"""Bounded adapter reload and beam20 timing, with no recommendation metrics."""
import argparse,hashlib,json,os,resource,sys,time,traceback
from pathlib import Path

def write(p,v):
 q=p.with_suffix('.partial');q.write_text(json.dumps(v,indent=2)+'\n');q.replace(p)

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--data-root',required=True);p.add_argument('--model',required=True);p.add_argument('--adapter',required=True);p.add_argument('--audit-output',required=True);p.add_argument('--output',required=True);p.add_argument('--run-commit',required=True);a=p.parse_args()
 rank=int(os.environ.get('LOCAL_RANK',0));out=Path(a.output);out.mkdir(parents=True,exist_ok=True);start=time.monotonic();r={'rank':rank,'argv':sys.argv,'RUN_COMMIT':a.run_commit,'status':'RUNNING','no_metrics':True,'timings':[]}
 try:
  import numpy as np,torch,torch.distributed as dist
  from transformers import LlamaForCausalLM,LlamaTokenizer
  from peft import PeftModel,get_peft_model_state_dict
  torch.cuda.set_device(rank);torch.manual_seed(42);dist.init_process_group('nccl')
  sys.path.insert(0,a.source)
  from data import SeqRecDataset
  from collator import TestCollator
  from types import SimpleNamespace
  args=SimpleNamespace(dataset='Instruments',data_path=a.data_root,max_his_len=20,his_sep=', ',index_file='.index.json',add_prefix=False)
  tok=LlamaTokenizer.from_pretrained(a.adapter,local_files_only=True);tok.pad_token_id=0;tok.padding_side='left';tok.model_max_length=2048
  samples=json.loads((Path(a.audit_output)/'probe_samples.json').read_text())
  for kind,identity in samples['a_data_identity'].items():
   path=Path(a.data_root)/'Instruments'/('Instruments.'+kind+'.json');assert hashlib.sha256(path.read_bytes()).hexdigest()==identity['sha256']
  audited=LlamaTokenizer.from_pretrained(Path(a.audit_output)/'tokenizer_a',local_files_only=True);assert tok.get_vocab()==audited.get_vocab()
  r['a_mapping_and_tokenizer_identity_verified']=True
  data=SeqRecDataset(args,mode='test');collate=TestCollator(args,tok);allowed=data.get_prefix_allowed_tokens_fn(tok)
  t=time.monotonic();base=LlamaForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,device_map={'':rank},local_files_only=True);base.resize_token_embeddings(len(tok));r['base_load_seconds']=time.monotonic()-t
  t=time.monotonic();model=PeftModel.from_pretrained(base,a.adapter,torch_dtype=torch.bfloat16,device_map={'':rank},is_trainable=False);model.eval();r['adapter_reload_seconds']=time.monotonic()-t
  # Verify every saved adapter tensor against the actual reloaded parameters; no permissive missing-state claim.
  path=Path(a.adapter)/'adapter_model.bin'
  if path.exists(): saved=torch.load(path,map_location='cpu',weights_only=True)
  else:
   from safetensors.torch import load_file
   saved=load_file(str(Path(a.adapter)/'adapter_model.safetensors'))
  loaded=get_peft_model_state_dict(model);assert set(saved)==set(loaded),(len(saved),len(loaded))
  mismatch=[]
  for k,v in saved.items():
   if not torch.equal(v.to(dtype=loaded[k].dtype),loaded[k].detach().cpu()):mismatch.append(k)
  assert not mismatch,mismatch[:10]
  r['adapter_exact_key_and_value_reload']={'keys':len(saved),'all_equal_in_inference_dtype':True};del saved,loaded
  samples=json.loads((Path(a.audit_output)/'probe_samples.json').read_text())
  reps=samples['test_prompt0_representatives'];r['selection']=reps
  # Source test uses vanilla HF attention and no temperature division at inference.
  r['generation_definition']={'model':'LlamaForCausalLM+PeftModel BF16; vanilla HF attention as original test_ddp.py','num_beams':20,'num_return_sequences':20,'max_new_tokens':10,'prefix_constraint':'original positional allowed-token sets, no Trie repair','prompt':0,'batch':1,'filtering_and_metrics':'not executed'}
  keys=list(reps) if isinstance(reps,dict) else list(range(len(reps)))
  for n,key in enumerate([keys[0]]+keys):
   d=reps[key] if isinstance(reps,dict) else reps[key];idx=d['row_idx'];data.set_prompt(int(d.get('prompt_id',0)))
   inputs,targets=collate([data[idx]]);inputs={k:v.to(rank) for k,v in inputs.items()}
   assert int(inputs['attention_mask'].sum())==d['input_length']
   torch.cuda.synchronize();t=time.monotonic()
   with torch.no_grad(): result=model.generate(**inputs,max_new_tokens=10,prefix_allowed_tokens_fn=allowed,num_beams=20,num_return_sequences=20,output_scores=True,return_dict_in_generate=True,early_stopping=True)
   torch.cuda.synchronize();seconds=time.monotonic()-t
   suffix=result.sequences[:,inputs['input_ids'].shape[1]:].cpu().tolist();assert len(suffix)==20
   r['timings'].append({'warmup':n==0,'selection':str(key),'row_idx':idx,'prompt_id':d.get('prompt_id',0),'input_tokens':int(inputs['input_ids'].numel()),'seconds':seconds,'generated_sequence_count':len(suffix),'generated_tokens':sum(len(z) for z in suffix),'output_token_ids':suffix,'eos_all':all(tok.eos_token_id in z for z in suffix)})
   assert r['timings'][-1]['eos_all'],'beam output did not reach SID EOS within fixed10 token budget'
   del result,inputs;write(out/f'inference_rank{rank}.json',r)
  r['status']='RELOAD_BEAM20_TECHNICAL_PASS'
 except BaseException as e:
  r['status']='FAIL';r['error']={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()};print(r['error']['traceback'],flush=True)
 finally:
  r['elapsed_seconds']=time.monotonic()-start;r['cpu_process_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
  if 'torch' in locals() and torch.cuda.is_initialized():r['cuda_peak_allocated_bytes']=torch.cuda.max_memory_allocated(rank);r['cuda_peak_reserved_bytes']=torch.cuda.max_memory_reserved(rank)
  write(out/f'inference_rank{rank}.json',r)
  if 'dist' in locals() and dist.is_initialized():dist.destroy_process_group()
 return 0 if r['status']=='RELOAD_BEAM20_TECHNICAL_PASS' else 2
if __name__=='__main__':raise SystemExit(main())
