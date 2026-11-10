#!/usr/bin/env python3
"""Round3 generation-only continuation of the immutable round2 adapter;0updates."""
import argparse,datetime as dt,hashlib,json,os,sys,time,traceback
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,r):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.partial');t.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');t.replace(p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--samples',required=True);p.add_argument('--adapter',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--run-commit',required=True);p.add_argument('--parent-job',required=True);a=p.parse_args()
 import torch,torch.distributed as dist
 from transformers import LlamaForCausalLM,LlamaTokenizer
 from peft import PeftModel,get_peft_model_state_dict
 rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE']);assert world==2 and len(a.run_commit)==40
 torch.cuda.set_device(local);dist.init_process_group('nccl',timeout=dt.timedelta(seconds=180));device=torch.device('cuda',local)
 cfg=json.loads(Path(a.config).read_text());samples=json.loads(Path(a.samples).read_text());out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
 report={'status':'RUNNING','RUN_COMMIT':a.run_commit,'parent_job':a.parent_job,'job_id':os.environ['GRID_JOB_ID'],'rank':rank,'optimizer_updates':0,'argv':sys.argv,'cwd':os.getcwd(),'script_sha256':sha(__file__),'generation_source_sha256':sha(Path(__file__).with_name('inference_probe.py')),'config_sha256':sha(a.config),'samples_sha256':sha(a.samples),'adapter':str(a.adapter),'task':'Only reload/evaluate the existing round2 bounded adapter; never construct optimizer or run backward'}
 path=out/('inference_rank%d.json'%rank)
 def event(stage,**values):
  report['stage']=stage;report['elapsed_seconds']=time.monotonic()-start;dump(path,report);row={'stage':stage,'rank':rank,'seconds':report['elapsed_seconds'],**values};print(json.dumps(row),flush=True)
 def guard():
  m=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
  assert m>=40*1024**3
  if time.monotonic()-start>2550:raise TimeoutError('Generation-only job wall guard')
 def maxrank(sec):
  x=torch.tensor(sec,device=device,dtype=torch.float64);dist.all_reduce(x,op=dist.ReduceOp.MAX);return float(x)
 try:
  sys.path.insert(0,str(Path(cfg['vendor_dir']).resolve()));import utils
  utils.set_seed(42);guard();event('before_fresh_model_reload')
  t=time.monotonic();tokenizer=LlamaTokenizer.from_pretrained(a.adapter,local_files_only=True)
  assert len(tokenizer)==32930
  order=json.loads((Path(a.adapter)/'sid_token_order.json').read_text());assert tokenizer.convert_tokens_to_ids(order['tokens'])==order['ids']==list(range(32000,32930))
  model=LlamaForCausalLM.from_pretrained(cfg['base_model'],torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,device_map={'':local},local_files_only=True);model.resize_token_embeddings(len(tokenizer))
  model=PeftModel.from_pretrained(model,a.adapter,torch_dtype=torch.bfloat16,device_map={'':local},is_trainable=False)
  torch.cuda.synchronize();report['reload_seconds_max_rank']=maxrank(time.monotonic()-t)
  saved=torch.load(Path(a.adapter)/'adapter_model.bin',map_location='cpu',weights_only=True);actual=get_peft_model_state_dict(model)
  assert len(actual)==450 and set(actual)==set(saved)
  checks={}
  for key,value in actual.items():
   cpu=value.detach().cpu();expected=saved[key];equal=cpu.shape==expected.shape and torch.equal(cpu,expected.to(cpu.dtype)) and torch.equal(cpu.to(expected.dtype),expected)
   assert equal,'Exact fresh reload values differ:'+key
   checks[key]={'shape':list(cpu.shape),'saved_dtype':str(expected.dtype),'native_loaded_dtype':str(cpu.dtype),'bidirectional_exact_value':equal}
  report['reload']={'status':'PASS','strict_all450_keys_shapes_values':True,'tokenizer_sid_order':True,'native_dtype_changes':sum(v['saved_dtype']!=v['native_loaded_dtype'] for v in checks.values()),'checks':checks};del saved,actual
  event('fresh_adapter_reload_verified')
  from inference_probe import run_inference_probe
  dataargs=dict(samples['args_namespace']);dataargs['data_path']=cfg['data_path'];dataargs['base_model']=cfg['base_model']
  report['generation']=run_inference_probe(model=model,tokenizer=tokenizer,data_args=argparse.Namespace(**dataargs),sample_doc=samples,config=cfg,output_dir=out,rank=rank,local_rank=local,event=event,guard=guard,max_rank_seconds=maxrank)
  report['status']='PASS' if report['generation']['status']=='PASS' else 'PARTIAL'
 except Exception as error:
  report.update(status='FAILED',error=repr(error),traceback=traceback.format_exc());traceback.print_exc()
 finally:
  report['elapsed_seconds']=time.monotonic()-start;report['peak_allocated_gib']=torch.cuda.max_memory_allocated(local)/1024**3;report['peak_reserved_gib']=torch.cuda.max_memory_reserved(local)/1024**3;dump(path,report)
  if dist.is_initialized():dist.destroy_process_group()
 print(json.dumps({'status':report['status'],'rank':rank,'optimizer_updates':0}),flush=True)
 raise SystemExit(0 if report['status']=='PASS' else 2)
if __name__=='__main__':main()
