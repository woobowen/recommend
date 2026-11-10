#!/usr/bin/env python3
"""Synthetic CPU API probe. No original weights, CUDA or optimizer updates."""
import argparse,hashlib,inspect,json,os,time,traceback
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
r={'scope':'Synthetic tiny CPU LLaMA+PEFT API only; not real7B/GPU feasibility','optimizer_updates':0,'status':'RUNNING'}
try:
 import torch,peft
 from transformers import LlamaConfig,LlamaForCausalLM
 from peft import LoraConfig,TaskType,get_peft_model,get_peft_model_state_dict,PeftModel
 torch.set_num_threads(2);torch.manual_seed(42)
 cfg=LlamaConfig(vocab_size=96,hidden_size=64,intermediate_size=128,num_hidden_layers=1,num_attention_heads=4,max_position_embeddings=128)
 base=LlamaForCausalLM(cfg);base.save_pretrained(out/'synthetic_base')
 model=get_peft_model(base,LoraConfig(r=8,lora_alpha=32,lora_dropout=.05,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj'],modules_to_save=['embed_tokens','lm_head'],bias='none',task_type=TaskType.CAUSAL_LM))
 r['before_bf16_dtypes']={n:str(v.dtype) for n,v in model.named_parameters() if v.requires_grad}
 model.bfloat16();model.train();ids=torch.randint(3,96,(2,12));labels=ids.clone();labels[:,:8]=-100
 loss=model(input_ids=ids,attention_mask=torch.ones_like(ids),labels=labels).loss;loss.backward()
 r['finite_forward_backward']=bool(torch.isfinite(loss)) and all(torch.isfinite(v.grad).all().item() for v in model.parameters() if v.requires_grad)
 r['gradient_norm_api']=float(torch.linalg.vector_norm(next(v.grad for v in model.parameters() if v.grad is not None),ord=float('inf'),dtype=torch.float32))
 model.save_pretrained(out/'synthetic_adapter',safe_serialization=False)
 saved=torch.load(out/'synthetic_adapter/adapter_model.bin',map_location='cpu',weights_only=True)
 fresh=LlamaForCausalLM.from_pretrained(out/'synthetic_base',torch_dtype=torch.bfloat16,local_files_only=True)
 loaded=PeftModel.from_pretrained(fresh,out/'synthetic_adapter',torch_dtype=torch.bfloat16,is_trainable=False)
 state=get_peft_model_state_dict(loaded)
 assert set(state)==set(saved)
 r['reload_dtypes']={n:{'saved':str(saved[n].dtype),'loaded':str(state[n].dtype),'shape_equal':state[n].shape==saved[n].shape,'float_values_equal':torch.equal(state[n].float(),saved[n].float())} for n in state}
 r['fresh_public_api_key_shape_value_equal']=all(v['shape_equal'] and v['float_values_equal'] for v in r['reload_dtypes'].values())
 r['required_reload_bf16_cast']=any(state[n].dtype!=saved[n].dtype for n in state)
 loaded.bfloat16();state=get_peft_model_state_dict(loaded)
 r['after_explicit_bf16_tensor_equal']=all(state[n].dtype==saved[n].dtype and torch.equal(state[n],saved[n]) for n in state)
 r['peft_reload_source_sha256']=hashlib.sha256(inspect.getsource(PeftModel.from_pretrained).encode()).hexdigest()
 r['cuda_initialized']=torch.cuda.is_initialized();assert not r['cuda_initialized']
 assert r['finite_forward_backward'] and r['fresh_public_api_key_shape_value_equal'] and r['after_explicit_bf16_tensor_equal']
 r['status']='SYNTHETIC_API_PASS'
except Exception as e:r.update(status='FAILED',error=repr(e),traceback=traceback.format_exc());traceback.print_exc()
(out/'compatibility.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'status':r['status'],'reload_cast':r.get('required_reload_bf16_cast')}));raise SystemExit(0 if r['status']=='SYNTHETIC_API_PASS' else 1)
