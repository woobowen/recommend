#!/usr/bin/env python3
"""Small CPU tensors: test native CPUAdam load, AdamW rule, imports and pinning."""
import argparse
import importlib
import importlib.metadata as md
import json
from pathlib import Path
import resource
import sys
import time
import torch
from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.git_version_info import installed_ops, accelerator_name, torch_info


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();start=time.monotonic()
    torch.manual_seed(42);torch.set_num_threads(4)
    x=torch.nn.Parameter(torch.randn(2048,dtype=torch.float32));y=torch.nn.Parameter(x.detach().clone())
    kw=dict(lr=5e-5,betas=(.9,.999),eps=1e-8,weight_decay=.01)
    cpu=DeepSpeedCPUAdam([x],adamw_mode=True,fp32_optimizer_states=True,**kw)
    ref=torch.optim.AdamW([y],foreach=False,**kw)
    diffs=[]
    for i in range(2):
        grad=torch.randn_like(x);x.grad=grad.clone();y.grad=grad.clone();cpu.step();ref.step()
        diffs.append(float((x-y).abs().max()))
    assert all(v<2e-6 for v in diffs),'CPUAdam/torch AdamW rule mismatch'
    state=cpu.state[x];assert all(v.device.type=='cpu' and v.dtype==torch.float32 for k,v in state.items() if isinstance(v,torch.Tensor))
    pin={}
    try:
        t=torch.empty(1024*1024,dtype=torch.uint8,pin_memory=True);pin={'bytes':t.numel(),'is_pinned':t.is_pinned()};del t
    except Exception as e:pin={'error':repr(e),'is_pinned':False}
    packages={}
    for name in ['torch','transformers','accelerate','deepspeed','numpy','tokenizers','huggingface_hub','safetensors','pydantic','ninja','psutil','cpuinfo']:
        m=importlib.import_module(name);packages[name]={'version':getattr(m,'__version__',None),'import_source':m.__file__}
    report={'status':'PASS','argv':sys.argv,'python':sys.executable,'packages':packages,'cuda_build':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'cuda_initialized':torch.cuda.is_initialized(),'installed_ops':installed_ops,'accelerator_name':accelerator_name,'compiled_torch_info':torch_info,'cpu_adam_class':type(cpu).__module__+'.'+type(cpu).__name__,'adamw_mode':cpu.adam_w_mode,'defaults':cpu.defaults,'max_abs_vs_torch_adamw_by_step':diffs,'exact_bitwise_equivalence_claimed':False,'state':{k:({'device':str(v.device),'dtype':str(v.dtype),'shape':list(v.shape)} if isinstance(v,torch.Tensor) else v) for k,v in state.items()},'memlock_bytes':resource.getrlimit(resource.RLIMIT_MEMLOCK),'pinned_memory_small_test':pin,'elapsed_seconds':time.monotonic()-start}
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'PASS','differences':diffs,'pinning':pin}))

if __name__=='__main__':main()
