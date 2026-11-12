#!/usr/bin/env python3
"""Replay the bound layer28 FA case, preserving tensor strides and native inputs."""
import argparse
import json
import os
from pathlib import Path
import time
import hashlib


def main():
    p=argparse.ArgumentParser();p.add_argument('--case-root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--process-repeat',type=int,default=-1);a=p.parse_args()
    if a.process_repeat==-1:
        import subprocess,sys
        for repeat in range(3):
            subprocess.run([sys.executable,__file__,'--case-root',a.case_root,'--output',a.output,
                            '--process-repeat',str(repeat)],check=True)
        return
    import torch
    import flash_attn.flash_attn_interface as fa
    import importlib.metadata
    from formal_compare import Compare,write
    import numpy as np
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    rank=int(os.environ['RANK']);root=Path(a.case_root)/('boundary.rank%d'%rank)
    out=Path(a.output)/('process%d.rank%d'%(a.process_repeat,rank));out.mkdir(parents=True,exist_ok=False)
    def load(name):
        path=root/('recompute.'+name+'.pt');cpu=torch.load(path,map_location='cpu',weights_only=False)
        meta=json.loads(path.with_suffix('.json').read_text())
        t=torch.empty_strided(meta['shape'],meta['stride'],dtype=cpu.dtype,device='cuda');t.copy_(cpu)
        assert list(t.stride())==meta['stride']
        assert hashlib.sha256(cpu.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()==meta['sha256']
        return t
    q,kv=load('unpad.q.post'),load('unpad.kv.post')
    cq,ck=load('native.cu_q'),load('native.cu_k')
    dout=load('native.dO');saved_q,saved_k,saved_v=load('native.q'),load('native.k'),load('native.v')
    saved_out,lse,rng=load('native.out'),load('native.lse'),load('native.rng')
    events=[json.loads(x) for x in (Path(a.case_root)/('events.rank%d.jsonl'%rank)).read_text().splitlines()]
    options=[x for x in events if x['event']=='fa_backward_ctx'];assert len(options)==1
    cfg=options[0];assert cfg['dropout_p']==0 and cfg['alibi_slopes'] is None and q.shape[-1]==128
    original=fa._wrapped_flash_attn_varlen_backward
    report={'rank':rank,'layer':28,'process_repeat':a.process_repeat,'case_root':str(root),
            'flash_attn':importlib.metadata.version('flash-attn'),'options':cfg,
            'device':torch.cuda.get_device_name(),'rows':[],'reconstruction':'all input shapes/dtypes/strides exact; new storage_offset0; values byte-bound to saved tensors',
            'input_strides':{n:list(t.stride()) for n,t in [('q',q),('kv',kv),('dO',dout),('native.q',saved_q),('native.k',saved_k),('native.v',saved_v)]}}
    refs={}
    for mode in [False,True]:
        for repeat in range(3):
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            before=torch.cuda.memory_allocated()
            t0=time.monotonic()
            output=fa.flash_attn_varlen_kvpacked_func(q,kv,cq,ck,cfg['max_q'],cfg['max_k'],0.0,
                    softmax_scale=cfg['softmax_scale'],causal=cfg['causal'],deterministic=mode)
            dq=torch.empty_like(saved_q)
            dkv=torch.empty((saved_k.shape[0],2,*saved_k.shape[1:]),dtype=saved_k.dtype,device='cuda')
            torch.cuda.synchronize();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) if repeat==0 else None
            if profiler:profiler.__enter__()
            start.record()
            original(dout,saved_q,saved_k,saved_v,saved_out,lse,dq,dkv[:,0],dkv[:,1],cq,ck,
                     cfg['max_q'],cfg['max_k'],cfg['dropout_p'],cfg['softmax_scale'],cfg['causal'],
                     *cfg['window_size'],cfg['softcap'],None,mode,rng_state=rng)
            end.record();torch.cuda.synchronize()
            if profiler:profiler.__exit__(None,None,None)
            tensors={'out':output.detach().cpu().contiguous(),'dQ':dq.detach().cpu().contiguous(),
                     'dK':dkv[:,0].detach().cpu().contiguous(),'dV':dkv[:,1].detach().cpu().contiguous()}
            key='det' if mode else 'default'
            if key not in refs:refs[key]=tensors
            cmp=Compare(torch,np)
            for name,t in tensors.items():cmp.tensor(refs[key][name],t,name,False)
            hashes={name:hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest() for name,t in tensors.items()}
            row={'mode':mode,'repeat':repeat,'hashes':hashes,'within_mode':cmp.tensors,
                 'backward_ms':start.elapsed_time(end),'wall_seconds':time.monotonic()-t0,
                 'allocated_before':before,'peak_allocated':torch.cuda.max_memory_allocated(),
                 'peak_reserved':torch.cuda.max_memory_reserved(),'native_dispatch_deterministic':mode}
            if profiler:
                row['cuda_kernels']=sorted(set(e.name for e in profiler.events() if e.device_type==torch.autograd.DeviceType.CUDA))
                profiler.export_chrome_trace(str(out/(key+'.profiler.json')))
            report['rows'].append(row)
            # Every repetition remains available for independent cross-process CPU recomputation.
            torch.save(tensors,out/(key+'.repeat%d.pt'%repeat))
            del dq,dkv,output
    cmp=Compare(torch,np)
    for name,t in refs['det'].items():cmp.tensor(refs['default'][name],t,name,False)
    report['between_modes']=cmp.tensors
    write(out/'result.json',report)
    print(json.dumps({'rank':rank,'process':a.process_repeat,'output':str(out),'status':'COMPLETE'}))


if __name__=='__main__':main()
