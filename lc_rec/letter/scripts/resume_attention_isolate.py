#!/usr/bin/env python3
"""DIAGNOSTIC_ONLY repeat exact real attention q/kv/dout without a model."""
import argparse
import json
import os
from pathlib import Path
import hashlib

def main():
    p=argparse.ArgumentParser();p.add_argument('--case-root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    import torch
    import flash_attn.flash_attn_interface as interface
    from resume_probe import tensor
    rank=int(os.environ['RANK']);torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    path=Path(a.case_root)/f'attention_case.rank{rank}.pt';case=torch.load(path,map_location='cpu',weights_only=False)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    report={'scope':'DIAGNOSTIC_ONLY','rank':rank,'case_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'case_bytes':path.stat().st_size,
            'function':case['function'],'source':case['source'],'args':[tensor(x) if isinstance(x,torch.Tensor) else x for x in case['args']],
            'dout':tensor(case['dout']),'atol':1e-6,'rtol':1e-3,'repeats':5,'modes':{}}
    all_grads={}
    for deterministic in [False,True]:
        mode=str(deterministic);records=[];reference=None;all_grads[mode]=[]
        for i in range(5):
            args=[x.cuda().clone() if isinstance(x,torch.Tensor) else x for x in case['args']]
            indices=[0,1] if 'kvpacked' in case['function'] else [0,1,2]
            for idx in indices:args[idx].requires_grad_(True)
            kw=dict(case['kwargs']);kw['deterministic']=deterministic
            output=getattr(interface,case['function'])(*args,**kw)
            grads=torch.autograd.grad(output,[args[idx] for idx in indices],grad_outputs=case['dout'].cuda())
            cpu=[x.detach().cpu() for x in grads];row={'repeat':i,'output':tensor(output),'gradients':[tensor(x) for x in cpu]}
            if reference is None:reference=cpu
            row['difference_vs_first']=[]
            for left,right in zip(reference,cpu):
                x=left.float();y=right.float();diff=(y-x).abs()
                row['difference_vs_first'].append({'different':int((left!=right).sum()),'outside_original_tolerance':int((diff>1e-6+1e-3*x.abs()).sum()),'max_abs':float(diff.max()),'reference_norm':float(x.norm()),'norm':float(y.norm())})
            records.append(row);all_grads[mode].append(cpu)
        report['modes'][mode]=records
    # Only small real attention gradients, never full model/optimizer payloads.
    torch.save(all_grads,out/f'gradient_repeats.rank{rank}.pt')
    (out/f'isolate.rank{rank}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'rank':rank,'modes':{k:[x['difference_vs_first'] for x in v] for k,v in report['modes'].items()}}))

if __name__=='__main__':main()
