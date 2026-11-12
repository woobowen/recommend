#!/usr/bin/env python3
"""CPU-only, fixed-tolerance comparisons for layer28 boundaries and full states."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def compare_tensor(left,right):
    import torch,numpy as np
    from formal_compare import Compare
    c=Compare(torch,np);c.tensor(left,right,'tensor',False)
    row=c.tensors[0]
    if left.shape==right.shape and left.is_floating_point():
        a,b=left.reshape(-1).double(),right.reshape(-1).double()
        delta=(b-a).abs();limit=1e-6+1e-3*a.abs()
        row['maximum_excess_over_limit']=float((delta-limit).clamp_min(0).max()) if a.numel() else 0
        indices=(a!=b).nonzero().flatten()[:24]
        row['raw_witness']=[{'flat_index':int(i),'reference':float(a[i]),'candidate':float(b[i]),
                             'abs_diff':float(delta[i]),'limit':float(limit[i])} for i in indices]
    return row


def boundaries(left,right,out):
    import torch
    from formal_compare import write,sha
    summary={'left':str(left),'right':str(right),'atol':1e-6,'rtol':1e-3,'ranks':[]}
    for rank in [0,1]:
        l=left/('boundary.rank%d'%rank);r=right/('boundary.rank%d'%rank)
        names={x.name for x in l.glob('*.pt')};other={x.name for x in r.glob('*.pt')}
        assert names and names==other, 'boundary coverage mismatch'
        rows={}
        for name in sorted(names):
            a=torch.load(l/name,map_location='cpu',weights_only=False)
            b=torch.load(r/name,map_location='cpu',weights_only=False)
            row=compare_tensor(a,b)
            row['left_descriptor']=json.loads((l/name).with_suffix('.json').read_text())
            row['right_descriptor']=json.loads((r/name).with_suffix('.json').read_text())
            rows[name]=row
        checks={}
        for side,root in [('left',l),('right',r)]:
            load=lambda n:torch.load(root/('recompute.'+n+'.pt'),map_location='cpu',weights_only=False)
            pairs=[('dispatch.dQ','native.dQ'),('native.dQ','unpad.q.post.grad'),
                   ('unpad.q.pre.grad','rope.q.post.grad'),('rope.q.pre.grad','q_proj.output.grad'),
                   ('pad.pre.grad','native.dO')]
            for a,b in pairs:checks[side+':'+a+'='+b]=compare_tensor(load(a).reshape(-1),load(b).reshape(-1))
            dkv=load('native.dKV')
            for i,k in enumerate(['dK','dV']):checks[side+':packed.'+k]=compare_tensor(dkv[:,i],load('dispatch.'+k))
            pre=load('unpad.q.pre');indices=load('unpad.q.indices')
            expected=torch.zeros_like(pre).reshape(-1,*pre.shape[2:]);expected[indices]=load('native.dQ')
            checks[side+':unpad_scatter']=compare_tensor(expected.reshape_as(pre),load('unpad.q.pre.grad'))
        summary['ranks'].append({'rank':rank,'tensor_count':len(rows),'rows':rows,'boundary_correspondence':checks})
    write(out/'boundary_comparison.json',summary)
    print(json.dumps({'boundary':[{ 'rank':r['rank'],'count':r['tensor_count'],'nonexact':[n for n,v in r['rows'].items() if not v['exact_equal']]} for r in summary['ranks']]}))


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'CPU-only'
    p=argparse.ArgumentParser();p.add_argument('--left');p.add_argument('--right');p.add_argument('--isolate')
    p.add_argument('--no-full-state',action='store_true');a=p.parse_args()
    from formal_compare import write,sha
    import torch
    torch.set_num_threads(8)
    out=Path('/data/bwwu/lc-rec/attention-20261112')/Path(os.environ['GRID_JOB_DIR']).name
    out.mkdir(parents=True,exist_ok=False)
    if a.isolate:
        root=Path(a.isolate);result={'root':str(root),'ranks':[]}
        for rank in [0,1]:
            reports=[json.loads((root/('process%d.rank%d'%(i,rank))/'result.json').read_text()) for i in range(3)]
            rows=[]
            for mode in ['default','det']:
                reference=torch.load(root/('process0.rank%d'%rank)/(mode+'.repeat0.pt'),map_location='cpu',weights_only=False)
                for process in range(3):
                    for repeat in range(3):
                        path=root/('process%d.rank%d'%(process,rank))/(mode+'.repeat%d.pt'%repeat)
                        tensors=torch.load(path,map_location='cpu',weights_only=False)
                        rows.append({'mode':mode,'process':process,'repeat':repeat,'source_sha256':sha(path),
                                     'tensors':{n:compare_tensor(t,tensors[n]) for n,t in reference.items()}})
            result['ranks'].append({'rank':rank,'process_reports':reports,'cross_process':rows})
        write(out/'isolate_comparison.json',result)
    else:
        left,right=Path(a.left),Path(a.right)
        boundaries(left,right,out)
        if not a.no_full_state:
            cmd=[sys.executable,str(Path(__file__).with_name('formal_compare.py')),'--continuous',str(left),'--resumed',str(right),'--output',str(out/'numeric_comparison.json')]
            proc=subprocess.run(cmd);assert proc.returncode in [0,2]
            write(out/'comparison_exit.json',{'argv':cmd,'exit_code':proc.returncode,'numeric_gate':'PASS' if proc.returncode==0 else 'FAIL'})


if __name__=='__main__':main()
