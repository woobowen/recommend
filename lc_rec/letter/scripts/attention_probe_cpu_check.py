#!/usr/bin/env python3
"""Exercise the identified Bool statistics bug's actual repaired observer on CPU."""
import os
from pathlib import Path
import json
from types import SimpleNamespace
from formal_train import write
from attention_boundary import Boundary


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import torch
    out=Path('/data/bwwu/lc-rec/attention-20261112')/Path(os.environ['GRID_JOB_DIR']).name
    out.mkdir(exist_ok=False);events=[]
    probe=SimpleNamespace(out=out,rank=0,event=lambda name,**row:events.append(dict(event=name,**row)))
    observer=Boundary(probe)
    identity={'phase':'cpu_fixture','rank':0,'scope':'synthetic statistic regression; no model/GPU'}
    for i,t in enumerate([torch.tensor([False,True]),torch.tensor([0.,1.],dtype=torch.bfloat16),
                          torch.tensor([0,1],dtype=torch.int64),torch.empty(0)]):
        observer.save('fixture%d'%i,t,identity)
        saved=torch.load(observer.root/('cpu_fixture.fixture%d.pt'%i),weights_only=False,map_location='cpu')
        assert torch.equal(t,saved) and saved.dtype==t.dtype
        assert events[-1]['max_abs']==(1.0 if t.numel() else 0.0)
    write(out/'statistic_regression.json',{'status':'PASS','fixtures':events,'CUDA_VISIBLE_DEVICES':''})


if __name__=='__main__':main()
