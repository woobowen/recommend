#!/usr/bin/env python3
"""Saved CPU entry for numeric comparison, RNG continuity and comparator regression."""
import argparse
import json
import os
from pathlib import Path
import sys

def main():
    p=argparse.ArgumentParser();p.add_argument('--left');p.add_argument('--right');p.add_argument('--checkpoint');p.add_argument('--output',required=True)
    a=p.parse_args();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import torch
    import numpy as np
    import enum
    from formal_compare import Compare
    from resume_probe import tree
    from resume_analyze import differences
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    class E(enum.IntEnum):A=1;B=2
    good=Compare(torch,np);good.compare({'enumclass':E,'member':E.A,'float':torch.tensor([1.])},{'enumclass':E,'member':E.A,'float':torch.tensor([1.])});assert not good.failures
    bad=Compare(torch,np);bad.compare({'member':E.A,'float':torch.tensor([1.])},{'member':E.B,'float':torch.tensor([1.01])});assert len(bad.failures)==2
    report={'comparator_positive':not good.failures,'comparator_negative_detected':len(bad.failures),'cuda_initialized':torch.cuda.is_initialized()}
    if a.checkpoint:
        report['rng']=[]
        for rank in [0,1]:
            saved=torch.load(Path(a.checkpoint)/f'loader_state_{rank}.pth',map_location='cpu',weights_only=False)
            loaded=json.loads((Path(a.right)/f'before_first_forward.rank{rank}.json').read_text())
            report['rng'].append({'rank':rank,'checkpoint_step':saved['global_step'],'loader_yielded_saved':saved['yielded'],
                                  'checkpoint_rng_vs_pre_forward':differences(tree(saved['checkpoint_rng']),loaded['rng'])})
    if a.left:
        import formal_compare
        sys.argv=[sys.argv[0],'--continuous',a.left,'--resumed',a.right,'--output',str(out/'numeric_comparison.json')]
        code=formal_compare.main();assert code in [0,2];report['numeric_comparison_exit_code']=code
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
