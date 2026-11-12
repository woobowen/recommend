#!/usr/bin/env python3
"""CPU-only historical checkpoint inventory and canonical S1 fingerprints."""
import argparse
import json
from pathlib import Path
import hashlib
import os
import datetime as dt
from resume_probe import tree

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import torch
    cp=Path(a.checkpoint);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    report={'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'checkpoint':str(cp),'files':[],'torch':torch.__version__,'cuda_initialized_before':torch.cuda.is_initialized()}
    for f in sorted(cp.rglob('*')):
        if not f.is_file():continue
        row={'path':str(f.relative_to(cp)),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
        report['files'].append(row)
        if f.name.endswith('model_states.pt') or f.name.endswith('optim_states.pt'):
            x=torch.load(f,map_location='cpu',weights_only=False)
            dest=out/(f.name+'.json');dest.write_text(json.dumps(tree(x),indent=2)+'\n');row['fingerprint_file']=dest.name
            row['top_keys']=list(x)
            del x
    report['cuda_initialized_after']=torch.cuda.is_initialized()
    (out/'cpu_inspection.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
