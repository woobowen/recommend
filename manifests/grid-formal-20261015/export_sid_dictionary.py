"""CPU read-only projection of existing SID tensors for independent auditing."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--path',type=Path,required=True);p.add_argument('--sha256',required=True)
    a=p.parse_args();assert os.environ.get('CUDA_VISIBLE_DEVICES','')==''
    assert a.path.is_file() and not a.path.is_symlink()
    assert hashlib.sha256(a.path.read_bytes()).hexdigest()==a.sha256
    import torch
    sid=torch.load(a.path,map_location='cpu',weights_only=True)
    assert sid.ndim==2 and sid.shape[0]==4 and sid.dtype in (torch.int32,torch.int64)
    rows=sid.t().tolist();legal=sorted({tuple(row) for row in rows})
    assert all(len(row)==4 and all(type(code)is int and code>=0 for code in row) for row in rows)
    print(json.dumps({'kind':'existing SID raw four-code dictionary projection; no upstream regeneration',
        'source':{'path':str(a.path),'sha256':a.sha256},'shape':list(sid.shape),
        'row_count':len(rows),'unique_four_code_count':len(legal),'duplicate_four_code_rows':len(rows)-len(legal),
        'legal_raw_sids':legal,'model_constructions':0,'GPU_model_calls':0},separators=(',',':')))


if __name__=='__main__':main()
