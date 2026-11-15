#!/usr/bin/env python3
"""Check first-epoch continuous and early real-model event streams."""
import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'letter/scripts'))
from formal_compare import ATOL,RTOL

def events(path):return [json.loads(x) for x in path.read_text().splitlines()]
def close(a,b):return math.isfinite(a) and math.isfinite(b) and abs(a-b)<=ATOL+RTOL*abs(a)
def check(a,b,rank):
    left,right=events(a/f'events.rank{rank}.jsonl'),events(b/f'events.rank{rank}.jsonl')
    lb=[x for x in left if x.get('event')=='batch' and x.get('epoch_index')==0]
    rb=[x for x in right if x.get('event')=='batch' and x.get('epoch_index')==0]
    keys=('step','epoch_index','loader_yielded','input_sha256','label_sha256','shape')
    assert len(lb)==len(rb)==10 and [[x[k] for k in keys] for x in lb]==[[x[k] for k in keys] for x in rb]
    ll=[x['loss'] for x in left if x.get('event')=='micro_loss' and x.get('epoch_index')==0]
    rl=[x['loss'] for x in right if x.get('event')=='micro_loss' and x.get('epoch_index')==0]
    assert len(ll)==len(rl)==10 and all(close(x,y) for x,y in zip(ll,rl))
    le=[x['values']['eval_loss'] for x in left if x.get('event')=='log' and 'eval_loss' in x['values']]
    re=[x['values']['eval_loss'] for x in right if x.get('event')=='log' and 'eval_loss' in x['values']]
    assert len(le)==4 and len(re)==1 and close(le[0],re[0])
    return {'rank':rank,'inputs_labels_order_exact':True,'losses_within_fixed_tolerance':True,'validation_within_fixed_tolerance':True,'continuous_validation_loss':le[0],'early_validation_loss':re[0]}
def main():
    p=argparse.ArgumentParser()
    for n in ('continuous','early','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    assert json.loads((a.continuous/'epoch1_rows.json').read_text())==json.loads((a.early/'epoch1_rows.json').read_text())
    report={'status':'PASS','scope':'first epoch actual input hashes, losses and validation','tolerance':{'atol':ATOL,'rtol':RTOL},'ranks':[check(a.continuous,a.early,r) for r in (0,1)]}
    with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({'status':report['status']}))
if __name__=='__main__':main()
