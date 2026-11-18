#!/usr/bin/env python3
"""Compare real-model boundary preflight to a cross-process resumed trajectory."""
import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]/'letter/scripts'))
from formal_compare import ATOL, RTOL, Compare, inspect_schema
from g3_identity import EXPERIMENT_ID, ASSETS, TRAINABLE_PARAMETERS, source_identity
import numpy as np
import torch
torch.set_num_threads(4)


def events(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def close(a, b):
    return math.isfinite(a) and math.isfinite(b) and abs(a-b) <= ATOL + RTOL*abs(a)


def main():
    p=argparse.ArgumentParser()
    for name in ('continuous','early','resumed','output'):p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args()
    report={'status':'FAIL','scope':'PREFLIGHT','description':'actual G3 model, four short preflight epochs, epoch1 resume',
            'experiment_id':EXPERIMENT_ID,'assets':ASSETS,
            'argv':sys.argv,'cwd':os.getcwd(),'python':sys.executable,
            'started_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'source_identity':source_identity([__file__,Path(__file__).resolve().parents[1]/'letter/scripts/formal_compare.py']),
            'tolerance':{'absolute':ATOL,'relative':RTOL,'fixed_before_observation':True,
                         'exact_required':['scheduler','all RNG','loader row/input/label sequence','discrete state','optimizer counters']},'ranks':[]}
    with a.output.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    for epoch in range(1,5):
        reference=json.loads((a.continuous/f'epoch{epoch}_rows.json').read_text())
        other=json.loads(((a.early if epoch==1 else a.resumed)/f'epoch{epoch}_rows.json').read_text())
        assert reference==other, 'row order changed in epoch '+str(epoch)
    for rank in (0,1):
        reports=[json.loads((root/f'rank{rank}.json').read_text()) for root in (a.continuous,a.early,a.resumed)]
        assert all(x['scope']=='PREFLIGHT' and x['experiment_id']==EXPERIMENT_ID for x in reports)
        assert len({x['run_commit'] for x in reports})==len({x['config_sha256'] for x in reports})==1
        assert reports[0]['source_identity']==reports[1]['source_identity']==reports[2]['source_identity']
        assert reports[0]['tokenizer_identity']==reports[1]['tokenizer_identity']==reports[2]['tokenizer_identity']
        assert all(x['parameter_update_validation']['status']=='PASS' and x['optimizer_membership_exact'] for x in reports)
        left=torch.load(a.continuous/f'boundary_pre_best_rank{rank}.pt',map_location='cpu',weights_only=False)
        right=torch.load(a.resumed/f'boundary_pre_best_rank{rank}.pt',map_location='cpu',weights_only=False)
        left_schema=inspect_schema(left,torch,'continuous')
        right_schema=inspect_schema(right,torch,'resumed')
        assert left_schema['trainable_elements']==right_schema['trainable_elements']==TRAINABLE_PARAMETERS
        assert left['global_step']==right['global_step']==10
        assert left['micro_steps']==right['micro_steps']==40
        assert left['accelerator_step']==right['accelerator_step']==40
        cmp=Compare(torch,np);cmp.compare(left,right)
        assert not cmp.failures, json.dumps(cmp.failures[:10],default=str)
        le=events(a.continuous/f'events.rank{rank}.jsonl')
        re=events(a.resumed/f'events.rank{rank}.jsonl')
        restored=[x for x in re if x.get('event')=='partial_restore_exact']
        assert len(restored)==1 and restored[0]['micro_steps']==10 and restored[0]['global_steps']==2
        assert restored[0]['optimizer_micro_step_id']==1
        lb=[x for x in le if x.get('event')=='batch' and x.get('epoch_index',-1)>=1]
        rb=[x for x in re if x.get('event')=='batch' and x.get('epoch_index',-1)>=1]
        keys=('step','epoch_index','loader_yielded','input_sha256','label_sha256','shape')
        assert len(lb)==len(rb)==30 and [[x[k] for k in keys] for x in lb]==[[x[k] for k in keys] for x in rb]
        ll=[x for x in le if x.get('event')=='micro_loss' and x.get('epoch_index',-1)>=1]
        rl=[x for x in re if x.get('event')=='micro_loss' and x.get('epoch_index',-1)>=1]
        assert len(ll)==len(rl)==30 and all(close(x['loss'],y['loss']) for x,y in zip(ll,rl))
        lu=[x for x in le if x.get('event')=='optimizer_step' and x['step']>2]
        ru=[x for x in re if x.get('event')=='optimizer_step' and x['step']>2]
        assert [x['step'] for x in lu]==[x['step'] for x in ru]==list(range(3,11))
        assert all(x['lr']==y['lr'] for x,y in zip(lu,ru))
        assert all(x['grad_norm'] is not None and y['grad_norm'] is not None and
                   close(x['grad_norm'],y['grad_norm']) for x,y in zip(lu,ru))
        lv=[x['values']['eval_loss'] for x in le if x.get('event')=='log' and 'eval_loss' in x['values']]
        # A resumed run may append to the original early-stop directory. Keep
        # original epoch-1 evidence, comparing only the resumed epoch-2..4 logs.
        rv=[x['values']['eval_loss'] for x in re if x.get('event')=='log' and 'eval_loss' in x['values'] and x.get('epoch',0)>=2]
        assert len(lv)==4 and len(rv)==3 and all(close(x,y) for x,y in zip(lv[1:],rv))
        report['ranks'].append({'rank':rank,'run_commit':reports[0]['run_commit'],
            'config_sha256':reports[0]['config_sha256'],
            'state_tensors':len(cmp.tensors),'tensor_diagnostics':cmp.tensors,'schema_continuous':left_schema,
            'schema_resumed':right_schema,'batch_label_order_exact':True,'losses_within_tolerance':True,
            'partial_restoration_exact':True,'optimizer_step_and_lr_sequence_exact':True,
            'gradient_norms_within_tolerance':True,
            'validation_within_tolerance':True,'validation_losses_continuous':lv,'validation_losses_resumed':rv,
            'state_and_log_identity':source_identity([a.continuous/f'boundary_pre_best_rank{rank}.pt',
                a.resumed/f'boundary_pre_best_rank{rank}.pt',a.continuous/f'events.rank{rank}.jsonl',
                a.resumed/f'events.rank{rank}.jsonl'])})
        del left,right
    report['status']='PASS';report['row_order_exact']=True
    report['finished_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'ranks':2}))


if __name__=='__main__':main()
