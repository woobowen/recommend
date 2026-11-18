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
from formal_compare import ATOL, RTOL, Compare, MASTER_KEYS, named_nodes, tensor_leaves
from g3_identity import (EXPERIMENT_ID, ASSETS, TRAINABLE_PARAMETERS, source_identity,
                         EXECUTION_TOPOLOGY, EXECUTION_PROFILE, verify_execution_profile)
from single_gpu_training import execution_schedule
import numpy as np
import torch
torch.set_num_threads(4)


def events(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def close(a, b):
    return math.isfinite(a) and math.isfinite(b) and abs(a-b) <= ATOL + RTOL*abs(a)


def inspect_single_schema(state, label):
    required={'trainable','optimizer','scheduler','global_step','micro_steps','accelerator_step','rng'}
    assert isinstance(state,dict) and required<=set(state), label+' lacks required trajectory state'
    assert len(state['trainable'])==450 and all(isinstance(t,torch.Tensor) for t in state['trainable'].values()), label+' must include all450 trainable tensors'
    elements=sum(t.numel() for t in state['trainable'].values())
    assert elements==TRAINABLE_PARAMETERS, label+' trainable element count'
    assert {'python','numpy','cpu','cuda','loader_generator'}<=set(state['rng']), label+' incomplete RNG state'
    assert isinstance(state['rng']['cuda'],(list,tuple)) and len(state['rng']['cuda'])==1, label+' requires one visible CUDA RNG state'
    masters=[];moments={'exp_avg':0,'exp_avg_sq':0}
    for key,path,value in named_nodes({'optimizer':state['optimizer']}):
        if key in MASTER_KEYS:
            for tensor_path,tensor in tensor_leaves(value,torch,path):
                if tensor.is_floating_point() and tensor.numel():
                    masters.append({'path':tensor_path,'elements':tensor.numel(),'dtype':str(tensor.dtype)})
        if key in moments and isinstance(value,torch.Tensor):moments[key]+=value.numel()
    assert masters and all(row['dtype']=='torch.float32' for row in masters), label+' requires explicit FP32 master partitions'
    total=sum(row['elements'] for row in masters)
    assert moments['exp_avg']==moments['exp_avg_sq']==total, label+' optimizer moment/master coverage'
    return {'trainable_tensors':450,'trainable_elements':elements,'master_partitions':masters,
            'master_partition_elements':total,'optimizer_moment_elements':moments,
            'rng_keys':list(state['rng']),'visible_cuda_rng_states':1}


def main():
    p=argparse.ArgumentParser()
    for name in ('continuous','early','resumed','output'):p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args()
    schedule=execution_schedule(131413,preflight=True)
    micro=schedule['per_epoch_microbatches'];final=schedule['max_steps']
    resume_step=schedule['epoch_steps'][0]
    report={'status':'FAIL','scope':'PREFLIGHT','description':'actual G3 model, four short preflight epochs, epoch1 resume',
            'experiment_id':EXPERIMENT_ID,'assets':ASSETS,
            'execution_topology':EXECUTION_TOPOLOGY,'execution_profile':dict(EXECUTION_PROFILE),
            'sample_flow':schedule,
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
    for rank in range(EXECUTION_PROFILE['world_size']):
        reports=[json.loads((root/f'rank{rank}.json').read_text()) for root in (a.continuous,a.early,a.resumed)]
        assert all(x['scope']=='PREFLIGHT' and x['experiment_id']==EXPERIMENT_ID for x in reports)
        for row in reports:
            assert row['execution_topology']==EXECUTION_TOPOLOGY
            verify_execution_profile(row['execution_profile'])
        assert len({x['run_commit'] for x in reports})==len({x['config_sha256'] for x in reports})==1
        assert reports[0]['source_identity']==reports[1]['source_identity']==reports[2]['source_identity']
        assert reports[0]['tokenizer_identity']==reports[1]['tokenizer_identity']==reports[2]['tokenizer_identity']
        assert all(x['parameter_update_validation']['status']=='PASS' and x['optimizer_membership_exact'] for x in reports)
        left=torch.load(a.continuous/f'boundary_pre_best_rank{rank}.pt',map_location='cpu',weights_only=False)
        right=torch.load(a.resumed/f'boundary_pre_best_rank{rank}.pt',map_location='cpu',weights_only=False)
        left_schema=inspect_single_schema(left,'continuous')
        right_schema=inspect_single_schema(right,'resumed')
        assert left_schema['trainable_elements']==right_schema['trainable_elements']==TRAINABLE_PARAMETERS
        assert left['global_step']==right['global_step']==final
        assert left['micro_steps']==right['micro_steps']==4*micro
        assert left['accelerator_step']==right['accelerator_step']==4*micro
        cmp=Compare(torch,np);cmp.compare(left,right)
        assert not cmp.failures, json.dumps(cmp.failures[:10],default=str)
        le=events(a.continuous/f'events.rank{rank}.jsonl')
        re=events(a.resumed/f'events.rank{rank}.jsonl')
        restored=[x for x in re if x.get('event')=='partial_restore_exact']
        assert len(restored)==1 and restored[0]['micro_steps']==micro and restored[0]['global_steps']==resume_step
        assert restored[0]['optimizer_micro_step_id']==micro%EXECUTION_PROFILE['gradient_accumulation_steps']-1
        lb=[x for x in le if x.get('event')=='batch' and x.get('epoch_index',-1)>=1]
        rb=[x for x in re if x.get('event')=='batch' and x.get('epoch_index',-1)>=1]
        keys=('step','epoch_index','loader_yielded','input_sha256','label_sha256','shape')
        assert len(lb)==len(rb)==3*micro and [[x[k] for k in keys] for x in lb]==[[x[k] for k in keys] for x in rb]
        ll=[x for x in le if x.get('event')=='micro_loss' and x.get('epoch_index',-1)>=1]
        rl=[x for x in re if x.get('event')=='micro_loss' and x.get('epoch_index',-1)>=1]
        assert len(ll)==len(rl)==3*micro and all(close(x['loss'],y['loss']) for x,y in zip(ll,rl))
        lu=[x for x in le if x.get('event')=='optimizer_step' and x['step']>resume_step]
        ru=[x for x in re if x.get('event')=='optimizer_step' and x['step']>resume_step]
        assert [x['step'] for x in lu]==[x['step'] for x in ru]==list(range(resume_step+1,final+1))
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
    print(json.dumps({'status':report['status'],'ranks':1}))


if __name__=='__main__':main()
