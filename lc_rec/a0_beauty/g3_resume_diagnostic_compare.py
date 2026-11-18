#!/usr/bin/env python3
"""CPU-only diagnostic comparison of two completed frozen G3 trajectories.

This never repairs, accepts or rewrites the failed early-stop report. It does
not replace the final fresh production boundary comparison or grant readiness.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
import traceback


def load(path):
    return json.loads(Path(path).read_text())


def identity(path):
    path=Path(path);h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
    return {'path':str(path.resolve()),'bytes':path.stat().st_size,'sha256':h.hexdigest()}


def write(path,value):
    temporary=path.with_name(path.name+'.partial')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temporary.replace(path)


def need(ok,message):
    if not ok:raise ValueError(message)


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False,description=__doc__)
    for key in ('continuous','resumed','output'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();started=time.monotonic()
    need(not args.output.exists(),'preserve prior diagnostic receipt')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    report={'status':'FAIL','scope':'DIAGNOSTIC_ONLY','final_preparation_acceptance':False,
            'replaces_final_fresh_boundary_comparison':False,'formal_experiment_started':False,
            'formal_launch_authorized':False,'argv':sys.argv,'cwd':os.getcwd(),'python':sys.executable,
            'started_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'script':identity(__file__)}
    write(args.output,report)
    try:
        roots=[args.continuous.resolve(strict=True),args.resumed.resolve(strict=True)]
        need(roots[0]!=roots[1],'distinct real trajectories required')
        records=[load(root/'rank0.json') for root in roots]
        need(all(row['scope']=='PREFLIGHT' and row['probe'] is True and
                 row['status']=='BOUNDARY_PREFLIGHT_COMPLETE' and row['global_step']==10 and
                 row['epoch']==4. for row in records),'both diagnostic trajectories must have actually completed')
        commit=records[0]['run_commit'];config_sha=records[0]['config_sha256']
        need(len(commit)==40 and all(c in '0123456789abcdef' for c in commit),'full source commit')
        fields=('run_commit','config_sha256','experiment_id','execution_topology','execution_profile',
                'source_identity','tokenizer_identity','data_identity')
        need(all(records[0][key]==records[1][key] for key in fields),'diagnostic run/config/input/source identities differ')
        scripts=Path(records[0]['argv'][0]).resolve().parent
        need(Path(records[1]['argv'][0]).resolve().parent==scripts,'different executed training source roots')
        version=load(scripts.parents[1]/'RUN_VERSION.json')
        need(version['run_commit']==commit,'diagnostic source export commit')
        manifest={str(Path(row['path']).resolve()):row for row in version['files']}
        imports=[scripts/'boundary_compare.py',scripts/'g3_identity.py',
                 scripts/'single_gpu_training.py',scripts.parent/'letter/scripts/formal_compare.py']
        for path in imports:
            actual=identity(path);fixed=manifest[str(path.resolve())]
            need(actual['bytes']==fixed['bytes'] and actual['sha256']==fixed['sha256'],
                 'diagnostic dependency differs from frozen run source: '+str(path))
        sys.path.insert(0,str(scripts));sys.path.insert(1,str(scripts.parent/'letter/scripts'))
        import torch
        import numpy as np
        import boundary_compare
        import formal_compare
        import g3_identity
        from single_gpu_training import execution_schedule
        torch.set_num_threads(4)
        need(Path(boundary_compare.__file__).resolve()==imports[0].resolve(),'wrong boundary comparator import')
        need(Path(formal_compare.__file__).resolve()==imports[3].resolve(),'wrong recursive comparator import')
        g3_identity.verify_execution_profile(records[0]['execution_profile'])
        need(records[0]['execution_topology']==g3_identity.EXECUTION_TOPOLOGY,'single GPU topology')
        need(records[0]['experiment_id']==g3_identity.EXPERIMENT_ID,'fixed G3 experiment')
        need({k:v['sha256'] for k,v in records[0]['data_identity'].items()}==g3_identity.ASSETS,'fixed G3 assets')
        need(formal_compare.ATOL==1e-6 and formal_compare.RTOL==1e-3,'original fixed comparison criterion')
        schedule=execution_schedule(131413,preflight=True);micro=schedule['per_epoch_microbatches']
        gas=records[0]['execution_profile']['gradient_accumulation_steps']
        need(micro==20 and gas==8 and schedule['max_steps']==10,'expected frozen diagnostic schedule')
        for row in records:
            need(row['optimizer_membership_exact'] and row['optimizer_parameter_count']==g3_identity.TRAINABLE_PARAMETERS,
                 'complete optimizer membership required')
        report.update(run_commit=commit,config_sha256=config_sha,execution_topology=records[0]['execution_topology'],
                      execution_profile=records[0]['execution_profile'],sample_flow=schedule,
                      criterion={'atol':formal_compare.ATOL,'rtol':formal_compare.RTOL,
                                 'exact':['scheduler','all RNG','input/label/order','all discrete state'],
                                 'fixed_before_observation':True},
                      actual_import_files=[identity(path) for path in imports]+[identity(torch.__file__),identity(np.__file__)])
        states=[root/'boundary_pre_best_rank0.pt' for root in roots]
        report['large_state_identity']=[identity(path) for path in states]
        left,right=[torch.load(path,map_location='cpu',weights_only=False) for path in states]
        report['state_schema']=[boundary_compare.inspect_single_schema(state,label)
                                for state,label in ((left,'continuous'),(right,'resumed'))]
        need(left['global_step']==right['global_step']==10 and left['micro_steps']==right['micro_steps']==80 and
             left['accelerator_step']==right['accelerator_step']==80,'complete ten-update/eighty-microbatch trajectories')
        comparison=formal_compare.Compare(torch,np);comparison.compare(left,right)
        report['state_comparison']={'tensor_diagnostics':comparison.tensors,'numpy_arrays':comparison.arrays,
                                   'exact_scalar_leaves':comparison.exact_scalar_leaves,'failures':comparison.failures}
        need(not comparison.failures,'full trajectory state differs under original fixed criterion')
        del left,right
        row_evidence=[]
        for epoch in range(1,5):
            paths=[root/f'epoch{epoch}_rows.json' for root in roots]
            rows=[load(path) for path in paths]
            need(rows[0]==rows[1],'serialized loader epoch row order differs at epoch'+str(epoch))
            need(len(rows[0]['ranks'])==1 and len(rows[0]['ranks'][0])==160,'fixed actual short-epoch row scope')
            row_evidence.extend(identity(path) for path in paths)
        logs=[[json.loads(line) for line in (root/'events.rank0.jsonl').read_text().splitlines()] for root in roots]
        fields=('step','epoch_index','loader_yielded','input_sha256','label_sha256','shape')
        batches=[[{key:row[key] for key in fields} for row in log
                  if row['event']=='batch' and row.get('epoch_index',-1)>=1] for log in logs]
        need(len(batches[0])==len(batches[1])==60 and batches[0]==batches[1],
             'all sixty resumed input/label/loader microbatches must match exactly')
        def close(x,y):return math.isfinite(x) and math.isfinite(y) and abs(x-y)<=formal_compare.ATOL+formal_compare.RTOL*abs(x)
        loss=[[row for row in log if row['event']=='micro_loss' and row.get('epoch_index',-1)>=1] for log in logs]
        need(len(loss[0])==len(loss[1])==60 and all(close(x['loss'],y['loss']) for x,y in zip(*loss)),
             'resumed microbatch loss sequence exceeds original criterion')
        updates=[[row for row in log if row['event']=='optimizer_step' and row['step']>2] for log in logs]
        need(all([row['step'] for row in rows]==list(range(3,11)) for rows in updates),'optimizer update sequence3..10')
        need(all(x['lr']==y['lr'] for x,y in zip(*updates)),'external scheduler/LR sequence must match exactly')
        need(all(x['grad_norm'] is not None and y['grad_norm'] is not None and close(x['grad_norm'],y['grad_norm'])
                 for x,y in zip(*updates)),'gradient norms exceed original fixed criterion')
        partial=[row for row in logs[1] if row['event']=='partial_restore_exact']
        need(len(partial)==1 and partial[0]['micro_steps']==20 and partial[0]['global_steps']==2 and
             partial[0]['optimizer_micro_step_id']==3,'strict immediate partial-gradient restoration proof')
        starts=[(i,row) for i,row in enumerate(logs[1]) if row['event']=='train_begin']
        index,begin=starts[-1];attempt=begin['attempt_job_id']
        strict=[row for row in logs[1][:index] if row['event']=='strict_ds_restored' and row['attempt_job_id']==attempt]
        need(len(strict)==1 and strict[0]['global_step']==2,'strict native new-process checkpoint2 restoration')
        replayed=[row for row in logs[1] if row['event']=='loader_replayed' and row['attempt_job_id']==attempt]
        need(len(replayed)==1 and replayed[0]['checkpoint_step']==2,'restored loader/RNG cursor proof')
        validation=[[row['values']['eval_loss'] for row in log if row['event']=='log' and
                     'eval_loss' in row.get('values',{}) and row.get('epoch',0)>=2] for log in logs]
        need(len(validation[0])==len(validation[1])==3 and all(close(x,y) for x,y in zip(*validation)),
             'resumed validation losses exceed fixed criterion')
        report.update(status='PASS',row_order_exact=True,row_evidence=row_evidence,
                      complete_resumed_microbatches=60,input_label_batch_sequence_exact=True,
                      micro_loss_within_fixed_tolerance=True,optimizer_step_and_LR_exact=True,
                      gradient_norms_within_fixed_tolerance=True,validation_within_fixed_tolerance=True,
                      immediate_partial_restore=partial[0],native_strict_restore=strict[0],loader_replay=replayed[0],
                      actual_raw_files=[identity(root/name) for root in roots for name in ('rank0.json','events.rank0.jsonl')],
                      diagnostic_attempt_job_id=attempt,
                      early_update_family_report_read_or_modified=False,
                      limits='Diagnostic restoration evidence only. Failed early report remains failed; final fresh production validation is still required.')
    except BaseException as error:
        report['error']={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()}
    finally:
        report.update(finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started,
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        write(args.output,report)
    print(json.dumps({'status':report['status'],'scope':report['scope'],'output':str(args.output)}),flush=True)
    return 0 if report['status']=='PASS' else 2


if __name__=='__main__':raise SystemExit(main())
