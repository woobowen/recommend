"""A2-only at-most-once queue using the accepted shared GPU scheduler.

Old controllers, failed quantization and all 34 historical stages remain read
only. No encode request or retry of a received job is present in this queue.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

from experiments.launch_20261017 import queue

STAGES = {'sportsa2-quantize','sportsa2-metadata','sportsa2-gpu-gate','sportsa2-train',
          'sportsa2-eval-0','sportsa2-eval-3','sportsa2-eval-4',
          'sportsa2-attention-0','sportsa2-attention-3','sportsa2-attention-4','sportsa2-stats'}


def require(value, message):
    if not value:
        raise ValueError(message)


def lineage(config, *, live):
    proof=config['recovery']
    records={x['path']:x for x in proof['frozen_evidence']}
    for path,record in records.items():
        require(queue.sha(path)==record['sha256'],'frozen lineage evidence changed: '+path)
    shared=queue.read(proof['shared_config_path'])
    state=queue.read(proof['shared_state_path'])
    require(state['status']=='TERMINAL_PENDING_RESULT_REVIEW','shared queue was not terminal')
    require(all(x['status']!='SUBMITTING' for x in state['stages'].values()),'unresolved historical submission')
    require({s['id'] for s in config['stages']}==STAGES and len(config['stages'])==11,'A2-only stage scope changed')
    require(set(config['blocked_workflows'])==queue.WORKFLOWS-{'sports-a2'},'historical workflows must not run')
    require(config['runner']==shared['runner'] and config['gpu_locks_dir']==shared['gpu_locks_dir'],
            'accepted runner or shared GPU locks changed')
    require(config['max_storage_bytes']==shared['max_storage_bytes'] and config['free_space']==shared['free_space'],
            'storage protection changed')
    require(set(shared['storage_roots']) <= set(config['storage_roots']),'old outputs omitted from storage accounting')
    old_absolute=shared['continuation']['original_absolute_queue_deadline']
    require(proof['original_absolute_queue_deadline']==old_absolute,'original absolute deadline changed')
    historical=queue.accounting(shared,state)
    require(abs(historical['used_gpu_request_seconds']-proof['historical_stage_gpu_seconds'])<1e-6,
            'historical accounting changed')
    for group,cap in config['budget_groups'].items():
        require(cap<=shared['budget_groups'][group]-historical['groups'][group]+1e-6,
                'group allowance was reset: '+group)
    require(config['gpu_budget_seconds']<=shared['gpu_budget_seconds']-historical['used_gpu_request_seconds']+1e-6,
            'overall allowance was reset')
    old_by={s['id']:s for s in shared['stages']}
    for stage in config['stages']:
        original=old_by[stage['id']]
        for key in ['workflow','kind','budget_group','gpu_uuids','gpu_candidates','resource_policy']:
            require(stage.get(key)==original.get(key),'shared resource/protocol setting changed: '+stage['id']+':'+key)
        require(stage['request']['job_id']!=original['request']['job_id'],'old job identity reused')
        require(stage['request']['timeout_seconds']<=original['request']['timeout_seconds'],'stage wall cap increased')
    quant=next(s for s in config['stages'] if s['kind']=='rkmeans')
    require(quant['request']['timeout_seconds']*2<=config['budget_groups']['sports-a2-rkmeans'],
            'quantizer exceeds its original remaining two-card allowance')
    root=Path(config['snapshot']['root'])
    require(root in Path(__file__).resolve().parents,'controller is outside its fixed snapshot')
    deployment=queue.read(root/'DEPLOYMENT.json')
    require(deployment['source_commit']==config['snapshot']['commit'],'new deployment commit mismatch')
    require(proof['source_review_status']=='FD_ENGINEERING_EQUIVALENCE_PASS','repair not admitted')
    if live:
        for job in proof['terminal_controller_ids']:
            actual=queue.runner_call(config,'query',Path(config['runner']['job_root'])/job)
            require(actual.get('cleanup_verified') and not any(actual.get(k) for k in
                    ['supervisor_alive','worker_alive','workgroup_live','worker_may_be_live']),
                    'old controller is not verified empty: '+job)
        for stage in shared['stages']:
            job=Path(config['runner']['job_root'])/stage['request']['job_id']
            if (job/'accepted.json').exists():
                actual=queue.runner_call(config,'query',job)
                require(actual.get('cleanup_verified') and not any(actual.get(k) for k in
                    ['supervisor_alive','worker_alive','workgroup_live','worker_may_be_live']),
                    'historical worker live or unconfirmed: '+job.name)
                if stage['id']=='sportsa2-quantize':
                    require(actual['status']=='INTERRUPTED' and actual['exit_code']==-9,'old failure identity changed')
                else:
                    require(actual['status']=='SUCCEEDED' and actual['exit_code']==0,'completed historical job changed')
            else:
                require(not job.exists(),'unresolved original unsubmitted job directory: '+job.name)
        for stage in config['stages']:
            require(not (Path(config['runner']['job_root'])/stage['request']['job_id']).exists(),
                    'new job ID already occupied; query, never resubmit')
    return historical


def install(config, path, *, live):
    history=lineage(config,live=live)
    base_wrapped=queue.wrapped_request
    base_accounting=queue.accounting
    base_gates=queue.check_gates
    absolute=dt.datetime.fromisoformat(config['recovery']['original_absolute_queue_deadline'])

    def wrapped(current, config_path, stage):
        result=base_wrapped(current,config_path,stage)
        script=str(Path(__file__).resolve())
        result['argv']=[result['argv'][0],'-u',script,'run-stage','--config',str(Path(config_path).resolve()),'--stage',stage['id']]
        inputs={r['path']:r for r in result['inputs']}
        inputs[script]=dict(path=script,sha256=queue.sha(script))
        result['inputs']=list(inputs.values())
        return result

    def accounting(current,state):
        value=base_accounting(current,state)
        total=history['used_gpu_request_seconds']+current['recovery']['prior_gpu_request_seconds']+value['used_gpu_request_seconds']
        value.update(historical_stage_gpu_seconds=history['used_gpu_request_seconds'],
                     prior_gpu_request_seconds=current['recovery']['prior_gpu_request_seconds'],
                     total_used_gpu_request_seconds=total,original_total_gpu_budget_seconds=143100,
                     original_total_remaining_gpu_seconds=143100-total,
                     original_absolute_queue_deadline=absolute.isoformat())
        return value

    def gates(stage):
        require(dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=stage['request']['timeout_seconds']+60)<=absolute,
                'original absolute deadline cannot contain this unchanged stage budget')
        return base_gates(stage)

    queue.wrapped_request=wrapped
    queue.accounting=accounting
    queue.check_gates=gates
    queue.queue_should_stop=lambda current,started:dt.datetime.now(dt.timezone.utc)>=absolute-dt.timedelta(seconds=60)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['validate','render-requests','supervise','run-stage','status'])
    p.add_argument('--config',required=True,type=Path);p.add_argument('--stage')
    args=p.parse_args();config=queue.read(args.config)
    if args.mode=='status':
        print(json.dumps(queue.read(Path(config['state_dir'])/'queue-status.json'),indent=2));return 0
    queue.validate(config,args.config)
    install(config,args.config,live=args.mode in ['validate','supervise'])
    if args.mode=='validate':
        print(json.dumps(dict(status='A2_ONLY_LINEAGE_BUDGET_DEADLINE_PASS',stages=11,
          summary=queue.validate(config,args.config),deadline=config['recovery']['original_absolute_queue_deadline'])));return 0
    if args.mode=='render-requests':
        print(json.dumps([queue.wrapped_request(config,args.config,s) for s in config['stages']],indent=2));return 0
    if args.mode=='run-stage':return queue.run_stage(config,args.config,args.stage)
    return queue.supervise(config,args.config)


if __name__=='__main__':
    try:sys.exit(main())
    except BaseException as error:
        if isinstance(error,SystemExit):raise
        print(json.dumps(dict(error=repr(error),action='STOP_NO_AUTOMATIC_RETRY')),flush=True);sys.exit(2)
