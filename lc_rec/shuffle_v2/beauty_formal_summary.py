"""Reduce the seven live Beauty observations and estimate remaining time."""
import argparse
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

PREFIX = 'letter-beauty-v2-'
USERS = 22363
MODE_DISPLAY = {'shuffle2':'S2', 'shuffle3a':'S3a', 'shuffle3b':'S3b', 'shuffle4':'S4'}


def classify(job):
    for stem,stage in (('gpu-formal-', 'generation'), ('cpu-eval-', 'evaluation')):
        if job.startswith(PREFIX+stem):
            tail=job.removeprefix(PREFIX+stem)
            return tail.rsplit('-',1)[0],stage
    return None,'paired-statistics'


def summarize(records, plan, assessment):
    timezone = ZoneInfo('America/New_York')
    times = [datetime.fromisoformat(r['observed_utc']) for r in records]
    assert len(times)==7 and (times[-1]-times[0]).total_seconds()>=1800
    assert all(a<b for a,b in zip(times,times[1:]))
    expected_uuid = plan['modes']['shuffle2']['gpu']['env']['CUDA_VISIBLE_DEVICES']
    parent_id = records[0]['parent_id']
    known_children = {r['job_id'] for phase in plan['modes'].values() for r in phase.values()}
    known_children.add(plan['paired']['job_id'])
    summaries = []
    last_by_job = {}
    for i,(r,timestamp) in enumerate(zip(records,times,strict=True)):
        assert r['parent_id']==parent_id and r['jobs'][parent_id]['status']=='RUNNING'
        assert not r.get('unexpected_active_formal')
        running_gpu = [job for job in known_children if job.startswith(PREFIX+'gpu-formal-')
                       and r['jobs'].get(job,{}).get('status')=='RUNNING']
        assert len(running_gpu)<=1
        active = [job for job in known_children if r['jobs'].get(job,{}).get('status')=='RUNNING']
        assert len(active)==1,active
        child=active[0]
        mode,stage=classify(child)
        job=r['jobs'][child]
        assert job.get('worker_pid_identity_alive') or (job.get('live_members') and job['supervisor_alive'])
        rank=r['rank_reports'].get(child,{})
        if rank:
            assert rank['assigned_users']==USERS or (i==0 and rank['assigned_users'] is None)
            assert not rank.get('error')
            if i:
                assert (rank['world_size'],rank['rank'],rank['local_rank'])==(1,0,0)
                assert rank['RUN_COMMIT']==plan['formal_run_commit']
        output=r['outputs'].get(child, {})
        count=output.get('complete_lines',0)
        size=output.get('snapshot_size',0)
        assert count>=rank.get('completed_users',0)
        if stage=='generation':
            observed = [g for g in r['gpu_compute'] if g['owned_formal_job_id']]
            assert len(observed)==1 and observed[0]['uuid']==expected_uuid and observed[0]['owned_formal_job_id']==child
            assert observed[0]['pid']==rank['pid']
        card=next(g for g in r['gpu_records'] if g['uuid']==expected_uuid)
        prior_count,prior_size=last_by_job.get(child,(0,0))
        assert count>=prior_count and size>=prior_size
        delta=count-prior_count
        last_by_job[child]=(count,size)
        summaries.append({'requested_minute':5*i,'actual_elapsed_seconds':(timestamp-times[0]).total_seconds(),
            'observed_utc':r['observed_utc'],'observed_new_york':timestamp.astimezone(timezone).isoformat(),
            'parent_id':parent_id,'child_id':child,'mode':mode,'stage':'loading' if not count and stage=='generation' else stage,
            'complete_users':count,'total_users':USERS if mode else None,'interval_user_increment':delta,
            'candidate_count_by_contract':20*count,'prediction_bytes':size,'interval_byte_growth':size-prior_size,
            'prediction_mtime_ns':output.get('mtime_ns'),'predictions_checkpoint_users':rank.get('completed_users'),
            'gpu_physical_index':card['physical_index'],'gpu_uuid':expected_uuid,'gpu_logical_device':0,
            'gpu_memory_mib':card['memory_used_mib'],'gpu_utilization_percent':card['utilization_percent'],
            'disk_free_bytes':r.get('disk_free_bytes'),'disk_df_at_t0':r.get('disk_df'),
            'cumulative_budget_spend':r['cumulative_budget_spend'],
            'rank_pid':rank.get('pid'),'world_size':rank.get('world_size'),
            'job_worker_pid':job['worker']['pid'],'worker_identity_alive':True})
    for i in range(2,len(summaries)):
        a,b,c=summaries[i-2:i+1]
        if a['child_id']==b['child_id']==c['child_id'] and b['stage']==c['stage']=='generation':
            assert not (b['interval_user_increment']==c['interval_user_increment']==0
                        and b['interval_byte_growth']==c['interval_byte_growth']==0), 'two intervals without progress'
    recent=summaries[-1]
    first_positive=next(r for r in summaries if r['child_id']==recent['child_id'] and r['complete_users']>0)
    seconds=recent['actual_elapsed_seconds']-first_positive['actual_elapsed_seconds']
    increment=recent['complete_users']-first_positive['complete_users']
    assert seconds>0 and increment>0
    rate=increment/seconds
    remaining_users=USERS-recent['complete_users']
    own=remaining_users/rate
    all_modes=list(plan['modes'])
    later=all_modes[all_modes.index(recent['mode'])+1:]
    group=own+sum(assessment['modes'][m]['formal_wall_projection_seconds'] for m in later)
    return {'status':'PASS_30MIN_FORMAL_BEAUTY_OBSERVATION','observations':summaries,
            'observation_duration_seconds':(times[-1]-times[0]).total_seconds(),
            'projection':{'scope':'extrapolation as of T+30; measured wall rate from first positive observation to T+30',
                'measured_interval_seconds':seconds,'measured_complete_user_increment':increment,
                'measured_users_per_minute':60*rate,'current_remaining_users':remaining_users,
                'current_remaining_seconds':own,'later_unstarted_modes':later,
                'later_estimate_source':'same 32-user preflight timing estimate per mode incl. model load',
                'estimated_group_remaining_gpu_seconds':group,'cpu_evaluation_reserve_per_mode_seconds':600,
                'pairing_reserve_seconds':2100,'excludes_cpu_and_review_from_gpu_estimate':True},
            'formal_metrics':'pending full generation and CPU evaluation',
            'cpu_wall_budget_uses_runner_workers':True}


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--observations',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--assessment',type=Path,required=True)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    records=[json.loads((args.observations/f't{minute:02}.json').read_text()) for minute in range(0,31,5)]
    result=summarize(records,json.loads(args.plan.read_text()),json.loads(args.assessment.read_text()))
    text=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(text)
    print(text,end='')


if __name__=='__main__':
    main()
