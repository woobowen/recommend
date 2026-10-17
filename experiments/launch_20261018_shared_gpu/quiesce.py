"""Precisely stop only the old CPU controller's future submissions, not children."""
import argparse
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import time


def identity(pid):
    p=Path('/proc')/str(pid)
    try:
        fields=(p/'stat').read_text().rsplit(')',1)[1].split()
        if fields[0]=='Z':return None
        return {'pid':pid,'start_ticks':fields[19],'uid':p.stat().st_uid,
            'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'argv':(p/'cmdline').read_bytes().decode().rstrip('\0').split('\0')}
    except (FileNotFoundError,ProcessLookupError):return None


def main():
    a=argparse.ArgumentParser();a.add_argument('--config',type=Path,required=True)
    a.add_argument('--expected-config-sha256',required=True);a.add_argument('--output',type=Path,required=True)
    args=a.parse_args();raw=args.config.read_bytes()
    assert not args.output.exists(),'quiescence receipt is immutable; inspect prior receipt before re-entry'
    assert hashlib.sha256(raw).hexdigest()==args.expected_config_sha256
    config=json.loads(raw);root=Path(config['runner']['job_root']);outer=root/config['queue_id']
    source=Path(config['snapshot']['root'])/'experiments/launch_20261017/queue.py'
    spec=importlib.util.spec_from_file_location('original_queue_quiesce',source)
    q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
    original_request=q.read(outer/'request.json')
    bound={x['path']:x['sha256'] for x in original_request['inputs']}
    assert bound[str(args.config)]==args.expected_config_sha256
    deployment=Path(config['snapshot']['root'])/'DEPLOYMENT.json'
    assert q.sha(deployment)==bound[str(deployment)]
    manifest=q.read(deployment);entry=next(x for x in manifest['files'] if x['path']=='experiments/launch_20261017/queue.py')
    assert q.sha(source)==entry['sha256']
    before=q.runner_call(config,'query',outer)
    expected=hashlib.sha256(q.canonical(original_request)).hexdigest();assert before['request_sha256']==expected
    signaled=None
    if before['status'] in q.ACTIVE:
        worker=before['worker'];assert worker['uid']==os.getuid() and identity(worker['pid'])==worker
        assert worker['argv']==original_request['argv']
        assert worker['argv'][-2:]==['--config',str(args.config)]
        for stage in config['stages']:
            job=root/stage['request']['job_id']
            if (job/'accepted.json').exists():
                child=q.runner_call(config,'query',job)
                if child['status'] in q.ACTIVE:
                    assert child['worker_group']['pgid']!=worker['pid']
                    assert child['worker_group']['session_id']!=worker['pid']
        # pidfd binds the signal to this exact process, never a reused PID/group.
        fd=os.pidfd_open(worker['pid'])
        try:
            assert identity(worker['pid'])==worker
            signal.pidfd_send_signal(fd,signal.SIGTERM,None,0)
            signaled=dict(worker,signal='SIGTERM',at=q.utc(),scope='old CPU queue worker only')
        finally:os.close(fd)
    deadline=time.monotonic()+50
    while True:
        after=q.runner_call(config,'query',outer)
        state=q.read(Path(config['state_dir'])/'queue-status.json')
        stopped=(state['status'] in {'STOPPED_NO_NEW_SUBMISSIONS','TERMINAL_PENDING_RESULT_REVIEW'}
            and after.get('cleanup_verified') and not any(after.get(k) for k in ['worker_alive','supervisor_alive','workgroup_live','worker_may_be_live'])
            and after['status'] not in q.ACTIVE)
        if stopped:break
        assert time.monotonic()<deadline,'old controller quiescence not proven; do not launch continuation'
        time.sleep(1)
    assert not any(v['status']=='SUBMITTING' for v in state['stages'].values())
    with (Path(config['state_dir'])/'supervisor.lock').open('r+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state_sha=q.sha(Path(config['state_dir'])/'queue-status.json')
        inventory={}
        for stage in config['stages']:
            job=root/stage['request']['job_id']
            if not (job/'accepted.json').exists():
                assert not job.exists(),'unaccepted existing job directory requires manual identity resolution'
                inventory[stage['id']]={'submitted':False,'planned_job_id':stage['request']['job_id']}
                continue
            actual=q.runner_call(config,'query',job)
            expected_request=q.wrapped_request(config,args.config,stage)
            assert q.read(job/'request.json')==expected_request
            assert actual['request_sha256']==hashlib.sha256(q.canonical(expected_request)).hexdigest()
            inventory[stage['id']]={'submitted':True,'actual':actual,'request':expected_request}
        assert state_sha==q.sha(Path(config['state_dir'])/'queue-status.json')
    receipt={'status':'OLD_CONTROLLER_QUIESCED','at':q.utc(),'old_config_path':str(args.config),
        'old_config_sha256':args.expected_config_sha256,'old_queue_request_path':str(outer/'request.json'),
        'old_queue_request_sha256':q.sha(outer/'request.json'),'old_queue_state_path':str(Path(config['state_dir'])/'queue-status.json'),
        'old_queue_state_sha256':state_sha,'old_queue_id':config['queue_id'],'before':before,'after':after,
        'signal':signaled,'old_state':state,'final_inventory':inventory,'old_lock_released':True,
        'original_absolute_queue_deadline':(dt.datetime.fromisoformat(state['started_at'])+dt.timedelta(seconds=config['max_queue_seconds'])).isoformat(),
        'research_jobs_stopped':0,'resubmitted_jobs':0,'old_source_and_config_modified':False}
    assert not args.output.exists(),'quiescence receipt is immutable'
    args.output.parent.mkdir(parents=True,exist_ok=True);q.atomic(args.output,receipt)
    print(json.dumps(receipt))


if __name__=='__main__':main()
