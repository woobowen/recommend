"""Queue only T09-T12 with unchanged v1.1 training code and per-GPU locks."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
JOBS=[('T09',0,'toys','shuffle'),('T10',1,'toys','mlp'),
      ('T11',2,'beauty','shuffle'),('T12',3,'beauty','mlp')]


def write(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2),encoding='utf-8')
    temp.replace(path)


def verify_source():
    reference=ROOT/'experiments/sid_order_5seed_v1/beauty/shuffle/seed_42/source_manifest.json'
    hashes=json.loads(reference.read_text())['files']
    # Freeze all original model/data code, configs, and the executed protocol.
    for name,value in hashes.items():
        if name.startswith(('src/','configs/','codex/sid_order_launch/')) or name in (
            'codex/plan_v1_1/run_task.py','codex/plan_v1_1/protocol.py','codex/plan_v1_1/attention.py'):
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value, f'Source changed: {name}'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--group',type=Path,required=True)
    ap.add_argument('--lane',type=int,choices=range(4))
    args=ap.parse_args()
    group=args.group.resolve()
    verify_source()
    if args.lane is None:
        assert not group.exists(),'Round already registered'
        for _,_,dataset,variant in JOBS:
            assert not (ROOT/f'experiments/sid_order_5seed_v1/{dataset}/{variant}/seed_44').exists()
        previous_group=ROOT/'codex/second_round_20261104'
        for gpu in range(4):
            previous=json.loads((previous_group/f'T{gpu+5:02d}_status.json').read_text())
            assert previous['gpu']==gpu and previous['train_seed']==43
            assert previous['state'] in ('waiting_for_previous_task','running','completed'), 'Previous task failed'
        group.mkdir(parents=True)
        lanes=[]
        for task,gpu,dataset,variant in JOBS:
            command=[sys.executable,str(Path(__file__).resolve()),'--group',str(group),'--lane',str(gpu)]
            with (group/f'{task}_supervisor.log').open('w') as log:
                proc=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,
                    stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
            lanes.append(dict(task_id=task,gpu=gpu,dataset=dataset,variant=variant,train_seed=44,supervisor_pid=proc.pid))
        report=dict(plan_version='1.1',round=3,time=time.time(),lanes=lanes,
            automatic_later_rounds=False,automatic_restart=False,wait_for_previous_task=True)
        write(group/'launch_manifest.json',report)
        print(json.dumps(report,indent=2),flush=True)
        return
    task,gpu,dataset,variant=JOBS[args.lane]
    out=ROOT/f'experiments/sid_order_5seed_v1/{dataset}/{variant}/seed_44'
    previous=ROOT/f'codex/second_round_20261104/T{gpu+5:02d}_status.json'
    path=group/f'{task}_status.json'
    status=dict(task_id=task,gpu=gpu,dataset=dataset,variant=variant,train_seed=44,
        supervisor_pid=os.getpid(),session_id=os.getsid(0),state='waiting_for_previous_task',
        output=str(out),previous_task=str(previous),time=time.time())
    write(path,status)
    try:
        with (ROOT/f'stage_gpu_locks/gpu_{gpu}.lock').open('a') as lock:
            while True:
                # A later lane must wait even when its predecessor is itself queued.
                old=json.loads(previous.read_text())
                if old['state']=='failed':
                    raise RuntimeError('Previous task failed; manual review required')
                if old['state']!='completed':
                    time.sleep(15)
                    continue
                try:
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    time.sleep(15)
                    continue
                old=json.loads(previous.read_text())
                assert old['state']=='completed' and old.get('exit_code')==0,'Previous task failed; manual review required'
                result=json.loads((Path(old['output'])/'stage_result.json').read_text())
                assert result['completed'],'Previous test/attention incomplete'
                usage=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}',
                    '--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
                if usage<512:
                    break
                fcntl.flock(lock,fcntl.LOCK_UN)
                time.sleep(15)
            verify_source()
            command=[sys.executable,'-u',str(ROOT/'codex/plan_v1_1/run_task.py'),
                '--dataset',dataset,'--variant',variant,'--gpu',str(gpu),'--seed','44',
                '--output',str(out),'--sample-dir',str(ROOT/'experiments/sid_order_5seed_v1/fixed_test_users')]
            with (group/f'{task}.log').open('w') as log:
                proc=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,
                    stderr=subprocess.STDOUT,close_fds=True)
                status.update(state='running',pid=proc.pid,argv=command,time=time.time())
                write(path,status)
                code=proc.wait()
            status.update(state='completed' if code==0 else 'failed',exit_code=code,time=time.time())
            write(path,status)
    except BaseException as exc:
        status.update(state='failed',error=str(exc),time=time.time())
        write(path,status)
        raise


if __name__=='__main__':
    main()
