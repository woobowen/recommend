"""Detach exactly the four first-round tasks; no automatic later seeds/restarts."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
JOBS=[('T01',0,'beauty','shuffle'),('T02',1,'beauty','mlp'),
      ('T03',2,'toys','shuffle'),('T04',3,'toys','mlp')]


def write(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2),encoding='utf-8')
    temp.replace(path)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--group',type=Path,required=True)
    ap.add_argument('--preflight',action='store_true')
    ap.add_argument('--lane',type=int,choices=range(4))
    args=ap.parse_args()
    group=args.group.resolve()
    if args.lane is None:
        if group.exists():
            ap.error('Group must be new')
        if not args.preflight:
            for _,gpu,dataset,variant in JOBS:
                output=ROOT/f'experiments/sid_order_5seed_v1/{dataset}/{variant}/seed_42'
                assert not output.exists(), f'Existing task output {output}'
        group.mkdir(parents=True)
        lanes=[]
        for task_id,gpu,dataset,variant in JOBS:
            cmd=[sys.executable,str(Path(__file__).resolve()),'--group',str(group),'--lane',str(gpu)]
            if args.preflight:
                cmd.append('--preflight')
            with (group/f'{task_id}_supervisor.log').open('w') as log:
                proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,start_new_session=True,close_fds=True)
            lanes.append(dict(task_id=task_id,gpu=gpu,dataset=dataset,variant=variant,
                train_seed=42,supervisor_pid=proc.pid))
        manifest=dict(plan_version='1.1',time=time.time(),preflight=args.preflight,
            detachment='new session; stdin /dev/null; stdout/stderr persistent files',
            automatic_later_rounds=False,automatic_restart=False,lanes=lanes)
        write(group/'launch_manifest.json',manifest)
        print(json.dumps(manifest,indent=2),flush=True)
        return
    task_id,gpu,dataset,variant=JOBS[args.lane]
    locks=ROOT/'stage_gpu_locks'
    locks.mkdir(exist_ok=True)
    status=dict(task_id=task_id,dataset=dataset,variant=variant,train_seed=42,gpu=gpu,
        supervisor_pid=os.getpid(),session_id=os.getsid(0),state='preparing',time=time.time())
    status_path=group/f'{task_id}_status.json'
    write(status_path,status)
    try:
        with (locks/f'gpu_{gpu}.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            usage=subprocess.check_output(['nvidia-smi',f'--id={gpu}',
                '--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
            assert int(usage.strip())<512, f'GPU {gpu} not idle'
            out=(group/f'{dataset}_{variant}') if args.preflight else ROOT/f'experiments/sid_order_5seed_v1/{dataset}/{variant}/seed_42'
            cmd=[sys.executable,'-u',str(ROOT/'codex/plan_v1_1/run_task.py'),'--dataset',dataset,
                '--variant',variant,'--gpu',str(gpu),'--seed','42','--output',str(out),
                '--sample-dir',str(ROOT/'experiments/sid_order_5seed_v1/fixed_test_users')]
            if args.preflight:
                cmd.append('--preflight')
            with (group/f'{task_id}.log').open('w') as log:
                proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,
                    stderr=subprocess.STDOUT,close_fds=True)
                status.update(state='running',pid=proc.pid,output=str(out),argv=cmd,time=time.time())
                write(status_path,status)
                code=proc.wait()
            status.update(state='completed' if code==0 else 'failed',exit_code=code,time=time.time())
            write(status_path,status)
    except BaseException as exc:
        status.update(state='failed',error=str(exc),time=time.time())
        write(status_path,status)
        raise


if __name__=='__main__':
    main()
