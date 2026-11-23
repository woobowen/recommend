"""Finite preflight queue: wait for verified weights, run two smoke tests."""
from pathlib import Path
import datetime
import fcntl
import json
import os
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
launch=HERE/'runs'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
launch.mkdir(parents=True)
status={'pid':os.getpid(),'launch':str(launch),'formal_training':False,'stage':'waiting_for_verified_weights'}
(HERE/'latest_queue.json').write_text(json.dumps(status,indent=2))

def update(stage,**kw):
    status.update(stage=stage,time_unix=time.time(),**kw)
    tmp=launch/'status.json.tmp';tmp.write_text(json.dumps(status,indent=2));tmp.replace(launch/'status.json')
    print(json.dumps(status),flush=True)

try:
    update('waiting_for_verified_weights')
    deadline=time.monotonic()+3*3600
    while not (HERE/'assets_audit.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Weights not ready after 3 hours')
        time.sleep(10)
    report=json.loads((HERE/'assets_audit.json').read_text())
    assert all(r['downloaded'] for r in report['model_files'])
    # Hold locks through both tests; never reserve GPUs 2/3.
    locks=[]
    for i in (0,1):
        f=(HERE/f'gpu{i}.lock').open('w');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    gpu_csv=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    occupied={int(line.split(',')[0]):int(line.split(',')[1]) for line in gpu_csv.splitlines()}
    assert all(occupied[i]<100 for i in (0,1)),occupied
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',TOKENIZERS_PARALLELISM='false',
             HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled')
    for ds in ['Instruments','Beauty']:
        update('running_smoke',dataset=ds)
        with (launch/f'{ds}.log').open('wb',buffering=0) as log:
            p=subprocess.Popen([sys.executable,str(HERE/'smoke_test.py'),'--dataset',ds,
                '--model',str(ROOT/'models/huggyllama-llama-7b'),'--out',str(launch/ds)],
                stdout=log,stderr=log,env=env)
            update('running_smoke',dataset=ds,worker_pid=p.pid)
            rc=p.wait()
        if rc:raise RuntimeError(f'{ds} smoke exited {rc}; inspect {launch/ds}')
        result=json.loads((launch/ds/'result.json').read_text())
        assert result['complete']
    update('complete',complete=True)
except Exception as exc:
    update('failed',error=repr(exc),complete=False)
    raise
