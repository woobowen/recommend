"""Two detached, independent GPU queues: official Instruments and Yelp SIDs."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/home/jcy/GRID_wmh_Qwen_TIGER_20261009')
HERE=ROOT/'codex/letter_official_downstream_20261116'
TASKS=[('instruments',0),('yelp',1)]


def write(path,value):
    temp=path.with_suffix(f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temp.replace(path)


def read(path):return json.loads(path.read_text())


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def now():return datetime.now(timezone.utc).isoformat()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--group',type=Path)
    ap.add_argument('--lane',type=int,choices=[0,1]);args=ap.parse_args()
    if args.lane is None:
        # A completed/failed queue remains inspectable; never duplicate an active queue.
        with (HERE/'launch.lock').open('a') as guard:
            fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
            previous=HERE/'latest_queue.json'
            if previous.exists():
                old=read(previous)
                for task in old['tasks']:
                    path=Path(old['group'])/f'{task["dataset"]}_status.json'
                    if path.exists():
                        st=read(path)
                        assert st['state'] in ['failed','completed'],f'Existing queue is active: {path}'
            group=args.group or ROOT/'experiments'/('letter_official_instruments_yelp_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
            group=group.resolve();assert not group.exists()
            audits={dataset:read(HERE/'inputs'/dataset/'mapping_audit.json') for dataset,_ in TASKS}
            assert all(a['complete'] for a in audits.values())
            group.mkdir(parents=True)
            frozen={str(p):sha(p) for p in [HERE/'run_dataset.py',HERE/'prepare_data.py',
                ROOT/'codex/plan_v1_1/protocol.py',ROOT/'codex/sid_order_launch/sid_order_shuffle_model.py',
                ROOT/'configs/experiment/tiger_train_flat.yaml']}
            for dataset,_ in TASKS:
                a=audits[dataset]
                frozen[str(HERE/'inputs'/dataset/'mapping_audit.json')]=sha(HERE/'inputs'/dataset/'mapping_audit.json')
                frozen[a['sid_path']]=a['sid_sha256']
                frozen.update(a['official_sources'])
                frozen.update({str(ROOT/p):h for p,h in a['data_files'].items()})
            write(group/'frozen_files.json',frozen)
            tasks=[]
            for lane,(dataset,gpu) in enumerate(TASKS):
                task=dict(dataset=dataset,physical_gpu=gpu,items=audits[dataset]['items'],
                    users=audits[dataset]['users'],output=str(group/dataset/'formal'))
                write(group/f'{dataset}_status.json',dict(state='queued',**task,updated_utc=now()))
                cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--group',str(group),'--lane',str(lane)]
                with (group/f'{dataset}_supervisor.log').open('w') as log:
                    child=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,
                        stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
                task.update(supervisor_pid=child.pid);tasks.append(task)
            manifest=dict(group=str(group),created_utc=now(),tasks=tasks,
                official_repository_commit='8d0154e28de37dbb6e24871c508ad8ddb1921cda',
                train_seed=42,eval_seed=2026,metrics=['recall@5','recall@10','ndcg@5','ndcg@10'],
                one_fresh_fit_per_dataset=True,modes=['unshuffle','shuffle_3','shuffle_4'],
                same_best_checkpoint_for_three_tests=True,automatic_preflight_then_formal=True,
                detached=True,automatic_restart=False,failure_scope='Stop only the failed dataset lane.')
            write(group/'launch_manifest.json',manifest);write(HERE/'latest_queue.json',manifest)
            print(json.dumps(manifest,indent=2),flush=True)
        return
    group=args.group.resolve();dataset,gpu=TASKS[args.lane]
    status_file=group/f'{dataset}_status.json'
    status=dict(state='waiting_for_gpu',dataset=dataset,physical_gpu=gpu,
                supervisor_pid=os.getpid(),session_id=os.getsid(0),updated_utc=now())
    def update(state,**extra):
        status.update(state=state,updated_utc=now(),**extra);write(status_file,status)
    def verify_frozen():
        for p,h in read(group/'frozen_files.json').items():assert sha(Path(p))==h,f'File changed: {p}'
    update('waiting_for_gpu')
    try:
        locks=ROOT/'stage_gpu_locks';locks.mkdir(exist_ok=True)
        with (locks/f'gpu_{gpu}.lock').open('a') as lock:
            while True:
                try:
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    used=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}',
                        '--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
                    if used<512:break
                    fcntl.flock(lock,fcntl.LOCK_UN)
                    update('waiting_for_gpu',reason=f'GPU memory used: {used} MiB')
                except BlockingIOError:
                    update('waiting_for_gpu',reason='Stage GPU lock held by another job')
                time.sleep(15)
            for stage,preflight in [('preflight',True),('formal',False)]:
                verify_frozen()
                output=group/dataset/stage
                assert not output.exists()
                cmd=[sys.executable,'-u',str(HERE/'run_dataset.py'),'--dataset',dataset,
                     '--gpu',str(gpu),'--output',str(output)]
                if preflight:cmd.append('--preflight')
                with (group/f'{dataset}_{stage}.log').open('w') as log:
                    child=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,
                        stdout=log,stderr=subprocess.STDOUT,close_fds=True)
                    update('running',stage=stage,pid=child.pid,output=str(output),argv=cmd)
                    code=child.wait()
                assert code==0,f'{stage} exited {code}; see {group}/{dataset}_{stage}.log'
                result=read(output/'stage_result.json')
                assert result['completed'] and result['dataset']==dataset
                assert set(result['test_metrics'])=={'unshuffle','shuffle_3','shuffle_4'}
                if not preflight:
                    assert all(t['full_test_set'] and t['evaluated_users']==result['users']
                               for t in result['test_metrics'].values())
            verify_frozen()
            update('completed',metrics=str(output/'metrics.md'),exit_code=0)
    except BaseException as exc:
        update('failed',error=str(exc));raise


if __name__=='__main__':main()
