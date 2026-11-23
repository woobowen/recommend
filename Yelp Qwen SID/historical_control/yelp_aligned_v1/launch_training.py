"""Launch the authorized Yelp plan in four detached, fail-stop GPU queues."""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
PLAN=HERE/'experiment_plan.json'
GROUP=HERE/'training_queue_v1'
UPSTREAM=ROOT/'experiments/yelp_rkmeans_qwen_aligned_v1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def write(path,value):
    path=Path(path)
    temp=path.with_suffix(f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def verify_frozen():
    frozen=json.loads((GROUP/'frozen_inputs_sources.json').read_text())
    for path,expected in frozen.items():
        assert sha(path)==expected, f'Frozen file changed: {path}'


def aggregate(plan):
    metrics=['test/recall@5','test/recall@10','test/ndcg@5','test/ndcg@10']
    values={mode:{metric:[] for metric in metrics} for mode in ['unshuffled','shuffle_3','shuffle_4','mlp']}
    per_seed={}
    for task in sorted(plan['tasks'],key=lambda t:(t['seed'],t['variant'])):
        result=json.loads((Path(task['output'])/'stage_result.json').read_text())
        assert result['completed'] and result['train_seed']==task['seed'] and not result['preflight_only']
        for mode,test in result['test_metrics'].items():
            assert test['full_test_set'] and not test['attention_export']
            record=test['metrics'][0]
            for metric in metrics:
                value=float(record[metric])
                assert math.isfinite(value)
                values[mode][metric].append(value)
                per_seed[(task['seed'],mode,metric)]=value
    rows=[]
    for mode,entries in values.items():
        for metric,seq in entries.items():
            assert len(seq)==5
            rows.append(dict(condition=mode,metric=metric,n=5,mean=statistics.mean(seq),std=statistics.stdev(seq)))
    paired=[]
    for mode in ('shuffle_3','shuffle_4','mlp'):
        for metric in metrics:
            deltas=[per_seed[s,mode,metric]-per_seed[s,'unshuffled',metric] for s in range(42,47)]
            paired.append(dict(comparison=mode+' - unshuffled',metric=metric,n=5,
                mean=statistics.mean(deltas),std=statistics.stdev(deltas),seed_deltas=deltas))
    write(GROUP/'five_seed_summary.json',dict(completed_utc=now(),summary=rows,paired_differences=paired))
    with (GROUP/'five_seed_summary.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=['condition','metric','n','mean','std'])
        writer.writeheader()
        writer.writerows(rows)


def refresh_plan():
    with (GROUP/'plan.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        plan=json.loads(PLAN.read_text())
        states=[]
        for task in plan['tasks']:
            status=json.loads((GROUP/f"{task['id']}_status.json").read_text())
            task.update(status=status['state'],completed=status['state']=='completed',status_file=str(GROUP/f"{task['id']}_status.json"))
            states.append(status['state'])
        plan.update(updated_utc=now(),formal_training_started=any(s in ('running','completed','failed') for s in states))
        if all(s=='completed' for s in states):
            aggregate(plan)
            plan['state']='completed'
        elif any(s in ('failed','blocked_previous_failed') for s in states):
            plan['state']='attention_required'
        else:
            plan['state']='running'
        write(PLAN,plan)


def run_lane(gpu):
    plan=json.loads((GROUP/'authorized_plan.json').read_text())
    tasks=[t for t in plan['tasks'] if t['gpu']==gpu]
    spec=importlib.util.spec_from_file_location('gpu_reservation',ROOT/'_运行/run_pipeline.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for index,task in enumerate(tasks):
        path=GROUP/f"{task['id']}_status.json"
        status=dict(task_id=task['id'],gpu=gpu,variant=task['variant'],train_seed=task['seed'],
            output=task['output'],state='waiting_for_gpu',supervisor_pid=os.getpid(),
            session_id=os.getsid(0),updated_utc=now())
        def update(state,**extra):
            status.update(state=state,updated_utc=now(),**extra)
            write(path,status)
        update('waiting_for_gpu')
        try:
            def waiting(reason):
                update('waiting_for_gpu',reason=reason)
            with module.reserve_stage_gpus([str(gpu)],waiting):
                verify_frozen()
                assert not Path(task['output']).exists(), 'Fresh output required'
                cmd=task['argv']
                assert '--preflight' not in cmd and '--prepare-only' not in cmd
                with (GROUP/f"{task['id']}.log").open('w') as log:
                    proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,
                        stderr=subprocess.STDOUT,close_fds=True)
                    update('running',pid=proc.pid,argv=cmd)
                    refresh_plan()
                    code=proc.wait()
                assert code==0, f"{task['id']} exited with code {code}"
                result=json.loads((Path(task['output'])/'stage_result.json').read_text())
                assert result['completed'] and not result['preflight_only']
                modes=['mlp'] if task['variant']=='mlp' else ['unshuffled','shuffle_3','shuffle_4']
                assert set(result['test_metrics'])==set(result['attention_groups'])==set(modes)
                for mode in modes:
                    assert result['test_metrics'][mode]['full_test_set']
                    assert result['attention_groups'][mode]==dict(users=100,complete=True)
                update('completed',exit_code=0)
                refresh_plan()
        except BaseException as exc:
            update('failed',error=str(exc))
            for later in tasks[index+1:]:
                write(GROUP/f"{later['id']}_status.json",dict(task_id=later['id'],gpu=gpu,
                    variant=later['variant'],train_seed=later['seed'],state='blocked_previous_failed',
                    previous_failed_task=task['id'],updated_utc=now()))
            refresh_plan()
            traceback.print_exc()
            raise


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--lane',type=int,choices=range(4))
    args=ap.parse_args()
    if args.lane is not None:
        run_lane(args.lane)
        return
    with (HERE/'training_launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not GROUP.exists(), 'Training queue already registered; inspect existing queue'
        plan=json.loads(PLAN.read_text())
        readiness=json.loads((UPSTREAM/'readiness_verified.json').read_text())
        assert readiness['ready_for_formal_training']
        assert plan['state']=='ready_not_started' and len(plan['tasks'])==10
        assert all(not Path(t['output']).exists() for t in plan['tasks'])
        frozen=dict(readiness['artifact_sha256'])
        frozen.update(json.loads((UPSTREAM/'original_code_hashes.json').read_text()))
        manifest=json.loads((Path(plan['data_dir'])/'file_manifest.json').read_text())
        frozen.update({str(Path(plan['data_dir'])/p):h for p,h in manifest.items()})
        # Match model/training/data code to the successful read-only preparation.
        checked=json.loads((UPSTREAM/'readonly_shuffle/source_manifest.json').read_text())['files']
        for path,expected in checked.items():
            if path.startswith(('src/','configs/','codex/plan_v1_1/','codex/sid_order_launch/')) or path=='codex/yelp_aligned_v1/run_task.py':
                assert sha(ROOT/path)==expected, f'Source differs from verified preparation: {path}'
        for folder,pattern in [('src','*.py'),('configs','*.yaml'),('codex/plan_v1_1','*.py'),
                               ('codex/sid_order_launch','*_model.py'),('codex/yelp_aligned_v1','*.py')]:
            frozen.update({str(p):sha(p) for p in (ROOT/folder).rglob(pattern)})
        assert all(sha(p)==h for p,h in frozen.items())
        GROUP.mkdir()
        write(GROUP/'frozen_inputs_sources.json',frozen)
        plan.update(user_request='Start training according to the saved Yelp five-seed plan.',
            authorization='Human user: 启动训练',authorized_utc=now(),downstream_training_authorized=True,
            auto_launch=True,state='launching',queue_dir=str(GROUP),automatic_retry=False)
        for task in plan['tasks']:
            task.update(status='queued',completed=False)
            write(GROUP/f"{task['id']}_status.json",dict(task_id=task['id'],gpu=task['gpu'],
                variant=task['variant'],train_seed=task['seed'],output=task['output'],state='queued',updated_utc=now()))
        write(GROUP/'authorized_plan.json',plan)
        write(PLAN,plan)
        supervisors=[]
        for gpu in range(4):
            with (GROUP/f'GPU{gpu}_supervisor.log').open('w') as log:
                proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--lane',str(gpu)],
                    cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                    start_new_session=True,close_fds=True)
            supervisors.append(dict(gpu=gpu,pid=proc.pid,tasks=[t['id'] for t in plan['tasks'] if t['gpu']==gpu]))
        launch=dict(launched_utc=now(),supervisors=supervisors,
            detached=True,stdin='/dev/null',logs_persistent=True,automatic_retry=False,
            failed_lane_stops=True,aggregate_after_all_10_complete=True)
        write(GROUP/'launch_manifest.json',launch)
        print(json.dumps(launch,indent=2),flush=True)


if __name__=='__main__':
    main()
