"""Detached diagnosis after GPU 2's final formal task, without changing that queue."""
import argparse
from datetime import datetime,timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
GROUP=HERE/'extended_seed43_queue'
OUT=ROOT/'experiments/yelp_diagnostic_patience40_seed43'
PREVIOUS=ROOT/'codex/yelp_aligned_v1/training_queue_v1/Y07_status.json'


def write(value):
    value.update(updated_utc=datetime.now(timezone.utc).isoformat())
    tmp=GROUP/'status.tmp'
    tmp.write_text(json.dumps(value,indent=2))
    tmp.replace(GROUP/'status.json')


def main():
    global GROUP
    ap=argparse.ArgumentParser();ap.add_argument('--worker',action='store_true');ap.add_argument('--immediate',action='store_true');args=ap.parse_args()
    if args.immediate:
        GROUP=HERE/'extended_seed43_immediate'
    if not args.worker:
        assert not GROUP.exists() and not OUT.exists()
        GROUP.mkdir()
        with (GROUP/'supervisor.log').open('w') as log:
            cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker']+(['--immediate'] if args.immediate else [])
            p=subprocess.Popen(cmd,cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
        print(json.dumps(dict(pid=p.pid,output=str(OUT),immediate=args.immediate)))
        return
    status=dict(state='waiting_for_formal_task',pid=os.getpid(),session_id=os.getsid(0),gpu=2,
        previous=str(PREVIOUS),output=str(OUT),patience=40,seed=43,variant='shuffle',automatic_restart=False)
    write(status)
    try:
        while not args.immediate:
            previous=json.loads(PREVIOUS.read_text())
            if previous['state']=='completed':
                result=json.loads((Path(previous['output'])/'stage_result.json').read_text())
                assert result['completed']
                break
            if previous['state'] in ('failed','blocked_previous_failed'):
                raise RuntimeError('Formal task failed; diagnosis will not start automatically')
            time.sleep(20)
        spec=importlib.util.spec_from_file_location('original_pipeline',ROOT/'_运行/run_pipeline.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        def waiting(reason):
            status.update(state='waiting_for_gpu',reason=reason);write(status)
        with module.reserve_stage_gpus(['2'],waiting):
            frozen=json.loads((ROOT/'codex/yelp_aligned_v1/training_queue_v1/frozen_inputs_sources.json').read_text())
            import hashlib
            for path,expected in frozen.items():
                assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
            cmd=[sys.executable,'-u',str(HERE/'run_extended_seed43.py'),'--dataset','yelp_aligned_v1',
                '--variant','shuffle','--gpu','2','--seed','43','--output',str(OUT),
                '--sample-dir',str(ROOT/'experiments/sid_order_yelp_5seed_v1/fixed_test_users')]
            with (GROUP/'training.log').open('w') as log:
                p=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,close_fds=True)
                status.update(state='running',training_pid=p.pid,argv=cmd);write(status)
                code=p.wait()
            assert code==0,f'Diagnostic failed with exit code {code}'
            assert json.loads((OUT/'stage_result.json').read_text())['completed']
            status.update(state='completed',exit_code=0);write(status)
    except BaseException as exc:
        status.update(state='failed',error=str(exc));write(status)
        raise


if __name__=='__main__':
    main()
