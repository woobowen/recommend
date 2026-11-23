"""Detached GPU-locked preflight followed by one formal Beauty fit and three tests."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/home/jcy/GRID_wmh_Qwen_TIGER_20261009')
HERE=ROOT/'codex/letter_upstream_transformer_20261115'


def write(path,data):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    tmp.replace(path)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--group',type=Path)
    ap.add_argument('--gpu',type=int,default=0)
    ap.add_argument('--worker',action='store_true')
    args=ap.parse_args()
    if not args.worker:
        group=args.group or ROOT/'experiments'/('letter_official_beauty_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
        group=group.resolve();group.mkdir(parents=True,exist_ok=False)
        audit=json.loads((HERE/'inputs/beauty/mapping_audit.json').read_text())
        assert audit['complete']
        cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--group',str(group),'--gpu',str(args.gpu),'--worker']
        with (group/'supervisor.log').open('w') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,
                                   stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
        result=dict(group=str(group),supervisor_pid=child.pid,physical_gpu=args.gpu,
                    train_seed=42,eval_seed=2026,experiments=['unshuffle','shuffle_3','shuffle_4'],
                    upstream='official_LETTER_Beauty_SID',one_shared_best_checkpoint=True,
                    train_once=True,automatic_preflight_then_formal=True,automatic_restart=False,
                    started_at=datetime.now().isoformat())
        write(group/'launch_manifest.json',result);write(HERE/'latest_beauty_run.json',result)
        print(json.dumps(result,indent=2));return
    group=args.group.resolve();status_path=group/'job_status.json'
    status=dict(state='waiting_for_gpu',time=time.time(),physical_gpu=args.gpu,
                supervisor_pid=os.getpid(),session_id=os.getsid(0),group=str(group))
    write(status_path,status)
    try:
        locks=ROOT/'stage_gpu_locks';locks.mkdir(exist_ok=True)
        with (locks/f'gpu_{args.gpu}.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            used=int(subprocess.check_output(['nvidia-smi',f'--id={args.gpu}',
                '--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
            assert used<512,f'GPU {args.gpu} not idle: {used} MiB'
            for stage,preflight in [('preflight',True),('formal',False)]:
                output=group/stage
                cmd=[sys.executable,'-u',str(HERE/'run_beauty.py'),'--output',str(output),'--gpu',str(args.gpu)]
                if preflight:cmd.append('--preflight')
                with (group/f'{stage}.log').open('w') as log:
                    child=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,
                                           stdout=log,stderr=subprocess.STDOUT,close_fds=True)
                    status.update(state='running',stage=stage,pid=child.pid,output=str(output),
                                  argv=cmd,time=time.time());write(status_path,status)
                    code=child.wait()
                assert code==0,f'{stage} exited {code}; see {group}/{stage}.log'
                result=json.loads((output/'stage_result.json').read_text())
                assert result['completed'] and len(result['test_metrics'])==3
            status.update(state='completed',time=time.time(),metrics=str(group/'formal/metrics.md'))
            write(status_path,status)
    except BaseException as e:
        status.update(state='failed',error=str(e),time=time.time());write(status_path,status)
        raise


if __name__=='__main__':main()
