#!/usr/bin/env python3
"""Server-owned A0 train -> best adapter -> all-user test with one budget.

Runs within the accepted at-most-once runner. Torchrun owns its worker sessions
and forwards termination; the runner checks every surviving marked session.
A persistent seed ledger prevents duplicate/rebudgeted
attempts. Checkpoint payloads stay on /data, not in runner's small export tree.
"""
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from formal_train import write,sha
from formal_ledger import reconcile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['release','run-commit','job-id','kind','precision-approval']:
        p.add_argument('--'+n,required=True)
    p.add_argument('--resume');p.add_argument('--resume-evaluation',action='store_true')
    p.add_argument('--prelaunch-review');a=p.parse_args()
    assert a.kind in ['acceptance-continuous','acceptance-resume','formal']
    root=Path('/data/bwwu/lc-rec/a0-normal-20261111');root.mkdir(parents=True,exist_ok=True)
    job=Path(os.environ['GRID_JOB_DIR']);assert job.name==a.job_id
    scripts=Path(a.release)/'lc_rec/letter/scripts'
    config=Path(a.release)/'lc_rec/letter/config/formal_instruments_seed42.json'
    if a.kind=='formal':
        assert a.prelaunch_review
        gate=json.loads(Path(a.prelaunch_review).read_text())
        assert gate['status']=='PASS' and gate['RUN_COMMIT']==a.run_commit
        assert gate['config_sha256']==sha(config)
        assert gate['formal_train_sha256']==sha(scripts/'formal_train.py')
    source=Path(a.release)/'lc_rec/letter/vendor/LETTER/LETTER-LC-Rec'
    data=source.parent/'data'
    model='/data/bwwu/lc-rec/assets/llama-7b/4782ad278652c7c71b72204d462d6d01eaaf7549'
    output=root/a.job_id;output.mkdir(exist_ok=True)
    training=(root/'formal_seed42/training') if a.kind=='formal' else output/'training'
    training.parent.mkdir(exist_ok=True,parents=True)
    if a.resume and a.kind=='formal':assert Path(a.resume).resolve().parent==training.resolve()
    if a.resume and a.kind=='formal':
        complete=sorted((x for x in training.glob('checkpoint-*') if (x/'complete.json').is_file()),key=lambda x:int(x.name.split('-')[-1]))
        assert complete and Path(a.resume).resolve()==complete[-1].resolve(),'only latest complete checkpoint may resume training'
    ledger_path=root/('formal_seed42_ledger.json' if a.kind=='formal' else 'acceptance_ledger.json')
    status_path=job/'output/pipeline.json'
    started=time.monotonic();utc=lambda:dt.datetime.now(dt.timezone.utc).isoformat()
    status={'status':'PREFLIGHT','job_id':a.job_id,'kind':a.kind,'run_commit':a.run_commit,
            'output':str(output),'started_utc':utc(),'pid':os.getpid(),'stages':[]}
    with (root/'a0_gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ledger=reconcile(ledger_path)
        assert all(x.get('ended_utc') for x in ledger['attempts']), 'query prior unresolved job; no blind restart'
        assert all(x['job_id']!=a.job_id for x in ledger['attempts'])
        if a.kind=='formal' and ledger['attempts']: assert a.resume or a.resume_evaluation,'second fresh formal seed forbidden'
        used=sum(x['wall_seconds'] for x in ledger['attempts'])
        limit=(345600-used) if a.kind=='formal' else (3600-sum(x['gpu_device_seconds'] for x in ledger['attempts']))/2
        assert limit>0
        gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv'],text=True)
        assert len(gpu.strip().splitlines())==1,'GPU compute processes already present'
        status['gpu_start']=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used','--format=csv'],text=True)
        entry={'job_id':a.job_id,'kind':a.kind,'run_commit':a.run_commit,'started_utc':utc(),'resume':a.resume,'wall_limit_seconds':limit}
        ledger['attempts'].append(entry);write(ledger_path,ledger)
        write(status_path,status)
        children=[]

        def resource():
            mem={r.split(':')[0]:int(r.split()[1])*1024 for r in Path('/proc/meminfo').read_text().splitlines() if len(r.split())>=2}
            disk=os.statvfs(root)
            v={'utc':utc(),'elapsed_seconds':time.monotonic()-started,'mem_available':mem['MemAvailable'],
               'swap_used':mem['SwapTotal']-mem['SwapFree'],'disk_available':disk.f_bavail*disk.f_frsize,
               'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)}
            with (job/'output/resources.jsonl').open('a') as f:f.write(json.dumps(v)+'\n')
            assert v['mem_available']>=40*1024**3,'HOST_MEMORY_RESERVE_40GIB'
            assert v['disk_available']>=max(100*1024**3,disk.f_blocks*disk.f_frsize//10),'DATA_DISK_RESERVE'
            assert time.monotonic()-started<limit,'CUMULATIVE_BUDGET_EXHAUSTED'

        def run(stage,args):
            resource()
            command=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2',*map(str,args)]
            record={'stage':stage,'argv':command,'started_utc':utc()};status['stages'].append(record)
            status.update(status='RUNNING',phase=stage);write(status_path,status)
            # No new session: inherited identity markers and process group stay
            # under the existing runner's pidfd-safe timeout/cleanup mechanism.
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL);children.append(child)
            while child.poll() is None:
                resource()
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:pass
            record.update(exit_code=child.returncode,ended_utc=utc());write(status_path,status)
            if child.returncode:raise RuntimeError(stage+' failed: '+str(child.returncode))

        try:
            args=[scripts/'formal_train.py','--source-root',source,'--data-root',data,'--model',model,
                  '--output',training,'--config',config,'--run-commit',a.run_commit,
                  '--precision-approval',a.precision_approval]
            if a.resume:args+=['--resume',a.resume]
            if a.kind!='formal':args+=['--probe-stop','4','--probe-save','2']
            if a.resume_evaluation:
                assert a.kind=='formal' and not a.resume and ledger['attempts'][:-1]
                for r in [0,1]:
                    proof=json.loads((training/('rank%d.json'%r)).read_text())
                    assert proof['status']=='TRAINING_COMPLETE' and proof['global_step']==8240 and proof['epoch']==4.0
                    assert proof['run_commit']==a.run_commit and proof['config_sha256']==sha(config)
                selected=json.loads((training/'best_selection.json').read_text())
                assert selected['run_commit']==a.run_commit
                assert sha(training/'best_adapter/adapter_model.bin')==selected['adapter_sha256']
                status['stages'].append({'stage':'train','status':'REUSED_VERIFIED_COMPLETE_TRAINING','training':str(training),'best_selection':selected})
            else:
                if a.kind=='formal' and (training/'best_selection.json').exists():
                    raise RuntimeError('training already completed: verify and use --resume-evaluation; never retrain from older best')
                run('train',args)
            if a.kind=='formal':
                selected=json.loads((training/'best_selection.json').read_text())
                run('test',[scripts/'formal_evaluate.py','--source-root',source,'--data-root',data,'--model',model,
                            '--adapter',training/'best_adapter','--output',output/'test','--run-commit',a.run_commit])
                metrics=json.loads((output/'test/metrics.json').read_text())
                status.update(best_selection=selected,metrics=metrics)
                for r in [0,1]:assert json.loads((training/('rank%d.json'%r)).read_text())['status']=='TRAINING_COMPLETE'
            status.update(status='SUCCESS' if a.kind=='formal' else 'ACCEPTANCE_PATH_COMPLETE',exit_code=0)
        except BaseException as e:
            import traceback
            status.update(status='FAILED',error=repr(e),traceback=traceback.format_exc(),exit_code=1)
            # The runner owns verified child termination. Exit nonzero and let
            # its identity-checked cleanup handle any surviving task workers.
            raise
        finally:
            elapsed=time.monotonic()-started
            entry.update(ended_utc=utc(),wall_seconds=elapsed,gpu_device_seconds=2*elapsed,status=status['status'])
            write(ledger_path,ledger)
            status.update(ended_utc=utc(),wall_seconds=elapsed,charged_gpu_device_seconds=2*elapsed)
            write(status_path,status)


if __name__=='__main__':main()
