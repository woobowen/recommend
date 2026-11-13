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
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"qwen_preflight_20261110/scripts"))
from common import write,sha
from ledger import reconcile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['release','run-commit','job-id','kind']:
        p.add_argument('--'+n,required=True)
    p.add_argument('--resume');p.add_argument('--resume-evaluation',action='store_true')
    p.add_argument('--prelaunch-review');p.add_argument('--continuous');p.add_argument('--resumed');a=p.parse_args()
    assert a.kind in ['cpu','compare','continuous','resume','formal']
    root=Path('/data/bwwu/lc-rec/a1-qwen3-06b-instruments-20261113');root.mkdir(parents=True,exist_ok=True)
    job=Path(os.environ['GRID_JOB_DIR']);assert job.name==a.job_id
    scripts=Path(a.release)/'lc_rec/a1_qwen3_06b/scripts'
    config=Path(a.release)/'lc_rec/a1_qwen3_06b/config/formal_instruments_seed42.json'
    if a.kind=='formal':
        assert a.prelaunch_review
        gate=json.loads(Path(a.prelaunch_review).read_text())
        assert gate['status']=='PASS' and gate['RUN_COMMIT']==a.run_commit
        assert gate['config_sha256']==sha(config)
        assert gate['source_sha256']=={str(p.relative_to(scripts.parent)):sha(p) for p in sorted(scripts.glob('*.py'))}
    letter=Path(a.release)/'lc_rec/letter/vendor/LETTER'
    source=letter/'LETTER-LC-Rec'
    data=source.parent/'data'
    model='/data/bwwu/lc-rec/qwen-preflight-20261110/models/Qwen3-0.6B-Base/da87bfb608c14b7cf20ba1ce41287e8de496c0cd'
    if a.kind in ['cpu','compare']:
        output=root/a.job_id;output.mkdir(parents=True,exist_ok=True)
        if a.kind=='cpu':
            command=[sys.executable,str(scripts/'cpu.py'),'--release',a.release,'--output',str(output/'cpu'),'--run-commit',a.run_commit]
        else:
            assert a.continuous and a.resumed
            assert all(root.resolve() in Path(v).resolve().parents for v in [a.continuous,a.resumed])
            command=[sys.executable,str(scripts/'compare.py'),'--continuous',a.continuous,'--resumed',a.resumed,'--output',str(output/'comparison.json')]
        subprocess.run(command,check=True)
        write(job/'output/pipeline.json',{'status':'CPU_COMPLETE','RUN_COMMIT':a.run_commit,'kind':a.kind,'argv':command})
        return
    asset_path=root/'letter-a1-q06-20261113-cpu-r0/cpu/assets_environment.json'
    assets=json.loads(asset_path.read_text())
    assert assets['status']=='PASS' and assets['model_revision']==Path(model).name
    assert assets['files']['model.safetensors']['bytes']==Path(model+'/model.safetensors').stat().st_size==1192135096
    assert assets['files']['model.safetensors']['sha256']=='cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba'
    assert assets['int8_serialization_contract']['status']=='PASS_CPU_SERIALIZATION_CONTRACT'
    output=root/a.job_id;output.mkdir(exist_ok=True)
    training=(root/'formal_seed42/training') if a.kind=='formal' else output/'training'
    training.parent.mkdir(exist_ok=True,parents=True)
    if a.resume and a.kind=='formal':assert Path(a.resume).resolve().parent==training.resolve()
    if a.resume and a.kind=='formal':
        complete=sorted((x for x in training.glob('checkpoint-*') if (x/'complete.json').is_file()),key=lambda x:int(x.name.split('-')[-1]))
        assert complete and Path(a.resume).resolve()==complete[-1].resolve(),'only latest complete checkpoint may resume training'
    ledger_path=root/('formal_seed42_ledger.json' if a.kind=='formal' else
                     'preflight_ledger.json')
    status_path=job/'output/pipeline.json'
    started=time.monotonic();utc=lambda:dt.datetime.now(dt.timezone.utc).isoformat()
    status={'status':'PREFLIGHT','job_id':a.job_id,'kind':a.kind,'run_commit':a.run_commit,
            'output':str(output),'started_utc':utc(),'pid':os.getpid(),'stages':[],
            'asset_gate':{'path':str(asset_path),'sha256':sha(asset_path),'model_revision':assets['model_revision']}}
    with (root/'a1_gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ledger=reconcile(ledger_path)
        assert all(x.get('ended_utc') for x in ledger['attempts']), 'query prior unresolved job; no blind restart'
        assert all(x['job_id']!=a.job_id for x in ledger['attempts'])
        if a.kind=='formal' and ledger['attempts']: assert a.resume or a.resume_evaluation,'second fresh formal seed forbidden'
        used=sum(x['wall_seconds'] for x in ledger['attempts'])
        # Reserve 20s for identity-checked worker cleanup inside the total cap.
        limit=(345600-used-20) if a.kind=='formal' else (3600-sum(x['gpu_device_seconds'] for x in ledger['attempts'])-40)/2
        assert limit>0
        gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv'],text=True)
        assert len(gpu.strip().splitlines())==1,'GPU compute processes already present'
        status['gpu_start']=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used','--format=csv'],text=True)
        entry={'job_id':a.job_id,'kind':a.kind,'run_commit':a.run_commit,'started_utc':utc(),'resume':a.resume,'wall_limit_seconds':limit}
        ledger['attempts'].append(entry);write(ledger_path,ledger)
        write(status_path,status)
        children=[]
        activity={'worklog_bytes':-1,'last_progress':time.monotonic()}

        def resource():
            log_bytes=(job/'work.log').stat().st_size
            if log_bytes!=activity['worklog_bytes']:
                activity.update(worklog_bytes=log_bytes,last_progress=time.monotonic())
            mem={r.split(':')[0]:int(r.split()[1])*1024 for r in Path('/proc/meminfo').read_text().splitlines() if len(r.split())>=2}
            disk=os.statvfs(root)
            v={'utc':utc(),'elapsed_seconds':time.monotonic()-started,'mem_available':mem['MemAvailable'],
               'swap_used':mem['SwapTotal']-mem['SwapFree'],'disk_available':disk.f_bavail*disk.f_frsize,
               'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True),
               'seconds_without_worklog_progress':time.monotonic()-activity['last_progress']}
            with (job/'output/resources.jsonl').open('a') as f:f.write(json.dumps(v)+'\n')
            assert v['mem_available']>=40*1024**3,'HOST_MEMORY_RESERVE_40GIB'
            assert v['disk_available']>=max(100*1024**3,disk.f_blocks*disk.f_frsize//10),'DATA_DISK_RESERVE'
            assert time.monotonic()-started<limit,'CUMULATIVE_BUDGET_EXHAUSTED'
            assert v['seconds_without_worklog_progress']<7200,'NO_TASK_LOG_PROGRESS_FOR_TWO_HOURS'

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
                try:child.wait(timeout=min(30,max(.01,limit-(time.monotonic()-started))))
                except subprocess.TimeoutExpired:pass
            record.update(exit_code=child.returncode,ended_utc=utc());write(status_path,status)
            if child.returncode:raise RuntimeError(stage+' failed: '+str(child.returncode))

        try:
            args=[scripts/'train.py','--letter',letter,'--model',model,
                  '--output',training,'--config',config,'--run-commit',a.run_commit]
            if a.resume:args+=['--resume',a.resume]
            if a.kind in ['continuous','resume']:args+=['--probe-stop','12','--probe-save','2','--preflight-interfaces']
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
            if a.kind=='continuous':
                run('test_interfaces',[scripts/'evaluate.py','--source-root',source,'--data-root',data,'--model',model,
                            '--adapter',training/'preflight_adapter','--output',output/'test_preflight','--run-commit',a.run_commit,
                            '--preflight-users-per-rank','8'])
                for r in [0,1]:
                    assert json.loads((output/'test_preflight'/('rank%d.json'%r)).read_text())['status']=='PREFLIGHT_INTERFACE_PASS'
            if a.kind=='formal':
                selected=json.loads((training/'best_selection.json').read_text())
                assert selected['run_commit']==a.run_commit
                assert sha(training/'best_adapter/adapter_model.bin')==selected['adapter_sha256']
                for r in [0,1]:
                    proof=json.loads((training/('rank%d.json'%r)).read_text())
                    assert proof['status']=='TRAINING_COMPLETE' and proof['global_step']==8240 and proof['epoch']==4.0
                    assert proof['run_commit']==a.run_commit and proof['config_sha256']==sha(config)
                run('test',[scripts/'evaluate.py','--source-root',source,'--data-root',data,'--model',model,
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
