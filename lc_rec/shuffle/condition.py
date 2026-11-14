"""One immutable accepted formal condition: two ranks, then CPU verification."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from gpu_pipeline import gpu_idle


def atomic(path,value):
    tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser()
    for n in ['model-key','shuffle-n','output','cpu-root','run-commit','job-id','prelaunch-gate']:
        p.add_argument('--'+n,required=True)
    p.add_argument('--previous-job')
    a=p.parse_args();assert a.model_key in ['a0','a1'] and a.shuffle_n in ['3','4']
    root=Path(__file__).resolve().parent;cfg=json.loads((root/'config.json').read_text());b=cfg['baselines'][a.model_key]
    gate=json.loads(Path(a.prelaunch_gate).read_text())
    assert gate['status']=='PASS' and gate['SHUFFLE_RUN_COMMIT']==a.run_commit
    assert gate['A1_NORMAL_REVIEW']=='PASS'
    receipt=Path(gate['preflight_receipt']['path'])
    assert hashlib.sha256(receipt.read_bytes()).hexdigest()==gate['preflight_receipt']['sha256']
    assert json.loads(receipt.read_text())['status']=='PASS'
    preflight_status=json.loads((Path('/home/bwwu/grid-audits/sid-extensions-p0/jobs')/gate['preflight_job_id']/'status.json').read_text())
    assert preflight_status['status']=='SUCCEEDED' and preflight_status['cleanup']['verified']
    for name,expected in gate['core_source_sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected
    for model,expected in gate['cpu_audit_sha256'].items():
        assert hashlib.sha256((Path(a.cpu_root)/model/'audit.json').read_bytes()).hexdigest()==expected
    common=Path(a.cpu_root)/'shared_permutations.jsonl'
    assert hashlib.sha256(common.read_bytes()).hexdigest()==gate['shared_permutation_manifest_sha256']
    if a.previous_job:
        prev=Path('/home/bwwu/grid-audits/sid-extensions-p0/jobs')/a.previous_job
        status=json.loads((prev/'status.json').read_text())
        assert status['status']=='SUCCEEDED' and status['exit_code']==0 and status['cleanup']['verified']
    data=Path(a.output).parent;out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    assert json.loads((Path(a.cpu_root)/'summary.json').read_text())['status']=='PASS_CPU_PREPARATION'
    lock=data/'gpu.lock';ledger_path=data/'formal_ledger.json'
    with lock.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {'attempts':[]}
        for prior in ledger['attempts']:
            state=json.loads((Path('/home/bwwu/grid-audits/sid-extensions-p0/jobs')/prior['job_id']/'status.json').read_text())
            assert state['status']=='SUCCEEDED' and state['cleanup']['verified']
            prior['runner_wall_seconds']=(dt.datetime.fromisoformat(state['ended_at'])-dt.datetime.fromisoformat(state['started_at'])).total_seconds()
        used=sum(x['runner_wall_seconds'] for x in ledger['attempts'])
        cap=14400 if a.model_key=='a0' else 10800
        assert used+cap+1800<=64800,'Conservative total18h includes planned CPU margin'
        assert not any(x['job_id']==a.job_id for x in ledger['attempts'])
        attempt={'job_id':a.job_id,'model_key':a.model_key,'shuffle_n':int(a.shuffle_n),
                 'RUN_COMMIT':a.run_commit,'status':'RUNNING','started_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
                 'runner_wall_seconds':0,'gpu_stage_seconds':0,'gpu_device_seconds':0}
        ledger['attempts'].append(attempt);started=time.monotonic();atomic(ledger_path,ledger)
        try:
            gpu_idle()
            argv=[b['python'],'-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(root/'entry.py'),
              '--model-key',a.model_key,'--shuffle-n',a.shuffle_n,'--cpu-audit',str(Path(a.cpu_root)/a.model_key/'audit.json'),
              '--source-root',b['source-root'],'--data-root',b['data-root'],'--model',b['model'],'--adapter',b['adapter'],
              '--output',str(out/'test'),'--run-commit',a.run_commit,'--max-seconds',str(cap-60)]
            attempt['generation_argv']=argv;atomic(ledger_path,ledger)
            gpu_started=time.monotonic()
            with (out/'generation.log').open('x') as log,(out/'resources.jsonl').open('x') as resources:
                proc=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                while proc.poll() is None:
                    elapsed=time.monotonic()-gpu_started
                    memory={x.split(':')[0]:int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith(('MemAvailable:','SwapFree:','SwapTotal:'))}
                    sample={'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'gpu_stage_seconds':elapsed,'host':memory,
                      'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv'],text=True),
                      'disk_free_bytes':os.statvfs(out).f_bavail*os.statvfs(out).f_frsize}
                    resources.write(json.dumps(sample)+'\n');resources.flush()
                    assert memory['MemAvailable']>=40*1024**3 and sample['disk_free_bytes']>=20*1024**3
                    if elapsed>=cap-30:
                        proc.terminate();raise TimeoutError('Formal inference condition hard cap reserve')
                    attempt.update(gpu_stage_seconds=elapsed,gpu_device_seconds=2*elapsed,runner_wall_seconds=time.monotonic()-started)
                    atomic(ledger_path,ledger);time.sleep(15)
                code=proc.returncode
            attempt.update(generation_exit_code=code,gpu_stage_seconds=time.monotonic()-gpu_started)
            attempt['gpu_device_seconds']=2*attempt['gpu_stage_seconds'];atomic(ledger_path,ledger);assert code==0
            gpu_idle()
            reports=[json.loads((out/'test'/('rank%d.json'%rank)).read_text()) for rank in [0,1]]
            assert all(x['status']=='PASS' and x['completed_users']==12386 for x in reports)
            files=[root/'entry.py',root/'runtime.py',root/'core.py',root/'config.json']
            for rank in [0,1]:files += [out/'test'/('derived.rank%d.py'%rank),out/'test'/('derived.rank%d.diff'%rank)]
            atomic(out/'execution_sources.json',{'RUN_COMMIT':a.run_commit,
                'rank_entry_sha256_field_means':'frozen original entry; actual derived execution is separately hashed here',
                'files':[{'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]})
            cpu_python=cfg['baselines']['a1']['python'];env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
            argv=[cpu_python,str(root/'evaluate_condition.py'),'--model-key',a.model_key,'--test-dir',str(out/'test'),
                  '--output',str(out/'evaluation'),'--run-commit',a.run_commit,'--shuffle-n',a.shuffle_n,
                  '--cpu-audit',str(Path(a.cpu_root)/a.model_key/'audit.json')]
            tick=time.monotonic()
            with (out/'evaluation.log').open('x') as log:
                r=subprocess.run(argv,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1700)
            attempt.update(evaluation_argv=argv,evaluation_exit_code=r.returncode,cpu_evaluation_seconds=time.monotonic()-tick)
            assert r.returncode==0
            assert json.loads((out/'evaluation/summary.json').read_text())['status']=='PASS'
            attempt['status']='PASS'
        except BaseException as error:
            attempt['status']='FAIL';attempt['error']=repr(error);raise
        finally:
            attempt['runner_wall_seconds']=time.monotonic()-started
            attempt['ended_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
            atomic(ledger_path,ledger);atomic(out/'summary.json',attempt)
    print(json.dumps(attempt))


if __name__=='__main__':main()
