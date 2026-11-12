#!/usr/bin/env python3
"""Bounded, server-owned diagnosis under the accepted A0 runner."""
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

ROOT=Path('/data/bwwu/lc-rec/attention-20261112')
OLD=Path('/data/bwwu/lc-rec/a0-normal-20261111/letter-a0-20261111-acceptance-cont-r3/training/checkpoint-2')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-commit',required=True);p.add_argument('--kind',choices=['inspect','continuous','resume','attention-isolate'],required=True)
    p.add_argument('--resume');p.add_argument('--diagnostic-attention-deterministic',action='store_true')
    p.add_argument('--capture-attention-case',action='store_true');p.add_argument('--attention-case-root')
    p.add_argument('--full-first-backward-trace',action='store_true');p.add_argument('--diagnostic-torch-deterministic',action='store_true');a=p.parse_args()
    job=Path(os.environ['GRID_JOB_DIR']);assert job.name.startswith('letter-attention-20261112-')
    release=Path(__file__).resolve().parents[3];scripts=release/'lc_rec/letter/scripts'
    assert json.loads((release/'RUN_VERSION.json').read_text())['RUN_COMMIT']==a.run_commit, 'RUN_COMMIT label differs from immutable deployment'
    ROOT.mkdir(parents=True,exist_ok=True);out=ROOT/job.name;out.mkdir(exist_ok=False)
    cfg=json.loads((release/'lc_rec/letter/config/attention_boundary_default.json').read_text())
    utc=lambda:dt.datetime.now(dt.timezone.utc).isoformat()
    started=time.monotonic();status={'job_id':job.name,'run_commit':a.run_commit,'kind':a.kind,'started_utc':utc(),'status':'PREFLIGHT','gpu_device_seconds':0}
    write(out/'pipeline.json',status)
    with Path('/data/bwwu/lc-rec/a0-normal-20261111/a0_gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ledger_path=ROOT/'ledger.json'
        ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {'attempts':[],'gpu_limit_seconds':7200}
        assert all(x.get('ended_utc') for x in ledger['attempts']), 'unresolved prior job: inspect same ID'
        assert len([x for x in ledger['attempts'] if x['kind']!='inspect'])<cfg['jobs_limit']
        used=sum(x['gpu_device_seconds'] for x in ledger['attempts'])
        # Reserve 20 seconds per GPU for launch/runner cleanup beyond this loop.
        limit=min(1780,(7200-used)/2-20) if a.kind!='inspect' else 600
        assert limit>0
        entry=dict(status);ledger['attempts'].append(entry);write(ledger_path,ledger)
        try:
            if a.kind=='inspect':
                assert os.environ['CUDA_VISIBLE_DEVICES']==''
                argv=[sys.executable,str(scripts/'resume_inspect.py'),'--checkpoint',str(OLD),'--output',str(out/'inspection')]
            else:
                compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv'],text=True)
                assert len(compute.strip().splitlines())==1,'GPU occupied'
                reserve=(24 if a.kind=='continuous' else 6 if a.kind=='resume' else 1)*1024**3
                assert sum(f.stat().st_size for f in ROOT.rglob('*') if f.is_file() and not f.is_symlink())<cfg['persistent_bytes_limit']-reserve
                source=release/'lc_rec/letter/vendor/LETTER/LETTER-LC-Rec'
                diagnostic_config=('resume_torch_deterministic_diagnostic.json' if a.diagnostic_torch_deterministic else 'resume_full_trace_diagnostic.json' if a.full_first_backward_trace else 'attention_boundary_det.json' if a.diagnostic_attention_deterministic else 'resume_capture_diagnostic.json' if a.capture_attention_case else 'attention_boundary_default.json')
                argv=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2',str(scripts/'formal_train.py'),
                    '--source-root',str(source),'--data-root',str(source.parent/'data'),
                    '--model','/data/bwwu/lc-rec/assets/llama-7b/4782ad278652c7c71b72204d462d6d01eaaf7549',
                    '--output',str(out/'training'),'--config',str(release/'lc_rec/letter/config/formal_instruments_seed42.json'),
                    '--run-commit',a.run_commit,'--precision-approval',str(release/'lc_rec/letter/evidence/formal_20261111/precision_approval.json'),
                    '--probe-stop','4','--probe-save','2','--diagnosis',str(release/'lc_rec/letter/config'/diagnostic_config)]
                if a.kind=='resume':argv+=['--resume',a.resume or str(OLD)]
                if a.kind=='attention-isolate':
                    assert a.attention_case_root
                    argv=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2',str(scripts/'attention_case_replay.py'),'--case-root',a.attention_case_root,'--output',str(out/'isolate'),'--process-repeat','-1']
            status.update(argv=argv,wall_limit_seconds=limit,status='RUNNING');write(out/'pipeline.json',status)
            child_env=dict(os.environ)
            if a.diagnostic_torch_deterministic:child_env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
            status['diagnostic_environment']={'CUBLAS_WORKSPACE_CONFIG':child_env.get('CUBLAS_WORKSPACE_CONFIG')}
            write(out/'pipeline.json',status)
            child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,env=child_env)
            while child.poll() is None:
                elapsed=time.monotonic()-started
                assert elapsed<limit,'diagnosis budget limit; runner owns verified cleanup'
                disk=os.statvfs(ROOT);assert disk.f_bavail*disk.f_frsize>100*1024**3
                mem={x.split(':')[0]:int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if len(x.split())>=2}
                assert mem['MemAvailable']>40*1024**3
                if a.kind!='inspect':
                    resource={'utc':utc(),'elapsed_seconds':elapsed,'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv'],text=True),'mem_available':mem['MemAvailable'],'disk_available':disk.f_bavail*disk.f_frsize}
                    with (out/'resources.jsonl').open('a') as f:f.write(json.dumps(resource)+'\n')
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:pass
            status['exit_code']=child.returncode
            assert child.returncode==0, 'child failed '+str(child.returncode)
            status['status']='DIAGNOSTIC_PATH_COMPLETE'
        except BaseException as e:
            import traceback
            status.update(status='FAILED',error=repr(e),traceback=traceback.format_exc());raise
        finally:
            elapsed=time.monotonic()-started
            status.update(ended_utc=utc(),wall_seconds=elapsed,gpu_device_seconds=2*(elapsed+20) if a.kind!='inspect' else 0)
            entry.update(status);write(out/'pipeline.json',status);write(ledger_path,ledger)

if __name__=='__main__':main()
