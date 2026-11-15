#!/usr/bin/env python3
"""Bounded inference-only Beauty preflight; never launches training or full tests."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from preflight_budget import ledger
from preflight_subprocess import run

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

release=Path(sys.argv[1]).resolve(strict=True);commit=sys.argv[2];audit=Path(sys.argv[3]).resolve(strict=True)
assert len(commit)==40 and all(c in '0123456789abcdef' for c in commit)
manifest=json.loads((release/'export-manifest.json').read_text());assert manifest['git_commit']==commit
for row in manifest['files']:
    q=Path(row['path']);assert not q.is_absolute() and '..' not in q.parts
    p=release/q;assert p.is_file() and not p.is_symlink() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
assert sha(audit)=='eaadaa8b3eb0509d58ec587e9be6ac4f13c7e19e6489b7f39991635a11c8e634'
assert os.environ['GRID_JOB_ID']=='letter-a0-beauty-infer-20261116-r0'
root=Path('/data/bwwu/lc-rec/a0-beauty-20261115')/os.environ['GRID_JOB_ID'];assert not root.exists()
job=Path(os.environ['GRID_JOB_DIR']);out=job/'output'
adapter=Path('/data/bwwu/lc-rec/a0-beauty-20261115/letter-a0-beauty-e2e-20261116-r0/continuous/preflight_adapter')
assert (adapter/'adapter_model.bin').stat().st_size==580231186 and sha(adapter/'adapter_model.bin')=='e56abd90bd3cc55d3786df04971a670bc12c54578110d255b5e7f3f4ec76af07'
model=Path('/data/bwwu/lc-rec/assets/llama-7b/4782ad278652c7c71b72204d462d6d01eaaf7549')
data=Path('/home/bwwu/grid-audits/sid-extensions-p0/transfers/letter-a0-beauty-author-8d0154e28de3')
source=release/'lc_rec/letter/vendor/LETTER/LETTER-LC-Rec';users=release/'lc_rec/a0_beauty/config/preflight_users_seed42.json'
assert sha(users)=='714450a0e9479ed272266a5b4d3c2150d83742e4b70c38f80da888d4fa0aaa39'
a=json.loads(audit.read_text());assert a['normal_test_dir']==str(root/'normal') and a['manifest']['sha256']==sha(users)
assert os.environ['CUDA_VISIBLE_DEVICES']=='0,1'
gpus=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True)
rows=[x.split(', ') for x in gpus.strip().splitlines()];assert [x[0] for x in rows]==['GPU-dfff0e95-1db6-e2f4-3431-55bb852fa0ae','GPU-59cac3bd-3645-668b-79c8-839e0d878893'] and all(int(x[1])<100 for x in rows)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
budget=ledger('/home/bwwu/grid-audits/sid-extensions-p0/jobs',os.environ['GRID_JOB_ID'],max_wall=1740,limit=7200)
base_ids={'letter-a0-beauty-boundary-model-20261115-r0','letter-a0-beauty-cross-resume-20261115-r0','letter-a0-beauty-e2e-20261116-r0','letter-a0-beauty-zero-resume-20261115-r0','letter-a0-beauty-zero-state-20261115-r0'}
assert base_ids <= {x['job_id'] for x in budget['jobs']}
base=sum(x['charged_device_seconds'] for x in budget['jobs'] if x['job_id'] in base_ids)
assert abs(base-3316.546833369881)<0.00001
additional=sum(x['charged_device_seconds'] for x in budget['jobs'] if x['job_id'] not in base_ids)
wall=min(budget['wall_cap_seconds'],(3600-additional)/2-60);assert wall>0
mem=lambda: int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
assert mem()>=40*1024**3 and shutil.disk_usage('/data').free>=100*1024**3 and shutil.disk_usage(str(out)).free>=100*1024**3
root.mkdir(parents=True,exist_ok=False)
rec={'job_id':os.environ['GRID_JOB_ID'],'run_commit':commit,'training_run_commit':'f54a2b505fd2d8767109a11d330a7064aff8af5e','budget':budget,'historical_charge':base,'additional_prior_charge':additional,'incremental_limit_device_seconds':3600,'total_limit_device_seconds':7200,'wall_cap_seconds':wall,'status':'STARTED','stages':[],'output_root':str(root),'gpu_start':gpus}
start=time.monotonic()
try:
    launcher=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2']
    common=['--source-root',str(source),'--data-root',str(data),'--model',str(model),'--adapter',str(adapter),'--run-commit',commit,'--preflight-users-per-rank','16','--preflight-manifest',str(users)]
    for stage in ('normal','shuffle_acceptance'):
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        target=root/stage
        if stage=='normal': argv=launcher+[str(release/'lc_rec/a0_beauty/formal_evaluate.py')]+common+['--output',str(target)]
        else: argv=launcher+[str(release/'lc_rec/a0_beauty/shuffle_entry.py'),'--model-key','a0','--shuffle-n','0','--acceptance','--cpu-audit',str(audit)]+common+['--output',str(target)]
        item={'mode':stage,'argv':argv,'started_utc':dt.datetime.now(dt.timezone.utc).isoformat()};rec['stages'].append(item)
        def observe():
            disk=shutil.disk_usage('/data');m=mem();other=shutil.disk_usage(str(out))
            sample={'mode':stage,'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'charged_device_seconds_upper_bound':2*(time.monotonic()-start),'memory_available':m,'data_free':disk.free,'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)}
            with (out/'inference_resources.jsonl').open('a') as f:f.write(json.dumps(sample)+'\n')
            if m<40*1024**3 or disk.free<max(100*1024**3,disk.total//10) or other.free<max(100*1024**3,other.total//10):raise RuntimeError('resource reserve breached')
        run(argv,start+wall,item,observe)
        item['ended_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
        for rank in (0,1):
            info=json.loads((target/('rank%d.json'%rank)).read_text());assert info['status']=='PREFLIGHT_INTERFACE_PASS' and info['completed_users']==16
            if stage=='shuffle_acceptance':
                result=json.loads((target/('acceptance.rank%d.json'%rank)).read_text());assert result['status']=='PASS' and result['users']==16
    rec['status']='STAGES_EXECUTED_AWAITING_INDEPENDENT_CHECK'
except BaseException as e:
    rec['status']='TIMED_OUT' if isinstance(e,TimeoutError) else 'FAILED';rec['error']=repr(e);raise
finally:
    rec['wall_seconds']=time.monotonic()-start;rec['charged_device_seconds_upper_bound']=2*rec['wall_seconds']
    (out/'inference_guard.json').write_text(json.dumps(rec,indent=2)+'\n')
