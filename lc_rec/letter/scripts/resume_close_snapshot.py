#!/usr/bin/env python3
"""Light read-only server closing observation; no model import or new job."""
from pathlib import Path
import datetime as dt
import json
import os
import socket
import subprocess

assert socket.gethostname()=='zwlab-01' and os.getuid()==1005
root=Path('/data/bwwu/lc-rec/resume-20261112');support=Path('/home/bwwu/lc-rec/letter-resume-20261112')
jobroot=Path('/home/bwwu/grid-audits/sid-extensions-p0/jobs')
report={'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'host':socket.gethostname(),'uid':os.getuid(),'jobs':[],'gpu_device_seconds':0}
for job in sorted(jobroot.glob('letter-resume-20261112-*')):
 request=json.loads((job/'request.json').read_text());status=json.loads((job/'status.json').read_text())
 row={k:status.get(k) for k in ['job_id','status','exit_code','started_at','ended_at']}
 row.update(capture_version=request['capture_version'],argv=request['argv'],cuda_visible_devices=request['env'].get('CUDA_VISIBLE_DEVICES'),cleanup_verified=status.get('cleanup',{}).get('verified'),charged_gpu_device_seconds=0)
 if row['cuda_visible_devices'] not in ['',None]:
  elapsed=(dt.datetime.fromisoformat(status['ended_at'])-dt.datetime.fromisoformat(status['started_at'])).total_seconds()
  p=root/job.name/'pipeline.json';pipeline=json.loads(p.read_text()) if p.exists() else {}
  row['charged_gpu_device_seconds']=max(2*elapsed,pipeline.get('gpu_device_seconds',0));report['gpu_device_seconds']+=row['charged_gpu_device_seconds']
 report['jobs'].append(row)
report['active_job_ids']=[r['job_id'] for r in report['jobs'] if r['status'] not in ['SUCCEEDED','FAILED','CANCELLED','TIMED_OUT']]
assert not report['active_job_ids']
assert report['gpu_device_seconds']<=7200
report['gpu_device_minutes']=report['gpu_device_seconds']/60
report['storage']={}
for p in [root,support]:
 files=[f for f in p.rglob('*') if f.is_file() and not f.is_symlink()]
 report['storage'][str(p)]={'logical_bytes':sum(f.stat().st_size for f in files),'files':len(files)}
report['total_new_persistent_bytes']=sum(x['logical_bytes'] for x in report['storage'].values());assert report['total_new_persistent_bytes']<=80*1024**3
report['resources']={k:subprocess.check_output(v,text=True) for k,v in {
 'gpu':['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv'],
 'compute':['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv'],
 'ram':['free','-b'],'disk':['df','-B1','/data/bwwu','/home/bwwu']}.items()}
old=Path('/data/bwwu/lc-rec/a0-normal-20261111')
assert not (old/'formal_seed42').exists() and not (old/'formal_seed42_ledger.json').exists()
report['A0_FORMAL_NORMAL']='NOT_STARTED';report['installations']=[]
print(json.dumps(report))
