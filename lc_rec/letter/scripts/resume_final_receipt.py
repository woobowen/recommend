#!/usr/bin/env python3
"""CPU-only final source, job, budget, resource and original-file receipt."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import pwd
import socket
import subprocess

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--commits',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert (socket.gethostname(),os.getuid(),pwd.getpwuid(os.getuid()).pw_name)==('zwlab-01',1005,'bwwu')
    root=Path('/data/bwwu/lc-rec/resume-20261112');support=Path('/home/bwwu/lc-rec/letter-resume-20261112')
    report={'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'host':socket.gethostname(),'uid':os.getuid(),'sources':[],'jobs':[]}
    for commit in json.loads(a.commits):
        release=support/'releases'/commit;v=json.loads((release/'RUN_VERSION.json').read_text())
        assert v['RUN_COMMIT']==commit
        for row in v['files']:assert digest(release/row['path'])==row['sha256'],str(release/row['path'])
        report['sources'].append(v)
    old=json.loads((root/'letter-resume-20261112-cpu-r0/inspection/cpu_inspection.json').read_text())
    cp=Path(old['checkpoint']);protected=[]
    for row in old['files']:
        f=cp/row['path'];actual=digest(f)
        assert actual==row['sha256'] and f.stat().st_size==row['bytes'],str(f)
        protected.append({'path':str(f),'bytes':row['bytes'],'sha256':actual})
    report['historical_checkpoint_rehashed_unchanged']=protected
    jobroot=Path('/home/bwwu/grid-audits/sid-extensions-p0/jobs');charged=0
    for job in sorted(jobroot.glob('letter-resume-20261112-*')):
        request=json.loads((job/'request.json').read_text());status=json.loads((job/'status.json').read_text())
        row={'job_id':job.name,'status':status['status'],'exit_code':status.get('exit_code'),'capture_version':request['capture_version'],'argv':request['argv'],'charged_gpu_device_seconds':0}
        pipeline=root/job.name/'pipeline.json'
        if pipeline.exists():row['pipeline']=json.loads(pipeline.read_text())
        gpu=request['env'].get('CUDA_VISIBLE_DEVICES') not in ['',None]
        if gpu:
            assert status.get('ended_at') and status.get('started_at'),job.name
            elapsed=(dt.datetime.fromisoformat(status['ended_at'])-dt.datetime.fromisoformat(status['started_at'])).total_seconds()
            row['charged_gpu_device_seconds']=max(2*elapsed,row.get('pipeline',{}).get('gpu_device_seconds',0))
            charged+=row['charged_gpu_device_seconds']
        row['gpu_request']=gpu;row['cleanup_verified']=status.get('cleanup',{}).get('verified')
        report['jobs'].append(row)
    report['ledger']=json.loads((root/'ledger.json').read_text())
    report['charged_gpu_device_seconds']=charged;report['charged_gpu_device_minutes']=charged/60
    assert charged<=7200
    report['new_storage']={}
    for path in [root,support]:
        fs=[f for f in path.rglob('*') if f.is_file() and not f.is_symlink()]
        report['new_storage'][str(path)]={'logical_bytes':sum(f.stat().st_size for f in fs),'files':len(fs)}
    report['total_new_persistent_bytes']=sum(v['logical_bytes'] for v in report['new_storage'].values())
    assert report['total_new_persistent_bytes']<=80*1024**3
    report['resources']={k:subprocess.check_output(v,text=True) for k,v in {
      'gpu':['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv'],
      'compute':['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv'],
      'ram':['free','-b'],'disk':['df','-B1','/data/bwwu','/home/bwwu'],
      'processes':['ps','-u','bwwu','-o','pid,ppid,lstart,args']}.items()}
    assert len(report['resources']['compute'].strip().splitlines())==1
    oldroot=Path('/data/bwwu/lc-rec/a0-normal-20261111')
    assert not (oldroot/'formal_seed42').exists() and not (oldroot/'formal_seed42_ledger.json').exists()
    report['A0_FORMAL_NORMAL']='NOT_STARTED';report['new_installations']=[]
    report['ended_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'charged_gpu_device_minutes':charged/60,'persistent_bytes':report['total_new_persistent_bytes'],'original_files_unchanged':len(protected),'sources':len(report['sources']),'A0_FORMAL_NORMAL':report['A0_FORMAL_NORMAL']}))

if __name__=='__main__':main()
