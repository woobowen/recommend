#!/usr/bin/env python3
"""Reuse accepted runner for immutable CPU-only offline evaluation exports."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import formal_control as fc

SUPPORT=Path('/home/addaswsw/recommend-support/letter-a0-final-audit-20261113')
REMOTE='/home/bwwu/lc-rec/letter-a0-final-audit-20261113'
RUN='4c6a83d9ba42dc2fe737792759e410ef1874da76'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['deploy','submit','query','collect'])
    p.add_argument('--commit',required=True)
    p.add_argument('--job',required=True)
    a=p.parse_args();SUPPORT.mkdir(parents=True,exist_ok=True)
    assert subprocess.check_output(['git','branch','--show-current'],cwd=fc.REPO,text=True).strip()=='wbw-lc-rec'
    commit=subprocess.check_output(['git','rev-parse',a.commit],cwd=fc.REPO,text=True).strip()
    fc.gridctl.valid(a.job)
    t=fc.transport.Transport('new-server')
    dep=json.loads((fc.SUPPORT/'receipts'/('deployment-'+RUN+'.json')).read_text())
    t.profile=dict(t.profile,runner=dep['runner'],runner_sha256=dep['runner_sha256'])
    release=REMOTE+'/releases/'+commit
    if a.mode=='deploy':
        paths=['lc_rec/letter/scripts/offline_metrics.py','lc_rec/letter/scripts/offline_audit_pipeline.py','lc_rec/letter/config/a0_evaluation_v1.json','lc_rec/letter/review/final_audit_20261113/independent_extended.py','lc_rec/letter/review/final_audit_20261113/statistics_verify.py','lc_rec/letter/review/final_audit_20261113/external/official_cce.py','lc_rec/letter/review/final_audit_20261113/external/LICENSE']
        archive=SUPPORT/(commit+'.tar.gz');rows=[]
        with tempfile.TemporaryDirectory(dir=SUPPORT) as temp:
            with tarfile.open(archive,'w:gz') as tf:
                for name in paths:
                    data=subprocess.check_output(['git','show',commit+':'+name],cwd=fc.REPO)
                    path=Path(temp)/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data);tf.add(path,arcname=name,recursive=False)
                    rows.append({'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
        remote_archive=REMOTE+'/transfers/'+archive.name
        receipt=t.put(archive,remote_archive)
        code=f"""from pathlib import Path,PurePosixPath
import json,tarfile,hashlib
p=Path({remote_archive!r});root=Path({release!r});rows={rows!r}
assert hashlib.sha256(p.read_bytes()).hexdigest()=={receipt['sha256']!r}
with tarfile.open(p,'r:gz') as tf:
 for m in tf.getmembers():
  q=PurePosixPath(m.name);assert m.isfile() and not q.is_absolute() and '..' not in q.parts
 if not root.exists():root.mkdir(parents=True);tf.extractall(root,filter='data')
for r in rows:
 q=root/r['path'];assert q.is_file() and not q.is_symlink() and q.stat().st_size==r['bytes'] and hashlib.sha256(q.read_bytes()).hexdigest()==r['sha256']
print(json.dumps({{'evaluation_commit':{commit!r},'release':str(root),'files':rows,'not_a_git_checkout':True}}))
"""
        result=json.loads(t.python(code));fc.transport.atomic(SUPPORT/('deployment-'+commit+'.json'),result)
    elif a.mode=='submit':
        deployment=json.loads((SUPPORT/('deployment-'+commit+'.json')).read_text())
        old=json.loads((fc.REPO/'lc_rec/letter/evidence/autograd_20261110/jobs/letter-20261110-autograd-cpu-r0/request.json').read_text())
        req={k:old[k] for k in ['schema','host','user','uid','namespace','deployment_sha256']}
        env=dict(old['env']);env.update(CUDA_VISIBLE_DEVICES='',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');env.pop('DS_ACCELERATOR',None)
        req.update(job_id=a.job,capture_version=commit,cwd=release,env=env,timeout_seconds=6000,
                   argv=[fc.PYTHON,release+'/lc_rec/letter/scripts/offline_audit_pipeline.py','--release',release,'--evaluation-commit',commit,'--output','/data/bwwu/lc-rec/a0-final-audit-20261113/'+a.job,'--job-id',a.job],
                   inputs=[{'path':release+'/'+r['path'],'sha256':r['sha256']} for r in deployment['files']])
        path=SUPPORT/(a.job+'.request.json')
        if path.exists():assert json.loads(path.read_text())==req
        else:fc.transport.atomic(path,req)
        result=fc.gridctl.submit(t,path)
    else:
        result=fc.gridctl.query(t,a.job)
        if a.mode=='collect':
            root='/data/bwwu/lc-rec/a0-final-audit-20261113/'+a.job
            job=t.profile['job_root']+'/'+a.job
            code=f"""from pathlib import Path
import json,hashlib
result={{}}
for base,prefix in [(Path({root!r}),'results'),(Path({job!r}),'job')]:
 for p in base.rglob('*'):
  if p.is_file() and not p.is_symlink() and p.suffix in ['.json','.log'] and p.stat().st_size<8*1024**2:
   data=p.read_bytes();result[prefix+'/'+str(p.relative_to(base))]={{'text':data.decode(),'sha256':hashlib.sha256(data).hexdigest()}}
print(json.dumps(result))
"""
            files=json.loads(t.python(code,timeout=60))
            target=fc.REPO/'lc_rec/letter/evidence/final_audit_20261113/jobs'/a.job
            for name,v in files.items():
                dest=target/name;dest.parent.mkdir(parents=True,exist_ok=True);data=v['text'].encode();assert hashlib.sha256(data).hexdigest()==v['sha256']
                if dest.exists():assert dest.read_bytes()==data
                else:dest.write_bytes(data)
            fc.transport.atomic(SUPPORT/(a.job+'.collection.json'),{'files':{k:v['sha256'] for k,v in files.items()}})
    fc.transport.atomic(SUPPORT/(a.mode+'-'+a.job+'.json'),result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
