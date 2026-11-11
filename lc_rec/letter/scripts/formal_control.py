#!/usr/bin/env python3
"""A0 scoped Windows SSH/SCP controller; reuse accepted runner protocol.

No shared transport/profile is edited. Formal runner differs only by accepting
the user's 96h timeout. All state is under the existing job namespace and this
task's support/data directories; no remote code agent is installed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

REPO=Path(__file__).resolve().parents[3]
SUPPORT=Path('/home/addaswsw/recommend-support/letter-a0-normal-20261111')
REMOTE='/home/bwwu/lc-rec/letter-a0-normal-20261111'
PYTHON='/data/bwwu/envs/letter-20261110/bin/python'
sys.path.insert(0,'/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
import transport
import gridctl


def python_c(self,code,timeout=45):
    return self.ssh(shlex.join(['python3','-I','-S','-B','-c',code]),timeout=timeout)
transport.Transport.python=python_c


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['deploy','submit','query','logs','collect','compare','selftest'])
    p.add_argument('--commit');p.add_argument('--job')
    p.add_argument('--kind',choices=['acceptance-continuous','acceptance-resume','formal'])
    p.add_argument('--resume');p.add_argument('--resume-evaluation',action='store_true')
    p.add_argument('--continuous');p.add_argument('--resumed')
    a=p.parse_args();SUPPORT.mkdir(exist_ok=True,parents=True)
    t=transport.Transport('new-server')
    assert subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()=='wbw-lc-rec'
    sha=transport.sha
    def save(name,obj):transport.atomic(SUPPORT/'receipts'/name,obj)
    commit=subprocess.check_output(['git','rev-parse',a.commit or 'HEAD'],cwd=REPO,text=True).strip()
    release=REMOTE+'/releases/'+commit
    runner_bytes=subprocess.check_output(['git','show',commit+':lc_rec/letter/scripts/formal_runner.py'],cwd=REPO)
    runner_sha=hashlib.sha256(runner_bytes).hexdigest()
    runner=t.profile['remote_support']+'/scripts/runner-a0-96h-'+runner_sha[:16]+'.py'
    t.profile=dict(t.profile,runner=runner,runner_sha256=runner_sha)
    if a.mode=='deploy':
        t.identity()
        archive=SUPPORT/(commit+'.tar.gz')
        subprocess.run(['git','archive','--format=tar.gz','--output='+str(archive),commit,'lc_rec/letter'],cwd=REPO,check=True)
        upload=t.profile['remote_support']+'/transfers/letter-a0-'+commit+'.tar.gz'
        receipt=t.put(archive,upload)
        # Inherit the exact accepted sidecar, while preserving the old runner.
        sidecar=t.python('from pathlib import Path\nprint(Path('+repr(t.profile['remote_support']+'/scripts/runner-398768a732bbfebd.py.deployment.json')+').read_text(),end="")')
        local_sidecar=SUPPORT/'deployment.json';local_sidecar.write_text(sidecar)
        assert sha(local_sidecar)==t.profile['deployment_sha256']
        t.put(local_sidecar,runner+'.deployment.json')
        local_runner=SUPPORT/'runners'/(runner_sha+'.py')
        local_runner.parent.mkdir(exist_ok=True);local_runner.write_bytes(runner_bytes)
        t.put(local_runner,runner)
        code=f"""from pathlib import Path,PurePosixPath
import hashlib,json,tarfile
archive=Path({upload!r});root=Path({release!r});rows=[]
assert hashlib.sha256(archive.read_bytes()).hexdigest()=={receipt['sha256']!r}
with tarfile.open(archive,'r:gz') as tf:
 members=tf.getmembers();names=[m.name for m in members];assert len(names)==len(set(names))
 for m in members:
  p=PurePosixPath(m.name);assert not p.is_absolute() and '..' not in p.parts and (m.isfile() or m.isdir())
 if not root.exists():root.mkdir(parents=True);tf.extractall(root,filter='data')
 for m in members:
  if m.isfile():
   data=tf.extractfile(m).read();path=root/m.name;assert path.is_file() and not path.is_symlink() and path.read_bytes()==data
   rows.append({{'path':m.name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}})
r={{'RUN_COMMIT':{commit!r},'release':str(root),'files':rows,'not_a_git_checkout':True,'runner':{runner!r},'runner_sha256':{runner_sha!r}}}
v=root/'RUN_VERSION.json'
if v.exists():assert json.loads(v.read_text())==r
else:v.write_text(json.dumps(r,indent=2)+'\\n')
print(json.dumps(r))
"""
        r=json.loads(t.python(code,timeout=60));save('deployment-'+commit+'.json',r)
        print(json.dumps({'release':release,'files_verified':len(r['files']),'runner_sha256':runner_sha}));return
    if a.mode=='compare':
        assert a.continuous and a.resumed
        old=json.loads((REPO/'lc_rec/letter/evidence/autograd_20261110/jobs/letter-20261110-autograd-cpu-r0/request.json').read_text())
        command=['env','-u','DS_ACCELERATOR',*[k+'='+v for k,v in old['env'].items()],PYTHON,release+'/lc_rec/letter/scripts/formal_compare.py','--continuous',a.continuous,
                 '--resumed',a.resumed,'--output','/data/bwwu/lc-rec/a0-normal-20261111/acceptance_comparison.json']
        r=t.ssh(shlex.join(command),timeout=180)
        print(r);return
    assert a.job and a.job.startswith('letter-a0-20261111-');gridctl.valid(a.job)
    if a.mode in ['submit','selftest']:
        assert a.kind or a.mode=='selftest'
        gate=None
        if a.kind=='formal':
            gate=SUPPORT/'receipts/prelaunch_review.json'
            assert gate.is_file()
            review=json.loads(gate.read_text())
            assert review['status']=='PASS' and review['RUN_COMMIT']==commit
            assert review['config_sha256']==sha(REPO/'lc_rec/letter/config/formal_instruments_seed42.json')
            assert review['formal_train_sha256']==sha(REPO/'lc_rec/letter/scripts/formal_train.py')
        deployment=json.loads((SUPPORT/'receipts'/('deployment-'+commit+'.json')).read_text())
        old=json.loads((REPO/'lc_rec/letter/evidence/autograd_20261110/jobs/letter-20261110-autograd-gpu-r0/request.json').read_text())
        req={k:old[k] for k in ['schema','host','user','uid','namespace','deployment_sha256']}
        approval=release+'/lc_rec/letter/evidence/formal_20261111/precision_approval.json'
        argv=[PYTHON,release+'/lc_rec/letter/scripts/formal_pipeline.py','--release',release,'--run-commit',commit,
              '--job-id',a.job,'--kind',a.kind,'--precision-approval',approval]
        if a.resume:argv+=['--resume',a.resume]
        if a.resume_evaluation:argv+=['--resume-evaluation']
        env=dict(old['env']);env['PYTHONUNBUFFERED']='1'
        if a.mode=='selftest':
            argv=[PYTHON,release+'/lc_rec/letter/scripts/runner_timeout_fixture.py']
            env['CUDA_VISIBLE_DEVICES']='';env.pop('DS_ACCELERATOR',None)
        req.update(job_id=a.job,capture_version=commit,cwd=release,argv=argv,env=env,
          timeout_seconds=5 if a.mode=='selftest' else 345600 if a.kind=='formal' else 1800,
          inputs=[{'path':release+'/'+x['path'],'sha256':x['sha256']} for x in deployment['files'] if
                  x['path'].startswith(('lc_rec/letter/scripts/','lc_rec/letter/config/')) or x['path'].endswith('/precision_approval.json')])
        if gate:
            remote_gate=t.profile['remote_support']+'/requests/'+a.job+'.prelaunch.json'
            t.put(gate,remote_gate)
            req['argv']+=['--prelaunch-review',remote_gate]
            req['inputs'].append({'path':remote_gate,'sha256':sha(gate)})
        path=SUPPORT/'requests'/(a.job+'.json')
        if path.exists():assert json.loads(path.read_text())==req
        else:transport.atomic(path,req)
        r=gridctl.submit(t,path)
    else:
        r=gridctl.query(t,a.job)
        job=t.profile['job_root']+'/'+a.job
        if a.mode=='logs':print(t.ssh(shlex.join(['tail','-n','65',job+'/work.log'])))
        if a.mode=='collect':
            # Lightweight external receipts; payloads never cross to WSL.
            code=f"""from pathlib import Path
import json,hashlib
job=Path({job!r});root=Path('/data/bwwu/lc-rec/a0-normal-20261111')/{a.job!r}
result={{}}
roots=[(job,'job'),(root,'run')]
request=json.loads((job/'request.json').read_text())
if '--kind' in request['argv'] and request['argv'][request['argv'].index('--kind')+1]=='formal':
 roots.append((Path('/data/bwwu/lc-rec/a0-normal-20261111/formal_seed42/training'),'formal_training'))
for base,prefix in roots:
 for p in base.rglob('*'):
  if p.is_file() and not p.is_symlink() and p.suffix in ['.json','.jsonl','.log'] and p.stat().st_size<=8*1024**2 and not p.name.startswith('predictions'):
   result[prefix+'/'+str(p.relative_to(base))]={{'text':p.read_bytes().decode('utf-8'),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}}
print(json.dumps(result))
"""
            data=json.loads(t.python(code,timeout=60))
            dest=REPO/'lc_rec/letter/evidence/formal_20261111/jobs'/a.job
            for n,v in data.items():
                path=dest/n;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(v['text'].encode('utf-8'))
                assert sha(path)==v['sha256']
            save('collected-'+a.job+'.json',{'job_id':a.job,'files':{k:v['sha256'] for k,v in data.items()}})
    save(a.mode+'-'+a.job+'.json',r)
    print(json.dumps({k:v for k,v in r.items() if k not in ['artifacts','worker_group','cleanup']},indent=2))


if __name__=='__main__':main()
