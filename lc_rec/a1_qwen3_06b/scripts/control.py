#!/usr/bin/env python3
"""A1 scoped client of the accepted Windows transport and persistent runner."""
import argparse, datetime as dt, hashlib, json, shlex, subprocess, sys, tarfile
from pathlib import Path
REPO=Path(__file__).resolve().parents[3]
TASK=REPO/'lc_rec/a1_qwen3_06b'
SUPPORT=Path('/home/addaswsw/recommend-support/a1-qwen3-06b-instruments-20261113')
REMOTE='/home/bwwu/lc-rec/a1-qwen3-06b-instruments-20261113'
DATA='/data/bwwu/lc-rec/a1-qwen3-06b-instruments-20261113'
PYTHON='/data/bwwu/envs/qwen-preflight-20261110-qwen3/bin/python'
MODEL='/data/bwwu/lc-rec/qwen-preflight-20261110/models/Qwen3-0.6B-Base/da87bfb608c14b7cf20ba1ce41287e8de496c0cd'
PREFIX='letter-a1-q06-20261113-'
sys.path.insert(0,'/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
import transport, gridctl
transport.Transport.python=lambda self,code,timeout=45:self.ssh(shlex.join(['python3','-I','-S','-B','-c',code]),timeout=timeout)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('mode',choices=['deploy','submit','query','logs','collect','observe'])
 p.add_argument('--commit');p.add_argument('--job');p.add_argument('--kind',choices=['cpu','selftest','continuous','resume','formal','compare'])
 p.add_argument('--resume');p.add_argument('--resume-evaluation',action='store_true');a=p.parse_args()
 assert subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()=='wbw-lc-rec'
 commit=subprocess.check_output(['git','rev-parse',a.commit or 'HEAD'],cwd=REPO,text=True).strip()
 t=transport.Transport('new-server');release=REMOTE+'/releases/'+commit
 runner_bytes=subprocess.check_output(['git','show',commit+':lc_rec/a1_qwen3_06b/scripts/runner.py'],cwd=REPO)
 runner_sha=hashlib.sha256(runner_bytes).hexdigest();runner=t.profile['remote_support']+'/scripts/runner-a1-96h-'+runner_sha[:16]+'.py'
 t.profile=dict(t.profile,runner=runner,runner_sha256=runner_sha)
 def save(n,v):transport.atomic(SUPPORT/'receipts'/n,v)
 if a.mode=='deploy':
  identity=t.identity();save('identity.json',identity)
  archive=SUPPORT/(commit+'.tar.gz');archive.parent.mkdir(parents=True,exist_ok=True)
  paths=['lc_rec/a1_qwen3_06b','lc_rec/qwen_preflight_20261110/scripts/common.py','lc_rec/qwen_preflight_20261110/scripts/cpu_audit.py','lc_rec/letter/vendor/LETTER','lc_rec/letter/scripts/offline_metrics.py','lc_rec/letter/config/a0_evaluation_v1.json','lc_rec/letter/EVALUATION_PROTOCOL_V1.md','lc_rec/letter/review/final_audit_20261113','lc_rec/letter/review/formal_det_20261112/independent_metrics.py','lc_rec/letter/scripts/runner_timeout_fixture.py']
  subprocess.run(['git','archive','--format=tar.gz','--output='+str(archive),commit,*paths],cwd=REPO,check=True)
  upload=t.profile['remote_support']+'/transfers/a1-q06-'+commit+'.tar.gz';sent=t.put(archive,upload)
  sidecar=t.python('from pathlib import Path\nprint(Path('+repr(t.profile['remote_support']+'/scripts/runner-398768a732bbfebd.py.deployment.json')+').read_text(),end="")')
  local=SUPPORT/'deployment.json';local.write_text(sidecar);assert transport.sha(local)==t.profile['deployment_sha256'];t.put(local,runner+'.deployment.json')
  local=SUPPORT/'runners'/(runner_sha+'.py');local.parent.mkdir(exist_ok=True);local.write_bytes(runner_bytes);t.put(local,runner)
  code=f"""from pathlib import Path,PurePosixPath
import json,hashlib,tarfile
root=Path({release!r});archive=Path({upload!r});assert hashlib.sha256(archive.read_bytes()).hexdigest()=={sent['sha256']!r}
rows=[]
with tarfile.open(archive) as tf:
 members=tf.getmembers();assert len(members)==len(set(m.name for m in members))
 for m in members:assert not PurePosixPath(m.name).is_absolute() and '..' not in PurePosixPath(m.name).parts and (m.isfile() or m.isdir())
 if not root.exists():root.mkdir(parents=True);tf.extractall(root,filter='data')
 for m in members:
  if m.isfile():
   data=tf.extractfile(m).read();q=root/m.name;assert q.is_file() and not q.is_symlink() and q.read_bytes()==data
   rows.append({{'path':m.name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}})
r={{'RUN_COMMIT':{commit!r},'release':str(root),'not_a_git_checkout':True,'runner':{runner!r},'runner_sha256':{runner_sha!r},'files':rows}}
p=root/'RUN_VERSION.json'
if p.exists():assert json.loads(p.read_text())==r
else:p.write_text(json.dumps(r,indent=2)+'\\n')
print(json.dumps(r))"""
  r=json.loads(t.python(code,timeout=90));save('deployment-'+commit+'.json',r)
  print(json.dumps({'release':release,'files_verified':len(r['files']),'runner_sha256':runner_sha}));return
 assert a.job and a.job.startswith(PREFIX);gridctl.valid(a.job)
 job=t.profile['job_root']+'/'+a.job
 if a.mode=='submit':
  assert a.kind
  deployment=json.loads((SUPPORT/'receipts'/('deployment-'+commit+'.json')).read_text())
  old=json.loads((REPO/'lc_rec/letter/evidence/autograd_20261110/jobs/letter-20261110-autograd-gpu-r0/request.json').read_text())
  req={k:old[k] for k in ['schema','host','user','uid','namespace','deployment_sha256']}
  env=dict(old['env']);env.update(PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
  env.pop('DS_ACCELERATOR',None);env['TORCH_EXTENSIONS_DIR']=DATA+'/torch_extensions'
  argv=[PYTHON,release+'/lc_rec/a1_qwen3_06b/scripts/pipeline.py','--release',release,'--run-commit',commit,'--job-id',a.job,'--kind',a.kind]
  if a.resume:argv+=['--resume',a.resume]
  if a.resume_evaluation:argv+=['--resume-evaluation']
  timeout=1800
  inputs=[{'path':release+'/'+r['path'],'sha256':r['sha256']} for r in deployment['files'] if '/scripts/' in r['path'] or '/config/' in r['path'] or '/vendor/' in r['path']]
  if a.kind=='selftest':
   argv=[PYTHON,release+'/lc_rec/letter/scripts/runner_timeout_fixture.py'];env['CUDA_VISIBLE_DEVICES']='';timeout=5
  elif a.kind=='cpu':
   env['CUDA_VISIBLE_DEVICES']='';timeout=300
  elif a.kind=='compare':env['CUDA_VISIBLE_DEVICES']='';timeout=300
  else:
   ledger_name='formal_seed42_ledger.json' if a.kind=='formal' else 'preflight_ledger.json'
   code='import sys,json\nsys.path.insert(0,'+repr(release+'/lc_rec/a1_qwen3_06b/scripts')+')\nfrom ledger import reconcile\nprint(json.dumps(reconcile('+repr(DATA+'/'+ledger_name)+')))'
   ledger=json.loads(t.python(code));used=sum(x['wall_seconds'] for x in ledger['attempts'])
   assert a.kind!='formal' or not ledger['attempts'] or a.resume or a.resume_evaluation
   timeout=int((345600 if a.kind=='formal' else 1800)-used-10);assert timeout>0
   if a.kind=='formal':
    gate=SUPPORT/'receipts/prelaunch_review.json';review=json.loads(gate.read_text())
    assert review['status']=='PASS' and review['RUN_COMMIT']==commit
    assert review['source_sha256']=={str(p.relative_to(TASK)):transport.sha(p) for p in sorted((TASK/'scripts').glob('*.py'))}
    remote_gate=t.profile['remote_support']+'/requests/'+a.job+'.prelaunch.json';t.put(gate,remote_gate)
    argv+=['--prelaunch-review',remote_gate];inputs.append({'path':remote_gate,'sha256':transport.sha(gate)})
  req.update(job_id=a.job,capture_version=commit,cwd=release,argv=argv,env=env,inputs=inputs,timeout_seconds=timeout)
  path=SUPPORT/'requests'/(a.job+'.json')
  if path.exists():assert json.loads(path.read_text())==req
  else:transport.atomic(path,req)
  r=gridctl.submit(t,path)
 else:
  r=gridctl.query(t,a.job)
  if a.mode=='logs':print(t.ssh(shlex.join(['tail','-n','45',job+'/work.log'])))
  if a.mode in ['collect','observe']:
   code=f"""from pathlib import Path
import json,hashlib,subprocess,datetime
job=Path({job!r});root=Path({DATA!r});run=root/{a.job!r}
r={{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':job.name,'files':{{}}}}
for base,prefix in [(job,'job'),(run,'run'),(root/'formal_seed42/training','formal_training')]:
 if base.exists():
  for q in base.rglob('*'):
   if q.is_file() and not q.is_symlink() and q.suffix in ['.json','.jsonl','.log'] and q.stat().st_size<=8*1024**2 and not q.name.startswith('predictions'):
    b=q.read_bytes();r['files'][prefix+'/'+str(q.relative_to(base))]={{'text':b.decode(),'sha256':hashlib.sha256(b).hexdigest()}}
r['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu,temperature.gpu','--format=csv'],text=True)
r['ram']=subprocess.check_output(['free','-b'],text=True);r['disk']=subprocess.check_output(['df','-B1',str(root) if root.exists() else '/data/bwwu'],text=True)
print(json.dumps(r))"""
   data=json.loads(t.python(code,timeout=60))
   dest=TASK/'evidence/jobs'/a.job
   for n,v in data['files'].items():
    q=dest/n;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(v['text'].encode());assert transport.sha(q)==v['sha256']
   save('collected-'+a.job+'.json',{**{k:v for k,v in data.items() if k!='files'},'files':{k:v['sha256'] for k,v in data['files'].items()}})
 save(a.mode+'-'+a.job+'.json',r)
 print(json.dumps({k:r.get(k) for k in ['job_id','status','exit_code','cleanup_verified','worker_alive','supervisor_alive','workgroup_live']},indent=2))
if __name__=='__main__':main()
