from pathlib import Path
import base64,hashlib,json,shutil,sys
sys.path.insert(0,'/home/addaswsw/recommend-worktrees/lc-rec/lc_rec/a0_beauty')
import closure_ops as op
import transport
local=op.LOCAL;t=transport.Transport('new-server');t.profile=dict(t.profile,remote_support=op.BASE)
code=f'''from pathlib import Path
import base64,hashlib,json,datetime,subprocess,os
base=Path({op.BASE!r});files={{}}
for p in base.rglob('*'):
 if not p.is_file() or p.is_symlink():continue
 rel=str(p.relative_to(base))
 if '/releases/' in '/'+rel and not p.name=='RUN_VERSION.json':continue
 if p.name.startswith('predictions') or 'training/checkpoint-' in rel or '/best_adapter/' in rel:continue
 if p.suffix not in ('.json','.jsonl','.log','.npy','.npz'):continue
 if p.stat().st_size>2*1024**2:continue
 b=p.read_bytes();files[rel]={{'data':base64.b64encode(b).decode(),'sha256':hashlib.sha256(b).hexdigest()}}
prior=[]
for root,pattern in [(Path('/home/bwwu/grid-audits/sid-extensions-p0/jobs'),'letter-beauty*/request.json'),(base/'cpu-jobs','*/request.json')]:
 for reqpath in root.glob(pattern):
  req=json.loads(reqpath.read_text())
  if req.get('env',{{}}).get('CUDA_VISIBLE_DEVICES')!='':continue
  p=reqpath.parent/'status.json';s=json.loads(p.read_text())
  assert s['status'] not in ('ACCEPTED','RUNNING') and s.get('cleanup',{{}}).get('verified'),str(reqpath)
  seconds=(datetime.datetime.fromisoformat(s['ended_at'])-datetime.datetime.fromisoformat(s['started_at'])).total_seconds()
  prior.append({{'job_id':req['job_id'],'wall_seconds':seconds,'status':s['status'],'exit_code':s['exit_code'],'status_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}})
logical=sum(p.stat().st_size for p in base.rglob('*') if p.is_file())
assert logical<8*1024**3
budget={{'wall_seconds_upper_bound_including_nested_jobs':sum(x['wall_seconds'] for x in prior),'limit_seconds':7200,'temporary_logical_bytes':logical,'temporary_limit_bytes':8*1024**3,'attempts':prior,'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}}
assert budget['wall_seconds_upper_bound_including_nested_jobs']<7200
versions=subprocess.check_output([{op.PYTHON!r},'-c',"import sys,json,importlib.metadata as m;print(json.dumps(dict(python=sys.version,packages={{x:m.version(x) for x in ['torch','transformers','deepspeed','numpy']}})))"],text=True)
fixture=Path({op.FIXTURE!r});summary=json.loads((fixture/'summary.json').read_text())
source={{'fixture_root':str(fixture),'summary_sha256':hashlib.sha256((fixture/'summary.json').read_bytes()).hexdigest(),'files':[row for condition in summary['conditions'].values() for row in condition['prediction_files']]}}
print(json.dumps({{'files':files,'budget':budget,'versions':json.loads(versions),'source_inputs':source}}))'''
result=json.loads(t.python(code,timeout=120));dest=op.REPO/'lc_rec/a0_beauty/evidence/production-closure/final-server'
for n,v in result['files'].items():
 path=dest/n;path.parent.mkdir(parents=True,exist_ok=True);b=base64.b64decode(v['data']);assert hashlib.sha256(b).hexdigest()==v['sha256'];path.write_bytes(b)
for n in ('budget','versions','source_inputs'):
 transport.atomic(dest/(n+'.json'),result[n])
for name in ('run_followup_jobs.py','collect_final.py'):
 (dest/name).write_bytes((local/name).read_bytes())
(dest/'local-transport').mkdir(exist_ok=True)
for pattern in ('*.request.json','*.submit.json','*.status.json'):
 for p in local.glob(pattern):(dest/'local-transport'/p.name).write_bytes(p.read_bytes())
print(json.dumps({'files':len(result['files']),'budget':{k:v for k,v in result['budget'].items() if k!='attempts'},'versions':result['versions']}))
