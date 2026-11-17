"""Prepare an at-most-once CPU supplement; never invoke original controller."""
import datetime as dt, hashlib, json, os, resource, subprocess, sys, time
from pathlib import Path
START=time.monotonic()
def load(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def ident(p):
 p=Path(p);assert p.is_file() and not p.is_symlink();return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def write(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
c=load(Path(__file__).with_name('config.json'));root=Path(c['original_root']);sroot=Path(c['supplement_root']);p=load(root/'plan.json');state=load(root/'state.json')
assert hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()==c['original_plan_digest']
assert p['run_commit']==c['original_run'] and p['config_sha256']==c['config_sha256']
assert state['attempts'][-1]['job_id']==c['original_failed_job'] and state['attempts'][-1]['status']=='TIMED_OUT' and 'collect' not in state['stages']
assert sha(p['runner'])=='cc1a1e277faa9bc713a4d0d1b620248d47c398e8ed17d4ed3d7ab52805cc657e'
assert sha(p['runner']+'.deployment.json')=='eafd22574dc2d07fd253822732a572200032fcd7e88654140021e2e045087894'
job=Path(p['job_root'])/c['job_id']
if job.exists():
 print(json.dumps({'existing_job':str(job),'action':'QUERY_ONLY_NO_RESUBMIT'}));sys.exit(3)
assert not sroot.exists(), 'supplement root already exists; reconcile before preparation'
for a in state['attempts']:
 st=load(Path(p['job_root'])/a['job_id']/'status.json');assert st.get('ended_at') and st.get('cleanup',{}).get('verified')
oldcpu=sum(a['charged_seconds'] for a in state['attempts'] if a['stage'] not in ('train','normal','s3','s4'))
assert oldcpu+c['cpu_budget_seconds']<=p['budget']['cpu']
assert int(next(x.split()[1] for x in open('/proc/meminfo') if x.startswith('MemAvailable:')))*1024>=40*1024**3
v=os.statvfs(root);assert v.f_bavail*v.f_frsize>=max(100*1024**3,v.f_blocks*v.f_frsize//10)
s3=root/'attempts/paired/a0/s3/paired_comparison.json';assert sha(s3)==c['s3_sha256']
original=[root/'plan.json',root/'plan.identity.json',root/'state.json',Path(p['runner']),Path(p['runner']+'.deployment.json')]
original+=list(Path('/data/bwwu/lc-rec/a0-beauty-20261115/formal-experiment-registry').glob('*.json'))
failed=Path(p['job_root'])/c['original_failed_job'];original += [x for x in failed.iterdir() if x.is_file() and x.suffix in ('.json','.log')]
original += [s3,s3.with_name('paired_bootstrap_difference_means.npy')]
inputs=[]
for row in p['inputs']:
 assert sha(row['path'])==row['sha256'];inputs.append(ident(row['path']))
for cond in ('normal','s3','s4'):
 primary=root/f'attempts/{cond}_eval/a0/primary';m=load(primary/'metrics.json');assert m['status']=='PASS' and m['users']==22363 and m['run_commit']==c['original_run']
 for row in m['per_user_files']+m['prediction_files']:
  assert ident(row['path'])['sha256']==row['sha256']
 original += [primary/n for n in ('metrics.json','per_user.csv','per_user_metrics.npz','groups.json','bootstrap.json','bootstrap_draw_means.npy')]
 original += [root/f'attempts/{cond}/a0/test/rank{r}.json' for r in (0,1)]
 inputs += [ident(row['path']) for row in m['prediction_files']]
original += [root/'training/best_selection.json']+list((root/'training/checkpoint_receipts').glob('*.json'))
assert load(root/'training/best_selection.json')['adapter_sha256']==c['best_adapter_sha256']
sroot.mkdir(parents=True);(sroot/'original-evidence').mkdir()
protected=[ident(x) for x in sorted(set(original))];inputs+=protected
for f in (Path(__file__),Path(__file__).with_name('config.json'),Path(__file__).with_name('worker.py')):inputs.append(ident(f))
# Preserve small original receipts by byte copy; large CSV/NPZ/predictions stay server-side.
for f in original:
 f=Path(f)
 if f.stat().st_size<250000 and f.suffix in ('.json','.log'):
  dest=sroot/'original-evidence'/hashlib.sha256(str(f).encode()).hexdigest()[:12];dest.write_bytes(f.read_bytes())
write(sroot/'protected-inputs.json',protected)
write(sroot/'input-manifest.json',list({x['path']:x for x in inputs}.values()))
deploy=load(Path(__file__).with_name('deployment.json'));env=dict(p['env']);env.update(CUDA_VISIBLE_DEVICES='',PYTHONOPTIMIZE='0',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2')
request={'schema':1,'host':p['namespace']['host'],'user':p['namespace']['user'],'uid':p['namespace']['uid'],'namespace':p['namespace']['namespace'],'deployment_sha256':p['namespace']['deployment_sha256'],'capture_version':deploy['aux_commit'],'job_id':c['job_id'],'cwd':str(sroot),'argv':[p['python'],str(Path(__file__).with_name('worker.py')),'--config',str(Path(__file__).with_name('config.json'))],'env':env,'timeout_seconds':c['request_timeout_seconds'],'inputs':load(sroot/'input-manifest.json')+[ident(sroot/'input-manifest.json'),ident(sroot/'protected-inputs.json'),ident(Path(__file__).with_name('deployment.json'))],'supplemental':True,'designation':c['designation'],'original_plan_digest':c['original_plan_digest'],'original_failed_job':c['original_failed_job']}
write(sroot/'request.json',request)
write(sroot/'prelaunch.json',{'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'original_cpu_charged_seconds':oldcpu,'new_cpu_limit_seconds':c['cpu_budget_seconds'],'aux_commit':deploy['aux_commit'],'elapsed_seconds':time.monotonic()-START,'cpu_seconds':resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,'protected_files':len(protected),'inputs':len(request['inputs']),'job_id':c['job_id']})
print(json.dumps(load(sroot/'prelaunch.json')))
