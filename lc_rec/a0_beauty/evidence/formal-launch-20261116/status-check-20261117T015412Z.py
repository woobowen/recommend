import json,datetime,hashlib,os,subprocess
from pathlib import Path
R=Path('/data/bwwu/lc-rec/a0-beauty-20261115/production-closure-20261115/formal-candidate-5b2917d/formal-run')
def read(p):return json.loads(Path(p).read_text())
def utc(t=None):return datetime.datetime.now(datetime.timezone.utc).isoformat() if t is None else datetime.datetime.fromtimestamp(t,datetime.timezone.utc).isoformat()
p=read(R/'plan.json');s=read(R/'state.json');a=s['attempts'][-1];j=Path(p['job_root'])/a['job_id'];status=read(j/'status.json');q=read(j/'request.json');expected={str(x['pid']):x for x in status.get('worker_group',{}).get('known_members',[])}
for x in (status.get('worker',{}),status.get('supervisor',{})):
 if x:expected[str(x['pid'])]=x
z={'utc':utc(),'host':os.uname().nodename,'plan_digest':hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest(),'identity':read(R/'plan.identity.json'),'state':s,'request':q,'runner_status':status,'budget':p['budget'],'stage_seconds':p['stage_seconds'],'processes':[],'records':[],'logs':[],'outputs':[]}
boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  if proc.stat().st_uid!=os.getuid():continue
  argv=(proc/'cmdline').read_bytes().decode().split('\0')[:-1]; stat=(proc/'stat').read_text().rsplit(')',1)[1].split();controller=any(x.endswith('/production_control.py') for x in argv) and 'run' in argv and str(R/'plan.json') in argv
  if controller or proc.name in expected:
   e=expected.get(proc.name);z['processes'].append({'pid':int(proc.name),'ppid':int(stat[1]),'state':stat[0],'start_ticks':stat[19],'cpu_ticks':int(stat[11])+int(stat[12]),'argv':argv,'runner_identity_match':None if not e else e['start_ticks']==stat[19] and e['boot_id']==boot,'controller':controller})
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
paths=[R/'training/best_selection.json']+list((R/'training/checkpoint_receipts').glob('*.json'))
for stage,row in s['stages'].items():
 out=Path(row['output']);sr=out/'stage_result.json'
 if sr.is_file():
  r=read(sr);z['records'].append({'path':str(sr),'verified':r.get('verified'),'stage':r.get('stage'),'proof':r.get('proof') if stage in ('train','verify_best_adapter') else None})
 for relative in ('primary/metrics.json','test/rank0.json','test/rank1.json','s3/paired_comparison.json','s4/paired_comparison.json','summary.json'):
  f=out/relative
  if f.is_file():paths.append(f)
 z['outputs'] += [{'path':str(f),'bytes':f.stat().st_size,'mtime_utc':utc(f.stat().st_mtime)} for f in out.glob('*') if f.is_file()]
 for c in ('s3','s4'):
  if (out/c).is_dir():z['outputs'] += [{'path':str(f),'bytes':f.stat().st_size,'mtime_utc':utc(f.stat().st_mtime)} for f in (out/c).iterdir() if f.is_file()]
for f in paths:
 if f.stat().st_size<=200000:
  data=f.read_bytes();z['records'].append({'path':str(f),'sha256':hashlib.sha256(data).hexdigest(),'record':json.loads(data)})
for cp in sorted((R/'training').glob('checkpoint-*')):
 f=cp/'trainer_state.json'
 if f.is_file():
  t=read(f);z['records'].append({'path':str(f),'complete_exists':(cp/'complete.json').is_file(),'record':{k:t.get(k) for k in ('global_step','epoch','best_metric','best_model_checkpoint')},'validation':[x for x in t.get('log_history',[]) if 'eval_loss' in x]})
for f in (R/'controller.log',j/'work.log',j/'supervisor.log',R/'training/events.rank0.jsonl'):
 if not f.is_file():continue
 st=f.stat()
 with f.open('rb') as stream:stream.seek(max(0,st.st_size-16000));data=stream.read(16000)
 z['logs'].append({'path':str(f),'source_bytes':st.st_size,'mtime_utc':utc(st.st_mtime),'tail_sha256':hashlib.sha256(data).hexdigest(),'tail':data.decode(errors='replace')})
for key,cmd in [('gpu',['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader']),('gpu_apps',['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'])]:z[key]=subprocess.check_output(cmd,text=True)
z['mem_available_bytes']=int(next(x.split()[1] for x in open('/proc/meminfo') if x.startswith('MemAvailable:')))*1024
v=os.statvfs(R);z['disk_available_bytes']=v.f_bavail*v.f_frsize;z['finished_utc']=utc();print(json.dumps(z,ensure_ascii=False,indent=2))
