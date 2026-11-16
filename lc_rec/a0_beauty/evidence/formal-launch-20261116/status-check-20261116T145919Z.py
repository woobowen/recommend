import json,datetime,hashlib,os,subprocess
from pathlib import Path
root=Path('/data/bwwu/lc-rec/a0-beauty-20261115/production-closure-20261115/formal-candidate-5b2917d/formal-run')
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
read=lambda p:json.loads(Path(p).read_text())
plan=read(root/'plan.json'); state=read(root/'state.json'); active=state['attempts'][-1]; job=Path(plan['job_root'])/active['job_id']; status=read(job/'status.json'); req=read(job/'request.json')
result={'observed_utc':now(),'host':os.uname().nodename,'plan_digest':hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest(),'plan_identity':read(root/'plan.identity.json'),'stage':active['stage'],'attempt':active['attempt'],'job_id':active['job_id'],'state_status':state['status'],'stages':state['stages'],'attempts':[{k:a.get(k) for k in ('stage','attempt','job_id','status','charged_seconds')} for a in state['attempts']],'runner':{k:v for k,v in status.items() if k!='worker_group'},'request_argv':req['argv'],'request_sha256':hashlib.sha256(json.dumps(req,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest(),'budget':plan['budget'],'stage_seconds':plan['stage_seconds'],'processes':[],'files':[]}
boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(); expected={str(x['pid']):x for x in status.get('worker_group',{}).get('known_members',[])}
for x in (status.get('worker',{}),status.get('supervisor',{})):
 if x:expected[str(x['pid'])]=x
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  if p.stat().st_uid!=os.getuid():continue
  argv=(p/'cmdline').read_bytes().decode().split('\0')[:-1]; raw=(p/'stat').read_text().rsplit(')',1)[1].split(); ticks=raw[19]
  controller=any('production_control.py' in x for x in argv) and 'run' in argv and str(root/'plan.json') in argv
  important=p.name in expected and (p.name in (str(status.get('worker',{}).get('pid')),str(status.get('supervisor',{}).get('pid'))) or any(x.endswith('/formal_train.py') for x in argv))
  if controller or important:
   e=expected.get(p.name); result['processes'].append({'pid':int(p.name),'ppid':int(raw[1]),'sid':int(raw[3]),'state':raw[0],'start_ticks':ticks,'boot_id':boot,'argv':argv,'runner_identity_match': None if not e else ticks==e['start_ticks'] and boot==e['boot_id'],'is_controller':controller})
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
paths=[job/'work.log',job/'supervisor.log',root/'controller.log',root/'training/events.rank0.jsonl',root/'training/events.rank1.jsonl']
for p in paths:
 if not p.is_file():continue
 st=p.stat(); limit=24000
 with p.open('rb') as f:f.seek(max(0,st.st_size-limit)); data=f.read(limit)
 result['files'].append({'path':str(p),'source_size_at_read':st.st_size,'mtime_utc':datetime.datetime.fromtimestamp(st.st_mtime,datetime.timezone.utc).isoformat(),'tail_sha256':hashlib.sha256(data).hexdigest(),'tail':data.decode(errors='replace')})
result['checkpoints']=[]
for cp in sorted((root/'training').glob('checkpoint-*')):
 row={'path':str(cp),'complete_exists':(cp/'complete.json').is_file()}
 if (cp/'trainer_state.json').is_file():
  s=read(cp/'trainer_state.json');row['trainer_state']={k:s.get(k) for k in ('global_step','epoch','best_metric','best_model_checkpoint','log_history')}
 if row['complete_exists']:
  m=read(cp/'complete.json');row['complete_summary']={k:v for k,v in m.items() if k!='files'}
 result['checkpoints'].append(row)
result['receipts']=[{'path':str(p),'record':read(p)} for p in sorted((root/'training/checkpoint_receipts').glob('*.json'))]
result['best_selection']=read(root/'training/best_selection.json') if (root/'training/best_selection.json').is_file() else None
for key,cmd in [('gpu',['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader']),('gpu_apps',['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader']),('process_times',['ps','-p',','.join(str(x['pid']) for x in result['processes']),'-o','pid,ppid,sid,lstart,etime,args'])]:result[key]=subprocess.check_output(cmd,text=True)
result['mem_available_bytes']=int(next(x.split()[1] for x in open('/proc/meminfo') if x.startswith('MemAvailable:')))*1024
v=os.statvfs(root);result['disk_available_bytes']=v.f_bavail*v.f_frsize
result['finished_utc']=now();print(json.dumps(result,ensure_ascii=False,indent=2))
