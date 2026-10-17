"""One bounded, read-only observation of the immutable shared continuation queue; never submit or signal."""
import contextlib
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

BASE=Path('/home/addaswsw/recommend-support/grid-a-launch-20261017')
DEPLOYMENT=BASE/'shared-20261018/queue-deployment.json'
sys.path.insert(0,'/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
from transport import Transport,atomic

deployment=json.loads(DEPLOYMENT.read_text())
saved=(BASE/'health_query.py').read_text()
health_code=saved.split("code=r'''",1)[1].split("'''.replace('CONFIG_PATH'",1)[0].replace('CONFIG_PATH',repr(deployment['remote_config']))
health_code=health_code.replace('ga-20261017-queue-v1','ga-20261018-shared-queue-v1')
extra=r'''
import contextlib,io,math,hashlib
buf=io.StringIO()
with contextlib.redirect_stdout(buf):
 exec(HEALTH_CODE,globals())
observed=json.loads(buf.getvalue())
def tail_jsonl(p,limit=16384):
 if not p.is_file():return []
 with p.open('rb') as stream:
  offset=max(0,p.stat().st_size-limit);stream.seek(offset);lines=stream.read().decode('utf-8','replace').splitlines()
 if offset and lines:lines=lines[1:]
 values=[]
 for line in lines:
  try:values.append(json.loads(line))
  except json.JSONDecodeError:pass
 return values
extras={}
for job,entry in observed['jobs'].items():
 if not isinstance(entry['status'],dict):continue
 out=root/job/'output';details={}
 for name in ('deadline.json','training_result.json','best_checkpoint_load.json','latest_recovery.json','initialization.json','structure.json','admission.json'):
  p=out/name
  if p.is_file() and p.stat().st_size<=262144:
   details[name]={'mtime_utc':datetime.datetime.fromtimestamp(p.stat().st_mtime,datetime.timezone.utc).isoformat(),'data':json.loads(p.read_text())}
 if '-train-' in job:
  updates=tail_jsonl(out/'optimizer_updates.jsonl')
  details['last_optimizer_update']=updates[-1] if updates else None
  p=out/'validations.jsonl'
  if p.is_file() and p.stat().st_size<=1048576:
   validations=[json.loads(s) for s in p.read_text().splitlines() if s.strip()]
   if validations:
    best=max(validations,key=lambda x:x['metrics']['val/recall@5'])
    details['validation_history_mechanical']={'completed_count':len(validations),'all_full_coverage_exact':all(x['targets']==35598 and x['exact_key_target_multiset'] for x in validations),
     'best_global_step':best['global_step'],'best_validation_number':best['validation_number'],'best_recall5':best['metrics']['val/recall@5'],
     'validations_since_first_best':len(validations)-best['validation_number'],'derived_from':'existing validations.jsonl; not a checkpoint callback read or result audit'}
  checkpoints=[]
  for p in sorted((out/'checkpoints').glob('*.ckpt')):
   st=p.stat();checkpoints.append({'path':str(p),'bytes':st.st_size,'mtime_utc':datetime.datetime.fromtimestamp(st.st_mtime,datetime.timezone.utc).isoformat(),'hash':'NOT_REHASHED_WHILE_LIVE'})
  details['checkpoint_files']=checkpoints
 extras[job]=details
observed['monitor_details']=extras
observed['remote_config_sha256']=hashlib.sha256(pathlib.Path(config_path).read_bytes()).hexdigest()
observed['read_only_monitor']=True
print(json.dumps(observed))
'''.replace('HEALTH_CODE',repr(health_code))
observed=json.loads(Transport('new-server').python(extra,timeout=60))
stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
record=DEPLOYMENT.parent/('health-monitor-'+stamp+'.json')
atomic(record,observed)
issues=[]
expected_sha=deployment['config_sha256']
if observed['remote_config_sha256']!=expected_sha:issues.append('IMMUTABLE_QUEUE_CONFIG_HASH_MISMATCH')
state=observed['queue_state'];stage_counts={}
if state:
 for s in state['stages'].values():stage_counts[s['status']]=stage_counts.get(s['status'],0)+1
 if not state['disk']['ok']:issues.append('QUEUE_DISK_GUARD_FALSE')
 age=(dt.datetime.fromisoformat(observed['observed_at_utc'])-dt.datetime.fromisoformat(state['observed_at'])).total_seconds()
 if state['status']=='RUNNING' and age>90:issues.append('QUEUE_STATE_STALE_OVER_90S')
else:issues.append('QUEUE_STATE_MISSING')
summary_jobs={}
for job,entry in observed['jobs'].items():
 status=entry['status']
 if not isinstance(status,dict):continue
 if status['status']=='RUNNING' and not all(status.get(k) for k in ('worker_alive','supervisor_alive','workgroup_live')):issues.append(job+':LIVE_IDENTITY_INCONSISTENT')
 if status['status'] in ('FAILED','TIMED_OUT','CONTROL_LOST'):issues.append(job+':'+status['status'])
 if '-train-' in job or '-quantize-' in job or job=='ga-20261018-shared-queue-v1':
  extra=observed['monitor_details'].get(job,{})
  update=extra.get('last_optimizer_update')
  if update:
   if not all(__import__('math').isfinite(float(x)) for x in update['returned_training_losses']):issues.append(job+':NONFINITE_LAST_TRAIN_LOSS')
   if not update.get('optimizer',{}).get('all_available_gradients_finite',False):issues.append(job+':LAST_GRADIENT_GUARD_FALSE')
  summary_jobs[job]={'status':status['status'],'exit_code':status.get('exit_code'),'progress':entry.get('details',{}).get('progress.json',{}).get('data'),
   'latest_validation':entry.get('details',{}).get('latest_validation.json',{}).get('data'),
   'last_optimizer_update':update,'validation_history':extra.get('validation_history_mechanical'),'checkpoints':extra.get('checkpoint_files'),
   'training_result':extra.get('training_result.json'),'best_checkpoint_gate':extra.get('best_checkpoint_load.json')}
 for log in entry.get('logs',{}).values():
  if any(p in log['tail'] for p in ('CUDA out of memory','torch.OutOfMemoryError','FloatingPointError:','Traceback (most recent call last):')):issues.append(job+':ERROR_IN_LOG_TAIL')
toys={}
for name in ('toys5300','toys5000'):
 entries=[e for j,e in observed['jobs'].items() if j.startswith('ga-20261017-'+name+'-')]
 okay=len(entries)==7 and all(isinstance(e['status'],dict) and e['status']['status']=='SUCCEEDED' and e['status'].get('exit_code')==0 and e['status'].get('cleanup_verified') and not e['status'].get('workgroup_live') for e in entries)
 toys[name]='COMPUTATION_COMPLETE_REVIEW_PENDING' if okay else 'CHECK_OR_INCOMPLETE'
summary={'at':observed['observed_at_cst'],'record':str(record),'queue_status':state['status'] if state else None,'stage_counts':stage_counts,'toys':toys,
 'jobs':summary_jobs,'accounting':state.get('accounting') if state else None,'gpu':observed['gpu'],'process_owners':observed['process_owners'],
 'disk_free':{p:x['free'] for p,x in observed['disk'].items()},'issues':issues,'source_commit':deployment['source_commit'],'method_changes':False}
atomic(record.with_name(record.stem+'-summary.json'),summary)
print(json.dumps(summary,ensure_ascii=False,indent=2))
