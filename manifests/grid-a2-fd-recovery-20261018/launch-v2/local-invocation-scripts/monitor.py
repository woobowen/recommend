"""One read-only observation of this A2 recovery; never submit, restart or signal."""
import datetime as dt
import json
from pathlib import Path
import sys
sys.path.insert(0,'/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
from transport import Transport,atomic

B=Path(__file__).resolve().parent
d=json.loads((B/'queue-deployment.json').read_text())
code=r'''
import pathlib,json,importlib.util,datetime,subprocess,shutil,hashlib,math
path=pathlib.Path(CONFIG)
c=json.loads(path.read_text());S=pathlib.Path(c['snapshot']['root'])
spec=importlib.util.spec_from_file_location('recovery_read_only_queue',S/'experiments/launch_20261017/queue.py')
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
J=pathlib.Path(c['runner']['job_root'])
def tail(p,n=32768):
 if not p.is_file():return ''
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-n));return f.read().decode('utf-8','replace')
def last(p):
 text=tail(p)
 for line in reversed(text.splitlines()):
  try:return json.loads(line)
  except json.JSONDecodeError:pass
 return None
def read(p):return json.loads(p.read_text()) if p.is_file() else None
jobs={}
for stage in c['stages']:
 job=J/stage['request']['job_id'];out=job/'output'
 if not (job/'accepted.json').is_file():
  jobs[stage['id']]=dict(job_id=job.name,status='NOT_SUBMITTED',directory_exists=job.exists());continue
 status=q.runner_call(c,'query',job)
 result=read(out/'RESULT.json')
 if result and 'jobs' in result:
  result=dict(result,jobs=[{k:v for k,v in item.items() if k not in ['per_sample_query_pairs','least_similar_three_under_declared_mean_cosine']} for item in result['jobs']])
 fd=out/'fd'
 info=dict(job_id=job.name,status=status,progress=read(out/'progress.json'),
           last_optimizer=last(out/'optimizer_updates.jsonl'),last_validation=last(out/'validations.jsonl'),
           latest_validation=read(out/'latest_validation.json'),training_result=read(out/'training_result.json'),
           result=result,stage_exit=read(out/'stage-exit.json'),
           log_tail=tail(job/'work.log',8192),supervisor_tail=tail(job/'supervisor.log',2048))
 if fd.is_dir():
  info['fd']={p.name:(last(p) if p.suffix=='.jsonl' else read(p)) for p in sorted(fd.glob('*')) if p.is_file() and p.suffix in ['.json','.jsonl'] and '-batches' not in p.name}
  info['last_real_batch']={p.name:last(p) for p in sorted(fd.glob('*-batches.jsonl'))}
 if (out/'validations.jsonl').is_file() and (out/'validations.jsonl').stat().st_size<4194304:
  rows=[json.loads(line) for line in (out/'validations.jsonl').read_text().splitlines() if line]
  if rows:
   best=max(rows,key=lambda x:x['metrics']['val/recall@5'])
   info['validation_summary']=dict(events=len(rows),all_full_exact=all(x['targets']==35598 and x['exact_key_target_multiset'] for x in rows),best_step=best['global_step'],best_recall5=best['metrics']['val/recall@5'],events_since_best=len(rows)-best['validation_number'])
 info['checkpoints']=[dict(path=str(p),bytes=p.stat().st_size,mtime=p.stat().st_mtime,hash='not rehashed while live') for p in sorted((out/'checkpoints').glob('*.ckpt'))]
 jobs[stage['id']]=info
queuejob=q.runner_call(c,'query',J/c['queue_id']) if (J/c['queue_id']/'accepted.json').is_file() else None
state=read(pathlib.Path(c['state_dir'])/'queue-status.json')
gpu=q.gpu_snapshot()
print(json.dumps(dict(at_utc=q.utc(),at_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
 config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source=c['snapshot'],
 queue_runner=queuejob,queue_state=state,jobs=jobs,gpu=gpu,
 gpu_process_owners={str(p['pid']):subprocess.run(['ps','-p',str(p['pid']),'-o','pid=,uid=,user=,etime=,stat='],capture_output=True,text=True).stdout.strip() for card in gpu.values() for p in card['processes']},
 disk={p:shutil.disk_usage(p)._asdict() for p in ['/home/bwwu','/data/bwwu']},
 original_failure_status=read(J/'ga-20261017-sportsa2-quantize-v1/status.json'),read_only=True)))
'''.replace('CONFIG',repr(d['remote_config']))
value=json.loads(Transport('new-server').python(code,timeout=60))
assert value['config_sha256']==d['config_sha256']
stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
p=B/'monitor'/('observation-'+stamp+'.json');atomic(p,value)
summary=dict(at=value['at_cst'],record=str(p),queue=value['queue_state']['status'] if value['queue_state'] else None,
 counts={},gpu=value['gpu'],disk_free={p:x['free'] for p,x in value['disk'].items()},jobs={})
for name,item in value['jobs'].items():
 status=item['status'];name_status=status['status'] if isinstance(status,dict) else status
 summary['counts'][name_status]=summary['counts'].get(name_status,0)+1
 if isinstance(status,dict):
  info=dict(status=name_status,exit_code=status.get('exit_code'),cleanup=status.get('cleanup_verified'),
    progress=item.get('progress'),last_optimizer=item.get('last_optimizer'),last_validation=item.get('last_validation'),
    validation_summary=item.get('validation_summary'),result=item.get('result'),training_result=item.get('training_result'),
    errors_in_tail=[line for line in item['log_tail'].splitlines() if any(x in line for x in ['Error','Traceback','nan','inf','timeout'])])
  if 'fd' in item:
   info['fd_limits']={k:v for k,v in item['fd'].items() if '-limits' in k}
   info['fd_latest']={k:v for k,v in item['fd'].items() if '-samples' in k}
   info['workers_recorded']={str(rank):sum(f'rank-{rank}-worker' in k for k in item['fd']) for rank in (0,1)}
  summary['jobs'][name]=info
summary['accounting']=value['queue_state'].get('accounting') if value['queue_state'] else None
atomic(p.with_name(p.stem+'-summary.json'),summary)
atomic(B/'monitor/latest-summary.json',summary)
print(json.dumps(summary,ensure_ascii=False,indent=2))
