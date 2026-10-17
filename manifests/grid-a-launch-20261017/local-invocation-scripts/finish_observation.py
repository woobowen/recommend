"""Close the launch observation using actual saved checkpoints, never run models."""
import argparse
import datetime as dt
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,'/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
from transport import Transport,atomic

parser=argparse.ArgumentParser()
parser.add_argument('--compiled',type=Path,required=True)
args=parser.parse_args();root=args.compiled
health={name:json.loads((root/('health-'+name+'.json')).read_text()) for name in ('T0','T10','T20')}
start=dt.datetime.fromisoformat(health['T0']['observed_at_utc'])
end=dt.datetime.fromisoformat(health['T20']['observed_at_utc'])
assert (end-start).total_seconds()>=1200,'Twenty real minutes must elapse after observed computation'
final=health['T20'];config=json.loads((root/'queue-config.json').read_text());state=final['queue_state']
assert state and state['config_sha256']==json.loads((root/'queue-deployment.json').read_text())['config_sha256']
machine_path=config['state_dir']+'/health.jsonl'
code='from pathlib import Path;import json;print(json.dumps([json.loads(l) for l in Path('+repr(machine_path)+').read_text().splitlines() if l.strip()]))'
machine=json.loads(Transport('new-server').python(code))
times=[dt.datetime.fromisoformat(r['observed_at']) for r in machine]
selected=[r for r,t in zip(machine,times) if start-dt.timedelta(seconds=60)<=t<=end]
selected_times=[dt.datetime.fromisoformat(r['observed_at']) for r in selected]
gaps=[(b-a).total_seconds() for a,b in zip(selected_times,selected_times[1:])]
continuity={'server_log':machine_path,'samples':len(selected),'first':selected[0]['observed_at'],
 'last':selected[-1]['observed_at'],'maximum_gap_seconds':max(gaps,default=0),
 'first_boundary_gap_seconds':max(0,(selected_times[0]-start).total_seconds()),
 'last_boundary_gap_seconds':(end-selected_times[-1]).total_seconds(),
 'disk_guard_all_ok':all(r['disk']['ok'] for r in selected),'samples_data':selected}
atomic(root/'health-continuity.json',continuity)
source=json.loads((root/'index.json').read_text())['source_deployment']
prior=config['prior_gpu_request_usage']['seconds'];gpu_seconds=prior
durations={}
for stage in config['stages']:
 name=stage['request']['job_id'];entry=final['jobs'][name]
 if not isinstance(entry['status'],dict):continue
 s=entry['status'];begin=dt.datetime.fromisoformat(s['started_at'])
 stop=dt.datetime.fromisoformat(s['ended_at']) if s.get('ended_at') else end
 seconds=(stop-begin).total_seconds();durations[name]=seconds
 gpu_seconds+=seconds*len(stage.get('gpu_uuids',[]) or ([stage['gpu_uuid']] if stage.get('gpu_uuid') else []))
issues={}
for stage in config['stages']:
 item=state['stages'][stage['id']]
 if item['status'].startswith('BLOCKED'):
  issues[stage['id']]=item
for name,entry in final['jobs'].items():
 s=entry['status']
 if isinstance(s,dict) and s['status']=='RUNNING':
  if not all(s.get(k) for k in ('worker_alive','supervisor_alive','workgroup_live')):issues[name]='Live process identity inconsistent'
 for log in entry.get('logs',{}).values():
  if any(x in log['tail'] for x in ('CUDA out of memory','torch.OutOfMemoryError','Traceback (most recent call last)','FloatingPointError:')):
   issues[name]='Runtime error in current formal log'
if not continuity['disk_guard_all_ok'] or continuity['maximum_gap_seconds']>90 or continuity['last_boundary_gap_seconds']>90:
 issues['observation_continuity']='Machine observation or disk boundary failed'
rows=[]
for workflow in ('toys-a-5300','toys-a-5000','sports-a1','sports-a2'):
 stages=[s for s in config['stages'] if s['workflow']==workflow]
 received=[s for s in stages if isinstance(final['jobs'][s['request']['job_id']]['status'],dict)]
 live=[s for s in received if final['jobs'][s['request']['job_id']]['status']['status'] in ('ACCEPTED','RUNNING')]
 chosen=(live or received or [None])[-1]
 pending=[{'stage':s['id'],'status':state['stages'][s['id']]['status'],'planned_job_id':s['request']['job_id']}
  for s in stages if state['stages'][s['id']]['status']!='SUCCEEDED']
 rows.append({'workflow':workflow,'actual_job_id':chosen['request']['job_id'] if chosen else None,
  'actual_stage':chosen['id'] if chosen else None,'actual_status':final['jobs'][chosen['request']['job_id']]['status']['status'] if chosen else 'NOT_SUBMITTED',
  'actual_worker_wall_seconds':durations.get(chosen['request']['job_id'],0) if chosen else 0,
  'remaining_stages':pending,'completed_stages':[s['id'] for s in stages if state['stages'][s['id']]['status']=='SUCCEEDED']})
a1=final['jobs']['ga-20261017-sportsa1-train-v1'];initial=health['T0']['jobs']['ga-20261017-sportsa1-train-v1']
assert initial['details']['progress.json']['data']['global_step']>0
assert health['T0']['jobs']['ga-20261017-sportsa2-encode-v1']['details']['progress.json']['data']['processed_items_rank0']>0
progress=a1['details'].get('progress.json',{}).get('data',{})
assert progress.get('global_step',0)>initial['details']['progress.json']['data']['global_step']
status='PARTIAL_LAUNCHED' if issues else 'LAUNCH_HANDOFF_READY'
report={'status':status,'observation_start_cst':health['T0']['observed_at_cst'],'observation_end_cst':final['observed_at_cst'],
 'continuous_observation_seconds':(end-start).total_seconds(),'continuous_observation_pass':not issues,
 'source_commit':source['manifest']['source_commit'],'snapshot':source['root'],
 'published_before_launch_commit':'98579e415dc48bfa3a6c5dc50effe70cf25670fd',
 'workflows':rows,'issues':issues,'resources':{'gpu_request_seconds_used':gpu_seconds,
  'gpu_hours_used':gpu_seconds/3600,'gpu_hours_remaining':(143100-gpu_seconds)/3600,
  'hard_gpu_seconds':143100,'before_formal_gpu_seconds':prior,'current_gpu':final['gpu'],
  'disk':final['disk'],'new_logical_bytes':state['disk']['files_logical_bytes'],
  'cpu_available':final['memory_available'],'loadavg':final['loadavg']},
 'machine_continuity':{k:v for k,v in continuity.items() if k!='samples_data'},
 'queue_id':'ga-20261017-queue-v1','queue_status':state['status'],'correction_rounds_used':2,'correction_rounds_maximum':3,
 'new_installations':[],'scientific_result_review':'NOT_PERFORMED; launch/health only',
 'AI_background_monitoring_after_handoff':False,'GPT_UPLOAD_REQUIRED':'NO'}
atomic(root/'handoff-report.json',report)
print(json.dumps(report,indent=2))
