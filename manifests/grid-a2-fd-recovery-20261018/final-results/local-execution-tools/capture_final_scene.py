#!/usr/bin/env python3
"""Final scoped read-only capture, refusing to declare a live queue terminal."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import sys

BASE=Path(__file__).resolve().parent
sys.path.insert(0,'/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
from transport import Transport, atomic

REMOTE=r'''
import collections,datetime,fcntl,hashlib,importlib.util,json,pathlib,shutil
config_path=pathlib.Path(CONFIG)
c=json.loads(config_path.read_text())
spec=importlib.util.spec_from_file_location('final_read_only_queue',pathlib.Path(c['snapshot']['root'])/'experiments/launch_20261017/queue.py')
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
root=pathlib.Path(c['runner']['job_root'])
def read(p):return json.loads(pathlib.Path(p).read_text())
def identity(p):
 p=pathlib.Path(p)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
frozen=[]
for expected in c['recovery']['frozen_evidence']:
 actual=identity(expected['path'])
 assert all(actual[k]==expected[k] for k in ['bytes','sha256']),actual
 frozen.append(actual)
original_config=next(r['path'] for r in frozen if '/compiled/b6203414ee58a400/queue-config-' in r['path'])
original=read(original_config)
state=read(pathlib.Path(c['state_dir'])/'queue-status.json')
terminal={'SUCCEEDED','FAILED','INTERRUPTED','TIMED_OUT','WORKERS_TERMINATED'}
assert state['status'] not in {'RUNNING','ACCEPTED','QUEUED','SUBMITTING'},state['status']
names={s['request']['job_id'] for s in original['stages']+c['stages']}
names.update(c['recovery']['terminal_controller_ids'])
names.update(c['prior_gpu_request_usage']['jobs'])
names.update([c['queue_id'],'ga-20261018-a2-fd-cpu-equivalence-v1',
 'ga-20261017-toys5300-metadata-v1','ga-20261017-toys5000-metadata-v1','ga-20261017-sportsa1-metadata-v1'])
jobs={}
for name in sorted(names):
 job=root/name
 if not (job/'accepted.json').is_file():
  assert not job.exists(),'unaccepted directory exists: '+name
  jobs[name]=dict(status='NOT_SUBMITTED',directory_exists=False)
  continue
 # runner.query can persist a control-loss receipt for a live-looking status.
 # Refuse that branch before invoking it; this capture does not repair state.
 raw=read(job/'status.json')
 assert raw['status'] in terminal,(name,raw['status'])
 request=read(job/'request.json');accepted=read(job/'accepted.json')
 request_sha=hashlib.sha256(q.canonical(request)).hexdigest()
 assert accepted['request_sha256']==request_sha,(name,'accepted request identity')
 value=q.runner_call(c,'query',job)
 assert value['status'] in terminal,value
 assert value['request_sha256']==request_sha,(name,'query request identity')
 assert value['cleanup_verified'] and not value['workgroup_live'] and not value['supervisor_alive'],value
 jobs[name]=dict(query=value,request=request,accepted=accepted,
                 raw_status=raw,cleanup=read(job/'cleanup.json'))
assert jobs[c['queue_id']]['query']['status'] in terminal
source=[]
for expected in EXPECTED_SOURCE['files']:
 actual=identity(pathlib.Path(c['snapshot']['root'])/expected['path'])
 assert all(actual[k]==expected[k] for k in ['bytes','sha256']),actual
 source.append(actual)
locks=[]
for gpu in c['approved_gpu_uuids']:
 path=pathlib.Path(c['gpu_locks_dir'])/(gpu+'.lock')
 assert path.is_file() and not path.is_symlink()
 with path.open('r') as stream:
  try:
   fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
   available=True
   fcntl.flock(stream,fcntl.LOCK_UN)
  except BlockingIOError:available=False
 locks.append(dict(path=str(path),available=available,nonblocking_existing_file_probe=True))
assert all(v['available'] for v in locks),locks
new_usage=0.;groups=collections.defaultdict(float);details=[]
for stage in c['stages']:
 name=stage['request']['job_id'];doc=jobs[name]
 if 'query' not in doc:continue
 raw=doc['raw_status'];request=doc['request']
 start=datetime.datetime.fromisoformat(raw['started_at']);end=datetime.datetime.fromisoformat(raw['ended_at'])
 wall=(end-start).total_seconds();uuids=request.get('queue_stage',{}).get('gpu_uuids',[]);devices=len(uuids)
 assert len(uuids)==len(set(uuids)),(name,'duplicate GPU UUID')
 assert devices==len(stage.get('gpu_uuids',[])),(name,uuids,stage.get('gpu_uuids'))
 assert all(gpu in c['approved_gpu_uuids'] for gpu in uuids),(name,uuids)
 allocated=q.allocation_stage(stage,uuids)
 assert q.gpus(allocated)==uuids
 meta=request['queue_stage']
 for key,wanted in dict(id=stage['id'],workflow=stage['workflow'],
  config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),command=stage['request']['argv'],
  budget_group=stage['budget_group'],source_commit=c['snapshot']['commit']).items():
  assert meta[key]==wanted,(name,key,meta[key],wanted)
 assert request['env']['CUDA_VISIBLE_DEVICES']==','.join(uuids),(name,'CUDA allocation')
 assert request['cwd']==c['snapshot']['root'],(name,'cwd')
 assert request['capture_version']==stage['request']['capture_version'],(name,'capture identity')
 assert request['timeout_seconds']==stage['request']['timeout_seconds'],(name,'wall cap')
 assert end<=datetime.datetime.fromisoformat(c['recovery']['original_absolute_queue_deadline']),(name,'absolute deadline')
 used=wall*devices;new_usage+=used
 if devices:groups[stage['budget_group']]+=used
 assert wall<=request['timeout_seconds']+1,(name,wall,request['timeout_seconds'])
 details.append(dict(job_id=name,status=raw['status'],exit_code=raw['exit_code'],
  worker_wall_seconds=wall,requested_gpus=devices,gpu_uuids=uuids,gpu_requested_seconds=used,
  budget_group=stage['budget_group'],cleanup_verified=True))
for group,used in groups.items():assert used<=c['budget_groups'][group]+1e-5,(group,used)
def old_requested_usage(names):
 total=0.;rows=[]
 for name in names:
  doc=jobs[name]
  if 'query' not in doc:continue
  raw=doc['raw_status'];request=doc['request'];uuids=request['queue_stage']['gpu_uuids']
  assert len(uuids)==len(set(uuids)) and set(uuids)<=set(c['approved_gpu_uuids']),(name,uuids)
  assert request['env']['CUDA_VISIBLE_DEVICES']==','.join(uuids),(name,'historical CUDA allocation')
  wall=(datetime.datetime.fromisoformat(raw['ended_at'])-datetime.datetime.fromisoformat(raw['started_at'])).total_seconds()
  value=wall*len(uuids);total+=value
  rows.append(dict(job_id=name,worker_wall_seconds=wall,gpu_uuids=uuids,gpu_requested_seconds=value))
 return total,rows
old_names=[stage['request']['job_id'] for stage in original['stages']]
prior_names=c['prior_gpu_request_usage']['jobs']
assert not set(old_names)&set(prior_names)
historical,historical_rows=old_requested_usage(old_names)
prior,prior_rows=old_requested_usage(prior_names)
assert abs(historical-c['recovery']['historical_stage_gpu_seconds'])<1e-5,(historical,'historical baseline')
assert abs(prior-c['recovery']['prior_gpu_request_seconds'])<1e-5,(prior,'prior baseline')
total=historical+prior+new_usage
assert total<=143100+1e-5,total
observed=state['accounting']['used_gpu_request_seconds']
assert abs(observed-new_usage)<1e-4,(observed,new_usage)
out=dict(status='TERMINAL_SCENE_AND_RESOURCE_IDENTITY_CAPTURED',scientific_final_pass=False,
 captured_utc=q.utc(),config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
 source_commit=c['snapshot']['commit'],source_snapshot=c['snapshot']['root'],
 source_snapshot_sha256=EXPECTED_SOURCE['snapshot_sha256'],source_members_rehashed=source,
 frozen_evidence_unchanged=frozen,queue_state=state,jobs=jobs,locks=locks,gpus=q.gpu_snapshot(),
 disk={p:shutil.disk_usage(p)._asdict() for p in ['/home/bwwu','/data/bwwu']},
 resources=dict(stage_details=details,new_gpu_requested_seconds=new_usage,new_groups=dict(groups),
 historical_stage_gpu_seconds=historical,prior_gpu_requested_seconds=prior,
 historical_stage_recalculation=historical_rows,prior_recalculation=prior_rows,
 total_gpu_requested_seconds=total,total_budget_gpu_seconds=143100,remaining_gpu_seconds=143100-total,
 original_absolute_deadline=c['recovery']['original_absolute_queue_deadline']),
 activity=dict(known_accepted_jobs=sum('query' in v for v in jobs.values()),
 known_not_submitted=sum('query' not in v for v in jobs.values()),
 active_project_workers=0,active_known_supervisors=0),
 remote_file_writes=0,model_loads=0,scope='Only explicitly named task jobs, frozen small metadata/source files and existing task locks; no unrelated account scan.')
print(json.dumps(out))
'''

def main():
    deployment=json.loads((BASE/'queue-deployment.json').read_text())
    source=json.loads((BASE/'source-deployment.json').read_text())['manifest']
    code=REMOTE.replace('CONFIG',repr(deployment['remote_config'])).replace('EXPECTED_SOURCE',repr(source))
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    script=BASE/'final-scene'/('remote-'+stamp+'.py')
    script.parent.mkdir(exist_ok=True)
    script.write_text(code)
    value=json.loads(Transport('new-server').python(code,timeout=60))
    assert value['config_sha256']==deployment['config_sha256']
    value['remote_script_sha256']=hashlib.sha256(code.encode()).hexdigest()
    path=BASE/'final-scene'/('scene-'+stamp+'.json')
    atomic(path,value)
    print(json.dumps(dict(path=str(path),status=value['status'],activity=value['activity'],
                         resources=value['resources'],scientific_final_pass=False),ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
