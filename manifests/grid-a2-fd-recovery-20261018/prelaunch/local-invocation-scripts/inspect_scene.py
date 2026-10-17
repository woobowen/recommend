"""Bounded read-only scene check for the authorized A2 FD recovery."""
import json
import sys
from pathlib import Path
sys.path.insert(0, '/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
import gridctl
from transport import Transport, atomic

BASE = Path(__file__).resolve().parent
t = Transport('new-server')
ids = ['ga-20261017-queue-v1', 'ga-20261018-shared-queue-v1',
       'ga-20261017-sportsa2-encode-v1', 'ga-20261017-sportsa2-quantize-v1',
       'ga-20261017-sportsa1-train-v1', 'ga-20261017-sportsa1-stats-v1',
       'ga-20261017-toys5300-stats-v1', 'ga-20261017-toys5000-stats-v1']
statuses = {job: gridctl.query(t, job) for job in ids}
code = r'''
import pathlib, json, importlib.util, shutil, subprocess, os, fcntl, hashlib, datetime
B=pathlib.Path('/home/bwwu/grid-audits/sid-extensions-p0/grid-a-launch-20261017')
S=B/'snapshots/cdfd283823bd1928'
spec=importlib.util.spec_from_file_location('scene_queue',S/'experiments/launch_20261017/queue.py')
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
p=B/'shared-20261018/compiled/queue-config-d5528b2c185395bd.json'
c=q.read(p);old=q.read(c['continuation']['old_config_path']);J=pathlib.Path(c['runner']['job_root'])
states={};locks={}
for v in [old,c]:
 d=pathlib.Path(v['state_dir']);states[v['queue_id']]=q.read(d/'queue-status.json')
 with (d/'supervisor.lock').open('r+') as f:
  try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks[str(d/'supervisor.lock')]='AVAILABLE';fcntl.flock(f,fcntl.LOCK_UN)
  except BlockingIOError:locks[str(d/'supervisor.lock')]='BUSY'
for p in sorted(pathlib.Path(c['gpu_locks_dir']).glob('*')):
 if p.is_file():
  with p.open('r+') as f:
   try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks[str(p)]='AVAILABLE';fcntl.flock(f,fcntl.LOCK_UN)
   except BlockingIOError:locks[str(p)]='BUSY'
gpu=q.gpu_snapshot()
processes=[]
for d in pathlib.Path('/proc').iterdir():
 if not d.name.isdigit():continue
 try:
  if d.stat().st_uid!=os.getuid():continue
  args=(d/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if '/home/bwwu/grid-audits/sid-extensions-p0/' in args and d.name!=str(os.getpid()):
   processes.append(dict(pid=int(d.name),argv=args,stat=(d/'stat').read_text()))
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
e=J/'ga-20261017-sportsa2-encode-v1/output/item_embeddings.pt'
r=q.read(e.parent/'stage_result.json');eh=hashlib.sha256(e.read_bytes()).hexdigest()
assert eh==r['sha256']
missing=[]
for s in c['stages']:
 if s['workflow'] not in ['Toys-A-5300','Toys-A-5000','Sports-A1']:continue
 d=J/s['request']['job_id']
 if not (d/'status.json').is_file() or q.read(d/'status.json')['status']!='SUCCEEDED':missing.append(str(d))
newids=['ga-20261018-a2-fd-queue-v1','ga-20261018-sportsa2-quantize-fd-recovery-v1']
print(json.dumps(dict(at_utc=q.utc(),at_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
 configs={str(p):q.sha(p),c['continuation']['old_config_path']:q.sha(pathlib.Path(c['continuation']['old_config_path']))},
 config=c,old_config=old,states=states,locks=locks,gpu=gpu,
 gpu_owners={str(p['pid']):subprocess.run(['ps','-p',str(p['pid']),'-o','pid=,uid=,user=,etime=,stat='],capture_output=True,text=True).stdout for g in gpu.values() for p in g['processes']},
 project_processes=processes,disk={p:shutil.disk_usage(p)._asdict() for p in ['/home/bwwu','/data/bwwu']},
 embedding=dict(path=str(e),bytes=e.stat().st_size,sha256=eh,original_receipt=r),
 old_quantizer_inventory=[dict(path=str(p),bytes=p.stat().st_size) for p in sorted((J/'ga-20261017-sportsa2-quantize-v1/output').rglob('*')) if p.is_file()],
 old_completed_jobs_not_succeeded=missing,new_id_conflicts=[x for x in newids if (J/x).exists()],
 accounting=q.accounting(c,states[c['queue_id']]))))
'''
scene = json.loads(t.python(code, timeout=60))
scene['runner_statuses'] = statuses
atomic(BASE/'initial-scene.json', scene)
print(json.dumps({k:scene[k] for k in ('at_cst','locks','gpu','gpu_owners','project_processes','disk','old_completed_jobs_not_succeeded','new_id_conflicts','accounting')},ensure_ascii=False,indent=2))
print(json.dumps({'record':str(BASE/'initial-scene.json'),'statuses':{k:{n:v.get(n) for n in ['status','exit_code','cleanup_verified','worker_alive','supervisor_alive','workgroup_live']} for k,v in statuses.items()},'embedding':scene['embedding']},indent=2))
