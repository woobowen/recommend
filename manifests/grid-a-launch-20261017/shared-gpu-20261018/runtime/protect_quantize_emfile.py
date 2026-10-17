"""One-shot emergency protection of the exact failing original A2 quantizer only."""
from pathlib import Path
import hashlib,importlib.util,json,os,signal,time
BASE=Path('/home/bwwu/grid-audits/sid-extensions-p0/grid-a-launch-20261017')
SOURCE=BASE/'snapshots/cdfd283823bd1928/experiments/launch_20261017/queue.py'
CONFIG=BASE/'shared-20261018/compiled/queue-config-d5528b2c185395bd.json'
OUTPUT=BASE/'shared-20261018/quantize-emfile-protection.json'
assert not OUTPUT.exists(),'inspect existing protection receipt; never repeat blindly'
assert hashlib.sha256(CONFIG.read_bytes()).hexdigest()=='d5528b2c185395bd5c77803189f97a756891dd2f5db4d798fbb5f2e1deb04df3'
spec=importlib.util.spec_from_file_location('q',SOURCE);q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
config=q.read(CONFIG);job=Path(config['runner']['job_root'])/'ga-20261017-sportsa2-quantize-v1'
request=q.read(job/'request.json');before=q.runner_call(config,'query',job)
assert before['request_sha256']==hashlib.sha256(q.canonical(request)).hexdigest()
assert request['queue_stage']['id']=='sportsa2-quantize'
log=job/'work.log';raw=log.read_bytes();progress=job/'output/progress.json'
assert b'Too many open files' in raw and b'[Errno 24]' in raw
receipt={'at':q.utc(),'reason':'Repeated DataLoader EMFILE; no valid quantizer progress; protect original stage without retry',
 'before':before,'request':request,'log_sha256_before':hashlib.sha256(raw).hexdigest(),'log_bytes_before':len(raw),
 'log_tail_before':raw[-32768:].decode('utf-8','replace'),'progress_before':q.read(progress) if progress.exists() else None,
 'signals':[],'scientific_configuration_changed':False,'new_research_jobs':0,'other_jobs_signaled':0}
assert not progress.exists(),'progress now exists; reassess actual error, do not stop blindly'
def identity(pid):
 p=Path('/proc')/str(pid)
 try:
  f=(p/'stat').read_text().rsplit(')',1)[1].split()
  if f[0]=='Z':return None
  return {'pid':pid,'uid':p.stat().st_uid,'start_ticks':f[19],
   'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
   'argv':(p/'cmdline').read_bytes().decode().rstrip('\0').split('\0')}
 except (FileNotFoundError,ProcessLookupError):return None
receipt['owned_process_limits']=[]
for member in before.get('workgroup_observation',{}).get('owned_live',[]):
 p=Path('/proc')/str(member['pid'])
 try:
  receipt['owned_process_limits'].append({'pid':member['pid'],'fd_count':len(list((p/'fd').iterdir())),
   'limits':(p/'limits').read_text(),'argv':(p/'cmdline').read_bytes().decode().rstrip('\0').split('\0')})
 except (FileNotFoundError,ProcessLookupError):pass
q.atomic(OUTPUT.with_name('quantize-emfile-before-protection.json'),receipt)
if before['status'] in q.ACTIVE:
 supervisor=before['supervisor'];assert supervisor['uid']==os.getuid() and identity(supervisor['pid'])==supervisor
 assert supervisor['argv']==[config['runner']['python'],'-I','-S','-B',config['runner']['script'],'supervise',str(job)]
 fd=os.pidfd_open(supervisor['pid'])
 try:
  assert identity(supervisor['pid'])==supervisor
  signal.pidfd_send_signal(fd,signal.SIGTERM,None,0)
  receipt['signals'].append({'target':supervisor,'signal':'SIGTERM','at':q.utc(),'mechanism':'pidfd; accepted runner supervisor performs owned-group cleanup'})
 finally:os.close(fd)
deadline=time.monotonic()+45
while True:
 after=q.runner_call(config,'query',job)
 if after['status'] not in q.ACTIVE and after.get('cleanup_verified') and not any(after.get(k) for k in ['worker_alive','supervisor_alive','workgroup_live','worker_may_be_live']):break
 assert time.monotonic()<deadline,'cleanup not yet proven; preserve evidence and query same original job'
 time.sleep(1)
receipt.update(after=after,finished_at=q.utc(),status='A2_QUANTIZER_PROTECTED_NO_RETRY',new_queue_not_signaled=True)
q.atomic(OUTPUT,receipt);print(json.dumps({'status':receipt['status'],'after':after,'signals':receipt['signals']}))
