#!/usr/bin/env python3
"""Isolated CPU-only ownership and signal probe; never submits a real job."""
import hashlib,importlib.util,json,os,pathlib,pwd,signal,socket,subprocess,sys,tempfile,time
import derive_formal_runner as derive

def main():
 original=pathlib.Path(__file__).resolve().parents[1]/'letter/scripts/formal_runner.py'
 with tempfile.TemporaryDirectory(prefix='beauty-runner-cpu-') as temp:
  root=pathlib.Path(temp); candidate=root/'candidate.py';candidate.write_bytes(derive.derive(original))
  jobs=root/'jobs';jobs.mkdir(); side=pathlib.Path(str(candidate)+'.deployment.json')
  side.write_text(json.dumps({'job_root':str(jobs),'hostname':socket.gethostname(),'uid':os.getuid(),'user':pwd.getpwuid(os.getuid()).pw_name,'namespace':'SYNTHETIC_CPU_ONLY'}))
  spec=importlib.util.spec_from_file_location('beauty_runner_cpu_probe',candidate);runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
  jobid='letter-a0-beauty-cpu-runner-check-isolated';jobdir=jobs/jobid;jobdir.mkdir()
  req={'schema':1,'host':socket.gethostname(),'uid':os.getuid(),'user':pwd.getpwuid(os.getuid()).pw_name,'namespace':'SYNTHETIC_CPU_ONLY','deployment_sha256':hashlib.sha256(side.read_bytes()).hexdigest(),'capture_version':'SYNTHETIC_CPU_ONLY','job_id':jobid,'argv':[sys.executable,'-c','pass'],'cwd':str(root),'timeout_seconds':345600,'inputs':[],'env':{}}
  runner.verify(req)
  req['timeout_seconds']=345601
  try:runner.verify(req)
  except AssertionError:pass
  else:raise AssertionError('overlong request accepted')
  env=os.environ.copy();env.update(GRID_JOB_ID=jobid,GRID_JOB_DIR=str(jobdir))
  sleeper=[sys.executable,'-c','import time;time.sleep(120)']
  leader=worker=control=None
  try:
   leader=subprocess.Popen(sleeper,env=env,start_new_session=True);worker=subprocess.Popen(sleeper,env=env,start_new_session=True)
   control=subprocess.Popen(sleeper,start_new_session=True)
   group=runner.workgroup(leader,req,jobdir);deadline=time.monotonic()+5
   while True:
    obs=runner.observe_group(group);runner.remember(group,obs)
    found={x['pid'] for x in obs['owned_live']}
    if {leader.pid,worker.pid}<=found:break
    if time.monotonic()>deadline:raise RuntimeError('marked isolated worker not proven: '+repr(obs))
    time.sleep(.05)
   assert control.pid not in found
   marked=next(x for x in obs['owned_live'] if x['pid']==worker.pid)
   assert marked['proof']=='A0_MARKED_WORKER_SESSION'
   assert runner.signal_member(group,marked,signal.SIGTERM)=='SENT'
   worker.wait(timeout=5);assert control.poll() is None and leader.poll() is None
   leader_item=next(x for x in obs['owned_live'] if x['pid']==leader.pid)
   assert runner.signal_member(group,leader_item,signal.SIGTERM)=='SENT';leader.wait(timeout=5)
   assert control.poll() is None
   final=runner.observe_group(group)
   assert final['verified_empty'],repr(final)
   # Full runner lifecycle on a distinct synthetic-only job.  It starts only
   # two sleeping Python processes; no model, CUDA or formal job is involved.
   jobid2='letter-a0-beauty-cpu-runner-check-lifecycle';jobdir2=jobs/jobid2
   child="import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'],start_new_session=True); time.sleep(120)"
   request={**req,'job_id':jobid2,'timeout_seconds':2,'argv':[sys.executable,'-c',child]}
   request_path=root/'synthetic_request.json';request_path.write_text(json.dumps(request))
   cmd=[sys.executable,'-I','-S','-B',str(candidate)]
   accepted=json.loads(subprocess.check_output(cmd+['submit',str(jobs),str(request_path)],text=True))
   assert accepted['submission']=='ACCEPTED'
   deadline=time.monotonic()+20
   while True:
    status=json.loads(subprocess.check_output(cmd+['query',str(jobdir2)],text=True))
    if status['status'] not in ('ACCEPTED','RUNNING'):break
    if time.monotonic()>deadline:raise TimeoutError('isolated synthetic runner lifecycle')
    time.sleep(.1)
   assert status['status']=='TIMED_OUT' and status['cleanup_verified'] and not status['workgroup_live'],status
   assert any(x['member']['pid']!=status['worker']['pid'] for x in status['cleanup']['events']), 'escaped worker not cleaned'
   duplicate=json.loads(subprocess.check_output(cmd+['submit',str(jobs),str(request_path)],text=True))
   assert duplicate['submission']=='EXISTING_NO_RESTART'
   assert control.poll() is None
   print(json.dumps({'status':'PASS_CPU_ISOLATED','candidate_sha256':hashlib.sha256(candidate.read_bytes()).hexdigest(),'timeout_345600_accept':True,'timeout_345601_rejected':True,'marked_session_owned_and_pidfd_signalled':True,'synthetic_submit_timeout_cleanup_and_no_duplicate':True,'unmarked_control_untouched':True,'cleanup_verified':True,'formal_submit':False}))
  finally:
   for p in (worker,leader,control):
    if p is not None and p.poll() is None:
     p.terminate()
     try:p.wait(timeout=5)
     except subprocess.TimeoutExpired:p.kill();p.wait(timeout=5)
if __name__=='__main__':main()
