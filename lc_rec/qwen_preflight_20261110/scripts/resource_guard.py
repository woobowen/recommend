"""One GPU job, existing runner, task-wide 360/per-model120 device-minute limits."""
import argparse, datetime, fcntl, json, os, shutil, subprocess, sys, time
from pathlib import Path
from common import write

def main():
 p=argparse.ArgumentParser();p.add_argument('--key',required=True);p.add_argument('--job-id',required=True);p.add_argument('--output',required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
 assert a.command[0]=='--' and os.environ['CUDA_VISIBLE_DEVICES']=='0,1' and os.environ['GRID_JOB_ID']==a.job_id
 # Keep every potentially new compiler/runtime cache in the accounted task tree.
 cache_root=Path('/data/bwwu/lc-rec/qwen-preflight-20261110').resolve()
 for variable in ['TMPDIR','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','TORCH_EXTENSIONS_DIR','XDG_CACHE_HOME','HF_HOME','CUDA_CACHE_PATH']:
  if variable in os.environ:
   folder=Path(os.environ[variable]);assert folder.resolve().is_relative_to(cache_root);folder.mkdir(parents=True,exist_ok=True)
 import psutil
 root=Path('/home/bwwu/lc-rec/qwen-preflight-20261110');root.mkdir(parents=True,exist_ok=True);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 ledger_path=root/'gpu-budget.json'
 with (root/'gpu-budget.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {'limits':{'task_device_seconds':21600,'model_device_seconds':7200,'job_wall_seconds':2700},'runs':[]}
  assert all(x.get('end_utc') for x in ledger['runs']),'Original job unresolved; never reset budget'
  assert not any(x['job_id']==a.job_id for x in ledger['runs'])
  used=sum(x['charged_gpu_device_seconds'] for x in ledger['runs']);model_used=sum(x['charged_gpu_device_seconds'] for x in ledger['runs'] if x['key']==a.key)
  # Five-second polling plus bounded descendant termination must fit the hard
  # allowance. Leave thirty wall seconds before either cumulative ceiling.
  limit=min(2660,(21600-used)/2-30,(7200-model_used)/2-30);assert limit>0
  gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],text=True);assert not gpu.strip(),'GPU already occupied; no competing performance measurement'
  def allocated():
   total=0
   for base in [root,Path('/data/bwwu/lc-rec/qwen-preflight-20261110'),Path('/data/bwwu/envs/qwen-preflight-20261110-qwen3'),Path('/data/bwwu/envs/qwen-preflight-20261110-qwen35')]:
    if not base.exists():continue
    for d,dirs,files in os.walk(base,followlinks=False):
     dirs[:]=[x for x in dirs if not (Path(d)/x).is_symlink()]
     for name in files:
      q=Path(d)/name
      try:
       if not q.is_symlink():total+=q.stat().st_blocks*512
      except FileNotFoundError:pass
   # This task's job outputs are outside the task root.
   jobs=Path(os.environ['GRID_JOB_DIR']).parent
   for job in jobs.glob('qwen-20261110-*'):
    for d,dirs,files in os.walk(job,followlinks=False):
     dirs[:]=[x for x in dirs if not (Path(d)/x).is_symlink()]
     for name in files:
      q=Path(d)/name
      try:
       if not q.is_symlink():total+=q.stat().st_blocks*512
      except FileNotFoundError:pass
   # Accepted transport keeps task archives/requests in shared support paths.
   # Count only this task's known prefixes; do not scan unrelated artifacts.
   support=jobs.parent
   for folder,pattern in [('transfers','qwen-preflight-*'),('requests','qwen-20261110-*')]:
    for q in (support/folder).glob(pattern):
     try:
      if q.is_file() and not q.is_symlink():total+=q.stat().st_blocks*512
     except FileNotFoundError:pass
   return total
  def mem():return psutil.virtual_memory().available
  assert mem()>=40*1024**3
  started=time.monotonic();entry={'job_id':a.job_id,'key':a.key,'argv':a.command[1:],'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'limit_seconds':limit,'charging':'two devices x entire child lifetime, including imports/load/save/error/inference'};ledger['runs'].append(entry);write(ledger_path,ledger)
  child=subprocess.Popen(a.command[1:],stdin=subprocess.DEVNULL);owner=psutil.Process(child.pid);tracked={(owner.pid,owner.create_time()):owner};reason=None;minimum_mem=mem();rss_peak=0;disk_peak=0;count=0
  try:
   with (out/'resource-samples.jsonl').open('a',buffering=1) as stream:
    while True:
     try:
      for q in [owner]+owner.children(recursive=True):tracked.setdefault((q.pid,q.create_time()),q)
     except psutil.Error:pass
     alive=[];rss=0;swap=0
     for ident,q in tracked.items():
      try:
       if q.create_time()!=ident[1] or not q.is_running() or q.status()==psutil.STATUS_ZOMBIE:continue
       alive.append(q);rss+=q.memory_info().rss
       for line in Path('/proc',str(q.pid),'status').read_text().splitlines():
        if line.startswith('VmSwap:'):swap+=int(line.split()[1])*1024
      except (psutil.Error,OSError):pass
     elapsed=time.monotonic()-started;available=mem();minimum_mem=min(minimum_mem,available);rss_peak=max(rss_peak,rss)
     row={'elapsed_seconds':elapsed,'mem_available_bytes':available,'task_tree_rss_bytes':rss,'task_swap_bytes':swap,'pids':[q.pid for q in alive]}
     if count%3==0:
      row['nvidia_smi']=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
      size=allocated();disk_peak=max(disk_peak,size);row['new_disk_allocated_bytes']=size
      if size>60*1024**3:reason='TASK_60GIB_DISK_BOUND'
      for location in ['/data/bwwu','/home/bwwu']:
       d=shutil.disk_usage(location)
       if d.free<max(30*1024**3,d.total//10):reason='DISK_RESERVE_BOUND'
     stream.write(json.dumps(row)+'\n');count+=1
     if available<40*1024**3:reason='CPU_40GIB_RESERVE_BOUND'
     if swap>64*1024**2:reason='TASK_SWAP_BOUND'
     if elapsed>=limit:reason='GPU_TIME_BOUND'
     if reason or child.poll() is not None:break
     time.sleep(5)
  finally:
   def still_owned(ident,q):
    try:return q.create_time()==ident[1] and q.is_running() and q.status()!=psutil.STATUS_ZOMBIE and q.uids().real==os.getuid()
    except psutil.Error:return False
   remaining=[(ident,q) for ident,q in tracked.items() if still_owned(ident,q)]
   if remaining:
    for ident,q in reversed(list(tracked.items())):
     try:
      if q.create_time()==ident[1] and q.uids().real==os.getuid():q.terminate()
     except psutil.Error:pass
    psutil.wait_procs(list(tracked.values()),timeout=3)
    for ident,q in reversed(list(tracked.items())):
     try:
      if q.create_time()==ident[1] and q.is_running() and q.uids().real==os.getuid():q.kill()
     except psutil.Error:pass
   code=child.wait(timeout=10);live=[q.pid for ident,q in tracked.items() if still_owned(ident,q)]
   entry.update(end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started,child_exit_code=code,stop_reason=reason,mem_available_min_bytes=minimum_mem,task_tree_rss_sample_peak_bytes=rss_peak,new_disk_allocated_peak_bytes=disk_peak,tracked_live_pids_after_cleanup=live,cleanup_verified=not live)
   entry['charged_gpu_device_seconds']=2*entry['elapsed_seconds'];write(ledger_path,ledger);write(out/'resource-summary.json',entry)
  sys.exit(125 if live else 124 if reason else code if code>=0 else 128-code)
if __name__=='__main__':main()
