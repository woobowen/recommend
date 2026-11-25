"""Frozen single-GPU GRID baseline: validated reuse -> bounded smoke -> fresh train/test."""
import argparse,fcntl,hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path('/home/jcy/GRID_wmh_Qwen_TIGER_20261009')
PY=ROOT/'environments/grid/bin/python'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def state(run,phase,**kw):write(run/'status.json',dict(phase=phase,time=time.strftime('%Y-%m-%d %H:%M:%S %z'),supervisor_pid=os.getpid(),**kw))

def worker(run):
 locks=[]
 try:
  for p in [ROOT/'stage_gpu_locks/gpu_0.lock',ROOT/'codex/letter_lc_rec_smoke_20261121/gpu0.lock',run/'job.lock']:
   f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
  gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
  procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
  row=next(r for r in gpu.splitlines() if r.split(',')[0].strip()=='0');_,uuid,used=map(str.strip,row.split(','))
  assert int(used)<512 and uuid not in procs,'GPU 0 occupied; no existing work will be stopped'
  write(run/'gpu_start.json',dict(gpus=gpu,processes=procs,physical_gpu=0))
  frozen=run/'source';env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4',
    'TF_CPP_MIN_LOG_LEVEL':'3','TF_NUM_INTRAOP_THREADS':'1','TF_NUM_INTEROP_THREADS':'1','HYDRA_FULL_ERROR':'1','PYTHONUNBUFFERED':'1'}
  for key in ['PROJECT_ROOT','METADATA_DIR','LOCAL_RANK','RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT']:env.pop(key,None)
  def execute(name,argv,cwd,timeout):
   folder=run/name;folder.mkdir(exist_ok=True);write(folder/'command.json',dict(argv=list(map(str,argv)),cwd=str(cwd),CUDA_VISIBLE_DEVICES='0',timeout_seconds=timeout))
   state(run,name)
   with (folder/'console.log').open('ab',buffering=0) as log:
    p=subprocess.Popen(list(map(str,argv)),cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log)
    state(run,name,child_pid=p.pid)
    try:rc=p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
     p.terminate()
     try:p.wait(timeout=30)
     except subprocess.TimeoutExpired:p.kill();p.wait()
     write(folder/'exit.json',dict(returncode=p.returncode,timed_out=True,time=time.time()))
     raise
   write(folder/'exit.json',dict(returncode=rc,time=time.time()));assert rc==0,f'{name} failed exit={rc}'
  execute('prepare',[PY,run/'runner/prepare_data.py','--run',run],frozen,600)
  import yaml
  cfg=yaml.safe_load((run/'reference/toys_effective.yaml').read_text())
  cfg['paths']['root_dir']=str(frozen);cfg['data_dir']=str(run/'data')
  cfg['semantic_id_path']=str(ROOT/'codex/instrument_semantic_hierarchy_20261115/runs/20261115_104723/upstream/semantic_ids.pt')
  cfg['ckpt_path']=None
  configdir=frozen/'configs';(configdir/'instruments_grid.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
  tracked={str(p.relative_to(run)):sha(p) for base in [frozen,run/'runner',run/'reference'] for p in sorted(base.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
  write(run/'source_manifest.json',tracked)
  import importlib.metadata as metadata
  packages={n:metadata.version(n) for n in ['torch','lightning','transformers','hydra-core']}
  packages['tensorflow']=__import__('tensorflow').__version__
  write(run/'environment.json',dict(python=sys.version,packages=packages,gpu=0))
  write(run/'effective_changes.json',dict(reference='yjc-grid@6200d5875e264b5e6ec9841cf2cdb3b4a12d71a4 Toys final effective config',
   changes=['paths.root_dir','data_dir','semantic_id_path'],fresh_training=True,training_parameters_unchanged=True))
  base=[PY,frozen/'_运行/grid_stage.py','--config-name=instruments_grid']
  execute('smoke',base+[f'hydra.run.dir={run}/smoke','trainer.max_steps=2','trainer.val_check_interval=16',
    '+trainer.limit_val_batches=2','+trainer.limit_test_batches=2'],frozen,1800)
  smoke=json.loads((run/'smoke/stage_result.json').read_text());assert smoke['global_step']==2 and smoke['test_metrics']
  assert all(__import__('math').isfinite(float(v)) for m in smoke['test_metrics'] for v in m.values())
  write(run/'preflight.json',dict(passed=True,optimizer_updates=2,validation_batches=2,test_batches=2,fresh_formal_checkpoint=None))
  assert all(sha(run/p)==h for p,h in tracked.items()),'Source changed after preflight'
  assert sha(cfg['semantic_id_path'])=='0f12825d1de428e8cba9d9827537be229446f0f076effbf6bd695efa42c83015'
  execute('formal',base+[f'hydra.run.dir={run}/formal'],frozen,14*86400)
  result=json.loads((run/'formal/stage_result.json').read_text());assert result.get('test_metrics') and Path(result['checkpoint']).exists()
  write(run/'result.json',result);state(run,'complete',returncode=0,result=result)
 except BaseException as e:
  state(run,'failed',returncode=1,error=repr(e),traceback=traceback.format_exc());raise
 finally:
  for f in reversed(locks):f.close()

def main():
 p=argparse.ArgumentParser();p.add_argument('--worker',type=Path);args=p.parse_args()
 if args.worker:worker(args.worker);return
 here=Path(__file__).resolve().parent
 launch_lock=(here/'launch.lock').open('a');fcntl.flock(launch_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (here/'latest_run.json').exists():
  old=json.loads((here/'latest_run.json').read_text());proc=Path('/proc')/str(old['supervisor_pid'])
  if proc.exists() and str(old['run']) in (proc/'cmdline').read_bytes().decode(errors='replace'):
   raise RuntimeError('An existing Instruments GRID supervisor is alive; inspect instead of duplicating')
 run=ROOT/'experiments'/('instruments_grid_qwen_'+time.strftime('%Y%m%d_%H%M%S'));run.mkdir()
 source=run/'source';source.mkdir();runner=run/'runner';runner.mkdir();(run/'reference').mkdir()
 for part in ['src','configs','_运行']:shutil.copytree(ROOT/part,source/part,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 shutil.copy2(ROOT/'.project-root',source/'.project-root')
 for name in ['prepare_data.py','run_experiment.py']:shutil.copy2(here/name,runner/name)
 shutil.copy2(here/'toys_effective.yaml',run/'reference/toys_effective.yaml')
 files={str(p.relative_to(run)):sha(p) for base in [source,runner,run/'reference'] for p in sorted(base.rglob('*')) if p.is_file()}
 write(run/'launch.json',dict(user_authorized=True,argv=sys.argv,python=str(PY),run=str(run),files=files,
  baseline_commit='6200d5875e264b5e6ec9841cf2cdb3b4a12d71a4',gpu=0,seed=42,monitor_interval_seconds=900))
 with (run/'supervisor.log').open('ab',buffering=0) as log:
  child=subprocess.Popen([str(PY),'-u',str(runner/'run_experiment.py'),'--worker',str(run)],cwd=source,
   stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
 receipt=dict(run=str(run),supervisor_pid=child.pid,time=time.time());write(here/'latest_run.json',receipt);print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
