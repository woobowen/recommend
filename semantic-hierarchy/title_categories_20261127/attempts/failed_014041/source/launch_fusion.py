"""Detached supervisor with fixed sources, genuine stage returns and finite budget."""
from pathlib import Path
import argparse,fcntl,json,os,shutil,subprocess,sys,time,traceback
from runtime_common import ROOT,read_json,write_json,sha,status
HERE=ROOT/'codex/title_categories_hierarchy_20261127'
ENCODER=ROOT/'environments/llm2vec_sid_eval/bin/python'
GRID=ROOT/'environments/grid/bin/python'

def worker(run):
 children=[]
 try:
  manifest=read_json(run/'launch.json')
  for name,h in manifest['source_sha256'].items():assert sha(run/'source'/name)==h,name
  def execute(name,argv,env,timeout):
   out=run/name;out.mkdir(exist_ok=True);write_json(out/'command.json',dict(argv=list(map(str,argv)),cwd=str(run/'source'),gpus=env.get('CUDA_VISIBLE_DEVICES'),timeout_seconds=timeout))
   with (out/'console.log').open('ab',buffering=0) as log:
    p=subprocess.Popen(list(map(str,argv)),cwd=run/'source',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log)
    status(run/'status.json',name,child_pid=p.pid)
    try:rc=p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
     p.terminate()
     try:p.wait(timeout=30)
     except subprocess.TimeoutExpired:p.kill();p.wait()
     write_json(out/'exit.json',dict(returncode=p.returncode,timed_out=True));raise
   write_json(out/'exit.json',dict(returncode=rc));assert rc==0,f'{name} exit {rc}'
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
  execute('prepare',[GRID,run/'source/prepare_experiment.py','--run',run],env,900)
  execute('metric_preflight',[ENCODER,run/'source/test_metrics.py'],env,300)
  # Check each actual model file against its existing complete download manifest.
  model_manifest=read_json(ROOT/'codex/llm2vec_sid_hierarchy_20261113/preparation/model_download_verified.json');assert model_manifest['complete']
  selected=[]
  for row in model_manifest['files']:
   if any(row['file'].startswith('models/'+n+'/') for n in ['Meta-Llama-3-8B-Instruct','LLM2Vec-Meta-Llama-3-8B-Instruct-mntp','LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised']):
    p=ROOT/row['file'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'];selected.append(row)
  assert any('model-00001' in r['file'] for r in selected) and any('supervised/adapter_model' in r['file'] for r in selected)
  write_json(run/'model_verified.json',dict(passed=True,files=selected,original_manifest_sha256=sha(ROOT/'codex/llm2vec_sid_hierarchy_20261113/preparation/model_download_verified.json')))
  import importlib.metadata as metadata
  write_json(run/'environment.json',dict(python=sys.version,packages={n:metadata.version(n) for n in ['torch','transformers','peft','llm2vec','numpy','accelerate']},model_sharding='16 layers per GPU; fp16; MNTP merged + supervised adapter',original_encoder_source_sha256=sha(run/'source/original_encoder.py')))
  write_json(run/'effective_config.json',dict(encoding=read_json(run/'encoding_config.json'),analysis=dict(epsilon=1e-6,minimum_group_size=2,permutations=999,bootstrap_repetitions=2000,primary='fusion_complete',supplementary=['fusion_all_available','title_complete'],primary_holm_tests=4,top_paths='proper both edges; rank by level3 distinct item count',sid_depths=[1,2,3]),gpus={'beauty':[0,1],'instruments':[2,3]}))
  for domain,gpus in [('beauty','0,1'),('instruments','2,3')]:
   (run/domain).mkdir(exist_ok=True);argv=[str(ENCODER),'-u',str(run/'source/encode_fields.py'),'--run',str(run),'--dataset',domain]
   denv=dict(env,CUDA_VISIBLE_DEVICES=gpus);log=(run/domain/'console.log').open('ab',buffering=0)
   child=subprocess.Popen(argv,cwd=run/'source',env=denv,stdin=subprocess.DEVNULL,stdout=log,stderr=log);log.close()
   children.append((domain,child));write_json(run/domain/'command.json',dict(argv=argv,cwd=str(run/'source'),CUDA_VISIBLE_DEVICES=gpus))
  status(run/'status.json','encoding',children={name:p.pid for name,p in children},deadline_seconds=48*3600)
  deadline=time.monotonic()+48*3600
  while any(p.poll() is None for _,p in children):
   for name,p in children:
    if p.poll() is not None and p.returncode:raise RuntimeError(f'{name} encoder exit {p.returncode}')
   if time.monotonic()>deadline:raise TimeoutError('Encoding budget exceeded 48h')
   time.sleep(5)
  for name,p in children:
   write_json(run/name/'exit.json',dict(returncode=p.returncode));assert p.returncode==0
   assert read_json(run/name/'encoding_complete.json')['complete']
  execute('analysis_run',[ENCODER,run/'source/analyze.py','--run',run],env,24*3600)
  assert read_json(run/'all_results.json')['complete'];status(run/'status.json','complete',returncode=0,result=str(run/'all_results.json'))
 except BaseException as e:
  for name,p in children:
   if p.poll() is None:
    p.terminate()
    try:p.wait(timeout=30)
    except subprocess.TimeoutExpired:p.kill();p.wait()
   write_json(run/name/'exit.json',dict(returncode=p.returncode))
  status(run/'status.json','failed',returncode=1,error=repr(e),traceback=traceback.format_exc());raise

def main():
 p=argparse.ArgumentParser();p.add_argument('--worker',type=Path);a=p.parse_args()
 if a.worker:worker(a.worker);return
 lock=(HERE/'launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (HERE/'latest_run.json').exists():
  old=read_json(HERE/'latest_run.json');proc=Path('/proc')/str(old['pid'])
  if proc.exists() and old['run'] in (proc/'cmdline').read_bytes().decode(errors='replace'):raise RuntimeError('Existing supervisor alive')
 run=HERE/'runs'/time.strftime('%Y%m%d_%H%M%S');(run/'source').mkdir(parents=True)
 for f in HERE.glob('*.py'):shutil.copy2(f,run/'source'/f.name)
 for n in ['beauty_g3_item_mapping.csv','实验设计.md','review.md']:
  shutil.copy2(HERE/n,run/'source'/n)
 original=ROOT/'codex/llm2vec_sid_hierarchy_20261113'
 for src,dest in [('encoder_worker.py','original_encoder.py'),('semantic_metrics.py','semantic_metrics.py'),('test_metrics.py','test_metrics.py'),('runtime_common.py','runtime_common.py')]:shutil.copy2(original/src,run/'source'/dest)
 for n in ['launch_review.md']:
  if (HERE/n).exists():shutil.copy2(HERE/n,run/'source'/n)
 write_json(run/'launch.json',dict(user_authorized=True,authorization='2026-11-27 user: launch approved plan; Instruments aligned9922 shared axis',source_sha256={p.name:sha(p) for p in (run/'source').iterdir() if p.is_file()},run=str(run),created=time.time(),gpus={'beauty':[0,1],'instruments':[2,3]},monitor_interval_seconds=900))
 with (run/'supervisor.log').open('ab',buffering=0) as log:
  p=subprocess.Popen([str(ENCODER),'-u',str(run/'source/launch_fusion.py'),'--worker',str(run)],cwd=run/'source',stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
 receipt=dict(run=str(run),pid=p.pid,time=time.time());write_json(HERE/'latest_run.json',receipt);print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
