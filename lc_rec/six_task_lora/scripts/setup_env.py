#!/usr/bin/env python3
"""Create a task-only venv; reuse previously audited libraries read-only. No pip."""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
env=Path('/data/bwwu/envs/lc-six-task-20261110')
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024>=40*1024**3
assert not env.exists(), 'never overwrite existing environment'
steps=[]
def run(argv):
    start=time.monotonic();r=subprocess.run(argv,capture_output=True,text=True)
    steps.append({'argv':argv,'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'seconds':time.monotonic()-start})
    (out/'setup_log.json').write_text(json.dumps(steps,indent=2)+'\n');print(r.stdout,flush=True);print(r.stderr,flush=True);r.check_returncode()
run([sys.executable,'-m','venv','--system-site-packages',str(env)])
site=env/'lib/python3.10/site-packages'
shared=Path('/data/bwwu/envs/lc-rec-offload-20261108/lib/python3.10/site-packages')
assert shared.is_dir()
pth=site/'lc-six-task-readonly-framework.pth';pth.write_text(str(shared)+'\n/data/bwwu/envs/letter-20261110/lib/python3.10/site-packages\n')
run([str(env/'bin/python'),'-c',"import torch,transformers,peft,deepspeed,accelerate,numpy,sentencepiece,safetensors,psutil,json,importlib.metadata as m; print(json.dumps({'versions':{p:m.version(p) for p in ['torch','transformers','peft','deepspeed','accelerate','numpy','sentencepiece','safetensors','psutil']},'locations':{p:__import__(p).__file__ for p in ['torch','transformers','peft','deepspeed','accelerate']},'cuda_initialized':torch.cuda.is_initialized()}))"])
(out/'environment_changes.json').write_text(json.dumps({'new_environment':str(env),'system_packages':[],'language_packages_installed':[],'toolchains_installed':[],'configs_added':[{'path':str(pth),'text':pth.read_text(),'sha256':hashlib.sha256(pth.read_bytes()).hexdigest()}],'shared_packages_modified':False,'reason':'Separate task environment with read-only audited LC-Rec components; no LETTER model or runtime patches'},indent=2)+'\n')
print('ENVIRONMENT_READY',flush=True)
