#!/usr/bin/env python3
"""Round3: fill audited mandatory dependencies, reuse successful native build."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--package-dir',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);pkg=Path(a.package_dir)
    env=os.environ.copy();cuda='/data/bwwu/lc-rec/offload-20261108/cuda-12.4';python='/data/bwwu/envs/lc-rec-offload-20261108/bin/python'
    env.update(CUDA_HOME=cuda,DS_ACCELERATOR='cuda',PATH=cuda+'/bin:/data/bwwu/envs/lc-rec-offload-20261108/bin:'+env.get('PATH',''),LD_LIBRARY_PATH=cuda+'/lib64:'+env.get('LD_LIBRARY_PATH',''),OMP_NUM_THREADS='4')
    r={'argv':sys.argv,'started':time.time(),'status':'STARTED','commands':[]}
    def run(cmd,name,timeout=180):
        r['commands'].append({'argv':cmd,'log':name})
        with (out/name).open('w') as f:subprocess.run(cmd,check=True,stdout=f,stderr=subprocess.STDOUT,env=env,timeout=timeout)
    try:
        m=json.loads((pkg/'manifest-r3.json').read_text())
        for x in m['files']:
            q=pkg/x['name'];assert q.stat().st_size==x['bytes'] and hashlib.sha256(q.read_bytes()).hexdigest()==x['sha256']
        run([python,'-m','pip','install','--no-index','--no-deps','--no-cache-dir','--find-links',str(pkg),'einops==0.8.1','msgpack==1.1.0','nvidia-ml-py==12.570.86'],'mandatory-install.log')
        run([python,'-B',str(Path(__file__).with_name('verify_cpu_adam.py')),'--output',str(out/'cpu-adam.json')],'cpu-adam.log')
        check='''import importlib.metadata as m,json
from packaging.requirements import Requirement
from transformers import Trainer,TrainingArguments,LlamaForCausalLM
from deepspeed.runtime.zero.config import DeepSpeedZeroConfig
from accelerate import Accelerator
rows=[]
for name in ['deepspeed','transformers','accelerate']:
 for spec in m.requires(name):
  req=Requirement(spec)
  if req.marker and not req.marker.evaluate({'extra':''}):continue
  version=m.version(req.name);assert version in req.specifier
  rows.append({'distribution':name,'requirement':str(req),'actual':version})
print(json.dumps({'status':'PASS','imports':['Trainer','TrainingArguments','LlamaForCausalLM','DeepSpeedZeroConfig','Accelerator'],'mandatory_dependencies':rows},indent=2))
'''
        run([python,'-I','-B','-c',check],'training-imports.json')
        native=json.loads((out/'cpu-adam.json').read_text())['native_cpu_adam']['path'];run(['ldd',native],'native-ldd.txt')
        assert 'not found' not in (out/'native-ldd.txt').read_text()
        run([python,'-m','pip','freeze','--all'],'pip-freeze.txt')
        r['status']='ENVIRONMENT_AND_CPU_ADAM_VERIFIED'
    except BaseException as e:r['status']='FAILED';r['error']=repr(e);raise
    finally:r['elapsed_seconds']=time.time()-r['started'];(out/'environment-completion.json').write_text(json.dumps(r,indent=2)+'\n')

if __name__=='__main__':main()
