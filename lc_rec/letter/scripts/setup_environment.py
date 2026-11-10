"""Isolated user venv with inherited read-only framework; only installs pinned local wheels."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
root=Path('/home/bwwu/lc-rec/letter-20261110');env=Path('/data/bwwu/envs/letter-20261110');out=root/'environment';out.mkdir(parents=True,exist_ok=True)
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024>=40*1024**3
assert not env.exists(),'do not replace an existing environment'
base='/data/bwwu/envs/grid-20261009-e61baf3/bin/python'
steps=[]
def run(argv):
 start=time.time();p=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 steps.append({'argv':argv,'returncode':p.returncode,'output':p.stdout,'seconds':time.time()-start});(out/'install_log.json').write_text(json.dumps(steps,indent=2)+'\n');print(p.stdout,flush=True);p.check_returncode()
run([base,'-m','venv','--system-site-packages',str(env)])
site=env/'lib/python3.10/site-packages'
(site/'letter-readonly-framework.pth').write_text('/data/bwwu/envs/lc-rec-offload-20261108/lib/python3.10/site-packages\n')
manifest=json.loads((root/'dependency_downloads.json').read_text())
paths=[]
for r in manifest:
 p=root/'wheels'/r['filename'];assert p.stat().st_size==r['size'];assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'];paths.append(str(p))
run([str(env/'bin/python'),'-m','pip','install','--no-index','--no-deps','--no-cache-dir',*paths])
run([str(env/'bin/python'),'-c',"import torch,transformers,peft,bitsandbytes,flash_attn,fastchat,einops,deepspeed,importlib.metadata as m,json; from fastchat.train.llama2_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn; from peft import prepare_model_for_int8_training; print(json.dumps({p:m.version(p) for p in ['torch','transformers','peft','bitsandbytes','flash-attn','fschat','einops','deepspeed','accelerate']})); print('CUDA_CONTEXT',torch.cuda.is_initialized())"])
print('ISOLATED_ENVIRONMENT_READY',flush=True)
