import hashlib,importlib.metadata as m,json,os,subprocess,sys
from pathlib import Path
print(json.dumps({'python':sys.executable,'version':sys.version,'packages':{p:(m.version(p) if m.packages_distributions().get(p.replace('-','_')) else None) for p in ['torch','transformers','accelerate','deepspeed','peft','bitsandbytes','fschat','flash_attn','safetensors']}}))
for root in ['/data/bwwu/lc-rec/assets','/home/bwwu/lc-rec','/data/bwwu/envs/lc-rec-offload-20261108/lib/python3.10/site-packages']:
 p=Path(root)
 print('ROOT',root)
 if 'site-packages' in root:
  print([q.name for q in p.glob('*.pth')]);[print(q.name,q.read_text()) for q in p.glob('*.pth')]
 else:
  print('\n'.join(str(q) for q in p.glob('*')))
for root in ['/data/bwwu/lc-rec/assets/data','/data/bwwu/lc-rec/assets/llama-7b/4782ad278652c7c71b72204d462d6d01eaaf7549']:
 p=Path(root);print(root, [(str(q),q.stat().st_size) for q in p.glob('*')])
