import json
from pathlib import Path
import transport
root=Path(__file__).resolve().parents[1];t=transport.Transport('new-server')
code='''import os,json,datetime
from pathlib import Path
roots=['/data/bwwu/envs/lc-rec-offload-20261108','/data/bwwu/lc-rec/offload-20261108','/home/bwwu/lc-rec/offload-20261108','/data/bwwu/lc-rec/assets/llama-7b/4782ad278652c7c71b72204d462d6d01eaaf7549']
for p in Path('/home/bwwu/lc-rec/releases').iterdir():
 if (p/'lc_rec/offload').is_dir():roots.append(str(p))
for p in Path('/home/bwwu/lc-rec/deploy').glob('*.tar'):
 if p.stem in [Path(r).name for r in roots]:roots.append(str(p))
for p in Path('/home/bwwu/grid-audits/sid-extensions-p0/jobs').glob('lc-rec-offload-*'):roots.append(str(p))
seen=set();rows=[]
for root in roots:
 p=Path(root);total=allocated=files=symlinks=0
 candidates=[p] if p.is_file() else (Path(d)/n for d,dirs,names in os.walk(p,followlinks=False) for n in names)
 for f in candidates:
  if str(f) in seen:continue
  seen.add(str(f))
  try:s=f.lstat()
  except FileNotFoundError:continue
  total+=s.st_size;allocated+=s.st_blocks*512;files+=1;symlinks+=f.is_symlink()
 rows.append({'root':root,'logical_bytes':total,'allocated_bytes':allocated,'files':files,'symlinks_not_followed':symlinks})
volumes=[]
for r in ['/data/bwwu','/home/bwwu']:
 v=os.statvfs(r);volumes.append({'path':r,'available_bytes':v.f_bavail*v.f_frsize,'total_bytes':v.f_blocks*v.f_frsize,'required_safety_bytes':max(30*1024**3,(v.f_blocks*v.f_frsize)//10)})
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Conservative entire same-task roots, including existing13.48GB LLaMA weights and all offload releases/deploy tar. Path-based logical size; symlink targets excluded. Not a full-account scan.','roots':rows,'logical_total_bytes':sum(r['logical_bytes'] for r in rows),'allocated_total_bytes':sum(r['allocated_bytes'] for r in rows),'volumes':volumes,'limit_bytes':100*1024**3,'instantaneous_peak':False}
assert out['logical_total_bytes']<out['limit_bytes']
assert all(v['available_bytes']>=v['required_safety_bytes'] for v in volumes)
print(json.dumps(out))'''
s=t.python(code,timeout=45);p=root/'evidence'/'logical-storage-20261109.json';p.write_text(s)
v=json.loads(s);print(json.dumps({k:v[k] for k in ['utc','logical_total_bytes','allocated_total_bytes','volumes']}))
