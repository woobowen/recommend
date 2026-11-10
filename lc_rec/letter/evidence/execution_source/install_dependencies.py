import hashlib,json,sys,zipfile
from pathlib import Path
from transport import Transport
from gridctl import submit
t=Transport('new-server');root=Path('/home/bwwu/lc-rec/letter-20261110');local=Path('/home/addaswsw/recommend-worktrees/lc-rec/lc_rec/letter/evidence/dependency_downloads.json');rows=json.loads(local.read_text());cache=Path('/home/addaswsw/letter-task-20261110/dependencies/chunks');cache.mkdir(exist_ok=True)
for row in rows:
 p=Path(row['local'])
 with zipfile.ZipFile(p) as z: assert z.testzip() is None
 chunklist=[]
 with p.open('rb') as f:
  idx=0
  while b:=f.read(16*1024**2):
   c=cache/(p.name+'.'+str(idx)+'.chunk');c.write_bytes(b);remote=str(root/'wheel-chunks'/c.name);print(t.put(c,remote),flush=True);chunklist.append(remote);idx+=1
 dest=str(root/'wheels'/p.name)
 code="""from pathlib import Path
import hashlib,json,os
parts=%r;q=Path(%r);q.parent.mkdir(parents=True,exist_ok=True)
if not q.exists():
 temp=q.with_suffix(q.suffix+'.assembling')
 with temp.open('wb') as out:
  for part in parts:
   with open(part,'rb') as inp:
    while b:=inp.read(4*1024**2):out.write(b)
 assert temp.stat().st_size==%d and hashlib.sha256(temp.read_bytes()).hexdigest()==%r
 temp.replace(q)
assert q.stat().st_size==%d and hashlib.sha256(q.read_bytes()).hexdigest()==%r
print(json.dumps({'path':str(q),'bytes':q.stat().st_size,'verified':True}))
"""%(chunklist,dest,row['size'],row['sha256'],row['size'],row['sha256'])
 print(t.python(code,timeout=45),flush=True)
print(t.put(local,str(root/'dependency_downloads.json')),flush=True)
profile=t.profile;commit='7a7365c6033864c232a4c2b042bcc9c494c938b3';release=str(root/'releases'/commit);job='letter-20261110-environment-r0';script=release+'/lc_rec/letter/scripts/setup_environment.py';repo=Path('/home/addaswsw/recommend-worktrees/lc-rec')
r={'schema':1,'job_id':job,'capture_version':commit,'argv':['/data/bwwu/envs/grid-20261009-e61baf3/bin/python',script],'cwd':release,'env':{'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','DS_ACCELERATOR':'cuda','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'},'inputs':[{'path':script,'sha256':hashlib.sha256((repo/'lc_rec/letter/scripts/setup_environment.py').read_bytes()).hexdigest()},{'path':str(root/'dependency_downloads.json'),'sha256':hashlib.sha256(local.read_bytes()).hexdigest()}],'timeout_seconds':240,**{k:(profile['hostname'] if k=='host' else profile[k]) for k in ['host','uid','user','namespace','deployment_sha256']}}
q=Path('/home/addaswsw/letter-task-20261110/control/requests')/(job+'.json');q.write_text(json.dumps(r,indent=2)+'\n');print(submit(t,str(q)),flush=True)
