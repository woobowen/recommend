#!/usr/bin/env python3
import base64,hashlib,json,sys
from pathlib import Path
from transport import Transport,ROOT,atomic
jobs=sys.argv[1:];t=Transport('new-server')
files=['request.json','accepted.json','status.json','attempt.json','cleanup.json','supervisor.log','work.log','output/setup.json','output/pip-install.log','output/deepspeed-build.log','output/cpu-adam.log','output/cpu-adam.json','output/deepspeed-packaging.patch','output/pip-freeze.txt','output/environment-completion.json','output/mandatory-install.log','output/training-imports.json','output/native-ldd.txt','output/nvcc.log','output/venv.log','output/resource-summary.json','output/resource-samples.jsonl','output/probe/probe_rank0.json','output/probe/probe_rank1.json']
for job in jobs:
 assert job.startswith('lc-rec-offload-20261108-') and '/' not in job
 code='''from pathlib import Path
import base64,hashlib,json
root=Path(JOB);rows=[]
for name in FILES:
 p=root/name
 if not p.is_file():continue
 assert not p.is_symlink() and p.stat().st_size<30*1024**2
 b=p.read_bytes();rows.append({'path':name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'base64':base64.b64encode(b).decode()})
print(json.dumps(rows))
'''.replace('JOB',repr(t.profile['job_root']+'/'+job)).replace('FILES',repr(files))
 rows=json.loads(t.python(code,timeout=60));out=ROOT/'light-evidence'/job;out.mkdir(parents=True,exist_ok=True)
 for r in rows:
  b=base64.b64decode(r.pop('base64'));assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256'];p=out/r['path'];p.parent.mkdir(parents=True,exist_ok=True)
  if p.exists():assert p.read_bytes()==b,'do not overwrite previous evidence; query after terminal state'
  else:p.write_bytes(b)
 atomic(out/'retrieval.json',{'host':t.profile['hostname'],'job_id':job,'files':rows,'verified':True,'transport':'existing Windows OpenSSH; remote SHA/bytes checked locally; selected lightweight files, no build tree'})
 print(job,len(rows),'files verified',sum(r['bytes'] for r in rows))
