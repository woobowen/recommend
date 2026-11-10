"""Hash-bound Windows SCP staging of the 15 official wheels; no installation."""
import hashlib,json,shlex,subprocess,sys,tarfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from control import *

def main():
 t=transport.Transport('new-server');packages={}
 for family in ['qwen3','qwen35']:
  cfg=json.loads((TASK/'config'/('dependencies_'+family+'.json')).read_text())
  for row in cfg['packages']:packages[row['filename']]=row
 wheelhouse=SUPPORT/'wheels';archive=SUPPORT/'official-wheels.tar'
 with tarfile.open(archive,'w') as tf:
  for name,row in sorted(packages.items()):
   p=wheelhouse/name;assert p.stat().st_size==row['bytes'] and transport.sha(p)==row['sha256'];tf.add(p,arcname=name,recursive=False)
 receipt=t.put(archive,t.profile['remote_support']+'/transfers/qwen-preflight-20261110-official-wheels.tar');destination=DATA+'/wheels'
 compact={name:{'bytes':row['bytes'],'sha256':row['sha256']} for name,row in packages.items()}
 code=f"""import json,hashlib,tarfile;from pathlib import Path
archive=Path({receipt['path']!r});assert hashlib.sha256(archive.read_bytes()).hexdigest()=={receipt['sha256']!r};rows={compact!r};root=Path({destination!r});root.mkdir(parents=True,exist_ok=True)
with tarfile.open(archive) as tf:
 members=tf.getmembers();assert len(members)==len(set(m.name for m in members)) and set(m.name for m in members)==set(rows)
 for m in members:
  assert m.isfile() and '/' not in m.name and '..' not in m.name
  data=tf.extractfile(m).read();r=rows[m.name];assert len(data)==r['bytes'] and hashlib.sha256(data).hexdigest()==r['sha256'];p=root/m.name
  if p.exists():assert p.read_bytes()==data
  else:
   temp=p.with_name(p.name+'.partial');temp.write_bytes(data);temp.replace(p)
print(json.dumps({{'destination':str(root),'files':len(rows),'bytes':sum(r['bytes'] for r in rows.values()),'all_sha256_verified':True,'transfer':{receipt!r}}}))"""
 result=json.loads(t.python(code,timeout=90));transport.atomic(TASK/'evidence/wheels_transfer.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
