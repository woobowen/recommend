"""Stage verified official small files for CPU-only tokenizer/meta-model checks.

This is explicitly not publication of a complete model asset. Formal weights
use assets.py and the accepted download jobs separately.
"""
import json,tarfile
from pathlib import Path
from control import *

def main():
 t=transport.Transport('new-server');rows={};source=Path('/tmp/qwen_official_review');archive=SUPPORT/'tokenizer-metadata.tar';SUPPORT.mkdir(parents=True,exist_ok=True)
 with tarfile.open(archive,'w') as tf:
  for folder in sorted((TASK/'metadata').iterdir()):
   if not folder.is_dir():continue
   manifest=json.loads((folder/'manifest.json').read_text())
   for x in manifest['files']:
    if x['kind']=='weight':continue
    rel=folder.name+'/'+manifest['revision']+'/'+x['path'];p=source/folder.name/x['path'];assert p.stat().st_size==x['size'] and transport.sha(p)==x['sha256']
    rows[rel]={'bytes':x['size'],'sha256':x['sha256']};tf.add(p,arcname=rel,recursive=False)
 sent=t.put(archive,t.profile['remote_support']+'/transfers/qwen-preflight-tokenizer-metadata.tar');dest=DATA+'/tokenizer_metadata'
 code=f"""import json,tarfile,hashlib;from pathlib import Path,PurePosixPath
root=Path({dest!r});root.mkdir(parents=True,exist_ok=True);archive=Path({sent['path']!r});assert hashlib.sha256(archive.read_bytes()).hexdigest()=={sent['sha256']!r};rows={rows!r}
with tarfile.open(archive) as tf:
 members=tf.getmembers();assert len(members)==len(set(m.name for m in members)) and set(m.name for m in members)==set(rows)
 for m in members:
  name=PurePosixPath(m.name);assert m.isfile() and not name.is_absolute() and '..' not in name.parts
  data=tf.extractfile(m).read();row=rows[m.name];assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256'];p=root/m.name;p.parent.mkdir(parents=True,exist_ok=True)
  if p.exists():assert p.read_bytes()==data
  else:
   temp=p.with_name(p.name+'.partial');temp.write_bytes(data);temp.replace(p)
print(json.dumps({{'root':str(root),'files':rows,'complete_model_assets':False,'purpose':'CPU tokenizer/static audit only','all_sha256_verified':True}}))"""
 result=json.loads(t.python(code,timeout=90));transport.atomic(TASK/'evidence/tokenizer_metadata_transfer.json',result);print(json.dumps({'root':dest,'files':len(rows),'bytes':sum(x['bytes'] for x in rows.values()),'all_sha256_verified':True}))
if __name__=='__main__':main()
