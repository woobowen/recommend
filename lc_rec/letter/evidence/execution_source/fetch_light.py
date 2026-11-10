"""Fetch manifest-verified terminal task evidence while retaining large adapters on server."""
import hashlib,json,sys,tarfile
from pathlib import Path
from transport import Transport
from gridctl import query
job=sys.argv[1];t=Transport('new-server');status=query(t,job)
assert not status['worker_alive'] and not status['workgroup_live'] and status['cleanup_verified']
remote_job=t.profile['job_root']+'/'+job;remote_tar=t.profile['remote_support']+'/light_exports/'+job+'.tar.gz'
code="""import hashlib,json,tarfile
from pathlib import Path
root=Path(%r);archive=Path(%r);archive.parent.mkdir(parents=True,exist_ok=True)
files=[];excluded=[]
for p in sorted(root.rglob('*')):
 if not p.is_file() or p.is_symlink() or p.name.endswith(('.partial','.tmp','.lock')) or p.name in ['evidence.tar.gz','export-manifest.json']:continue
 if p.stat().st_size>8*1024**2 or p.suffix in ['.bin','.safetensors','.pt']:
  excluded.append({'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'reason':'retained server-side, referenced by probe save/reload receipt'});continue
 files.append({'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
manifest=archive.with_suffix('.manifest.json');manifest.write_text(json.dumps({'job':str(root),'files':files,'excluded':excluded},indent=2)+'\\n')
with tarfile.open(archive,'w:gz') as tf:
 for r in files:tf.add(root/r['path'],arcname=r['path'],recursive=False)
 tf.add(manifest,arcname='light-export-manifest.json',recursive=False)
print(json.dumps({'path':str(archive),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}))
"""%(remote_job,remote_tar)
meta=json.loads(t.python(code,timeout=45));out=Path('/home/addaswsw/letter-task-20261110/control/jobs/new-server')/job;archive=out/'light-evidence.tar.gz';t.get_verified(remote_tar,archive,meta,timeout=120);dest=out/'retrieved-light';dest.mkdir(exist_ok=True)
with tarfile.open(archive) as tf:
 for m in tf.getmembers():assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
 tf.extractall(dest,filter='data')
manifest=json.loads((dest/'light-export-manifest.json').read_text())
for r in manifest['files']:
 p=dest/r['path'];assert p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
print(json.dumps({'local':str(dest),'files':len(manifest['files']),'excluded':manifest['excluded'],'all_hashes_verified':True},indent=2))
