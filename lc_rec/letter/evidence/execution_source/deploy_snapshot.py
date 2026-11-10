import hashlib,io,json,shlex,subprocess,sys,tarfile
from pathlib import Path
from transport import Transport
repo=Path('/home/addaswsw/recommend-worktrees/lc-rec');commit=sys.argv[1]
assert subprocess.check_output(['git','rev-parse',commit],cwd=repo,text=True).strip()==commit
root=Path('/home/addaswsw/letter-task-20261110/control');archive=root/(commit+'.tar.gz')
data=subprocess.check_output(['git','archive','--format=tar',commit,'lc_rec/letter'],cwd=repo)
files=[]
with tarfile.open(fileobj=io.BytesIO(data)) as tf,tarfile.open(archive,'w:gz') as dst:
 for item in tf.getmembers():
  if item.isfile():
   payload=tf.extractfile(item).read();files.append({'path':item.name,'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()});dst.addfile(item,io.BytesIO(payload))
 version=json.dumps({'RUN_COMMIT':commit,'files':files},indent=2).encode();item=tarfile.TarInfo('RUN_VERSION.json');item.size=len(version);dst.addfile(item,io.BytesIO(version))
t=Transport('new-server');remote=t.profile['remote_support']+'/deploy/'+archive.name
print(t.put(archive,remote),flush=True)
release=t.profile['remote_support']+'/releases/'+commit
code="""import hashlib,json,tarfile
from pathlib import Path
p=Path(%r);out=Path(%r)
assert hashlib.sha256(p.read_bytes()).hexdigest()==%r
if not out.exists():
 out.mkdir(parents=True)
 with tarfile.open(p) as tf:
  for m in tf.getmembers():
   assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
  tf.extractall(out,filter='data')
v=json.loads((out/'RUN_VERSION.json').read_text());assert v['RUN_COMMIT']==%r
for r in v['files']:
 q=out/r['path'];assert q.stat().st_size==r['bytes'] and hashlib.sha256(q.read_bytes()).hexdigest()==r['sha256']
print(json.dumps({'release':str(out),'commit':v['RUN_COMMIT'],'files':len(v['files']),'all_hashes_verified':True}))
"""%(remote,release,hashlib.sha256(archive.read_bytes()).hexdigest(),commit)
x=t.python(code,timeout=50);(root/(commit+'.deployment.json')).write_text(x);print(x)
