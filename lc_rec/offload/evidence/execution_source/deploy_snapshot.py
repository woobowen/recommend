#!/usr/bin/env python3
"""Task-specific reuse of verified Windows OpenSSH transport; only lc_rec is exported."""
import hashlib,io,json,pathlib,subprocess,sys,tarfile
import transport
transport.STAGE=pathlib.Path('/mnt/c/Users/lenovo/AppData/Local/Temp/lc-rec-offload-20261108')
repo=pathlib.Path('/home/addaswsw/recommend-worktrees/lc-rec');root=pathlib.Path(__file__).resolve().parents[1]
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
payload=subprocess.check_output(['git','archive','--format=tar',commit,'lc_rec'],cwd=repo)
archive=root/'snapshots'/f'{commit}.tar';archive.parent.mkdir(exist_ok=True)
manifest=[]
with tarfile.open(fileobj=io.BytesIO(payload)) as source:
    for m in source:
        if m.isfile():
            data=source.extractfile(m).read();manifest.append({'path':m.name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
version={'RUN_COMMIT':commit,'files':manifest,'git_export':True,'has_dot_git':False}
with tarfile.open(archive,'w') as target,tarfile.open(fileobj=io.BytesIO(payload)) as source:
    for m in source:
        target.addfile(m,source.extractfile(m) if m.isfile() else None)
    data=(json.dumps(version,indent=2)+'\n').encode();m=tarfile.TarInfo('RUN_VERSION.json');m.size=len(data);target.addfile(m,io.BytesIO(data))
t=transport.Transport('new-server')
remote='/home/bwwu/lc-rec/releases/'+commit
t.put(archive,'/home/bwwu/lc-rec/deploy/'+archive.name)
code='''import pathlib,tarfile,json,hashlib
p=pathlib.Path(ARCHIVE);dest=pathlib.Path(DEST)
if not dest.exists():
 dest.mkdir(parents=True)
 with tarfile.open(p) as tf:
  for m in tf.getmembers():
   q=pathlib.PurePosixPath(m.name)
   assert not q.is_absolute() and '..' not in q.parts and (m.isfile() or m.isdir())
  tf.extractall(dest,filter='data')
v=json.loads((dest/'RUN_VERSION.json').read_text())
for row in v['files']:
 q=dest/row['path'];assert q.stat().st_size==row['bytes'] and hashlib.sha256(q.read_bytes()).hexdigest()==row['sha256']
print(json.dumps({'RUN_COMMIT':v['RUN_COMMIT'],'path':str(dest),'file_count':len(v['files']),'verified':True,'dot_git':(dest/'.git').exists()}))
'''.replace('ARCHIVE',repr('/home/bwwu/lc-rec/deploy/'+archive.name)).replace('DEST',repr(remote))
result=json.loads(t.python(code,timeout=120));(root/'evidence'/f'deploy-{commit}.json').write_text(json.dumps({'result':result,'version':version},indent=2)+'\n');print(json.dumps(result))
