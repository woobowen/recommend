#!/usr/bin/env python3
"""Move only this task's same-source public data/tokenizer/wheels to owned server storage."""
import hashlib,json,pathlib,tarfile
import transport
transport.STAGE=pathlib.Path('/mnt/c/Users/lenovo/AppData/Local/Temp/lc-rec-20261108')
root=pathlib.Path(__file__).resolve().parents[1];payload=root/'public-relay'
files=[{'path':str(p.relative_to(payload)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(payload.rglob('*')) if p.is_file()]
assert all(not r['path'].endswith(('.safetensors','.bin')) for r in files)
identity={'source_receipt':'public_relay.json','files':files};(payload/'asset-manifest.json').write_text(json.dumps(identity,indent=2)+'\n')
archive=root/'snapshots/public-relay.tar'
with tarfile.open(archive,'w') as t:
    for p in sorted(payload.rglob('*')):
        if p.is_file():t.add(p,arcname=str(p.relative_to(payload)),recursive=False)
t=transport.Transport('new-server');t.profile['remote_support']='/data/bwwu/lc-rec'
digest=hashlib.sha256(archive.read_bytes()).hexdigest();remote='/data/bwwu/lc-rec/relay/'+digest+'.tar'
receipt=t.put(archive,remote)
code='''import pathlib,tarfile,json,hashlib
dest=pathlib.Path('/data/bwwu/lc-rec/assets');dest.mkdir(parents=True,exist_ok=True)
with tarfile.open(ARCHIVE) as tf:
 for m in tf.getmembers():
  q=pathlib.PurePosixPath(m.name);assert m.isfile() and not q.is_absolute() and '..' not in q.parts
  p=dest/m.name;p.parent.mkdir(parents=True,exist_ok=True);b=tf.extractfile(m).read()
  if p.exists():assert hashlib.sha256(p.read_bytes()).hexdigest()==hashlib.sha256(b).hexdigest(), 'existing asset conflict'
  else:p.write_bytes(b)
identity=json.loads((dest/'asset-manifest.json').read_text())
for row in identity['files']:
 p=dest/row['path'];assert p.stat().st_size==row['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
print(json.dumps({'verified':True,'files':len(identity['files']),'bytes':sum(r['bytes'] for r in identity['files']),'destination':str(dest)}))
'''.replace('ARCHIVE',repr(remote))
result=json.loads(t.python(code,timeout=120));(root/'evidence/asset_relay.json').write_text(json.dumps({'transfer':receipt,'extraction':result,'identity':identity},indent=2)+'\n');print(json.dumps(result))
