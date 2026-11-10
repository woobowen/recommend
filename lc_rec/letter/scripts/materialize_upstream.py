import concurrent.futures as cf,hashlib,json,urllib.request
from pathlib import Path
root=Path(__file__).resolve().parents[1]
rows=json.loads((root/'evidence/upstream_files.json').read_text())
def get(r):
 p=root/'vendor/LETTER'/r['path'];p.parent.mkdir(parents=True,exist_ok=True)
 b=urllib.request.urlopen(r['download_url'],timeout=60).read()
 assert len(b)==r['size']
 assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==r['sha']
 p.write_bytes(b)
 return dict(path=r['path'],bytes=len(b),git_blob=r['sha'],sha256=hashlib.sha256(b).hexdigest(),url=r['download_url'])
with cf.ThreadPoolExecutor(max_workers=5) as ex: out=list(ex.map(get,rows))
(root/'evidence/upstream_materialization.json').write_text(json.dumps({'commit':'8d0154e28de37dbb6e24871c508ad8ddb1921cda','directory_metadata_source':'connected GitHub fetch','payload_materialization':'fixed raw download URL; verified against connector Git blob IDs','files':out},indent=2)+'\n')
print(json.dumps({'files':len(out),'bytes':sum(x['bytes'] for x in out)}))
