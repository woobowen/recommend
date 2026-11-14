"""CPU-only full byte identity audit of exact frozen assets; no model import."""
import argparse,datetime,hashlib,json,os
from pathlib import Path
import time

p=argparse.ArgumentParser(allow_abbrev=False)
p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);a=p.parse_args()
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
manifest=Path(a.manifest);request=json.loads(manifest.read_text());start=time.monotonic()
rows=[]
for expected in request['files']:
    path=Path(expected['path']);before=path.stat();h=hashlib.sha256();tick=time.monotonic()
    assert path.is_file() and before.st_size==expected['bytes']
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    after=path.stat()
    assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)
    assert h.hexdigest()==expected['sha256'],str(path)
    rows.append({**expected,'actual_sha256':h.hexdigest(),'stable_stat_during_hash':True,'seconds':time.monotonic()-tick})
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
result={'status':'PASS_FULL_BYTE_ASSET_HASHES','GPU_used':False,
 'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'wall_seconds':time.monotonic()-start,
 'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'reference_sources':request['reference_sources'],'bytes_hashed':sum(x['bytes'] for x in rows),'files':rows}
with out.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
