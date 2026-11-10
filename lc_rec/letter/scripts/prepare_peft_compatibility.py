"""Pinned, small CPU-side dependency materialization for native BF16 API compatibility."""
import hashlib,json,urllib.request,zipfile
from pathlib import Path
root=Path('/home/addaswsw/letter-task-20261110/dependencies');rows=[]
for name,version in [('peft','0.10.0'),('huggingface-hub','0.20.3')]:
 j=json.load(urllib.request.urlopen(f'https://pypi.org/pypi/{name}/{version}/json',timeout=30));f=next(x for x in j['urls'] if x['filename'].endswith('none-any.whl'));p=root/f['filename']
 if not p.exists():p.write_bytes(urllib.request.urlopen(f['url'],timeout=45).read())
 h=hashlib.sha256(p.read_bytes()).hexdigest();assert p.stat().st_size==f['size'] and h==f['digests']['sha256']
 with zipfile.ZipFile(p) as z:assert z.testzip() is None
 rows.append({'package':name,'version':version,'filename':f['filename'],'size':f['size'],'sha256':h,'url':f['url'],'local':str(p),'requires_dist':j['info']['requires_dist']})
Path('lc_rec/letter/evidence/dependency_native_bf16.json').write_text(json.dumps(rows,indent=2)+'\n');print([{'package':r['package'],'bytes':r['size']} for r in rows])
