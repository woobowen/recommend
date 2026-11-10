"""Download only pinned dependency wheels to the task cache; no package install locally."""
import concurrent.futures as cf,hashlib,json,urllib.request,subprocess
from pathlib import Path
root=Path('/home/addaswsw/letter-task-20261110/dependencies');root.mkdir(parents=True,exist_ok=True)
want={'peft':'0.5.0','bitsandbytes':'0.45.5','fschat':'0.2.31','einops':'0.8.1'}
rows=[]
for pkg,v in want.items():
 j=json.load(urllib.request.urlopen(f'https://pypi.org/pypi/{pkg}/{v}/json',timeout=30))
 f=next(x for x in j['urls'] if x['filename'].endswith('.whl') and ('none-any' in x['filename'] or 'manylinux_2_24_x86_64' in x['filename']))
 rows.append({'package':pkg,'version':v,**{k:f[k] for k in ['url','filename','size','digests']}})
# Official prebuilt FlashAttention wheel matches existing CPython3.10 / torch2.6 / CUDA12 / CXX11 ABI=False.
j=json.load(urllib.request.urlopen('https://api.github.com/repos/Dao-AILab/flash-attention/releases/tags/v2.7.4.post1',timeout=30))
f=next(x for x in j['assets'] if 'cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl' in x['name'])
rows.append({'package':'flash-attn','version':'2.7.4.post1','url':f['browser_download_url'],'filename':f['name'],'size':f['size']})
def get(r):
 p=root/r['filename']
 if not p.exists() or p.stat().st_size!=r['size']:
  partial=p.with_suffix(p.suffix+'.partial')
  with (root/(r['filename']+'.download.log')).open('ab') as log:
   subprocess.run(['curl','--fail','--location','--continue-at','-','--retry','3','--retry-delay','1','--connect-timeout','15','--max-time','360','--output',str(partial),r['url']],stdout=log,stderr=log,check=True)
  assert partial.stat().st_size==r['size'];partial.replace(p)
 h=hashlib.sha256(p.read_bytes()).hexdigest()
 if 'digests' in r:assert h==r['digests']['sha256']
 return dict(r,sha256=h,local=str(p))
with cf.ThreadPoolExecutor(max_workers=5) as ex: result=list(ex.map(get,rows))
Path('lc_rec/letter/evidence/dependency_downloads.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps([{'package':x['package'],'bytes':x['size'],'sha256':x['sha256']} for x in result]))
