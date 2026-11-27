"""Download immutable assets, validate every byte/member and safely restore them."""
from pathlib import Path
import argparse,hashlib,json,shutil,urllib.request,zipfile
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--output',required=True,type=Path);p.add_argument('--metadata-only',action='store_true');a=p.parse_args()
 root=a.root.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);cache=out/'downloads';cache.mkdir(exist_ok=True)
 manifest=json.loads((root/'assets.json').read_text(encoding='utf8'))
 for s in manifest['assets']:
  if a.metadata_only and 'metadata' not in s['name']:continue
  f=cache/s['name']
  if not f.exists():
   tmp=f.with_suffix('.partial')
   with urllib.request.urlopen(s['url'],timeout=180) as src,tmp.open('wb') as dst:shutil.copyfileobj(src,dst,8*1024*1024)
   assert tmp.stat().st_size==s['bytes'] and sha(tmp)==s['sha256'];tmp.replace(f)
  assert f.stat().st_size==s['bytes'] and sha(f)==s['sha256']
  with zipfile.ZipFile(f) as z:
   assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
   rows=json.loads(z.read('MANIFEST.json'));expected={r['member'] for r in rows}|{'MANIFEST.json'};assert set(z.namelist())==expected
   for row in rows:
    dest=(out/row['member']).resolve();assert dest.is_relative_to(out)
    h=hashlib.sha256();n=0
    with z.open(row['member']) as src:
     for b in iter(lambda:src.read(8*1024*1024),b''):h.update(b);n+=len(b)
    assert n==row['bytes'] and h.hexdigest()==row['sha256']
    if dest.exists():assert sha(dest)==row['sha256'],'Refusing to overwrite differing file: '+str(dest);continue
    dest.parent.mkdir(parents=True,exist_ok=True)
    with z.open(row['member']) as src,dest.open('wb') as dst:shutil.copyfileobj(src,dst,8*1024*1024)
  print('Verified and restored',s['name'],flush=True)
if __name__=='__main__':main()
