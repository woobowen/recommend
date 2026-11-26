"""Download verified release bytes and restore the selected checkpoint/predictions."""
from pathlib import Path
import argparse,hashlib,json,urllib.request,zipfile,shutil
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--predictions-only',action='store_true');args=p.parse_args();root=args.root
 spec=json.loads((root/'assets.json').read_text());cache=root/'downloads';cache.mkdir(exist_ok=True)
 for s in spec['assets']:
  if args.predictions_only and s['name']!='beauty-llama-predictions.zip':continue
  f=cache/s['name']
  if not f.exists():
   tmp=f.with_suffix('.partial')
   with urllib.request.urlopen(s['url'],timeout=120) as src,tmp.open('wb') as dst:shutil.copyfileobj(src,dst,8*1024*1024)
   assert sha(tmp)==s['sha256'];tmp.replace(f)
  assert f.stat().st_size==s['bytes'] and sha(f)==s['sha256']
  if s['name']=='beauty-llama-predictions.zip':
   with zipfile.ZipFile(f) as z:
    assert z.testzip() is None
    for m in z.namelist():assert (root/m).resolve().is_relative_to(root.resolve())
    z.extractall(root)
 if not args.predictions_only:
  target=root/'checkpoint/adapter_model.safetensors'
  if not target.exists():
   tmp=target.with_suffix('.partial')
   with tmp.open('wb') as out:
    for s in spec['adapter']['parts']:
     with (cache/s['name']).open('rb') as src:shutil.copyfileobj(src,out,8*1024*1024)
   assert sha(tmp)==spec['adapter']['sha256'];tmp.replace(target)
  assert sha(target)==spec['adapter']['sha256']
 print('Verified requested assets; no model inference performed.')
if __name__=='__main__':main()
