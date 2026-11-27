"""Retrieve the data archive linked by the fixed official LC-Rec README."""
from pathlib import Path
import hashlib,io,json,requests,zipfile
HERE=Path(__file__).resolve().parent
URL='https://drive.google.com/uc?export=download&id=19tsIxV_LbErh5FfYhT732M82y0V-3bxR'
def main():
 path=HERE/'LCRec_Instruments_original.zip'
 if not path.exists():
  r=requests.get(URL,timeout=(30,180));r.raise_for_status();assert r.content[:2]==b'PK',r.headers.get('Content-Type')
  path.write_bytes(r.content)
 with zipfile.ZipFile(path) as z:
  assert z.testzip() is None
  print('\n'.join(z.namelist()),flush=True)
  for name in z.namelist():
   if name.endswith(('.item2id','.user2id','.inter.json','.item.json')):
    target=HERE/'author_mapping'/Path(name).name;target.parent.mkdir(exist_ok=True);assert not target.exists() or target.read_bytes()==z.read(name);target.write_bytes(z.read(name))
 record=dict(source_readme='https://github.com/RUCAIBox/LC-Rec/blob/e6f1b3b3f8b4fee55a49a7b3dfeba29dbfcf0669/README.md',folder='https://drive.google.com/drive/folders/1RcJ2M1l5zWPHYuGd9l5Gibcs5w5aI3y6',url=URL,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
 (HERE/'author_download_receipt.json').write_text(json.dumps(record,indent=2))
if __name__=='__main__':main()
