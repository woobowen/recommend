"""Check new publication bytes, explicit scope, local links and original source identity."""
from pathlib import Path
import hashlib,json,re,subprocess
from urllib.parse import unquote
HERE=Path(__file__).resolve().parent;REPO=HERE/'repo';DEST=REPO/'semantic-hierarchy/title_categories_20261127'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()=='371b4498281b39ea87d0d065ff200e24ff4912b0'
 sources=json.loads((DEST/'run/launch.json').read_text(encoding='utf8'))['source_sha256']
 for n,h in sources.items():assert sha(DEST/'run/source'/n)==h,n
 missing=[];paths={};total=0
 for p in DEST.rglob('*'):
  if not p.is_file():continue
  assert '__pycache__' not in p.parts and p.suffix not in ['.pyc','.safetensors'],str(p)
  assert p.stat().st_size<100*1024*1024
  rel=p.relative_to(REPO).as_posix();paths[rel]=sha(p);total+=p.stat().st_size
  if p.suffix in ['.md','.py','.json','.jsonl','.txt','.log','.yaml']:
   s=p.read_text(encoding='utf8')
   assert not re.search(r'gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{25,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY',s),rel
  if p.suffix=='.md' and (p.parent==DEST or p.name=='README.md' and p.parent==DEST/'alignment'):
   for link in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf8')):
    if '://' in link or link.startswith('#'):continue
    if not (p.parent/unquote(link.split('#')[0])).exists():missing.append([rel,link])
 assert not missing,missing
 receipt=dict(passed=True,files=len(paths),bytes=total,frozen_source_files=len(sources),secret_patterns_checked=True,top_level_local_links_valid=True,
  review='Author self-verification. Genuine prior launch nonauthor records included; no new nonauthor publication review.',paths=paths)
 (HERE/'validation.json').write_text(json.dumps(receipt,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in receipt.items() if k!='paths'}))
if __name__=='__main__':main()
