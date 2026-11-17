#!/usr/bin/env python3
"""Validate only local navigation, Git identities and protected content."""
import json,os,pathlib,re,subprocess,urllib.parse
BASE='2a7d193af47a533a8f0e1ab5691902ae3040970b';GRID='9767729a5114e21b9c949892af3d759baca6a98e'
root=pathlib.Path(__file__).resolve().parents[4];os.chdir(root)
def run(*args):return subprocess.check_output(['git',*args])
def text(ref,path):return run('show',ref+':'+path)
def need(ok,message):
 if not ok:raise AssertionError(message)
need(run('branch','--show-current').decode().strip()=='wbw-sid-order','topic branch')
need(subprocess.run(['git','merge-base','--is-ancestor',BASE,'HEAD']).returncode==0,'approved base ancestry')
need(run('rev-parse','refs/heads/wbw-grid').decode().strip()==GRID,'local GRID ref changed')
need(run('rev-parse','refs/heads/wbw-lc-rec').decode().strip()==BASE,'local LC ref changed')
paths=run('diff','--cached','--name-only','-z',BASE).decode().split('\0');paths=[p for p in paths if p]
allowed={'README.md','docs/research/README.md','docs/research/sid-order/README.md','docs/research/sid-order/STRUCTURE.md','docs/research/sid-order/checks/check_structure.py','docs/research/sid-order/checks/second_level_checks.json'}
need(set(paths)<=allowed,'protected paths staged: '+str(set(paths)-allowed))
source_path='docs/research/sid-order/sources.json';old=text(BASE,source_path);new=pathlib.Path(source_path).read_bytes();need(old==new,'sources.json changed')
d=json.loads(new);rows=d['conditions'];need(len(rows)==21 and len({r['id'] for r in rows})==21,'condition coverage');need(sum(len(r['metrics']) for r in rows)==192,'metric coverage')
need(b'TIMED_OUT' in pathlib.Path('docs/research/sid-order/STRUCTURE.md').read_bytes() and b'NOT_RUN' in pathlib.Path('docs/research/sid-order/STRUCTURE.md').read_bytes(),'Beauty failure history')
need(BASE.encode() in pathlib.Path('docs/research/sid-order/STRUCTURE.md').read_bytes() and GRID.encode() in pathlib.Path('docs/research/sid-order/STRUCTURE.md').read_bytes(),'branch sources')
a=text(BASE,'README.md');b=pathlib.Path('README.md').read_bytes();i=a.index(b'\n')+1;need(b.startswith(a[:i]) and b.endswith(a[i:]),'root README historical bytes')
a=text(BASE,'docs/research/sid-order/README.md').decode().splitlines();b=iter(pathlib.Path('docs/research/sid-order/README.md').read_text().splitlines());need(all(any(x==y for y in b) for x in a),'theme original lines')
files=['README.md','docs/research/README.md','docs/research/sid-order/README.md','docs/research/sid-order/STRUCTURE.md'];checked=0
for name in files:
 p=pathlib.Path(name)
 for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
  u=urllib.parse.urlsplit(link);path=urllib.parse.unquote(u.path)
  if u.netloc=='github.com' and path.startswith('/woobowen/recommend/blob/'):
   ref,path=path.split('/blob/',1)[1].split('/',1);need(bool(re.fullmatch('[0-9a-f]{40}',ref)),'historical link not fixed');need(subprocess.run(['git','cat-file','-e',ref+':'+path],stderr=subprocess.DEVNULL).returncode==0,'missing fixed '+link);checked+=1
  elif not u.scheme and not u.netloc and path:
   target=(p.parent/path).resolve();need(target.exists(),'missing relative '+name+' '+link);checked+=1
   if u.fragment:
    headings=re.findall(r'^#{1,6}\s+(.+)$',target.read_text(),re.M); anchors=[re.sub(r'[^\w -]','',h.lower()).replace(' ','-') for h in headings];need(urllib.parse.unquote(u.fragment) in anchors,'missing anchor '+link)
need('[SID 顺序扰动：GRID 与 LETTER-LC-Rec](docs/research/sid-order/README.md)' in pathlib.Path('README.md').read_text(),'root theme entry')
need('[GRID 实现与结果](GRID.md)' in pathlib.Path('docs/research/sid-order/README.md').read_text() and '[LETTER-LC-Rec 实现与结果](LETTER.md)' in pathlib.Path('docs/research/sid-order/README.md').read_text(),'route links')
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1');v=json.loads(subprocess.check_output(['python3','docs/research/sid-order/checks/validate.py'],env=env));t=json.loads(subprocess.check_output(['python3','docs/research/sid-order/checks/test_beauty_roles.py'],env=env));need(v['status']=='PASS' and t['status']=='PASS' and len(t['negative_cases'])==6,'inherited targeted checks')
print(json.dumps({'status':'PASS','base':BASE,'branch':'wbw-sid-order','staged_paths':paths,'protected_path_delta':'none outside allowlist','sources_json_byte_identical_to_base':True,'conditions':len(rows),'historical_metric_values':192,'new_document_links_checked':checked,'previous_source_validation':v,'previous_beauty_negative_tests':t,'scope':'local git objects and documentation only; no model, server or LFS payload verification'},ensure_ascii=False,indent=2))
