"""Normal scoped commit/push and exact remote tree/content readback."""
from pathlib import Path
import hashlib,json,subprocess,sys,time
from urllib.parse import quote
HERE=Path(__file__).resolve().parent;REPO=HERE/'repo';BRANCH='sid-semantic-hierarchy';BASE='371b4498281b39ea87d0d065ff200e24ff4912b0'
sys.path.insert(0,str(HERE.parent/'github_delivery_audit_20261121'))
from github_api import GitHub
def git(*args):return subprocess.check_output(['git',*args],cwd=REPO)
def main():
 api=GitHub();api.session.proxies.update({'https':'http://127.0.0.1:7897'})
 assert api.call('GET','git/ref/heads/'+BRANCH)['object']['sha']==BASE,'Target branch changed; inspect without overwriting'
 assert git('rev-parse','HEAD').decode().strip()==BASE
 staged=git('diff','--cached','--name-only','-z').decode().strip('\0').split('\0')
 assert all(p.startswith('semantic-hierarchy/title_categories_20261127/') or p in ['README.md','semantic-hierarchy/README.md'] for p in staged)
 expected={};content={}
 for p in staged:
  raw=git('show',':'+p);expected[p]=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest();content[p]=hashlib.sha256(raw).hexdigest()
  assert not raw.startswith(b'version https://git-lfs'),p
  if '/run/source/' in p:
   assert raw==(REPO/p).read_bytes(),p
 (HERE/'staged_manifest.json').write_text(json.dumps(dict(parent=BASE,branch=BRANCH,expected_blobs=expected,content_sha256=content),indent=2),encoding='utf8')
 subprocess.run(['git','-c','user.name=Jingchen Yin','-c','user.email=234159278+Justin-hub-coder@users.noreply.github.com','commit','-m','Publish common-axis Title Categories SID hierarchy experiment'],cwd=REPO,check=True)
 commit=git('rev-parse','HEAD').decode().strip()
 (HERE/'prepared_commit.json').write_text(json.dumps(dict(commit=commit,parent=BASE,branch=BRANCH),indent=2))
 subprocess.run(['git','push','origin','HEAD:refs/heads/'+BRANCH],cwd=REPO,check=True)
 assert api.call('GET','git/ref/heads/'+BRANCH)['object']['sha']==commit
 after=api.call('GET','git/trees/'+commit+'?recursive=1');assert not after.get('truncated')
 tree={x['path']:x['sha'] for x in after['tree'] if x['type']=='blob'};before=json.loads((HERE/'tree_before.json').read_text(encoding='utf8'));old={x['path']:x['sha'] for x in before['tree'] if x['type']=='blob'}
 changed={p for p in tree.keys()|old.keys() if tree.get(p)!=old.get(p)};assert changed==set(expected)
 assert all(tree[p]==h for p,h in expected.items())
 prefix='semantic-hierarchy/title_categories_20261127/'
 keys=['README.md','semantic-hierarchy/README.md']+[prefix+p for p in ['README.md','analysis.md','reproduction.md','metrics.csv','run/all_results.json','run/input_audit.json','run/effective_config.json','run/source/encode_fields.py','run/source/semantic_metrics.py','assets.json','verification/cpu_verification.json','fetch_assets.py','prepare_runtime.py']]
 checks=[]
 for p in keys:
  r=api.session.get('https://raw.githubusercontent.com/woobowen/recommend/'+commit+'/'+quote(p),timeout=90);r.raise_for_status();assert hashlib.sha256(r.content).hexdigest()==content[p],p
  checks.append(dict(path=p,bytes=len(r.content),sha256=hashlib.sha256(r.content).hexdigest()))
 start=json.loads((HERE/'starting_branches.json').read_text());now={x['name']:x['commit']['sha'] for x in api.call('GET','branches?per_page=100')};other=[dict(branch=x['name'],before=x['commit']['sha'],after=now.get(x['name']),unchanged=x['commit']['sha']==now.get(x['name'])) for x in start if x['name']!=BRANCH]
 receipt=dict(commit=commit,parent=BASE,branch=BRANCH,all_changed_blobs_verified=len(expected),raw_files=checks,other_branches=other,time_unix=time.time(),normal_git_commit_and_push=True)
 (HERE/'branch_readback.json').write_text(json.dumps(receipt,indent=2),encoding='utf8');print(json.dumps(dict(commit=commit,files=len(expected),raw_reads=len(checks))),flush=True)
if __name__=='__main__':main()
