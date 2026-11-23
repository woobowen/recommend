"""Bounded read-only experiment and GitHub tree inventory; no model calls."""
from pathlib import Path
import sys,json,subprocess,datetime
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'github_delivery_audit_20261121'))
from github_api import GitHub
api=GitHub()
records=[]
for branch in ['yjc-grid','SID-Order-Invariance','sid-semantic-hierarchy']:
    ref=api.call('GET','git/ref/heads/'+branch)
    sha=ref['object']['sha']
    tree=api.call('GET','git/trees/'+sha+'?recursive=1')
    (HERE/('inventory_tree_'+branch+'.json')).write_text(json.dumps(tree,ensure_ascii=False,indent=2),encoding='utf8')
    paths=[x['path'] for x in tree['tree']]
    records.append({'branch':branch,'sha':sha,'truncated':tree['truncated'],'path_count':len(paths),'yelp_matches':[x for x in paths if 'yelp' in x.lower()], 'letter_matches':[x for x in paths if 'letter' in x.lower()], 'experiment_paths':[x for x in paths if x.endswith(('stage_result.json','result.json','test_results.json','task.json'))]})
(HERE/'inventory_github_matches.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf8')
remote=r'''
from pathlib import Path
import json,datetime
root=Path('/home/jcy/GRID_wmh_Qwen_TIGER_20261009/experiments')
records=[]
for run in sorted(root.iterdir()):
 if not run.is_dir(): continue
 found=[]
 for p in run.rglob('*.json'):
  if p.is_symlink() or len(p.relative_to(run).parts)>5 or p.stat().st_size>1500000: continue
  if not any(x in p.name.lower() for x in ['status','result','progress','summary','completed','diagnosis']): continue
  try: obj=json.loads(p.read_text())
  except Exception: continue
  found.append({'path':str(p),'bytes':p.stat().st_size,'value':obj})
 records.append({'run':str(run),'records':found})
print(json.dumps({'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'roots':records},ensure_ascii=False))
'''
r=subprocess.run(['ssh','-o','ConnectTimeout=20','ECNU-Server','python3 -'],input=remote,text=True,capture_output=True,timeout=180)
if r.returncode: raise RuntimeError(r.stderr)
data=json.loads(r.stdout)
(HERE/'inventory_server_records.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'github_branches':[{k:v for k,v in x.items() if k not in ['experiment_paths','yelp_matches','letter_matches']} for x in records] if False else [{'branch':x['branch'],'sha':x['sha'],'yelp_matches':len(x['yelp_matches']),'letter_matches':len(x['letter_matches'])} for x in json.loads((HERE/'inventory_github_matches.json').read_text(encoding='utf8'))],'server_roots':[{ 'run':x['run'],'records':len(x['records'])} for x in data['roots']]},ensure_ascii=False))
