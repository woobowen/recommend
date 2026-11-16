"""Create a new server export from the verified historical export, never overwrite it."""
import base64
import hashlib
import json
import subprocess
from pathlib import Path

RUN='6e1b211d9c37f358c43b1540b40b17f6703150d2'
REPO=Path(__file__).resolve().parents[2]
BASE=Path(__file__).parent/'evidence/production-closure/final-server/releases/e9a3a631929600757b7a99b5763540c6dd9d2b99/RUN_VERSION.json'
SHA=subprocess.check_output(['git','rev-parse',RUN],cwd=REPO,text=True).strip()
old=json.loads(BASE.read_text())
paths=[r['relative'] for r in old['files']]
paths.append('lc_rec/a0_beauty/test_prelaunch_guards.py')
rows=[]; changed={}
for rel in paths:
    content=subprocess.check_output(['git','show',SHA+':'+rel],cwd=REPO)
    h=hashlib.sha256(content).hexdigest()
    rows.append({'relative':rel,'bytes':len(content),'sha256':h})
    previous=next((r for r in old['files'] if r['relative']==rel),None)
    if not previous or previous['sha256']!=h:changed[rel]=base64.b64encode(content).decode('ascii')
remote=r'''import sys,json,pathlib,shutil,base64,hashlib
x=json.load(sys.stdin)
base=pathlib.Path('/data/bwwu/lc-rec/a0-beauty-20261115/production-closure-20261115/releases')
source=base/'e9a3a631929600757b7a99b5763540c6dd9d2b99'; target=base/x['commit']
if target.exists(): raise SystemExit('refusing existing release')
shutil.copytree(source,target,symlinks=False)
for rel,data in x['changed'].items():
 p=target/rel
 if not p.resolve().is_relative_to(target.resolve()): raise ValueError('bad path')
 p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(data,validate=True))
rows=[]
for row in x['rows']:
 p=target/row['relative']; data=p.read_bytes()
 if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('export mismatch '+str(p))
 rows.append(dict(path=str(p),**row))
v={'run_commit':x['commit'],'release':str(target),'files':rows}
(target/'RUN_VERSION.json').write_text(json.dumps(v,sort_keys=True,indent=2)+'\n')
print(json.dumps({'run_commit':x['commit'],'release':str(target),'count':len(rows),'manifest_sha256':hashlib.sha256((target/'RUN_VERSION.json').read_bytes()).hexdigest(),'changed':sorted(x['changed'])}))'''
cmd=['/mnt/c/Windows/System32/OpenSSH/ssh.exe','-o','BatchMode=yes','-o','ConnectTimeout=12','new-server','python3','-c',"'"+remote.replace("'","'\\''")+"'"]
r=subprocess.run(cmd,input=json.dumps({'commit':SHA,'rows':rows,'changed':changed}),text=True,capture_output=True,timeout=60)
print(r.stdout,end=''); print(r.stderr,end='',file=__import__('sys').stderr);r.check_returncode()
