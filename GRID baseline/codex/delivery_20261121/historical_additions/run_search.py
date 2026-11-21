from pathlib import Path
import base64
import hashlib
import json
import subprocess
HERE=Path(__file__).resolve().parent
source=(HERE/'remote_bounded_search.py').read_bytes()
r=subprocess.run(['ssh','-o','ConnectTimeout=15','ECNU-Server','/home/jcy/GRID_wmh_Qwen_TIGER_20261009/environments/grid/bin/python','-'],input=source,capture_output=True,timeout=90)
(HERE/'ssh_stderr.txt').write_bytes(r.stderr)
assert r.returncode==0,r.stderr.decode(errors='replace')
j=json.loads(r.stdout);saved=[]
for i,row in enumerate(j.pop('payloads')):
    raw=base64.b64decode(row.pop('base64'));assert hashlib.sha256(raw).hexdigest()==row['sha256']
    name=str(i+1).zfill(2)+'_'+row['path'].split('/')[-1]
    p=HERE/'originals'/name;p.parent.mkdir(exist_ok=True);p.write_bytes(raw)
    saved.append({**row,'local':str(p.relative_to(HERE))})
j['acquired_originals']=saved
j['query_script_sha256']=hashlib.sha256(source).hexdigest()
(HERE/'search_receipt.json').write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'scopes':len(j['inventories']),'acquired_originals':len(saved),'name_candidates':len(j['name_candidates']),'content_matches':len(j['content_matches']),'reference_files':len(j['full_test_reference_inspection'])}))
