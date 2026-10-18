#!/usr/bin/env python3
"""Retrieve targeted small terminal originals and full logs, no checkpoints."""
from pathlib import Path
import base64,hashlib,json,sys
sys.path.insert(0,'/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
from transport import Transport
P=Path(__file__).resolve().parent;T=Transport('new-server')
terminal=json.loads((P/'terminal-review-20261018T042225Z.json').read_text())
JOB='/home/bwwu/grid-audits/sid-extensions-p0/jobs/ga-20261018-sportsa2-train-fd-recovery-v1'
target=P/'terminal-originals';target.mkdir(exist_ok=True)
receipts=[]
for r in terminal['original_artifacts_verified']:
 if r['path'].endswith('.ckpt'):continue
 receipt=T.get_verified(JOB+'/'+r['path'],target/r['path'],{'bytes':r['bytes'],'sha256':r['sha256']},timeout=60);receipts.append(receipt)
code='''from pathlib import Path
import base64,hashlib,json
job=Path(JOB)
out={}
for name in ['request.json','status.json','cleanup.json','accepted.json','output/invocation.json','output/deadline.json','output/stage-entry.json','output/stage-exit.json','output/latest_validation.json','output/progress.json']:
 p=job/name;b=p.read_bytes();assert len(b)<1_000_000
 out[name]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'b64':base64.b64encode(b).decode()}
print(json.dumps(out))
'''.replace('JOB',repr(JOB))
for name,r in json.loads(T.python(code,timeout=45)).items():
 b=base64.b64decode(r.pop('b64'));assert hashlib.sha256(b).hexdigest()==r['sha256'];dst=target/name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(b);receipts.append({'remote':JOB+'/'+name,'local':str(dst),**r,'verified':True})
(P/'terminal-copy-receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
print(json.dumps({'files':len(receipts),'bytes':sum(r['bytes'] for r in receipts),'checkpoint_downloads':0,'target':str(target)}))
