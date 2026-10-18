#!/usr/bin/env python3
"""One bounded read-only A2 training observation; not a polling service."""
from pathlib import Path
import datetime,json,sys,hashlib,base64
sys.path.insert(0,'/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
from transport import Transport
OUT=Path(__file__).resolve().parent
code='''from pathlib import Path
import base64,datetime,hashlib,json,os
root=Path('/home/bwwu/grid-audits/sid-extensions-p0/jobs')
job=root/'ga-20261018-sportsa2-train-fd-recovery-v1'
out={'captured_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'atomic_snapshot':False,'files':{},'log_prefixes':{}}
for name in ['request.json','status.json','accepted.json','output/invocation.json','output/used_config.json','output/effective_config.yaml','output/training_protocol.json','output/actual_train_loader.json','output/structure.json','output/deadline.json','output/progress.json','output/latest_validation.json','output/training_result.json','output/best_checkpoint_load.json','output/stage-entry.json']:
 p=job/name
 if not p.is_file():continue
 b=p.read_bytes();assert len(b)<1_000_000
 out['files'][name]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'b64':base64.b64encode(b).decode()}
for name,limit in [('output/optimizer_updates.jsonl',256),('output/validations.jsonl',256)]:
 p=job/name
 if not p.is_file():continue
 rows=[]
 with p.open('rb') as f:
  for i,line in enumerate(f):
   if i>=limit:break
   try:json.loads(line)
   except json.JSONDecodeError:break
   rows.append(line)
 b=b''.join(rows)
 out['files'][name]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'b64':base64.b64encode(b).decode()}
 out['log_prefixes'][name]={'complete_prefix_records':len(rows),'bounded_record_limit':limit,'source_bytes_observed':p.stat().st_size,'whole_file_captured':len(b)==p.stat().st_size}
quant=root/'ga-20261018-sportsa2-quantize-fd-recovery-v1'
out['quantizer_result']=json.loads((quant/'output/RESULT.json').read_text())
out['quantizer_status']={k:v for k,v in json.loads((quant/'status.json').read_text()).items() if k in ['job_id','status','exit_code','cleanup','ended_at']}
inv=json.loads((job/'output/invocation.json').read_text());sid=Path(inv['effective_spec']['sid']['path'])
out['training_sid']={'path':str(sid),'bytes':sid.stat().st_size,'sha256':hashlib.sha256(sid.read_bytes()).hexdigest()}
snap=Path('/home/bwwu/grid-audits/sid-extensions-p0/grid-a-launch-20261017/snapshots/22286966cd544773')
dep=json.loads((snap/'DEPLOYMENT.json').read_text());out['source_commit']=dep['source_commit'];out['snapshot_sha256']=dep['snapshot_sha256']
out['source_files']=[{'path':rel,'sha256':hashlib.sha256((snap/rel).read_bytes()).hexdigest()} for rel in ['experiments/launch_20261017/train.py','experiments/launch_20261017/training_protocol.json','experiments/beauty_extensions/author/models.py','experiments/beauty_extensions/common.py']]
out['GPU_or_model_calls']=0;out['submit_or_signal_calls']=0
print(json.dumps(out))
'''
raw=json.loads(Transport('new-server').python(code,timeout=45))
stamp=raw['captured_at_utc'].replace(':','').replace('+','_')
folder=OUT/('observation-'+stamp);folder.mkdir(exist_ok=False)
for name,r in raw.pop('files').items():
 b=base64.b64decode(r.pop('b64'));assert hashlib.sha256(b).hexdigest()==r['sha256']
 p=folder/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
raw['observation_directory']=str(folder)
(folder/'observation.json').write_text(json.dumps(raw,indent=2)+'\n')
(OUT/'latest_observation_path.json').write_text(json.dumps({'path':str(folder)},indent=2)+'\n')
print(json.dumps({'observation_directory':str(folder),'training_sid':raw['training_sid'],'quantizer_status':raw['quantizer_status']['status'],'captured_at_utc':raw['captured_at_utc'],'log_prefixes':raw['log_prefixes']}))
