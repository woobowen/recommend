#!/usr/bin/env python3
"""Nonauthor lightweight SSH read of final effective training records only."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'lc_rec/letter/scripts'))
import formal_control

RUN = '4c6a83d9ba42dc2fe737792759e410ef1874da76'
REL = '/home/bwwu/lc-rec/letter-a0-normal-det-20261112/releases/' + RUN
TRAIN = '/data/bwwu/lc-rec/a0-normal-20261111/formal_seed42/training'
source_paths = ['lc_rec/letter/scripts/formal_train.py',
                'lc_rec/letter/scripts/formal_attention.py',
                'lc_rec/letter/scripts/formal_evaluate.py',
                'lc_rec/letter/config/formal_instruments_seed42.json']
expected = {p:hashlib.sha256(subprocess.check_output(['git','show',RUN+':'+p],cwd=ROOT)).hexdigest()
            for p in source_paths}
code = '''from pathlib import Path
import datetime,hashlib,json,os,socket
RUN={run!r};rel=Path({rel!r});training=Path({train!r});expected={expected!r}
assert os.getuid()==1005 and socket.gethostname()=='zwlab-01'
version=json.loads((rel/'RUN_VERSION.json').read_text())
assert version['RUN_COMMIT']==RUN and version['not_a_git_checkout'] is True
source=[]
for name,wanted in expected.items():
 p=rel/name;data=p.read_bytes();actual=hashlib.sha256(data).hexdigest()
 assert actual==wanted
 source.append(dict(path=str(p),bytes=len(data),sha256=actual))
reports=[]
for rank in [0,1]:
 p=training/('rank%d.json'%rank);data=p.read_bytes();r=json.loads(data)
 assert r['run_commit']==RUN and r['status']=='TRAINING_COMPLETE'
 ta=r['training_arguments'];ds=r['effective_ds'];inventory=r['parameter_inventory']
 trainable={{k:v for k,v in inventory.items() if v['requires_grad']}}
 count=0
 for v in trainable.values():
  n=1
  for x in v['shape']:n*=x
  count+=n
 assert len(trainable)==450 and count==289652736
 result={{k:r[k] for k in ['rank','status','run_commit','config_sha256','global_step','epoch',
 'data','data_identity','base_model_identity','training_arguments','effective_ds',
 'optimizer_membership_exact','initial_global_step','a_only_bf16','out_of_place',
 'zero2_no_sync_compat','formal_attention','checkpoint_storage_adaptation']}}
 result['trainable_tensors']=len(trainable);result['trainable_elements']=count
 result['trainable_policy_valid']=all(('lora_' in k or 'modules_to_save.default' in k) for k in trainable)
 result['original_embedding_head_frozen']=all(not v['requires_grad'] for k,v in inventory.items()
   if 'original_module' in k and ('embed_tokens' in k or 'lm_head' in k))
 result['source_file']={{'path':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}}
 reports.append(result)
print(json.dumps(dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 host=socket.gethostname(),uid=os.getuid(),RUN_COMMIT=RUN,source_files=source,ranks=reports,
 scope='Read only final rank JSON and four small fixed-source files; no payload load, no GPU, no remote write'),allow_nan=True))
'''.format(run=RUN,rel=REL,train=TRAIN,expected=expected)

out = Path(__file__).with_name('terminal_effective_raw.json')
assert not out.exists()
response = formal_control.transport.Transport('new-server').python(code,timeout=45)
report = json.loads(response)
with out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({'output':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
                  'source_ranks':[r['source_file'] for r in report['ranks']]}))
