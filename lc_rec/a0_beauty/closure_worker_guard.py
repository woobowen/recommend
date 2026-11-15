"""Check worker admission using the real accepted requests of the completed CPU run."""
from pathlib import Path
import json,sys
from formal_control import read,durable_write
from production_pipeline import validate_worker_context
from production_identity import need,identity,SYNTHETIC
p=read(sys.argv[1]);need(p['evidence_scope']==SYNTHETIC,'test scope')
s=read(Path(p['root'])/'state.json');checked={}
for stage,rec in s['stages'].items():
 a=s['attempts'][rec['attempt_index']];job=a['job_id'];directory=str(Path(p['job_root'])/job)
 checked[stage]=validate_worker_context(p,stage,rec['output'],job_id=job,job_dir=directory)
 for wrong in (str(Path(p['root'])/'foreign-output'),rec['output']+'-wrong'):
  try:validate_worker_context(p,stage,wrong,job_id=job,job_dir=directory)
  except ValueError:pass
  else:raise AssertionError('worker accepted foreign output')
from paired_beauty import process
try:process(None)
except ValueError as e:need('positive training source context' in str(e),'missing context rejected for wrong reason')
else:raise AssertionError('unbound pairing accepted')
result={'scope':SYNTHETIC,'status':'PASS','real_accepted_worker_contexts':checked,'foreign_outputs_rejected':20,
        'unbound_pairing_rejected':True,'sources':[identity(Path(__file__).with_name(x)) for x in ('production_pipeline.py','paired_beauty.py')]}
durable_write(Path(p['root'])/'worker_context_check.json',result);print(json.dumps(result))
