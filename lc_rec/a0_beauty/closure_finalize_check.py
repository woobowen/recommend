"""Targeted verification for terminal-state and backwards-clock budget guards."""
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch
from formal_control import read,durable_write
from production_control import Control
from closure_acceptance import factory
from production_pipeline import validate_stage
from production_identity import need,identity,SYNTHETIC

planpath=Path(sys.argv[1]);p=read(planpath);need(p['evidence_scope']==SYNTHETIC,'test scope only')
c=Control(planpath,request_factory=factory);before=read(c.state_path)
need(all(x['status']=='COMPLETE' for x in before['stages'].values()),'completed pipeline input')
for stage,v in before['stages'].items():validate_stage(p,stage,v['output'])
reconcile=copy.deepcopy(before);reconcile['stages']['collect']['status']='RUNNING';durable_write(c.state_path,reconcile)
after=c.step();need(after['status']=='COMPLETE' and after['stages']['collect']['status']=='COMPLETE','final stage did not close pipeline')
query=c.runner.query;target=after['attempts'][-1]['job_id'];actual=query(target)
try:
    with patch.object(c.runner,'query',side_effect=lambda job:dict(actual,ended_at='2000-01-01T00:00:00+00:00') if job==target else query(job)):
        try:c.step()
        except ValueError as e:need('clock moved backwards' in str(e),'wrong clock rejection');rejected=repr(e)
        else:raise AssertionError('backwards-clock refund accepted')
finally:durable_write(c.state_path,after)
result={'scope':SYNTHETIC,'status':'PASS','all_ten_stage_artifacts_revalidated':True,'terminal_state':after['status'],
        'backwards_clock_rejected':rejected,'production_control':identity(Path(__file__).with_name('production_control.py'))}
durable_write(Path(p['root'])/'finalize_check.json',result);print(json.dumps(result))
