"""Recheck changed control/negative paths against an existing completed CPU pipeline."""
import json
from pathlib import Path
import sys
import time
from closure_acceptance import factory,negative_tests,runner_tests
from production_control import Control
from production_identity import need,identity,SYNTHETIC
from formal_control import read,durable_write

p=read(sys.argv[1]);need(p['evidence_scope']==SYNTHETIC,'synthetic-only followup')
c=Control(sys.argv[1],request_factory=factory)
need(all(x['status']=='COMPLETE' for x in read(c.state_path)['stages'].values()),'reuse only a completed pipeline')
suffix=identity(Path(__file__).with_name('closure_acceptance.py'))['sha256'][:12]
root=Path(p['root']);start=time.monotonic()
tests=negative_tests(p,Path(sys.argv[1]),c.runner);durable_write(root/('fault_acceptance_'+suffix+'.json'),tests)
processes=runner_tests(p,c.runner,root/('torchrun_followup_'+suffix))
# No train/generation/evaluator stage is submitted by this followup.
result={'scope':SYNTHETIC,'status':'PASS','negative_tests':tests,'runner_tests':processes,
        'elapsed_seconds':time.monotonic()-start,'source_files':[identity(Path(__file__).with_name(n)) for n in
         ('production_control.py','production_identity.py','production_pipeline.py','closure_acceptance.py','closure_followup.py')],
        'reused_pipeline_run_commit':p['run_commit']}
durable_write(root/('followup_'+suffix+'.json'),result);print(json.dumps(result))
