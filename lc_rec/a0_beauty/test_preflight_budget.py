#!/usr/bin/env python3
"""CPU regression of Beauty's receipt-based preflight accounting."""
import json
import tempfile
from pathlib import Path
from preflight_budget import REQUIRED, ledger

costs=[28.825041640549898,30.232078190892935,58.17027708142996,197.967664770782]
names=['letter-a0-beauty-zero-state-20261115-r0','letter-a0-beauty-zero-resume-20261115-r0','letter-a0-beauty-cross-resume-20261115-r0','letter-a0-beauty-boundary-model-20261115-r0']

def reject(call):
    try: call()
    except (ValueError,FileNotFoundError): return
    raise AssertionError('invalid ledger was accepted')

with tempfile.TemporaryDirectory() as temp:
    root=Path(temp)
    for i,(name,cost) in enumerate(zip(names,costs)):
        out=root/name/'output';out.mkdir(parents=True)
        (root/name/'status.json').write_text(json.dumps({'job_id':name,'status':'FAILED' if i==3 else 'SUCCEEDED','worker_may_be_live':False,'cleanup':{'verified':True}}))
        (out/'guard.json').write_text(json.dumps({'job_id':name,'charged_device_seconds_upper_bound':cost}))
    r=ledger(root)
    assert len(r['jobs'])==4 and abs(r['prior_device_seconds']-315.1950616836558)<1e-9
    assert r['wall_cap_seconds']==1500 and r['remaining_device_seconds']>3284
    missing=root/names[-1]/'output/guard.json';saved=missing.read_bytes();missing.unlink()
    reject(lambda:ledger(root));missing.write_bytes(saved)
    d=json.loads(saved);d['job_id']=names[0];missing.write_text(json.dumps(d))
    reject(lambda:ledger(root));missing.write_bytes(saved)
    extra=root/'letter-a0-beauty-extra';extra.mkdir()
    reject(lambda:ledger(root));extra.rmdir()
    d=json.loads(saved);d['charged_device_seconds_upper_bound']=3500;missing.write_text(json.dumps(d))
    reject(lambda:ledger(root))
print('PASS: failed historical attempt charged; missing, duplicate identity, unknown and exhausted ledger rejected')
