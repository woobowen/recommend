"""Fail-closed accounting of all Beauty preflight runner receipts."""
import json
import math
from pathlib import Path

REQUIRED = {
    'letter-a0-beauty-zero-state-20261115-r0',
    'letter-a0-beauty-zero-resume-20261115-r0',
    'letter-a0-beauty-cross-resume-20261115-r0',
    'letter-a0-beauty-boundary-model-20261115-r0',
}
LIMIT = 3600.0


def ledger(jobs_root, current_job=None, reserve_seconds=60.0, max_wall=1500.0, limit=LIMIT):
    root = Path(jobs_root)
    paths = list(root.glob('letter-a0-beauty-*'))
    names = [p.name for p in paths]
    if len(names) != len(set(names)) or not REQUIRED.issubset(names):
        raise ValueError('missing or duplicate required historical jobs')
    rows=[]
    for p in sorted(paths):
        if p.name == current_job:
            continue
        status=json.loads((p/'status.json').read_text())
        if status.get('job_id') != p.name or status.get('status') not in ('SUCCEEDED','FAILED','TIMED_OUT','CANCELLED'):
            raise ValueError('job identity or terminal state invalid: '+p.name)
        if status.get('worker_may_be_live') is not False or not status.get('cleanup',{}).get('verified'):
            raise ValueError('job cleanup not verified: '+p.name)
        candidates=[x for x in (p/'output/guard.json', p/'output/boundary_guard.json',p/'output/inference_guard.json') if x.is_file()]
        if len(candidates)!=1:
            raise ValueError('missing or ambiguous accounting: '+p.name)
        receipt=json.loads(candidates[0].read_text())
        if receipt.get('job_id')!=p.name:
            raise ValueError('receipt identity mismatch: '+p.name)
        cost=receipt.get('charged_device_seconds_upper_bound')
        if not isinstance(cost,(float,int)) or isinstance(cost,bool) or not math.isfinite(cost) or cost<=0:
            raise ValueError('invalid charge: '+p.name)
        rows.append({'job_id':p.name,'status':status['status'],'charged_device_seconds':cost})
    total=math.fsum(r['charged_device_seconds'] for r in rows)
    remaining=limit-total
    wall=min(max_wall,remaining/2-reserve_seconds)
    if wall<=0: raise ValueError('GPU budget exhausted or reserve unavailable')
    return {'jobs':rows,'prior_device_seconds':total,'remaining_device_seconds':remaining,
            'cleanup_reserve_wall_seconds':reserve_seconds,'wall_cap_seconds':wall,'limit_device_seconds':limit}
