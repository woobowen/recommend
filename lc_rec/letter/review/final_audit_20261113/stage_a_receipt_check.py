#!/usr/bin/env python3
"""Nonauthor checks of the real complete Stage A CPU execution receipt."""
import hashlib
import json
from pathlib import Path


def main():
    repo=Path(__file__).resolve().parents[4]
    evidence=repo/'lc_rec/letter/evidence/final_audit_20261113'
    independent=evidence/'independent_original.json'
    status=evidence/'stage_a_status.json'
    original=evidence/'metrics.json'
    a=json.loads(independent.read_text());s=json.loads(status.read_text());o=json.loads(original.read_text())
    checks={}
    def check(name,value):
        checks[name]=bool(value)
        if not value:raise AssertionError(name)
    run='4c6a83d9ba42dc2fe737792759e410ef1874da76'
    check('cpu_runner_success',s['status']=='SUCCEEDED' and s['exit_code']==0
          and s['cleanup_verified'] and not s['worker_alive'] and not s['supervisor_alive'])
    check('cpu_job_identity',s['job_id']=='letter-a0-20261112-final-audit-20261113-stageA-r1')
    check('independent_success',a['status']=='PASS' and a['exit_code']==0)
    check('run_identity',a['run_commit']==run==o['RUN_COMMIT'])
    script=repo/'lc_rec/letter/review/formal_det_20261112/independent_metrics.py'
    check('independent_executed_script',a['script_sha256']==hashlib.sha256(script.read_bytes()).hexdigest()
          =='d21d0eb6479564cd83c534a6a7dd926e48c8d50af6e72578280e94c87fc10473')
    check('actual_worker_argv',s['worker']['argv'][0]=='/data/bwwu/envs/letter-20261110/bin/python'
          and s['worker']['argv'][1:]==a['argv'])
    check('full_coverage',a['actual_test_users']==24772 and a['beam_candidates_per_user']==20
          and a['skipped_users']==0 and a['anomalies']['candidate_count']==495440)
    check('fixed_rank0',a['prediction_files'][0]['users']==12386
          and a['prediction_files'][0]['bytes']==190712530
          and a['prediction_files'][0]['sha256']=='bdcbdde15d3ac6850708888e055aa67ae61212dbb675d9f915b9947b2a208bde')
    check('fixed_rank1',a['prediction_files'][1]['users']==12386
          and a['prediction_files'][1]['bytes']==190154918
          and a['prediction_files'][1]['sha256']=='f5d2ce538bfe86b024c38070f74cce3411a457554ae08ec610ec26987570c629')
    check('original_metrics_unchanged',a['original_metrics']['sha256']
          ==hashlib.sha256(original.read_bytes()).hexdigest()
          =='7cbbc077385923f284ee800b4318f2aefd4fd561b13c5925a24ae4e3429d471e')
    check('unchanged_anomalies',a['anomalies']==o['anomalies'])
    check('unchanged_tolerances',a['tolerances']=={'metric_mean':1e-13,'metric_sum':1e-9})
    for m in ['hit@5','hit@10','ndcg@5','ndcg@10']:
        check(m+'_tolerance',abs(a['metrics'][m]-o['metrics'][m])<=1e-13
              and abs(a['metric_sums'][m]-o['metric_sums'][m])<=1e-9)
    result={'status':'PASS','ORIGINAL_METRICS_INDEPENDENT_AUDIT':'PASS',
        'checks':checks,'RUN_COMMIT':run,'metric_deltas':a['metric_deltas'],
        'inputs':{p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                  for p in [independent,status,original]},
        'scope':'Reviewer inspected independent source and actual full CPU execution receipt; '
                'no second full prediction scan or GPU rerun',
        'semantics':'Complete user relevance and mathematical definitions agree. '
                    'HR sums/means are exact; NDCG accumulation differs only at documented '
                    'floating-point roundoff scale. This is not bitwise aggregate equality.',
        'correction':'Round0 failed before reading predictions because --data-root was incorrect; '
                     'raw failure preserved. One task-level path correction, unchanged script and '
                     'metric tolerances, then successful full r1 execution.'}
    output=Path(__file__).with_name('stage_a_receipt_check.json')
    with output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'status':'PASS','checks':len(checks),'output':str(output)}))


if __name__=='__main__':main()
