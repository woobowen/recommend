#!/usr/bin/env python3
"""Bounded CPU-only final audit pipeline; outputs independent of original A0."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    with Path(path).open('x') as out:
        json.dump(value, out, indent=2, allow_nan=False)
        out.write('\n')


def fingerprint(path):
    p=Path(path)
    return {'path':str(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,
            'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',type=Path,required=True)
    p.add_argument('--evaluation-commit',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--job-id',required=True)
    args=p.parse_args()
    started=time.monotonic()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    run='4c6a83d9ba42dc2fe737792759e410ef1874da76'
    original=Path('/home/bwwu/lc-rec/letter-a0-normal-det-20261112/releases')/run
    original_data=original/'lc_rec/letter/vendor/LETTER/data'
    oldroot=Path('/data/bwwu/lc-rec/a0-normal-20261111')
    test=oldroot/'letter-a0-20261112-normal-det-seed42-r0/test'
    training=oldroot/'formal_seed42/training'
    stage=Path('/data/bwwu/lc-rec/a0-final-audit-20261113/stage_a_r1/independent_original.json')
    require_stage=read(stage)
    assert require_stage['status']=='PASS' and require_stage['run_commit']==run
    required_hashes=['bdcbdde15d3ac6850708888e055aa67ae61212dbb675d9f915b9947b2a208bde','f5d2ce538bfe86b024c38070f74cce3411a457554ae08ec610ec26987570c629']
    assert [f['sha256'] for f in require_stage['prediction_files']]==required_hashes
    protected=[oldroot/'formal_seed42_ledger.json',test/'metrics.json',test/'rank0.json',test/'rank1.json',training/'best_selection.json',training/'rank0.json',training/'rank1.json']
    before=[fingerprint(x) for x in protected]
    src=args.release/'lc_rec/letter'
    config=src/'config/a0_evaluation_v1.json'
    review=src/'review/final_audit_20261113'
    official=review/'external/official_cce.py'
    cfg=read(config)
    stage_records=[]
    status={'job_id':args.job_id,'original_run_commit':run,'evaluation_commit':args.evaluation_commit,
            'started_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'gpu_used':False,'stages':stage_records,'status':'RUNNING'}
    commands=[('primary',[sys.executable,str(src/'scripts/offline_metrics.py'),'--data-root',str(original_data),'--test-dir',str(test),'--stage-a-report',str(stage),'--config',str(config),'--official-cce',str(official),'--output',str(out/'primary'),'--evaluation-commit',args.evaluation_commit]),
              ('independent_extended',[sys.executable,str(review/'independent_extended.py'),'--data-root',str(original_data),'--test-dir',str(test),'--stage-a-report',str(stage),'--output',str(out/'independent_extended.json'),'--official-cce',str(official),'--official-sha256',cfg['official_cce_sha256'],'--per-user',str(out/'primary/per_user.csv')]),
              ('independent_statistics',[sys.executable,str(review/'statistics_verify.py'),'--data-root',str(original_data/'Instruments'),'--per-user',str(out/'primary/per_user.csv'),'--primary-bootstrap',str(out/'primary/bootstrap.json'),'--primary-draws',str(out/'primary/bootstrap_draw_means.npy'),'--self-test','--output',str(out/'independent_statistics.json')])]
    try:
        for name,argv in commands:
            tick=time.monotonic()
            with (out/(name+'.stdout.log')).open('x') as log:
                r=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,5900-(time.monotonic()-started)))
            stage_records.append({'stage':name,'argv':argv,'exit_code':r.returncode,'elapsed_seconds':time.monotonic()-tick})
            assert r.returncode==0, name+' failed; preserve all results'
        a,b=read(out/'primary/metrics.json'),read(out/'independent_extended.json')
        differences={key:a['metrics'][key]-b['metrics'][key] for key in a['metrics']}
        assert len(differences)==16 and max(abs(v) for v in differences.values())<1e-13
        mapping={'candidates':'candidates','valid':'valid','invalid':'invalid','duplicates':'duplicate_occurrences','colliding':'collision_candidates','users_invalid':'users_with_invalid','users_duplicate':'users_with_duplicate','users_colliding':'users_with_collision','unique_valid_sids':'unique_valid_sids','unique_valid_sid_coverage':'unique_valid_sid_coverage'}
        for k in ('5','10','20'):
            for left,right in mapping.items():
                assert a['sid_statistics'][k][left]==b['sid_diagnostics'][k][right],(k,left)
        assert all(a['first_target_rank_users'][str(r)]==b['first_target_rank_counts'][str(r)] for r in range(1,21))
        assert a['first_target_rank_users']['miss']==b['first_target_rank_counts']['not_in_top20']
        assert all(read(out/'independent_statistics.json').get(k) is not None for k in ['status'])
        after=[fingerprint(x) for x in protected]
        assert before==after, 'Protected original metadata changed'
        write(out/'crosscheck.json',{'status':'PASS','metric_mean_deltas':differences,'all_sid_counts_and_coverage_equal':True,'first_rank_histogram_equal':True,'protected_before':before,'protected_after':after})
        status['status']='PASS'
    except BaseException as e:
        status['status']='FAIL';status['error']=repr(e)
        raise
    finally:
        status.update(elapsed_seconds=time.monotonic()-started,python=sys.executable,argv=sys.argv,
                      finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                      driver_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                      children_cpu_user_seconds=resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime,
                      children_cpu_system_seconds=resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime)
        write(out/'pipeline.json',status)
        print(json.dumps(status),flush=True)


if __name__=='__main__':
    main()
