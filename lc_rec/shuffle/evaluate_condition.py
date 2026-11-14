"""Reuse frozen ranking/CCE/statistics implementations on new frozen predictions."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    for n in ['model-key','test-dir','output','run-commit']:p.add_argument('--'+n,required=True)
    a=p.parse_args();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    root=Path(__file__).resolve().parent;lc=root.parent;out=Path(a.output);out.mkdir(exist_ok=False)
    evaluation_commit=json.loads((lc.parent/'RUN_VERSION.json').read_text())['RUN_COMMIT']
    cfg=json.loads((root/'config.json').read_text());base=cfg['baselines'][a.model_key]
    test=Path(a.test_dir);records=[];tick=time.monotonic()
    def run(name,argv):
        start=time.monotonic()
        with (out/(name+'.log')).open('x') as f:r=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT,timeout=1500)
        records.append({'stage':name,'argv':argv,'exit_code':r.returncode,'wall_seconds':time.monotonic()-start})
        (out/'stages.json').write_text(json.dumps(records,indent=2)+'\n');assert r.returncode==0,name
    predictions=[]
    for rank in [0,1]:
        report=json.loads((test/('rank%d.json'%rank)).read_text())
        assert report['status']=='PASS' and report['RUN_COMMIT']==a.run_commit and report['completed_users']==12386
        assert report['adapter_files']['adapter_model.bin']==base['adapter_identity']
        path=test/('predictions.rank%d.jsonl'%rank)
        predictions.append({'rank':rank,'path':str(path.resolve()),'bytes':path.stat().st_size,'sha256':sha(path)})
    # Legacy schema label is an interface contract of the pinned independent
    # reviewer; task/model fields distinguish both new A0/A1 conditions explicitly.
    manifest={'schema':'a1_frozen_prediction_manifest_v1','task_schema':'letter_shuffle_frozen_prediction_v1',
              'model_key':a.model_key,'status':'FROZEN_INPUTS','gpu_run_commit':a.run_commit,
              'test_dir':str(test.resolve()),'predictions':predictions}
    (out/'prediction_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    original=lc/'letter/review/formal_det_20261112/independent_metrics.py'
    run('independent_original',[sys.executable,str(original),'--data-root',base['data-root'],
                              '--test-dir',str(test),'--output',str(out/'independent_original.json')])
    evaluation_cfg=json.loads((lc/'letter/config/a0_evaluation_v1.json').read_text())
    evaluation_cfg['original_run_commit']=a.run_commit
    evaluation_cfg['model_revision']=Path(base['model']).name
    evaluation_cfg['bootstrap']['scope']='Overall16 conditional on fixed trained model and fixed shuffle realization, not training-seed or shuffle-realization variance'
    (out/'evaluation_v1.json').write_text(json.dumps(evaluation_cfg,indent=2)+'\n')
    official=lc/'letter/review/final_audit_20261113/external/official_cce.py'
    primary_path=lc/'a1_qwen3_06b/scripts/offline_metrics.py'
    spec=importlib.util.spec_from_file_location('frozen_primary',primary_path)
    primary=importlib.util.module_from_spec(spec);spec.loader.exec_module(primary)
    primary.RUN=a.run_commit;primary.PRED=[(r['bytes'],r['sha256']) for r in predictions]
    start=time.monotonic()
    primary.evaluate(SimpleNamespace(output=out/'primary',test_dir=test,config=out/'evaluation_v1.json',
        official_cce=official,stage_a_report=out/'independent_original.json',data_root=Path(base['data-root']),evaluation_commit=evaluation_commit))
    records.append({'stage':'primary','source':str(primary_path),'sha256':sha(primary_path),'wall_seconds':time.monotonic()-start,'exit_code':0})
    run('independent_extended',[sys.executable,str(lc/'a1_qwen3_06b/review/independent_extended.py'),
        '--data-root',base['data-root'],'--test-dir',str(test),'--stage-a-report',str(out/'independent_original.json'),
        '--manifest',str(out/'prediction_manifest.json'),'--per-user',str(out/'primary/per_user.csv'),
        '--official-cce',str(official),'--official-sha256',evaluation_cfg['official_cce_sha256'],
        '--output',str(out/'independent_extended.json')])
    main=json.loads((out/'primary/metrics.json').read_text());ind=json.loads((out/'independent_extended.json').read_text())
    assert set(main['metrics'])==set(ind['metrics']) and len(main['metrics'])==16
    assert max(abs(main['metrics'][k]-ind['metrics'][k]) for k in main['metrics'])<=1e-13
    run('independent_statistics',[sys.executable,str(lc/'letter/review/final_audit_20261113/statistics_verify.py'),
        '--data-root',str(Path(base['data-root'])/'Instruments'),'--per-user',str(out/'primary/per_user.csv'),
        '--primary-bootstrap',str(out/'primary/bootstrap.json'),'--primary-draws',str(out/'primary/bootstrap_draw_means.npy'),
        '--self-test','--output',str(out/'independent_statistics.json')])
    normal_primary=(Path('/data/bwwu/lc-rec/a0-final-audit-20261113/letter-a0-20261112-final-audit-20261113-offline-r1/primary')
                    if a.model_key=='a0' else Path(base['normal_test_dir']).parent/'offline_evaluation/primary')
    run('paired',[sys.executable,str(root/'paired_and_changes.py'),'--normal-primary',str(normal_primary),
        '--shuffle-primary',str(out/'primary'),'--normal-test',base['normal_test_dir'],
        '--shuffle-test',str(test),'--output',str(out/'paired')])
    assert json.loads((out/'independent_statistics.json').read_text())['status']=='PASS'
    result={'status':'PASS','INPUT_PREDICTION_RUN_COMMIT':a.run_commit,'EVALUATION_RUN_COMMIT':evaluation_commit,
            'model_key':a.model_key,'GPU_used':False,'wall_seconds':time.monotonic()-tick,'stages':records}
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
