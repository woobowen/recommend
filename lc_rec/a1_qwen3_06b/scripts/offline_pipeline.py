"""CPU-only original/extended/CCE/statistics/paired A1 evaluation and audit."""
import argparse,datetime,json,os,subprocess,sys,time
from pathlib import Path
from freeze_manifest import freeze,sha,read

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ['release','evaluation-commit','gpu-run-commit','test-dir','training','data-root','output','job-id']:p.add_argument('--'+n,required=True)
 a=p.parse_args();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 out=Path(a.output);out.mkdir(parents=True,exist_ok=False);release=Path(a.release)
 task=release/'lc_rec/a1_qwen3_06b';old=release/'lc_rec/letter';scripts=task/'scripts'
 manifest=freeze(a.test_dir,a.training,a.gpu_run_commit,out/'prediction_manifest.json')
 before=manifest['metadata_files'];cfg=read(old/'config/a0_evaluation_v1.json')
 cfg['original_run_commit']=a.gpu_run_commit;cfg['model_revision']='da87bfb608c14b7cf20ba1ce41287e8de496c0cd'
 with (out/'a1_evaluation_v1.json').open('x') as f:f.write(json.dumps(cfg,indent=2)+'\n')
 official=old/'review/final_audit_20261113/external/official_cce.py'
 records=[];started=time.monotonic();status={'status':'RUNNING','gpu_used':False,'GPU_RUN_COMMIT':a.gpu_run_commit,'EVALUATION_RUN_COMMIT':a.evaluation_commit,'job_id':a.job_id,'stages':records}
 commands=[('independent_original',[sys.executable,str(old/'review/formal_det_20261112/independent_metrics.py'),'--data-root',a.data_root,'--test-dir',a.test_dir,'--output',str(out/'independent_original.json')]),
 ('primary',[sys.executable,str(scripts/'offline_metrics.py'),'--data-root',a.data_root,'--test-dir',a.test_dir,'--stage-a-report',str(out/'independent_original.json'),'--config',str(out/'a1_evaluation_v1.json'),'--official-cce',str(official),'--output',str(out/'primary'),'--evaluation-commit',a.evaluation_commit,'--manifest',str(out/'prediction_manifest.json')]),
 ('independent_extended',[sys.executable,str(task/'review/independent_extended.py'),'--data-root',a.data_root,'--test-dir',a.test_dir,'--stage-a-report',str(out/'independent_original.json'),'--output',str(out/'independent_extended.json'),'--official-cce',str(official),'--official-sha256',cfg['official_cce_sha256'],'--per-user',str(out/'primary/per_user.csv'),'--manifest',str(out/'prediction_manifest.json')]),
 ('independent_statistics',[sys.executable,str(old/'review/final_audit_20261113/statistics_verify.py'),'--data-root',str(Path(a.data_root)/'Instruments'),'--per-user',str(out/'primary/per_user.csv'),'--primary-bootstrap',str(out/'primary/bootstrap.json'),'--primary-draws',str(out/'primary/bootstrap_draw_means.npy'),'--self-test','--output',str(out/'independent_statistics.json')]),
 ('paired',[sys.executable,str(scripts/'paired.py'),'--a1',str(out/'primary'),'--output',str(out/'paired')])]
 try:
  for name,argv in commands:
   tick=time.monotonic()
   with (out/(name+'.stdout.log')).open('x') as log:r=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,1700-(time.monotonic()-started)))
   records.append({'stage':name,'argv':argv,'exit_code':r.returncode,'elapsed_seconds':time.monotonic()-tick});assert r.returncode==0,name
  main=read(out/'primary/metrics.json');independent=read(out/'independent_extended.json')
  assert len(main['metrics'])==16 and set(main['metrics'])==set(independent['metrics'])
  delta={k:main['metrics'][k]-independent['metrics'][k] for k in main['metrics']};assert max(abs(v) for v in delta.values())<1e-13
  mapping={'candidates':'candidates','valid':'valid','invalid':'invalid','duplicates':'duplicate_occurrences','colliding':'collision_candidates','users_invalid':'users_with_invalid','users_duplicate':'users_with_duplicate','users_colliding':'users_with_collision','unique_valid_sids':'unique_valid_sids','unique_valid_sid_coverage':'unique_valid_sid_coverage'}
  for k in ['5','10','20']:
   for left,right in mapping.items():assert main['sid_statistics'][k][left]==independent['sid_diagnostics'][k][right],(k,left)
  assert all(main['first_target_rank_users'][str(r)]==independent['first_target_rank_counts'][str(r)] for r in range(1,21))
  assert main['first_target_rank_users']['miss']==independent['first_target_rank_counts']['not_in_top20']
  assert read(out/'independent_statistics.json')['status']=='PASS'
  for name,v in before.items():assert Path(name).stat().st_size==v['bytes'] and sha(name)==v['sha256']
  status.update(status='PASS',all_16_mean_deltas=delta,original_metadata_unchanged=True,manifest_sha256=sha(out/'prediction_manifest.json'))
 except BaseException as e:status.update(status='FAIL',error=repr(e));raise
 finally:
  status['elapsed_seconds']=time.monotonic()-started;status['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
  with (out/'pipeline.json').open('x') as f:f.write(json.dumps(status,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
