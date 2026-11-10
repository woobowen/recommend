"""CPU regression of the actual failed metadata type and guarded quota arithmetic."""
import argparse, datetime, json, shutil, sys, tempfile
from pathlib import Path
from common import sha,write
from update_budget import charged_attempts

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-commit',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();task=Path(__file__).parents[1];ledger=task/'evidence/before_round3_update-budget.json';raw=json.loads(ledger.read_text());original=ledger.read_bytes()
 with tempfile.TemporaryDirectory(prefix='qwen-round3-cpu-') as directory:
  root=Path(directory);path=root/'metadata.json';write(path,{'missing_keys':set(),'unexpected_keys':set(),'ordered':{'b','a'},'temperature':.8});assert json.loads(path.read_text())=={'missing_keys':[],'unexpected_keys':[],'ordered':['a','b'],'temperature':.8}
  rejected=False
  try:write(path,{'unsupported':object()})
  except TypeError:rejected=True
  assert rejected
  charged,proof=charged_attempts(raw['attempted_updates'],task,task/'evidence/jobs');assert len(proof)==1 and len(raw['attempted_updates'])-len(charged)==1
  assert sum(x['route']=='F' for x in charged)==2 and not any(x['key']=='q17' and x['route']=='F' for x in charged)
  # Synthetic bookkeeping only: no optimizer/model is instantiated or updated.
  synthetic=raw['attempted_updates']+[{'job_id':'synthetic-future-'+key,'key':key,'route':'F','index_in_job':i} for key in ['q17','q35'] for i in [0,1]]
  future,_=charged_attempts(synthetic,task,task/'evidence/jobs');assert sum(x['route']=='F' for x in future)==6;assert all(sum(x['key']==key and x['route']=='F' for x in future)==2 for key in ['q06','q17','q35'])
  # Hash tampering cannot silently grant quota.
  bad=json.loads((task/'config/update_reconciliations.json').read_text());bad['certified_failed_zero_update_jobs'][0]['raw_files_sha256']['status.json']='0'*64
  (root/'config').mkdir();write(root/'config/update_reconciliations.json',bad);review=bad['certified_failed_zero_update_jobs'][0]['independent_review_path'];(root/review).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(task/review,root/review)
  rejected=False
  try:charged_attempts(raw['attempted_updates'],root,task/'evidence/jobs')
  except AssertionError:rejected=True
  assert rejected
 assert ledger.read_bytes()==original and 'torch' not in sys.modules
 result={'status':'PASS_CPU_METADATA_AND_EXPLICIT_QUOTA_REGRESSION','run_commit':a.run_commit,'argv':sys.argv,'cwd':str(Path.cwd()),'interpreter':sys.executable,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':{x:sha(Path(__file__).with_name(x)) for x in ['common.py','probe.py','update_budget.py','check_round3.py']},'actual_ledger_sha256_unchanged':sha(ledger),'actual_failed_attempt_retained':True,'actual_completed_global_updates_F17':0,'synthetic_only_future_limit_F':6,'synthetic_only_per_model_limit_F':2,'unknown_metadata_type_still_rejected':True,'tampered_proof_rejected':True,'no_torch_import_or_CUDA':True,'reconciliation':proof};write(a.output,result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
