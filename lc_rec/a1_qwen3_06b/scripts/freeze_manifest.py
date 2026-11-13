"""Freeze A1 prediction identities before offline evaluation; never use A0 hashes."""
import argparse,datetime,hashlib,json
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def freeze(test,training,run,output):
 test,training=Path(test).resolve(),Path(training).resolve()
 assert len(run)==40 and all(c in '0123456789abcdef' for c in run)
 metrics=read(test/'metrics.json');selected=read(training/'best_selection.json')
 assert metrics['RUN_COMMIT']==selected['run_commit']==run
 assert metrics['actual_test_users']==24772
 for rank in (0,1):
  train=read(training/f'rank{rank}.json')
  assert train['run_commit']==run and train['status']=='TRAINING_COMPLETE' and train['global_step']==8240 and train['epoch']==4.0
 adapter=training/'best_adapter/adapter_model.bin';adapter_sha=sha(adapter)
 assert adapter_sha==selected['adapter_sha256']==metrics['adapter_model_sha256']
 rows=[];identity=None
 for rank in (0,1):
  report=read(test/f'rank{rank}.json');assert report['status']=='PASS' and report['rank']==rank and report['RUN_COMMIT']==run
  assert report['completed_users']==report['assigned_users']==12386
  assert report['model']['revision']=='da87bfb608c14b7cf20ba1ce41287e8de496c0cd'
  assert report['adapter_files']['adapter_model.bin']['sha256']==adapter_sha
  same={k:report[k] for k in ['model','data_identity','author_sources','generation','allowed_callback_sha256','adapter_files']}
  assert identity is None or identity==same;identity=same
  p=test/f'predictions.rank{rank}.jsonl';before=p.stat();digest=sha(p);after=p.stat()
  assert (before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_size,after.st_mtime_ns,after.st_ctime_ns)
  inherited=metrics['prediction_files'][rank]
  assert digest==inherited['sha256'] and before.st_size==inherited['bytes']
  rows.append({'rank':rank,'path':str(p),'bytes':before.st_size,'sha256':digest,'mtime_ns':before.st_mtime_ns,'ctime_ns':before.st_ctime_ns})
 result={'schema':'a1_frozen_prediction_manifest_v1','status':'FROZEN_INPUTS','gpu_run_commit':run,'test_dir':str(test),'training_dir':str(training),'predictions':rows,'adapter_sha256':adapter_sha,'model_and_protocol':identity,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'metadata_files':{str(p):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in [test/'metrics.json',test/'rank0.json',test/'rank1.json',training/'best_selection.json',training/'rank0.json',training/'rank1.json']}}
 with Path(output).open('x') as f:f.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
 return result
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ['test-dir','training','gpu-run-commit','output']:p.add_argument('--'+n,required=True)
 a=p.parse_args();freeze(a.test_dir,a.training,a.gpu_run_commit,a.output)
if __name__=='__main__':main()
