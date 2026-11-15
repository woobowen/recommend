"""Synthetic negative tests for formal pairing identity; no real job is implied."""
import copy, hashlib, json, tempfile
from pathlib import Path
from offline_metrics import ASSETS, verify_bootstrap_config
from paired_beauty import validate_provenance,validate_metric_matrix
from offline_metrics import KEYS
import numpy as np

def write(p,v): p.write_text(json.dumps(v))
def main():
 cfg={'unit':'user','replacement':True,'repetitions':5000,'seed':20261113,'rng':'numpy Generator PCG64','percentiles':[2.5,97.5],'percentile_method':'linear'}
 verify_bootstrap_config(cfg)
 matrix=np.zeros((22363,16),dtype=np.float64);validate_metric_matrix(np,KEYS,matrix)
 for keys,values in [(KEYS[::-1],matrix),(KEYS[:-1],matrix),(KEYS,matrix[:2]),(KEYS,np.full((22363,16),np.nan))]:
  try:validate_metric_matrix(np,keys,values)
  except AssertionError:pass
  else:raise AssertionError('invalid metric matrix accepted')
 for key in cfg:
  b=copy.deepcopy(cfg); b[key]=None
  try: verify_bootstrap_config(b)
  except ValueError: pass
  else: raise AssertionError(key)
 with tempfile.TemporaryDirectory() as d:
  root=Path(d);primary=root/'primary';test=root/'test';primary.mkdir();test.mkdir()
  run='1'*40;adapter={'adapter_model.bin':{'sha256':'2'*64,'bytes':42}}
  records=[]
  for rank in (0,1):
   p=test/f'predictions.rank{rank}.jsonl';p.write_text('SYNTHETIC\n')
   records.append({'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
   report={'status':'PASS','rank':rank,'preflight_only':False,'RUN_COMMIT':run,'completed_users':11182-rank,'data_identity':{k:{'sha256':v} for k,v in ASSETS.items()},'adapter_files':adapter}
   write(test/f'rank{rank}.json',report)
  per_user=[]
  for name in ('per_user.csv','per_user_metrics.npz'):
   p=primary/name;p.write_text('SYNTHETIC\n')
   per_user.append({'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
  write(primary/'metrics.json',{'status':'PASS','users':22363,'run_commit':run,'assets':{k:{'sha256':v} for k,v in ASSETS.items()},'prediction_files':records,'per_user_files':per_user})
  assert validate_provenance(primary,test)['run_commit']==run
  original=json.loads((test/'rank1.json').read_text())
  for name,mutate in [('preflight',lambda x:x.update(preflight_only=True)),('run',lambda x:x.update(RUN_COMMIT='3'*40)),('adapter',lambda x:x.update(adapter_files={'adapter_model.bin':{'sha256':'4'*64}})),('user_count',lambda x:x.update(completed_users=16)),('data',lambda x:x['data_identity']['inter'].update(sha256='5'*64))]:
   changed=copy.deepcopy(original);mutate(changed);write(test/'rank1.json',changed)
   try: validate_provenance(primary,test)
   except (AssertionError,KeyError): pass
   else: raise AssertionError(name)
  write(test/'rank1.json',original);(test/'predictions.rank1.jsonl').write_text('CHANGED\n')
  try: validate_provenance(primary,test)
  except AssertionError: pass
  else: raise AssertionError('prediction mismatch')
  (test/'predictions.rank1.jsonl').write_text('SYNTHETIC\n')
  (primary/'per_user.csv').write_text('CHANGED\n')
  try: validate_provenance(primary,test)
  except AssertionError: pass
  else: raise AssertionError('per-user artifact mismatch')
 print('PASS: frozen bootstrap and synthetic provenance negative cases; no formal job evidence')
if __name__=='__main__':main()
