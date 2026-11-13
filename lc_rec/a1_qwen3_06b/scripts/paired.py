"""Paired user bootstrap: fixed A0 versus frozen-protocol A1, no GPU use."""
import argparse,csv,hashlib,json,math,time
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 import numpy as np
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--a1',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 a0=Path('/data/bwwu/lc-rec/a0-final-audit-20261113/letter-a0-20261112-final-audit-20261113-offline-r1/primary')
 assert sha(a0/'per_user.csv')=='423309c6765395e42fc6a18a43706a4653d71692b184c65fe5755f78a34df6fb'
 assert sha(a0/'per_user_metrics.npz')=='d46590f04dcd0f8438b67ccea636b3310a61ff081e146305032c614f444a1689'
 assert not a.output.exists();a.output.mkdir(parents=True)
 frozen_keys=([f'hit@{k}' for k in (1,5,10,20)]+[f'{m}@{k}' for m in ('ndcg','mrr') for k in (5,10,20)]+[f'{m}@{k}' for m in ('ItemHit','ItemNDCG') for k in (5,10,20)])
 rows=[];mat=[];keys=None
 for root in [a0,a.a1]:
  with (root/'per_user.csv').open() as f:rows.append(list(csv.DictReader(f)))
  with np.load(root/'per_user_metrics.npz',allow_pickle=False) as z:
   k=z['keys'].tolist();v=z['values'];axis=str(z['user_axis_sha256'])
   assert axis=='1daf63ce4c77b0265777a46b61939006aafab0276b9106c742de602c8c7e867a'
   assert v.dtype==np.float64 and v.shape==(24772,16) and np.array_equal(z['row_idx'],np.arange(24772))
   assert k==frozen_keys and np.isfinite(v).all()
   csv_values=np.asarray([[float(row[key]) for key in frozen_keys] for row in rows[-1]],dtype=np.float64)
   assert csv_values.shape==v.shape and np.array_equal(csv_values,v), 'CSV/NPZ full per-user values differ'
   assert all(row['protocol_version']=='letter_instruments_beam20_v1' for row in rows[-1])
   assert keys is None or keys==k;keys=k;mat.append(v.copy())
 assert len(rows[0])==len(rows[1])==24772
 fields=['row_idx','user_id','user_axis_sha256','target_item_id','target_sid','target_group_size','visible_history_length','history_group','target_training_frequency','popularity_group','protocol_version']
 for i,(left,right) in enumerate(zip(*rows)):
  assert int(left['row_idx'])==i
  assert all(left[k]==right[k] for k in fields),(i,left,right)
 difference=mat[1]-mat[0];rng=np.random.Generator(np.random.PCG64(20261113));draws=np.empty((5000,16),dtype=np.float64)
 start=time.monotonic()
 for begin in range(0,5000,8):
  selected=rng.integers(0,24772,size=(min(8,5000-begin),24772),dtype=np.int64)
  draws[begin:begin+len(selected)]=difference[selected].mean(axis=1)
 low,high=np.percentile(draws,[2.5,97.5],axis=0,method='linear')
 np.save(a.output/'paired_bootstrap_difference_means.npy',draws)
 values={}
 for j,k in enumerate(keys):
  baseline=math.fsum(mat[0][:,j])/24772;candidate=math.fsum(mat[1][:,j])/24772;d=candidate-baseline
  values[k]={'A0':baseline,'A1':candidate,'absolute_A1_minus_A0':d,'relative_difference':d/baseline if baseline else None,'difference_CI95_low':float(low[j]),'difference_CI95_high':float(high[j])}
 result={'status':'PASS_PAIRED_USER_AXIS','users':24772,'identity_fields_exact':fields,'user_axis_sha256':axis,'bootstrap':{'repetitions':5000,'seed':20261113,'rng':'numpy PCG64','replacement':True,'draw_order':'replicate-major int64, same selected user indices for both models','percentile_method':'linear','seconds':time.monotonic()-start},'metrics':values,'inputs':{str(root):{n:sha(root/n) for n in ['per_user.csv','per_user_metrics.npz']} for root in [a0,a.a1]},'limits':'Conditional on two fixed trained configurations; not training-seed variance or evidence isolating the SID effect.'}
 (a.output/'paired_comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'status':result['status'],'users':24772}))
if __name__=='__main__':main()
