"""CPU-only saved output, actual frozen sources, and item/SID/data-axis checks."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import collections,hashlib,json,os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TF_CPP_MIN_LOG_LEVEL='3')
import torch
import tensorflow as tf
tf.config.set_visible_devices([],'GPU');torch.set_num_threads(2)
ROOT=Path('/home/jcy/GRID_wmh_Qwen_TIGER_20261009');OUT=ROOT/'codex/experiment_publication_20261123'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(p.read_text())
def rows(folder):
 opts=tf.data.Options();opts.threading.private_threadpool_size=1;opts.threading.max_intra_op_parallelism=1
 for p in sorted(folder.glob('*.tfrecord.gz')):
  for r in tf.data.TFRecordDataset(str(p),compression_type='GZIP').with_options(opts):yield tf.train.Example.FromString(r.numpy()).features.feature

def main():
 result={'time_beijing':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'gpu_work':False,'model_forward':False,'letter':[],'yelp':{},'source_snapshots':[]}
 paths=[ROOT/'experiments/letter_official_beauty_20261115_125008/formal',ROOT/'experiments/letter_official_instruments_yelp_20261116_142754/instruments/formal',ROOT/'experiments/letter_official_instruments_yelp_20261116_142754/yelp/formal']
 for run in paths:
  j=load(run/'stage_result.json');audit=load(run/'mapping_audit.json');ds=j['dataset']
  assert j['completed'] and not j['preflight_only'];sid=torch.load(audit['sid_path'],map_location='cpu',weights_only=True)
  official={Path(p).name.split('.')[-2]:load(Path(p)) for p in audit['official_sources']}
  assert list(sid.shape)==[4,audit['items']]
  for i in range(audit['items']):assert sid[:,i].tolist()==[int(s.split('_')[1][:-1]) for s in official['index'][str(i)]]
  split_counts={}
  for split,trim in [('training',2),('evaluation',1),('testing',0)]:
   seen=set()
   for f in rows(Path(audit['original_data_path'])/split):
    uid=int(f['user_id'].int64_list.value[0]);seq=list(f['sequence_data'].int64_list.value);assert uid not in seen;seen.add(uid)
    expected=official['inter'][str(uid)];expected=expected[:-trim] if trim else expected
    if split!='training':expected=expected[-21:]
    assert seq==expected,(ds,split,uid)
   assert seen=={int(i) for i in official['inter']};split_counts[split]=len(seen)
  checkpoint=Path(j['checkpoint']);assert sha(checkpoint)==j['checkpoint_sha256']
  checks={}
  for mode in ['unshuffle','shuffle_3','shuffle_4']:
   v=load(run/f'test_{mode}.json');assert v==j['test_metrics'][mode]
   assert v['full_test_set'] and v['evaluated_users']==audit['users'] and v['checkpoint_sha256']==j['checkpoint_sha256']
   assert all(0<=v['metrics'][0]['test/'+k]<=1 for k in ['recall@5','recall@10','ndcg@5','ndcg@10'])
   checks[mode]=v['metrics'][0]
  result['letter'].append({'dataset':ds,'run':str(run),'items':audit['items'],'users':audit['users'],'unique_sid_columns':int(torch.unique(sid.T,dim=0).shape[0]),'sid_exact_with_official':True,'all_split_sequences_exact':split_counts,'checkpoint_sha256':j['checkpoint_sha256'],'three_saved_test_records_equal':True,'test_metrics':checks})
 for run in paths+[ROOT/'experiments/sid_order_yelp_5seed_v1/shuffle/seed_43',ROOT/'experiments/yelp_diagnostic_patience40_seed43']:
  m=load(run/'source_manifest.json')['files'];bad=[]
  for p,h in m.items():
   f=run/'source_snapshot'/p
   if not f.exists() or sha(f)!=h:bad.append(p)
  assert not bad,(run,bad)
  result['source_snapshots'].append({'run':str(run),'files':len(m),'mismatches':bad})
 upstream=ROOT/'experiments/yelp_rkmeans_qwen_aligned_v1';sid=torch.load(upstream/'semantic_ids.pt',map_location='cpu',weights_only=True);vectors=torch.load(upstream/'item_embeddings.pt',map_location='cpu',weights_only=True)
 if isinstance(vectors,dict):result['yelp']['embedding_object_keys']=list(vectors)
 else:assert vectors.shape[0]==sid.shape[1];result['yelp']['embedding_shape']=list(vectors.shape)
 result['yelp']['sid_shape']=list(sid.shape);result['yelp']['unique_sid_columns']=len(torch.unique(sid.T,dim=0));result['yelp']['data_axis']='Qwen aligned_v1 18775 items; distinct from LETTER 20033'
 for label,run in [('patience10',ROOT/'experiments/sid_order_yelp_5seed_v1/shuffle/seed_43'),('patience40',ROOT/'experiments/yelp_diagnostic_patience40_seed43')]:
  j=load(run/'stage_result.json');assert j['completed'] and sha(Path(j['checkpoint']))==j['checkpoint_sha256']
  assert set(j['test_metrics'])=={'unshuffled','shuffle_3','shuffle_4'}
  for mode,v in j['test_metrics'].items():assert v==load(run/f'test_{mode}.json') and v['full_test_set'] and v['checkpoint_sha256']==j['checkpoint_sha256']
  result['yelp'][label]={'global_step':j['global_step'],'checkpoint':j['checkpoint'],'checkpoint_sha256':j['checkpoint_sha256'],'three_saved_tests_equal':True}
 (OUT/'cpu_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
