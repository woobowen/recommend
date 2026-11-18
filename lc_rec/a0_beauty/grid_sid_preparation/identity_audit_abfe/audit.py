#!/usr/bin/env python3
"""CPU identity audit: fixed manifest, actual GRID TFRecords, P5 and LETTER."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',TF_CPP_MIN_LOG_LEVEL='3',TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1',OMP_NUM_THREADS='2')
import ast, csv, hashlib, json, pickle, sys, tempfile
from collections import Counter, defaultdict
from pathlib import Path
import tensorflow as tf
import torch

torch.set_num_threads(2)
ROOT=Path('/home/bwwu/grid-audits/sid-extensions-p0/g3-identity-abfe')
DATA=Path('/data/bwwu/sid-extensions-assets/563dc50b1c7e/original_assets/data/amazon_data/beauty')
RUN=Path('/data/bwwu/sid-extensions-assets/563dc50b1c7e/experiments/beauty_rkmeans_qwen_20261010_012253')
LETTER=Path('/home/bwwu/grid-audits/sid-extensions-p0/transfers/letter-a0-beauty-author-8d0154e28de3/Beauty')
N=12101

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def need(v,msg):
 if not v: raise ValueError(msg)
def j(p): return json.loads(p.read_text())
def records(p):
 for raw in tf.compat.v1.io.tf_record_iterator(str(p),options=tf.io.TFRecordOptions(compression_type='GZIP')):
  yield tf.train.Example.FromString(raw).features.feature

def main():
 need(sha(ROOT/'dataset_source_manifest.json')=='4f83d14bd94a216ebbaaef6eea9515ff86fc248a1202c0db73ad8b057c9b9229','manifest hash')
 entries=[x for x in j(ROOT/'dataset_source_manifest.json')['files'] if x['path'].startswith('amazon_data/beauty/')]
 expected={x['path'].removeprefix('amazon_data/beauty/'):x for x in entries}
 actual={p.relative_to(DATA).as_posix() for p in DATA.glob('*/*.tfrecord.gz')}
 need(len(expected)==549 and len(entries)==549 and actual==set(expected),f'shard set mismatch missing={set(expected)-actual} extra={actual-set(expected)}')
 for rel,e in sorted(expected.items()):
  p=DATA/rel; need(p.stat().st_size==e['size'] and sha(p)==e['sha256'],f'shard {rel}')
 print('manifest bytes and exact shard set PASS',flush=True)
 maps=ROOT/'datamaps.json'; seq=ROOT/'sequential_data.txt'
 need(sha(maps)=='79bce2481b59b88362621dd07843661a308819f243a00c67540c69b348123d3a','p5 maps')
 need(sha(seq)=='226cce9c3105299ca0db9615d7d3fb32b3175e90da43100ae352599f0f0107b8','p5 sequences')
 m=j(maps); identity=j(ROOT/'p5_beauty_item_identity.json')
 need(identity['item2id']==m['item2id'] and identity['id2item']==m['id2item'],'p5 projection')
 need(set(m['id2item'])==set(map(str,range(1,N+1))) and len(m['item2id'])==N,'p5 key coverage')
 need(all(isinstance(a,str) and a and m['item2id'].get(a)==k for k,a in m['id2item'].items()),'p5 bijection')
 sequences={}
 for line in seq.read_text().splitlines():
  x=list(map(int,line.split())); uid=x[0]-1; need(uid not in sequences,'duplicate p5 user'); sequences[uid]=[i-1 for i in x[1:]]
 items={}
 for rel in sorted(x for x in expected if x.startswith('items/')):
  for f in records(DATA/rel):
   ids=list(f['id'].int64_list.value); texts=list(f['text'].bytes_list.value)
   need(len(ids)==len(texts)==1 and ids[0] not in items,'item duplicate or malformed'); items[ids[0]]=texts[0]
 need(set(items)==set(range(N)),'items coverage')
 split={}; observed=set()
 for name,trim in [('training',2),('evaluation',1),('testing',0)]:
  seen=set(); comparisons=0
  for rel in sorted(x for x in expected if x.startswith(name+'/')):
   for f in records(DATA/rel):
    u=list(f['user_id'].int64_list.value); need(len(u)==1 and u[0] not in seen and u[0] in sequences,'user duplicate or unknown'); u=u[0]; seen.add(u)
    ids=list(f['sequence_data'].int64_list.value); text=list(f['text'].bytes_list.value); ref=sequences[u][:-trim] if trim else sequences[u]; ref=ref if name=='training' else ref[-21:]
    need(ids==ref,f'sequence {name} {u}'); need(len(ids)==len(text) and all(items[i]==t for i,t in zip(ids,text)),f'text {name} {u}')
    observed.update(ids); comparisons+=len(ids)
  need(seen==set(sequences),f'user coverage {name}'); split[name]={'users':len(seen),'item_occurrences':comparisons}; print(name,split[name],flush=True)
 need(observed==set(range(N)),'matched sequence item coverage')
 pkl=RUN/'sid_generation/pickle/merged_predictions.pkl'; g3=RUN/'semantic_ids.pt'; gen=RUN/'sid_generation/pickle/merged_predictions_tensor.pt'
 for p,h in [(pkl,'dcbfb61369cde2627cea50cc42bb80be753e2d9854b85f1825460174486bd590'),(g3,'d41adcfb22a9a47d5e0424c068f4c74f33cc9fdbaf5db08540fbfa0e3d720e18'),(gen,'d5241c43708cb9e5ce0c1e812b965a0b5f92e38d94aead8246bcebeea62e385f')]: need(sha(p)==h,f'asset {p}')
 rows=pickle.loads(pkl.read_bytes()); need(type(rows)==list and len(rows)==N,'prediction length')
 keys=[]
 for row in rows:
  need(type(row)==dict and type(row.get('item_id')) is int and type(row.get('cluster_ids')) is list and len(row['cluster_ids'])==3 and all(type(x) is int and 0<=x<=255 for x in row['cluster_ids']),'bad prediction')
  keys.append(row['item_id'])
 need(len(set(keys))==N and set(keys)==set(items),'prediction key coverage')
 src=ROOT/'tensor_utils.py';need(sha(src)=='93b88bc47a6569f89453d90a7cb60fd690469762eaf1743d3a89b287deeaa26d','source')
 names={'merge_list_of_keyed_tensors_to_single_tensor','deduplicate_rows_in_tensor','transpose_tensor_from_file'}
 nodes=[x for x in ast.parse(src.read_text()).body if isinstance(x,ast.FunctionDef) and x.name in names];need({x.name for x in nodes}==names,'functions')
 ns={'torch':torch,'open_local_or_remote':lambda file_path,mode='rb':open(file_path,mode)}
 module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[]);exec(compile(ast.fix_missing_locations(module),str(src),'exec'),ns)
 merged=ns['merge_list_of_keyed_tensors_to_single_tensor'](rows,'item_id','cluster_ids')
 with tempfile.TemporaryDirectory() as t:
  tmp=str(Path(t)/'rows.pt');torch.save(merged,tmp);ns['deduplicate_rows_in_tensor'](tmp);ns['transpose_tensor_from_file'](tmp); rebuilt=torch.load(tmp,map_location='cpu',weights_only=True)
 sid=torch.load(g3,map_location='cpu',weights_only=True);generated=torch.load(gen,map_location='cpu',weights_only=True)
 need(all(list(x.shape)==[4,N] and x.dtype==torch.int64 for x in (sid,rebuilt,generated)),'shape/dtype');need(torch.equal(sid,rebuilt) and torch.equal(sid,generated),'tensor values');need(torch.unique(sid.T,dim=0).shape[0]==N,'unique sid')
 with (ROOT/'beauty_g3_item_mapping.csv').open(newline='') as f: mapping=list(csv.DictReader(f))
 need(len(mapping)==N,'mapping rows')
 for i,r in enumerate(mapping):
  need(int(r['sid_column'])==int(r['grid_item_id'])==i and r['original_item_id']==m['id2item'][str(i+1)] and r['p5_item_id']==str(i+1) and [int(r[f'sid_{z}']) for z in range(1,5)]==sid[:,i].tolist(),f'mapping {i}')
 need(len({r['original_item_id'] for r in mapping})==N,'ASIN uniqueness')
 letter={name:j(LETTER/f'Beauty.{name}.json') for name in ('inter','item','index')}
 userkeys=set(map(str,sequences)); need(set(letter['inter'])==userkeys,'letter user set')
 differences=[u for u in sequences if letter['inter'][str(u)]!=sequences[u]]
 need(not differences,f'letter full sequences mismatch count={len(differences)} first={differences[:10]}')
 need(set(letter['item'])==set(letter['index'])==set(map(str,range(N))),'letter item/index keys')
 refs={i for values in letter['inter'].values() for i in values};need(refs==set(range(N)),'letter references')
 bridge=ROOT/'bridge.csv'
 with bridge.open('x',newline='') as f:
  w=csv.writer(f,lineterminator='\n');w.writerow(['grid_item_id','sid_column','original_item_id','letter_item_id','sid_1','sid_2','sid_3','sid_4'])
  for i,r in enumerate(mapping):w.writerow([i,i,r['original_item_id'],i,*[r[f'sid_{z}'] for z in range(1,5)]])
 result={'status':'PASS','source_commit':'abfe7827e9c586a10edcb47c1f4a0c2a6ae62f67','files':len(expected),'splits':split,'matched_item_coverage':len(observed),'p5_users':len(sequences),'letter_full_sequences_equal':len(sequences),'letter_refs':len(refs),'prediction_records':len(rows),'tensor_value_differences':0,'bridge_rows':N,'bridge_sha256':sha(bridge),'letter_hashes':{n:sha(LETTER/f'Beauty.{n}.json') for n in letter},'script_sha256':sha(Path(__file__)),'argv':sys.argv,'python':sys.version,'torch':torch.__version__,'tensorflow':tf.__version__,'gpu_used':False}
 (ROOT/'result.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
