#!/usr/bin/env python3
"""Isolated full-axis synthetic evaluator exercise; never creates formal receipts."""
import argparse,hashlib,json,math,os,time,sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import offline_metrics as primary
from uneven_sampler import assignment,assignment_sha256

LABEL='SYNTHETIC_CPU_INTEGRATION_ONLY'

def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--repo-root',type=Path,required=True);p.add_argument('--source-commit',required=True);a=p.parse_args()
 if LABEL not in str(a.output) or a.output.exists():raise ValueError('new, explicitly SYNTHETIC_CPU_INTEGRATION_ONLY output required')
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU only')
 a.output.mkdir(parents=True);start=time.monotonic();write(a.output/'SYNTHETIC_NOTICE.json',{'designation':LABEL,'real_model_predictions':False,'formal_job_receipt':False})
 def safety():
  if time.monotonic()-start>6900:raise TimeoutError('CPU wall budget')
  size=sum(p.stat().st_size for p in a.output.rglob('*') if p.is_file())
  if size>8*1024**3:raise RuntimeError('CPU temporary space budget')
 assets={}
 for kind,digest in primary.ASSETS.items():
  path=a.data_root/'Beauty'/('Beauty.'+kind+'.json')
  if sha(path)!=digest:raise ValueError('Beauty asset identity '+kind)
  assets[kind]=json.loads(path.read_bytes())
 idx,inter=assets['index'],assets['inter'];users=list(inter)
 assert len(users)==22363 and len(idx)==12101
 assert hashlib.sha256(json.dumps(users,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()=='7430079e2b917dfa5e4bb8ead2702fb6c622fb48ffb684dd2889780f9480af7f'
 sid={k:''.join(v) for k,v in idx.items()};sizes=Counter(sid.values());catalog=sorted(sizes)
 assert len(sizes)==12088 and sum(n>1 for n in sizes.values())==11 and sum(n for n in sizes.values() if n>1)==24
 shards=[assignment(len(users),r,2) for r in (0,1)]
 assert tuple(map(assignment_sha256,shards))==primary.RANK_ASSIGNMENT_SHA
 cfg=json.loads((a.repo_root/'lc_rec/letter/config/a0_evaluation_v1.json').read_text())
 source_commit=a.source_commit
 assert len(source_commit)==40 and all(c in '0123456789abcdef' for c in source_commit)
 cfg.update(protocol_version='letter_beauty_beam20_v1',dataset='Beauty',users=22363,original_run_commit=source_commit,sid_coverage_denominator=12088,asset_sha256=primary.ASSETS,user_axis_sha256='7430079e2b917dfa5e4bb8ead2702fb6c622fb48ffb684dd2889780f9480af7f')
 write(a.output/'synthetic_config.json',cfg);official=a.repo_root/'lc_rec/letter/review/final_audit_20261113/external/official_cce.py'
 patterns=[0,4,9,19,None,0]
 results={}
 for condition,offset in [('Normal',0),('S3',1),('S4',2)]:
  safety();root=a.output/condition;root.mkdir();test=root/'synthetic_predictions';test.mkdir();counts=Counter();sums=Counter();files=[];patterns_seen=Counter()
  for rank,rows in enumerate(shards):
   path=test/f'predictions.rank{rank}.jsonl'
   with path.open('x') as stream:
    for i in rows:
     uid=users[i];item=str(inter[uid][-1]);target=sid[item];mode=(i+offset)%len(patterns);pos=patterns[mode];patterns_seen[str(pos)]+=1
     vals=[x for x in catalog[(i*17)%len(catalog):(i*17)%len(catalog)+40] if x!=target][:20]
     if len(vals)<20:vals += [x for x in catalog[:40] if x!=target][:20-len(vals)]
     if pos is not None:vals[pos]=target
     if mode==5:vals[1]=target
     if i%7==0 and 18!=pos:vals[18]='<synthetic_invalid_sid>'
     if i%11==0 and pos not in (16,17):vals[17]=vals[16]
     scores=[float(20-j if j!=5 else 16) for j in range(20)]
     candidates=[{'generation_order':j,'raw_decoded':'Response: '+s,'sid':s,'sequence_score':scores[j]} for j,s in enumerate(vals)]
     order=sorted(range(20),key=lambda j:scores[j],reverse=True);ranked=[vals[j] for j in order];rel=[int(s==target) for s in ranked]
     for k in (5,10):
      sums[f'hit@{k}']+=float(any(rel[:k]));sums[f'ndcg@{k}']+=math.fsum(v/math.log2(j+2) for j,v in enumerate(rel[:k]))
     counts['invalid_complete_sid_candidates']+=sum(x not in sizes for x in ranked)
     counts['colliding_sid_candidate_occurrences']+=sum(sizes.get(x,0)>1 for x in ranked)
     row={'row_idx':i,'user_id':uid,'target_item_id':item,'target':target,'rank':rank,'prompt_id':0,'candidates':candidates,'sorted_generation_order':order,'relevance':rel,'generation_seconds':.01,'designation':LABEL}
     stream.write(json.dumps(row,separators=(',',':'))+'\n')
   files.append({'rank':rank,'bytes':path.stat().st_size,'sha256':sha(path),'path':str(path)})
   write(test/f'rank{rank}.json',{'designation':LABEL,'run_commit':source_commit,'assignment_sha256':assignment_sha256(rows)})
  stage={'designation':LABEL,'status':'PASS','run_commit':source_commit,'actual_test_users':len(users),'prediction_files':files,'metrics':{k:v/len(users) for k,v in sums.items()},'anomalies':dict(counts)}
  write(root/'synthetic_stage_a.json',stage)
  primary.evaluate(SimpleNamespace(output=root/'primary',test_dir=test,stage_a_report=root/'synthetic_stage_a.json',config=a.output/'synthetic_config.json',official_cce=official,data_root=a.data_root,run_commit=source_commit,evaluation_commit=source_commit))
  metrics=json.loads((root/'primary/metrics.json').read_text());assert metrics['official_cce_full_user_comparisons']==22363
  results[condition]={'users':len(users),'candidates':len(users)*20,'patterns':dict(patterns_seen),'prediction_files':files,'metrics_sha256':sha(root/'primary/metrics.json'),'full_official_cce_compares':metrics['official_cce_full_user_comparisons'],'synthetic_metrics':metrics['metrics']}
  print(json.dumps({'condition':condition,'status':'SYNTHETIC_EVALUATION_COMPLETE','elapsed_seconds':time.monotonic()-start}),flush=True)
 # Pairing implementation deliberately does not call formal provenance gate with synthetic identities.
 from importlib.util import spec_from_file_location,module_from_spec
 verify=a.repo_root/'lc_rec/letter/review/final_audit_20261113/statistics_verify.py';spec=spec_from_file_location('independent',verify);ind=module_from_spec(spec);spec.loader.exec_module(ind)
 pair={}
 with np.load(a.output/'Normal/primary/per_user_metrics.npz') as z:normal=z['values'].copy();keys=z['keys'].tolist()
 assert keys==primary.KEYS and normal.shape==(22363,16)
 for condition in ('S3','S4'):
  safety()
  with np.load(a.output/condition/'primary/per_user_metrics.npz') as z:
   assert z['keys'].tolist()==keys and z['values'].shape==normal.shape
   delta=z['values']-normal
  rng=np.random.Generator(np.random.PCG64(20261113));draws=np.empty((5000,16))
  for begin in range(0,5000,8):
   selected=rng.integers(0,22363,size=(min(8,5000-begin),22363),dtype=np.int64);draws[begin:begin+len(selected)]=delta[selected].mean(axis=1)
  ci=np.percentile(draws,[2.5,97.5],axis=0,method='linear');ref,other=ind.bootstrap_columns(delta)
  mx=float(np.max(np.abs(draws-other)));mi=float(np.max(np.abs(ci-ref)));assert mx<=1e-13 and mi<=1e-13
  pair[condition]={'repetitions':5000,'max_draw_difference':mx,'max_interval_difference':mi,'mean_difference':dict(zip(keys,delta.mean(axis=0).tolist()))}
  print(json.dumps({'condition':condition,'paired_synthetic_complete':True,'elapsed_seconds':time.monotonic()-start}),flush=True)
 safety();write(a.output/'summary.json',{'designation':LABEL,'users':22363,'conditions':results,'paired_bootstrap':pair,'candidate_count_total':1341780,'formal_paired_entry_executed':False,'formal_provenance_not_simulated':True,'elapsed_seconds':time.monotonic()-start})
 print(json.dumps({'status':'SYNTHETIC_CPU_PRIMARY_AND_BOOTSTRAP_PASS','designation':LABEL,'elapsed_seconds':time.monotonic()-start}),flush=True)
if __name__=='__main__':main()
