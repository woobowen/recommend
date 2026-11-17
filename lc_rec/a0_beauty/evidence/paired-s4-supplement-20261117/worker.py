"""CPU-only S4 supplement plus bounded frozen-artifact audit and aggregation."""
import argparse, csv, datetime as dt, hashlib, importlib.util, json, math, os, resource, subprocess, sys, time
from collections import Counter
from pathlib import Path

def load(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def identity(p):return {'path':str(p),'bytes':Path(p).stat().st_size,'sha256':sha(p)}
def write(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def need(ok,msg):
 if not ok:raise ValueError(msg)
def event(name,**kw):print(json.dumps({'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'event':name,**kw}),flush=True)
def audit(c,root,out,plan):
 import numpy as np
 from offline_metrics import KEYS,ASSETS,ranking,cce,normalize,module
 from production_identity import verify_predictions
 n=22363;tol=1e-13;matrix={};rows={};metrics={};groups={};tops={};audits={};maxerr=0.;cceerr=0.
 source=Path(plan['release']);official=module(source/'lc_rec/letter/review/final_audit_20261113/external/official_cce.py')
 data=Path(plan['data_root'])/'Beauty';assets={k:identity(data/f'Beauty.{k}.json') for k in ASSETS}
 need({k:r['sha256'] for k,r in assets.items()}==ASSETS,'frozen data SHA')
 index=load(data/'Beauty.index.json');inter=load(data/'Beauty.inter.json');users=list(inter);sid_for={k:''.join(v) for k,v in index.items()};sizes=Counter(sid_for.values())
 axis=hashlib.sha256(json.dumps(users,ensure_ascii=False,separators=(',',':')).encode()).hexdigest();need(axis=='7430079e2b917dfa5e4bb8ead2702fb6c622fb48ffb684dd2889780f9480af7f','user axis')
 best=load(root/'attempts/verify_best_adapter/a0/stage_result.json')['proof']
 need(best['adapter_files']['adapter_model.bin']['sha256']==c['best_adapter_sha256'],'best adapter identity')
 freq=Counter({k:0 for k in index})
 for seq in inter.values():freq.update(str(x) for x in seq[:-2])
 positives=sorted(v for v in freq.values() if v);q1=positives[math.ceil(len(positives)/3)-1];q2=positives[math.ceil(2*len(positives)/3)-1]
 for cond in ('normal','s3','s4'):
  event('audit_condition_begin',condition=cond)
  primary=root/f'attempts/{cond}_eval/a0/primary';test=root/f'attempts/{cond}/a0/test';m=load(primary/'metrics.json');metrics[cond]=m
  need(m['status']=='PASS' and m['users']==n and list(m['metrics'])==KEYS and m['official_cce_full_user_comparisons']==n,'formal metrics scope')
  # Frozen source verifier, with training proof already revalidated by the preceding fixed S4 CLI.
  verify_predictions(test,c['original_run'],best,cond)
  with (primary/'per_user.csv').open() as f:rr=list(csv.DictReader(f))
  with np.load(primary/'per_user_metrics.npz',allow_pickle=False) as z:
   arr=z['values'].copy();need(arr.dtype==np.float64 and arr.shape==(n,16) and np.isfinite(arr).all(),'metric dtype/shape/finite');need(z['keys'].tolist()==KEYS and np.array_equal(z['row_idx'],np.arange(n)) and str(z['user_axis_sha256'])==axis,'NPZ axis')
  need(len(rr)==n and np.array_equal(arr,np.asarray([[float(r[k]) for k in KEYS] for r in rr])),'CSV/NPZ exact')
  for i,r in enumerate(rr):
   seq=inter[users[i]];item=str(seq[-1]);hlen=min(20,len(seq)-1);pop='UNSEEN' if freq[item]==0 else 'LONG_TAIL' if freq[item]<=q1 else 'MID' if freq[item]<=q2 else 'POPULAR';hg='0-5' if hlen<=5 else '6-10' if hlen<=10 else '11-20'
   need(r['row_idx']==str(i) and r['user_id']==users[i] and r['user_axis_sha256']==axis and r['target_item_id']==item and r['target_sid']==sid_for[item] and r['target_group_size']==str(sizes[sid_for[item]]) and r['visible_history_length']==str(hlen) and r['history_group']==hg and r['popularity_group']==pop and r['target_training_frequency']==str(freq[item]) and r['protocol_version']=='letter_beauty_beam20_v1','all-user metadata')
  matrix[cond]=arr;rows[cond]=rr;means={k:math.fsum(arr[:,j])/n for j,k in enumerate(KEYS)};point=max(abs(means[k]-m['metrics'][k]) for k in KEYS);need(point<=tol,'saved point estimates')
  # Read frozen predictions once for numerical audit and list-change crosscheck; never generate predictions.
  seen=set();lists={};rank_counts=[];rawerr=0.;sidstats={str(k):Counter() for k in (5,10,20)};coverage={str(k):set() for k in (5,10,20)}
  for rank in (0,1):
   path=test/f'predictions.rank{rank}.jsonl';before=path.stat();count=0;assignment=[]
   with path.open() as f:
    for line in f:
     r=json.loads(line);i=r['row_idx'];need(i not in seen and 0<=i<n and r['rank']==rank and r['prompt_id']==0,'prediction axis/Prompt0');seen.add(i);assignment.append(i);count+=1
     need(r['user_id']==users[i] and r['target_item_id']==rr[i]['target_item_id'] and r['target']==rr[i]['target_sid'],'prediction pairing')
     candidates=r['candidates'];need(len(candidates)==20,'Beam20');order=sorted(range(20),key=lambda j:candidates[j]['sequence_score'],reverse=True);need(order==r['sorted_generation_order'],'score order')
     for j,x in enumerate(candidates):need(x['generation_order']==j and normalize(x['raw_decoded'])==x['sid'] and math.isfinite(x['sequence_score']),'candidate integrity')
     ss=[candidates[j]['sid'] for j in order];lists[i]=ss;values,_=ranking(ss,r['target']);correct=cce(ss,r['target'],sizes);reference=official.compute_cce_metrics(ss,r['target'],sizes,[5,10,20]);cceerr=max(cceerr,max(abs(correct[k]-reference[k]) for k in reference));need(cceerr<=1e-14,'official CCE tolerance');values.update(correct)
     rawerr=max(rawerr,max(abs(values[k]-arr[i,j]) for j,k in enumerate(KEYS)))
     for k in (5,10,20):
      v=ss[:k];valid=sum(x in sizes for x in v);dups=k-len(set(v));coll=sum(sizes.get(x,0)>1 for x in v);sidstats[str(k)].update(candidates=k,valid=valid,invalid=k-valid,duplicates=dups,colliding=coll,users_invalid=int(valid<k),users_duplicate=int(dups>0),users_colliding=int(coll>0));coverage[str(k)].update(x for x in v if x in sizes)
   after=path.stat();need((before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_size,after.st_mtime_ns,after.st_ctime_ns),'prediction changed during audit');report=load(test/f'rank{rank}.json');need(count==report['completed_users']==(11182 if rank==0 else 11181),'rank count');need(hashlib.sha256(json.dumps(assignment,separators=(',',':')).encode()).hexdigest()==report['assignment_sha256'],'rank assignment');rank_counts.append(count)
  need(len(seen)==n and rawerr<=tol,'all 16 raw metrics');tops[cond]=lists
  for k,counts in sidstats.items():
   need(all(m['sid_statistics'][k][key]==v for key,v in counts.items()),'SID counts')
   need(m['sid_statistics'][k]['unique_valid_sids']==len(coverage[k]) and m['sid_statistics'][k]['coverage_denominator']==12088,'coverage counts')
  g=load(primary/'groups.json');groups[cond]=g;need((g['q1'],g['q2'])==(q1,q2),'group cutpoints');gerr=0.
  for field,labels in g['groups'].items():
   for label,saved in labels.items():
    ids=[i for i,r in enumerate(rr) if r[field]==label];need(saved['users']==len(ids),'group denominator')
    for j,k in enumerate(KEYS):
     if not ids:need(saved['metrics'][k] is None,'empty group')
     else:gerr=max(gerr,abs(math.fsum(arr[i,j] for i in ids)/len(ids)-saved['metrics'][k]))
  need(gerr<=tol,'group mean tolerance')
  boot=load(primary/'bootstrap.json');draws=np.load(primary/'bootstrap_draw_means.npy',allow_pickle=False);need(draws.shape==(5000,16) and draws.dtype==np.float64 and np.isfinite(draws).all(),'saved primary bootstrap');ci=np.percentile(draws,[2.5,97.5],axis=0,method='linear');berr=max(max(abs(boot['intervals'][k][bound]-ci[b,j]) for b,bound in enumerate(('low','high'))) for j,k in enumerate(KEYS));need(berr<=tol,'primary saved intervals')
  audits[cond]={'users':n,'rank_counts':rank_counts,'raw_prediction_metric_max_difference':rawerr,'point_estimate_max_difference':point,'group_max_difference':gerr,'stored_primary_CI_max_difference':float(berr),'primary_bootstrap_recomputed':False,'primary_note':'Stored draw array and interval consistency checked; no new primary bootstrap production run.'};maxerr=max(maxerr,rawerr,point,gerr,float(berr))
 paired={};pair_audit={}
 for cond,path in [('s3',root/'attempts/paired/a0/s3'),('s4',out/'s4')]:
  r=load(path/'paired_comparison.json');need(r['status']=='PASS' and r['scope']=='FORMAL' and r['users']==n and list(r['metrics'])==KEYS,'paired scope');need(r['bootstrap']=={'repetitions':5000,'seed':20261113,'rng':'PCG64','replacement':True,'percentile':'95% linear','scope':'User uncertainty conditional on fixed trained model and fixed shuffle realization; not training-seed or permutation variance.'},'paired protocol')
  if cond=='s3':need(sha(path/'paired_comparison.json')==c['s3_sha256'],'S3 original SHA')
  for condition in ('normal',cond):
   primary=root/f'attempts/{condition}_eval/a0/primary'
   need(r['inputs'][str(primary)]=={name:sha(primary/name) for name in ('per_user.csv','per_user_metrics.npz')},'paired input hashes')
  need(all(x['adapter_files']==best['adapter_files'] and x['run_commit']==c['original_run'] for x in r['prediction_provenance']),'paired same best model')
  ver=r['independent_verification'];need(ver['tolerance']==tol and ver['maximum_draw_difference']<=tol and ver['maximum_interval_difference']<=tol and sha(ver['source'])==ver['source_sha256'],'frozen independent verification')
  draws=np.load(path/'paired_bootstrap_difference_means.npy',allow_pickle=False);need(draws.dtype==np.float64 and draws.shape==(5000,16) and np.isfinite(draws).all(),'paired draws');ci=np.percentile(draws,[2.5,97.5],axis=0,method='linear');pe=0.
  for j,k in enumerate(KEYS):
   normal=math.fsum(matrix['normal'][:,j])/n;shuffle=math.fsum(matrix[cond][:,j])/n;delta=shuffle-normal;expected={'Normal':normal,'Shuffle':shuffle,'absolute_difference':delta,'relative_difference':delta/normal if normal else None,'difference_CI95_low':float(ci[0,j]),'difference_CI95_high':float(ci[1,j])}
   for key,v in expected.items():
    if v is None:need(r['metrics'][k][key] is None,'null relative difference')
    else:pe=max(pe,abs(r['metrics'][k][key]-v))
  need(pe<=tol,'paired 16 points/intervals')
  changes={'users':n,'top1_changed':sum(tops['normal'][i][0]!=tops[cond][i][0] for i in range(n)),'top20_ordered_changed':sum(tops['normal'][i]!=tops[cond][i] for i in range(n)),'top20_set_changed':sum(set(tops['normal'][i])!=set(tops[cond][i]) for i in range(n)),'hit_transitions':{}}
  for k in [k for k in KEYS if k.startswith('hit@')]:
   j=KEYS.index(k);changes['hit_transitions'][k]={f'{a}->{b}':int(np.sum((matrix['normal'][:,j]==a)&(matrix[cond][:,j]==b))) for a in (0,1) for b in (0,1)}
  for k in ('top1_changed','top20_ordered_changed','top20_set_changed'):changes[k+'_fraction']=changes[k]/n
  need(changes==r['changes'],'list changes and transitions');paired[cond]=r;pair_audit[cond]={'status':'PASS','max_point_relative_or_CI_difference':pe,'frozen_independent_verification':ver,'bootstrap_identity':identity(path/'paired_bootstrap_difference_means.npy'),'source_result':identity(path/'paired_comparison.json'),'production_recomputed':cond=='s4'}
 # Numeric-only user matrices for external review: no user IDs or raw recommendations.
 np.savez_compressed(out/'public_metric_matrices.npz',keys=np.asarray(KEYS),row_idx=np.arange(n),user_axis_sha256=np.asarray(axis),normal=matrix['normal'],s3=matrix['s3'],s4=matrix['s4'])
 summary={'designation':c['designation'],'supplemental':True,'ORIGINAL_PIPELINE_STATUS':'PAIRED_TIMED_OUT_COLLECT_NOT_RUN','SUPPLEMENTAL_CPU_ANALYSIS_STATUS':'COMPLETE','review_status':'COMPUTATION_AND_INTERNAL_AUDIT_COMPLETE_PENDING_EXTERNAL_REVIEW','original_run':c['original_run'],'original_plan_digest':c['original_plan_digest'],'original_failed_job':c['original_failed_job'],'users':n,'metric_keys':KEYS,'user_axis_sha256':axis,'best_selection':load(root/'training/best_selection.json'),'conditions':{k:{'metrics':metrics[k]['metrics'],'sid_statistics':metrics[k]['sid_statistics'],'collision_catalog':metrics[k]['collision_catalog'],'groups':groups[k]} for k in metrics},'paired':{k:{'metrics':v['metrics'],'changes':v['changes'],'bootstrap':v['bootstrap']} for k,v in paired.items()},'audit':{'status':'PASS','tolerance':tol,'condition_audits':audits,'pair_audits':pair_audit,'maximum_condition_difference':maxerr,'maximum_official_CCE_difference':cceerr,'data_assets':assets,'NONAUTHOR_REVIEW':'PENDING','EXTERNAL_GPT_FINAL_REVIEW':'PENDING'},'limits':['Conditional on fixed Beauty data/model/seed/shuffle realization.','User bootstrap does not estimate across-training-seed or across-permutation variation.','Performance decrease alone does not establish hierarchical SID semantics.','S4 versus S3 has no dedicated paired significance test here.','One training seed cannot establish cross-seed robustness.','No Instruments population is included.','Subgroup means are descriptive; overall CIs are not subgroup CIs.']}
 write(out/'beauty_supplemental_final_summary.json',summary);return summary

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);a=ap.parse_args();c=load(a.config);root=Path(c['original_root']);out=Path(c['supplement_root']);plan=load(root/'plan.json');started=time.monotonic();status='FAILED';code=1
 need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('PYTHONOPTIMIZE')=='0' and __debug__,'CPU/assertion environment');need(os.environ.get('GRID_JOB_ID')==c['job_id'],'Runner job binding');need(c['request_timeout_seconds']+c['cleanup_reserve_seconds']==c['cpu_budget_seconds']==1800,'budget binding')
 sys.path.insert(0,str(Path(plan['release'])/'lc_rec/a0_beauty'))
 protected=load(out/'protected-inputs.json')
 try:
  for row in protected:need(sha(row['path'])==row['sha256'],'protected original changed before compute: '+row['path'])
  argv=[plan['python'],str(Path(plan['release'])/'lc_rec/a0_beauty/paired_beauty.py'),'--normal-primary',str(root/'attempts/normal_eval/a0/primary'),'--shuffle-primary',str(root/'attempts/s4_eval/a0/primary'),'--normal-test',str(root/'attempts/normal/a0/test'),'--shuffle-test',str(root/'attempts/s4/a0/test'),'--output',str(out/'s4'),'--training-root',str(root/'training'),'--training-run-commit',c['original_run'],'--training-config-sha256',c['config_sha256']]
  write(out/'actual-argv.json',{'argv':argv,'cwd':str(out),'python':sys.executable,'env':{k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','PYTHONOPTIMIZE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')},'deployment':load(Path(__file__).with_name('deployment.json'))})
  event('fixed_s4_cli_begin');t=time.monotonic();subprocess.run(argv,check=True);event('fixed_s4_cli_complete',elapsed_seconds=time.monotonic()-t)
  audit(c,root,out,plan)
  for row in protected:need(sha(row['path'])==row['sha256'],'protected original changed after compute: '+row['path'])
  write(out/'original-preservation.json',{'status':'PASS','all_original_protected_hashes_unchanged':True,'protected_files':len(protected),'original_plan_unchanged':True,'original_ledger_unchanged':True,'original_failed_attempt_preserved':True})
  outputs=[identity(x) for x in out.rglob('*') if x.is_file() and x.name not in ('output-manifest.json','budget.json') and 'original-evidence' not in x.parts]
  write(out/'output-manifest.json',outputs);status='PASS';code=0
 finally:
  elapsed=time.monotonic()-started;u=resource.getrusage(resource.RUSAGE_SELF);children=resource.getrusage(resource.RUSAGE_CHILDREN)
  write(out/'budget.json',{'status':status,'elapsed_seconds':elapsed,'cpu_user_system_seconds_including_children':u.ru_utime+u.ru_stime+children.ru_utime+children.ru_stime,'wall_budget_seconds':1800,'request_limit_seconds':1680,'cleanup_reserve_seconds':120,'new_gpu_compute':0,'original_cpu_seconds':load(out/'prelaunch.json')['original_cpu_charged_seconds'],'combined_cpu_wall_seconds_before_runner_cleanup':load(out/'prelaunch.json')['original_cpu_charged_seconds']+elapsed,'peak_rss_self_bytes':u.ru_maxrss*1024,'finished_utc':dt.datetime.now(dt.timezone.utc).isoformat()});event('worker_complete',status=status,elapsed_seconds=elapsed)
 return code
if __name__=='__main__':sys.exit(main())
