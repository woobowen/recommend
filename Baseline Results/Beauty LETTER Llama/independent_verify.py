"""Independent read-only verification of saved Beauty candidate rows; no GPU/model load."""
from pathlib import Path
import hashlib,importlib.util,json,math,collections,datetime
ROOT=Path('/home/jcy/GRID_wmh_Qwen_TIGER_20261009')
RUN=ROOT/'experiments/letter_beauty_seqrec_batch64_20261122_155523'
AUTHOR=ROOT/'codex/letter_lc_rec_smoke_20261121/vendor/LETTER'
def load(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def module(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
evaluate=module('independent_author_evaluate',AUTHOR/'LETTER-LC-Rec/evaluate.py')
protocol=module('independent_grid_protocol',RUN/'runner/shuffle_grid_protocol.py')
config=load(RUN/'Beauty/training/config.json');trained=load(RUN/'Beauty/training/training_result.json')
result=load(RUN/'Beauty/testing/evaluation_result.json');launch=load(RUN/'launch.json')
assert all(sha(RUN/p)==h for p,h in launch['source_sha256'].items())
assert load(RUN/'Beauty/progress.json')['complete']
adapter=Path(trained['best_checkpoint'])/'adapter_model.safetensors';assert sha(adapter)==trained['best_adapter_sha256']==result['checkpoint_sha256']
inputs=AUTHOR/'data/Beauty';indices=load(inputs/'Beauty.index.json');inters=load(inputs/'Beauty.inter.json');users=list(inters)
assert len(users)==22363
items={''.join(c) for c in indices.values()};counts=collections.Counter(tuple(c) for c in indices.values())
ambiguous=sum(counts[tuple(indices[str(inters[u][-1])])]>1 for u in users)
state=load(adapter.parent/'trainer_state.json')
evals=[(x['step'],x['eval_loss']) for x in state['log_history'] if 'eval_loss' in x]
assert min(v for s,v in evals)==trained['best_eval_loss']
keys=['hit@5','hit@10','ndcg@5','ndcg@10'];rows_by_mode={};checks={}
for mode,n in [('unshuffle',0),('shuffle_3',3),('shuffle_4',4)]:
 out=RUN/'Beauty/testing';folder=out/mode;summary=result['results'][mode]
 assert sha(inputs/'Beauty.index.json')==summary['indices_sha256']
 assert sha(inputs/'Beauty.inter.json')==summary['interactions_sha256']
 assert sha(RUN/'runner/test_recommendation.py')==summary['evaluator_source_sha256']
 sums={k:0. for k in keys};done=changed=history=valid=dupes=0;hashes=[]
 for p in sorted(folder.glob('part-*.json')):
  hashes.append(dict(path=p.name,sha256=sha(p)))
  for row in load(p):
   u=users[done];assert row['row']==done and row['user_id']==u
   assert row['target']==''.join(indices[str(inters[u][-1])])
   assert len(row['candidates'])==len(row['scores'])==20
   assert all(math.isfinite(s) for s in row['scores'])
   metrics=evaluate.get_metrics_results(evaluate.get_topk_results(row['candidates'],row['scores'],[row['target']],20,all_items=None),keys)
   assert metrics==row['metrics']
   expected_history=len(inters[u][:-1][-20:]);assert row['history_items']==expected_history
   expected=[list(protocol.permutation(n,int(u),i,'beauty',2026))+list(range(n,4)) if n else [0,1,2,3] for i in range(expected_history)]
   assert expected==row['permutations']
   changed_here=sum(p!=[0,1,2,3] for p in expected);assert changed_here==row['changed_items']
   valid_here=sum(c in items for c in row['candidates']);assert valid_here==row['valid_candidates']
   if n==0:assert row['original_input_sha256']==row['input_sha256']
   elif changed_here==0:assert row['original_input_sha256']==row['input_sha256']
   else:assert row['original_input_sha256']!=row['input_sha256']
   for k in keys:sums[k]+=metrics[k]
   dupes+=len(row['candidates'])-len(set(row['candidates']));history+=expected_history;changed+=changed_here;valid+=valid_here;done+=1
 assert done==22363
 for k in keys:
  key=k.replace('hit@','recall@');assert abs(sums[k]/done-summary['metrics'][key])<1e-14
 assert changed==summary['changed_history_items'] and history==summary['total_history_items']
 assert valid/(20*done)==summary['valid_candidate_fraction']
 assert summary['ambiguous_sid_test_targets']==ambiguous
 checks[mode]=dict(rows=done,metric_recompute_all_rows=True,targets_and_user_order=True,permutations_match_protocol=True,changed_items=changed,total_history=history,duplicate_candidate_extra=dupes,part_count=len(hashes),part_hashes=hashes)
report=dict(time_beijing=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),passed=True,
 run=str(RUN),read_only=True,no_model_load_no_gpu=True,launch_sources_verified=len(launch['source_sha256']),
 adapter_sha256=trained['best_adapter_sha256'],best_checkpoint=trained['best_checkpoint'],best_eval_loss=trained['best_eval_loss'],
 selected_checkpoint_eval_history=evals,items=len(indices),unique_full_sids=len(items),users=len(users),ambiguous_targets=ambiguous,
 config=config,conditions=checks,limits=['Independent CPU recomputation of saved candidates, not model reevaluation.',
 'Prompt token arrays were not saved; input hashes checked for within-row consistency, not regenerated from full prompts.',
 'Publication remote bytes and fetch/reproduction tools have not yet been reviewed.'])
print(json.dumps(report,indent=2))
