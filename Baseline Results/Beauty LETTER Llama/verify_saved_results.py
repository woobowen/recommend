"""CPU-only candidate-level metric and permutation recomputation; no model inference."""
from pathlib import Path
import argparse,collections,hashlib,importlib.util,json,math,csv
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);args=p.parse_args();root=args.root
 meta=root/'metadata' if (root/'metadata').exists() else root
 pred=root/'predictions' if (root/'predictions').exists() else root
 evaluator=module('letter_eval',meta/'vendor/LETTER/LETTER-LC-Rec/evaluate.py')
 protocol=module('grid_shuffle',meta/'run/runner/shuffle_grid_protocol.py')
 interactions=json.loads((meta/'vendor/LETTER/data/Beauty/Beauty.inter.json').read_text());indices=json.loads((meta/'vendor/LETTER/data/Beauty/Beauty.index.json').read_text())
 users=list(interactions);valid={''.join(s) for s in indices.values()};counts=collections.Counter(''.join(s) for s in indices.values())
 summary=json.loads((meta/'run/Beauty/testing/evaluation_result.json').read_text());assert len(users)==22363
 checks={};table=[]
 for mode,n in [('unshuffle',0),('shuffle_3',3),('shuffle_4',4)]:
  sums=collections.Counter();done=changed=history=validn=ambiguous=0
  for f in sorted((pred/'run/Beauty/testing'/mode).glob('part-*.json')):
   for r in json.loads(f.read_text()):
    u=users[done];h=interactions[u][:-1][-20:]
    assert r['row']==done and r['user_id']==u and r['target']==''.join(indices[str(interactions[u][-1])])
    assert len(r['scores'])==len(r['candidates'])==20 and all(math.isfinite(s) for s in r['scores'])
    metrics=evaluator.get_metrics_results(evaluator.get_topk_results(r['candidates'],r['scores'],[r['target']],20,all_items=None),['hit@5','hit@10','ndcg@5','ndcg@10'])
    assert all(abs(metrics[k]-r['metrics'][k])<1e-12 for k in metrics)
    expected=[list(protocol.permutation(n,int(u),j,'beauty',2026))+list(range(n,4)) if n else list(range(4)) for j in range(len(h))]
    assert r['permutations']==expected and r['history_items']==len(h)
    c=sum(v!=[0,1,2,3] for v in expected);assert c==r['changed_items']
    v=sum(s in valid for s in r['candidates']);assert v==r['valid_candidates']
    if n==0:assert r['original_input_sha256']==r['input_sha256']
    for k,vv in metrics.items():sums[k]+=vv
    done+=1;changed+=c;history+=len(h);validn+=v;ambiguous+=counts[r['target']]>1
  assert done==22363
  m={k.replace('hit','recall'):v/done for k,v in sums.items()}
  raw=summary['results'][mode]
  assert all(abs(m[k]-raw['metrics'][k])<1e-12 for k in m)
  assert changed==raw['changed_history_items'] and history==raw['total_history_items'] and ambiguous==raw['ambiguous_sid_test_targets']
  assert abs(validn/(20*done)-raw['valid_candidate_fraction'])<1e-12
  checks[mode]=dict(users=done,candidate_metrics_exact=True,targets_and_user_order_exact=True,permutations_replayed=True,changed_items=changed,ambiguous_targets=ambiguous,metrics=m)
  table.append(dict(mode=mode,**m))
 out=dict(passed=True,scope='All 67089 saved user-condition rows: candidates+scores to author metrics, official target identities, shuffle permutations. CPU recomputation only; no fresh model reevaluation.',results=checks)
 (root/'cpu_verification.json').write_text(json.dumps(out,indent=2)+'\n')
 keys=['recall@5','recall@10','ndcg@5','ndcg@10'];base=table[0]
 with (root/'metrics.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['mode']+keys+[k+'_change_pct' for k in keys])
  for r in table:w.writerow([r['mode']]+[r[k] for k in keys]+[(r[k]/base[k]-1)*100 for k in keys])
 lines=['| 模式 | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |','|---|---:|---:|---:|---:|']
 for r in table:lines.append('| '+r['mode']+' | '+' | '.join(f'{r[k]:.6f}'+('' if r is base else f' ({(r[k]/base[k]-1)*100:+.2f}%)') for k in keys)+' |')
 (root/'metrics.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print(json.dumps(out),flush=True)
if __name__=='__main__':main()
