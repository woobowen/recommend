"""Evaluate both frozen SID trees against the same independently encoded vectors."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
from pathlib import Path
import argparse,time
import numpy as np
from runtime_common import read_json,read_jsonl,write_json,write_jsonl,sha,status
from semantic_metrics import TreeMetrics,ConditionalControls,holm_adjust

def evaluate(run,domain,method,variant,mask,z,codes,permutations):
 out=run/'analysis'/domain/method/variant;out.mkdir(parents=True)
 vectors=z[mask];sid=codes[mask,:3];ids=np.flatnonzero(mask)
 tree=TreeMetrics(sid);cos=tree.observed(vectors);observed=tree.statistics(cos)
 nodes=[];edges=[];paths=[];gaps=[]
 for d in [1,2,3]:
  for j,p in enumerate(tree.prefixes[d]):
   nodes.append(dict(depth=d,prefix=p,items=int(tree.counts[d][j]),mean_pairwise_cosine=float(cos[d][j]) if np.isfinite(cos[d][j]) else None))
  if d>1:
   for j,p in enumerate(tree.prefixes[d]):
    parent=tree.parents[d][j];delta=cos[d][j]-cos[d-1][parent]
    edges.append(dict(transition=f'{d-1}_to_{d}',prefix=p,parent=p[:-1],child_items=int(tree.counts[d][j]),parent_items=int(tree.counts[d-1][parent]),proper=bool(tree.edge_masks[d][j]),delta=float(delta) if np.isfinite(delta) else None))
   g=tree.gap_values(cos,d)
   gaps.extend(dict(transition=f'{d-1}_to_{d}',parent=tree.prefixes[d-1][j],gap=float(g[j])) for j in np.flatnonzero(np.isfinite(g)))
 for j in np.flatnonzero(tree.counts[3]>=2):
  second=tree.parents[3][j];first=tree.parents[2][second];vals=[float(cos[1][first]),float(cos[2][second]),float(cos[3][j])]
  paths.append(dict(prefix=tree.prefixes[3][j],counts=[int(tree.counts[1][first]),int(tree.counts[2][second]),int(tree.counts[3][j])],cosines=vals,
    proper_both_edges=bool(tree.path_mask[j]),monotonic=bool(vals[1]-vals[0]>1e-6 and vals[2]-vals[1]>=-1e-6),item_ids=ids[tree.groups[3][j]].tolist()))
 nodes.sort(key=lambda r:(-r['items'],r['depth'],r['prefix']))
 for rank,r in enumerate(nodes,1):r['global_frequency_rank']=rank
 paths.sort(key=lambda r:(-r['counts'][2],r['prefix']))
 write_jsonl(out/'prefix_nodes.jsonl',nodes);write_jsonl(out/'parent_child_edges.jsonl',edges);write_jsonl(out/'complete_paths.jsonl',paths)
 write_jsonl(out/'top20_proper_paths.jsonl',[r for r in paths if r['proper_both_edges']][:20]);write_jsonl(out/'branch_gaps.jsonl',gaps)
 coverage={str(d):dict(nodes=len(tree.groups[d]),evaluable_nodes=int((tree.counts[d]>=2).sum()),singletons=int((tree.counts[d]==1).sum()),covered_items=int(tree.counts[d][tree.counts[d]>=2].sum()),coverage=float(tree.counts[d][tree.counts[d]>=2].sum()/len(ids))) for d in [1,2,3]}
 # Check the sum identity against an explicit pair matrix on real evaluable nodes.
 checked=0
 for d in [1,2,3]:
  small=[j for j,g in enumerate(tree.groups[d]) if 2<=len(g)<=50][:3]
  for j in small:
   b=vectors[tree.groups[d][j]].astype(np.float64);gram=b@b.T
   assert abs(gram[np.triu_indices(len(b),1)].mean()-cos[d][j])<1e-10;checked+=1
 summary=dict(dataset=domain,method=method,variant=variant,items=len(ids),original_items=len(mask),selected_item_ids_sha256=None,
   node_macro_cosines={str(d):float(cos[d][tree.counts[d]>=2].mean()) for d in [1,2,3]},observed=observed,coverage=coverage,
   proper_edges={f'{d-1}_to_{d}':int(tree.edge_masks[d].sum()) for d in [2,3]},proper_paths=int(tree.path_mask.sum()),
   cluster_bootstrap_95ci=tree.cluster_intervals(cos,np.random.default_rng(20261113+77),2000),explicit_pairwise_nodes_checked=checked,
   same_common_vectors_and_masks=True,threshold_sensitivity_runs=False)
 np.save(out/'selected_item_ids.npy',ids);summary['selected_item_ids_sha256']=sha(out/'selected_item_ids.npy')
 write_json(out/'observed_summary.json',summary)
 if permutations:
  controls=ConditionalControls(tree,vectors);records=[];comparisons={}
  for depth in [1,2]:
   rng=np.random.default_rng(20261113+depth);samples=[]
   for it in range(999):
    stat=tree.statistics(controls.cosines(controls.permute(depth,rng),cos,depth+1));samples.append(stat);records.append(dict(parent_depth=depth,iteration=it,**stat))
    if it%100==0:status(run/'analysis_status.json','permutations',dataset=domain,method=method,variant=variant,parent_depth=depth,completed=it,total=999)
   keys=['delta_12','positive_12','gap_12','path_monotonic'] if depth==1 else ['delta_23','positive_23','gap_23','path_monotonic']
   comparisons[str(depth)]={}
   for k in keys:
    if observed[k] is None:continue
    a=np.array([r[k] for r in samples]);comparisons[str(depth)][k]=dict(observed=observed[k],null_mean=float(a.mean()),p_one_sided=float((1+np.sum(a>=observed[k]))/1000),null_95percent=np.quantile(a,[.025,.975]).tolist())
  summary['conditional_controls']=comparisons;summary['primary_p_value']=comparisons['1']['delta_12']['p_one_sided']
  write_jsonl(out/'permutation_statistics.jsonl',records)
 write_json(out/'summary.json',summary);write_json(out/'evaluation_complete.json',dict(complete=True,summary_sha256=sha(out/'summary.json')))
 return summary

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run;results=[]
 for domain in ['beauty','instruments']:
  folder=run/domain;manifest=read_json(folder/'encoding_complete.json');assert manifest['complete']
  for n,h in manifest['outputs'].items():assert sha(folder/n)==h
  masks=np.load(folder/'field_valid.npy');both=masks.all(axis=1);anyfield=masks.any(axis=1)
  fused=np.load(folder/'fused_normalized.npy',mmap_mode='r');title=np.load(folder/'title_normalized.npy',mmap_mode='r')
  t=np.load(folder/'title_raw.npy',mmap_mode='r');c=np.load(folder/'categories_raw.npy',mmap_mode='r')
  for i in np.flatnonzero(anyfield):
   raw=(t[i]+c[i])/masks[i].sum();expected=raw/np.linalg.norm(raw.astype(np.float64))
   assert np.max(abs(expected-fused[i]))<1e-6
  for method in ['qwen','letter']:
   records=read_jsonl(run/'inputs'/domain/(method+'_sids.jsonl'));assert [r['item_id'] for r in records]==list(range(len(fused)))
   assert sha(run/'inputs'/domain/(method+'_sids.jsonl'))==read_json(run/'input_audit.json')['domains'][domain]['sid_sha256'][method]
   codes=np.array([r['sid'] for r in records])
   for variant,mask,z in [('fusion_complete',both,fused),('fusion_all_available',anyfield,fused),('title_complete',both,title)]:
    status(run/'analysis_status.json','observed',dataset=domain,method=method,variant=variant)
    results.append(evaluate(run,domain,method,variant,mask,z,codes,variant=='fusion_complete'))
 primary=[r for r in results if r['variant']=='fusion_complete']
 adjusted=holm_adjust({r['dataset']+'_'+r['method']:r['primary_p_value'] for r in primary})
 for r in primary:r['holm_p_value']=adjusted[r['dataset']+'_'+r['method']]
 write_json(run/'all_results.json',dict(complete=True,primary=primary,all_variants=results,primary_holm_family=list(adjusted),primary_holm=adjusted))
 lines=['| Dataset | SID | n | S1 | S2 | S3 | Δ12 | Δ23 | positive12 | positive23 | monotonic | Gap12 | Gap23 | L3 coverage | Holm p12 |',
 '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 def f(v):return 'NA' if v is None else f'{v:.6f}'
 for r in primary:
  obs=r['observed'];vals=[r['node_macro_cosines'][str(d)] for d in [1,2,3]]+[obs[k] for k in ['delta_12','delta_23','positive_12','positive_23','path_monotonic','gap_12','gap_23']]+[r['coverage']['3']['coverage'],r['holm_p_value']]
  lines.append('| '+r['dataset']+' | '+r['method']+' | '+str(r['items'])+' | '+' | '.join(map(f,vals))+' |')
 (run/'指标.md').write_text('\n'.join(lines)+'\n',encoding='utf8');status(run/'analysis_status.json','complete',analyses=len(results))
if __name__=='__main__':main()
