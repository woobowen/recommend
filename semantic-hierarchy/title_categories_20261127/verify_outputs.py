"""Recompute saved-vector statistics on CPU, without running the encoder."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys
import numpy as np
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf8'))
def jl(p):return [json.loads(s) for s in p.read_text(encoding='utf8').splitlines()]
def compare(a,b,label):
 if isinstance(a,dict):
  for k,v in a.items():compare(v,b[k],label+'.'+k)
 elif isinstance(a,(list,tuple)):
  assert len(a)==len(b),label
  for i,(x,y) in enumerate(zip(a,b)):compare(x,y,label+str(i))
 elif a is None:assert b is None,label
 elif isinstance(a,(int,float,np.number)):assert abs(float(a)-float(b))<1e-10,(label,a,b)
 else:assert a==b,(label,a,b)
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--output',required=True,type=Path);args=p.parse_args();run=args.run
 sys.path.insert(0,str(run/'source'));from semantic_metrics import TreeMetrics,holm_adjust
 assert load(run/'status.json')['returncode']==0
 launch=load(run/'launch.json')
 for n,h in launch['source_sha256'].items():assert sha(run/'source'/n)==h,n
 results=load(run/'all_results.json');checks=[];node_count=edge_count=path_count=permutation_count=0
 for domain in ['beauty','instruments']:
  folder=run/domain;manifest=load(folder/'encoding_complete.json')
  for n,h in manifest['outputs'].items():assert sha(folder/n)==h,n
  maps=jl(run/'inputs'/domain/'item_text_map.jsonl');texts=jl(run/'inputs'/domain/'unique_texts.jsonl')
  t=np.load(folder/'title_raw.npy');c=np.load(folder/'categories_raw.npy');z=np.load(folder/'fused_normalized.npy');tz=np.load(folder/'title_normalized.npy');u=np.load(folder/'unique_raw.npy');masks=np.load(folder/'field_valid.npy');norms=np.load(folder/'field_norms.npy')
  assert [r['item_id'] for r in maps]==list(range(len(maps))) and z.shape==(len(maps),4096)
  for i,r in enumerate(maps):
   ti,ci=r['title_text_id'],r['categories_text_id'];valid=np.array([ti>=0,ci>=0]);assert np.array_equal(masks[i],valid)
   for idx,raw,label in [(ti,t[i],'Title'),(ci,c[i],'Categories')]:
    if idx>=0:assert texts[idx]['text']==label+': '+r[label.lower()] and np.array_equal(raw,u[idx])
    else:assert np.all(raw==0)
   mean=(t[i]+c[i])/max(1,valid.sum());length=np.linalg.norm(mean.astype(np.float64))
   assert np.isfinite(mean).all() and length>1e-12
   assert np.max(abs(z[i]-mean/length))<1e-6
   tn=np.linalg.norm(t[i].astype(np.float64));cn=np.linalg.norm(c[i].astype(np.float64))
   assert np.max(abs(norms[i]-[tn,cn,length]))<1e-6
   assert np.max(abs(tz[i]-(t[i]/tn if tn else t[i])))<1e-6
  assert np.isfinite(z).all() and np.max(abs(np.linalg.norm(z.astype(np.float64),axis=1)-1))<1e-5
  for method in ['qwen','letter']:
   codes=np.array([r['sid'] for r in jl(run/'inputs'/domain/(method+'_sids.jsonl'))])
   for variant,mask,vectors in [('fusion_complete',masks.all(1),z),('fusion_all_available',masks.any(1),z),('title_complete',masks.all(1),tz)]:
    out=run/'analysis'/domain/method/variant;stored=load(out/'summary.json')
    assert sha(out/'summary.json')==load(out/'evaluation_complete.json')['summary_sha256']
    assert sha(out/'selected_item_ids.npy')==stored['selected_item_ids_sha256']
    assert np.array_equal(np.load(out/'selected_item_ids.npy'),np.flatnonzero(mask))
    tree=TreeMetrics(codes[mask,:3]);cos=tree.observed(vectors[mask]);compare(tree.statistics(cos),stored['observed'],domain+method+variant)
    compare(tree.cluster_intervals(cos,np.random.default_rng(20261113+77),2000),stored['cluster_bootstrap_95ci'],'bootstrap')
    for depth in [1,2,3]:
     count=tree.counts[depth];compare(float(cos[depth][count>=2].mean()),stored['node_macro_cosines'][str(depth)],'layer')
     assert int(count[count>=2].sum())==stored['coverage'][str(depth)]['covered_items']
     compare(count[count>=2].sum()/mask.sum(),stored['coverage'][str(depth)]['coverage'],'coverage')
    for node in jl(out/'prefix_nodes.jsonl'):
     d=node['depth'];j=tree.prefixes[d].index(tuple(node['prefix']));assert node['items']==int(tree.counts[d][j])
     compare(node['mean_pairwise_cosine'],float(cos[d][j]) if np.isfinite(cos[d][j]) else None,'node');node_count+=1
    for edge in jl(out/'parent_child_edges.jsonl'):
     d=len(edge['prefix']);j=tree.prefixes[d].index(tuple(edge['prefix']));parent=tree.parents[d][j]
     assert edge['proper']==bool(tree.edge_masks[d][j]);value=cos[d][j]-cos[d-1][parent]
     compare(edge['delta'],float(value) if np.isfinite(value) else None,'edge');edge_count+=1
    for path in jl(out/'complete_paths.jsonl'):
     j=tree.prefixes[3].index(tuple(path['prefix']));b=tree.parents[3][j];a=tree.parents[2][b]
     compare(path['cosines'],[float(cos[1][a]),float(cos[2][b]),float(cos[3][j])],'path')
     assert path['proper_both_edges']==bool(tree.path_mask[j]);path_count+=1
    for r in jl(out/'branch_gaps.jsonl'):
     d=int(r['transition'][-1]);j=tree.prefixes[d-1].index(tuple(r['parent']));compare(r['gap'],float(tree.gap_values(cos,d)[j]),'gap')
    if variant=='fusion_complete':
     records=jl(out/'permutation_statistics.jsonl');assert len(records)==1998
     for depth,values in stored['conditional_controls'].items():
      rows=[r for r in records if r['parent_depth']==int(depth)];assert len(rows)==999 and [r['iteration'] for r in rows]==list(range(999))
      for k,v in values.items():
       samples=np.array([r[k] for r in rows]);compare(samples.mean(),v['null_mean'],'null_mean');compare(np.quantile(samples,[.025,.975]).tolist(),v['null_95percent'],'null_ci')
       compare((1+np.sum(samples>=stored['observed'][k]))/1000,v['p_one_sided'],'p')
     permutation_count+=len(records)
    top=next(r for r in results['all_variants'] if (r['dataset'],r['method'],r['variant'])==(domain,method,variant));compare(stored,top,'all_results')
    checks.append(dict(dataset=domain,method=method,variant=variant,items=int(mask.sum()),passed=True))
 pvals={r['dataset']+'_'+r['method']:r['primary_p_value'] for r in results['primary']};compare(holm_adjust(pvals),results['primary_holm'],'holm')
 report=dict(passed=True,scope='All saved-vector field expansion, fusion, normalization, 12 tree analyses, all node/edge/path/gap records, bootstrap95CI, saved permutation distributions and p/Holm. No model loading or fresh embedding generation. Does not rerun permutation RNG assignments.',frozen_sources=len(launch['source_sha256']),analyses=checks,node_records=node_count,edge_records=edge_count,path_records=path_count,permutation_records=permutation_count,review='self verification; prior genuine nonauthor launch reviews retained; no new independent reviewer in this side conversation')
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
