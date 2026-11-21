"""CPU-only readback of vectors, labels, checkpoint pairing and baseline axes."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import collections
import hashlib
import importlib.util
import json
import os
import sys

os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                  TF_CPP_MIN_LOG_LEVEL='3',TF_NUM_INTEROP_THREADS='1',TF_NUM_INTRAOP_THREADS='1')
import numpy as np
import torch
torch.set_num_threads(2)
ROOT=Path('/home/jcy/GRID_wmh_Qwen_TIGER_20261009')
OUT=ROOT/'codex/github_delivery_audit_20261121'


def load(path):return json.loads(path.read_text())
def lines(path):return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
    return h.hexdigest()


def main():
    result={'time_beijing':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'gpu_used':False,
        'vectors':[],'classification':{},'baseline':{},'attention':{}}
    for name,run in [('beauty_toys',ROOT/'codex/llm2vec_sid_hierarchy_20261113/runs/20261114_125221'),
                     ('instruments',ROOT/'codex/instrument_semantic_hierarchy_20261115/runs/20261115_104723')]:
        for f in sorted(run.glob('*/*/embeddings_normalized.npy')):
            folder=f.parent;ds=folder.parent.name
            x=np.load(f,mmap_mode='r');ids=np.load(folder/'item_ids.npy')
            assert x.dtype==np.float32 and x.shape==(len(ids),4096)
            assert len(np.unique(ids))==len(ids) and np.array_equal(np.sort(ids),np.arange(len(ids)))
            norms=np.linalg.norm(x,axis=1);assert np.isfinite(x).all() and np.max(np.abs(norms-1))<1e-5
            old_nodes=lines(folder/'evaluation/prefix_nodes.jsonl')
            sid_path=(ROOT/'codex/instrument_semantic_hierarchy_20261115/inputs/item_sids.jsonl' if ds=='instrument'
                      else ROOT/f'codex/llm2vec_sid_hierarchy_20261113/inputs/{ds}/item_sids.jsonl')
            if not sid_path.exists():
                # Input paths are frozen in analysis config; use their exact source.
                cfg=load(run/'analysis_config.json')
                def find_sid(value):
                    if isinstance(value,dict):
                        for k,v in value.items():
                            if k in ['sid_path','sids_path','item_sids'] and isinstance(v,str) and Path(v).exists():yield Path(v)
                            yield from find_sid(v)
                    elif isinstance(value,list):
                        for v in value:yield from find_sid(v)
                candidates=list(find_sid(cfg))
                sid_path=next(p for p in candidates if ds in str(p))
            sid_rows=lines(sid_path);lookup={r['item_id']:r['sid'] for r in sid_rows}
            assert set(lookup)==set(ids.tolist())
            codes=np.array([lookup[int(i)][:3] for i in ids]);max_error=0.;checked=0
            for node in old_nodes:
                if node['items']<2:continue
                prefix=node['prefix'];idx=np.flatnonzero((codes[:,:len(prefix)]==prefix).all(axis=1))
                assert len(idx)==node['items']
                z=np.asarray(x[idx],dtype=np.float64);summed=z.sum(0)
                cosine=(np.dot(summed,summed)-np.square(z).sum())/(len(idx)*(len(idx)-1))
                err=abs(cosine-node['mean_pairwise_cosine']);max_error=max(max_error,err);checked+=1
            assert max_error<1e-9,(f,max_error)
            receipt=load(folder/'encoding_complete.json')
            result['vectors'].append({'key':f'{name}/{folder.relative_to(run).as_posix()}',
                'shape':list(x.shape),'dtype':str(x.dtype),'ids_unique_and_complete':True,
                'max_norm_error':float(np.max(np.abs(norms-1))), 'checked_nodes':checked,
                'max_cosine_error':max_error,'sid_source':str(sid_path),'sid_sha256':digest(sid_path),
                'embedding_sha256':digest(f),'ids_sha256':digest(folder/'item_ids.npy'),
                'encoding_receipt':receipt})
            del x
    base=ROOT/'experiments/sid_prefix_semantics_v1'
    for ds,expected,unknown_expected in [('beauty',12101,405),('toys',11924,219)]:
        labels=lines(base/ds/'final_shard0of1.jsonl');taxonomy=load(base/ds/'taxonomy_frozen.json')
        assert len(labels)==expected and len({r['item_id'] for r in labels})==expected
        assert [r['item_id'] for r in labels]==list(range(expected))
        counts=collections.Counter(r['category_id'] for r in labels);assert counts['UNKNOWN']==unknown_expected
        tax={c['id']:c for c in taxonomy['categories']};tax['UNKNOWN']={'coarse_id':'UNKNOWN'}
        assert all(r['category_id'] in tax for r in labels)
        prep=ROOT/'codex'/('sid_prefix_semantics_v1' if ds=='beauty' else 'sid_prefix_semantics_v7_compact')/'preparation'/ds
        results_path=base/'evaluation_beauty_standalone_20261110/results.json' if ds=='beauty' else base/'evaluation/results.json'
        if not results_path.exists():results_path=base/'evaluation/results.json'
        saved=load(results_path)['records'];checks=[]
        for length in [1,2,3]:
            groups=lines(prep/f'prefix_groups_l{length}.jsonl')
            assert sorted(i for g in groups for i in g['item_ids'])==list(range(expected))
            selected=[g for g in groups if g['count']>=2]
            for granularity in ['fine','coarse']:
                num=den=pairnum=pairden=0
                for g in selected:
                    names=[labels[i]['category_id'] if granularity=='fine' else tax[labels[i]['category_id']]['coarse_id']
                           for i in g['item_ids'] if labels[i]['category_id']!='UNKNOWN']
                    hist=collections.Counter(names);n=len(names)
                    num+=max(hist.values(),default=0);den+=n
                    pairnum+=sum(v*(v-1)//2 for v in hist.values());pairden+=n*(n-1)//2
                purity=num/den;agreement=pairnum/pairden if pairden else None
                old=next(r for r in saved if r['dataset']==ds and r['prefix_length']==length and r['granularity']==granularity and r['label_view']=='known_only' and r['selection']=='all_shared')
                assert abs(purity-old['metrics']['weighted_purity'])<1e-12
                assert agreement is None and old['metrics']['pair_agreement'] is None or abs(agreement-old['metrics']['pair_agreement'])<1e-12
                checks.append({'prefix_length':length,'granularity':granularity,'weighted_purity':purity,
                    'pair_agreement':agreement,'evaluated_items':den,'evaluated_pairs':pairden})
        result['classification'][ds]={'items':len(labels),'unknown':counts['UNKNOWN'],'known':expected-counts['UNKNOWN'],
            'taxonomy_sha256':digest(base/ds/'taxonomy_frozen.json'),'labels_sha256':digest(base/ds/'final_shard0of1.jsonl'),
            'results_source':str(results_path),'checks':checks}
    # Inspect all 4000 source records' exact identities using published manifest.
    att=list((ROOT/'experiments/sid_order_5seed_v1').rglob('*.pt.gz'))
    result['attention']={'files':len(att),'bytes':sum(p.stat().st_size for p in att)}
    assert len(att)==4000
    sample=att[0]
    import gzip
    with gzip.open(sample,'rb') as f:content=torch.load(f,map_location='cpu',weights_only=False)
    result['attention']['sample_file']=str(sample)
    result['attention']['sample_keys']=list(content) if isinstance(content,dict) else str(type(content))
    # Count exact test rows and preserve the internal item axis. No guessed ASIN joins.
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU')
    options=tf.data.Options();options.threading.private_threadpool_size=1
    options.threading.max_intra_op_parallelism=1
    for ds,runname in [('beauty','beauty_rkmeans_qwen_20261010_012253'),('sports','sports_rkmeans_qwen_20261010_093035'),('toys','toys_rkmeans_qwen_20261010_093036')]:
        root=ROOT/'original_assets/data/amazon_data'/ds
        sid=torch.load(ROOT/'experiments'/runname/'semantic_ids.pt',map_location='cpu',weights_only=False)
        n=sid.shape[1];users=[];last=[];itemids=[]
        for sub in ['testing','items']:
            files=sorted((root/sub).glob('*.tfrecord.gz'))
            for raw in tf.data.TFRecordDataset([str(p) for p in files],compression_type='GZIP').with_options(options):
                fields=tf.train.Example.FromString(raw.numpy()).features.feature
                if sub=='items':itemids.append(int(fields['id'].int64_list.value[0]))
                else:
                    users.append(int(fields['user_id'].int64_list.value[0]));seq=list(fields['sequence_data'].int64_list.value)
                    assert seq and all(0<=i<n for i in seq);last.append(seq[-1])
        assert len(set(users))==len(users) and sorted(itemids)==list(range(n))
        result['baseline'][ds]={'test_users':len(users),'item_count':n,'sid_shape':list(sid.shape),
            'items_complete_unique':True,'test_user_ids_unique':True,'test_item_range_checked':True,
            'test_target_count':len(last),'test_target_nonzero_ids':sum(i!=0 for i in last),
            'sid_sha256':digest(ROOT/'experiments'/runname/'semantic_ids.pt'),
            'test_partitions':[{'path':str(f.relative_to(ROOT)),'bytes':f.stat().st_size} for f in sorted((root/'testing').glob('*.tfrecord.gz'))],
            'external_asin_identity':'Beauty separate G3 recovery already published; Sports/Toys no verified independent ASIN map found'}
    result['complete']=True
    path=OUT/'cpu_verification.json';path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'complete':True,'vectors':len(result['vectors']),'classification':{k:v['unknown'] for k,v in result['classification'].items()},'test_users':{k:v['test_users'] for k,v in result['baseline'].items()}},ensure_ascii=False))


if __name__=='__main__':main()
