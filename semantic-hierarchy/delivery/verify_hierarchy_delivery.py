"""Recompute labels, verify SID-derived groups, and inspect saved vector receipts.
Does not load embedding matrices or recompute cosine similarities.
"""
import collections
import hashlib
import json
from pathlib import Path


def load(p): return json.loads(p.read_text(encoding='utf-8'))
def lines(p): return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]


def main():
    here = Path(__file__).resolve().parent; cpu = load(here/'cpu_verification.json')
    report = {}
    for ds in ['beauty', 'toys']:
        source = here.parent/'results/classification'/ds
        path = source/'final_shard0of1.jsonl'; labels = lines(path)
        old = cpu['classification'][ds]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == old['labels_sha256']
        assert len(labels) == len({r['item_id'] for r in labels}) == old['items']
        assert [r['item_id'] for r in labels] == list(range(len(labels)))
        assert sum(r['category_id']=='UNKNOWN' for r in labels) == old['unknown']
        assert hashlib.sha256((source/'taxonomy_frozen.json').read_bytes()).hexdigest() == old['taxonomy_sha256']
        tax = {r['id']: r for r in load(source/'taxonomy_frozen.json')['categories']}
        sid_rows = lines(here/'classification_inputs'/ds/'item_sids.jsonl')
        sid_map = {r['item_id']: r['sid'] for r in sid_rows}
        assert len(sid_rows) == len(sid_map) and set(sid_map) == set(range(len(labels))), 'SID item axis mismatch'
        group_inputs = []
        checks = []
        for length in [1,2,3]:
            groups = lines(here/'classification_inputs'/ds/f'prefix_groups_l{length}.jsonl')
            assert sorted(i for g in groups for i in g['item_ids']) == list(range(len(labels)))
            expected_groups = collections.defaultdict(list)
            for item_id, sid in sid_map.items(): expected_groups[tuple(sid[:length])].append(item_id)
            actual_groups = {tuple(g['prefix']): sorted(g['item_ids']) for g in groups}
            assert len(actual_groups) == len(groups), 'Duplicate prefix group'
            assert actual_groups == {k: sorted(v) for k,v in expected_groups.items()}, 'Prefix membership differs from SID'
            assert all(g['count'] == len(g['item_ids']) and len(g['prefix']) == length for g in groups), 'Group count/depth mismatch'
            group_inputs.append({'depth':length,'groups':len(groups),'group_sha256':hashlib.sha256((here/'classification_inputs'/ds/f'prefix_groups_l{length}.jsonl').read_bytes()).hexdigest()})
            for granularity in ['fine','coarse']:
                num=den=pn=pd=0
                for g in groups:
                    if g['count']<2: continue
                    names=[labels[i]['category_id'] for i in g['item_ids'] if labels[i]['category_id']!='UNKNOWN']
                    if granularity=='coarse': names=[tax[c]['coarse_id'] for c in names]
                    hist=collections.Counter(names); n=len(names)
                    num+=max(hist.values(),default=0);den+=n
                    pn+=sum(v*(v-1)//2 for v in hist.values());pd+=n*(n-1)//2
                values={'weighted_purity':num/den, 'pair_agreement':pn/pd if pd else None}
                expected=next(r for r in old['checks'] if r['prefix_length']==length and r['granularity']==granularity)
                for k,v in values.items(): assert v==expected[k] or v is not None and abs(v-expected[k])<1e-12
                checks.append({'prefix_length':length,'granularity':granularity,**values})
        report[ds]={'items':len(labels),'unknown':old['unknown'],'recomputed_main_rows':checks,'sid_sha256':hashlib.sha256((here/'classification_inputs'/ds/'item_sids.jsonl').read_bytes()).hexdigest(),'sid_group_checks':group_inputs}
    vectors=load(here/'vector_index.json'); assert len(vectors)==6
    assert all(r['ids_unique_and_complete'] and r['max_norm_error']<1e-5 and r['max_cosine_error']<1e-9 for r in vectors)
    print(json.dumps({'classification':report,'checks':{'label_metrics':{'status':'recomputed','rows':12},'group_source':{'status':'verified_from_sid','datasets':2,'depths':[1,2,3]},'vector_receipts':{'status':'saved_receipts_checked','records':6,'source':'vector_index.json'},'direct_vector_cosines':{'status':'not_run','reason':'This command does not load embedding matrices; use scripts/evaluate_embeddings.py'}},'gpu_work':False},indent=2))


if __name__ == '__main__': main()
