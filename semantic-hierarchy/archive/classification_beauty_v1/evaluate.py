"""CPU evaluation after frozen text-only labels have been produced."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import time

import numpy as np

from common import roots, sha, read_jsonl, write_json, write_jsonl
from run_stage import read_complete_labels


def contingency(group_ids, label_ids, groups, classes):
    return np.bincount(group_ids * classes + label_ids,
                       minlength=groups * classes).reshape(groups, classes)


def scores(counts):
    sizes = counts.sum(axis=1)
    total = int(sizes.sum())
    pairs = int((sizes * (sizes - 1) // 2).sum())
    majority = counts.max(axis=1)
    return {
        'weighted_purity': float(majority.sum()/total) if total else None,
        'macro_purity': float((majority[sizes > 0]/sizes[sizes > 0]).mean()) if total else None,
        'pair_agreement': float((counts * (counts-1)//2).sum()/pairs) if pairs else None,
        'evaluated_items': total, 'evaluated_pairs': pairs,
        'groups_with_known_items': int((sizes > 0).sum()),
        'groups_with_evaluable_pairs': int((sizes >= 2).sum())}


def compare_null(actual, samples):
    if actual is None:
        return {'actual': None, 'reason': 'No evaluable items or pairs'}
    values = np.array([s for s in samples if s is not None], dtype=float)
    if not len(values):
        return {'actual': actual, 'reason': 'No evaluable null samples'}
    mean = float(values.mean())
    return {'actual': actual, 'null_mean': mean,
            'null_95pct_interval': np.quantile(values, [.025, .975]).tolist(),
            'absolute_gain': actual-mean, 'ratio_to_null': actual/mean if mean > 0 else None,
            'permutation_p_one_sided': float((1 + (values >= actual-1e-12).sum())/(len(values)+1))}


def bh_adjust(records):
    # Predeclared family: all shared groups, known labels, both granularities,
    # both datasets and all three lengths, for the two main metrics (24 tests).
    family = []
    for record in records:
        if record['selection'] == 'all_shared' and record['label_view'] == 'known_only':
            for metric in ['weighted_purity', 'pair_agreement']:
                result = record['null_comparison'][metric]
                if 'permutation_p_one_sided' in result:
                    family.append(result)
    ordered = sorted(family, key=lambda x: x['permutation_p_one_sided'])
    best = 1.
    for index in range(len(ordered)-1, -1, -1):
        best = min(best, ordered[index]['permutation_p_one_sided']*len(ordered)/(index+1))
        ordered[index]['bh_q_primary_family'] = best


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute', action='store_true')
    args = ap.parse_args()
    if not args.execute:
        ap.error('Evaluation requires --execute after user confirmation and final labels')
    cfg, root, prep, out = roots()
    output = out / 'evaluation'
    output.mkdir(parents=True, exist_ok=True)
    audit = json.loads((prep / 'data_preparation_verified.json').read_text())
    records = []
    provenance = {'config_sha256': sha(Path(__file__).with_name('config.json')), 'datasets': {}}
    for dataset, spec in cfg['datasets'].items():
        n = spec['expected_items']
        taxonomy_path = out / dataset / 'taxonomy_frozen.json'
        taxonomy = json.loads(taxonomy_path.read_text())
        lookup = {c['id']: c for c in taxonomy['categories']}
        lookup['UNKNOWN'] = {'id': 'UNKNOWN', 'name': 'UNKNOWN', 'coarse_id': 'UNKNOWN', 'coarse_name': 'UNKNOWN'}
        labels = read_complete_labels(out / dataset, 'final', n)
        complete = json.loads(next((out / dataset).glob('final_shard*of*.complete.json')).read_text())
        assert complete['taxonomy_sha256'] == sha(taxonomy_path)
        assert complete['items_sha256'] == audit['datasets'][dataset]['classifier_items_sha256']
        assert complete['config_sha256'] == provenance['config_sha256']
        assert sha(prep / dataset / 'classifier_items.jsonl') == complete['items_sha256']
        texts = {r['item_id']: r['text'] for r in read_jsonl(prep / dataset / 'classifier_items.jsonl')}
        assert all(r['category_id'] in lookup for r in labels)
        unknown = np.array([r['category_id'] == 'UNKNOWN' for r in labels])
        provenance['datasets'][dataset] = {'taxonomy_sha256': sha(taxonomy_path),
            'unknown_items': int(unknown.sum()), 'needs_review_items': sum(r['needs_review'] for r in labels),
            'label_files_sha256': {p.name: sha(p) for p in (out / dataset).glob('final_shard*of*.jsonl')}}
        for length in cfg['prefix_lengths']:
            group_path = prep / dataset / f'prefix_groups_l{length}.jsonl'
            ranked = read_jsonl(group_path)
            # Prefix membership is revalidated against the exact original SID file.
            import torch
            sid_path = Path(audit['datasets'][dataset]['sid_path'])
            assert sha(sid_path) == audit['datasets'][dataset]['sid_sha256']
            sids = torch.load(sid_path, map_location='cpu', weights_only=True).T.tolist()
            seen = []
            for group in ranked:
                assert group['count'] == len(group['item_ids'])
                assert all(sids[i][:length] == group['prefix'] for i in group['item_ids'])
                seen.extend(group['item_ids'])
            assert sorted(seen) == list(range(n))
            assert ranked == sorted(ranked, key=lambda r: (-r['count'], r['prefix']))
            group_ids = np.empty(n, dtype=np.int64)
            for g, row in enumerate(ranked):
                group_ids[row['item_ids']] = g
            original_sizes = np.array([r['count'] for r in ranked])
            selections = {'all_shared': np.flatnonzero(original_sizes >= 2),
                          'all_min5': np.flatnonzero(original_sizes >= 5)}
            selections.update({f'top{k}_shared': np.flatnonzero(original_sizes >= 2)[:k]
                               for k in cfg['top_k']})
            for granularity in ['fine', 'coarse']:
                names = [r['category_id'] if granularity == 'fine' else lookup[r['category_id']]['coarse_id']
                         for r in labels]
                group_detail = []
                for row in ranked:
                    if row['count'] < 2:
                        continue
                    distribution = Counter(names[i] for i in row['item_ids'])
                    valid_counts = Counter({c: v for c, v in distribution.items() if c != 'UNKNOWN'})
                    known = sum(valid_counts.values())
                    group_detail.append(dict(row, category_counts=dict(distribution),
                        known_items=known, known_purity=max(valid_counts.values())/known if known else None,
                        representative_texts=[{'item_id': i, 'text': texts[i], 'category_id': names[i]}
                                              for i in row['item_ids'][:5]]))
                write_jsonl(output / dataset / f'groups_l{length}_{granularity}.jsonl', group_detail)
                for view in ['known_only', 'including_unknown']:
                    active = ~unknown if view == 'known_only' else np.ones(n, dtype=bool)
                    active_names = np.array(names)[active]
                    if not len(active_names):
                        raise ValueError('All products are UNKNOWN; labels are not an adequate reference')
                    categories, codes = np.unique(active_names, return_inverse=True)
                    groups_active = group_ids[active]
                    actual_table = contingency(groups_active, codes, len(ranked), len(categories))
                    nulls = {key: {m: [] for m in ['weighted_purity', 'pair_agreement']} for key in selections}
                    rng = np.random.default_rng(cfg['seed'])
                    for repeat in range(cfg['permutations']):
                        shuffled = rng.permutation(codes)
                        table = contingency(groups_active, shuffled, len(ranked), len(categories))
                        for key, selected in selections.items():
                            measured = scores(table[selected])
                            for metric in nulls[key]:
                                nulls[key][metric].append(measured[metric])
                    global_counts = np.bincount(codes)
                    global_pairs = len(codes)*(len(codes)-1)//2
                    random_pair = float((global_counts*(global_counts-1)//2).sum()/global_pairs) if global_pairs else None
                    for key, selected in selections.items():
                        measured = scores(actual_table[selected])
                        records.append({'dataset': dataset, 'prefix_length': length,
                            'granularity': granularity, 'label_view': view, 'selection': key,
                            'selected_groups': len(selected), 'original_items': int(original_sizes[selected].sum()),
                            'coverage_all_items': float(original_sizes[selected].sum()/n),
                            'evaluated_coverage_all_items': measured['evaluated_items']/n,
                            'metrics': measured, 'global_random_pair_agreement': random_pair,
                            'null_comparison': {m: compare_null(measured[m], nulls[key][m]) for m in nulls[key]}})
                    print(dataset, length, granularity, view, 'permutations_complete', flush=True)
    bh_adjust(records)
    write_json(output / 'results.json', {'records': records, 'provenance': provenance,
        'permutations': cfg['permutations'], 'created_unix': time.time(),
        'null_interval_is_not_effect_confidence_interval': True})
    fields = ['dataset', 'prefix_length', 'granularity', 'label_view', 'selection', 'selected_groups',
              'coverage_all_items', 'evaluated_coverage_all_items', 'weighted_purity', 'pair_agreement',
              'purity_null_mean', 'pair_null_mean', 'purity_gain', 'pair_gain']
    with (output / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            purity, pair = r['null_comparison']['weighted_purity'], r['null_comparison']['pair_agreement']
            writer.writerow({**{k: r[k] for k in fields[:8]},
                'weighted_purity': r['metrics']['weighted_purity'], 'pair_agreement': r['metrics']['pair_agreement'],
                'purity_null_mean': purity.get('null_mean'), 'pair_null_mean': pair.get('null_mean'),
                'purity_gain': purity.get('absolute_gain'), 'pair_gain': pair.get('absolute_gain')})


if __name__ == '__main__':
    main()
