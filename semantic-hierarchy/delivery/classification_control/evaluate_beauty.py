"""Evaluate completed Beauty labels without touching the active pipeline or its outputs."""
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path('/home/jcy/GRID_wmh_Qwen_TIGER_20261009')
SCIENCE = ROOT / 'codex/sid_prefix_semantics_v1'
sys.path.insert(0, str(SCIENCE))
import numpy as np
from common import roots, sha, read_jsonl, write_json
from run_stage import read_complete_labels
from evaluate import contingency, scores, compare_null


def main():
    cfg, root, prep, out = roots()
    dataset = 'beauty'
    n = cfg['datasets'][dataset]['expected_items']
    audit = json.loads((prep / 'data_preparation_verified.json').read_text())['datasets'][dataset]
    labels = read_complete_labels(out / dataset, 'final', n)
    receipt = json.loads((out / dataset / 'final_shard0of1.complete.json').read_text())
    taxonomy_path = out / dataset / 'taxonomy_frozen.json'
    taxonomy = json.loads(taxonomy_path.read_text())
    assert receipt['taxonomy_sha256'] == sha(taxonomy_path)
    assert receipt['items_sha256'] == audit['classifier_items_sha256'] == sha(prep / dataset / 'classifier_items.jsonl')
    assert receipt['config_sha256'] == sha(SCIENCE / 'config.json')
    assert receipt['code_sha256']['evaluate.py'] == sha(SCIENCE / 'evaluate.py')
    lookup = {c['id']: c for c in taxonomy['categories']}
    lookup['UNKNOWN'] = {'coarse_id': 'UNKNOWN'}
    assert all(r['category_id'] in lookup for r in labels)
    unknown = np.array([r['category_id'] == 'UNKNOWN' for r in labels])
    import torch
    assert sha(audit['sid_path']) == audit['sid_sha256']
    sids = torch.load(audit['sid_path'], map_location='cpu', weights_only=True).T.tolist()
    records = []
    for length in cfg['prefix_lengths']:
        ranked = read_jsonl(prep / dataset / f'prefix_groups_l{length}.jsonl')
        assert ranked == sorted(ranked, key=lambda r: (-r['count'], r['prefix']))
        seen = []
        group_ids = np.empty(n, dtype=np.int64)
        for g, row in enumerate(ranked):
            assert row['count'] == len(row['item_ids'])
            assert all(sids[i][:length] == row['prefix'] for i in row['item_ids'])
            seen.extend(row['item_ids'])
            group_ids[row['item_ids']] = g
        assert sorted(seen) == list(range(n))
        sizes = np.array([r['count'] for r in ranked])
        selections = {'all_shared': np.flatnonzero(sizes >= 2), 'all_min5': np.flatnonzero(sizes >= 5)}
        selections.update({f'top{k}_shared': np.flatnonzero(sizes >= 2)[:k] for k in cfg['top_k']})
        for granularity in ['fine', 'coarse']:
            names = [r['category_id'] if granularity == 'fine' else lookup[r['category_id']]['coarse_id'] for r in labels]
            for view in ['known_only', 'including_unknown']:
                active = ~unknown if view == 'known_only' else np.ones(n, dtype=bool)
                categories, codes = np.unique(np.array(names)[active], return_inverse=True)
                assert len(codes)
                groups_active = group_ids[active]
                table = contingency(groups_active, codes, len(ranked), len(categories))
                nulls = {k: {m: [] for m in ['weighted_purity', 'pair_agreement']} for k in selections}
                rng = np.random.default_rng(cfg['seed'])
                for repeat in range(cfg['permutations']):
                    shuffled = rng.permutation(codes)
                    null_table = contingency(groups_active, shuffled, len(ranked), len(categories))
                    for key, selected in selections.items():
                        result = scores(null_table[selected])
                        for metric in nulls[key]:
                            nulls[key][metric].append(result[metric])
                global_counts = np.bincount(codes)
                global_pairs = len(codes) * (len(codes) - 1) // 2
                random_pair = float((global_counts * (global_counts - 1) // 2).sum() / global_pairs)
                for key, selected in selections.items():
                    measured = scores(table[selected])
                    # Independently verify the observed combinatorial metrics with Python integers.
                    counts = table[selected].tolist()
                    total = sum(sum(row) for row in counts)
                    pairs = sum(sum(row) * (sum(row) - 1) // 2 for row in counts)
                    numerator = sum(v * (v - 1) // 2 for row in counts for v in row)
                    assert np.isclose(measured['weighted_purity'], sum(max(row) for row in counts) / total)
                    assert np.isclose(measured['pair_agreement'], numerator / pairs)
                    records.append({'dataset': dataset, 'prefix_length': length, 'granularity': granularity,
                        'label_view': view, 'selection': key, 'selected_groups': len(selected),
                        'original_items': int(sizes[selected].sum()), 'coverage_all_items': float(sizes[selected].sum() / n),
                        'evaluated_coverage_all_items': measured['evaluated_items'] / n, 'metrics': measured,
                        'global_random_pair_agreement': random_pair,
                        'null_comparison': {m: compare_null(measured[m], nulls[key][m]) for m in nulls[key]}})
                print(length, granularity, view, 'complete', flush=True)
    target = out / 'evaluation_beauty_standalone_20261110'
    report = {'records': records, 'permutations': cfg['permutations'], 'created_unix': time.time(),
        'created_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'null_interval_is_not_effect_confidence_interval': True,
        'bh_q_status': 'Deferred until the prespecified 24-test family includes both datasets; no partial-family adjustment.',
        'provenance': {'dataset': dataset, 'total_items': n, 'known_items': int((~unknown).sum()),
            'unknown_items': int(unknown.sum()), 'needs_review_items': sum(r['needs_review'] for r in labels),
            'config_sha256': sha(SCIENCE / 'config.json'), 'taxonomy_sha256': sha(taxonomy_path),
            'labels_sha256': receipt['sha256'], 'sid_sha256': audit['sid_sha256'],
            'metrics_source_sha256': sha(SCIENCE / 'evaluate.py'), 'standalone_script_sha256': sha(__file__),
            'fine_categories': len(taxonomy['categories']), 'coarse_categories': len(taxonomy['coarse_categories']),
            'human_accuracy_measured': False}}
    write_json(target / 'results.json', report)
    lines = ['# Beauty SID prefix evaluation', '', f"Created: {report['created_at']}", '',
        f"Items: {n}; known: {int((~unknown).sum())}; UNKNOWN: {int(unknown.sum())}.", '',
        'Primary metrics exclude UNKNOWN. Prefix selection uses original distinct item counts, before label exclusion.', '',
        'Purity/pair gain is observed minus the mean of 1,000 label permutations, in percentage points.', '',
        '|Selection|Granularity|Prefix length|Groups|Covered items|Coverage|Evaluated coverage|Purity|Null purity|Purity gain (pp)|Pair agreement|Null pair|Pair gain (pp)|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in records:
        if row['label_view'] != 'known_only':
            continue
        p = row['null_comparison']['weighted_purity']; a = row['null_comparison']['pair_agreement']
        lines.append(f"|{row['selection']}|{row['granularity']}|{row['prefix_length']}|{row['selected_groups']}|{row['original_items']}|{100*row['coverage_all_items']:.2f}%|{100*row['evaluated_coverage_all_items']:.2f}%|{100*p['actual']:.2f}%|{100*p['null_mean']:.2f}%|{100*p['absolute_gain']:.2f}|{100*a['actual']:.2f}%|{100*a['null_mean']:.2f}%|{100*a['absolute_gain']:.2f}|")
    lines += ['', report['bh_q_status'], '', 'The reference labels are Llama-generated product categories, not human-verified ground truth.']
    (target / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('OUTPUT', target, flush=True)
    for row in records:
        if row['label_view'] == 'known_only' and row['selection'] in ['all_shared', 'top20_shared']:
            print(json.dumps(row, ensure_ascii=False), flush=True)
    print('PROVENANCE', json.dumps(report['provenance']), flush=True)


if __name__ == '__main__':
    main()
