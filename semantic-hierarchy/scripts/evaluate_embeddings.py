"""Portable CPU entry point; reuse the unchanged historical metric implementation."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('OMP_NUM_THREADS', '4')
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / 'archive' / 'llm2vec_beauty_toys'))
from semantic_metrics import ConditionalControls, TreeMetrics


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value, lines=False):
    if lines:
        data = ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in value)
    else:
        data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    path.write_text(data, encoding='utf-8')


def finite(value):
    return float(value) if np.isfinite(value) else None


def evaluate(vectors, codes, seed=20261113, permutations=999, bootstrap=2000):
    tree = TreeMetrics(codes, epsilon=1e-6)
    cosines = tree.observed(vectors)
    observed = tree.statistics(cosines)
    nodes, edges, paths, gaps = [], [], [], []
    coverage = {}
    for d in (1, 2, 3):
        valid = tree.counts[d] >= 2
        coverage[str(d)] = {
            'all_nodes': len(valid), 'evaluable_nodes': int(valid.sum()),
            'covered_distinct_items': int(tree.counts[d][valid].sum()),
            'coverage': float(tree.counts[d][valid].sum() / len(codes))}
        for i, prefix in enumerate(tree.prefixes[d]):
            nodes.append({'prefix': list(prefix), 'depth': d, 'items': int(tree.counts[d][i]),
                          'mean_pairwise_cosine': finite(cosines[d][i])})
    nodes.sort(key=lambda x: (-x['items'], x['depth'], x['prefix']))
    for rank, node in enumerate(nodes, 1):
        node['global_frequency_rank'] = rank
    for d in (2, 3):
        for i, prefix in enumerate(tree.prefixes[d]):
            parent = tree.parents[d][i]
            edges.append({'parent_prefix': list(tree.prefixes[d - 1][parent]),
                          'child_prefix': list(prefix), 'transition': f'{d-1}_to_{d}',
                          'parent_items': int(tree.counts[d-1][parent]),
                          'child_items': int(tree.counts[d][i]),
                          'parent_cosine': finite(cosines[d-1][parent]),
                          'child_cosine': finite(cosines[d][i]),
                          'delta': finite(cosines[d][i] - cosines[d-1][parent]),
                          'eligible_proper_refinement': bool(tree.edge_masks[d][i]),
                          'same_membership': bool(tree.counts[d][i] == tree.counts[d-1][parent])})
        values = tree.gap_values(cosines, d)
        for i in np.flatnonzero(np.isfinite(values)):
            gaps.append({'parent_prefix': list(tree.prefixes[d-1][i]),
                         'transition': f'{d-1}_to_{d}', 'gap': float(values[i])})
    for i, prefix in enumerate(tree.prefixes[3]):
        if tree.counts[3][i] < 2:
            continue
        second = tree.parents[3][i]; first = tree.parents[2][second]
        scores = [float(cosines[1][first]), float(cosines[2][second]), float(cosines[3][i])]
        paths.append({'prefix': list(prefix),
                      'counts': [int(tree.counts[1][first]), int(tree.counts[2][second]), int(tree.counts[3][i])],
                      'cosines': scores, 'proper_both_edges': bool(tree.path_mask[i]),
                      'monotonic_as_image': bool(scores[1] - scores[0] > tree.eps and
                                                  scores[2] - scores[1] >= -tree.eps)})
    paths.sort(key=lambda p: (-p['counts'][2], p['prefix']))
    intervals = tree.cluster_intervals(cosines, np.random.default_rng(seed + 77), bootstrap)
    controls = ConditionalControls(tree, vectors)
    comparisons, records = {}, []
    for parent_depth in (1, 2):
        rng = np.random.default_rng(seed + parent_depth)
        samples = {key: [] for key, value in observed.items() if value is not None}
        for iteration in range(permutations):
            mapping = controls.permute(parent_depth, rng)
            stats = tree.statistics(controls.cosines(mapping, cosines, parent_depth + 1))
            records.append({'parent_depth': parent_depth, 'iteration': iteration, **stats})
            for key in samples:
                samples[key].append(stats[key])
        comparison = {}
        for key, values in samples.items():
            values = np.asarray(values)
            comparison[key] = {
                'observed': observed[key], 'null_mean': float(values.mean()),
                'observed_minus_null_mean': float(observed[key] - values.mean()),
                'null_reference_95percent': np.quantile(values, [.025, .975]).tolist(),
                'p_one_sided': float((1 + (values >= observed[key]).sum()) / (permutations + 1))}
        comparisons[f'within_depth_{parent_depth}'] = comparison
    summary = {
        'items': len(codes), 'observed': observed, 'coverage': coverage,
        'node_macro_cosines': {str(d): finite(cosines[d][tree.counts[d] >= 2].mean())
                               if np.any(tree.counts[d] >= 2) else None for d in (1, 2, 3)},
        'proper_complete_paths': int(tree.path_mask.sum()),
        'shared_three_code_paths': len(paths), 'cluster_bootstrap_95ci': intervals,
        'random_controls': comparisons, 'permutations_per_control': permutations,
        'seed': seed, 'bootstrap_repetitions': bootstrap, 'minimum_group_size': 2,
        'epsilon': tree.eps, 'path_condition': 'S1<S2<=S3 with proper subsets',
        'primary_evidence': 'first_to_second_prefix',
        'main_p_value': comparisons['within_depth_1'].get('delta_12', {}).get('p_one_sided'),
        'holm_adjustment': 'Not applied automatically; declare the primary test family first.'}
    return summary, {'prefix_nodes': nodes, 'parent_child_edges': edges, 'complete_paths': paths,
                     'top20_three_code_paths': paths[:20], 'branch_gaps': gaps,
                     'permutation_statistics': records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--embeddings', type=Path, required=True, help='Unit-normalized N x D .npy')
    parser.add_argument('--item-ids', type=Path, required=True, help='N item IDs in embedding-row order')
    parser.add_argument('--sids', type=Path, required=True, help='JSONL rows: item_id, sid (>=3 integers)')
    parser.add_argument('--output', type=Path, required=True, help='New directory; never overwrites saved results')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--variant', choices=['title', 'title_brand'], required=True)
    parser.add_argument('--seed', type=int, default=20261113)
    parser.add_argument('--permutations', type=int, default=999)
    parser.add_argument('--bootstrap', type=int, default=2000)
    args = parser.parse_args()
    if args.permutations < 1 or args.bootstrap < 1:
        parser.error('permutations and bootstrap must be positive')
    vectors = np.load(args.embeddings, mmap_mode='r', allow_pickle=False)
    ids = np.load(args.item_ids, allow_pickle=False)
    if vectors.ndim != 2 or len(vectors) < 2 or vectors.shape[1] == 0:
        raise ValueError('Expected at least two nonempty embedding rows')
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or len(ids) != len(vectors) or len(set(ids.tolist())) != len(ids):
        raise ValueError('Embedding IDs must be unique integers and match row count')
    if not np.issubdtype(vectors.dtype, np.floating) or not np.isfinite(vectors).all():
        raise ValueError('Embeddings must be finite real floating-point vectors')
    if np.max(np.abs(np.linalg.norm(vectors.astype(np.float64), axis=1) - 1)) >= 1e-5:
        raise ValueError('Expected pre-normalized unit vectors; no implicit normalization is performed')
    rows = [json.loads(line) for line in args.sids.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if any(type(row.get('item_id')) is not int or len(row.get('sid', [])) < 3 or
           any(type(x) is not int or x < 0 for x in row['sid']) for row in rows):
        raise ValueError('SID rows require integer item_id and >=3 nonnegative integer codes')
    sid_map = {row['item_id']: row['sid'][:3] for row in rows}
    if len(sid_map) != len(rows) or set(sid_map) != set(ids.tolist()):
        raise ValueError('SID IDs must be unique and exactly match the embedding ID set')
    codes = np.asarray([sid_map[int(i)] for i in ids], dtype=np.int64)
    args.output.mkdir(parents=True, exist_ok=False)
    summary, tables = evaluate(vectors, codes, args.seed, args.permutations, args.bootstrap)
    summary.update(dataset=args.dataset, variant=args.variant, provenance={
        'embedding_sha256': sha(args.embeddings), 'item_ids_sha256': sha(args.item_ids),
        'sid_sha256': sha(args.sids),
        'metric_source_sha256': sha(BASE / 'archive/llm2vec_beauty_toys/semantic_metrics.py'),
        'entry_point': 'portable publication evaluator; no encoding or training performed'})
    for name, table in tables.items():
        save(args.output / f'{name}.jsonl', table, lines=True)
    save(args.output / 'summary.json', summary)
    print(json.dumps({'output': str(args.output), 'observed': summary['observed'],
                      'coverage_3': summary['coverage']['3']['coverage']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
