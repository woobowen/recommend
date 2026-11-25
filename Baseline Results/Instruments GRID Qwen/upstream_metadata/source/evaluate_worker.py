"""Associate completed text embeddings with frozen SID tables and evaluate."""
import os
os.environ.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
                  CUDA_VISIBLE_DEVICES='')
import argparse
from pathlib import Path
import time
import numpy as np
from semantic_metrics import TreeMetrics, ConditionalControls
from runtime_common import read_json, read_jsonl, sha, status, write_json, write_jsonl


def value(v):
    return float(v) if np.isfinite(v) else None


def write_plots(out, tree, cosines, nodes, paths):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 140})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].boxplot([cosines[d][tree.counts[d] >= 2] for d in [1, 2, 3]], tick_labels=['1', '2', '3'], showfliers=False)
    axes[0].set(xlabel='Shared prefix length', ylabel='Mean pairwise cosine', title='All evaluable prefix nodes')
    for depth, label in [(2, '1 to 2'), (3, '2 to 3')]:
        mask = tree.edge_masks[depth]
        delta = cosines[depth][mask] - cosines[depth - 1][tree.parents[depth][mask]]
        axes[1].hist(delta, bins=40, alpha=.55, density=True, label=label)
    axes[1].axvline(0, color='black', linewidth=.8)
    axes[1].set(xlabel='Child minus parent cosine', ylabel='Density', title='Proper refinements'); axes[1].legend()
    ranked = sorted(paths, key=lambda row: (-row['counts'][2], row['prefix']))[:20]
    for row in ranked:
        axes[2].plot([1, 2, 3], row['cosines'], alpha=.55, linewidth=1)
    axes[2].set(xticks=[1, 2, 3], xlabel='Shared prefix length', ylabel='Mean pairwise cosine', title='Top 20 three-code prefixes')
    fig.tight_layout(); fig.savefig(out / 'semantic_refinement.png'); plt.close(fig)


def evaluate_variant(run, dataset, variant, cfg):
    directory = run / dataset / variant
    out = directory / 'evaluation'
    out.mkdir(exist_ok=True)
    state_path = run / dataset / 'evaluation_status.json'
    manifest = read_json(directory / 'encoding_complete.json')
    assert manifest['complete']
    assert manifest['embedding_sha256'] == sha(directory / 'embeddings_normalized.npy')
    spec = cfg['datasets'][dataset]
    assert sha(spec['sid_path']) == spec['sid_sha256']
    rows = read_jsonl(spec['sid_path'])
    ids = np.load(directory / 'item_ids.npy')
    assert np.array_equal(ids, [row['item_id'] for row in rows])
    codes = np.array([row['sid'][:3] for row in rows], dtype=np.int64)
    vectors = np.load(directory / 'embeddings_normalized.npy', mmap_mode='r')
    assert len(vectors) == spec['items'] == len(rows)
    tree = TreeMetrics(codes, cfg['epsilon'])
    status(state_path, 'observed_metrics', dataset=dataset, variant=variant)
    cosines = tree.observed(vectors)
    observed = tree.statistics(cosines)
    nodes, edges, paths = [], [], []
    for depth in [1, 2, 3]:
        for index, prefix in enumerate(tree.prefixes[depth]):
            nodes.append({'prefix': list(prefix), 'depth': depth, 'items': int(tree.counts[depth][index]),
                          'mean_pairwise_cosine': value(cosines[depth][index])})
    nodes.sort(key=lambda row: (-row['items'], row['depth'], row['prefix']))
    for rank, node in enumerate(nodes, 1):
        node['global_frequency_rank'] = rank
    for depth in [2, 3]:
        for child, prefix in enumerate(tree.prefixes[depth]):
            parent = tree.parents[depth][child]
            parent_count = int(tree.counts[depth - 1][parent])
            child_count = int(tree.counts[depth][child])
            delta = value(cosines[depth][child] - cosines[depth - 1][parent])
            edges.append({'parent_prefix': list(tree.prefixes[depth - 1][parent]),
                          'child_prefix': list(prefix), 'transition': f'{depth-1}_to_{depth}',
                          'parent_items': parent_count, 'child_items': child_count,
                          'parent_cosine': value(cosines[depth - 1][parent]),
                          'child_cosine': value(cosines[depth][child]), 'delta': delta,
                          'eligible_proper_refinement': bool(tree.edge_masks[depth][child]),
                          'same_membership': parent_count == child_count})
    for third, prefix in enumerate(tree.prefixes[3]):
        if tree.counts[3][third] < 2:
            continue
        second = tree.parents[3][third]; first = tree.parents[2][second]
        scores = [float(cosines[1][first]), float(cosines[2][second]), float(cosines[3][third])]
        paths.append({'prefix': list(prefix),
                      'counts': [int(tree.counts[1][first]), int(tree.counts[2][second]), int(tree.counts[3][third])],
                      'cosines': scores, 'delta_12': scores[1]-scores[0], 'delta_23': scores[2]-scores[1],
                      'proper_both_edges': bool(tree.path_mask[third]),
                      'monotonic_as_image': bool(scores[1]-scores[0] > cfg['epsilon'] and scores[2]-scores[1] >= -cfg['epsilon'])})
    paths.sort(key=lambda row: (-row['counts'][2], row['prefix']))
    write_jsonl(out / 'prefix_nodes.jsonl', nodes)
    write_jsonl(out / 'parent_child_edges.jsonl', edges)
    write_jsonl(out / 'complete_paths.jsonl', paths)
    write_jsonl(out / 'top20_three_code_paths.jsonl', paths[:20])
    gaps = []
    for depth in [2, 3]:
        gap = tree.gap_values(cosines, depth)
        for parent in np.flatnonzero(np.isfinite(gap)):
            gaps.append({'parent_prefix': list(tree.prefixes[depth-1][parent]),
                         'transition': f'{depth-1}_to_{depth}', 'gap': float(gap[parent])})
    write_jsonl(out / 'branch_gaps.jsonl', gaps)
    coverage = {str(depth): {'all_nodes': len(tree.groups[depth]),
                             'evaluable_nodes': int((tree.counts[depth] >= 2).sum()),
                             'singleton_nodes': int((tree.counts[depth] == 1).sum()),
                             'covered_distinct_items': int(tree.counts[depth][tree.counts[depth] >= 2].sum()),
                             'coverage': float(tree.counts[depth][tree.counts[depth] >= 2].sum() / len(rows)),
                             'maximum_node_items': int(tree.counts[depth].max())}
                for depth in [1, 2, 3]}
    intervals = tree.cluster_intervals(cosines, np.random.default_rng(cfg['seed']+77), cfg['bootstrap_repetitions'])
    edge_summary = {}
    for depth in [2, 3]:
        mask = tree.edge_masks[depth]
        delta = cosines[depth][mask] - cosines[depth-1][tree.parents[depth][mask]]
        edge_summary[f'{depth-1}_to_{depth}'] = {
            'eligible_edges': int(mask.sum()), 'median_delta': float(np.median(delta)),
            'positive_edges': int((delta > cfg['epsilon']).sum()),
            'ties': int((np.abs(delta) <= cfg['epsilon']).sum()),
            'negative_edges': int((delta < -cfg['epsilon']).sum()),
            'same_membership_edges': int((tree.counts[depth] == tree.counts[depth-1][tree.parents[depth]]).sum()),
            'item_weighted_delta': float(np.average(delta, weights=tree.counts[depth][mask]))}
    summary = {'dataset': dataset, 'variant': variant, 'items': len(rows),
               'primary_evidence': 'first_to_second_prefix',
               'primary_metric': 'mean_pairwise_cosine', 'observed': observed,
               'cluster_bootstrap_95ci': intervals, 'coverage': coverage,
               'edge_summary': edge_summary,
               'shared_three_code_paths': len(paths), 'proper_complete_paths': int(tree.path_mask.sum()),
               'node_macro_cosines': {str(d): float(cosines[d][tree.counts[d]>=2].mean()) for d in [1,2,3]},
               'path_condition': 'S1 < S2 <= S3, epsilon=1e-6; main denominator requires proper subsets',
               'embedding_sha256': manifest['embedding_sha256'], 'sid_sha256': spec['sid_sha256'],
               'threshold_sensitivity_runs': False}
    write_json(out / 'observed_summary.json', summary)
    status(state_path, 'building_parent_gram_cache', dataset=dataset, variant=variant)
    controls = ConditionalControls(tree, vectors)
    permutation_records, comparisons = [], {}
    started = time.monotonic()
    for parent_depth in [1, 2]:
        rng = np.random.default_rng(cfg['seed'] + parent_depth)
        samples = {key: [] for key in observed}
        for iteration in range(cfg['permutations']):
            mapping = controls.permute(parent_depth, rng)
            randomized = controls.cosines(mapping, cosines, parent_depth+1)
            statistics = tree.statistics(randomized)
            for key in samples:
                samples[key].append(statistics[key])
            permutation_records.append({'parent_depth': parent_depth, 'iteration': iteration, **statistics})
            if iteration % 100 == 0:
                status(state_path, 'permutations', dataset=dataset, variant=variant,
                       parent_depth=parent_depth, completed=iteration, total=cfg['permutations'],
                       elapsed_seconds=time.monotonic()-started)
        comparison = {}
        for key, values in samples.items():
            if observed[key] is None:
                continue
            array = np.asarray(values)
            comparison[key] = {'observed': observed[key], 'null_mean': float(array.mean()),
                               'observed_minus_null_mean': float(observed[key]-array.mean()),
                               'null_reference_95percent': np.quantile(array, [.025,.975]).tolist(),
                               'p_one_sided': float((1 + (array >= observed[key]).sum()) / (len(array)+1))}
        comparisons[f'within_depth_{parent_depth}'] = comparison
    summary['random_controls'] = comparisons
    summary['permutations_per_control'] = cfg['permutations']
    summary['permutation_seconds'] = time.monotonic()-started
    summary['main_p_value'] = comparisons['within_depth_1']['delta_12']['p_one_sided']
    summary['secondary_p_value'] = comparisons['within_depth_2']['delta_23']['p_one_sided']
    write_jsonl(out / 'permutation_statistics.jsonl', permutation_records)
    write_json(out / 'summary.json', summary)
    write_plots(out, tree, cosines, nodes, paths)
    write_json(out / 'evaluation_complete.json', {'complete': True, 'finished_unix': time.time(),
                                                'summary_sha256': sha(out / 'summary.json')})
    status(state_path, 'variant_complete', dataset=dataset, variant=variant)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run', required=True)
    ap.add_argument('--dataset', choices=['instrument'], required=True)
    args = ap.parse_args(); run = Path(args.run)
    cfg = read_json(run / 'analysis_config.json')
    try:
        for variant in cfg['variants']:
            evaluate_variant(run, args.dataset, variant, cfg)
        status(run / args.dataset / 'evaluation_status.json', 'complete', dataset=args.dataset)
    except Exception as exc:
        status(run / args.dataset / 'evaluation_status.json', 'failed', dataset=args.dataset,
               error_type=type(exc).__name__, error=str(exc)[:1000])
        raise


if __name__ == '__main__':
    main()
