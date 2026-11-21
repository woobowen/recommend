#!/usr/bin/env python3
"""Recompute the complete Shuffle V2 table and all five paired comparisons offline.

Requires NumPy and the adjacent final-stats-independent.py. Evidence is read
only. Original prompts and model weights are unnecessary for this scoring audit.
"""
import argparse
import csv
from collections import Counter
from hashlib import sha256
import importlib.util
import json
from math import fsum
from pathlib import Path

import numpy as np


def independent_scorer():
    path = Path(__file__).with_name('final-stats-independent.py')
    spec = importlib.util.spec_from_file_location('independent_raw_scorer', path)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    return scorer


def load(path):
    return json.loads(path.read_bytes())


def csvrows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def export_scores(root, stats):
    inters = load(root/'Instruments.inter.json')
    index = load(root/'Instruments.index.json')
    assert len(inters) == stats.COUNT and len(index) == 9922
    user_axis = stats.jsonsha(list(inters))
    sid_for = {i: ''.join(parts) for i, parts in index.items()}
    sizes = Counter(sid_for.values())
    assert len(sizes) == 9897 and sum(n > 1 for n in sizes.values()) == 23
    targets = [sid_for[str(items[-1])] for items in inters.values()]
    dictionary = load(root/'beam_dictionary.json')
    sids = []
    for sid, raw_suffix, token_ids, eos_pos in dictionary:
        assert raw_suffix.strip().replace(' ', '') == sid
        assert len(token_ids) >= 5 and eos_pos == 4 and token_ids[4] == 2
        sids.append(sid)
    shape = (len(stats.MODES), stats.COUNT, 20)
    candidates = np.load(root/'beam_dictionary_indices.npy', allow_pickle=False)
    scores = np.load(root/'sequence_scores.npy', allow_pickle=False)
    ranks = np.load(root/'sorted_generation_order.npy', allow_pickle=False)
    relevance = np.load(root/'ranked_relevance.npy', allow_pickle=False)
    assert all(a.shape == shape for a in (candidates, scores, ranks, relevance))
    assert candidates.dtype == np.dtype('<u4') and scores.dtype == np.dtype('<f8')
    assert ranks.dtype == relevance.dtype == np.dtype('u1')
    assert candidates.min() >= 0 and candidates.max() < len(dictionary)
    assert np.isfinite(scores).all() and relevance.max() <= 1
    # Input index j is generation order. Stable sort preserves it for ties.
    assert np.array_equal(np.argsort(-scores, kind='stable', axis=2), ranks)
    ranked_ids = np.take_along_axis(candidates, ranks, axis=2)
    values = np.empty((*shape[:2], len(stats.KEYS)), dtype=np.float64)
    diagnostics = {}
    for c, mode in enumerate(stats.MODES):
        counts = Counter()
        for i, target in enumerate(targets):
            ranked = [sids[k] for k in ranked_ids[c, i]]
            assert [int(sid == target) for sid in ranked] == relevance[c, i].tolist()
            counts.update(invalid=sum(sid not in sizes for sid in ranked),
                          duplicates=20-len(set(ranked)),
                          colliding=sum(sizes.get(sid, 0) > 1 for sid in ranked))
            values[c, i] = stats.score(ranked, target, sizes)
        diagnostics[mode] = dict(counts)
    return values, user_axis, diagnostics


def verify_bootstrap(values, axis, result, stats):
    labels = list(stats.NAMES)
    summary = load(result/'summary.json')
    assert summary['status'] == 'PASS_V2_PAIRED' and summary['users'] == stats.COUNT
    assert summary['user_axis_sha256'] == axis and set(summary['comparisons']) == set(labels)
    spec = summary['bootstrap']
    assert (spec['rng'], spec['seed'], spec['replacement'], spec['replicates'], spec['percentile_method'],
            spec['bounds'], spec['multiple_comparison_adjustment']) == ('NumPy PCG64', 20261113, True, 5000, 'linear', [2.5,97.5], False)
    means = [[fsum(col)/stats.COUNT for col in matrix.T] for matrix in values]
    deltas = np.stack([values[b]-values[a] for a,b in stats.PAIRS])
    with np.load(result/'arrays.npz', allow_pickle=False) as data:
        assert data['keys'].tolist() == stats.KEYS and data['condition'].tolist() == list(stats.MODES)
        assert data['comparison'].tolist() == labels and np.array_equal(data['row_idx'], np.arange(stats.COUNT))
        assert data['user_axis_sha256'].item() == axis
        values_err = float(np.max(np.abs(data['per_user_values']-values)))
        deltas_err = float(np.max(np.abs(data['per_user_differences']-deltas)))
        expected_reps = data['bootstrap_difference_means']
    assert values_err < stats.TOL and deltas_err < stats.TOL
    assert expected_reps.shape == (len(labels), 5000, len(stats.KEYS))
    means_csv = csvrows(result/'metrics.csv')
    assert len(means_csv) == len(stats.MODES)
    max_point_error = 0.0
    for idx, mode in enumerate(stats.MODES):
        assert means_csv[idx]['condition'] == mode
        for col, key in enumerate(stats.KEYS):
            published = summary['metrics'][mode][key]
            assert published == float(means_csv[idx][key])
            max_point_error = max(max_point_error, abs(means[idx][col]-published))
    source = deltas.transpose(1,0,2).reshape(stats.COUNT,-1).copy()
    rng = np.random.Generator(np.random.PCG64(20261113))
    digest = sha256()
    computed = np.empty((5000,source.shape[1]), dtype=np.float64)
    # Independent reduction via one vector of user multiplicities per draw.
    for i in range(0,5000,20):
        indices = rng.integers(0,stats.COUNT,size=(min(20,5000-i),stats.COUNT),dtype=np.int64)
        digest.update(indices.astype('<i8',copy=False).tobytes())
        weights = np.asarray([np.bincount(row,minlength=stats.COUNT) for row in indices],dtype=np.float64)
        computed[i:i+len(weights)] = weights.dot(source)/stats.COUNT
    assert digest.hexdigest() == spec['draw_indices_sha256_little_endian']
    computed = computed.reshape(5000,len(labels),len(stats.KEYS)).transpose(1,0,2)
    draw_err = float(np.max(np.abs(computed-expected_reps)))
    sorted_draws = np.sort(computed,axis=1)
    limits = []
    for p in (0.025,0.975):
        q = p*4999
        k = int(q)
        x = q-k
        limits.append(sorted_draws[:,k]*(1-x)+sorted_draws[:,k+1]*x)
    comparisons = csvrows(result/'paired.csv')
    assert len(comparisons) == len(labels)*len(stats.KEYS)
    max_ci_error = 0.0
    for comp, label in enumerate(labels):
        left,right = stats.PAIRS[comp]
        saved = summary['comparisons'][label]
        assert len(saved) == len(stats.KEYS)
        for metric, key in enumerate(stats.KEYS):
            item = saved[metric]
            row = comparisons[comp*len(stats.KEYS)+metric]
            assert row['comparison'] == label and row['metric'] == item['metric'] == key
            for site in ('left','right','absolute_difference','relative_decrease_fraction','difference_CI95_low','difference_CI95_high'):
                assert item[site] == (float(row[site]) if row[site] else None)
            a = means[left][metric]
            b = means[right][metric]
            absolute = fsum(deltas[comp,:,metric])/stats.COUNT
            max_point_error = max(max_point_error,abs(a-item['left']),abs(b-item['right']),abs(absolute-item['absolute_difference']))
            if a:
                assert abs(-absolute/a-item['relative_decrease_fraction']) < stats.TOL
            else:
                assert item['relative_decrease_fraction'] is None
            max_ci_error = max(max_ci_error,abs(limits[0][comp,metric]-item['difference_CI95_low']),
                               abs(limits[1][comp,metric]-item['difference_CI95_high']))
    assert max(max_point_error, max_ci_error, draw_err) < stats.TOL
    return {'all_cells_compared': values.size, 'paired_differences_compared': deltas.size,
            'bootstrap_means_compared': computed.size, 'interval_endpoints_compared': 2*len(labels)*len(stats.KEYS),
            'draw_indices_sha256_little_endian': digest.hexdigest(), 'max_abs_per_user': values_err,
            'max_abs_per_user_difference': deltas_err, 'max_abs_point': float(max_point_error),
            'max_abs_bootstrap_draw': draw_err, 'max_abs_ci': float(max_ci_error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--export', type=Path, required=True, help='Directory containing full exported candidates and axes')
    parser.add_argument('--statistics', type=Path, required=True, help='Directory containing metrics.csv, paired.csv, summary.json and arrays.npz')
    args = parser.parse_args()
    scorer = independent_scorer()
    values, axis, diagnostics = export_scores(args.export, scorer)
    proof = verify_bootstrap(values,axis,args.statistics,scorer)
    print(json.dumps({'status':'PASS_OFFLINE_INDEPENDENT', 'users_per_condition': scorer.COUNT,
                      'mode_order': scorer.MODES, 'user_axis_sha256': axis, 'candidate_diagnostics': diagnostics,
                      'numpy': np.__version__, **proof}, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
