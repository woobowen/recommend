"""Check engineering-only Beauty metrics and bootstrap from independent raw-row results."""
import argparse
from collections import Counter
import csv
from hashlib import file_digest, sha256
import json
import time
from pathlib import Path

import numpy as np


MODES = ('normal', 'shuffle2', 'shuffle3a', 'shuffle3b', 'shuffle4')
PAIRS = ((1, 0), (2, 0), (3, 0), (4, 0), (3, 2))
PAIRS_NAMES = tuple(f'{MODES[right]}-{MODES[left]}' for right, left in PAIRS)
CUTOFFS = (5, 10, 20)
METRICS = [*(f'hit@{k}' for k in (1, *CUTOFFS)),
           *(f'{key}@{k}' for key in ('ndcg', 'mrr', 'ItemHit', 'ItemNDCG') for k in CUTOFFS)]
SEED, REPLICATES, USERS, TOLERANCE = 20261113, 5000, 32, 1e-13


def sampled_means(matrices, replicates):
    """Weight each user by its count; use one sampled sequence for every condition."""
    sample_hash = sha256()
    users = matrices.shape[1]
    rng = np.random.Generator(np.random.PCG64(SEED))
    means = []
    for _ in range(replicates):
        sampled = rng.integers(users, size=users, dtype=np.int64)
        sample_hash.update(sampled.astype('<i8', copy=False).tobytes())
        weights = Counter(sampled.tolist())
        weighted = sum(matrices[:, i, :] * count for i, count in sorted(weights.items())) / users
        means.append(weighted)
    return np.asarray(means), sample_hash.hexdigest()


def intervals(draws):
    """Compute linear interpolation directly at the requested sample order statistics."""
    ordered = np.sort(draws, axis=1)
    bounds = []
    for quantile in (0.025, 0.975):
        position = (ordered.shape[1] - 1) * quantile
        lower, fraction = int(position), position - int(position)
        bounds.append(ordered[:, lower] * (1 - fraction) + ordered[:, lower + 1] * fraction)
    return np.asarray(bounds)


def maxabs(actual, expected):
    aa, ee = np.asarray(actual), np.asarray(expected)
    assert aa.shape == ee.shape
    assert np.isfinite(aa).all() and np.isfinite(ee).all()
    return float(np.max(np.abs(aa - ee)))


def identity(path):
    with path.open('rb') as src:
        return {'bytes': path.stat().st_size, 'sha256': file_digest(src, 'sha256').hexdigest()}


def run(review, paired):
    assert review['reviewer'] == '/root/research_review' and review['status'] == 'PASS'
    selection = review['global_row_indices']
    assert len(set(selection)) == USERS
    assert review['metric_keys'] == METRICS
    matrices = np.asarray([review['engineering_matrix'][mode] for mode in MODES], dtype=np.float64)
    assert matrices.shape == (5, USERS, 16)
    points = matrices.mean(axis=1)
    means, expected_sample_sha = sampled_means(matrices, REPLICATES)
    differences = np.asarray([matrices[right] - matrices[left] for right, left in PAIRS])
    draws = np.asarray([means[:, right] - means[:, left] for right, left in PAIRS])
    bounds = intervals(draws)
    with np.load(paired / 'arrays.npz', allow_pickle=False) as raw:
        assert raw['keys'].tolist() == METRICS
        assert tuple(raw['condition'].tolist()) == MODES
        assert tuple(raw['comparison'].tolist()) == PAIRS_NAMES
        assert raw['row_idx'].tolist() == selection
        assert str(raw['user_axis_sha256']) == review['sample_user_axis_sha256']
        assert all(raw[k].dtype == np.float64 for k in
                   ('per_user_values', 'per_user_differences', 'bootstrap_difference_means'))
        maxima = {
            'per_user_values': maxabs(raw['per_user_values'], matrices),
            'per_user_differences': maxabs(raw['per_user_differences'], differences),
            'bootstrap_difference_means': maxabs(raw['bootstrap_difference_means'], draws),
        }
    summary = json.loads((paired / 'summary.json').read_text())
    assert summary['status'] == 'PASS_BEAUTY_PREFLIGHT_EVALUATION_INTERFACE'
    assert summary['users'] == USERS and summary['formal_results'] is False
    assert summary['RUN_COMMIT'] == review['RUN_COMMIT']
    assert summary['cpu_manifest_sha256'] == review['cpu_manifest_sha256']
    assert summary['global_user_axis_sha256'] == review['global_user_axis_sha256']
    assert summary['user_axis_sha256'] == review['sample_user_axis_sha256']
    assert summary['preflight_summary_sha256'] == review['preflight_summary_sha256']
    assert summary['preflight_candidates'] == 2560 and summary['normal_sample_candidates'] == 640
    assert summary['normal_sample_matches_archive'] is True
    for mode in MODES[1:]:
        facts = summary['preflight'][mode]
        assert facts['predictions_sha256'] == review['modes'][mode]['files']['predictions.rank0.jsonl']['sha256']
        assert facts['rank_sha256'] == review['modes'][mode]['files']['rank0.json']['sha256']
    assert summary['preflight_assignment'] == selection
    assert tuple(summary['comparison_order']) == PAIRS_NAMES
    sampling = summary['bootstrap']
    expected = {'rng': 'NumPy PCG64', 'seed': SEED, 'replacement': True,
                'replicates': REPLICATES, 'percentile_method': 'linear',
                'bounds': [2.5, 97.5], 'multiple_comparison_adjustment': False,
                'draw_indices_sha256_little_endian': expected_sample_sha}
    assert all(sampling[key] == value for key, value in expected.items())
    assert tuple(summary['comparisons']) == PAIRS_NAMES
    point_error = 0.0
    ci_error = 0.0
    expected_csv = []
    for i, (comparison, (right, left)) in enumerate(zip(PAIRS_NAMES, PAIRS, strict=True)):
        assert [row['metric'] for row in summary['comparisons'][comparison]] == METRICS
        for j, entry in enumerate(summary['comparisons'][comparison]):
            delta = points[right, j] - points[left, j]
            point_error = max(point_error, maxabs(
                [entry['left'], entry['right'], entry['absolute_difference'], entry['percentage_point_difference']],
                [points[left, j], points[right, j], delta, 100 * delta]))
            if points[left, j]:
                point_error = max(point_error, abs(entry['relative_change_fraction'] - delta / points[left, j]))
            else:
                assert entry['relative_change_fraction'] is None
            ci_error = max(ci_error, maxabs(
                [entry['difference_CI95_low'], entry['difference_CI95_high']], bounds[:, i, j]))
            expected_csv.append({'comparison': comparison, **entry})
    with (paired / 'paired.csv').open(newline='') as stream:
        for actual, expected in zip(csv.DictReader(stream), expected_csv, strict=True):
            for key, value in expected.items():
                readback = actual[key] if key in ('comparison', 'metric') else float(actual[key]) if actual[key] else None
                assert readback == value, (key, readback, value)
    with (paired / 'metrics.csv').open(newline='') as stream:
        for i, (mode, observed) in enumerate(zip(MODES, csv.DictReader(stream), strict=True)):
            assert observed['condition'] == mode and len(observed) == 17
            observed_points = [float(observed[key]) for key in METRICS]
            assert observed_points == [summary['metrics'][mode][key] for key in METRICS]
            point_error = max(point_error, maxabs(observed_points, points[i]))
    changes = summary['changes_from_normal']
    with (paired / 'changes_from_normal.csv').open(newline='') as stream:
        for i, (observed, reported) in enumerate(zip(csv.DictReader(stream), changes, strict=True)):
            condition, j = divmod(i, 16)
            mode, metric = MODES[condition], METRICS[j]
            assert (observed['condition'], observed['metric']) == (reported['condition'], reported['metric']) == (mode, metric)
            for key in ('value', 'normal', 'delta', 'percentage_point_difference', 'relative_change_fraction'):
                readback = float(observed[key]) if observed[key] else None
                assert readback == reported[key]
            change = points[condition, j] - points[0, j]
            point_error = max(point_error, maxabs(
                [reported['value'], reported['normal'], reported['delta'], reported['percentage_point_difference']],
                [points[condition, j], points[0, j], change, 100 * change]))
            rel = reported['relative_change_fraction']
            if points[0, j]:
                point_error = max(point_error, abs(rel - change / points[0, j]))
            else:
                assert rel is None
    assert len(changes) == 80
    maxima.update(point_estimates=point_error, confidence_bounds=ci_error)
    assert max(maxima.values()) <= TOLERANCE, maxima
    return {
        'status': 'PASS_INDEPENDENT_PREFLIGHT_STATISTICS', 'reviewer': '/root/research_review',
        'scope': 'Engineering-only frozen 32-user GPU sample and archived Normal subset',
        'formal_results': False, 'RUN_COMMIT': review['RUN_COMMIT'], 'job_id': review['job_id'],
        'comparison_metric_rows': len(expected_csv), 'users': USERS, 'shared_samples': REPLICATES,
        'expected_sample_sha256': expected_sample_sha, 'maximum_absolute_differences': maxima,
        'files': {name: identity(paired / name) for name in ('summary.json', 'arrays.npz', 'paired.csv', 'metrics.csv', 'changes_from_normal.csv')},
    }


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--raw-review', type=Path, required=True)
    parser.add_argument('--paired', type=Path, required=True)
    args = parser.parse_args()
    assert __debug__, 'assertions must remain enabled'
    started = time.monotonic()
    receipt = run(json.loads(args.raw_review.read_text()), args.paired)
    receipt['review_source'] = identity(Path(__file__))
    receipt['local_cpu_wall_seconds'] = time.monotonic() - started
    receipt['raw_review_identity'] = identity(args.raw_review)
    print(json.dumps(receipt, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
