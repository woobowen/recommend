"""Check saved LETTER point estimates and confidence intervals without loading a model."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

EVIDENCE = Path(__file__).resolve().parent
BEAUTY = Path('lc_rec/a0_beauty/evidence/paired-s4-supplement-20261117/evidence')
G3 = Path('lc_rec/a0_beauty/grid_sid_preparation/g3_conversion/production_preparation/cpu_closure/results/compute')
A0 = Path('lc_rec/letter/evidence/final_audit_20261113/jobs/letter-a0-20261112-final-audit-20261113-offline-r1/results/primary')
A1 = Path('lc_rec/a1_qwen3_06b/evidence/status_checks/20261114T034030Z/existing_results/offline_evaluation/primary')


def read_json(path):
    return json.loads(path.read_text())


def read_matrix(path, field='values'):
    with np.load(path, allow_pickle=False) as archive:
        names = tuple(archive['keys'].tolist())
        rows = archive['row_idx']
        values = archive[field]
        axis = str(archive['user_axis_sha256'])
    assert len(names) == len(set(names)) == 16, path
    assert values.dtype == np.float64 and values.shape == (len(rows), len(names)), path
    assert np.array_equal(rows, np.arange(len(rows))) and np.isfinite(values).all(), path
    assert (values >= 0).all() and (values <= 1 + 1e-13).all(), path
    return names, values, axis


def means(values):
    return np.array([math.fsum(col) / len(col) for col in values.T])


def match_point(label, matrix, metrics):
    keys, values, axis = matrix
    doc = read_json(metrics)
    assert len(values) == doc['users'] and axis == doc['user_axis_sha256'], metrics
    expected = np.array([doc['metrics'][key] for key in keys])
    point = means(values)
    np.testing.assert_allclose(expected, point, atol=1e-13, rtol=0, err_msg=label)
    return {'comparison': label, 'users': len(values), 'metrics': len(keys), 'user_axis_sha256': axis,
            'max_point_error': float(np.max(np.abs(expected - point)))}


def match_pair(label, normal, treatment, paired, draws):
    keys, left, axis = normal
    right_keys, right, right_axis = treatment
    doc = read_json(paired)
    assert keys == right_keys and axis == right_axis and left.shape == right.shape, paired
    assert len(left) == doc['users'], paired
    archive = np.load(draws, allow_pickle=False)
    assert archive.shape == (5000, len(keys)) and np.isfinite(archive).all(), draws
    cutoffs = np.percentile(archive, (2.5, 97.5), axis=0, method='linear')
    actual = np.vstack((means(left), means(right), means(right - left), cutoffs))
    fields = ('Normal', 'Shuffle', 'absolute_difference', 'difference_CI95_low', 'difference_CI95_high')
    expected = np.array([[doc['metrics'][key][field] for key in keys] for field in fields])
    np.testing.assert_allclose(expected, actual, atol=1e-13, rtol=0, err_msg=label)
    return {'comparison': label, 'users': len(left), 'metrics': len(keys), 'user_axis_sha256': axis,
            'saved_draws_per_metric': len(archive), 'max_pair_field_error': float(np.max(np.abs(expected - actual)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=EVIDENCE.parents[2])
    args = parser.parse_args()
    out = []
    for model, original, prefix in [('a0', A0, 'letter-a0-20261112-'), ('a1', A1, 'letter-a1-q06-20261113-')]:
        normal = read_matrix(EVIDENCE / 'instruments-normal' / model / 'per_user_metrics.npz')
        out.append(match_point(f'Instruments {model} Normal', normal, args.repo / original / 'metrics.json'))
        for degree in (3, 4):
            results = args.repo / 'lc_rec/shuffle/results' / f'{prefix}shuffle{degree}-instruments-seed42-20261114-r0/evaluation'
            treatment = read_matrix(results / 'primary/per_user_metrics.npz')
            out.append(match_point(f'Instruments {model} V1 S{degree}', treatment, results / 'primary/metrics.json'))
            paired = results / 'paired'
            out.append(match_pair(f'Instruments {model} V1 S{degree} − Normal', normal, treatment,
                                  paired / 'paired_comparison.json', paired / 'paired_bootstrap_difference_means.npy'))
    for group in ('LETTER Beauty V1', 'G3 Beauty V1'):
        results = args.repo / (BEAUTY if group == 'LETTER Beauty V1' else G3)
        matrices = {}
        for mode in ('normal', 's3', 's4'):
            if group == 'LETTER Beauty V1':
                matrix = results / 'results/public_metric_matrices.npz'
                metrics = results / 'original' / f'{mode}-primary/metrics.json'
                matrices[mode] = read_matrix(matrix, mode)
            else:
                primary = results / 'original' / mode / 'primary'
                matrices[mode], metrics = read_matrix(primary / 'per_user_metrics.npz'), primary / 'metrics.json'
            out.append(match_point(f'{group} {mode}', matrices[mode], metrics))
        for mode in ('s3', 's4'):
            if group == 'LETTER Beauty V1':
                paired = results / ('original/s3' if mode == 's3' else 'results/s4')
            else:
                paired = results / 'paired' / mode
            out.append(match_pair(f'{group} {mode} − Normal', matrices['normal'], matrices[mode],
                                  paired / 'paired_comparison.json', paired / 'paired_bootstrap_difference_means.npy'))
    print(json.dumps({'status': 'PASS', 'tolerance_abs': 1e-13, 'checked': out}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
