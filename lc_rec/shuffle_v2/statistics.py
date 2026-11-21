"""Combine the five frozen evaluation matrices using paired PCG64 user sampling."""
import argparse
import csv
from hashlib import sha256
import json
import math
import os
from pathlib import Path

from lc_rec.shuffle_v2 import core
from lc_rec.shuffle_v2.evaluate import LC, module
from lc_rec.shuffle_v2.replay import CONFIG, dump, frozen

CONDITIONS = ('normal', *core.MODES)
COMPARISONS = tuple(('normal', mode) for mode in core.MODES) + (('shuffle3a', 'shuffle3b'),)
IDENTITY_FIELDS = ('row_idx', 'user_id', 'user_axis_sha256', 'target_item_id', 'target_sid',
                   'target_group_size', 'original_interaction_length', 'visible_history_length',
                   'history_group', 'target_training_frequency', 'popularity_group', 'protocol_version')


def read_metrics(root, mode, users, axis):
    import numpy as np
    summary = json.loads((root / 'summary.json').read_text())
    assert summary['status'] == 'PASS_V2_EVALUATION' and summary['condition'] == mode
    primary = root / 'primary'
    matrix_path, row_path = (primary / name for name in ('per_user_metrics.npz', 'per_user.csv'))
    with np.load(matrix_path, allow_pickle=False) as raw:
        keys, values = raw['keys'].tolist(), raw['values']
        assert len(keys) == 16 and values.shape == (users, len(keys)) and values.dtype == np.float64
        assert np.array_equal(raw['row_idx'], np.arange(users)) and str(raw['user_axis_sha256']) == axis
    with row_path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    identities = [tuple(row[field] for field in IDENTITY_FIELDS) for row in rows]
    assert [row['row_idx'] for row in rows] == [str(i) for i in range(users)]
    assert np.array_equal(values, np.asarray([[float(row[k]) for k in keys] for row in rows]))
    assert np.isfinite(values).all()
    hashes = {str(path): frozen.sha(path) for path in (matrix_path, row_path, root / 'summary.json')}
    return keys, identities, values, hashes


def paired_means(deltas, users):
    import numpy as np
    rng = np.random.Generator(np.random.PCG64(20261113))
    draws = np.empty((len(deltas), 5000, 16), dtype=np.float64)
    sample_hash = sha256()
    for start in range(0, 5000, 8):
        sample = rng.integers(0, users, size=(min(8, 5000 - start), users), dtype=np.int64)
        sample_hash.update(sample.astype('<i8', copy=False).tobytes(order='C'))
        for k, diff in enumerate(deltas):
            draws[k, start:start + len(sample)] = diff[sample].mean(axis=1)
    return draws, sample_hash.hexdigest()


def main():
    import numpy as np
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('condition_roots', type=Path, nargs=len(CONDITIONS), metavar='EVALUATION_DIR')
    parser.add_argument('--config', type=Path, default=CONFIG)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    config = json.loads(args.config.read_text())
    users, axis = (config[k] for k in ('users', 'user_axis_sha256'))
    args.output.mkdir(parents=True, exist_ok=False)
    rows = [read_metrics(root, mode, users, axis) for root, mode in zip(args.condition_roots, CONDITIONS, strict=True)]
    keys, identities, _, _ = rows[0]
    assert all(keys == names and identities == ids for names, ids, _, _ in rows)
    matrices = {mode: row[2] for mode, row in zip(CONDITIONS, rows, strict=True)}
    main_table = {mode: {key: math.fsum(matrix[:, j]) / users for j, key in enumerate(keys)}
                  for mode, matrix in matrices.items()}
    deltas = [matrices[b] - matrices[a] for a, b in COMPARISONS]
    means, sample_hash = paired_means(deltas, users)
    limits = np.percentile(means, [2.5, 97.5], axis=1, method='linear')
    independent_path = LC / 'letter/review/final_audit_20261113/statistics_verify.py'
    independent = module('frozen_statistics_reference', independent_path)
    comparisons, checks = {}, {}
    for i, (a, b) in enumerate(COMPARISONS):
        name = f'{b}-{a}'
        other_limits, other_means = independent.bootstrap_columns(deltas[i])
        checks[name] = {'max_draw_difference': float(np.max(np.abs(means[i] - other_means))),
                        'max_ci_difference': float(np.max(np.abs(limits[:, i, :] - other_limits)))}
        assert max(checks[name].values()) <= 1e-13
        comparisons[name] = []
        for j, key in enumerate(keys):
            left, right = main_table[a][key], main_table[b][key]
            delta = right - left
            point = {'metric': key, 'left': left, 'right': right, 'absolute_difference': delta,
                                      'relative_decrease_fraction': -delta / left if left else None,
                                      'difference_CI95_low': float(limits[0, i, j]), 'difference_CI95_high': float(limits[1, i, j])}
            if config.get('dataset') == 'Beauty':
                point.update(relative_change_fraction=delta / left if left else None,
                             percentage_point_difference=100 * delta)
            comparisons[name].append(point)
    paired = {'status': 'PASS_V2_PAIRED', 'users': users, 'user_axis_sha256': axis, 'protocol': core.PROTOCOL,
              'identity_policy': core.IDENTITY_POLICY, 'shuffle_seed': core.SEED,
              'comparisons': comparisons, 'metrics': main_table,
              'inputs': {mode: row[3] for mode, row in zip(CONDITIONS, rows, strict=True)},
              'bootstrap': {'rng': 'NumPy PCG64', 'seed': 20261113, 'replacement': True, 'replicates': 5000,
                            'draw_order': 'replicate-major then user-major; int64 integers(0, users)',
                            'draw_indices_sha256_little_endian': sample_hash, 'percentile_method': 'linear',
                            'bounds': [2.5, 97.5], 'multiple_comparison_adjustment': False,
                            'scope': 'Fixed verified user axis, model and one seed-42 intervention instance',
                            'nonauthor_reference_source_sha256': frozen.sha(independent_path), 'numeric_checks': checks}}
    if config.get('dataset') == 'Beauty':
        normal = main_table['normal']
        normal_changes = []
        for mode, values in main_table.items():
            for key, value in values.items():
                delta = value - normal[key]
                normal_changes.append({'condition': mode, 'metric': key, 'value': value, 'normal': normal[key],
                                       'delta': delta, 'percentage_point_difference': 100 * delta,
                                       'relative_change_fraction': delta / normal[key] if normal[key] else None})
        paired['changes_from_normal'] = normal_changes
        with (args.output / 'changes_from_normal.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(normal_changes[0]))
            writer.writeheader()
            writer.writerows(normal_changes)
    dump(args.output / 'summary.json', paired)
    with (args.output / 'metrics.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=('condition', *keys))
        writer.writeheader()
        writer.writerows(dict(condition=mode, **values) for mode, values in main_table.items())
    with (args.output / 'paired.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=('comparison', *next(iter(comparisons.values()))[0]))
        writer.writeheader()
        writer.writerows(dict(comparison=name, **point) for name, records in comparisons.items() for point in records)
    np.savez_compressed(args.output / 'arrays.npz', keys=keys, condition=CONDITIONS, comparison=list(comparisons),
                        row_idx=np.arange(users), user_axis_sha256=axis, per_user_values=np.asarray(list(matrices.values())),
                        per_user_differences=np.asarray(deltas), bootstrap_difference_means=means)


if __name__ == '__main__':
    main()
