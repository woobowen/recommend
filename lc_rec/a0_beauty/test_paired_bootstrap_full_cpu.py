#!/usr/bin/env python3
"""Full-axis synthetic test of the production paired bootstrap, not a job receipt."""
import hashlib
import json
import argparse
from pathlib import Path
import time
import numpy as np
from paired_beauty import module, paired_bootstrap, validate_metric_matrix

EXPECTED = {
    'Normal': '718e70481baeb6d6acbc025b1ab72fc9aadcc9e2ff5b4efb830c627a593d15d9',
    'S3': '9524712cebfb87da487bab990b114e3065d8242b4426667d208b2c4ab775a7f0',
    'S4': 'ea56b2e4b498c3e6f573985cfe1f1799eeeb6dee5adfc7d7a61da7b2bd5f92d2',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixture-root', type=Path)
    args = parser.parse_args()
    start = time.monotonic()
    path = Path(__file__).resolve().parents[1] / 'letter/review/final_audit_20261113/statistics_verify.py'
    if args.fixture_root:
        matrices = {}
        for name, expected in EXPECTED.items():
            f = args.fixture_root / name / 'per_user_metrics.npz'
            if hashlib.sha256(f.read_bytes()).hexdigest() != expected:
                raise ValueError('fixture SHA mismatch: ' + name)
            with np.load(f, allow_pickle=False) as z:
                values = z['values'].copy()
                validate_metric_matrix(np, z['keys'].tolist(), values)
                if not np.array_equal(z['row_idx'], np.arange(22363)) or str(z['user_axis_sha256']) != '7430079e2b917dfa5e4bb8ead2702fb6c622fb48ffb684dd2889780f9480af7f':
                    raise ValueError('user axis mismatch')
                matrices[name] = values
        differences = {name: matrices[name] - matrices['Normal'] for name in ('S3', 'S4')}
    else:
        rows = np.arange(22363, dtype=np.float64)[:, None]
        columns = np.arange(16, dtype=np.float64)[None, :]
        differences = {'generated': ((rows * (columns + 1)) % 19 - 9) / 19}
    errors = {}
    for name, difference in differences.items():
        draws, low, high = paired_bootstrap(np, difference)
        intervals, independent = module(path).bootstrap_columns(difference)
        draw_error = float(np.max(np.abs(draws - independent)))
        interval_error = float(np.max(np.abs(np.array([low, high]) - intervals)))
        if draw_error > 1e-13 or interval_error > 1e-13:
            raise AssertionError('production and independent bootstrap disagree')
        errors[name] = {'max_draw_error': draw_error, 'max_interval_error': interval_error}
    for bad in (difference[:2], np.full((22363, 16), np.nan), difference.astype(np.float32)):
        try:
            paired_bootstrap(np, bad)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid matrix accepted')
    print(json.dumps({'designation': 'SYNTHETIC_CPU_INTEGRATION_ONLY',
                      'users': 22363, 'metrics': 16, 'repetitions': 5000,
                      'independent_source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'comparisons': errors, 'fixture_sha256': EXPECTED if args.fixture_root else None,
                      'elapsed_seconds': time.monotonic() - start, 'status': 'PASS'}))


if __name__ == '__main__':
    main()
