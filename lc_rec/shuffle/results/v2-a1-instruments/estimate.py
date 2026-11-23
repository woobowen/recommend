"""Project formal latency/bytes from the A1 full CPU axis and actual 32-user timings."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np

MODES = ('shuffle2', 'shuffle3a', 'shuffle3b', 'shuffle4')


def load(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def class_of(n):
    return '4-6' if n <= 6 else '7-12' if n <= 12 else '13-20'


def bytes_in_formal(row):
    row = dict(row)
    ev = {k: v for k, v in row['shuffle'].items() if k not in ('groups', 'original_input_ids')}
    ev['actual_prefill'] = {k: v for k, v in ev['actual_prefill'].items()
                           if k in ('input_sha256', 'attention_mask_sha256', 'position_ids_sha256',
                                    'cache_length', 'beam_expansion', 'all_beams_ids_mask_positions_exact')}
    row['shuffle'] = ev
    return len((json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('evidence', type=Path)
    p.add_argument('destination', type=Path)
    args = p.parse_args()
    replay = args.evidence / 'cpu-replay-r0'
    gpu = args.evidence / 'gpu-preflight-r0'
    cpu_normal = args.evidence / 'cpu-normal-r0'
    cpu_assess = args.evidence / 'cpu-assess-r0'
    records = list(rows(replay / 'inputs.jsonl'))
    n = len(records)
    assert n == 24772
    full_hist = Counter(row['visible_history_length'] for row in records)
    full_tokens = Counter(row['encoded_length'] for row in records)
    full = Counter(class_of(row['visible_history_length']) for row in records)
    estimates = {}
    bf16 = load(gpu / 'summary.json')
    sample_tokens = sample_hist = None
    for mode in MODES:
        run = bf16['stages'][mode]
        bucket = defaultdict(list)
        contract = list(rows(gpu / mode / 'predictions.rank0.jsonl'))
        lengths = {row['row_idx']: row['shuffle']['history_group_count'] for row in contract}
        hist = Counter(lengths.values())
        toks = Counter(len(row['input_token_ids']) for row in contract)
        if sample_hist is None:
            sample_hist, sample_tokens = hist, toks
        assert hist == sample_hist and toks == sample_tokens
        for row in contract:
            bucket[class_of(lengths[row['row_idx']])].append(
                (row['generation_seconds'], bytes_in_formal(row)))
        assert set(bucket) == set(full)
        elapsed = [row['generation_seconds'] for row in contract]
        row_outer_overhead = max(0, run['generation_seconds'] - sum(elapsed)) / len(contract)
        startup = max(0, run['seconds'] - run['generation_seconds'])
        q = lambda values, fraction: float(np.quantile(values, fraction, method='linear'))
        band = {str(v): startup + n * row_outer_overhead + sum(
            full[k] * q([x[0] for x in group], v) for k, group in bucket.items()) for v in (.1, .9)}
        estimates[mode] = {
            'sample_users': len(contract), 'observed_launch_through_exit_seconds': run['seconds'],
            'observed_strict_load_seconds': run['model_load_seconds'],
            'observed_generation_loop_seconds': run['generation_seconds'],
            'observed_peak_allocated_bytes': run['gpu_peak_allocated_bytes'],
            'observed_peak_reserved_bytes': run['gpu_peak_reserved_bytes'],
            'buckets': {k: {'sample': len(v), 'population': full[k],
                            'mean_generate_seconds': float(np.mean([x[0] for x in v])),
                            'mean_projected_formal_row_bytes': float(np.mean([x[1] for x in v]))}
                        for k, v in bucket.items()},
            'launch_exit_overhead_seconds': startup, 'per_row_non_generate_loop_overhead_seconds': row_outer_overhead,
            'reweighted_wall_seconds': startup + n * row_outer_overhead + sum(
                full[k] * float(np.mean([x[0] for x in v])) for k, v in bucket.items()),
            'unweighted_wall_seconds': startup + n * (row_outer_overhead + float(np.mean(elapsed))),
            'p10_p90_wall_sensitivity_seconds': band,
            'projected_raw_prediction_bytes': round(sum(
                full[k] * float(np.mean([x[1] for x in v])) for k, v in bucket.items())),
        }
    normal = load(cpu_normal / 'summary.json')
    confirm = load(cpu_assess / 'summary.json')
    paired = load(cpu_assess / 'paired/summary.json')
    # A bounded estimate of the actual full-shape arithmetic, using archived Normal
    # as an input-only stand-in. No condition results are generated or saved.
    values = np.load(cpu_normal / 'primary/per_user_metrics.npz')['values']
    assert values.shape == (24772, 16)
    rng = np.random.Generator(np.random.PCG64(20261113))
    start = time.monotonic()
    draw_count = 64
    for _ in range(draw_count // 8):
        ix = rng.integers(0, n, (8, n), dtype=np.int64)
        for _ in range(5):
            assert np.isfinite(values[ix].mean(axis=1)).all()
    arithmetic_seconds = time.monotonic() - start
    body = {
        'status': 'ESTIMATE_FROM_ACTUAL_A1_PREFLIGHT', 'full_users': n,
        'full_histogram_history_groups': dict(sorted(full_hist.items())), 'prefight_histogram_history_groups': dict(sorted(sample_hist.items())),
        'full_histogram_a1_token_length': dict(sorted(full_tokens.items())), 'preflight_histogram_a1_token_length': dict(sorted(sample_tokens.items())),
        'modes': estimates, 'group_point_seconds': sum(x['reweighted_wall_seconds'] for x in estimates.values()),
        'group_low_high_sensitivity_seconds': {percent: sum(
            x['p10_p90_wall_sensitivity_seconds'][percent] for x in estimates.values()) for percent in ('0.1', '0.9')},
        'raw_formal_predictions_bytes': sum(x['projected_raw_prediction_bytes'] for x in estimates.values()),
        'formal_bootstrap_indices_bytes_without_header': 5000 * n * 8,
        'cpu_normal_actual_seconds': normal['elapsed_seconds'], 'cpu_preflight_assess_seconds': confirm['elapsed_seconds'],
        'cpu_paired_sample_actual_seconds': paired['elapsed_seconds'],
        'full_shape_local_arithmetic_fixture_seconds': arithmetic_seconds,
        'full_shape_arithmetic_projection_seconds': arithmetic_seconds * 5000 / draw_count,
        'cpu_full_replay_actual_seconds': load(replay / 'audit.json')['elapsed_seconds'],
        'scope': ('History-stratum means weighted by full-axis counts. Sensitivity sums conditional sample '
                  'P10/P90, not an uncertainty interval. The 32 fixed samples overrepresent long histories, '
                  'cover token lengths 71 through 167, and include extra embedding/weight witnesses; actual '
                  'CPU/GPU contention, cache warmup, read/write and JSON overhead can differ. CPU fixture uses '
                  'archived Normal values only, on WSL, and times the full paired arithmetic without file I/O.'),
    }
    with args.destination.open('x') as out:
        json.dump(body, out, indent=2, allow_nan=False)
        out.write('\n')


if __name__ == '__main__':
    main()
