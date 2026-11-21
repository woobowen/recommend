"""Summarize measured preflight timing for the future, unsubmitted Beauty plan."""
import argparse
import json
import math
from pathlib import Path
import statistics

from lc_rec.shuffle_v2 import beauty, core


def measure(root, config):
    summary = json.loads((root / 'summary.json').read_text())
    assert summary['status'] == 'PASS_V2_PREFLIGHT'
    assert [stage['mode']['mode'] for stage in summary['stages']] == list(core.MODES)
    users = config['users']
    by_mode = {}
    for stage in summary['stages']:
        mode = stage['mode']['mode']
        folder = root / mode
        rank = json.loads((folder / 'rank0.json').read_text())
        output = folder / 'predictions.rank0.jsonl'
        with output.open() as stream:
            rows = [json.loads(line) for line in stream]
        assert stage['exit_code'] == 0 and len(rows) == rank['completed_users'] == 32
        candidates = sum(len(row['candidates']) for row in rows)
        assert candidates == 640
        per_user = [row['generation_seconds'] for row in rows]
        generation_wall = rank['elapsed_generation_seconds']
        load_wall = rank['model_load_and_strict_verification_seconds']
        wrapper_other = max(0, stage['seconds'] - load_wall - generation_wall)
        expected_generation = generation_wall / len(rows) * users
        forecast = load_wall + wrapper_other + expected_generation
        assert all(math.isfinite(value) and value >= 0 for value in (load_wall, generation_wall, wrapper_other, forecast))
        by_mode[mode] = {
            'users_sampled': len(rows), 'raw_candidates': candidates,
            'measured_attempt_wall_seconds': stage['seconds'],
            'measured_model_load_seconds': load_wall,
            'measured_generation_seconds': generation_wall,
            'other_measured_seconds': wrapper_other,
            'per_user_generate_median_seconds': statistics.median(per_user),
            'per_user_generate_max_seconds': max(per_user),
            'formal_generation_extrapolated_seconds': expected_generation,
            'formal_wall_projection_seconds': forecast,
            'output_bytes_sample': output.stat().st_size,
            'raw_jsonl_formal_bytes_projection': math.ceil(output.stat().st_size / len(rows) * users),
        }
    values = [entry['formal_wall_projection_seconds'] for entry in by_mode.values()]
    budget = config['budget']
    fixed_margin = 1.25
    return {
        'status': 'BEAUTY_PREFLIGHT_TIMING_ESTIMATE',
        'scope': 'Engineering projection from the same frozen 32 users per mode; no formal submission or result',
        'run_commit': summary['RUN_COMMIT'],
        'basis': 'Mode wall generation time / 32 × 22363, plus measured loading and other stage overhead',
        'uncertainty': 'Fixed input-feature-selected sample, transient server load/cache, serialization and fsync variability; estimate is not an upper bound or a statistical interval.',
        'modes': by_mode, 'projected_single_gpu_total_seconds': sum(values),
        'sensitivity_multiplier': fixed_margin,
        'sensitivity_max_mode_seconds': max(values) * fixed_margin,
        'sensitivity_total_seconds': sum(values) * fixed_margin,
        'fits_formal_attempt_under_sensitivity': max(values) * fixed_margin < budget['formal_attempt_wall_seconds'],
        'fits_formal_total_under_sensitivity': sum(values) * fixed_margin < budget['formal_total_device_seconds'],
        'projected_raw_formal_jsonl_bytes': sum(entry['raw_jsonl_formal_bytes_projection'] for entry in by_mode.values()),
        'cpu_reservation_seconds': {'each_evaluation': 600, 'joint_bootstrap': 2100},
        'cpu_projection_basis': 'Use archived full Normal evaluation time as a size-matched baseline, plus single-shard boundary checks; bootstrap reservation covers shared 5000 draws and independent checks.',
        'review_timing': 'Separate after computation, not projected as GPU time',
    }


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--preflight-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=beauty.CONFIG)
    args = parser.parse_args()
    result = measure(args.preflight_dir, json.loads(args.config.read_text()))
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
