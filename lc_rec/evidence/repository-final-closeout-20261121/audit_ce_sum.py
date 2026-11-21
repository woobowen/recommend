#!/usr/bin/env python3
"""Reconstruct the 14 historical CE sum differences without importing NumPy."""
import argparse
import ast
import datetime
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

GROUPING_SHA256 = 'aaeb799cb5888ff9a54b13d0bf0f77e0de20d37ed6fe179a9cba91f4869a7874'


def json_file(path):
    return json.loads(path.read_bytes())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def builtin_calls(source):
    assert sha256(source) == GROUPING_SHA256
    parsed = ast.parse(source.read_bytes())
    grouping = next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == 'ce_grouping')
    calls = {}
    for node in ast.walk(grouping):
        if isinstance(node, ast.Assign) and any(isinstance(var, ast.Name) and var.id == 'total' for var in node.targets):
            calls['sum_sample_loss_weights'] = node.value
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and key.value == 'original_grouped_ce_task_weight':
                    calls['original_grouped_ce_task_weight'] = value
    assert len(calls) == 2 and all(isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == 'sum' for value in calls.values())
    assert all(not (isinstance(node, ast.Name) and node.id == 'sum' and isinstance(node.ctx, ast.Store)) for node in ast.walk(parsed))
    return {key: {'line': value.lineno, 'call': ast.unparse(value)} for key, value in calls.items()}


def simple_sum(weights):
    total = 0.0
    for weight in weights:
        total += weight
    return total


def exact(a, b):
    return a.hex() == b.hex()


def audit(args):
    calls = builtin_calls(args.source)
    historical = json_file(args.grouping)
    assert sha256(args.samples) == historical['source_files']['samples']['sha256']
    samples = json_file(args.samples)
    old = json.load(sys.stdin) if args.differences == '-' else json_file(Path(args.differences))
    diffs = old['mixed_ce_weighting_only_differences']
    assert len(diffs) == len({record['pointer'] for record in diffs}) == 14
    observations = []
    for record in diffs:
        section, plan_name, micro, node, *fields = record['pointer'].split('/')
        assert (section, node) == ('mixed_global128', 'ce_weighting')
        group = historical[section][plan_name][micro][node]
        sample_order = [row['sample_id'] for row in group['per_sample']]
        assert sample_order == samples['plans'][plan_name] and len(sample_order) == 128
        if len(fields) == 1:
            field, = fields
            assert field == 'sum_sample_loss_weights'
            task = None
            value = group[field]
        else:
            family, task, field = fields
            assert (family, field) == ('per_task', 'original_grouped_ce_task_weight')
            value = group[family][task][field]
        weights = [row['sample_loss_weight'] for row in group['per_sample']
                   if task is None or samples['records'][row['sample_id']]['task'] == task]
        assert weights and all(type(w) is float for w in weights)
        original, replay = record['original'], record['replayed']
        p, f, builtin = simple_sum(weights), math.fsum(weights), sum(weights)
        assert exact(value, original) and exact(original, p) and exact(replay, f)
        assert not exact(original, replay) and exact(abs(original - replay), record['abs_delta'])
        expected = original if args.expect == 'historical' else replay
        assert exact(builtin, expected), (record['pointer'], sys.version, builtin, expected)
        observations.append({'pointer': record['pointer'], 'float_count': len(weights),
                             'weights_hex_sha256': hashlib.sha256(('\n'.join(w.hex() for w in weights) + '\n').encode()).hexdigest(),
                             'historical': original, 'prior_replay': replay, 'abs_delta': abs(original - replay),
                             'left_to_right': p, 'math_fsum': f, 'builtin_sum': builtin,
                             'float64_hex': {'historical': original.hex(), 'replay': replay.hex(), 'current_builtin': builtin.hex()}})
    assert 'numpy' not in sys.modules
    return {'status': 'PASS', 'capture_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'python_executable': sys.executable, 'python': sys.version, 'implementation': platform.python_implementation(),
            'numeric_backend': 'Python built-in float; NumPy not imported', 'expect': args.expect,
            'source_sha256': sha256(args.source), 'grouping_sha256': sha256(args.grouping), 'samples_sha256': sha256(args.samples),
            'source_calls': calls, 'historical_numpy_metadata': old['numpy_historical'], 'prior_replay_numpy_metadata': old['numpy_replay'],
            'verified_fields': len(observations), 'maximum_abs_delta': max(r['abs_delta'] for r in observations), 'fields': observations}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--grouping', type=Path, required=True)
    parser.add_argument('--samples', type=Path, required=True)
    parser.add_argument('--differences', required=True, help='Previous difference JSON path, or - for stdin.')
    parser.add_argument('--expect', required=True, choices=['historical', 'replay'])
    print(json.dumps(audit(parser.parse_args()), indent=2))


if __name__ == '__main__':
    main()
