"""Project the entire saved workload to fields needed for numeric and eligibility review."""
import argparse
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

ONLY_ORIGINAL = {'random_choices', 'input_sha256_utf8', 'full_sha256_utf8'}
SOURCE_JOB = 'lc-six-20261110-cpu-r0'
GROUPING_JOB = 'lc-six-20261110-grouping-r2'


def file_sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def normalized(value):
    return json.loads(json.dumps(value, allow_nan=False))


def differences(actual, earlier, path, deltas):
    if isinstance(actual, list):
        for i, (left, right) in enumerate(zip(actual, earlier, strict=True)):
            differences(left, right, f'{path}/{i}', deltas)
    elif isinstance(actual, dict):
        assert actual.keys() == earlier.keys(), path
        for name in actual:
            differences(actual[name], earlier[name], f'{path}/{name}', deltas)
    elif actual != earlier:
        assert isinstance(actual, float) and isinstance(earlier, float), path
        assert '/ce_weighting/' in path and math.isclose(actual, earlier, rel_tol=0, abs_tol=4 * math.ulp(1.0)), path
        deltas.append({'field': path, 'historical': earlier, 'replayed': actual, 'abs': abs(actual-earlier)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    exp = args.experiment.resolve()
    src = exp / 'evidence/jobs' / SOURCE_JOB / 'output/workload'
    original = src / 'records.jsonl.gz'
    summary = json.loads((src / 'workload_summary.json').read_bytes())
    grouping = json.loads((exp / 'evidence/jobs' / GROUPING_JOB / 'output/grouping/grouping.json').read_bytes())
    sys.path.insert(0, str(exp / 'scripts'))
    from analyze_workload import describe_subset
    from analyze_grouping import run
    import numpy as np

    projected = args.output / 'records-numeric.jsonl.gz'
    seen = collections.Counter()
    original_sha = file_sha(original)
    assert original_sha == grouping['source_files']['records']['sha256']
    with gzip.open(original, 'rt', encoding='utf-8') as old, projected.open('wb') as to_file:
        with gzip.GzipFile(fileobj=to_file, filename='', mode='wb', mtime=0, compresslevel=5) as new:
            for i, line in enumerate(old):
                row = json.loads(line)
                assert row['concat_index'] == i
                seen[row['task']] += 1
                projected_row = {k: v for k, v in row.items() if k not in ONLY_ORIGINAL}
                assert len(projected_row) + len(ONLY_ORIGINAL) == len(row)
                new.write(json.dumps(projected_row, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode() + b'\n')
    assert seen == summary['dataset_rows'] and sum(seen.values()) == summary['concat_rows']
    with gzip.open(projected, 'rt', encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream]
    for task, expected in summary['per_task'].items():
        result = describe_subset(rows, [i for i, row in enumerate(rows) if row['task'] == task], np)
        assert normalized(result) == expected, task
    full = describe_subset(rows, range(len(rows)), np)
    assert normalized(full) == {k: summary['joint'][k] for k in full}
    replay = run(projected, src / 'samples.json')
    original_replay = run(original, src / 'samples.json')
    assert {k: v for k,v in replay.items() if k != 'source_files'} == {k: v for k,v in original_replay.items() if k != 'source_files'}
    deltas = []
    for k in ('full_records', 'task_counts', 'full_dataset_zero_shift_label_records', 'full_epoch_fixed_realization',
              'mixed_global128', 'permutation_sha256_int64_little_endian'):
        if k == 'mixed_global128':
            differences(normalized(replay[k]), grouping[k], k, deltas)
        else:
            assert normalized(replay[k]) == grouping[k], k
    print(json.dumps({'origin': str(original), 'origin_bytes': original.stat().st_size, 'origin_sha256': original_sha,
                      'projection_bytes': projected.stat().st_size, 'projection_sha256': file_sha(projected),
                      'omitted_per_row_fields': sorted(ONLY_ORIGINAL), 'rows': len(rows), 'task_counts': seen,
                      'verified': ['seven descriptive group summaries', 'full grouping layouts and fixed permutation',
                                   'both mixed-global128 grouping, padding and token weights'],
                      'same_environment_original_equal': True, 'historical_numpy': grouping['numpy_version'],
                      'current_numpy': np.__version__, 'historical_roundoff': deltas,
                      'scope': 'Numeric and eligibility review of all rows. Frozen text/RNG witnesses are separate original samples.'}, indent=2))


if __name__ == '__main__':
    main()
