#!/usr/bin/env python3
"""Independent Shuffle V2 Beauty review from archived beam predictions.

Uses no project evaluator or statistics imports. This checks candidate identity,
metrics and paired statistics. The generation entry and model boundary are
reviewed separately against their source and execution evidence.
"""

import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import time

import numpy as np


MODES = ('normal', 'shuffle2', 'shuffle3a', 'shuffle3b', 'shuffle4')
PAIRS = ((0, 1), (0, 2), (0, 3), (0, 4), (2, 3))
PAIR_NAMES = tuple(f'{MODES[b]}-{MODES[a]}' for a, b in PAIRS)
ITEM_CUTOFFS = (5, 10, 20)
METRICS = ([f'hit@{k}' for k in (1, *ITEM_CUTOFFS)]
           + [f'{m}@{k}' for m in ('ndcg', 'mrr', 'ItemHit', 'ItemNDCG') for k in ITEM_CUTOFFS])
USERS = 22363
BEAMS = 20
BOOTSTRAP_SEED = 20261113
REPLICATES = 5000
TOLERANCE = 1e-13
DISCOUNT = (0.0,) + tuple(1 / math.log2(rank + 1) for rank in range(1, BEAMS + 1))
FROZEN_RUN = '6e1b211d9c37f358c43b1540b40b17f6703150d2'
FROZEN_AXIS = '7430079e2b917dfa5e4bb8ead2702fb6c622fb48ffb684dd2889780f9480af7f'
FROZEN_ADAPTER = 'dd9f3d2983af7ce85ace99534bd23df9362d9cd91ac39d0b356b219cb790b1c1'
FROZEN_DATA = {
    'index': 'be3fb890e59b5443824cdebc560ef349191b97f20d3aecad4727de2d6b212c82',
    'inter': '8809584b8b78fa6561771c88911ed65bd716fc9926fca1e581e6a1ab578ccfd5',
    'item': '74aeddf911548aadfcd1dab326c16678ed4ac74457e874da300d63a94bdb2330',
}
FROZEN_NORMAL_PREDICTIONS = (
    '90acbd6bc0bb11c9173332035b43fc4bf40391128ff5a6dab0069486b5c80519',
    '74c8e582e8d142731e71371fc508e1549a9fbd61f34f162f1ae69ad9af0c3bf4',
)
FROZEN_NORMAL_PRIMARY = {
    'metrics.json': '0b5390078d2c53aa441dce220725c5413873500bdf1343e69f4c57583842d9c7',
    'per_user.csv': 'a832d8d7757f76e4b81a0c779f539b3ab88ddd7fcca873aefc34fef4920479d6',
    'per_user_metrics.npz': 'da3fd116943e17635f9acd611684e8aa42b6295cba559521c9808d5e3afc1184',
}
FROZEN_NORMAL_ASSIGNMENTS = (
    '8eb669c968e313c79df1bba3f16b16bd94ce1e0d6697728ff528c1da3de01363',
    'ea9c57674be765e0424ac991e986823a5b8587cfee3d162c3b38a46deaf4f173',
)


def require(condition, context):
    if not condition:
        raise ValueError(str(context))


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()


def read(path):
    return json.loads(path.read_bytes())


def file_identity(path):
    h = sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1 << 20), b''):
            h.update(data)
    return {'path': str(path), 'sha256': h.hexdigest(), 'bytes': path.stat().st_size}


def catalog(root):
    data, identities = {}, {}
    for name, expected_hash in FROZEN_DATA.items():
        path = root / f'Beauty.{name}.json'
        raw = path.read_bytes()
        digest = sha256(raw).hexdigest()
        require(digest == expected_hash, ('Frozen dataset SHA', path))
        identities[name] = {'path': str(path), 'sha256': digest, 'bytes': len(raw)}
        data[name] = json.loads(raw)
    catalog = {item: ''.join(symbols) for item, symbols in data['index'].items()}
    sizes = Counter(catalog.values())
    require(len(catalog) == 12101 and set(catalog) == set(data['item']), 'Catalog items')
    require(len(sizes) == 12088 and Counter(size for size in sizes.values() if size > 1) == {2: 9, 3: 2},
            'SID collision profile')
    users = tuple(data['inter'])
    require(len(users) == USERS and sha256(compact(users)).hexdigest() == FROZEN_AXIS, 'Frozen user axis')
    symbol_ids = {symbol: token for token, symbol in enumerate(
        sorted({symbol for symbols in data['index'].values() for symbol in symbols}), 32000)}
    require(len(symbol_ids) == 965, 'Added token count')
    allowed = [{symbol_ids[symbols[position]]: symbols[position] for symbols in data['index'].values()}
               for position in range(4)]
    records = []
    for user, sequence in data['inter'].items():
        require(sequence and all(str(item) in catalog for item in sequence), ('Catalog membership', user))
        item = str(sequence[-1])
        records.append((user, item, catalog[item]))
    return records, sizes, allowed, identities


def rank_beams(sids, scores):
    require(len(sids) == len(scores) == BEAMS, 'Beam count')
    require(all(math.isfinite(score) for score in scores), 'Nonfinite score')
    # Explicit generation-order tie key independently implements stable descending sorting.
    order = sorted(range(BEAMS), key=lambda i: (-scores[i], i))
    return order, [sids[i] for i in order]


def row_metrics(ranked, target, multiplicities):
    listed = [rank for rank, sid in enumerate(ranked, 1) if sid == target]
    first = listed[0] if listed else BEAMS + 1
    # Expand each candidate SID to item ranks. Invalid SIDs occupy one rank, and
    # the first target SID is the only group with expected credit for its item.
    target_slots = []
    occupied = 0
    if listed:
        for sid in ranked[:first]:
            width = multiplicities.get(sid, 1)
            slots = range(occupied + 1, occupied + width + 1)
            occupied += width
            if sid == target:
                target_slots.extend(slots)
    target_width = multiplicities[target]
    result = [float(first <= cutoff) for cutoff in (1, *ITEM_CUTOFFS)]
    result.extend(math.fsum(DISCOUNT[rank] for rank in listed if rank <= cutoff)
                  for cutoff in ITEM_CUTOFFS)
    result.extend(1 / first if first <= cutoff else 0.0 for cutoff in ITEM_CUTOFFS)
    result.extend(sum(rank <= cutoff for rank in target_slots) / target_width for cutoff in ITEM_CUTOFFS)
    result.extend(math.fsum(DISCOUNT[rank] for rank in target_slots if rank <= cutoff) / target_width
                  for cutoff in ITEM_CUTOFFS)
    return result, int(bool(listed)) * first


def read_raw(mode, root, records, multiplicities, allowed, claims, expected_run):
    shards = (11182, 11181) if mode == 'normal' else (USERS,)
    require(set(claims) == set(range(len(shards))), (mode, 'prediction manifest ranks'))
    matrix = np.empty((USERS, len(METRICS)), dtype=np.float64)
    first_ranks = np.empty(USERS, dtype=np.int64)
    seen = set()
    assignments, source_receipts = [], []
    diagnostics = {k: Counter() for k in ITEM_CUTOFFS}
    for rank, count in enumerate(shards):
        path = root / f'predictions.rank{rank}.jsonl'
        rank_path = root / f'rank{rank}.json'
        report = read(rank_path)
        require(report['status'] == 'PASS' and report['rank'] == rank and report['local_rank'] == rank,
                (mode, rank, 'rank terminal status'))
        require(report['world_size'] == len(shards) and report['completed_users'] == count,
                (mode, rank, 'rank world or user count'))
        require(report['RUN_COMMIT'] == expected_run, (mode, rank, 'run commit'))
        require(report['adapter_files']['adapter_model.bin']['sha256'] == FROZEN_ADAPTER,
                (mode, rank, 'original Beauty adapter identity'))
        require({k: v['sha256'] for k, v in report['data_identity'].items()} == FROZEN_DATA,
                (mode, rank, 'asset identity in rank report'))
        digest = sha256()
        assignment = []
        size = 0
        with path.open('rb') as stream:
            for line_number, raw in enumerate(stream, 1):
                digest.update(raw)
                size += len(raw)
                row = json.loads(raw)
                i = row['row_idx']
                where = (mode, rank, line_number, i)
                require(type(i) is int and 0 <= i < USERS and i not in seen, (where, 'row axis'))
                seen.add(i)
                assignment.append(i)
                user, item, target = records[i]
                require((row['user_id'], row['target_item_id'], row['target']) == (user, item, target),
                        (where, 'user / target'))
                require(row['rank'] == rank and row['prompt_id'] == 0, (where, 'rank / Prompt0'))
                sids, scores = [], []
                for beam, candidate in enumerate(row['candidates']):
                    require(candidate['generation_order'] == beam, (where, 'candidate generation order'))
                    _, marker, response = candidate['raw_decoded'].rpartition('Response:')
                    sid = response.strip().replace(' ', '')
                    require(marker and sid == candidate['sid'], (where, 'decoded candidate parsing'))
                    tokens = candidate['generated_token_ids']
                    require(len(tokens) >= 5 and tokens[4] == 2 and candidate['eos_position'] == 4,
                            (where, 'canonical EOS'))
                    require(all(token in (0, 2) for token in tokens[5:]), (where, 'padding after EOS'))
                    token_sid = ''.join(slot.get(token, '\x00') for slot, token in zip(allowed, tokens[:4], strict=True))
                    require(token_sid == sid, (where, 'canonical slot token IDs'))
                    score = candidate['sequence_score']
                    require(type(score) in (int, float), (where, 'numeric score'))
                    sids.append(sid)
                    scores.append(score)
                order, ranked = rank_beams(sids, scores)
                require(row['sorted_generation_order'] == order, (where, 'stable score sort'))
                require(row['relevance'] == [int(sid == target) for sid in ranked], (where, 'relevance'))
                matrix[i], first_ranks[i] = row_metrics(ranked, target, multiplicities)
                for cutoff, stats in diagnostics.items():
                    subset = ranked[:cutoff]
                    valid = sum(sid in multiplicities for sid in subset)
                    stats['candidates'] += len(subset)
                    stats['valid'] += valid
                    stats['invalid'] += len(subset) - valid
                    stats['duplicates'] += len(subset) - len(set(subset))
                    stats['colliding'] += sum(multiplicities[sid] > 1 for sid in subset)
        require(len(assignment) == count, (mode, rank, 'raw prediction count'))
        assignment_digest = sha256(compact(assignment)).hexdigest()
        require(assignment_digest == report['assignment_sha256'], (mode, rank, 'sampler identity'))
        claim = claims[rank]
        require((digest.hexdigest(), size) == (claim['sha256'], claim['bytes']), (mode, rank, 'prediction SHA/size'))
        require(claim['assignment_sha256'] == assignment_digest, (mode, rank, 'primary assignment SHA'))
        if mode == 'normal':
            require(digest.hexdigest() == FROZEN_NORMAL_PREDICTIONS[rank], ('Frozen Normal raw SHA', rank))
            require(assignment_digest == FROZEN_NORMAL_ASSIGNMENTS[rank], ('Frozen Normal assignment', rank))
        source_receipts.append({'rank': rank, 'rows': count, 'bytes': size, 'sha256': digest.hexdigest(),
                                'path': str(path), 'assignment_sha256': assignment_digest,
                                'rank_report': file_identity(rank_path)})
        assignments.append(assignment)
    require(len(seen) == USERS, (mode, 'full user coverage'))
    return matrix, first_ranks, assignments, source_receipts, diagnostics


def point_estimates(values):
    return np.array([math.fsum(column) / len(values) for column in values.T], dtype=np.float64)


def difference(actual, expected):
    require(np.shape(actual) == np.shape(expected), ('Incompatible array shapes', np.shape(actual), np.shape(expected)))
    require(np.isfinite(actual).all() and np.isfinite(expected).all(), 'Expected finite values')
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))


def check_primary(mode, root, report, matrix, records, first_ranks, diagnostics):
    identities = {name: file_identity(root / name) for name in FROZEN_NORMAL_PRIMARY}
    if mode == 'normal':
        require({name: result['sha256'] for name, result in identities.items()} == FROZEN_NORMAL_PRIMARY,
                'Original Normal evaluation identity')
    with np.load(root / 'per_user_metrics.npz', allow_pickle=False) as saved:
        require(saved['values'].dtype == np.float64 and saved['keys'].tolist() == METRICS, (mode, 'NPZ metric axis'))
        require(np.array_equal(saved['row_idx'], np.arange(USERS)), (mode, 'NPZ row indices'))
        require(str(saved['user_axis_sha256']) == FROZEN_AXIS, (mode, 'NPZ user axis'))
        stored = saved['values']
    with (root / 'per_user.csv').open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        require(len(reader.fieldnames) == len(set(reader.fieldnames)), (mode, 'CSV duplicated field'))
        for index, (row, expected) in enumerate(zip(reader, records, strict=True)):
            actual = tuple(row[key] for key in ('user_id', 'target_item_id', 'target_sid'))
            require(actual == expected and int(row['row_idx']) == index, (mode, 'CSV user/target', index))
            require(row['user_axis_sha256'] == FROZEN_AXIS and int(row['first_target_rank']) == first_ranks[index],
                    (mode, 'CSV axis/rank', index))
            require(np.array_equal([float(row[key]) for key in METRICS], stored[index]), (mode, 'CSV/NPZ', index))
    point = point_estimates(matrix)
    maximum = difference(matrix, stored)
    mean_maximum = difference(point, [report['metrics'][key] for key in METRICS])
    require(max(maximum, mean_maximum) <= TOLERANCE, (mode, 'recomputed metrics differ', maximum, mean_maximum))
    for cutoff, counts in diagnostics.items():
        actual = report['sid_statistics'][str(cutoff)]
        require(all(actual[key] == number for key, number in counts.items()), (mode, 'candidate anomaly counts', cutoff))
    return point, {'max_per_user_delta': maximum, 'max_point_delta': mean_maximum,
                   'metrics': dict(zip(METRICS, point.tolist(), strict=True)),
                   'candidate_diagnostics': {str(k): dict(stats) for k, stats in diagnostics.items()},
                   'primary_files': identities}


def bootstrap(matrices, replicates=REPLICATES):
    """Bootstrap via occurrence-count weights, then subtract condition means."""
    conditions, n, metric_count = matrices.shape
    by_user = np.ascontiguousarray(matrices.transpose((1, 0, 2))).reshape(n, conditions * metric_count)
    means = np.empty((replicates, conditions, metric_count), dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    sample_identity = sha256()
    for begin in range(0, replicates, 32):
        stop = min(begin + 32, replicates)
        weights = np.empty((stop - begin, n), dtype=np.float64)
        for row in weights:
            indices = rng.integers(n, size=n, dtype=np.int64)
            sample_identity.update(indices.astype('<i8', copy=False).tobytes())
            row[:] = np.bincount(indices, minlength=n)
        block = (weights @ by_user) / n
        means[begin:stop] = block.reshape(stop - begin, conditions, metric_count)
    draws = np.stack([means[:, right, :] - means[:, left, :] for left, right in PAIRS])
    return draws, sample_identity.hexdigest()


def linear_bounds(draws):
    ordered = np.sort(draws, axis=1)
    bounds = []
    for percentile in (0.025, 0.975):
        position = (ordered.shape[1] - 1) * percentile
        lower = math.floor(position)
        share = position - lower
        bounds.append((1 - share) * ordered[:, lower] + share * ordered[:, math.ceil(position)])
    return np.asarray(bounds)


def verify_pairs(root, matrices, points):
    summary_path, arrays_path = root / 'summary.json', root / 'arrays.npz'
    summary = read(summary_path)
    settings = summary['bootstrap']
    require(summary['status'] == 'PASS_V2_PAIRED' and summary['users'] == USERS, 'Paired status/users')
    require(summary['user_axis_sha256'] == FROZEN_AXIS, 'Paired axis')
    require(summary['shuffle_seed'] == 42 and summary['identity_policy'] == 'allow_identity'
            and summary['protocol'] == 'GRID-LETTER-SHUFFLE-V2', 'Paired research protocol')
    require(settings['rng'] == 'NumPy PCG64' and settings['seed'] == BOOTSTRAP_SEED,
            'Paired RNG/seed')
    require(settings['replacement'] is True and settings['replicates'] == REPLICATES,
            'Paired sample definition')
    require(settings['percentile_method'] == 'linear' and settings['bounds'] == [2.5, 97.5]
            and settings['multiple_comparison_adjustment'] is False, 'Paired interval definition')
    draws, digest = bootstrap(matrices)
    require(settings['draw_indices_sha256_little_endian'] == digest, 'Paired sample stream identity')
    estimates = linear_bounds(draws)
    with np.load(arrays_path, allow_pickle=False) as arrays:
        require(arrays['keys'].tolist() == METRICS and arrays['condition'].tolist() == list(MODES)
                and arrays['comparison'].tolist() == list(PAIR_NAMES), 'Paired condition/metric axes')
        require(np.array_equal(arrays['row_idx'], np.arange(USERS)), 'Paired array user rows')
        require(str(arrays['user_axis_sha256']) == FROZEN_AXIS, 'Paired array user identity')
        for key in ('per_user_values', 'per_user_differences', 'bootstrap_difference_means'):
            require(arrays[key].dtype == np.float64, ('Float64 result storage', key))
        max_values = difference(arrays['per_user_values'], matrices)
        expected_differences = np.stack([matrices[b] - matrices[a] for a, b in PAIRS])
        max_differences = difference(arrays['per_user_differences'], expected_differences)
        max_draws = difference(arrays['bootstrap_difference_means'], draws)
    max_bounds = max_points = 0.0
    require(set(summary['metrics']) == set(MODES), 'Paired condition table')
    for mode, point in zip(MODES, points, strict=True):
        actual = summary['metrics'][mode]
        require(set(actual) == set(METRICS), ('Paired table metric keys', mode))
        max_points = max(max_points, difference([actual[k] for k in METRICS], point))
    require(set(summary['comparisons']) == set(PAIR_NAMES), 'Paired comparisons')
    for pair, (name, (left, right)) in enumerate(zip(PAIR_NAMES, PAIRS, strict=True)):
        comparison = summary['comparisons'][name]
        require([row['metric'] for row in comparison] == METRICS, (name, 'full comparison rows'))
        for metric, row in enumerate(comparison):
            baseline, intervention = points[left, metric], points[right, metric]
            delta = intervention - baseline
            reference = (baseline, intervention, delta, 100 * delta)
            observed = (row['left'], row['right'], row['absolute_difference'], row['percentage_point_difference'])
            max_points = max(max_points, difference(observed, reference))
            if baseline:
                max_points = max(max_points, difference(row['relative_change_fraction'], delta / baseline))
            else:
                require(row['relative_change_fraction'] is None, (name, row['metric'], 'zero baseline'))
            max_bounds = max(max_bounds, difference((row['difference_CI95_low'], row['difference_CI95_high']),
                                                    estimates[:, pair, metric]))
    normal_changes = summary['changes_from_normal']
    normal_keys = ((mode, metric) for mode in MODES for metric in METRICS)
    for (mode, metric), row in zip(normal_keys, normal_changes, strict=True):
        require((row['condition'], row['metric']) == (mode, metric), 'Normal-relative table order')
        original, baseline = summary['metrics'][mode][metric], summary['metrics']['normal'][metric]
        delta = original - baseline
        require((row['value'], row['normal'], row['delta'], row['percentage_point_difference'], row['relative_change_fraction'])
                == (original, baseline, delta, 100 * delta, delta / baseline if baseline else None), (mode, metric, 'Normal-relative table'))
    metrics_csv, paired_csv, changes_csv = (root / name for name in ('metrics.csv', 'paired.csv', 'changes_from_normal.csv'))
    with metrics_csv.open(newline='', encoding='utf-8') as stream:
        for mode, row in zip(MODES, csv.DictReader(stream), strict=True):
            require(row['condition'] == mode and all(float(row[key]) == summary['metrics'][mode][key] for key in METRICS),
                    (mode, 'condition CSV metric values/order'))
    comparisons = ((name, row) for name in PAIR_NAMES for row in summary['comparisons'][name])
    with paired_csv.open(newline='', encoding='utf-8') as stream:
        for (name, expected), row in zip(comparisons, csv.DictReader(stream), strict=True):
            require(row['comparison'] == name and row['metric'] == expected['metric'], (name, 'paired CSV order'))
            for key in ('left', 'right', 'absolute_difference', 'percentage_point_difference', 'relative_change_fraction',
                        'difference_CI95_low', 'difference_CI95_high'):
                actual = float(row[key]) if row[key] else None
                require(actual == expected[key], (name, row['metric'], key, 'paired CSV precision'))
    with changes_csv.open(newline='', encoding='utf-8') as stream:
        for expected, row in zip(normal_changes, csv.DictReader(stream), strict=True):
            require((row['condition'], row['metric']) == (expected['condition'], expected['metric']), 'Normal-relative CSV order')
            for key in ('value', 'normal', 'delta', 'percentage_point_difference', 'relative_change_fraction'):
                actual = float(row[key]) if row[key] else None
                require(actual == expected[key], (row['condition'], row['metric'], key, 'Normal-relative CSV precision'))
    maxima = {'per_user': max_values, 'per_user_pair_differences': max_differences,
              'bootstrap_draws': max_draws, 'point_estimates': max_points, 'paired_CI_bounds': max_bounds}
    require(max(maxima.values()) <= TOLERANCE, ('Paired arrays or summary differ', maxima))
    return {'comparisons': len(PAIRS) * len(METRICS), 'max_differences': maxima,
            'bootstrap_sample_sha256': digest, 'files': [file_identity(path) for path in
                                                        (summary_path, arrays_path, metrics_csv, paired_csv, changes_csv)]}


def selftest():
    sids = ['collision', 'invalid', 'target', 'target'] + ['other'] * 16
    order, ranked = rank_beams(sids, [2.0, 1.0, 1.0, 1.0] + [0.0] * 16)
    require(order == list(range(20)), 'Stable score fixture')
    metrics, first = row_metrics(ranked, 'target', {'collision': 3, 'target': 2, 'other': 1})
    require(first == 3 and metrics[4] == DISCOUNT[3] + DISCOUNT[4], 'Repeated SID NDCG fixture')
    require(metrics[10] == 0.5 and metrics[13] == DISCOUNT[5] / 2, 'Invalid and collision slot fixture')
    require(not any(row_metrics(ranked, 'absent', {'absent': 1})[0]), 'Target absent fixture')
    fixture = np.arange(5 * 6 * 3, dtype=np.float64).reshape(5, 6, 3) / 31
    draws, digest = bootstrap(fixture, replicates=11)
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    sample = rng.integers(0, fixture.shape[1], size=(11, fixture.shape[1]), dtype=np.int64)
    expected = np.stack([(fixture[b] - fixture[a])[sample].mean(axis=1) for a, b in PAIRS])
    require(digest == sha256(sample.astype('<i8', copy=False).tobytes()).hexdigest(), 'Bootstrap fixture index stream')
    require(difference(draws, expected) <= TOLERANCE, 'Bootstrap fixture draws')
    require(difference(linear_bounds(draws), np.percentile(expected, [2.5, 97.5], axis=1, method='linear')) <= TOLERANCE,
            'Percentile fixture')
    return 'PASS'


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--selftest', action='store_true')
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--condition', action='append', nargs=3, metavar=('MODE', 'TEST_DIR', 'PRIMARY_DIR'))
    parser.add_argument('--v2-run-commit')
    parser.add_argument('--paired-dir', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.selftest:
        print(json.dumps({'selftest': selftest()}))
        return
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Review requires no visible GPU')
    require(args.condition and args.data_dir and args.output, 'Use --data-dir, --condition and --output')
    require(not args.output.exists(), ('Review receipt already exists', args.output))
    modes = tuple(condition[0] for condition in args.condition)
    if args.paired_dir:
        require(modes == MODES, 'Paired review requires all five ordered conditions')
        require(args.v2_run_commit and len(args.v2_run_commit) == 40
                and all(c in '0123456789abcdef' for c in args.v2_run_commit), 'Expected full V2 Git commit')
    else:
        require(modes == ('normal',), 'Normal-only review requires only the archived Normal condition')
    started = time.monotonic()
    records, sizes, allowed, identities = catalog(args.data_dir)
    results, matrices, points = {}, [], []
    normal_assignment = None
    for mode, test, primary in args.condition:
        primary = Path(primary)
        report = read(primary / 'metrics.json')
        expected_run = FROZEN_RUN if mode == 'normal' else args.v2_run_commit
        require(report['users'] == USERS and report['user_axis_sha256'] == FROZEN_AXIS,
                (mode, 'primary user axis'))
        claims = {record['rank']: record for record in report['prediction_files']}
        require(len(claims) == len(report['prediction_files']), (mode, 'duplicate prediction claims'))
        matrix, first, assignments, receipts, diag = read_raw(mode, Path(test), records, sizes, allowed, claims, expected_run)
        if mode == 'normal':
            normal_assignment = [None] * USERS
            normal_assignment[::2], normal_assignment[1::2] = assignments
        else:
            require(assignments[0] == normal_assignment, (mode, 'lossless odd-axis reconstruction'))
        point, checks = check_primary(mode, primary, report, matrix, records, first, diag)
        results[mode] = {'prediction_sources': receipts, **checks}
        matrices.append(matrix)
        points.append(point)
        print(json.dumps({'condition': mode, 'users': len(matrix), 'max_per_user_delta': checks['max_per_user_delta']}), flush=True)
    paired = verify_pairs(args.paired_dir, np.asarray(matrices), np.asarray(points)) if args.paired_dir else None
    result = {'status': 'PASS_INDEPENDENT_BEAUTY_PAIRED_REVIEW' if paired else 'PASS_INDEPENDENT_BEAUTY_NORMAL_REVIEW',
              'tolerance': TOLERANCE, 'selftest': selftest(), 'source': file_identity(Path(__file__)),
              'users': USERS, 'user_axis_sha256': FROZEN_AXIS, 'data': identities, 'conditions': results,
              'paired': paired, 'numpy': np.__version__, 'elapsed_wall_seconds': time.monotonic() - started}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'output': str(args.output), 'seconds': result['elapsed_wall_seconds']}))


if __name__ == '__main__':
    main()
