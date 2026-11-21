#!/usr/bin/env python3
"""Verify that v1.5 LETTER compact records support the prior frozen audit results."""
import argparse
from collections import Counter
from itertools import permutations
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

from compact_predictions import canonical, file_digest, iter_rows

ATOL = 1e-13
RANK_CUTOFFS = (5, 10, 20)
V1 = 'GRID-LETTER-SHUFFLE-V1'


def read(path):
    return json.loads(path.read_bytes())


def frozen_results(frozen, repo, condition):
    from verify_saved_metrics import A0, A1

    family, *rest = condition.split('/')
    mode = rest[-1]
    if family == 'g3-beauty':
        results = repo / frozen.G3
        primary = results / 'original' / mode / 'primary'
        pair = results / 'paired' / mode
    elif family == 'letter-beauty':
        results = repo / frozen.BEAUTY
        primary = results / 'original' / f'{mode}-primary'
        pair = results / ('original/s3' if mode == 's3' else 'results/s4')
    elif mode == 'normal':
        primary = repo / (A0 if rest[0] == 'a0' else A1)
        pair = None
    else:
        prefix = 'letter-a0-20261112' if rest[0] == 'a0' else 'letter-a1-q06-20261113'
        results = repo / 'lc_rec/shuffle/results' / f'{prefix}-{mode}-instruments-seed42-20261114-r0/evaluation'
        primary, pair = results / 'primary', results / 'paired'
    return primary / 'metrics.json', pair


def verify_v1(row, catalogs):
    shuffle = row.get('shuffle')
    if shuffle is None:
        return 0
    evidence = shuffle['evidence']
    n, seed, row_id = shuffle['n'], shuffle['seed'], row['row_idx']
    assert evidence['seed'] == seed == 42 and evidence['protocol_version'] == V1
    assert evidence['row_idx'] == row_id == shuffle['row_idx']
    catalog = catalogs[n]
    limit = ((1 << 256) // len(catalog)) * len(catalog)
    for occurrence, group in enumerate(evidence['groups']):
        assert group['visible_history_item_index'] == occurrence
        counter = 0
        while True:
            key = [V1, seed, n, row_id, occurrence, counter]
            value = int.from_bytes(hashlib.sha256(canonical(key)).digest(), 'big')
            if value < limit:
                permutation_id = value % len(catalog)
                break
            counter += 1
        assert group['rejection_counter'] == counter and group['permutation_id'] == permutation_id
        expected = list(catalog[permutation_id]) + ([3] if n == 3 else [])
        assert group['permutation'] == expected and group['nonidentity'] is True
        assert group['shuffled_ids'] == [group['original_ids'][j] for j in expected]
        assert sorted(group['original_ids']) == sorted(group['shuffled_ids'])
    return len(evidence['groups'])


def check_condition(frozen, repo, members, record):
    condition, path = record['condition'], members / record['member']
    assert path.stat().st_size == record['bytes'] and file_digest(path) == record['sha256']
    index, interactions = frozen.input_paths(repo, condition)
    sid_for = {item: ''.join(tokens) for item, tokens in read(index).items()}
    collision = Counter(sid_for.values())
    users = [(uid, str(seq[-1])) for uid, seq in read(interactions).items()]
    keys, expected, axis = frozen.expected_matrix(repo, condition)
    assert len(expected) == len(users)
    matrix = np.full_like(expected, np.nan)
    rankings = [None] * len(users)
    hist = Counter()
    counts = {k: Counter() for k in RANK_CUTOFFS}
    coverage = {k: set() for k in RANK_CUTOFFS}
    catalogs = {n: tuple(p for p in permutations(range(n)) if p != tuple(range(n))) for n in (3, 4)}
    groups = 0
    rank_receipts = []
    idx = 0
    end = record['original_shards'][0]['rows']
    h = hashlib.sha256()
    for row_number, row in enumerate(iter_rows(path), 1):
        h.update(canonical(row) + b'\n')
        i = row['row_idx']
        assert 0 <= i < len(users) and rankings[i] is None
        assert row['rank'] == idx and row['prompt_id'] == 0
        uid, item = users[i]
        target = sid_for[item]
        assert (row['user_id'], row['target_item_id'], row['target']) == (uid, item, target)
        rankings[i] = ranks = tuple(row['candidates'][j]['sid'] for j in row['sorted_generation_order'])
        first = ranks.index(target) + 1 if target in ranks else 'miss'
        hist[str(first)] += 1
        for k in RANK_CUTOFFS:
            top = ranks[:k]
            valid = sum(sid in collision for sid in top)
            duplicate = k - len(set(top))
            colliding = sum(collision.get(sid, 0) > 1 for sid in top)
            counts[k].update(candidates=k, valid=valid, invalid=k-valid, duplicates=duplicate, colliding=colliding,
                             users_invalid=valid<k, users_duplicate=duplicate>0, users_colliding=colliding>0)
            coverage[k].update(sid for sid in top if sid in collision)
        scores = frozen.score(ranks, target, collision)
        matrix[i, :] = [scores[key] for key in keys]
        groups += verify_v1(row, catalogs)
        if row_number == end:
            shard = record['original_shards'][idx]
            assert h.hexdigest() == shard['parsed_row_sha256']
            rank_receipts.append({'rank': idx, 'rows': shard['rows'], 'parsed_row_sha256': h.hexdigest()})
            h = hashlib.sha256()
            idx += 1
            if idx < len(record['original_shards']):
                end += record['original_shards'][idx]['rows']
    assert record['rows'] == len(users) == row_number == end and idx == len(record['original_shards'])
    assert np.isfinite(matrix).all()
    np.testing.assert_allclose(matrix, expected, atol=ATOL, rtol=0, err_msg=condition)
    metrics_path, pair_path = frozen_results(frozen, repo, condition)
    reference_metrics = read(metrics_path)
    assert len(users) == reference_metrics['users'] and reference_metrics['user_axis_sha256'] == axis
    error = max(abs(math.fsum(matrix[:, i]) / len(users) - reference_metrics['metrics'][k]) for i, k in enumerate(keys))
    assert error <= ATOL
    assert all(hist[key] == count for key, count in reference_metrics['first_target_rank_users'].items())
    for k in RANK_CUTOFFS:
        ref = reference_metrics['sid_statistics'][str(k)]
        assert all(val == ref[key] for key, val in counts[k].items()), (condition, k)
        assert len(coverage[k]) == ref['unique_valid_sids'] and len(collision) == ref['coverage_denominator']
    return {'condition': condition, 'rows': len(users), 'candidate_rows': len(users) * 20, 'verified_v1_groups': groups,
            'per_user_metric_cells': matrix.size, 'max_cell_error': float(np.max(np.abs(matrix - expected))),
            'max_metric_error': error, 'diagnostics_exact': True, 'rank_receipts': rank_receipts}, (matrix, keys, rankings, pair_path)


def check_pair(pair_path, normal, treatment):
    left, reference_keys, before, _ = normal
    right, keys, after, _ = treatment
    assert keys == reference_keys
    count = len(left)
    paired = read(pair_path / 'paired_comparison.json')
    ratings = paired['changes']
    assert paired['users'] == count == len(right) == ratings['users']
    assert ratings['top1_changed'] == sum(a[0] != b[0] for a, b in zip(before, after, strict=True))
    assert ratings['top20_ordered_changed'] == sum(a != b for a, b in zip(before, after, strict=True))
    assert ratings['top20_set_changed'] == sum(set(a) != set(b) for a, b in zip(before, after, strict=True))
    cutoffs = (1, *RANK_CUTOFFS)
    for cutoff in cutoffs:
        key = f'hit@{cutoff}'
        column = keys.index(key)
        transitions = Counter(f'{int(a)}->{int(b)}' for a, b in zip(left[:, column], right[:, column], strict=True))
        assert all(transitions[name] == val for name, val in ratings['hit_transitions'][key].items())
    draws = np.load(pair_path / 'paired_bootstrap_difference_means.npy', allow_pickle=False)
    assert draws.shape == (5000, len(keys))
    intervals = np.percentile(draws, (2.5, 97.5), axis=0, method='linear')
    for i, key in enumerate(keys):
        d = paired['metrics'][key]
        assert abs(d['absolute_difference'] - math.fsum(right[:, i] - left[:, i]) / count) <= ATOL
        assert abs(d['difference_CI95_low'] - intervals[0, i]) <= ATOL
        assert abs(d['difference_CI95_high'] - intervals[1, i]) <= ATOL
    return {'rows': count, 'metrics': len(keys), 'hit_transition_cutoffs': list(cutoffs),
            'ranking_change_counts_exact': True, 'archived_bootstrap_draws_per_metric': len(draws)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True, type=Path, help='Frozen LC-Rec delivery tree')
    parser.add_argument('--members', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / 'lc_rec/evidence/repository-delivery-20261120'))
    import verify_raw_predictions as frozen
    summary = read(args.members / 'manifest.json')
    conditions = []
    results = {}
    for record in summary['conditions']:
        checks, result = check_condition(frozen, repo, args.members, record)
        conditions.append(checks)
        results[record['condition']] = result
    pairs = {}
    for condition, result in results.items():
        if not condition.endswith('/normal'):
            pairs[condition] = check_pair(result[3], results[condition.rsplit('/', 1)[0] + '/normal'], result)
    assert len(conditions) == 12 and len(pairs) == 8
    print(json.dumps({'status': 'PASS', 'tolerance_abs': ATOL, 'conditions': conditions, 'pairs_against_normal': pairs}, indent=2))


if __name__ == '__main__':
    main()
