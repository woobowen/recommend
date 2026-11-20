"""Recompute every saved per-user metric directly from archived LC-Rec prediction shards."""
import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path
import re

import numpy as np

from verify_saved_metrics import BEAUTY, EVIDENCE, G3, read_matrix

DISCOUNTS = tuple(0.0 if rank == 0 else 1 / math.log2(rank + 1) for rank in range(21))
SID = re.compile(r'(?:<[abcd]_\d+>){4}')


def input_paths(repo, condition):
    family = condition.split('/')[0]
    if family == 'instruments':
        folder = repo / 'lc_rec/letter/vendor/LETTER/data/Instruments'
        return folder / 'Instruments.index.json', folder / 'Instruments.inter.json'
    folder = EVIDENCE / 'letter-beauty-author/Beauty'
    if family == 'g3-beauty':
        index = repo / 'lc_rec/a0_beauty/grid_sid_preparation/g3_conversion/Beauty.g3.index.json'
    else:
        index = folder / 'Beauty.index.json'
    return index, folder / 'Beauty.inter.json'


def expected_matrix(repo, condition):
    family, *rest = condition.split('/')
    mode = rest[-1]
    if family == 'g3-beauty':
        return read_matrix(repo / G3 / 'original' / mode / 'primary/per_user_metrics.npz')
    if family == 'letter-beauty':
        return read_matrix(repo / BEAUTY / 'results/public_metric_matrices.npz', mode)
    model = rest[0]
    if mode == 'normal':
        return read_matrix(EVIDENCE / 'instruments-normal' / model / 'per_user_metrics.npz')
    prefix = 'letter-a0-20261112' if model == 'a0' else 'letter-a1-q06-20261113'
    folder = f'{prefix}-{mode}-instruments-seed42-20261114-r0'
    return read_matrix(repo / 'lc_rec/shuffle/results' / folder / 'evaluation/primary/per_user_metrics.npz')


def score(sids, target, sizes):
    hits = [rank for rank, sid in enumerate(sids, 1) if sid == target]
    first = hits[0] if hits else 21
    offset = sum(sizes.get(sid, 1) for sid in sids[:first - 1])
    group_size = sizes[target]
    scores = {f'hit@{k}': first <= k for k in (1, 5, 10, 20)}
    for k in (5, 10, 20):
        scores[f'ndcg@{k}'] = math.fsum(DISCOUNTS[r] for r in hits if r <= k)
        scores[f'mrr@{k}'] = 1 / first if first <= k else 0.0
        visible = max(0, min(group_size, k - offset)) if hits else 0
        scores[f'ItemHit@{k}'] = visible / group_size
        scores[f'ItemNDCG@{k}'] = math.fsum(DISCOUNTS[r] for r in range(offset + 1, offset + visible + 1)) / group_size
    return scores


def verify_condition(raw, repo, condition):
    index_path, inter_path = input_paths(repo, condition)
    sid_for = {item: ''.join(tokens) for item, tokens in json.loads(index_path.read_bytes()).items()}
    sizes = Counter(sid_for.values())
    expected_users = [(uid, str(history[-1])) for uid, history in json.loads(inter_path.read_bytes()).items()]
    keys, expected, axis = expected_matrix(repo, condition)
    seen = set()
    matrix = np.full_like(expected, np.nan)
    files = sorted((raw / condition).glob('predictions.rank*.jsonl.gz'))
    assert files and len(expected_users) == len(matrix), condition
    for rank, path in enumerate(files):
        with gzip.open(path, 'rb') as stream:
            for line in stream:
                row = json.loads(line)
                i = row['row_idx']
                assert i not in seen and row['rank'] == rank and row['prompt_id'] == 0, (path, i)
                seen.add(i)
                uid, item = expected_users[i]
                target = sid_for[item]
                assert (row['user_id'], row['target_item_id'], row['target']) == (uid, item, target), (path, i)
                candidates = row['candidates']
                assert len(candidates) == 20, (path, i)
                order = sorted(range(20), key=lambda j: candidates[j]['sequence_score'], reverse=True)
                sids = []
                for j in order:
                    candidate = candidates[j]
                    prefix, response = candidate['raw_decoded'].rsplit('Response:', 1)
                    sid = response.strip().replace(' ', '')
                    assert candidate['generation_order'] == j and math.isfinite(candidate['sequence_score']), (path, i)
                    assert sid == candidate['sid'] and SID.fullmatch(sid), (path, i, 'candidate scope')
                    assert 'The user has interacted with items ' in prefix, (path, i, 'Prompt0')
                    sids.append(sid)
                assert order == row['sorted_generation_order'] and [int(sid == target) for sid in sids] == row['relevance'], (path, i)
                scores = score(sids, target, sizes)
                matrix[i, :] = [scores[key] for key in keys]
    assert seen == set(range(len(expected))) and np.isfinite(matrix).all(), condition
    np.testing.assert_allclose(matrix, expected, atol=1e-13, rtol=0, err_msg=condition)
    return {'condition': condition, 'shards': len(files), 'rows': len(expected), 'candidate_rows': len(expected) * 20,
            'grouped_sid_count': len(sizes), 'user_axis_sha256': axis,
            'per_user_metric_values': expected.size, 'maximum_abs_error': float(np.abs(matrix - expected).max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--repo', type=Path, default=EVIDENCE.parents[2])
    args = parser.parse_args()
    conditions = sorted({p.parent.relative_to(args.raw).as_posix() for p in args.raw.rglob('*.jsonl.gz')})
    assert len(conditions) == 12, conditions
    print(json.dumps({'status': 'PASS', 'atol': 1e-13, 'conditions': [verify_condition(args.raw, args.repo, c) for c in conditions]}, indent=2))


if __name__ == '__main__':
    main()
