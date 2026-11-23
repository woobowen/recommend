"""GRID-LETTER-SHUFFLE-V1: output[j] = original[permutation[j]].

Hash compact UTF-8 JSON [protocol,seed,n,row,item,counter] with SHA256.
Interpret the digest unsigned big-endian. Reject values >= floor(2**256/M)*M;
otherwise select value % M from lexicographic permutations excluding identity.
No global RNG, process hash, execution rank, worker or batch enters the key.
Implementation supplied by shuffle_core agent through an explicit text bridge.
"""
from collections import Counter
from functools import lru_cache
import hashlib
import itertools
import json
import operator

PROTOCOL_VERSION = 'GRID-LETTER-SHUFFLE-V1'


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode()


def canonical_sha256(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def integer(value):
    if isinstance(value, bool) or operator.index(value) < 0:
        raise ValueError('Expected nonnegative integer, not bool')
    return int(value)


@lru_cache(maxsize=2)
def permutation_catalog(n):
    if n not in (3, 4):
        raise ValueError('Shuffle n must be 3 or 4')
    return tuple(p for p in itertools.permutations(range(n)) if p != tuple(range(n)))


def _digest256(payload):
    return hashlib.sha256(payload).digest()


def _draw(n, row_idx, item_idx, seed):
    n, row_idx, item_idx, seed = map(integer, (n, row_idx, item_idx, seed))
    catalog = permutation_catalog(n)
    limit = ((1 << 256) // len(catalog)) * len(catalog)
    counter = 0
    while True:
        payload = canonical_bytes([PROTOCOL_VERSION, seed, n, row_idx, item_idx, counter])
        value = int.from_bytes(_digest256(payload), 'big')
        if value < limit:
            index = value % len(catalog)
            return index, catalog[index], counter
        counter += 1


def permutation(n, row_idx, item_idx, seed=42):
    return _draw(n, row_idx, item_idx, seed)[1]


def transform(input_ids, attention_mask, expected_groups, all_sid_ids, n, row_idx, seed=42):
    n, row_idx, seed = map(integer, (n, row_idx, seed))
    if n not in (0, 3, 4):
        raise ValueError('Expected OFF=0, Shuffle3=3 or Shuffle4=4')
    ids = [integer(t) for t in input_ids]
    mask = list(attention_mask)
    if len(ids) != len(mask) or any(x not in (0, 1) for x in mask):
        raise ValueError('Invalid mask or input length')
    active = [i for i, x in enumerate(mask) if x]
    if active and len(active) != active[-1]-active[0]+1:
        raise ValueError('Internal attention-mask holes are not supported')
    vocab = {integer(t) for t in all_sid_ids}
    if not vocab:
        raise ValueError('SID vocabulary must be complete and nonempty')
    groups = [[integer(t) for t in g] for g in expected_groups]
    if any(len(g) != 4 or any(t not in vocab for t in g) for g in groups):
        raise ValueError('Expected exactly four registered SID symbols per item')
    positions = [i for i, t in enumerate(ids) if t in vocab]
    if any(not mask[i] for i in positions) or len(positions) != 4*len(groups):
        raise ValueError('Extra/missing SID symbols or SID padding')
    result, records = list(ids), []
    for item_idx, group in enumerate(groups):
        pos = positions[4*item_idx:4*item_idx+4]
        if pos != list(range(pos[0], pos[0]+4)) or [ids[i] for i in pos] != group:
            raise ValueError('History SID boundary, item identity or item order mismatch')
        index, selected, counter = _draw(n, row_idx, item_idx, seed) if n else (None, (), None)
        full = selected + ((3,) if n == 3 else ()) if n else (0, 1, 2, 3)
        shuffled = [group[j] for j in full]
        assert not n or full != (0, 1, 2, 3)
        assert Counter(group) == Counter(shuffled)
        assert n == 4 or group[3] == shuffled[3]
        for i, token in zip(pos, shuffled): result[i] = token
        records.append({'visible_history_item_index': item_idx, 'start': pos[0],
                        'positions': pos, 'permutation': list(full), 'permutation_id': index,
                        'rejection_counter': counter, 'original_ids': group,
                        'shuffled_ids': shuffled, 'nonidentity': full != (0, 1, 2, 3)})
    history = set(positions)
    assert all(ids[i] == result[i] for i in range(len(ids)) if i not in history)
    assert Counter(ids) == Counter(result) and len(ids) == len(result)
    assert n != 0 or result == ids
    return result, {'protocol_version': PROTOCOL_VERSION, 'seed': seed, 'shuffle_n': n,
                    'row_idx': row_idx, 'history_group_count': len(groups),
                    'input_changed': ids != result, 'nonhistory_unchanged': True,
                    'attention_mask_unchanged': True, 'sid_multiset_conserved': True,
                    'original_sha256': canonical_sha256(ids),
                    'shuffled_sha256': canonical_sha256(result),
                    'attention_mask_sha256': canonical_sha256(mask), 'groups': records}
