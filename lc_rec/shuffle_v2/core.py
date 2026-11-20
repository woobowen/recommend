"""V2 permutes complete token IDs within verified visible history occurrences."""
from functools import lru_cache
import hashlib
from itertools import permutations

from lc_rec.shuffle.core import canonical_bytes, canonical_sha256, integer
from lc_rec.shuffle.core import transform as validate_v1_off

PROTOCOL = 'GRID-LETTER-SHUFFLE-V2'
IDENTITY_POLICY = 'allow_identity'
SEED = 42
MODES = {
    'shuffle2': (2, 3),
    'shuffle3a': (0, 1, 2),
    'shuffle3b': (1, 2, 3),
    'shuffle4': (0, 1, 2, 3),
}
SERIALIZATION = {'encoding': 'utf-8', 'ensure_ascii': True, 'sort_keys': True,
                 'separators': [',', ':'], 'allow_nan': False}


@lru_cache(maxsize=3)
def catalog(m):
    if m not in (2, 3, 4):
        raise ValueError('Expected two, three, or four active positions')
    return tuple(permutations(range(m)))


def _digest(payload):
    return hashlib.sha256(payload).digest()


def draw(m, row_idx, visible_item_idx):
    m, row_idx, visible_item_idx = map(integer, (m, row_idx, visible_item_idx))
    choices = catalog(m)
    limit = ((1 << 256) // len(choices)) * len(choices)
    counter = 0
    while True:
        payload = canonical_bytes([PROTOCOL, SEED, m, row_idx, visible_item_idx, counter])
        value = int.from_bytes(_digest(payload), 'big', signed=False)
        if value < limit:
            index = value % len(choices)
            return index, choices[index], counter
        counter += 1


def full_permutation(active, relative):
    full = list(range(4))
    for j, pos in enumerate(active):
        full[pos] = active[relative[j]]
    return full


def mode_metadata(mode):
    return {'protocol': PROTOCOL, 'identity_policy': IDENTITY_POLICY, 'mode': mode,
            'seed': SEED, 'active_positions': list(MODES[mode])}


def transform(input_ids, mask, groups, all_sid_ids, mode, row_idx):
    active = MODES[mode]
    # The frozen OFF path checks exact encoded SID boundaries, padding, the full
    # SID vocabulary and item identity/order without sampling any permutation.
    raw, verified = validate_v1_off(input_ids, mask, groups, all_sid_ids, 0, row_idx)
    result = raw.copy()
    records = verified['groups']
    for occurrence, group in enumerate(records):
        index, relative, rejected = draw(len(active), row_idx, occurrence)
        full = full_permutation(active, relative)
        original = group['original_ids']
        shuffled = [original[src] for src in full]
        for pos, token in zip(group['positions'], shuffled):
            result[pos] = token
        group.update(permutation=full, relative_permutation=list(relative), permutation_id=index,
                     rejection_counter=rejected, shuffled_ids=shuffled, nonidentity=index != 0,
                     tokens_changed=original != shuffled)
    return result, {**mode_metadata(mode), 'row_idx': row_idx,
                    'history_group_count': len(records), 'input_changed': result != raw,
                    'original_sha256': verified['original_sha256'],
                    'attention_mask_sha256': verified['attention_mask_sha256'],
                    'shuffled_sha256': canonical_sha256(result), 'groups': records}
