#!/usr/bin/env python3
"""Read the author Instruments assets and make an auditable numeric pairing projection."""
import argparse
import base64
from collections import Counter
import datetime
import gzip
import hashlib
import json
from pathlib import Path

AUTHOR_ZIP_SHA256 = '96aad0cf1496a8ffef31bf1ca3a0d1033ca49473dff46b2c2dc27763224f9453'
EXPECTED = {
    'user2id': 'ae170e33f5f1e71ae21d72760e10951cac16dacaa5a20f6ede0aa26ce014bc15',
    'user.json': '8fe65b5de6222e665d12052c8771f8a314813323b8470f54f2ca2920bf7af5b6',
    'inter.json': '0a11d47791d90483005da93c36d826679e20e4592d1d2b86518d5f56f218e743',
    'index.json': 'b94d16398a178220a117abda5effd5508069beaccc2b8110eab25ae50f2af600',
    'item.json': '1d9938e576a8d964e6b977a2b63aa56402840a788eaa6685f26f0d910a7133f2',
    'item2id': '76380000fb809cf19e841c69163310ec2d32b7b578277e2b659fbd1e72714688',
}
FIELDS = ['uid', 'interaction', 'train_target', 'train_history', 'test_target', 'test_history', 'explicit_preference_count']


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def integer_axis(data):
    pairs = [line.split() for line in data.decode().splitlines()]
    assert all(len(pair) == 2 for pair in pairs)
    keys = {row[0] for row in pairs}
    values = {int(row[1]) for row in pairs}
    assert len(keys) == len(values) == len(pairs)
    return values


def summarize(rows):
    count = Counter()
    seen_users = set()
    items = set()
    for user, interaction, train_target, train_history, test_target, test_history, preference_count in rows:
        assert user not in seen_users and len(interaction) > 3 and preference_count > 0
        seen_users.add(user)
        items.update(interaction)
        count['interactions'] += len(interaction)
        count['train_item_history_exact'] += (train_target == interaction[-3] and train_history == interaction[:-3])
        count['train_item_history_mismatch'] += (train_target != interaction[-3] or train_history != interaction[:-3])
        count['test_item_history_mismatch'] += (test_target != interaction[-1] or test_history != interaction[:-1])
        count['train_aux_target_is_validation'] += train_target == interaction[-2]
        count['train_aux_target_is_test'] += train_target == interaction[-1]
        count['train_aux_visible_recent20_contains_validation'] += interaction[-2] in train_history[-20:]
        count['train_aux_visible_recent20_contains_test'] += interaction[-1] in train_history[-20:]
        count['explicit_preferences_present'] += preference_count > 0
    return {'users': len(seen_users), 'distinct_interaction_items': len(items), **dict(count)}


def capture(root):
    data = {}
    records = {}
    for name, expected in EXPECTED.items():
        path = root / f'Instruments.{name}'
        payload = path.read_bytes()
        assert sha256(payload) == expected, path
        records[name] = {'server_path': str(path), 'sha256': expected, 'bytes': len(payload)}
        data[name] = json.loads(payload) if name.endswith('.json') else integer_axis(payload)
    archive = root.parent.parent / 'Instruments.zip'
    original = archive.read_bytes()
    assert sha256(original) == AUTHOR_ZIP_SHA256 and len(original) == 13612638
    inter, index, metadata, user = (data[name + '.json'] for name in ('inter', 'index', 'item', 'user'))
    train = user['user_vague_intention']['train']
    test = user['user_vague_intention']['test']
    preferences = user['user_explicit_preference']
    assert set(inter) == set(train) == set(test) == set(preferences)
    assert {int(key) for key in inter} == data['user2id']
    assert set(index) == set(metadata) and {int(key) for key in index} == data['item2id']
    counts = Counter(tuple(tokens) for tokens in index.values())
    assert all(len(codes) == 4 for codes in counts)
    rows = []
    for user_id, sequence in inter.items():
        tr, te = train[user_id], test[user_id]
        ids = [*sequence, tr['item'], *tr['inters'], te['item'], *te['inters']]
        assert all(type(item) is int and item in data['item2id'] for item in ids)
        rows.append([int(user_id), sequence, tr['item'], tr['inters'], te['item'], te['inters'], len(preferences[user_id])])
    projection = b''.join(json.dumps(row, separators=(',', ':')).encode() + b'\n' for row in rows)
    compressed = gzip.compress(projection, mtime=0)
    return {'captured_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'original_members': records,
            'author_zip': {'server_path': str(archive), 'sha256': AUTHOR_ZIP_SHA256, 'bytes': len(original)},
            'item_axis': {'metadata_index_and_mapping_identical': True, 'mapped_items': len(index),
                          'distinct_canonical_sids': len(counts), 'colliding_sid_groups': sum(v > 1 for v in counts.values())},
            'projection_fields': FIELDS, 'summary': summarize(rows),
            'projection_sha256': sha256(projection), 'projection_gzip_sha256': sha256(compressed),
            'projection_gzip_base64': base64.b64encode(compressed).decode()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--capture', type=Path)
    group.add_argument('--verify-projection', type=Path)
    args = parser.parse_args()
    if args.capture is not None:
        result = capture(args.capture)
    else:
        content = args.verify_projection.read_bytes()
        raw = gzip.decompress(content)
        result = {'projection_sha256': sha256(raw), 'projection_gzip_sha256': sha256(content),
                  'projection_fields': FIELDS, 'summary': summarize(json.loads(row) for row in raw.splitlines())}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
