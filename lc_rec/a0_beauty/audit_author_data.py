#!/usr/bin/env python3
"""Read fixed upstream Beauty assets and report their identities and structure."""
import collections
import hashlib
import json
import urllib.request

COMMIT = '8d0154e28de37dbb6e24871c508ad8ddb1921cda'
BLOBS = {'index': '6711ba430c7a6889e60f908554f72aae8a038039',
         'inter': '9378c6e88224ebd22e57325d89edb8cfb73291ac',
         'item': '098205baf4a483b818507dc98c4b9bc9151a69f0'}
BASE = f'https://raw.githubusercontent.com/HonghuiBao2000/LETTER/{COMMIT}/data/Beauty/Beauty.'


def main():
    assets, identities = {}, {}
    for kind, expected in BLOBS.items():
        url = BASE + kind + '.json'
        with urllib.request.urlopen(url, timeout=90) as response:
            raw = response.read()
        blob = hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest()
        if blob != expected:
            raise ValueError(f'{kind}: Git blob mismatch: {blob}')
        assets[kind] = json.loads(raw)
        identities[kind] = {'url': url, 'bytes': len(raw),
                            'sha256': hashlib.sha256(raw).hexdigest(), 'git_blob_sha1': blob}
    index, inter, metadata = (assets[k] for k in ('index', 'inter', 'item'))
    keys = set(index)
    references = {str(item) for sequence in inter.values() for item in sequence}
    if keys != set(metadata) or references - keys:
        raise ValueError('Item keys or interaction references do not match')
    lengths = collections.Counter(map(len, index.values()))
    if set(lengths) != {4}:
        raise ValueError('SID lengths differ from four')
    groups = collections.Counter(tuple(sid) for sid in index.values())
    seq_lengths = [len(seq) for seq in inter.values()]
    axis = json.dumps(list(inter), ensure_ascii=False, separators=(',', ':')).encode()
    result = {'source_commit': COMMIT, 'assets': identities, 'items': len(index),
              'users': len(inter), 'interactions': sum(seq_lengths),
              'sequence_length_min': min(seq_lengths), 'sequence_length_max': max(seq_lengths),
              'interaction_missing_item_keys': len(references - keys),
              'items_not_in_interactions': len(keys - references),
              'metadata_keys_match_index': keys == set(metadata),
              'sid_lengths': dict(lengths),
              'sid_layer_token_counts': [len({sid[i] for sid in index.values()}) for i in range(4)],
              'sid_distinct_tokens': len({token for sid in index.values() for token in sid}),
              'unique_sids': len(groups), 'collision_groups': sum(n > 1 for n in groups.values()),
              'collision_items': sum(n for n in groups.values() if n > 1),
              'train_prefix_count_derived': sum(max(0, n - 3) for n in seq_lengths),
              'validation_examples_if_two_per_user': 2 * len(inter),
              'user_axis_sha256_compact_json': hashlib.sha256(axis).hexdigest(),
              'limitations': ['Key matching does not independently establish metadata semantic correctness.',
                              'Counts are derived, not a run of the actual SeqRec loader.',
                              'Tokenizer, GRID record identity and model have not been checked.']}
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
