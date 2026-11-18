#!/usr/bin/env python3
"""Convert the pinned G3 tensor and audited bridge into a LETTER index."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

N = 12101
HASHES = {
    'g3': 'd41adcfb22a9a47d5e0424c068f4c74f33cc9fdbaf5db08540fbfa0e3d720e18',
    'bridge': '10a7ca66dba43b69232a7c96b48d3287df9314a70a9ec9083ff00acfcf2a8c4f',
    'item': '74aeddf911548aadfcd1dab326c16678ed4ac74457e874da300d63a94bdb2330',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f'duplicate JSON key: {key}')
        result[key] = value
    return result


def write_once(path, content):
    if path.exists():
        require(path.is_file() and path.read_bytes() == content, f'existing output differs: {path}')
    else:
        with path.open('xb') as output:
            output.write(content)


def main():
    parser = argparse.ArgumentParser()
    for name in ('g3', 'bridge', 'item', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    for name, expected in HASHES.items():
        require(sha(getattr(args, name)) == expected, f'{name} hash mismatch')
    tensor = torch.load(args.g3, map_location='cpu', weights_only=True)
    require(tensor.dtype == torch.int64 and tuple(tensor.shape) == (4, N), 'G3 shape or dtype')
    values = [set(tensor[position].tolist()) for position in range(4)]
    require(values == [set(range(256)), set(range(256)), set(range(256)), set(range(18))], 'G3 value sets')
    require(torch.unique(tensor.T, dim=0).shape[0] == N, 'G3 full SID collision')
    items = json.loads(args.item.read_text(), object_pairs_hook=unique_object)
    axis = set(map(str, range(N)))
    require(set(items) == axis, 'LETTER item axis')
    with args.bridge.open(newline='') as source:
        reader = csv.DictReader(source)
        require(reader.fieldnames == ['grid_item_id', 'sid_column', 'original_item_id', 'letter_item_id',
                                      'sid_1', 'sid_2', 'sid_3', 'sid_4'], 'bridge columns')
        rows = list(reader)
    require(len(rows) == N, 'bridge row count')
    index, columns, grid_ids, originals = {}, set(), set(), set()
    for row in rows:
        item, column, grid = row['letter_item_id'], int(row['sid_column']), int(row['grid_item_id'])
        require(item in axis and item not in index, 'missing, duplicate or invalid LETTER item')
        require(0 <= column < N and column not in columns, 'invalid or duplicate SID column')
        require(0 <= grid < N and grid not in grid_ids and grid == column, 'invalid GRID item')
        require(row['original_item_id'] and row['original_item_id'] not in originals, 'duplicate original item')
        digits = [int(row[f'sid_{position}']) for position in range(1, 5)]
        require(digits == tensor[:, column].tolist(), 'bridge and G3 SID mismatch')
        tokens = [f'<{prefix}_{digit}>' for prefix, digit in zip('abcd', digits)]
        require([int(token[3:-1]) for token in tokens] == digits, 'token round trip')
        index[item] = tokens
        columns.add(column)
        grid_ids.add(grid)
        originals.add(row['original_item_id'])
    require(set(index) == axis and columns == set(range(N)), 'incomplete coverage')
    require(len({tuple(tokens) for tokens in index.values()}) == N, 'index full SID collision')
    ordered = {str(item): index[str(item)] for item in range(N)}
    encoded = (json.dumps(ordered, ensure_ascii=False, separators=(',', ':')) + '\n').encode()
    receipt = {'inputs': HASHES, 'index_sha256': hashlib.sha256(encoded).hexdigest(),
               'items': N, 'unique_full_sid': N, 'collision_groups': 0,
               'distinct_values_per_position': list(map(len, values)), 'token_count': sum(map(len, values)),
               'scope': 'conversion-only; not a training, tokenizer or model acceptance'}
    args.output.mkdir(parents=True, exist_ok=True)
    write_once(args.output / 'Beauty.g3.index.json', encoded)
    write_once(args.output / 'conversion.json', (json.dumps(receipt, indent=2, sort_keys=True) + '\n').encode())
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
