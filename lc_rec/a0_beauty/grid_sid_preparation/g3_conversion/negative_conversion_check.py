#!/usr/bin/env python3
"""Exercise conversion's structural guards independently of pinned file hashes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import torch
import convert


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ('g3', 'bridge', 'item', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if any(sha(getattr(args, key)) != value for key, value in convert.HASHES.items()):
        raise ValueError('positive input identity')
    with args.bridge.open(newline='') as stream:
        reader = csv.DictReader(stream)
        fields, original = reader.fieldnames, list(reader)
    results = {}
    with tempfile.TemporaryDirectory(prefix='g3-negative-') as temporary:
        root = Path(temporary)
        for case in ('missing_row', 'duplicate_item', 'sid_mismatch', 'same_shape_wrong_g3'):
            rows = [dict(row) for row in original]
            tensor = args.g3
            if case == 'missing_row':
                rows.pop()
            elif case == 'duplicate_item':
                rows[1]['letter_item_id'] = rows[0]['letter_item_id']
            elif case == 'sid_mismatch':
                rows[1]['sid_1'] = str((int(rows[1]['sid_1']) + 1) % 256)
            else:
                value = torch.load(args.g3, map_location='cpu', weights_only=True)
                value[0, 0] = (int(value[0, 0]) + 1) % 256
                tensor = root / 'wrong.pt'
                torch.save(value, tensor)
            bridge = root / (case + '.csv')
            with bridge.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            convert.HASHES = {'g3': sha(tensor), 'bridge': sha(bridge), 'item': sha(args.item)}
            sys.argv = ['convert.py', '--g3', str(tensor), '--bridge', str(bridge), '--item', str(args.item),
                        '--output', str(root / case)]
            try:
                convert.main()
            except ValueError as error:
                message = str(error)
                expected = {'missing_row': 'bridge row count', 'duplicate_item': 'missing, duplicate or invalid LETTER item',
                            'sid_mismatch': 'bridge and G3 SID mismatch', 'same_shape_wrong_g3': 'bridge and G3 SID mismatch'}[case]
                if message != expected:
                    raise ValueError(f'{case}: unexpected rejection {message}') from error
                results[case] = message
            else:
                raise ValueError(case + ' unexpectedly passed')
    result = {'scope': 'synthetic mutations of real inputs; hash pins overridden only in test', 'rejections': results}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
