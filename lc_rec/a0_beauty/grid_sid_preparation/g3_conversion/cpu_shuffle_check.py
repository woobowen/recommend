#!/usr/bin/env python3
"""Exercise the existing shuffle on every real G3 test input, without weights."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ('source', 'shuffle', 'helpers', 'data', 'index', 'model', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    expected = {'index': '420fc060caba3a22b0f03900dfe26abc67da8753f3bac2803f7f3c8703a05c22',
                'inter': '8809584b8b78fa6561771c88911ed65bd716fc9926fca1e581e6a1ab578ccfd5'}
    for name, path in [('index', args.index), ('inter', args.data / 'Beauty.inter.json')]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected[name]:
            raise ValueError(name + ' identity')
    sys.path[:0] = [str(args.source), str(args.shuffle), str(args.helpers)]
    from data import SeqRecDataset
    from core import transform
    from uneven_sampler import assignment, validate_assignments, assignment_sha256
    from transformers import LlamaTokenizer

    with tempfile.TemporaryDirectory(prefix='g3-shuffle-') as temporary:
        root = Path(temporary)
        beauty = root / 'Beauty'
        beauty.mkdir()
        (beauty / 'Beauty.inter.json').symlink_to((args.data / 'Beauty.inter.json').resolve())
        (beauty / 'Beauty.index.json').symlink_to(args.index.resolve())
        options = SimpleNamespace(dataset='Beauty', data_path=str(root), max_his_len=20,
                                  his_sep=', ', index_file='.index.json', add_prefix=False)
        dataset = SeqRecDataset(options, mode='test', sample_num=-1)
        dataset.set_prompt(0)
        tokenizer = LlamaTokenizer.from_pretrained(args.model, local_files_only=True, model_max_length=2048)
        tokenizer.pad_token_id = 0
        tokens = dataset.get_new_tokens()
        if tokenizer.add_tokens(tokens) != 786 or len(tokenizer) != 32786:
            raise ValueError('vocabulary identity')
        lookup = {token: tokenizer.encode(token, add_special_tokens=False)[0] for token in tokens}
        if len(set(lookup.values())) != 786:
            raise ValueError('token ID collision')
        stream_hashes = {size: hashlib.sha256() for size in (0, 3, 4)}
        lengths, history_lengths = [], []
        axis = list(dataset.inters)
        if len(axis) != 22363 or digest(axis) != '7430079e2b917dfa5e4bb8ead2702fb6c622fb48ffb684dd2889780f9480af7f':
            raise ValueError('test user axis')
        for row_number, user in enumerate(axis):
            sequence = dataset.inters[user]
            visible = sequence[:-1][-20:]
            groups = [[lookup[token] for token in dataset.indices[str(item)]] for item in visible]
            row = dataset[row_number]
            if row['labels'] != ''.join(dataset.indices[str(sequence[-1])]):
                raise ValueError('target changed')
            encoded = tokenizer(row['input_ids'], return_attention_mask=True)
            ids, mask = encoded['input_ids'], encoded['attention_mask']
            if len(ids) > 2048 or not all(mask):
                raise ValueError('input length or mask')
            lengths.append(len(ids))
            history_lengths.append(len(visible))
            for size in (0, 3, 4):
                changed, evidence = transform(ids, mask, groups, set(lookup.values()), size, row_number)
                if size == 0 and changed != ids:
                    raise ValueError('OFF changed input')
                if size and visible and changed == ids:
                    raise ValueError('shuffle did not change input')
                stream_hashes[size].update((json.dumps([row_number, changed], separators=(',', ':')) + '\n').encode())
            padded, _ = transform([0, 0, 0] + ids, [0, 0, 0] + mask, groups, set(lookup.values()), 4, row_number)
            if padded[:3] != [0, 0, 0] or padded[3:] != changed:
                raise ValueError('padding changed')
        shards = [assignment(len(dataset), rank, 2) for rank in (0, 1)]
        validate_assignments(shards, len(dataset))
        if list(map(len, shards)) != [11182, 11181]:
            raise ValueError('rank assignment')
        result = {'scope': 'real full-input CPU shuffle only', 'users': len(axis),
                  'visible_history_groups': sum(history_lengths), 'max_input_length': max(lengths),
                  'rank_sizes': list(map(len, shards)), 'rank_assignment_sha256': list(map(assignment_sha256, shards)),
                  'input_stream_sha256': {str(key): value.hexdigest() for key, value in stream_hashes.items()},
                  'assets': expected}
        content = (json.dumps(result, indent=2, sort_keys=True) + '\n').encode()
        with args.output.open('xb') as stream:
            stream.write(content)
        print(content.decode(), end='')


if __name__ == '__main__':
    main()
