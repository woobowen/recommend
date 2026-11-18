#!/usr/bin/env python3
"""Check real LETTER datasets and base tokenizer using the converted G3 index."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

EXPECTED = {'index': '420fc060caba3a22b0f03900dfe26abc67da8753f3bac2803f7f3c8703a05c22',
            'item': '74aeddf911548aadfcd1dab326c16678ed4ac74457e874da300d63a94bdb2330',
            'inter': '8809584b8b78fa6561771c88911ed65bd716fc9926fca1e581e6a1ab578ccfd5'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ('source', 'data', 'index', 'model', 'config', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    for kind in ('item', 'inter'):
        if sha(args.data / f'Beauty.{kind}.json') != EXPECTED[kind]:
            raise ValueError(f'{kind} identity')
    if sha(args.index) != EXPECTED['index']:
        raise ValueError('index identity')
    sys.path.insert(0, str(args.source))
    from utils import load_datasets, set_seed
    from data import SeqRecDataset
    from collator import Collator
    from transformers import LlamaTokenizer

    set_seed(42)
    cfg = json.loads(args.config.read_text())
    with tempfile.TemporaryDirectory(prefix='g3-dataset-') as temporary:
        root = Path(temporary)
        beauty = root / 'Beauty'
        beauty.mkdir()
        for kind in ('item', 'inter'):
            (beauty / f'Beauty.{kind}.json').symlink_to((args.data / f'Beauty.{kind}.json').resolve())
        (beauty / 'Beauty.index.json').symlink_to(args.index.resolve())
        cfg['data_path'] = str(root)
        options = SimpleNamespace(**cfg)
        train, valid = load_datasets(options)
        test = SeqRecDataset(options, mode='test', sample_num=-1)
        test.set_prompt(0)
        if (len(train), len(valid), len(test)) != (131413, 44726, 22363):
            raise ValueError('dataset row counts')
        tokens = train.datasets[0].get_new_tokens()
        tokenizer = LlamaTokenizer.from_pretrained(args.model, model_max_length=2048, padding_side='right', local_files_only=True)
        tokenizer.pad_token_id = 0
        added = tokenizer.add_tokens(tokens)
        lookup = {token: tokenizer.encode(token, add_special_tokens=False) for token in tokens}
        if added != 786 or len(tokenizer) != 32786 or any(len(ids) != 1 for ids in lookup.values()):
            raise ValueError('token registration')
        ids = [value[0] for value in lookup.values()]
        if len(set(ids)) != 786 or set(ids) & set(tokenizer.all_special_ids):
            raise ValueError('token IDs')
        tokenizer_dir = root / 'tokenizer'
        tokenizer.save_pretrained(tokenizer_dir)
        reloaded = LlamaTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
        if tokenizer.get_vocab() != reloaded.get_vocab() or any(reloaded.encode(token, add_special_tokens=False) != value for token, value in lookup.items()):
            raise ValueError('tokenizer reload')
        batch = Collator(options, tokenizer)([train[0], train[1]])
        if not ((batch['labels'] != -100).any() and (batch['labels'] == -100).any()):
            raise ValueError('collator labels')
        allowed = test.get_prefix_allowed_tokens_fn(tokenizer)
        del allowed
        allowed_sizes = [len(test.allowed_tokens[position]) for position in range(5)]
        if allowed_sizes != [256, 256, 256, 18, 1]:
            raise ValueError('generation allowed sets')
        result = {'scope': 'real CPU dataset/tokenizer only', 'dataset_rows': [len(train), len(valid), len(test)],
                  'added_tokens': added, 'vocab_size': len(tokenizer), 'allowed_sizes': allowed_sizes,
                  'token_to_id_sha256': hashlib.sha256(json.dumps({key: value[0] for key, value in lookup.items()}, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                  'collator_shapes': {key: list(value.shape) for key, value in batch.items()},
                  'assets': EXPECTED, 'model': str(args.model), 'source': str(args.source)}
        content = (json.dumps(result, indent=2, sort_keys=True) + '\n').encode()
        with args.output.open('xb') as stream:
            stream.write(content)
        print(content.decode(), end='')


if __name__ == '__main__':
    main()
