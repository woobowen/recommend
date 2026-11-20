"""A0 full CPU replay from the frozen tokenizer, author collator and Normal shards."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from lc_rec.letter.scripts import formal_evaluate as frozen
from lc_rec.shuffle_v2 import core, reference

CONFIG = Path(__file__).with_name('a0_instruments.json')


def semantic_hash(value):
    # The historical axis/vocab receipt uses insertion order and unescaped UTF-8.
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()


def dump(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def asset_check(config):
    evidence = []
    for asset in config['files']:
        path = Path(asset['path'])
        stat = path.stat()
        actual = frozen.sha(path)
        after = path.stat()
        before_id = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        after_id = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        assert before_id == after_id and stat.st_size == asset.get('bytes', stat.st_size), path
        assert actual == asset['sha256'], (path, actual)
        evidence.append({'path': str(path), 'sha256': actual, 'stat_identity': before_id})
    return evidence


def frozen_data(config):
    from transformers import LlamaTokenizer
    base = config['baseline']
    sys.path.insert(0, base['source-root'])
    from data import SeqRecDataset
    from collator import TestCollator
    args = SimpleNamespace(dataset='Instruments', data_path=base['data-root'], max_his_len=20,
                           his_sep=', ', index_file='.index.json', add_prefix=False)
    data = SeqRecDataset(args, mode='test', sample_num=-1)
    data.set_prompt(0)
    tokenizer = LlamaTokenizer.from_pretrained(base['adapter'], local_files_only=True)
    tokenizer.pad_token_id, tokenizer.padding_side, tokenizer.model_max_length = 0, 'left', 2048
    assert semantic_hash(tokenizer.get_vocab()) == config['vocab_sha256']
    tokens = sorted({token for sid in data.indices.values() for token in sid})
    lookup = {token: tokenizer.encode(token, add_special_tokens=False) for token in tokens}
    assert len(lookup) == config['sid_tokens'] and all(len(ids) == 1 for ids in lookup.values())
    token_ids = {token: ids[0] for token, ids in lookup.items()}
    assert len(set(token_ids.values())) == config['sid_tokens']
    return data, tokenizer, TestCollator(args, tokenizer), token_ids


def read_normal(config, users, catalog, assignments):
    base = config['baseline']
    evaluator = frozen.load_evaluator(base['source-root'])
    raw_by_index = {}
    for rank, assignment in enumerate(assignments):
        folder = Path(base['normal_test_dir'])
        report = json.loads((folder / f'rank{rank}.json').read_text())
        assert report['status'] == 'PASS' and report['RUN_COMMIT'] == base['run-commit']
        assert report['completed_users'] == len(assignment) and report['rank'] == rank
        assert report['world_size'] == 2 and report['adapter_files']['adapter_model.bin'] == base['adapter_identity']
        assert report['assignment_sha256'] == core.canonical_sha256(assignment)
        with (folder / f'predictions.rank{rank}.jsonl').open() as stream:
            for index, line in zip(assignment, stream, strict=True):
                row = json.loads(line)
                assert row['row_idx'] == index and row['rank'] == rank and row['prompt_id'] == 0
                assert len(row['candidates']) == 20 and all(c['eos_position'] == 4 for c in row['candidates'])
                frozen.audit_prediction(row, users[index], catalog, evaluator)
                assert index not in raw_by_index
                raw_by_index[index] = core.canonical_sha256(row['input_token_ids'])
    assert len(raw_by_index) == config['users']
    return raw_by_index


def sample_rows(assignment, rows):
    chosen = []

    def take(pool, count):
        chosen.extend([idx for idx in pool if idx not in chosen][:count])

    for mode in core.MODES:
        take([i for i in assignment if not rows[i]['conditions'][mode]['input_changed']], 1)
        take([i for i in assignment if 0 in rows[i]['conditions'][mode]['permutation_ids']], 1)
    for low, high in ((4, 5), (6, 10), (11, 20)):
        take([i for i in assignment if low <= rows[i]['visible_history_length'] <= high], 2)
    by_length = sorted(assignment, key=lambda i: (rows[i]['encoded_length'], i))
    take([by_length[0], by_length[-1]], 2)
    take(assignment, 16 - len(chosen))
    assert len(chosen) == 16
    return chosen


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert sys.flags.optimize == 0 and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(CONFIG.read_text())
    assert config['modes'] == {mode: list(active) for mode, active in core.MODES.items()}
    assets = asset_check(config)
    dump(args.output / 'assets.json', assets)
    from torch.utils.data.distributed import DistributedSampler
    indices, catalog, users, _ = frozen.catalog_and_users(config['baseline']['data-root'])
    assert len(indices) == config['items'] and len(catalog) == config['sids']
    data, tokenizer, collate, token_ids = frozen_data(config)
    sid_ids = set(token_ids.values())
    user_ids = list(data.inters)
    assert user_ids == [user['user_id'] for user in users]
    assert semantic_hash(user_ids) == config['user_axis_sha256']
    assignments = [list(DistributedSampler(range(len(users)), num_replicas=2, rank=rank,
                                           shuffle=True, seed=0, drop_last=True)) for rank in range(2)]
    normal = read_normal(config, users, catalog, assignments)
    hist = Counter()
    input_streams = {mode: hashlib.sha256() for mode in core.MODES}
    stream_path = args.output / 'inputs.jsonl'
    rows, examples = [], {}
    frequency = {mode: Counter() for mode in core.MODES}
    positions = {mode: [0] * 4 for mode in core.MODES}
    permuted_positions = {mode: [0] * 4 for mode in core.MODES}
    unchanged_groups = Counter()
    unchanged_users = Counter()
    with stream_path.open('x') as stream:
        for index, uid in enumerate(user_ids):
            sequence = data.inters[uid]
            visible = sequence[:-1][-20:]
            groups = [[token_ids[token] for token in data.indices[str(item)]] for item in visible]
            inputs, labels = collate([data[index]])
            raw, mask = inputs['input_ids'][0].tolist(), inputs['attention_mask'][0].tolist()
            assert labels == [''.join(indices[str(sequence[-1])])] == [users[index]['target']]
            assert core.canonical_sha256(raw) == normal[index]
            assert all(mask) and len(raw) <= 2048
            hist[len(groups)] += 1
            row = {'row_idx': index, 'user_id': uid, 'target': labels[0],
                   'visible_history_length': len(groups), 'encoded_length': len(raw),
                   'original_sha256': normal[index], 'attention_mask_sha256': core.canonical_sha256(mask),
                   'conditions': {}}
            offsets = [pos for pos, token in enumerate(raw) if token in sid_ids]
            assert len(offsets) == 4 * len(groups)
            group_set = set(offsets)
            shared = [reference.relative(3, index, j) for j in range(len(groups))]
            for mode, active in core.MODES.items():
                new, evidence = core.transform(raw, mask, groups, sid_ids, mode, index)
                input_streams[mode].update(core.canonical_bytes([index, new]) + b'\n')
                assert len(new) == len(raw) and evidence['attention_mask_sha256'] == row['attention_mask_sha256']
                assert all(a == b for pos, (a, b) in enumerate(zip(raw, new, strict=True)) if pos not in group_set)
                entries = evidence['groups']
                permutation_ids = []
                for j, (group, record) in enumerate(zip(groups, entries, strict=True)):
                    wanted = shared[j] if len(active) == 3 else reference.relative(len(active), index, j)
                    pick, permutation, counter = wanted
                    start = offsets[j * 4]
                    assert offsets[j * 4:j * 4 + 4] == list(range(start, start + 4))
                    assert record['positions'] == list(range(start, start + 4))
                    assert raw[start:start + 4] == group
                    changed = new[start:start + 4]
                    assert changed == reference.remap(group, active, permutation) and sorted(changed) == sorted(group)
                    assert record['relative_permutation'] == permutation and record['rejection_counter'] == counter
                    assert record['permutation_id'] == pick and record['nonidentity'] == (pick != 0)
                    permutation_ids.append(pick)
                    frequency[mode][pick] += 1
                    unchanged_groups[mode] += group == changed
                    for pos in range(4):
                        positions[mode][pos] += group[pos] != changed[pos]
                        permuted_positions[mode][pos] += record['permutation'][pos] != pos
                        assert pos in active or changed[pos] == group[pos]
                assert evidence['input_changed'] == (raw != new)
                unchanged_users[mode] += raw == new
                row['conditions'][mode] = {'input_sha256': evidence['shuffled_sha256'],
                                           'input_changed': raw != new, 'permutation_ids': permutation_ids}
                if index in (0, 1233, 24771):
                    examples.setdefault(str(index), {})[mode] = {'ids': new, 'mask': mask, **evidence}
            assert row['conditions']['shuffle3a']['permutation_ids'] == row['conditions']['shuffle3b']['permutation_ids']
            rows.append(row)
            stream.write(json.dumps(row, separators=(',', ':'), ensure_ascii=False) + '\n')
    total = sum(n * count for n, count in hist.items())
    assert total == config['groups'] and min(hist) == 4 and max(hist) == 20
    first = collate([data[0], data[max(range(len(rows)), key=lambda i: rows[i]['encoded_length'])]])[0]
    for seq, mask in zip(first['input_ids'].tolist(), first['attention_mask'].tolist(), strict=True):
        encoded = [token for token, active in zip(seq, mask, strict=True) if active]
        assert mask == [0] * (len(seq) - len(encoded)) + [1] * len(encoded)
    # Verify the original tokenizer/collator's truncation, and fail closed on lost SID groups.
    text = ('word ' * 2300) + data[0]['input_ids']
    truncated, _ = collate([{'input_ids': text, 'labels': users[0]['target']}])
    assert len(truncated['input_ids'][0]) == tokenizer.model_max_length
    try:
        core.transform(truncated['input_ids'][0].tolist(), truncated['attention_mask'][0].tolist(),
                       [[token_ids[token] for token in indices[str(item)]] for item in data.inters[user_ids[0]][:-1][-20:]],
                       sid_ids, 'shuffle2', 0)
    except ValueError:
        pass
    else:
        raise AssertionError('Truncation lost valid SID groups without a failure')
    samples = {rank: sample_rows(assignment, rows) for rank, assignment in enumerate(assignments)}
    for rank, selected in samples.items():
        for mode in core.MODES:
            assert any(0 in rows[i]['conditions'][mode]['permutation_ids'] for i in selected), (rank, mode)
    dump(args.output / 'samples.json', {rank: [rows[i] for i in selected] for rank, selected in samples.items()})
    dump(args.output / 'examples.json', examples)
    summary = {'status': 'PASS_CPU_V2_FULL', 'protocol': core.PROTOCOL, 'identity_policy': core.IDENTITY_POLICY,
               'shuffle_seed': core.SEED, 'serialization': core.SERIALIZATION,
               'users': len(rows), 'items': len(indices), 'unique_sids': len(catalog), 'history_groups': total,
               'history_length_counts': hist, 'token_ids': token_ids,
               'tokenizer_vocab_sha256': config['vocab_sha256'], 'user_axis_sha256': config['user_axis_sha256'],
               'preflight_selection': 'Per rank in frozen sampler order: first natural unchanged user and first identity group per mode where present, up to two per history-length bin, encoded length extrema then sampler-order fill',
               'preflight_assignments': samples, 'full_rank_assignment_sha256': [core.canonical_sha256(a) for a in assignments],
               'normal_inputs_exact': True, 'manifest_sha256': frozen.sha(stream_path),
               'versions': {name: importlib.metadata.version(name) for name in ('torch','transformers','peft','accelerate','numpy','sentencepiece')},
               'modes': {mode: {**core.mode_metadata(mode), 'permutation_counts': [frequency[mode][i] for i in range(len(core.catalog(len(active))))],
                                'identity_permutations': frequency[mode][0], 'token_unchanged_groups': unchanged_groups[mode],
                                'unchanged_users': unchanged_users[mode], 'moved_ids_by_position': positions[mode],
                                'permuted_slots_by_position': permuted_positions[mode], 'fixed_position_violations': 0,
                                'outside_group_violations': 0, 'mask_or_length_violations': 0}
                         for mode, active in core.MODES.items()}}
    summary['input_stream_sha256'] = {mode: value.hexdigest() for mode, value in input_streams.items()}
    dump(args.output / 'audit.json', summary)
    print(json.dumps({key: summary[key] for key in ('status','users','history_groups','modes')}))


if __name__ == '__main__':
    main()
