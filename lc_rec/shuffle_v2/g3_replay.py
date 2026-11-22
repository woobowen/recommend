"""Rebuild every G3 Prompt0 and independently verify every V2 occurrence."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

from lc_rec.shuffle_v2 import core, g3_assets, reference
from lc_rec.shuffle_v2.replay import dump


def normal_inputs(config, normal_rank, users, path):
    axis = normal_rank['assigned_row_indices']
    assert len(axis) == len(users) and sorted(axis) == list(range(len(users)))
    assert core.canonical_sha256(axis) == config['full_assignment_sha256']
    assert normal_rank['status'] == 'PASS' and normal_rank['RUN_COMMIT'] == config['training_commit']
    assert normal_rank['rank'] == 0 and normal_rank['world_size'] == 1 and normal_rank['scope'] == 'FORMAL'
    g3_assets.verified(path, config['normal_predictions_sha256'])
    result = {}
    with path.open() as stream:
        for index, line in zip(axis, stream, strict=True):
            row = json.loads(line)
            assert all(row[key] == value for key, value in users[index].items())
            assert row['condition'] == 'normal' and row['rank'] == 0
            assert row['input_token_ids'] == row['original_input_token_ids']
            result[index] = (row['input_token_ids'], row['attention_mask'])
    return axis, result


def replay(output):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    config, source, normal_rank, adapter, paths = g3_assets.load()
    from collator import TestCollator
    from data import SeqRecDataset
    from preflight_users import validate
    from transformers import LlamaTokenizer
    from uneven_sampler import assignment
    indices, catalog, users, identities = source.catalog_and_users(config['data_root'])
    assert len(indices) == len(catalog) == 12101 and len(users) == 22363
    assert {key: row['sha256'] for key, row in identities.items()} == config['assets']
    args = SimpleNamespace(dataset='Beauty', data_path=config['data_root'], max_his_len=20, his_sep=', ',
                           index_file='.index.json', add_prefix=False)
    data = SeqRecDataset(args, mode='test', sample_num=-1)
    data.set_prompt(0)
    assert list(data.remapped_inters) == [row['user_id'] for row in users]
    normal_path = Path(config['formal_root']) / 'attempts/normal/a0/test/predictions.rank0.jsonl'
    axis, normal = normal_inputs(config, normal_rank, users, normal_path)
    assert assignment(len(users), 0, 1) == axis
    frozen = g3_assets.read(paths[g3_assets.MANIFEST])
    assert frozen['asset_sha256'] == config['assets']
    selected, = validate(frozen, users, [axis])
    assert len(selected) == len(set(selected)) == 32
    tokenizer = LlamaTokenizer.from_pretrained(adapter, local_files_only=True)
    tokenizer.pad_token_id, tokenizer.padding_side, tokenizer.model_max_length = 0, 'left', 2048
    token_identity = source.tokenizer_identity(tokenizer, indices)
    assert token_identity == normal_rank['tokenizer_identity']
    token_ids = {token: tokenizer.encode(token, add_special_tokens=False)[0]
                 for token in sorted({token for sid in indices.values() for token in sid})}
    sid_vocabulary = set(token_ids.values())
    assert sid_vocabulary == set(range(32000, 32786))
    assert hashlib.sha256(inspect.getsource(data.get_prefix_allowed_tokens_fn).encode()).hexdigest() == config['callback_sha256']
    collate = TestCollator(args, tokenizer)
    output.mkdir(parents=True, exist_ok=False)
    manifest = output / 'inputs.jsonl'
    history_lengths, frequencies, unchanged, unchanged_users = Counter(), {}, Counter(), Counter()
    moved = {mode: [0, 0, 0, 0] for mode in core.MODES}
    streams = {mode: hashlib.sha256() for mode in core.MODES}
    identity_rows = {mode: [] for mode in core.MODES}
    for mode in core.MODES:
        frequencies[mode] = Counter()
    with manifest.open('x') as stream:
        for index, user in enumerate(users):
            sequence = data.inters[user['user_id']]
            visible = [str(item) for item in sequence[:-1][-20:]]
            groups = [[token_ids[token] for token in indices[item]] for item in visible]
            inputs, labels = collate([data[index]])
            raw, mask = inputs['input_ids'][0].tolist(), inputs['attention_mask'][0].tolist()
            assert labels == [user['target']] and str(sequence[-1]) == user['target_item_id']
            assert (raw, mask) == normal[index] and all(mask)
            offsets = [position for position, token in enumerate(raw) if token in sid_vocabulary]
            assert len(offsets) == 4 * len(groups)
            offsets_set = set(offsets)
            boundaries = [offsets[i:i + 4] for i in range(0, len(offsets), 4)]
            for group, span in zip(groups, boundaries, strict=True):
                assert span == list(range(span[0], span[0] + 4))
                assert [raw[position] for position in span] == group
            history_lengths[len(groups)] += 1
            row = {**user, 'original_sha256': core.canonical_sha256(raw),
                   'attention_mask_sha256': core.canonical_sha256(mask), 'encoded_length': len(raw),
                   'groups': list(zip(visible, boundaries, groups, strict=True)), 'conditions': {}}
            shared = [reference.relative(3, index, j) for j in range(len(groups))]
            for mode, active in core.MODES.items():
                actual, proof = core.transform(raw, mask, groups, sid_vocabulary, mode, index)
                assert len(actual) == len(raw) and proof['attention_mask_sha256'] == row['attention_mask_sha256']
                assert all(actual[i] == token for i, token in enumerate(raw) if i not in offsets_set)
                picks = []
                for j, (item, span, record) in enumerate(zip(groups, boundaries, proof['groups'], strict=True)):
                    picked, relative, rejected = shared[j] if len(active) == 3 else reference.relative(len(active), index, j)
                    expected = reference.remap(item, active, relative)
                    assert record['positions'] == span and record['original_ids'] == item
                    assert (record['permutation_id'], record['relative_permutation'], record['rejection_counter']) == (picked, relative, rejected)
                    assert tuple(record['permutation'][p] for p in range(4) if p not in active) == tuple(p for p in range(4) if p not in active)
                    assert actual[span[0]:span[-1] + 1] == expected and sorted(item) == sorted(expected)
                    picks.append(picked)
                    frequencies[mode][picked] += 1
                    unchanged[mode] += item == expected
                    for slot in range(4):
                        moved[mode][slot] += item[slot] != expected[slot]
                assert proof['input_changed'] == (raw != actual)
                if raw == actual:
                    unchanged_users[mode] += 1
                    if index in selected:
                        identity_rows[mode].append(index)
                streams[mode].update(core.canonical_bytes([index, actual]) + b'\n')
                row['conditions'][mode] = {'input_sha256': proof['shuffled_sha256'], 'input_changed': raw != actual,
                                           'permutation_ids': picks}
            assert row['conditions']['shuffle3a']['permutation_ids'] == row['conditions']['shuffle3b']['permutation_ids']
            stream.write(json.dumps(row, separators=(',', ':'), ensure_ascii=False) + '\n')
    group_count = sum(count * length for length, count in history_lengths.items())
    assert group_count == 159966
    dump(output / 'audit.json', {
        'status': 'PASS_G3_CPU_V2_FULL', 'protocol': core.PROTOCOL, 'identity_policy': core.IDENTITY_POLICY,
        'users': len(users), 'items': len(indices), 'sids': len(catalog), 'collisions': 0, 'groups': group_count,
        'history_lengths': history_lengths, 'manifest_sha256': g3_assets.sha(manifest),
        'training_commit': config['training_commit'], 'assets': identities, 'tokenizer_identity': token_identity,
        'token_ids': token_ids, 'user_axis_sha256': core.canonical_sha256(users),
        'normal_sha256': config['normal_predictions_sha256'], 'normal_inputs_exact': True,
        'preflight_assignment': selected, 'preflight_manifest_sha256': g3_assets.sha(paths[g3_assets.MANIFEST]),
        'assignment_sha256': core.canonical_sha256(axis),
        'versions': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'peft', 'accelerate', 'numpy', 'sentencepiece')},
        'modes': {mode: {**core.mode_metadata(mode), 'permutation_counts': [frequencies[mode][i]
                            for i in range(len(core.catalog(len(active))))],
                        'identity_groups': unchanged[mode], 'identity_users': unchanged_users[mode],
                        'preflight_identity_rows': identity_rows[mode], 'moved_ids_by_slot': moved[mode],
                        'inputs_sha256': streams[mode].hexdigest()}
                  for mode, active in core.MODES.items()},
    })


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--output', required=True, type=Path)
    options = parser.parse_args()
    replay(options.output)


if __name__ == '__main__':
    main()
