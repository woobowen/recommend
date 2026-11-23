"""Verify the full A1 Instruments CPU input stream against frozen Normal."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from lc_rec.a1_qwen3_06b.scripts import evaluate as frozen
from lc_rec.qwen_preflight_20261110.scripts import common
from lc_rec.shuffle_v2 import core, reference


ROOT = Path(__file__).resolve().parents[2]
NORMAL_COMMIT = '0e3e2c6f5c1a76152ea0228390d94ffc1ff75ba5'
SAMPLE_COMMIT = 'f0242b9f90ae23a4979d688a287734ca9f88cea1'
SAMPLE_PATH = 'lc_rec/shuffle/evidence/cpu-bound-r0/a1/audit.json'
SAMPLE_SHA = '1a69eeaef99f46406c968367f0c0a3dc3e433675002185401a92ab7013964269'
SAMPLE_RANK_SHA = (
    'a39dc248f9cb5202f33caa0e2538aac295eb4c636cf8d0ff148098331a299834',
    'a2b0eac377797ecfdea1c2a7a0c04e907fcbde6ec088e23a0a2f820876bf6d22',
)
NORMAL_PREDICTIONS = (
    (184629790, '790506cf834bca27f4bb01d35d7d4048d494872f74ff4d9a8bcdac516c475b4d'),
    (184130524, '028a74dbab9ba2afaa018994b1518c465d20d58ce05218720634e5c42a4a2a45'),
)
HISTORICAL_VOCAB_SHA = '4d0e7267626dde1ab31259f328a81a4be2a923a824e2ae62c2bd74d4d01a4ccc'
VOCAB_SHA = '7c0e678397740771aa98ef5268786f15020e82537a8bffadf47c4829bb8533e9'
AXIS_SHA = '1daf63ce4c77b0265777a46b61939006aafab0276b9106c742de602c8c7e867a'
NORMAL_MANIFEST_SHA = '038107db291d363d5a5ba3324c694454f444d90b1059a6368d27f4508fd86a06'
VERSIONS = {'torch': '2.6.0', 'transformers': '4.57.6', 'peft': '0.21.2', 'accelerate': '1.12.0'}


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, data):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def semantic_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()


def vocab_hash(vocabulary):
    # Fast tokenizer vocabulary iteration is backed by an unordered Rust map.
    return hashlib.sha256(json.dumps(vocabulary, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def cpu_only():
    if sys.flags.optimize or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError("Use non-optimized Python and CUDA_VISIBLE_DEVICES='' for CPU jobs")


def repo_path(path):
    return ROOT / Path(path)


def asset_check(config, offline):
    def stat_identity(st):
        return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]

    evidence = []
    for asset in config['files']:
        path = repo_path(asset['path'])
        if offline and path.name in {'model.safetensors', 'adapter_model.bin'} and not path.is_file():
            evidence.append({'path': str(path), 'sha256': asset['sha256'], 'status': 'external model weights'})
            continue
        before = path.stat()
        digest = frozen.sha(path)
        after = path.stat()
        assert stat_identity(before) == stat_identity(after), path
        assert digest == asset['sha256'] and before.st_size == asset.get('bytes', before.st_size), path
        evidence.append({'path': str(path), 'bytes': before.st_size, 'sha256': digest, 'status': 'verified',
                         'stat_identity': stat_identity(before)})
    return evidence


def frozen_samples(config):
    audit_path = repo_path(config.get('frozen_samples', SAMPLE_PATH))
    assert frozen.sha(audit_path) == SAMPLE_SHA
    audit = load(audit_path)
    assert audit['status'] == 'PASS_CPU_FULL_INPUTS' and audit['model_key'] == 'a1'
    assert audit['user_axis_sha256'] == AXIS_SHA
    assert audit['tokenizer_vocab_sha256'] == HISTORICAL_VOCAB_SHA
    source = [{'role': 'historical CPU selection', 'sha256': SAMPLE_SHA, 'path': str(audit_path)}]
    for rank, digest in enumerate(SAMPLE_RANK_SHA):
        path = repo_path(f'lc_rec/shuffle/evidence/preflight-r1/a1/rank{rank}.json')
        assert frozen.sha(path) == digest
        report = load(path)
        assert report['rank'] == rank and report['completed_users'] == 16
        assert report['preflight_assignment'] == audit['preflight_assignments'][str(rank)]
        source.append({'role': f'historical A1 GPU rank {rank}', 'path': str(path), 'sha256': digest})
    selected = audit['preflight_assignments']['0'] + audit['preflight_assignments']['1']
    assert len(selected) == len(set(selected)) == 32
    return selected, source


def frozen_data(config, indices):
    from transformers import AutoTokenizer

    base = config['baseline']
    sys.path.insert(0, str(repo_path(base['source-root'])))
    from data import SeqRecDataset

    args = SimpleNamespace(dataset='Instruments', data_path=str(repo_path(base['data-root'])), max_his_len=20,
                           his_sep=', ', index_file='.index.json', add_prefix=False)
    data = SeqRecDataset(args, mode='test', sample_num=-1)
    data.set_prompt(0)
    tokens = sorted({token for sid in indices.values() for token in sid})
    reference_tokenizer, metadata = common.tokenizer_for(repo_path(base['model']), tokens)
    tokenizer = AutoTokenizer.from_pretrained(repo_path(base['adapter']), local_files_only=True, padding_side='left')
    vocabulary = tokenizer.get_vocab()
    assert vocabulary == reference_tokenizer.get_vocab()
    assert tokenizer.get_added_vocab() == reference_tokenizer.get_added_vocab()
    assert len(tokenizer) == 152587 and vocab_hash(vocabulary) == config['vocab_sha256'] == VOCAB_SHA
    assert (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.padding_side) == (
        None, 151643, 151643, 'left')
    assert data.indices == indices and not metadata['encodes_automatic_bos']
    token_ids = dict(zip(metadata['sid_tokens'], metadata['sid_ids'], strict=True))
    assert len(token_ids) == len(set(token_ids.values())) == 918

    def collate(rows):
        # Same A1 evaluate.py prompt0 collate, including the explicit mask.
        inputs = tokenizer([row['input_ids'] for row in rows], padding=True, truncation=False,
                           return_tensors='pt', return_attention_mask=True)
        assert inputs['input_ids'].shape[1] <= 2048
        return inputs, [row['labels'] for row in rows]

    return data, tokenizer, collate, token_ids, metadata


def assignments(users):
    from torch.utils.data.distributed import DistributedSampler
    array = range(users)
    ranks = [list(DistributedSampler(array, num_replicas=2, rank=rank, shuffle=True, seed=0,
                                    drop_last=True)) for rank in (0, 1)]
    single = list(DistributedSampler(array, num_replicas=1, rank=0, shuffle=True, seed=0, drop_last=True))
    assert len(ranks[0]) == len(ranks[1]) == 12386
    interleave = [idx for pair in zip(*ranks, strict=True) for idx in pair]
    assert sorted(single) == list(array) and interleave == single
    return single, ranks


def frozen_normal(config):
    base = config['baseline']
    normal_dir = repo_path(base['normal_test_dir'])
    manifest_path = repo_path(config.get('normal_manifest', normal_dir.parent / 'offline_evaluation/prediction_manifest.json'))
    assert frozen.sha(manifest_path) == NORMAL_MANIFEST_SHA
    manifest = load(manifest_path)
    assert (manifest['schema'], manifest['status'], manifest['gpu_run_commit']) == (
        'a1_frozen_prediction_manifest_v1', 'FROZEN_INPUTS', NORMAL_COMMIT)
    assert base['run-commit'] == NORMAL_COMMIT
    selection = load(repo_path(base['best_selection']))
    assert selection['run_commit'] == NORMAL_COMMIT
    assert selection['loaded_best_checkpoint_step'] == selection['completed_training_step'] == base['adapter_step'] == 8240
    assert Path(selection['checkpoint']).name == 'checkpoint-8240'
    assert selection['adapter_keys'] == 394 and selection['adapter_sha256'] == manifest['adapter_sha256']
    reports = [load(normal_dir / f'rank{rank}.json') for rank in (0, 1)]
    protocol = manifest['model_and_protocol']
    for rank, (report, doc, identity) in enumerate(zip(reports, manifest['predictions'], NORMAL_PREDICTIONS, strict=True)):
        assert report['RUN_COMMIT'] == NORMAL_COMMIT and report['rank'] == doc['rank'] == rank
        assert (report['status'], report['world_size'], report['completed_users'], report['assigned_users']) == (
            'PASS', 2, 12386, 12386)
        assert report['versions'] == VERSIONS
        assert (doc['bytes'], doc['sha256']) == identity
        assert all(report[key] == protocol[key] for key in (
            'model', 'data_identity', 'author_sources', 'generation', 'allowed_callback_sha256', 'adapter_files'))
        assert report['adapter_files']['adapter_model.bin']['sha256'] == manifest['adapter_sha256']
        assert report['adapter_files']['adapter_model.bin']['bytes'] == selection['adapter_bytes']
        actual = normal_dir / f'predictions.rank{rank}.jsonl'
        assert actual.stat().st_size == doc['bytes'], actual
    return normal_dir, manifest, reports


def normal_rows(path, assignment, users, rank, identity):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for idx, raw in zip(assignment, stream, strict=True):
            digest.update(raw)
            row = json.loads(raw)
            assert all(row[k] == users[idx][k] for k in ('row_idx', 'user_id', 'target_item_id', 'target'))
            assert row['rank'] == rank and row['prompt_id'] == 0 and len(row['candidates']) == 20
            yield idx, row, raw
    assert digest.hexdigest() == identity['sha256'], path


def verify_group(raw, transformed, positions, original, record, relative, active):
    pick, permutation, counter = relative
    assert record['positions'] == positions
    assert positions == list(range(positions[0], positions[0] + 4))
    start = positions[0]
    moved = transformed[start:start + 4]
    expected = reference.remap(original, active, permutation)
    assert raw[start:start + 4] == original and moved == expected and sorted(moved) == sorted(original)
    assert all(original[slot] == moved[slot] for slot in range(4) if slot not in active)
    assert record['relative_permutation'] == permutation and record['permutation_id'] == pick
    assert record['rejection_counter'] == counter and record['nonidentity'] == (pick != 0)
    return moved


def replay(config, output, offline):
    started = time.monotonic()
    cpu_only()
    assert (config['protocol'], config['identity_policy'], config['shuffle_seed']) == (
        core.PROTOCOL, core.IDENTITY_POLICY, core.SEED)
    assert config['modes'] == {mode: list(active) for mode, active in core.MODES.items()}
    assert (config['users'], config['items'], config['sids'], config['groups'], config['sid_tokens']) == (
        24772, 9922, 9897, 172929, 918)
    versions = {k: importlib.metadata.version(k) for k in VERSIONS}
    if not offline:
        assert versions == VERSIONS, versions
    output.mkdir(parents=True, exist_ok=False)
    assets = asset_check(config, offline)
    dump(output / 'assets.json', assets)
    base = config['baseline']
    indices, catalog, users, identities = frozen.catalog_and_users(repo_path(base['data-root']))
    assert (len(indices), len(catalog), len(users)) == (9922, 9897, 24772)
    data, tokenizer, collate, token_ids, metadata = frozen_data(config, indices)
    sid_ids = set(token_ids.values())
    user_ids = [row['user_id'] for row in users]
    assert list(data.inters) == list(data.remapped_inters) == user_ids
    assert len(data) == len(users) and semantic_hash(user_ids) == AXIS_SHA
    single, ranks = assignments(len(users))
    selected, sample_sources = frozen_samples(config)
    selected_set = set(selected)
    assert set(selected[:16]) <= set(ranks[0]) and set(selected[16:]) <= set(ranks[1])
    normal_dir, normal_manifest, reports = frozen_normal(config)
    assert metadata == {k: reports[0]['tokenizer'][k] for k in metadata}
    read_source = inspect.getsource(common.Allowed)
    assert all(hashlib.sha256(read_source.encode()).hexdigest() == report['allowed_callback_sha256'] for report in reports)
    normal_inputs, sample_bytes = {}, {}
    for rank, (assignment, report) in enumerate(zip(ranks, reports, strict=True)):
        assert core.canonical_sha256(assignment) == report['assignment_sha256']
        path = normal_dir / f'predictions.rank{rank}.jsonl'
        for idx, row, raw in normal_rows(path, assignment, users, rank, normal_manifest['predictions'][rank]):
            assert idx not in normal_inputs and row['input_nonpadding_tokens'] == len(row['input_token_ids'])
            normal_inputs[idx] = row['input_token_ids']
            if idx in selected_set:
                sample_bytes[idx] = raw
    assert len(normal_inputs) == len(users) and sample_bytes.keys() == selected_set
    with (output / 'normal-sample.jsonl').open('xb') as stream:
        stream.writelines(sample_bytes[idx] for idx in selected)

    histories, lengths = Counter(), Counter()
    mode_stats = {}
    for mode, active in core.MODES.items():
        catalog_rows = core.catalog(len(active))
        assert len(catalog_rows) == {2: 2, 3: 6, 4: 24}[len(active)]
        assert catalog_rows[0] == tuple(range(len(active))) and tuple(sorted(catalog_rows)) == catalog_rows
        mode_stats[mode] = {**core.mode_metadata(mode), 'permutation_catalog': catalog_rows,
                            'permutation_counts': [0] * len(catalog_rows), 'unchanged_users': 0,
                            'token_unchanged_groups': 0, 'moved_ids_by_position': [0] * 4,
                            'permuted_slots_by_position': [0] * 4}
    manifest_file = output / 'inputs.jsonl'
    hashes = {mode: hashlib.sha256() for mode in core.MODES}
    samples, max_index, max_length = {}, None, 0
    with manifest_file.open('xb') as stream:
        for idx, user in enumerate(users):
            sequence = data.inters[user['user_id']]
            visible = sequence[:-1][-20:]
            groups = [[token_ids[token] for token in indices[str(item)]] for item in visible]
            context, labels = collate([data[idx]])
            raw, mask = context['input_ids'][0].tolist(), context['attention_mask'][0].tolist()
            assert labels == [user['target']] and str(sequence[-1]) == user['target_item_id']
            assert data.inter_data[idx]['inters'] == ', '.join(''.join(indices[str(item)]) for item in visible)
            assert normal_inputs[idx] == raw and mask == [1] * len(raw)
            off, off_audit = core.validate_v1_off(raw, mask, groups, sid_ids, 0, idx)
            assert off == raw and off_audit['attention_mask_sha256'] == core.canonical_sha256(mask)
            histories[len(groups)] += 1
            lengths[len(raw)] += 1
            if len(raw) > max_length:
                max_index, max_length = idx, len(raw)
            positions = [pos for pos, token in enumerate(raw) if token in sid_ids]
            assert len(positions) == len(groups) * 4
            group_positions = set(positions)
            row = {**user, 'visible_history_length': len(groups), 'encoded_length': len(raw),
                   'original_sha256': off_audit['original_sha256'],
                   'attention_mask_sha256': off_audit['attention_mask_sha256'], 'conditions': {}}
            expected = {m: [reference.relative(m, idx, n) for n in range(len(groups))] for m in (2, 3, 4)}
            for mode, active in core.MODES.items():
                transformed, audit = core.transform(raw, mask, groups, sid_ids, mode, idx)
                assert len(transformed) == len(raw) and audit['attention_mask_sha256'] == row['attention_mask_sha256']
                assert all(before == after for p, (before, after) in enumerate(zip(raw, transformed, strict=True))
                           if p not in group_positions)
                stats = mode_stats[mode]
                draws = []
                for n, (group, record, relative) in enumerate(
                    zip(groups, audit['groups'], expected[len(active)], strict=True)
                ):
                    moved = verify_group(raw, transformed, positions[n * 4:n * 4 + 4], group, record, relative, active)
                    pick = relative[0]
                    draws.append(pick)
                    stats['permutation_counts'][pick] += 1
                    stats['token_unchanged_groups'] += moved == group
                    for slot in range(4):
                        stats['moved_ids_by_position'][slot] += moved[slot] != group[slot]
                        stats['permuted_slots_by_position'][slot] += record['permutation'][slot] != slot
                assert audit['input_changed'] == (transformed != raw)
                stats['unchanged_users'] += transformed == raw
                row['conditions'][mode] = {'input_sha256': audit['shuffled_sha256'],
                                           'input_changed': audit['input_changed'], 'permutation_ids': draws}
                hashes[mode].update(core.canonical_bytes([idx, transformed]) + b'\n')
            assert row['conditions']['shuffle3a']['permutation_ids'] == row['conditions']['shuffle3b']['permutation_ids']
            if idx in selected_set:
                samples[idx] = row
            stream.write(core.canonical_bytes(row) + b'\n')

    # A true left-padded pair tests explicit masks even though historical batch=1 has no pad.
    smallest = min(selected, key=lambda idx: samples[idx]['encoded_length'])
    pair = [smallest, max_index]
    assert len(normal_inputs[smallest]) < max_length
    batched, _ = collate([data[idx] for idx in pair])
    for idx, padded, mask in zip(pair, batched['input_ids'].tolist(), batched['attention_mask'].tolist(), strict=True):
        padding = len(padded) - len(normal_inputs[idx])
        assert padded == [tokenizer.pad_token_id] * padding + normal_inputs[idx]
        assert mask == [0] * padding + [1] * len(normal_inputs[idx])
    totals = sum(n * count for n, count in histories.items())
    assert totals == 172929 and max(histories) == 20
    for stats in mode_stats.values():
        assert sum(stats['permutation_counts']) == totals
        stats['identity_permutations'] = stats['permutation_counts'][0]
        stats['actual_changed_token_count'] = sum(stats['moved_ids_by_position'])
    digest = frozen.sha(manifest_file)
    dump(output / 'single-shard.json', {
        'status': 'PASS_SINGLE_SHARD_CPU', 'manifest_sha256': digest, 'users': len(users), 'history_groups': totals,
        'world_size': 1, 'rank': 0, 'assignment': single, 'assignment_sha256': core.canonical_sha256(single),
        'normal_rank_assignments': ranks, 'normal_rank_assignment_sha256': [core.canonical_sha256(r) for r in ranks],
        'exact_interleave_of_frozen_normal_shards': True, 'permutation_ids_independently_verified': True,
        'preflight_assignment': selected, 'preflight_assignment_sha256': core.canonical_sha256(selected),
        'preflight_selection_sources': sample_sources, 'preflight_selection_commit': SAMPLE_COMMIT,
    })
    dump(output / 'samples.json', [samples[idx] for idx in selected])
    summary = {
        'status': 'PASS_CPU_V2_FULL', 'protocol': core.PROTOCOL, 'identity_policy': core.IDENTITY_POLICY,
        'shuffle_seed': core.SEED, 'serialization': core.SERIALIZATION, 'users': len(users), 'items': len(indices),
        'unique_sids': len(catalog), 'collision_groups': sum(len(items) > 1 for items in catalog.values()),
        'history_groups': totals, 'mode_user_records': len(users) * 4, 'mode_history_groups': totals * 4,
        'history_length_counts': histories, 'encoded_length_counts': lengths, 'versions': versions,
        'dataset': 'Instruments', 'token_ids': token_ids, 'tokenizer_vocab_sha256': VOCAB_SHA,
        'tokenizer_vocab_serialization': 'Sort by token string, compact JSON UTF-8 with ensure_ascii=False',
        'historical_vocab_sha256_order_dependent': HISTORICAL_VOCAB_SHA,
        'tokenizer_metadata': metadata, 'user_axis_sha256': AXIS_SHA, 'data_identity': identities,
        'preflight_assignments': {'0': selected}, 'manifest_sha256': digest,
        'normal_inputs_exact': True, 'normal_input_only_off_exact': True,
        'normal_prediction_files': normal_manifest['predictions'],
        'normal_sample_sha256': frozen.sha(output / 'normal-sample.jsonl'),
        'input_stream_sha256': {mode: stream.hexdigest() for mode, stream in hashes.items()},
        'modes': mode_stats, 'elapsed_seconds': time.monotonic() - started,
        'same_m_sampler_and_mapping_checked_independently_for_every_history_group': True,
        'cpu_only': True, 'offline_replay': offline,
    }
    dump(output / 'audit.json', summary)
    print(json.dumps({k: summary[k] for k in ('status', 'users', 'history_groups', 'manifest_sha256', 'elapsed_seconds')}))


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--offline', action='store_true', help='Use packaged tokenizer/data without large model weights')
    args = parser.parse_args()
    replay(load(args.config), args.output, args.offline)


if __name__ == '__main__':
    main()
