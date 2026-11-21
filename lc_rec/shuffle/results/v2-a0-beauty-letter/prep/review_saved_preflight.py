"""Read saved Beauty evidence using only the standard library and frozen data."""
import argparse
from collections import Counter
from hashlib import file_digest, sha256
from itertools import permutations
import json
import math
import time
from pathlib import Path


MODES = {
    'shuffle2': (2, 3),
    'shuffle3a': (0, 1, 2),
    'shuffle3b': (1, 2, 3),
    'shuffle4': (0, 1, 2, 3),
}
PROTOCOL = 'GRID-LETTER-SHUFFLE-V2'
ADAPTER = {'bytes': 580231186, 'sha256': 'dd9f3d2983af7ce85ace99534bd23df9362d9cd91ac39d0b356b219cb790b1c1'}
ORIGINAL = '6e1b211d9c37f358c43b1540b40b17f6703150d2'
EXPECTED_AXIS = '7430079e2b917dfa5e4bb8ead2702fb6c622fb48ffb684dd2889780f9480af7f'
EXPECTED_SELECTION = '629b462bf517a8c86fea0b587491d20826e175f3d4652e8d8a6cfdae0a0b7700'
EXPECTED_REPLAY = '2e69ea72e47eb978e3d7375c47aa62cd3d3588fa633bdf7b6d1cf7000adc23b2'
EXPECTED_NORMAL = (
    '90acbd6bc0bb11c9173332035b43fc4bf40391128ff5a6dab0069486b5c80519',
    '74c8e582e8d142731e71371fc508e1549a9fbd61f34f162f1ae69ad9af0c3bf4',
)
GPU_ENVIRONMENT = {
    'CUDA_HOME': '/data/bwwu/lc-rec/offload-20261108/cuda-12.4',
    'LD_LIBRARY_PATH': '/data/bwwu/lc-rec/offload-20261108/cuda-12.4/lib64:/data/bwwu/envs/grid-20261009-e61baf3/lib/python3.10/site-packages/torch/lib',
}
RAW_AXIS = {
    'index': 'be3fb890e59b5443824cdebc560ef349191b97f20d3aecad4727de2d6b212c82',
    'inter': '8809584b8b78fa6561771c88911ed65bd716fc9926fca1e581e6a1ab578ccfd5',
    'item': '74aeddf911548aadfcd1dab326c16678ed4ac74457e874da300d63a94bdb2330',
}


def read(path):
    return json.loads(path.read_bytes())


def rows(path):
    with path.open() as stream:
        yield from map(json.loads, stream)


def identify(path):
    with path.open('rb') as stream:
        return {'bytes': path.stat().st_size, 'sha256': file_digest(stream, 'sha256').hexdigest()}


def digest(value):
    return sha256(json.dumps(value, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def reference(size, row, occurrence):
    possibilities = tuple(permutations(range(size)))
    cap = 2**256 - 2**256 % len(possibilities)
    counter = 0
    while True:
        key = json.dumps([PROTOCOL, 42, size, row, occurrence, counter], separators=(',', ':')).encode()
        integer = int.from_bytes(sha256(key).digest(), 'big')
        if integer < cap:
            index = integer % len(possibilities)
            return index, possibilities[index], counter
        counter += 1


def validate_candidate_row(row, legal, target):
    candidates = row['candidates']
    assert len(candidates) == 20 and [c['generation_order'] for c in candidates] == list(range(20))
    for c in candidates:
        seq = c['generated_token_ids']
        assert c['eos_position'] == 4 and seq[4] == 2
        assert all(seq[i] in legal[i] for i in range(4))
        assert c['sid'] == ''.join(legal[i][seq[i]] for i in range(4))
        assert set(seq[5:]) <= {0, 2}
        assert c['sid'] == c['raw_decoded'].split('Response:')[-1].strip().replace(' ', '')
        assert math.isfinite(c['sequence_score'])
    order = sorted(range(20), key=lambda j: candidates[j]['sequence_score'], reverse=True)
    assert row['sorted_generation_order'] == order
    assert row['relevance'] == [int(candidates[j]['sid'] == target) for j in order]


def compare_identity(row, normal):
    assert row['row_idx'] == normal['row_idx'] == 20014
    assert row['input_token_ids'] == normal['input_token_ids']
    assert row['sorted_generation_order'] == normal['sorted_generation_order']
    deviations = []
    for current, archived in zip(row['candidates'], normal['candidates'], strict=True):
        assert current['sid'] == archived['sid']
        assert current['generated_token_ids'] == archived['generated_token_ids']
        error = abs(current['sequence_score'] - archived['sequence_score'])
        assert error <= 1e-5 + 1e-5 * abs(archived['sequence_score'])
        deviations.append(error)
    return max(deviations)


def run(args):
    config = {}
    for kind, identity in RAW_AXIS.items():
        raw = (args.data / f'Beauty.{kind}.json').read_bytes()
        assert sha256(raw).hexdigest() == identity
        config[kind] = json.loads(raw)
    catalog, inter = config['index'], config['inter']
    users = list(inter)
    assert len(users) == 22363 and len(catalog) == 12101
    assert catalog.keys() == config['item'].keys()
    group_sizes = Counter(''.join(value) for value in catalog.values())
    assert len(group_sizes) == 12088 and Counter(n for n in group_sizes.values() if n > 1) == {2: 9, 3: 2}
    audit = read(args.cpu / 'audit.json')
    shard = read(args.cpu / 'single-shard.json')
    manifest = args.cpu / 'inputs.jsonl'
    assert sha256(manifest.read_bytes()).hexdigest() == shard['manifest_sha256'] == audit['manifest_sha256'] == EXPECTED_REPLAY
    assert audit['status'] == 'PASS_CPU_V2_FULL' and audit['history_groups'] == 159966
    assert audit['users'] == len(users) and audit['user_axis_sha256'] == digest(users) == EXPECTED_AXIS
    selected = shard['preflight_assignment']
    assert len(selected) == len(set(selected)) == 32 and digest(selected) == shard['preflight_assignment_sha256'] == EXPECTED_SELECTION
    expected_rows = {row['row_idx']: row for row in rows(manifest) if row['row_idx'] in selected}
    lookup = audit['token_ids']
    vocab = set(lookup.values())
    assert len(vocab) == len(lookup) == 965
    independent_lookup = {token: i for i, token in enumerate(sorted({t for sid in catalog.values() for t in sid}), 32000)}
    assert lookup == independent_lookup
    legal = [{lookup[sid[i]]: sid[i] for sid in catalog.values()} for i in range(4)]
    frozen_normal, normal_files = normal_subset(args.normal, set(selected))
    normal_boundary = read(args.normal / 'rank0.json')
    assert normal_boundary['RUN_COMMIT'] == ORIGINAL
    stages = read(args.preflight / 'summary.json')
    assert stages['status'] == 'PASS_V2_PREFLIGHT' and stages['RUN_COMMIT'] == args.commit
    assert [s['mode']['mode'] for s in stages['stages']] == list(MODES)
    observed_devices = {stage['cuda_devices']['uuid'] for stage in stages['stages']}
    assert len(observed_devices) == 1
    assert all(stage['preflight'] and stage['exit_code'] == 0 and stage['RUN_COMMIT'] == args.commit
               for stage in stages['stages'])
    uuid = observed_devices.pop()
    summary = {}
    matrix = {'normal': metric_matrix(frozen_normal, selected, group_sizes)}
    for name, slots in MODES.items():
        root = args.preflight / name
        rank = read(root / 'rank0.json')
        assert rank['status'] == 'PREFLIGHT_INTERFACE_PASS' and rank['preflight_only']
        assert rank['RUN_COMMIT'] == args.commit and rank['beauty_normal_binding'] == ORIGINAL
        assert rank['world_size'] == 1 and rank['rank'] == rank['local_rank'] == 0
        assert rank['completed_users'] == rank['assigned_users'] == 32
        assert rank['adapter_files']['adapter_model.bin'] == ADAPTER
        assert rank['gpu']['visible_devices'] == uuid
        assert rank['cuda_environment'] == {'actual_rank_environment': GPU_ENVIRONMENT,
                                             'llama_import_and_strict_reload': 'PASS'}
        assert rank['adapter_files'] == normal_boundary['adapter_files']
        assert rank['data_identity'] == normal_boundary['data_identity']
        assert rank['model'] == normal_boundary['model'] and rank['versions'] == normal_boundary['versions']
        assert rank['author_sources'] == normal_boundary['author_sources']
        assert rank['allowed_callback_sha256'] == normal_boundary['allowed_callback_sha256']
        assert rank['tokenizer']['added_tokens'] == lookup and rank['tokenizer']['vocab_size'] == 32965
        assert rank['strict_adapter_reload'] == normal_boundary['strict_adapter_reload']
        assert rank['strict_adapter_reload']['all_active_embedding_head_and_sid_rows_checked']
        assert rank['strict_adapter_reload']['no_missing_keys_tolerated']
        assert all(t['finite_and_exact_after_explicit_dtype_conversion']
                   for t in rank['strict_adapter_reload']['tensor_checks'].values())
        assert rank['generation']['prompt_id'] == 0 and rank['generation']['per_rank_batch_size'] == 1
        assert rank['strict_adapter_reload']['keys'] == 450
        assert rank['strict_adapter_reload']['elements'] == 290037760
        dtypes = Counter(r['loaded_dtype'] for r in rank['strict_adapter_reload']['tensor_checks'].values())
        assert dtypes == {'torch.float32': 448, 'torch.bfloat16': 2}
        assert rank['assignment_sha256'] == digest(selected)
        assert rank['generation']['filter_items'] is False
        effective = rank['shuffle']['effective_generation_config']
        expected_settings = {'num_beams': 20, 'num_return_sequences': 20, 'do_sample': False,
                             'max_new_tokens': 10, 'use_cache': True, 'early_stopping': True,
                             'eos_token_id': 2, 'pad_token_id': 0, 'length_penalty': 1.0}
        assert all(effective[k] == value for k, value in expected_settings.items())
        tally = Counter()
        identities = {}
        checked = {}
        for row_idx, row in zip(selected, rows(root / 'predictions.rank0.jsonl'), strict=True):
            checked[row_idx] = row
            uid = users[row_idx]
            sequence = inter[uid]
            held_out = str(sequence[-1])
            target = ''.join(catalog[held_out])
            assert (row['rank'], row['prompt_id'], row['row_idx']) == (0, 0, row_idx)
            assert (row['user_id'], row['target_item_id'], row['target']) == (uid, held_out, target)
            intervention = row['shuffle']
            assert intervention['mode'] == name and intervention['protocol'] == PROTOCOL
            assert intervention['identity_policy'] == 'allow_identity' and intervention['seed'] == 42
            assert intervention['active_positions'] == list(slots)
            assert intervention['cpu_manifest_sha256'] == EXPECTED_REPLAY
            assert intervention['row_idx'] == row_idx
            old = intervention['original_input_ids']
            mask = [1] * len(old)
            assert len(old) == row['input_nonpadding_tokens']
            baseline = expected_rows[row_idx]
            assert digest(old) == baseline['original_sha256']
            assert digest(mask) == baseline['attention_mask_sha256']
            group_offsets = [i for i, token in enumerate(old) if token in vocab]
            history = sequence[:-1][-20:]
            new = old.copy()
            picks = []
            assert len(group_offsets) == 4 * len(history)
            for occurrence, (item, proof) in enumerate(zip(history, intervention['groups'], strict=True)):
                group = [lookup[token] for token in catalog[str(item)]]
                offsets = group_offsets[occurrence * 4:(occurrence + 1) * 4]
                assert offsets == list(range(offsets[0], offsets[0] + 4))
                assert group == [old[offset] for offset in offsets]
                pick, pattern, counter = reference(len(slots), row_idx, occurrence)
                assert proof['positions'] == offsets and proof['relative_permutation'] == list(pattern)
                assert proof['permutation_id'] == pick and proof['rejection_counter'] == counter
                for destination, source in zip(slots, pattern, strict=True):
                    new[offsets[destination]] = group[slots[source]]
                assert proof['original_ids'] == group and proof['shuffled_ids'] == [new[offset] for offset in offsets]
                assert proof['nonidentity'] == (pick != 0)
                picks.append(pick)
                tally[pick] += 1
            planned = baseline['conditions'][name]
            assert picks == planned['permutation_ids']
            assert new == row['input_token_ids'] and digest(new) == planned['input_sha256']
            assert intervention['input_changed'] == planned['input_changed'] == (new != old)
            boundary = intervention['actual_prefill']
            assert boundary['input_ids'] == new and boundary['attention_mask'] == mask
            assert boundary['input_sha256'] == digest(new) and boundary['attention_mask_sha256'] == digest(mask)
            assert boundary['position_ids'] == list(range(len(old)))
            assert boundary['position_ids_sha256'] == digest(boundary['position_ids'])
            assert (boundary['cache_length'], boundary['beam_expansion'], boundary['lookup_exact']) == (0, 20, True)
            assert row['original_callback_verified_all_tokens']
            validate_candidate_row(row, legal, target)
            assert frozen_normal[row_idx]['target'] == target
            assert frozen_normal[row_idx]['user_id'] == uid
            assert old == frozen_normal[row_idx]['input_token_ids']
            if old == new:
                identities[row_idx] = compare_identity(row, frozen_normal[row_idx])
        assert list(identities) == ([20014] if name == 'shuffle2' else [])
        matrix[name] = metric_matrix(checked, selected, group_sizes)
        hashes = {leaf: identify(root / leaf) for leaf in ('predictions.rank0.jsonl', 'rank0.json', 'v2-binding.rank0.json')}
        binding = read(root / 'v2-binding.rank0.json')
        assert binding['mode'] == name and binding['V2_RUN_COMMIT'] == args.commit
        assert binding['derived_sha256'] == identify(root / 'derived.rank0.py')['sha256']
        summary[name] = {'users': 32, 'candidates': 640, 'catalog_histogram': dict(sorted(tally.items())), 'files': hashes,
                         'natural_identity_maximum_score_error': identities,
                         'actual_rank_cuda_environment': rank['cuda_environment']['actual_rank_environment'],
                         'strict_reload_dtype_counts': dict(dtypes),
                         'actual_model_class': rank['shuffle']['actual_model_module'],
                         'generation_seconds': rank['elapsed_generation_seconds'],
                         'load_seconds': rank['model_load_and_strict_verification_seconds']}
    assert summary['shuffle3a']['catalog_histogram'] == summary['shuffle3b']['catalog_histogram']
    inputs = {mode: identify(args.preflight / f'{mode}.request.json') for mode in MODES}
    exits = {mode: identify(args.preflight / f'{mode}.exit.json') for mode in MODES}
    result = {
        'reviewer': '/root/research_review', 'status': 'PASS', 'RUN_COMMIT': args.commit,
        'protocol': PROTOCOL, 'job_id': stages['job_id'],
        'preflight_summary_sha256': identify(args.preflight / 'summary.json')['sha256'],
        'replay_audit_sha256': identify(args.cpu / 'audit.json')['sha256'],
        'scope': ('Entire frozen 32-user sample in each of four independently checked conditions. '
                  'Model hook source and actual per-row receipts bind the live in-memory checks.'),
        'users_per_mode': len(selected), 'candidate_rows': len(MODES) * len(selected) * 20,
        'preflight_assignment_sha256': EXPECTED_SELECTION, 'device_uuid': uuid,
        'frozen_normal_predictions': normal_files, 'mode_requests': inputs, 'mode_exits': exits,
        'modes': summary, 'global_row_indices': selected, 'engineering_matrix': matrix,
        'sample_user_axis_sha256': digest([users[i] for i in selected]),
        'global_user_axis_sha256': EXPECTED_AXIS, 'cpu_manifest_sha256': EXPECTED_REPLAY,
        'metric_keys': METRICS,
    }
    if args.runner:
        state = read(args.runner / 'status.json')
        request = read(args.runner / 'request.json')
        cleanup = read(args.runner / 'cleanup.json')
        assert request['job_id'] == state['job_id'] == stages['job_id']
        assert request['capture_version'] == args.commit
        assert state['status'] == 'SUCCEEDED' and state['exit_code'] == 0 and state['cleanup']['verified']
        assert cleanup['verified'] and cleanup['last_observation']['verified_empty']
        assert request['env']['CUDA_VISIBLE_DEVICES'] == uuid
        assert all(request['env'][key] == value for key, value in GPU_ENVIRONMENT.items())
        result['runner_job_files'] = {key: identify(args.runner / key)
                                      for key in ('request.json', 'status.json', 'cleanup.json')}
        result['request_cuda_environment'] = GPU_ENVIRONMENT
    return result


def normal_subset(path, expected):
    selected = {}
    identities = []
    for i, known in enumerate(EXPECTED_NORMAL):
        source = path / f'predictions.rank{i}.jsonl'
        records = {}
        identity = sha256()
        with source.open('rb') as stream:
            for line in stream:
                identity.update(line)
                candidate = json.loads(line)
                if candidate['row_idx'] in expected:
                    assert candidate['row_idx'] not in records
                    records[candidate['row_idx']] = candidate
        assert identity.hexdigest() == known
        assert not selected.keys() & records.keys()
        selected.update(records)
        identities.append({'bytes': source.stat().st_size, 'sha256': known, 'rank': i})
    assert selected.keys() == expected
    return selected, identities


CUTOFFS = (5, 10, 20)
METRICS = [*(f'hit@{k}' for k in (1, *CUTOFFS)),
           *(f'{key}@{k}' for key in ('ndcg', 'mrr', 'ItemHit', 'ItemNDCG') for k in CUTOFFS)]


def metric_values(ranked, target, sizes):
    matching_positions = [i for i, sid in enumerate(ranked, 1) if sid == target]
    first = matching_positions[0] if matching_positions else 21
    item_prefix = sum(sizes.get(sid, 1) for sid in ranked[:first - 1])
    item_ranks = range(item_prefix + 1, item_prefix + sizes[target] + 1) if first <= 20 else ()
    return [
        *(float(first <= k) for k in (1, *CUTOFFS)),
        *(math.fsum(1.0 / math.log2(p + 1) for p in matching_positions if p <= k) for k in CUTOFFS),
        *(1.0 / first if first <= k else 0.0 for k in CUTOFFS),
        *(sum(p <= k for p in item_ranks) / sizes[target] for k in CUTOFFS),
        *(math.fsum(1.0 / math.log2(p + 1) for p in item_ranks if p <= k) / sizes[target] for k in CUTOFFS),
    ]


def metric_matrix(rows, selection, sizes):
    result = []
    for row_idx in selection:
        row = rows[row_idx]
        by_score = sorted(row['candidates'], key=lambda c: (-c['sequence_score'], c['generation_order']))
        result.append(metric_values([entry['sid'] for entry in by_score], row['target'], sizes))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--commit', required=True)
    for name in ('data', 'cpu', 'preflight', 'normal'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--runner', type=Path)
    assert __debug__, 'assertions must remain enabled'
    args = p.parse_args()
    started = time.monotonic()
    receipt = run(args)
    receipt['review_source'] = identify(Path(__file__))
    receipt['local_cpu_wall_seconds'] = time.monotonic() - started
    print(json.dumps(receipt, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
