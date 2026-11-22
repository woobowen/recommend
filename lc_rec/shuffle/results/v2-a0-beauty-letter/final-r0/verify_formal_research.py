"""Independent, bounded, read-only audit of the original LETTER Beauty formal run."""

from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
from hashlib import file_digest, sha256
from itertools import permutations
from math import isfinite
from pathlib import Path
import json
import resource
import time

ROOT = Path('/data/bwwu/lc-rec/letter-shuffle-v2-beauty')
BASE = Path('/home/bwwu/grid-audits/sid-extensions-p0')
PREFIX = 'letter-beauty-v2-'
RUN = '1182dfd914c20bd3dc11786587eae6879fba6b9d'
NORMAL_RUN = '6e1b211d9c37f358c43b1540b40b17f6703150d2'
RUNNER = 'cbb963fab68c34cc977653afc46f260e7ede962cb4a8f105458331a0e0e19a75'
REPLAY = '2e69ea72e47eb978e3d7375c47aa62cd3d3588fa633bdf7b6d1cf7000adc23b2'
ASSIGNMENT = '22ee58fe9f948f50e5cbc92160b6dc7df7e2f633cb9e8bf3121da5fb35bcb088'
NORMAL_FILES = (
    '90acbd6bc0bb11c9173332035b43fc4bf40391128ff5a6dab0069486b5c80519',
    '74c8e582e8d142731e71371fc508e1549a9fbd61f34f162f1ae69ad9af0c3bf4',
)
GPU = 'GPU-59cac3bd-3645-668b-79c8-839e0d878893'
PROTOCOL = 'GRID-LETTER-SHUFFLE-V2'
AXIS = 22363
BEAMS = 20
GROUPS = 159966
MODES = {'shuffle2': (2, 3), 'shuffle3a': (0, 1, 2), 'shuffle3b': (1, 2, 3), 'shuffle4': (0, 1, 2, 3)}
CATALOGS = {n: tuple(permutations(range(n))) for n in range(2, 5)}
NORM_TOLERANCE = 1e-5
DIMS = {
    'num_beams': BEAMS, 'num_return_sequences': BEAMS, 'max_new_tokens': 10,
    'early_stopping': True, 'do_sample': False, 'use_cache': True,
    'pad_token_id': 0, 'eos_token_id': 2, 'bos_token_id': 1, 'length_penalty': 1.0,
}
RAW_DATA = {
    'index': 'be3fb890e59b5443824cdebc560ef349191b97f20d3aecad4727de2d6b212c82',
    'inter': '8809584b8b78fa6561771c88911ed65bd716fc9926fca1e581e6a1ab578ccfd5',
    'item': '74aeddf911548aadfcd1dab326c16678ed4ac74457e874da300d63a94bdb2330',
}


def canonical(x):
    return json.dumps(x, separators=(',', ':'), sort_keys=True, ensure_ascii=True, allow_nan=False).encode()


def digest(x):
    return sha256(canonical(x)).hexdigest()


def hashfile(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def file_state(path):
    metadata = path.stat()
    return metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns


def read(path):
    return json.loads(path.read_bytes())


def receipt(name, expected_request, runner_path):
    folder = BASE / 'jobs' / name
    accepted, request, status, cleanup = (read(folder / leaf) for leaf in
                                          ('accepted.json', 'request.json', 'status.json', 'cleanup.json'))
    assert request == expected_request, name
    assert request['job_id'] == status['job_id'] == name and request['capture_version'] == RUN, name
    assert accepted['request_sha256'] == digest(request), name
    assert accepted['runner_sha256'] == RUNNER and accepted['runner_path'] == str(runner_path), name
    assert status['status'] == 'SUCCEEDED' and status['exit_code'] == 0, name
    assert status['cleanup'] == cleanup and not status['worker_may_be_live'], name
    assert cleanup['verified'] and cleanup['last_observation']['verified_empty'], name
    assert status['worker']['argv'] == request['argv'], name
    start, end = (datetime.fromisoformat(status[key]) for key in ('started_at', 'ended_at'))
    assert start < end, name
    return {'job_id': name, 'started_at': start.isoformat(), 'ended_at': end.isoformat(),
            'active_wall_seconds': (end - start).total_seconds(), 'exit_code': status['exit_code'],
            'cleanup_reason': cleanup['reason'], 'cleanup_verified_empty': True,
            'request_sha256': accepted['request_sha256'], 'receipt_sha256': {
                leaf: hashfile(folder / leaf) for leaf in ('accepted.json', 'request.json', 'status.json', 'cleanup.json')}}


def rank_review(mode, plan, config_path, normal_report, single):
    path = ROOT / f'gpu-formal-{mode}-r0'
    trial = path / 'test'
    r = read(trial / 'rank0.json')
    b = read(trial / 'v2-binding.rank0.json')
    launch = read(path / f'{mode}.request.json')
    finished = read(path / f'{mode}.exit.json')
    config = read(config_path)
    req = plan['modes'][mode]['gpu']
    assert req['env']['CUDA_VISIBLE_DEVICES'] == r['gpu']['visible_devices'] == GPU
    assert r['gpu']['name'] == 'NVIDIA GeForce RTX 4090'
    assert '--preflight-users-per-rank' not in b['effective_argv'] and not r['preflight_only']
    assert r['assigned_users'] == r['completed_users'] == AXIS and r['status'] == 'PASS'
    assert r['assignment_sha256'] == single['assignment_sha256'] == ASSIGNMENT
    assert r['rank'] == r['local_rank'] == 0 and r['world_size'] == 1
    assert r['RUN_COMMIT'] == b['V2_RUN_COMMIT'] == RUN
    assert r['cwd'] == req['cwd'] and r['interpreter'] == req['argv'][0]
    assert all(r['data_identity'][tag]['sha256'] == value for tag, value in RAW_DATA.items())
    assert all(r[key] == normal_report[key] for key in (
        'data_identity', 'adapter_files', 'model', 'author_sources', 'versions',
        'strict_adapter_reload', 'allowed_callback_sha256', 'tokenizer'
    ))
    assert r['beauty_normal_binding'] == NORMAL_RUN
    assert r['tokenizer']['vocab_size'] == 32965 and len(r['tokenizer']['added_tokens']) == 965
    adapter = r['adapter_files']['adapter_model.bin']
    assert adapter['sha256'] == 'dd9f3d2983af7ce85ace99534bd23df9362d9cd91ac39d0b356b219cb790b1c1'
    assert r['model']['revision'] == '4782ad278652c7c71b72204d462d6d01eaaf7549'
    assert r['peft_actual_weight_reads'][0]['path'] == config['baseline']['adapter'] + '/adapter_model.bin'
    assert r['strict_adapter_reload']['keys'] == 450
    assert r['strict_adapter_reload']['elements'] == 290037760
    assert r['strict_adapter_reload']['all_active_embedding_head_and_sid_rows_checked']
    assert r['strict_adapter_reload']['no_missing_keys_tolerated']
    dtypes = Counter(item['loaded_dtype'] for item in r['strict_adapter_reload']['tensor_checks'].values())
    assert dtypes == {'torch.float32': 448, 'torch.bfloat16': 2}
    assert r['cuda_environment'] == {'actual_rank_environment': config['gpu_environment'], 'llama_import_and_strict_reload': 'PASS'}
    assert all(req['env'][key] == value for key, value in config['gpu_environment'].items())
    assert r['generation']['per_rank_batch_size'] == 1 and r['generation']['prompt_id'] == 0
    assert r['generation']['filter_items'] is False and r['generation']['oom_beam_reduction'] is False
    sh = r['shuffle']
    assert sh['cpu_manifest_sha256'] == REPLAY and not sh['capture_lookup']
    for record in (sh, b):
        assert record['mode'] == mode and record['protocol'] == PROTOCOL and record['seed'] == 42
        assert record['identity_policy'] == 'allow_identity' and tuple(record['active_positions']) == MODES[mode]
    assert {key: sh['effective_generation_config'][key] for key in DIMS} == DIMS
    assert hashfile(trial / 'derived.rank0.py') == b['derived_sha256']
    assert r['entry_sha256'] == b['frozen_source_sha256'] == hashfile(Path(b['frozen_source']))
    assert launch['preflight'] is False and finished['exit_code'] == 0 and finished['RUN_COMMIT'] == RUN
    assert launch['cuda_devices']['uuid'] == GPU and launch['cuda_devices']['physical_index'] == 1
    return {'rank_report_sha256': hashfile(trial / 'rank0.json'), 'binding_sha256': hashfile(trial / 'v2-binding.rank0.json'),
            'gpu_physical_index': launch['cuda_devices']['physical_index'], 'gpu_logical_index': 0,
            'model': r['model'], 'adapter': adapter, 'loaded_adapter_tensor_dtypes': dict(dtypes),
            'strict_model_reload': r['cuda_environment']['llama_import_and_strict_reload'],
            'weight_load_seconds': r['model_load_and_strict_verification_seconds'],
            'generation_elapsed_seconds': r['elapsed_generation_seconds'],
            'gpu_peak_reserved_bytes': r['gpu_peak_reserved_bytes'], 'effective_generation': DIMS,
            'data_identity': r['data_identity'], 'source_binding': r['entry_sha256'],
            'assigned_users': r['assigned_users'], 'completed_users': r['completed_users']}


def row_reader(streams, hashes, key):
    raw = streams[key].readline()
    assert raw.endswith(b'\n'), key
    hashes[key].update(raw)
    return json.loads(raw)


def independent_slot_choices(position, occurrence):
    choices = {}
    for n, perms in CATALOGS.items():
        stop = 2**256 - (2**256 % len(perms))
        retry = 0
        while True:
            key = [PROTOCOL, 42, n, position, occurrence, retry]
            d = int.from_bytes(sha256(canonical(key)).digest(), 'big')
            if d < stop:
                ind = d % len(perms)
                choices[n] = (ind, perms[ind])
                break
            retry += 1
    return choices


def candidate_trace(row, legal, occurrences, stats):
    candidates = row['candidates']
    assert len(candidates) == BEAMS
    scores = []
    identities = []
    for j, cand in enumerate(candidates):
        assert cand['generation_order'] == j
        seq = cand['generated_token_ids']
        assert cand['eos_position'] == 4 and seq[4] == 2
        assert all(v in (0, 2) for v in seq[5:])
        sid = ''.join(leg[token] for leg, token in zip(legal, seq[:4], strict=True))
        assert cand['sid'] == sid
        assert sid == cand['raw_decoded'].rsplit('Response:', 1)[-1].strip().replace(' ', '')
        assert isfinite(cand['sequence_score'])
        scores.append(cand['sequence_score'])
        identities.append((tuple(seq), sid))
        group_size = occurrences[sid]
        stats['valid_catalog_candidates'] += group_size > 0
        stats['invalid_catalog_candidates'] += group_size == 0
        stats['colliding_catalog_candidates'] += group_size > 1
    sorted_order = sorted(range(BEAMS), key=scores.__getitem__, reverse=True)
    assert row['sorted_generation_order'] == sorted_order
    assert row['relevance'] == [int(identities[j][1] == row['target']) for j in sorted_order]
    return identities, scores, sorted_order


def compare_identity(rowid, history_count, formal, archived, report):
    f_ids, fs, f_order = formal
    n_ids, ns, n_order = archived
    maximum = max(abs(a - b) for a, b in zip(fs, ns, strict=True))
    outside = sum(abs(a - b) > NORM_TOLERANCE + NORM_TOLERANCE * abs(b) for a, b in zip(fs, ns, strict=True))
    report['rows'] += 1
    report['compared_score_slots'] += BEAMS
    report['slots_outside_score_tolerance'] += outside
    report['exact_slot_candidate_rows'] += f_ids == n_ids
    report['exact_sorted_order_rows'] += f_order == n_order
    report['exact_ranked_candidate_rows'] += [f_ids[j] for j in f_order] == [n_ids[j] for j in n_order]
    report['exact_positional_score_rows'] += fs == ns
    report['history_tolerance_pass_rows'] += f_ids == n_ids and f_order == n_order and outside == 0
    report['max_positional_score_abs_delta'] = max(report['max_positional_score_abs_delta'], maximum)
    if f_ids != n_ids or f_order != n_order or outside:
        report['departures'].append({'row_idx': rowid, 'history_groups': history_count,
                                     'generated_candidate_order_matches': f_ids == n_ids,
                                     'sorted_order_matches': f_order == n_order,
                                     'out_of_tolerance_scores': outside, 'max_positional_score_abs_delta': maximum})


def identity_report():
    return dict(rows=0, compared_score_slots=0, slots_outside_score_tolerance=0,
                exact_slot_candidate_rows=0, exact_sorted_order_rows=0, exact_ranked_candidate_rows=0,
                exact_positional_score_rows=0, history_tolerance_pass_rows=0, max_positional_score_abs_delta=0.0, departures=[])


def main():
    started_wall, started_cpu = time.monotonic(), time.process_time()
    started = datetime.now(timezone.utc)
    parent_name = PREFIX + 'group-series-r0'
    parent = read(BASE / 'jobs' / parent_name / 'request.json')
    planpath = Path(parent['argv'][parent['argv'].index('--plan') + 1])
    plan = read(planpath)
    assert plan['formal_run_commit'] == plan['orchestrator_commit'] == RUN
    assert list(plan['modes']) == list(MODES)
    assert any(entry['path'] == str(planpath) and entry['sha256'] == hashfile(planpath) for entry in parent['inputs'])
    runner_path = Path(plan['runner'])
    assert hashfile(runner_path) == plan['runner_sha256'] == RUNNER
    config_path = Path(plan['config'])
    config = read(config_path)
    assert config['users'] == AXIS and config['dataset'] == 'Beauty'
    release = read(Path(parent['cwd']) / 'RUN_VERSION.json')
    assert release['RUN_COMMIT'] == RUN and release['runner_sha256'] == RUNNER
    for item in release['files']:
        path = Path(parent['cwd']) / item['path']
        assert hashfile(path) == item['sha256']
    parent_receipt = receipt(parent_name, parent, runner_path)
    order = [plan['modes'][mode][stage] for mode in MODES for stage in ('gpu', 'cpu')] + [plan['paired']]
    children = [receipt(req['job_id'], req, runner_path) for req in order]
    chain = read(ROOT / 'group-series-r0' / 'summary.json')
    assert chain['status'] == 'PASS_V2_GROUP_COMPUTE' and chain['formal_run_commit'] == RUN
    assert [row['job_id'] for row in chain['stages']] == [row['job_id'] for row in children]
    assert all(a['ended_at'] <= b['started_at'] for a, b in zip(children, children[1:]))
    assert parent_receipt['started_at'] <= children[0]['started_at'] <= children[-1]['ended_at'] <= parent_receipt['ended_at']
    paired = read(ROOT / 'cpu-summary-r0' / 'summary.json')
    assert paired['status'] == 'PASS_V2_PAIRED' and paired['users'] == AXIS

    baseline = config['baseline']
    normaldir = Path(baseline['normal_test_dir'])
    datadir = Path(baseline['data-root']) / 'Beauty'
    replaydir = ROOT / 'cpu-replay-r3'
    audit = read(replaydir / 'audit.json')
    single = read(replaydir / 'single-shard.json')
    assert audit['status'] == 'PASS_CPU_V2_FULL' and audit['manifest_sha256'] == single['manifest_sha256'] == REPLAY
    assert audit['history_groups'] == single['history_groups'] == GROUPS
    assert single['world_size'] == 1 and single['exact_interleave_of_frozen_normal_shards']
    data = {}
    for tag, sha in RAW_DATA.items():
        path = datadir / f'Beauty.{tag}.json'
        raw = path.read_bytes()
        assert sha256(raw).hexdigest() == sha
        data[tag] = json.loads(raw)
    catalog, inter = data['index'], data['inter']
    users = list(inter)
    assert catalog.keys() == data['item'].keys() and len(catalog) == 12101
    assert len(users) == AXIS and digest(users) == audit['user_axis_sha256'] == config['user_axis_sha256']
    occurrences = Counter(''.join(sid) for sid in catalog.values())
    assert len(occurrences) == 12088 and sum(n > 1 for n in occurrences.values()) == 11
    vocab = {tok: i for i, tok in enumerate(sorted({t for sid in catalog.values() for t in sid}), 32000)}
    assert len(vocab) == 965 and vocab == audit['token_ids']
    legal = [{vocab[sid[i]]: sid[i] for sid in catalog.values()} for i in range(4)]
    vocab_ids = set(vocab.values())
    normal_reports = [read(normaldir / f'rank{i}.json') for i in (0, 1)]
    assert [r['completed_users'] for r in normal_reports] == [11182, 11181]
    assert all(r['RUN_COMMIT'] == NORMAL_RUN and r['status'] == 'PASS' for r in normal_reports)
    rank_reviews = {mode: rank_review(mode, plan, config_path, normal_reports[0], single) for mode in MODES}

    manifest = []
    mhash = sha256()
    with (replaydir / 'inputs.jsonl').open('rb') as stream:
        for i, line in enumerate(stream):
            assert line.endswith(b'\n')
            mhash.update(line)
            a = json.loads(line)
            assert (a['row_idx'], a['user_id']) == (i, users[i])
            manifest.append(a)
    assert len(manifest) == AXIS and mhash.hexdigest() == REPLAY

    raw_paths = {mode: ROOT / f'gpu-formal-{mode}-r0' / 'test' / 'predictions.rank0.jsonl' for mode in MODES}
    for i in (0, 1):
        raw_paths[f'normal{i}'] = normaldir / f'predictions.rank{i}.jsonl'
    before = {key: file_state(path) for key, path in raw_paths.items()}
    hashes = {key: sha256() for key in raw_paths}
    by_mode = {mode: Counter() for mode in MODES}
    move = {mode: [0]*4 for mode in MODES}
    histogram = {mode: Counter() for mode in MODES}
    identity = {mode: identity_report() for mode in MODES}
    input_rows = {mode: [None]*AXIS for mode in MODES}
    order_rows = []
    shared = natural_groups = 0
    normal_candidate_counts = Counter()
    grouped_vocab = {item: tuple(vocab[s] for s in sid) for item, sid in catalog.items()}
    mask_and_positions = {}
    with ExitStack() as stack:
        streams = {key: stack.enter_context(path.open('rb')) for key, path in raw_paths.items()}
        for pos in range(AXIS):
            old = row_reader(streams, hashes, f'normal{pos%2}')
            j = old['row_idx']
            order_rows.append(j)
            uid = users[j]
            target_id = str(inter[uid][-1])
            target = ''.join(catalog[target_id])
            history = [str(item) for item in inter[uid][:-1][-20:]]
            natural_groups += len(history)
            rpl = manifest[j]
            ids = old['input_token_ids']
            offsets = [k for k, value in enumerate(ids) if value in vocab_ids]
            assert len(offsets) == 4 * len(history) == 4 * rpl['visible_history_length'], j
            assert (old['user_id'], old['target_item_id'], old['target']) == (uid, target_id, target), j
            actual_len = len(ids)
            assert actual_len == old['input_nonpadding_tokens'] == rpl['encoded_length'], j
            assert digest(ids) == rpl['original_sha256'], j
            if actual_len not in mask_and_positions:
                mask_and_positions[actual_len] = digest([1] * actual_len), digest(list(range(actual_len)))
            mask_sha, position_sha = mask_and_positions[actual_len]
            assert rpl['attention_mask_sha256'] == mask_sha, j
            expected = {mode: ids.copy() for mode in MODES}
            ids_seen = {mode: [] for mode in MODES}
            for occurrence, item in enumerate(history):
                base = offsets[4 * occurrence]
                assert offsets[4*occurrence:4*occurrence+4] == list(range(base, base+4)), j
                group = grouped_vocab[item]
                assert tuple(ids[base:base+4]) == group, j
                choices = independent_slot_choices(j, occurrence)
                for mode, slots in MODES.items():
                    ind, permutation = choices[len(slots)]
                    ids_seen[mode].append(ind)
                    histogram[mode][ind] += 1
                    for destination, origin in zip(slots, permutation, strict=True):
                        moved = group[slots[origin]]
                        expected[mode][base+destination] = moved
                        move[mode][destination] += moved != group[destination]
            assert ids_seen['shuffle3a'] == ids_seen['shuffle3b'], j
            shared += len(history)
            reference_trace = None
            for mode, slots in MODES.items():
                row = row_reader(streams, hashes, mode)
                side = row['shuffle']
                prepared = rpl['conditions'][mode]
                new = expected[mode]
                assert ids_seen[mode] == prepared['permutation_ids'], (j, mode)
                assert (row['row_idx'], row['rank'], row['prompt_id']) == (j, 0, 0), (j, mode)
                assert (row['user_id'], row['target_item_id'], row['target']) == (uid, target_id, target), (j, mode)
                assert row['original_callback_verified_all_tokens'], (j, mode)
                assert (side['protocol'], side['identity_policy'], side['mode'], side['seed']) == (PROTOCOL, 'allow_identity', mode, 42), (j, mode)
                assert tuple(side['active_positions']) == slots and side['row_idx'] == j, (j, mode)
                assert side['history_group_count'] == len(history) and side['original_sha256'] == rpl['original_sha256'], (j, mode)
                assert side['cpu_manifest_sha256'] == REPLAY and side['attention_mask_sha256'] == mask_sha, (j, mode)
                assert row['input_nonpadding_tokens'] == actual_len and row['input_token_ids'] == new, (j, mode)
                actual = digest(new)
                assert side['shuffled_sha256'] == prepared['input_sha256'] == actual, (j, mode)
                assert side['input_changed'] == prepared['input_changed'] == (new != ids), (j, mode)
                prefill = side['actual_prefill']
                assert prefill['input_sha256'] == actual and prefill['attention_mask_sha256'] == mask_sha, (j, mode)
                assert prefill['position_ids_sha256'] == position_sha, (j, mode)
                assert (prefill['cache_length'], prefill['beam_expansion']) == (0, BEAMS), (j, mode)
                input_rows[mode][j] = canonical([j, new]) + b'\n'
                trace = candidate_trace(row, legal, occurrences, by_mode[mode])
                by_mode[mode]['complete_candidate_rows'] += BEAMS
                by_mode[mode]['fully_verified_input_rows'] += 1
                if new == ids:
                    if reference_trace is None:
                        reference_trace = candidate_trace(old, legal, occurrences, normal_candidate_counts)
                    compare_identity(j, len(history), trace, reference_trace, identity[mode])
        assert all(stream.read(1) == b'' for stream in streams.values())
    assert len(set(order_rows)) == AXIS and set(order_rows) == set(range(AXIS))
    assert digest(order_rows) == ASSIGNMENT
    assert [digest(order_rows[i::2]) for i in range(2)] == single['normal_rank_assignment_sha256']
    assert natural_groups == shared == GROUPS
    assert before == {key: file_state(path) for key, path in raw_paths.items()}
    raw_sha = {key: value.hexdigest() for key, value in hashes.items()}
    assert tuple(raw_sha[f'normal{i}'] for i in range(2)) == NORMAL_FILES

    mode_report = {}
    for mode in MODES:
        counts = [histogram[mode][n] for n in range(len(CATALOGS[len(MODES[mode])]))]
        assert counts == audit['modes'][mode]['permutation_counts'] and sum(counts) == GROUPS
        assert move[mode] == audit['modes'][mode]['moved_ids_by_position']
        assert identity[mode]['rows'] == audit['modes'][mode]['unchanged_users']
        all_inputs = sha256()
        for item in input_rows[mode]:
            assert item
            all_inputs.update(item)
        assert all_inputs.hexdigest() == audit['input_stream_sha256'][mode]
        eval_summary = read(ROOT / f'cpu-eval-{mode}-r0' / 'summary.json')
        device_summary = read(ROOT / f'gpu-formal-{mode}-r0' / 'test' / 'metrics.json')
        prec = eval_summary['prediction_files']
        assert len(prec) == 1 and prec[0]['path'] == str(raw_paths[mode])
        assert prec[0]['sha256'] == raw_sha[mode] and prec[0]['bytes'] == before[mode][2]
        assert prec[0]['users'] == AXIS and prec[0]['assignment_sha256'] == ASSIGNMENT
        assert device_summary['prediction_files'][0]['sha256'] == raw_sha[mode]
        assert eval_summary['status'] == 'PASS_V2_EVALUATION' and eval_summary['users'] == AXIS
        assert eval_summary['run_commit'] == RUN and eval_summary['evaluation_commit'] == RUN
        assert eval_summary['candidate_count'] == by_mode[mode]['complete_candidate_rows'] == AXIS * BEAMS
        assert eval_summary['official_cce_full_user_comparisons'] == AXIS
        full_stats = eval_summary['sid_statistics']['20']
        assert (full_stats['valid'], full_stats['invalid'], full_stats['colliding']) == tuple(by_mode[mode][key] for key in
                ('valid_catalog_candidates', 'invalid_catalog_candidates', 'colliding_catalog_candidates'))
        mode_report[mode] = dict(prediction_bytes=before[mode][2], prediction_sha256=raw_sha[mode],
                                 prediction_file=str(raw_paths[mode]), prediction_input_stream_sha256=all_inputs.hexdigest(),
                                 permutation_counts=counts, moved_token_ids_by_position=move[mode],
                                 fixed_slot_or_outside_group_violations=0, mismatch_of_input_hash_or_boundary=0,
                                 observations=dict(by_mode[mode]), natural_identity=identity[mode], rank=rank_reviews[mode])

    result = {'status': 'PASS_NONAUTHOR_FULL_FORMAL_RESEARCH_EVIDENCE', 'reviewer': '/root/formal_review',
              'scope': 'All saved 4x22363 formal input rows in the original run, with original Normal record interleave, CPU manifest, candidate structural evidence, no inference rerun.',
              'run_commit': RUN, 'runner_sha256': RUNNER, 'plan_sha256': hashfile(planpath),
              'parent': parent_receipt, 'children': children, 'checked_installed_source_files': len(release['files']),
              'dataset_identity': RAW_DATA, 'replay_file_sha256': REPLAY, 'normal_file_sha256': NORMAL_FILES,
              'formal_assignment_sha256': ASSIGNMENT, 'shared_s3a_s3b_relative_group_draws_checked': shared,
              'complete_formal_users': len(MODES) * AXIS, 'complete_formal_candidates': len(MODES) * AXIS * BEAMS,
              'natural_identity_tolerance': {'abs': NORM_TOLERANCE, 'relative_to_archived_score': NORM_TOLERANCE,
                                            'also_requires_same_generated_candidate_order_and_sorted_order': True},
              'modes': mode_report, 'started_at_utc': started.isoformat(),
              'completed_at_utc': datetime.now(timezone.utc).isoformat(),
              'server_wall_seconds': time.monotonic() - started_wall, 'server_cpu_seconds': time.process_time() - started_cpu,
              'server_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
