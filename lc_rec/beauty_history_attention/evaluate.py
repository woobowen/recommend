"""Frozen LETTER Beauty generation and Shuffle V2 evaluation for B and A'."""

import argparse
from contextlib import ExitStack
from copy import deepcopy
import datetime as dt
import difflib
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import socket
import sys
import time
import traceback
from types import SimpleNamespace

from lc_rec.beauty_history_attention.identity import (
    IDENTITY_FILE, tokenizer_identity, verify_adapter_identity, verify_base_model,
)
from lc_rec.shuffle.core import transform as verify_history_groups
from lc_rec.shuffle_v2 import beauty, core, statistics
from lc_rec.shuffle_v2.beauty_evaluate import derive_metrics
from lc_rec.shuffle_v2.replay import frozen, semantic_hash

CONDITIONS = ('normal', *core.MODES)
REPLAY_DIR = Path('/data/bwwu/lc-rec/letter-shuffle-v2-beauty/cpu-replay-r3')
# The three source hashes are from the actual historical V2 formal request.
REPLAY_SHA = {
    'audit.json': 'f5802e83be477126eccde54a1945ffacd900c9f4a9fa30999ac0e2543569a57a',
    'inputs.jsonl': '2e69ea72e47eb978e3d7375c47aa62cd3d3588fa633bdf7b6d1cf7000adc23b2',
    'single-shard.json': '00f4d70d297611a822191e7f3fa4aba45d4d657a2207e6a67824eba4b73cf2cd',
}
SOURCE_V2 = '1182dfd914c20bd3dc11786587eae6879fba6b9d'
CHECKPOINT = IDENTITY_FILE


def read(path):
    return json.loads(path.read_text())


def check_commit(value):
    if len(value) != 40 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Expected a complete lowercase commit SHA')


def read_replay(root, config):
    for name, digest in REPLAY_SHA.items():
        if frozen.sha(root / name) != digest:
            raise ValueError(f'Frozen Beauty V2 replay hash differs for {name}')
    audit, shard = (read(root / n) for n in ('audit.json', 'single-shard.json'))
    with (root / 'inputs.jsonl').open() as stream:
        rows = list(map(json.loads, stream))
    assert audit['status'] == 'PASS_CPU_V2_FULL' and audit['protocol'] == core.PROTOCOL
    assert shard['status'] == 'PASS_SINGLE_SHARD_CPU'
    assert len(rows) == audit['users'] == shard['users'] == config['users'] == 22363
    assert audit['user_axis_sha256'] == config['user_axis_sha256']
    assert audit['tokenizer_vocab_sha256'] == config['vocab_sha256']
    assert audit['manifest_sha256'] == shard['manifest_sha256'] == REPLAY_SHA['inputs.jsonl']
    assert all(row['row_idx'] == i for i, row in enumerate(rows))
    return audit, rows, shard


def preflight_rows(rows):
    # Decide from lengths and intervention identity only, before seeing scores.
    selected = [min(rows, key=lambda row: row['encoded_length']),
                max(rows, key=lambda row: row['encoded_length'])]
    for mode in core.MODES:
        identity = next((row for row in rows if not row['conditions'][mode]['input_changed']), None)
        if identity is not None:
            selected.append(identity)
    return list(dict.fromkeys(row['row_idx'] for row in selected))


def paired_inputs(raw, mask, groups, sid_vocab, replay_row):
    index = replay_row['row_idx']
    canonical, verified = verify_history_groups(raw, mask, groups, sid_vocab, 0, index)
    assert canonical == raw and len(raw) == replay_row['encoded_length']
    assert verified['original_sha256'] == replay_row['original_sha256']
    assert verified['attention_mask_sha256'] == replay_row['attention_mask_sha256']
    assert len(groups) == replay_row['visible_history_length']
    positions = [position for group in verified['groups'] for position in group['positions']]
    history_slots = set(positions)
    history = [i in history_slots for i in range(len(raw))]
    assert sum(history) == 4 * len(groups) and all(mask[i] for i in positions)
    common = dict(row_idx=index, original_sha256=verified['original_sha256'],
                  attention_mask_sha256=verified['attention_mask_sha256'],
                  cpu_manifest_sha256=REPLAY_SHA['inputs.jsonl'], history_group_count=len(groups),
                  history_sid_mask_sha256=core.canonical_sha256(history))
    conditions = {'normal': (raw, {**common, 'mode': 'normal', 'input_changed': False,
                                  'shuffled_sha256': common['original_sha256'], 'permutation_ids': []})}
    for mode in core.MODES:
        ids, evidence = core.transform(raw, mask, groups, sid_vocab, mode, index)
        assert [position for item in evidence['groups'] for position in item['positions']] == positions
        expected = replay_row['conditions'][mode]
        perm_ids = [group['permutation_id'] for group in evidence['groups']]
        assert (evidence['shuffled_sha256'], evidence['input_changed'], perm_ids) == (
            expected['input_sha256'], expected['input_changed'], expected['permutation_ids'])
        conditions[mode] = (ids, {**common, **core.mode_metadata(mode), 'permutation_ids': perm_ids,
                                 'input_changed': evidence['input_changed'],
                                 'shuffled_sha256': evidence['shuffled_sha256']})
    return history, conditions


def checkpoint_binding(adapter, baseline, purpose, identities):
    checkpoint = verify_adapter_identity(adapter, scope=purpose)
    assert checkpoint['model_revision'] == baseline['model_identity']['revision']
    assert checkpoint['data_identity'] == {kind: entry['sha256'] for kind, entry in identities.items()}
    assert checkpoint['adapter_sha256'] != baseline['adapter_identity']['sha256']
    return checkpoint


def generation_settings(model):
    params = dict(num_beams=20, num_return_sequences=20, max_new_tokens=10, early_stopping=True,
                  do_sample=False, output_scores=True, return_dict_in_generate=True)
    effective = deepcopy(model.generation_config)
    assert not effective.update(**params)
    assert effective.use_cache is True
    assert (effective.bos_token_id, effective.eos_token_id, effective.pad_token_id) == (1, 2, 0)
    return params, effective.to_dict()


def load_inference(adapter, report, config, source, author, index):
    """Strictly reload the frozen BF16/FP32 inference path before installing a mode."""
    import torch

    baseline = config['baseline']
    report['verified_base_files'] = verify_base_model(baseline['model'])
    started = time.monotonic()
    model_args = SimpleNamespace(adapter=str(adapter), model=baseline['model'])
    model, tokenizer = source.strict_reload(model_args, report, torch, 0, author, index)
    torch.cuda.synchronize()
    report['load_and_strict_verification_seconds'] = time.monotonic() - started
    assert report['model'] == baseline['model_identity']
    historical = read(Path(baseline['normal_test_dir']) / 'rank0.json')
    actual_checks = report['strict_adapter_reload']['tensor_checks']
    assert {key: value['loaded_dtype'] for key, value in actual_checks.items()} == {
        key: value['loaded_dtype'] for key, value in historical['strict_adapter_reload']['tensor_checks'].items()}
    model.requires_grad_(False)
    base = model.get_base_model()
    assert not getattr(base, 'is_loaded_in_4bit', False) and not getattr(base, 'is_loaded_in_8bit', False)
    assert not model.training and all(p.device == torch.device('cuda', 0) for p in model.parameters())
    assert base.model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16
    return model, tokenizer


def predict(model, base, tokenizer, source, allowed, ids, mask, history, settings, capture):
    import torch
    positions = mask.long().cumsum(-1) - 1
    positions.masked_fill_(mask == 0, 1)
    prompt_length = ids.shape[1]
    boundary, decodes = {}, []

    def before_forward(module, positional, named):
        seen = named['input_ids']
        seen_mask = named['attention_mask']
        position_ids = named['position_ids']
        past = named.get('past_key_values')
        seen_history = named['history_sid_mask']
        assert seen_history.dtype == torch.bool and seen_history.shape[0] == seen.shape[0] == 20
        assert torch.equal(seen_history[:, :prompt_length], history.expand(20, -1))
        if not boundary:
            assert past is None and torch.equal(seen, ids.expand(20, -1))
            assert torch.equal(seen_mask, mask.expand(20, -1))
            assert torch.equal(position_ids, positions.expand(20, -1))
            assert seen_history.shape == seen.shape
            boundary.update(input_sha256=core.canonical_sha256(seen[0].tolist()),
                            attention_mask_sha256=core.canonical_sha256(seen_mask[0].tolist()),
                            position_ids_sha256=core.canonical_sha256(position_ids[0].tolist()),
                            history_sid_mask_sha256=core.canonical_sha256(seen_history[0].tolist()),
                            cache_length=0, beam_expansion=20)
            if capture:
                boundary.update(input_ids=seen[0].tolist(), attention_mask=seen_mask[0].tolist(),
                                position_ids=position_ids[0].tolist(), history_sid_mask=seen_history[0].tolist())
            return
        cache_length = int(past[0][0].shape[-2])
        assert seen.shape[1] == 1 and seen_mask.shape[1] == cache_length + 1
        assert seen_history.shape[1] == prompt_length and cache_length >= prompt_length
        expected_pos = seen_mask.long().sum(dim=-1, keepdim=True) - 1
        assert torch.equal(position_ids, expected_pos)
        if capture:
            decodes.append({'cache_length': cache_length, 'position_ids': position_ids[:, 0].tolist(),
                            'history_length': seen_history.shape[1], 'query_length': seen.shape[1]})

    with ExitStack() as stack:
        if capture:
            embedding = model.get_input_embeddings()
            lookup = embedding(ids)

            def verify_lookup(module, positional, embedding_output):
                if 'lookup_exact' not in boundary:
                    assert boundary and torch.equal(embedding_output, lookup.expand_as(embedding_output))
                    boundary['lookup_exact'] = True

            stack.callback(embedding.register_forward_hook(verify_lookup).remove)
        stack.callback(base.register_forward_pre_hook(before_forward, with_kwargs=True).remove)
        tick = time.monotonic()
        generated = model.generate(input_ids=ids, attention_mask=mask, history_sid_mask=history,
                                   prefix_allowed_tokens_fn=allowed, **settings)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - tick
    assert boundary
    if capture:
        assert len(decodes) == 4 and boundary['lookup_exact']
        boundary['decodes'] = decodes
    input_ids = ids[0].tolist()
    predictions = generated.sequences.cpu().tolist()
    scores = generated.sequences_scores.float().cpu().tolist()
    decoded = tokenizer.batch_decode(generated.sequences, skip_special_tokens=True)
    assert len(predictions) == len(scores) == len(decoded) == 20 and all(map(math.isfinite, scores))
    candidates = []
    for order, (sequence, score, text) in enumerate(zip(predictions, scores, decoded, strict=True)):
        assert sequence[:prompt_length] == input_ids
        suffix = sequence[prompt_length:]
        eos = suffix.index(tokenizer.eos_token_id)
        assert eos == 4
        for position, token in enumerate(suffix[:eos + 1]):
            permitted = allowed(0, torch.tensor(input_ids + suffix[:position], dtype=torch.long))
            assert permitted is not None and token in permitted
        assert all(token in {tokenizer.eos_token_id, tokenizer.pad_token_id} for token in suffix[eos + 1:])
        candidates.append(dict(generation_order=order, generated_token_ids=suffix, eos_position=eos,
                               raw_decoded=text, sequence_score=score, sid=source.normalized_prediction(text)))
    return candidates, sorted(range(20), key=scores.__getitem__, reverse=True), elapsed, boundary


def generate(args):
    import torch
    from lc_rec.beauty_history_attention.attention import install_attention

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    config = read(args.config)
    baseline = config['baseline']
    check_commit(args.run_commit)
    assert 0 < args.max_seconds <= 96 * 3600
    assert torch.cuda.device_count() == 1 and int(os.environ.get('WORLD_SIZE', '1')) == 1
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    versions = {package: importlib.metadata.version(package) for package in baseline['versions']}
    assert versions == baseline['versions']
    report = {'status': 'RUNNING', 'argv': sys.argv, 'RUN_COMMIT': args.run_commit, 'purpose': args.purpose,
              'historical_training_commit': beauty.LETTER_RUN, 'historical_shuffle_run': SOURCE_V2,
              'host': socket.gethostname(), 'pid': os.getpid(), 'interpreter': sys.executable, 'cwd': str(Path.cwd()),
              'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'versions': versions,
              'gpu': {'name': torch.cuda.get_device_name(0), 'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                      'total_memory_bytes': torch.cuda.get_device_properties(0).total_memory},
              'cuda_environment': {key: os.environ.get(key) for key in config['gpu_environment']},
              'world_size': 1, 'rank': 0, 'local_rank': 0}
    run_report = output / 'generation.json'
    frozen.write(run_report, report)
    per_condition_reports = {}

    def guard():
        if time.monotonic() - started > args.max_seconds:
            raise TimeoutError('Beauty generation wall-time limit reached')
        mem = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                   if line.startswith('MemAvailable:'))
        disk = shutil.disk_usage(output).free
        report['min_host_free_bytes'] = min(mem, report.get('min_host_free_bytes', mem))
        report['min_output_free_bytes'] = min(disk, report.get('min_output_free_bytes', disk))
        if mem < 40 * 1024**3 or disk < 20 * 1024**3:
            raise RuntimeError('Frozen evaluator host memory/output disk reserve exhausted')

    try:
        guard()
        source_file = Path(baseline['entry'])
        source = beauty.frozen_module(source_file)
        author = source.author_source(baseline['source-root'])
        for item in [*baseline['author_sources'].values(), next(
                item for item in config['files'] if item['path'] == str(author / 'utils.py'))]:
            assert frozen.sha(Path(item['path'])) == item['sha256']
        report['author_sources'] = baseline['author_sources']
        report['frozen_entry_sha256'] = frozen.sha(source_file)
        index, catalog, users, identities = source.catalog_and_users(baseline['data-root'])
        report['data_identity'] = identities
        audit, manifest_rows, sampler = read_replay(args.replay_dir, config)
        assert semantic_hash([row['user_id'] for row in users]) == config['user_axis_sha256']
        assert all((row['user_id'], row['target']) == (other['user_id'], other['target'])
                   for row, other in zip(users, manifest_rows, strict=True))
        single, ranks, _ = beauty.frozen_assignments(config, users, identities)
        assert [core.canonical_sha256(rank) for rank in ranks] == audit['full_rank_assignment_sha256']
        assert core.canonical_sha256(single) == sampler['assignment_sha256']
        capture = args.purpose == 'preflight'
        assignment = preflight_rows(manifest_rows) if capture else single
        checkpoint = checkpoint_binding(args.adapter, baseline, args.purpose, identities)
        report['history_attention'] = checkpoint
        report['cpu_manifest_sha256'] = audit['manifest_sha256']
        report['checkpoint_manifest_sha256'] = frozen.sha(args.adapter / CHECKPOINT)
        sys.path.insert(0, str(author))
        from data import SeqRecDataset
        from collator import TestCollator
        from utils import set_seed

        set_seed(42)
        model, tokenizer = load_inference(args.adapter, report, config, source, author, index)
        assert {name: report['adapter_files'][name] for name in checkpoint['files']} == checkpoint['files']
        assert report['adapter_files'][CHECKPOINT]['sha256'] == report['checkpoint_manifest_sha256']
        assert semantic_hash(tokenizer.get_vocab()) == audit['tokenizer_vocab_sha256']
        assert tokenizer.get_added_vocab() == audit['token_ids']
        assert tokenizer_identity(tokenizer, audit['token_ids']) == checkpoint['tokenizer_identity']
        base = model.get_base_model()
        report['attention_runtime'] = install_attention(base, checkpoint['mode'])
        model.base_model_prepare_inputs_for_generation = base.prepare_inputs_for_generation
        settings, report['generation'] = generation_settings(model)
        data_args = SimpleNamespace(dataset='Beauty', data_path=baseline['data-root'], max_his_len=20,
                                    his_sep=', ', index_file='.index.json', add_prefix=False)
        dataset = SeqRecDataset(data_args, mode='test', sample_num=-1)
        dataset.set_prompt(0)
        assert list(dataset.remapped_inters) == [row['user_id'] for row in users]
        assert all(dataset[i]['labels'] == row['target'] for i, row in enumerate(users))
        allowed = dataset.get_prefix_allowed_tokens_fn(tokenizer)
        report['allowed_callback_sha256'] = hashlib.sha256(
            inspect.getsource(dataset.get_prefix_allowed_tokens_fn).encode()).hexdigest()
        assert report['allowed_callback_sha256'] == baseline['allowed_callback_sha256']
        collate = TestCollator(data_args, tokenizer)
        evaluator = source.load_evaluator(author)
        token_ids = audit['token_ids']
        sid_vocab = set(token_ids.values())
        report['assignment_sha256'] = core.canonical_sha256(assignment)
        report['assigned_users'] = len(assignment)
        report['expected_conditions'] = list(CONDITIONS)
        if capture:
            report['preflight_assignment'] = assignment
        frozen.write(run_report, report)
        guard()
        with ExitStack() as stack, torch.no_grad():
            streams = {}
            for mode in CONDITIONS:
                target = output / mode
                target.mkdir()
                streams[mode] = stack.enter_context((target / 'predictions.rank0.jsonl').open('x'))
                per_condition_reports[mode] = dict(report, condition=mode, completed_users=0,
                                                    elapsed_generation_seconds=0., identical_to_normal_rows=0)
            for position, i in enumerate(assignment, 1):
                guard()
                user = users[i]
                inputs, targets = collate([dataset[i]])
                seq = dataset.inters[user['user_id']]
                assert targets == [user['target']] == [''.join(index[str(seq[-1])])]
                groups = [[token_ids[symbol] for symbol in index[str(item)]] for item in seq[:-1][-20:]]
                raw, mask = inputs['input_ids'][0].tolist(), inputs['attention_mask'][0].tolist()
                history, conditions = paired_inputs(raw, mask, groups, sid_vocab, manifest_rows[i])
                mask_tensor = inputs['attention_mask'].to('cuda:0')
                history_tensor = torch.tensor([history], dtype=torch.bool, device='cuda:0')
                for mode, (ids, evidence) in conditions.items():
                    input_tensor = torch.tensor([ids], dtype=inputs['input_ids'].dtype, device='cuda:0')
                    candidates, order, elapsed, boundary = predict(
                        model, base, tokenizer, source, allowed, input_tensor, mask_tensor, history_tensor, settings, capture)
                    assert boundary['input_sha256'] == evidence['shuffled_sha256']
                    for key in ('attention_mask_sha256', 'history_sid_mask_sha256'):
                        assert boundary[key] == evidence[key]
                    evidence['actual_prefill'] = boundary
                    if capture:
                        evidence['original_input_ids'] = raw
                    row = dict(user, rank=0, prompt_id=0, input_token_ids=ids, input_nonpadding_tokens=sum(mask),
                               candidates=candidates, sorted_generation_order=order,
                               relevance=[int(candidates[j]['sid'] == targets[0]) for j in order],
                               generation_seconds=elapsed, original_callback_verified_all_tokens=True, shuffle=evidence)
                    source.audit_prediction(row, user, catalog, evaluator)
                    state = per_condition_reports[mode]
                    if mode == 'normal':
                        original = (candidates, order)
                    elif not evidence['input_changed']:
                        assert (candidates, order) == original, f'Identity generation differs: {i}/{mode}'
                        state['identical_to_normal_rows'] += 1
                    streams[mode].write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                    streams[mode].flush()
                    state['completed_users'] += 1
                    state['elapsed_generation_seconds'] += elapsed
                if position % 25 == 0:
                    for mode, state in per_condition_reports.items():
                        os.fsync(streams[mode].fileno())
                        frozen.write(output / mode / 'rank0.json', state)
                    print(json.dumps({'stage': 'five_conditions', 'completed_users': position,
                                      'assigned_users': len(assignment), 'elapsed_seconds': time.monotonic() - started}), flush=True)
            for stream in streams.values():
                os.fsync(stream.fileno())
        for mode, state in per_condition_reports.items():
            assert state['completed_users'] == len(assignment)
            state.update(status='PREFLIGHT_INTERFACE_PASS' if capture else 'PREDICTIONS_COMPLETE')
            if not capture:
                target = output / mode
                frozen.write(target / 'rank0.json', state)
                aggregate = source.aggregate_predictions([target / 'predictions.rank0.jsonl'], users, catalog,
                                                         evaluator, run_commit=args.run_commit)
                aggregate.update(RUN_COMMIT=args.run_commit, history_attention=checkpoint, condition=mode)
                frozen.write(target / 'metrics.json', aggregate)
                state.update(status='PASS')
        report.update(status='PREFLIGHT_INTERFACE_PASS' if capture else 'PASS',
                      conditions={mode: {'completed_users': state['completed_users'],
                                         'generation_seconds': state['elapsed_generation_seconds'],
                                         'identical_to_normal_rows': state['identical_to_normal_rows']}
                                  for mode, state in per_condition_reports.items()})
    except BaseException as error:
        report.update(status='FAIL', error={'type': type(error).__name__, 'message': str(error),
                                             'traceback': traceback.format_exc()})
        raise
    finally:
        resource = dict(finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                        elapsed_wall_seconds=time.monotonic() - started,
                        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        for mode, state in per_condition_reports.items():
            if report['status'] == 'FAIL':
                state.update(status='FAIL', error=report['error'])
            frozen.write(output / mode / 'rank0.json', dict(state, **resource))
        frozen.write(run_report, dict(report, **resource))


def verify_prediction(row, replay, mode, sid_ids):
    evidence = row['shuffle']
    expected = (dict(input_sha256=replay['original_sha256'], input_changed=False, permutation_ids=[])
                if mode == 'normal' else replay['conditions'][mode])
    assert evidence['mode'] == mode
    assert (row['row_idx'], row['user_id'], row['target']) == (replay['row_idx'], replay['user_id'], replay['target'])
    assert evidence['row_idx'] == row['row_idx']
    assert evidence['cpu_manifest_sha256'] == REPLAY_SHA['inputs.jsonl']
    assert (evidence['original_sha256'], evidence['attention_mask_sha256'], evidence['history_group_count']) == (
        replay['original_sha256'], replay['attention_mask_sha256'], replay['visible_history_length'])
    assert (evidence['input_changed'], evidence['shuffled_sha256'], evidence['permutation_ids']) == (
        expected['input_changed'], expected['input_sha256'], expected['permutation_ids'])
    if mode != 'normal':
        assert all(evidence[key] == value for key, value in core.mode_metadata(mode).items())
    nonpad = row['input_nonpadding_tokens']
    ids = row['input_token_ids']
    assert len(ids) == nonpad == replay['encoded_length']
    assert core.canonical_sha256(ids) == expected['input_sha256']
    history = [token in sid_ids for token in ids]  # Only this verified Prompt0 input omits the answer.
    assert sum(history) == 4 * replay['visible_history_length']
    assert core.canonical_sha256(history) == evidence['history_sid_mask_sha256']
    prefill = evidence['actual_prefill']
    assert prefill['input_sha256'] == expected['input_sha256']
    assert all(prefill[key] == evidence[key] for key in ('attention_mask_sha256', 'history_sid_mask_sha256'))
    assert prefill['position_ids_sha256'] == core.canonical_sha256(list(range(len(ids))))
    assert (prefill['cache_length'], prefill['beam_expansion']) == (0, 20)
    assert row['original_callback_verified_all_tokens']
    assert all(c['eos_position'] == 4 and c['generated_token_ids'][4] == 2 for c in row['candidates'])


def metrics(args):
    """Verify saved preflight output or compute the complete historical metrics and paired statistics."""
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    check_commit(args.evaluation_commit)
    config = read(args.config)
    generation = read(args.predictions / 'generation.json')
    baseline = config['baseline']
    purpose = generation['purpose']
    preflight = purpose == 'preflight'
    assert purpose in ('preflight', 'formal') and args.purpose == purpose
    assert generation['status'] == ('PREFLIGHT_INTERFACE_PASS' if preflight else 'PASS')
    assert generation['cpu_manifest_sha256'] == REPLAY_SHA['inputs.jsonl']
    assert generation['allowed_callback_sha256'] == baseline['allowed_callback_sha256']
    assert generation['model'] == baseline['model_identity']
    assert generation['history_attention']['checkpoint_purpose'] == purpose
    assert generation['history_attention']['adapter_sha256'] == generation['adapter_files']['adapter_model.bin']['sha256']
    assert generation['checkpoint_manifest_sha256'] == generation['adapter_files'][CHECKPOINT]['sha256']
    audit, rows, sampler = read_replay(args.replay_dir, config)
    assignment = preflight_rows(rows) if preflight else None
    assignment_sha = core.canonical_sha256(assignment) if preflight else sampler['assignment_sha256']
    count = len(assignment) if preflight else config['users']
    assert generation['assignment_sha256'] == assignment_sha and generation['assigned_users'] == count
    sid_ids = set(audit['token_ids'].values())
    original_config = read(Path(config['normal_evaluation_config']))
    assert original_config['user_axis_sha256'] == config['user_axis_sha256']
    run_commit = generation['RUN_COMMIT']
    check_commit(run_commit)
    original_config['original_run_commit'] = run_commit
    source_path, cce_path = (Path(config[k]) for k in ('frozen_metrics', 'official_cce'))
    source = beauty.frozen_dependency(config, source_path)
    official = beauty.frozen_dependency(config, cce_path)
    assert frozen.sha(cce_path) == original_config['official_cce_sha256']
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    config_path = output / 'evaluation-config.json'
    frozen.write(config_path, original_config)
    ranges = [(count, assignment_sha)]

    def check_row(row, mode):
        verify_prediction(row, rows[row['row_idx']], mode, sid_ids)

    if preflight:
        from collections import Counter

        _, catalog, users, identities = beauty.frozen_module(Path(baseline['entry'])).catalog_and_users(
            baseline['data-root'])
        sizes = Counter({sid: len(items) for sid, items in catalog.items()})
        assert identities == generation['data_identity']
        frozen.write(output / 'unit_tests.json', source.selftest(official))
    else:
        original = source_path.read_text()
        derived = derive_metrics(original)
        (output / 'offline_metrics.derived.py').write_text(derived)
        (output / 'offline_metrics.diff').write_text(''.join(difflib.unified_diff(
            original.splitlines(True), derived.splitlines(True), fromfile=str(source_path), tofile='V2-single-shard')))
        exec(compile(derived, str(source_path), 'exec'), source.__dict__)

    for mode in CONDITIONS:
        predictions = args.predictions / mode
        rank = read(predictions / 'rank0.json')
        assert rank['condition'] == mode and rank['status'] == generation['status']
        for key in ('RUN_COMMIT', 'history_attention', 'adapter_files', 'model', 'data_identity',
                    'cpu_manifest_sha256', 'allowed_callback_sha256', 'checkpoint_manifest_sha256'):
            assert rank[key] == generation[key]
        assert rank['completed_users'] == rank['assigned_users'] == count
        assert rank['assignment_sha256'] == assignment_sha
        if preflight:
            path = predictions / 'predictions.rank0.jsonl'
            with path.open() as stream:
                sample = []
                for i, line in zip(assignment, stream, strict=True):
                    row = json.loads(line)
                    assert row['row_idx'] == i and row['target_item_id'] == users[i]['target_item_id']
                    check_row(row, mode)
                    sids = [row['candidates'][j]['sid'] for j in row['sorted_generation_order']]
                    values, _ = source.ranking(sids, row['target'])
                    corrected = source.cce(sids, row['target'], sizes)
                    ref = official.compute_cce_metrics(sids, row['target'], sizes, [5, 10, 20])
                    assert all(abs(corrected[k] - ref[k]) <= 1e-14 for k in ref)
                    values.update(corrected)
                    assert set(values) == set(source.KEYS) and len(values) == 16
                    sample.append({'row_idx': i, 'target': row['target'], 'metrics': values})
            frozen.write(output / f'{mode}.json', {'scope': 'bounded technical check', 'condition': mode,
                                                     'predictions_sha256': frozen.sha(path), 'sample': sample})
        else:
            root = output / mode
            root.mkdir()
            options = SimpleNamespace(test_dir=predictions, output=root / 'primary',
                                      stage_a_report=predictions / 'metrics.json', config=config_path,
                                      run_commit=run_commit, evaluation_commit=args.evaluation_commit,
                                      official_cce=cce_path, data_root=Path(baseline['data-root']),
                                      assignments=ranges, check_row=lambda row, mode=mode: check_row(row, mode))
            summary = read(source.evaluate(options))
            summary.update(status='PASS_V2_EVALUATION', condition=mode, candidate_count=count * 20, shards=1,
                           history_attention=generation['history_attention'],
                           history_manifest_sha256=generation['checkpoint_manifest_sha256'],
                           evaluator_derived_sha256=hashlib.sha256(derived.encode()).hexdigest())
            frozen.write(root / 'summary.json', summary)
    if not preflight:
        old_argv = sys.argv
        try:
            sys.argv = ['lc_rec.shuffle_v2.statistics', '--output', str(output / 'paired'),
                        '--config', str(args.config), *(str(output / mode) for mode in CONDITIONS)]
            statistics.main()
        finally:
            sys.argv = old_argv
    frozen.write(output / 'summary.json', {'status': 'PASS_PREFLIGHT_METRICS' if preflight else 'PASS_V2_EVALUATION',
                                           'users_per_condition': count, 'condition_count': len(CONDITIONS),
                                           'run_commit': run_commit, 'evaluation_commit': args.evaluation_commit,
                                           'generation_sha256': frozen.sha(args.predictions / 'generation.json'),
                                           'checkpoint': generation['history_attention']})


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('generate', 'metrics'):
        sub = commands.add_parser(name, allow_abbrev=False)
        sub.add_argument('--config', type=Path, default=beauty.CONFIG)
        sub.add_argument('--replay-dir', type=Path, default=REPLAY_DIR)
        sub.add_argument('--output', type=Path, required=True)
        sub.add_argument('--purpose', required=True, choices=('preflight', 'formal'))
        if name == 'generate':
            sub.add_argument('--adapter', type=Path, required=True)
            sub.add_argument('--run-commit', required=True)
            sub.add_argument('--max-seconds', type=int, required=True)
        else:
            sub.add_argument('--predictions', type=Path, required=True)
            sub.add_argument('--evaluation-commit', required=True)
    if sys.flags.optimize:
        parser.error('Python optimization disables the verified replay guards')
    args = parser.parse_args()
    (generate if args.command == 'generate' else metrics)(args)


if __name__ == '__main__':
    main()
