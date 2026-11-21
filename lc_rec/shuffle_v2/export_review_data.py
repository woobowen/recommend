#!/usr/bin/env python3
"""Extract the fixed Instruments A0 evidence for offline review. CPU and NumPy only."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from itertools import permutations
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ACTIVE = {
    'shuffle2': (2, 3), 'shuffle3a': (0, 1, 2),
    'shuffle3b': (1, 2, 3), 'shuffle4': (0, 1, 2, 3),
}
MODES = ('normal', *ACTIVE)
PROTOCOL = 'GRID-LETTER-SHUFFLE-V2'
SEED = 42
CATALOGS = {m: tuple(permutations(range(m))) for m in (2, 3, 4)}
TOPS = (5, 10, 20)
BEAMS = 20
PAD = 65535
KEYS = ([f'hit@{k}' for k in (1, *TOPS)]
        + [f'{key}@{k}' for key in ('ndcg', 'mrr', 'ItemHit', 'ItemNDCG') for k in TOPS])


def require(condition, description):
    if not condition:
        raise ValueError(description)


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def compact(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def binary_identity(path):
    digest = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def copy_bound(source, destination, expected):
    with source.open('rb') as src, destination.open('xb') as dst:
        shutil.copyfileobj(src, dst)
    actual = binary_identity(destination)
    require(actual['sha256'] == expected, f'Asset digest mismatch: {source}')
    return {'source': str(source), **actual}


def draw(size, row, occurrence):
    catalog = CATALOGS[size]
    limit = (1 << 256) // len(catalog) * len(catalog)
    counter = 0
    while True:
        payload = compact([PROTOCOL, SEED, size, row, occurrence, counter])
        value = int.from_bytes(sha256(payload).digest(), 'big')
        if value < limit:
            pick = value % len(catalog)
            return pick, catalog[pick]
        counter += 1


def allocate(n, width):
    return {
        'beam_dictionary_indices': np.zeros((5, n, BEAMS), dtype='<u4'),
        'sequence_scores': np.zeros((5, n, BEAMS), dtype='<f8'),
        'sorted_generation_order': np.zeros((5, n, BEAMS), dtype='u1'),
        'ranked_relevance': np.zeros((5, n, BEAMS), dtype='u1'),
        'input_token_ids': np.full((5, n, width), PAD, dtype='<u2'),
        'input_nonpadding_tokens': np.zeros((5, n), dtype='<u2'),
        'input_changed': np.zeros((4, n), dtype='?'),
        'prefill_hash_bytes': np.zeros((4, n, 3, 32), dtype='u1'),
        'prefill_cache_length_and_beam_expansion': np.zeros((4, n, 2), dtype='u1'),
        'source_rank': np.zeros((5, n), dtype='u1'),
        'source_line_zero_based': np.zeros((5, n), dtype='<u4'),
        'generation_seconds': np.zeros((5, n), dtype='<f8'),
    }


def stream_predictions(root, config, audit, single, cpu_rows, index, inter, arrays, beams, run):
    n = len(inter)
    user_ids = list(inter)
    symbol_ids = audit['token_ids']
    id_symbols = {value: key for key, value in symbol_ids.items()}
    allowed = [{symbol_ids[sid[j]] for sid in index.values()} for j in range(4)]
    all_allowed = set(id_symbols)
    catalog = Counter(''.join(item) for item in index.values())
    dictionary_lookup = {}
    sources, counts = {}, {}
    for mode_i, mode in enumerate(MODES):
        mode_dir = root / f'cpu-eval-{mode}-r0'
        saved = read(mode_dir / 'prediction_manifest.json')
        expected_run = config['baseline']['run-commit'] if mode == 'normal' else run
        require(saved['condition'] == mode and saved['gpu_run_commit'] == expected_run, (mode, 'saved run'))
        require(len(saved['predictions']) == (2 if mode == 'normal' else 1), (mode, 'shards'))
        seen = set()
        sources[mode], diagnostic = [], Counter()
        for source in saved['predictions']:
            rank = source['rank']
            source_path = Path(source['path'])
            test_dir = (Path(config['baseline']['normal_test_dir']) if mode == 'normal'
                        else root / f'gpu-formal-{mode}-r0' / 'test')
            require(source_path == test_dir / f'predictions.rank{rank}.jsonl', (mode, source_path))
            rank_record = read(test_dir / f'rank{rank}.json')
            require(rank_record['status'] == 'PASS' and rank_record['RUN_COMMIT'] == expected_run, (mode, 'rank'))
            require(rank_record['world_size'] == (2 if mode == 'normal' else 1), (mode, 'world_size'))
            reference_assignment = (audit['full_rank_assignment_sha256'][rank] if mode == 'normal' else single['assignment_sha256'])
            assignments = []
            content_hash = sha256()
            byte_count = 0
            with source_path.open('rb') as stream:
                for line_number, raw in enumerate(stream):
                    content_hash.update(raw)
                    byte_count += len(raw)
                    row = json.loads(raw)
                    i = row['row_idx']
                    assert type(i) is int and 0 <= i < n and i not in seen
                    uid = user_ids[i]
                    sequence = inter[uid]
                    target_item = str(sequence[-1])
                    target = ''.join(index[target_item])
                    expected = cpu_rows[i]
                    assert row['user_id'] == uid == expected['user_id']
                    assert (row['target_item_id'], row['target']) == (target_item, target)
                    assert row['target'] == expected['target'] and row['prompt_id'] == 0 and row['rank'] == rank
                    assert row['original_callback_verified_all_tokens'] is True
                    input_ids = row['input_token_ids']
                    input_len = len(input_ids)
                    assert min(input_ids) >= 0 and max(input_ids) < PAD
                    assert input_len == row['input_nonpadding_tokens'] == expected['encoded_length']
                    assert sha256(compact(input_ids)).hexdigest() == (expected['original_sha256'] if mode == 'normal' else expected['conditions'][mode]['input_sha256'])
                    arrays['input_token_ids'][mode_i, i, :input_len] = input_ids
                    arrays['input_nonpadding_tokens'][mode_i, i] = input_len
                    if mode != 'normal':
                        intervention = row['shuffle']
                        desired = expected['conditions'][mode]
                        assert intervention['protocol'] == PROTOCOL and intervention['seed'] == SEED
                        assert intervention['identity_policy'] == 'allow_identity' and intervention['mode'] == mode
                        assert intervention['active_positions'] == list(ACTIVE[mode]) and intervention['row_idx'] == i
                        assert intervention['cpu_manifest_sha256'] == audit['manifest_sha256']
                        assert intervention['history_group_count'] == expected['visible_history_length']
                        assert intervention['input_changed'] == desired['input_changed']
                        assert intervention['original_sha256'] == expected['original_sha256']
                        assert intervention['attention_mask_sha256'] == expected['attention_mask_sha256']
                        assert intervention['shuffled_sha256'] == desired['input_sha256']
                        prefill = intervention['actual_prefill']
                        assert prefill['input_sha256'] == desired['input_sha256']
                        assert prefill['attention_mask_sha256'] == expected['attention_mask_sha256']
                        assert prefill['position_ids_sha256'] == sha256(compact(list(range(input_len)))).hexdigest()
                        assert prefill['cache_length'] == 0 and prefill['beam_expansion'] == BEAMS
                        arrays['input_changed'][mode_i-1, i] = intervention['input_changed']
                        for hash_index, name in enumerate(('input_sha256', 'attention_mask_sha256', 'position_ids_sha256')):
                            arrays['prefill_hash_bytes'][mode_i-1, i, hash_index] = np.frombuffer(bytes.fromhex(prefill[name]), dtype='u1')
                        arrays['prefill_cache_length_and_beam_expansion'][mode_i-1, i] = (prefill['cache_length'], prefill['beam_expansion'])
                    else:
                        assert 'shuffle' not in row
                    scores, sids = [], []
                    assert len(row['candidates']) == BEAMS
                    for j, candidate in enumerate(row['candidates']):
                        assert candidate['generation_order'] == j
                        assert candidate['eos_position'] == 4
                        score = candidate['sequence_score']
                        assert type(score) is float and math.isfinite(score)
                        prompt, separator, response = candidate['raw_decoded'].rpartition('Response:')
                        assert separator and response.strip().replace(' ', '') == candidate['sid']
                        assert prompt
                        tokens = tuple(candidate['generated_token_ids'])
                        assert len(tokens) >= 5 and tokens[4] == 2 and all(t in (0, 2) for t in tokens[5:])
                        assert all(tokens[k] in allowed[k] for k in range(4))
                        sid = candidate['sid']
                        assert ''.join(id_symbols[t] for t in tokens[:4]) == sid
                        key = (sid, response, tokens, candidate['eos_position'])
                        if key not in dictionary_lookup:
                            dictionary_lookup[key] = len(beams)
                            beams.append(key)
                        arrays['beam_dictionary_indices'][mode_i, i, j] = dictionary_lookup[key]
                        scores.append(score)
                        sids.append(sid)
                    order = sorted(range(BEAMS), key=scores.__getitem__, reverse=True)
                    relevance = [int(sids[j] == target) for j in order]
                    assert order == row['sorted_generation_order'] and relevance == row['relevance']
                    arrays['sequence_scores'][mode_i, i] = scores
                    arrays['sorted_generation_order'][mode_i, i] = order
                    arrays['ranked_relevance'][mode_i, i] = relevance
                    arrays['source_rank'][mode_i, i] = rank
                    arrays['source_line_zero_based'][mode_i, i] = line_number
                    assert math.isfinite(row['generation_seconds']) and row['generation_seconds'] >= 0
                    arrays['generation_seconds'][mode_i, i] = row['generation_seconds']
                    diagnostic['candidates'] += len(sids)
                    diagnostic['invalid'] += sum(sid not in catalog for sid in sids)
                    diagnostic['duplicates'] += len(sids) - len(set(sids))
                    diagnostic['colliding'] += sum(catalog[sid] > 1 for sid in sids)
                    seen.add(i)
                    assignments.append(i)
            require(byte_count == source['bytes'] == source_path.stat().st_size and content_hash.hexdigest() == source['sha256'], (mode, 'source identity'))
            require(reference_assignment == rank_record['assignment_sha256'] == sha256(compact(assignments)).hexdigest(), (mode, 'assignment'))
            expected_rows = n//2 if mode == 'normal' else n
            require(len(assignments) == expected_rows == rank_record['completed_users'], (mode, 'rows'))
            sources[mode].append({**source, 'rows': len(assignments), 'assignment_sha256': reference_assignment})
        require(seen == set(range(n)) and diagnostic['candidates'] == n * BEAMS, (mode, 'full coverage'))
        reference_stats = read(mode_dir / 'primary' / 'metrics.json')['sid_statistics']['20']
        for key, value in diagnostic.items():
            require(value == reference_stats[key], (mode, key, value, reference_stats[key]))
        counts[mode] = {'users': len(seen), **diagnostic}
        print(f'{mode}: {len(seen)} rows, {diagnostic["candidates"]} candidates', flush=True)
    return sources, counts


def check_inputs(arrays, cpu_rows, inter, index, audit):
    symbols = audit['token_ids']
    sid_ids = set(symbols.values())
    expected_axis = list(inter)
    counts = {mode: Counter() for mode in ACTIVE}
    for i, cpu_row in enumerate(cpu_rows):
        uid = expected_axis[i]
        history = inter[uid][:-1][-20:]
        assert cpu_row['user_id'] == uid and cpu_row['row_idx'] == i
        assert cpu_row['visible_history_length'] == len(history)
        length = cpu_row['encoded_length']
        baseline = arrays['input_token_ids'][0, i, :length].tolist()
        assert sha256(compact(baseline)).hexdigest() == cpu_row['original_sha256']
        assert sha256(compact([1]*length)).hexdigest() == cpu_row['attention_mask_sha256']
        offsets = [slot for slot, token in enumerate(baseline) if token in sid_ids]
        assert len(offsets) == 4 * len(history)
        groups = []
        for occurrence, item in enumerate(history):
            positions = offsets[occurrence*4:occurrence*4+4]
            group = [symbols[sid] for sid in index[str(item)]]
            assert positions == list(range(positions[0], positions[0]+4))
            assert [baseline[pos] for pos in positions] == group
            groups.append((positions, group))
        for mode_i, (mode, active) in enumerate(ACTIVE.items(), 1):
            restored = baseline.copy()
            picks = []
            for occurrence, (positions, group) in enumerate(groups):
                pick, permutation = draw(len(active), i, occurrence)
                picks.append(pick)
                for k, pos in enumerate(active):
                    restored[positions[pos]] = group[active[permutation[k]]]
            actual = arrays['input_token_ids'][mode_i, i, :length].tolist()
            expected = cpu_row['conditions'][mode]
            assert length == arrays['input_nonpadding_tokens'][mode_i, i] and actual == restored
            assert picks == expected['permutation_ids']
            assert (actual != baseline) == expected['input_changed'] == arrays['input_changed'][mode_i-1, i]
            digest = sha256(compact(actual)).digest()
            assert digest.hex() == expected['input_sha256']
            actual_hashes = arrays['prefill_hash_bytes'][mode_i-1, i]
            assert tuple(h.tobytes() for h in actual_hashes) == (digest, bytes.fromhex(cpu_row['attention_mask_sha256']), sha256(compact(list(range(length)))).digest())
            assert arrays['prefill_cache_length_and_beam_expansion'][mode_i-1, i].tolist() == [0, BEAMS]
            counts[mode]['users'] += 1
            counts[mode]['groups'] += len(groups)
            counts[mode]['identity_groups'] += picks.count(0)
            counts[mode]['unchanged_users'] += int(actual == baseline)
    for mode, seen in counts.items():
        assert seen['groups'] == audit['history_groups']
        assert seen['unchanged_users'] == audit['modes'][mode]['unchanged_users']
        assert seen['identity_groups'] == audit['modes'][mode]['identity_permutations']
    return counts


def score_row(sids, target, sizes):
    hits = [i + 1 for i, sid in enumerate(sids) if sid == target]
    first = hits[0] if hits else 0
    group_size = sizes[target]
    prefix = sum(sizes.get(sid, 1) for sid in sids[:first-1]) if first else 0
    columns = [float(0 < first <= k) for k in (1, *TOPS)]
    columns += [math.fsum(1 / math.log2(i + 1) for i in hits if i <= k) for k in TOPS]
    columns += [1 / first if 0 < first <= k else 0.0 for k in TOPS]
    credits = [min(group_size, max(0, k - prefix)) if first else 0 for k in TOPS]
    columns += [credit / group_size for credit in credits]
    columns += [math.fsum(1 / math.log2(prefix + q + 1) for q in range(1, credit+1)) / group_size for credit in credits]
    return columns


def check_scores(arrays, dictionary, index, inter, matrices):
    sizes = Counter(''.join(sid) for sid in index.values())
    targets = [''.join(index[str(sequence[-1])]) for sequence in inter.values()]
    n = len(targets)
    results = np.empty((len(MODES), n, len(KEYS)), dtype='<f8')
    for row, beam in enumerate(dictionary):
        sid, raw_suffix, _, eos = beam
        require(raw_suffix.strip().replace(' ', '') == sid and eos == 4, ('beam dictionary', row))
    for mode_i, mode in enumerate(MODES):
        for i, target in enumerate(targets):
            scores = arrays['sequence_scores'][mode_i, i].tolist()
            order = arrays['sorted_generation_order'][mode_i, i].tolist()
            assert all(math.isfinite(x) for x in scores)
            assert order == sorted(range(BEAMS), key=scores.__getitem__, reverse=True)
            generation_order = arrays['beam_dictionary_indices'][mode_i, i].tolist()
            sids = [dictionary[generation_order[j]][0] for j in order]
            assert [int(sid == target) for sid in sids] == arrays['ranked_relevance'][mode_i, i].tolist()
            results[mode_i, i] = score_row(sids, target, sizes)
    means = {mode: {key: math.fsum(results[i, :, j])/n for j, key in enumerate(KEYS)} for i, mode in enumerate(MODES)}
    error = None
    if matrices is not None:
        with np.load(matrices, allow_pickle=False) as archive:
            assert tuple(archive['condition'].tolist()) == MODES and archive['keys'].tolist() == KEYS
            assert np.array_equal(archive['row_idx'], np.arange(n))
            assert archive['per_user_values'].shape == results.shape
            error = float(np.max(np.abs(results - archive['per_user_values'])))
        require(error <= 1e-13, ('Frozen paired source matrices diverge', error))
    return {'metric_keys': KEYS, 'rows_per_condition': n, 'per_user_values_compared': results.size if matrices else 0,
            'maximum_absolute_difference_to_frozen_matrices': error, 'metrics': means}


def export(args):
    started = time.monotonic()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    config = read(args.config)
    root = args.root
    cpu_src = root / 'cpu-replay-r1'
    single_src = root / 'cpu-single-shard-r1' / 'audit.json'
    audit = read(cpu_src / 'audit.json')
    single = read(single_src)
    assert audit['status'] == 'PASS_CPU_V2_FULL' and audit['user_axis_sha256'] == config['user_axis_sha256']
    assert single['status'] == 'PASS_SINGLE_SHARD_CPU' and single['manifest_sha256'] == audit['manifest_sha256']
    assets = {x['path']: x['sha256'] for x in config['files']}
    data_root = Path(config['baseline']['data-root']) / 'Instruments'
    asset_copies = {}
    for kind in ('index', 'inter'):
        name = f'Instruments.{kind}.json'
        source = data_root / name
        asset_copies[name] = copy_bound(source, out / name, assets[str(source)])
    asset_copies['cpu-inputs.jsonl'] = copy_bound(cpu_src / 'inputs.jsonl', out / 'cpu-inputs.jsonl', audit['manifest_sha256'])
    for name, source in (('cpu-replay-audit.json', cpu_src/'audit.json'), ('cpu-single-shard-audit.json', single_src)):
        identity = binary_identity(source)
        asset_copies[name] = copy_bound(source, out/name, identity['sha256'])
    index, inter = (read(out / f'Instruments.{part}.json') for part in ('index', 'inter'))
    axis_digest = sha256(json.dumps(list(inter), ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    assert axis_digest == config['user_axis_sha256']
    assert len(index) == config['items'] and len(inter) == config['users']
    with (out / 'cpu-inputs.jsonl').open('rb') as stream:
        cpu_rows = [json.loads(row) for row in stream]
    assert len(cpu_rows) == len(inter)
    width = max(row['encoded_length'] for row in cpu_rows)
    arrays = allocate(len(inter), width)
    dictionary = []
    sources, coverage = stream_predictions(root, config, audit, single, cpu_rows, index, inter, arrays, dictionary, args.run_commit)
    inputs_check = check_inputs(arrays, cpu_rows, inter, index, audit)
    print('All input token arrays reconstructed with original item boundaries and prefill evidence.', flush=True)
    scores_check = check_scores(arrays, dictionary, index, inter, root / 'cpu-summary-r0' / 'arrays.npz')
    write(out / 'beam_dictionary.json', dictionary)
    array_index = {}
    for name, array in arrays.items():
        path = out / f'{name}.npy'
        np.save(path, array, allow_pickle=False)
        reopened = np.load(path, mmap_mode='r', allow_pickle=False)
        require(array.shape == reopened.shape and array.dtype == reopened.dtype and np.array_equal(array.view('u1'), reopened.view('u1')), (name, 'saved bytes'))
        array_index[path.name] = {'shape': list(array.shape), 'dtype': array.dtype.str, **binary_identity(path)}
    receipt = {'status': 'PASS_FULL_CPU_ONLY_REVIEW_EXTRACTION', 'schema': 'a0_shuffle_v2_review_evidence_v1',
               'review_exporter_sha256': binary_identity(Path(__file__))['sha256'], 'utc': datetime.now(timezone.utc).isoformat(),
               'run_commit': args.run_commit, 'normal_commit': config['baseline']['run-commit'], 'research_prediction_sources': sources,
               'fixed_inputs': asset_copies, 'user_axis_sha256': axis_digest, 'conditions_axis': MODES, 'shuffled_conditions_axis': tuple(ACTIVE),
               'beam_dictionary': {'path': 'beam_dictionary.json', 'rows': len(dictionary), **binary_identity(out/'beam_dictionary.json'),
                                   'fields': ('sid', 'raw_decoded_suffix_after_last_Response:', 'generated_token_ids', 'eos_position')},
               'array_files': array_index, 'candidate_coverage': coverage, 'independent_input_check': inputs_check,
               'score_recompute': scores_check, 'elapsed_seconds': time.monotonic() - started, 'numpy': np.__version__,
               'projection': 'Exact saved score (binary64), output token IDs/EOS, normalized SID and exact last Response: suffix, ranking/relevance, source occurrence and input token/prefill records. Repeated raw_decoded prompt prefix is omitted. No candidate, malformed response, duplicate candidate or miss is removed.',
               'input_storage_padding': PAD, 'prefill_hash_fields': ('input_sha256','attention_mask_sha256','position_ids_sha256'),
               'axes_note': 'Except prefill/input_changed, arrays use condition axis then globally aligned row_idx, then named field. Candidate arrays are indexed by original generation_order; sorted_generation_order maps ranked positions back to that axis. source_rank and source_line_zero_based recover original file order. input_token_ids store exact saved IDs before the indicated length, with 65535 as storage-only pad. The CPU/original asset records retain their exact bytes.'}
    write(out / 'review_export_manifest.json', receipt)
    print(json.dumps({k: receipt[k] for k in ('status', 'candidate_coverage', 'independent_input_check', 'score_recompute', 'elapsed_seconds')}, ensure_ascii=False))


def verify(args):
    root = args.directory
    manifest = read(root/'review_export_manifest.json')
    assert manifest['status'] == 'PASS_FULL_CPU_ONLY_REVIEW_EXTRACTION'
    assert tuple(manifest['conditions_axis']) == MODES
    arrays = {}
    for name, entry in manifest['array_files'].items():
        assert binary_identity(root/name) == {key: entry[key] for key in ('bytes','sha256')}
        value = np.load(root/name, mmap_mode='r', allow_pickle=False)
        assert list(value.shape) == entry['shape'] and value.dtype.str == entry['dtype']
        arrays[name.removesuffix('.npy')] = value
    for name, entry in manifest['fixed_inputs'].items():
        assert binary_identity(root/name) == {key: entry[key] for key in ('bytes','sha256')}
    beams = root/'beam_dictionary.json'
    assert binary_identity(beams) == {key: manifest['beam_dictionary'][key] for key in ('bytes','sha256')}
    index, inter = (read(root/f'Instruments.{part}.json') for part in ('index', 'inter'))
    with (root/'cpu-inputs.jsonl').open('rb') as stream:
        cpu_rows = [json.loads(row) for row in stream]
    inputs = check_inputs(arrays, cpu_rows, inter, index, read(root/'cpu-replay-audit.json'))
    scores = check_scores(arrays, read(beams), index, inter, args.matrices)
    assert scores['metrics'] == manifest['score_recompute']['metrics']
    print(json.dumps({'status': 'PASS_OFFLINE_REVIEW_DATA', 'input_check': inputs, 'score_check': scores}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    cli = parser.add_subparsers(dest='action', required=True)
    run = cli.add_parser('export', allow_abbrev=False)
    run.add_argument('--root', type=Path, required=True, help='Original V2 output directory, read-only.')
    run.add_argument('--config', type=Path, required=True, help='a0_instruments.json from the fixed RUN export.')
    run.add_argument('--run-commit', required=True)
    run.add_argument('--output', type=Path, required=True, help='Fresh destination for extracted review materials.')
    readback = cli.add_parser('verify', allow_abbrev=False)
    readback.add_argument('directory', type=Path, help='Directory of extracted review materials.')
    readback.add_argument('--matrices', type=Path, help='Optional original paired arrays.npz to verify all per-user values.')
    args = parser.parse_args()
    require(sys.flags.optimize == 0, 'Run with Python assertions enabled.')
    if args.action == 'export':
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Export runs only with CUDA_VISIBLE_DEVICES explicitly empty.')
        require(len(args.run_commit) == 40, 'Full RUN SHA required.')
        export(args)
    else:
        verify(args)


if __name__ == '__main__':
    main()
