"""Read-only prediction verification and frozen A1 evaluation, including paired bootstrap."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import struct
import time
from types import SimpleNamespace

from lc_rec.a1_qwen3_06b.scripts import offline_metrics as metrics
from lc_rec.shuffle_v2 import a1_replay as replay, core


CONDITIONS = ('normal', *core.MODES)
PAIRS = tuple(('normal', mode) for mode in core.MODES) + (('shuffle3a', 'shuffle3b'),)
CCE_PATH = 'lc_rec/letter/review/final_audit_20261113/external/official_cce.py'
CCE_SHA = '173c257a8711a885e8f6d5abcdb0ad2510ad4fab01c1700631b5ec261f50cd2d'
NORMAL_PRIMARY_SHA = {
    'per_user_metrics.npz': '6141f21905aa4544625c62b51dcc5a1bc275214a839b200b6e741409b9c2c197',
    'metrics.json': 'f92aaff15786fb6e71835ba876e98e4b59c15ad599d6384cb4a1c031c4af02aa',
}


def read_cpu(folder, users):
    load = replay.load
    audit = load(folder / 'audit.json')
    single = load(folder / 'single-shard.json')
    path = folder / 'inputs.jsonl'
    assert audit['status'] == 'PASS_CPU_V2_FULL' and single['status'] == 'PASS_SINGLE_SHARD_CPU'
    assert audit['manifest_sha256'] == single['manifest_sha256'] == replay.frozen.sha(path)
    assert (audit['protocol'], audit['identity_policy'], audit['shuffle_seed']) == (
        core.PROTOCOL, core.IDENTITY_POLICY, core.SEED)
    assert audit['user_axis_sha256'] == replay.AXIS_SHA
    with path.open('rb') as stream:
        rows = [json.loads(line) for line in stream]
    for row, expected in zip(rows, users, strict=True):
        assert all(row[k] == value for k, value in expected.items())
    return audit, single, rows


def validate_candidate(row, token_to_symbol, legal, users, catalog, original, official):
    _, anomalies = replay.frozen.audit_prediction(row, users[row['row_idx']], catalog, original)
    assert row['original_callback_verified_all_tokens'] is True
    for candidate in row['candidates']:
        suffix = candidate['generated_token_ids']
        assert len(suffix) <= 10 and len(suffix) >= 5
        assert candidate['eos_position'] == 4 and suffix[4] == 151643
        assert all(suffix[pos] in allowed for pos, allowed in enumerate(legal))
        assert all(token == 151643 for token in suffix[5:])
        assert ''.join(token_to_symbol[token] for token in suffix[:4]) == candidate['sid']
        score = candidate['sequence_score']
        assert struct.unpack('<f', struct.pack('<f', score))[0] == score
    sids = [row['candidates'][j]['sid'] for j in row['sorted_generation_order']]
    sizes = {sid: len(catalog[sid]) for sid in sids if sid in catalog}
    sizes[row['target']] = len(catalog[row['target']])
    values, _ = metrics.ranking(sids, row['target'])
    corrected = metrics.cce(sids, row['target'], sizes)
    official_values = official.compute_cce_metrics(sids, row['target'], sizes, [5, 10, 20])
    assert all(abs(corrected[k] - official_values[k]) <= 1e-14 for k in official_values)
    values.update(corrected)
    return [values[k] for k in metrics.KEYS], anomalies


def compare_frozen_matrix(config, values, averages):
    import numpy as np
    root = replay.repo_path(config['normal_primary'])
    receipts = {}
    for name, expected in NORMAL_PRIMARY_SHA.items():
        path = root / name
        digest = replay.frozen.sha(path)
        assert digest == expected, (path, digest)
        receipts[name] = {'path': str(path), 'sha256': digest}
    with np.load(root / 'per_user_metrics.npz', allow_pickle=False) as old:
        assert old['keys'].tolist() == metrics.KEYS
        assert np.array_equal(old['row_idx'], np.arange(values.shape[0]))
        assert str(old['user_axis_sha256']) == replay.AXIS_SHA and old['values'].dtype == np.float64
        assert np.array_equal(old['values'], values), np.max(np.abs(old['values'] - values))
    old_averages = replay.load(root / 'metrics.json')
    assert old_averages['run_commit'] == replay.NORMAL_COMMIT
    assert old_averages['metrics'] == averages
    return {'all_metric_values_exact': values.size, 'all_means_exact': len(averages),
            'archive': receipts, 'run_commit': old_averages['run_commit']}


def check_boundary(row, check, audit, mode, preflight):
    idx = row['row_idx']
    ids = row['input_token_ids']
    length = check['encoded_length']
    desired = check['conditions'][mode]
    ev = row['shuffle']
    meta = core.mode_metadata(mode)
    assert {key: ev[key] for key in meta} == meta
    assert ev['row_idx'] == idx and ev['cpu_manifest_sha256'] == audit['manifest_sha256']
    assert ev['input_changed'] is desired['input_changed']
    assert ev['original_sha256'] == check['original_sha256']
    assert ev['attention_mask_sha256'] == check['attention_mask_sha256']
    assert ev['shuffled_sha256'] == core.canonical_sha256(ids) == desired['input_sha256']
    assert len(ids) == row['input_nonpadding_tokens'] == length
    assert ev['history_group_count'] == check['visible_history_length']
    assert ev['eval_mode'] is True and ev['grad_enabled'] is False and ev['sequence_score_dtype'] == 'torch.float32'
    actual = ev['actual_prefill']
    assert actual['input_sha256'] == desired['input_sha256']
    assert actual['attention_mask_sha256'] == check['attention_mask_sha256']
    assert actual['position_ids_sha256'] == core.canonical_sha256(list(range(length)))
    assert actual['cache_length'] == 0 and actual['beam_expansion'] == 20
    assert actual['all_beams_ids_mask_positions_exact'] is True
    callback = ev['callback']
    assert callback['original_validation'] is True and callback['bound_prefix_sha256'] == desired['input_sha256']
    assert len(callback['generation_position_calls']) == 5 and all(n > 0 for n in callback['generation_position_calls'])
    assert callback['allowed_counts'] == [150, 256, 256, 256, 1]
    separator = callback['separator_ids']
    assert separator and ids[-len(separator):] == separator
    if preflight:
        assert actual['input_ids'] == ids and actual['attention_mask'] == [1] * length
        assert actual['position_ids'] == actual['cache_position'] == list(range(length))
        assert all(n == 0 for n in actual['initial_cache_layer_lengths'])
        assert actual['lookup_exact'] is True and actual['lookup_shape'] == [20, length, 1024]
        assert actual['lookup_dtype'] == 'torch.bfloat16'
        assert actual['lookup_sha256'] == actual['expected_lookup_sha256']
        original = ev['original_input_ids']
        assert core.canonical_sha256(original) == check['original_sha256']
        assert original[-len(separator):] == separator
        assert [g['permutation_id'] for g in ev['groups']] == desired['permutation_ids']
        occupied = set()
        for group in ev['groups']:
            positions = group['positions']
            assert len(positions) == 4 and not occupied.intersection(positions)
            occupied.update(positions)
            assert [original[p] for p in positions] == group['original_ids']
            assert [ids[p] for p in positions] == group['shuffled_ids']
        assert all(a == b or pos in occupied for pos, (a, b) in enumerate(zip(original, ids, strict=True)))


def compare_identity(actual, normal):
    assert actual['input_token_ids'] == normal['input_token_ids']
    fields = ('generation_order', 'generated_token_ids', 'eos_position', 'raw_decoded', 'sequence_score', 'sid')
    pairs = list(zip(actual['candidates'], normal['candidates'], strict=True))
    different_fields = [key for key in fields if any(new[key] != old[key] for new, old in pairs)]
    if actual['sorted_generation_order'] != normal['sorted_generation_order']:
        different_fields.append('sorted_generation_order')
    score_errors = [abs(new['sequence_score'] - old['sequence_score']) for new, old in pairs]
    return {'row_idx': actual['row_idx'], 'user_id': actual['user_id'], 'input_exact': True,
            'different_fields': different_fields, 'scores_exact': 'sequence_score' not in different_fields,
            'max_score_absolute_difference': max(score_errors)}


def mode_predictions(test, assignment, mode, preflight, config, audit, single, cpu_rows, run_commit):
    path = test / 'predictions.rank0.jsonl'
    report = replay.load(test / 'rank0.json')
    expected = 'PREFLIGHT_INTERFACE_PASS' if preflight else 'PASS'
    assert report['status'] == expected and report['world_size'] == 1 and report['rank'] == 0
    assert report['preflight_only'] is preflight
    if run_commit is not None:
        assert report['RUN_COMMIT'] == run_commit
    assert report['completed_users'] == report['assigned_users'] == len(assignment)
    assert report['assignment_sha256'] == core.canonical_sha256(assignment)
    assert report['formal_assignment_sha256'] == single['assignment_sha256']
    if preflight:
        assert report['preflight_assignment'] == assignment
    baseline = config['baseline']
    expected_identity = baseline['rank_identity']
    for key in ('adapter_files', 'versions'):
        assert report[key] == expected_identity[key]
    assert core.canonical_sha256(report['strict_adapter_reload']) == baseline['strict_reload_sha256']
    info = report['shuffle']
    assert info['parameter_versions_unchanged'] is True and info['generate_calls'] == len(assignment)
    assert info['parameter_sha256_before'] == info['parameter_sha256_after']
    if preflight:
        assert len(info['parameter_sha256_before']) == 64
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for idx, line in zip(assignment, stream, strict=True):
            digest.update(line)
            row = json.loads(line)
            assert row['row_idx'] == idx and row['rank'] == 0 and row['prompt_id'] == 0
            check_boundary(row, cpu_rows[idx], audit, mode, preflight)
            yield idx, row
    yield {'rank': 0, 'path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest(),
           'rows': len(assignment), 'run_commit': report['RUN_COMMIT'],
           'rank_report_sha256': replay.frozen.sha(test / 'rank0.json')}, None


def read_normal_sample(folder, audit, assignment):
    path = folder / 'normal-sample.jsonl'
    assert replay.frozen.sha(path) == audit['normal_sample_sha256']
    with path.open('rb') as stream:
        rows = [json.loads(line) for line in stream]
    assert [row['row_idx'] for row in rows] == assignment
    return {row['row_idx']: row for row in rows}


def averages_for(values):
    return {key: math.fsum(values[:, j]) / len(values) for j, key in enumerate(metrics.KEYS)}


def evaluate(args, config):
    import numpy as np
    started = time.monotonic()
    condition = 'normal' if args.action == 'normal' else args.mode
    preflight = condition != 'normal' and args.preflight
    base = config['baseline']
    index, catalog, users, _ = replay.frozen.catalog_and_users(replay.repo_path(base['data-root']))
    audit, single, cpu_rows = read_cpu(args.cpu_manifest, users)
    original = replay.frozen.load_evaluator(replay.repo_path(base['source-root']))
    official_path = replay.repo_path(config.get('official_cce', CCE_PATH))
    assert replay.frozen.sha(official_path) == CCE_SHA
    official = metrics.module(official_path)
    fixtures = metrics.selftest(official)
    symbol_ids = audit['token_ids']
    reverse = {value: key for key, value in symbol_ids.items()}
    assert len(reverse) == len(symbol_ids) == 918
    legal = [{symbol_ids[sid[pos]] for sid in index.values()} for pos in range(4)]
    assert [len(entries) for entries in legal] == [150, 256, 256, 256]
    selected = single['preflight_assignment'] if preflight else list(range(len(users)))
    row_positions = {idx: position for position, idx in enumerate(selected)}
    values = np.full((len(selected), len(metrics.KEYS)), np.nan, dtype=np.float64)
    predictions, identity_evidence = [], []
    anomalies = Counter()

    def collect(idx, row):
        position = row_positions[idx]
        assert np.isnan(values[position]).all()
        values[position], counts = validate_candidate(row, reverse, legal, users, catalog, original, official)
        anomalies.update(counts)

    if condition == 'normal':
        folder, original_manifest, reports = replay.frozen_normal(config)
        for rank, order in enumerate(single['normal_rank_assignments']):
            assert reports[rank]['assignment_sha256'] == core.canonical_sha256(order)
            path = folder / f'predictions.rank{rank}.jsonl'
            expected = original_manifest['predictions'][rank]
            for idx, row, _ in replay.normal_rows(path, order, users, rank, expected):
                assert core.canonical_sha256(row['input_token_ids']) == cpu_rows[idx]['original_sha256']
                collect(idx, row)
            predictions.append({**expected, 'path': str(path), 'rows': len(order), 'run_commit': replay.NORMAL_COMMIT})
    else:
        folder = args.test
        order = selected if preflight else single['assignment']
        historic = read_normal_sample(args.cpu_manifest, audit, selected) if preflight else {}
        assert len(order) == len(selected) and set(order) == set(selected)
        for idx, row in mode_predictions(folder, order, condition, preflight, config, audit, single, cpu_rows, args.run_commit):
            if row is None:
                predictions.append(idx)
                continue
            collect(idx, row)
            if preflight and not row['shuffle']['input_changed']:
                identity_evidence.append(compare_identity(row, historic[idx]))
        if preflight:
            wanted = [idx for idx in selected if not cpu_rows[idx]['conditions'][condition]['input_changed']]
            assert [entry['row_idx'] for entry in identity_evidence] == wanted

    assert np.isfinite(values).all() and anomalies['candidate_count'] == len(values) * 20
    averages = averages_for(values)
    frozen_comparison = compare_frozen_matrix(config, values, averages) if condition == 'normal' else None
    ids = [users[idx]['user_id'] for idx in selected]
    check = args.output.resolve()
    assert check != folder.resolve() and folder.resolve() not in check.parents, 'Read-only prediction directory'
    args.output.mkdir(parents=True, exist_ok=False)
    primary = args.output / 'primary'
    primary.mkdir()
    uid_axis = replay.semantic_hash(ids)
    np.savez_compressed(primary / 'per_user_metrics.npz', values=values, keys=np.asarray(metrics.KEYS),
                        row_idx=np.asarray(selected), user_axis_sha256=np.asarray(uid_axis))
    with (primary / 'per_user.csv').open('x', newline='', encoding='utf-8') as stream:
        fields = ('row_idx', 'user_id', 'target_item_id', 'target', *metrics.KEYS)
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({**users[idx], **dict(zip(metrics.KEYS, vector, strict=True))}
                         for idx, vector in zip(selected, values, strict=True))
    replay.dump(primary / 'unit_tests.json', fixtures)
    replay.dump(args.output / 'prediction_manifest.json', predictions)
    identity = {'natural_whole_input_identity_users': len(identity_evidence), 'comparisons': identity_evidence,
                'candidate_exact_users': sum(not user['different_fields'] for user in identity_evidence),
                'scope': 'Historical Normal comparison on natural identity preflight users only'}
    if preflight:
        replay.dump(args.output / 'natural_identity.json', identity)
    summary = {
        'status': 'PASS_V2_EVALUATION', 'condition': condition, 'users': len(values),
        'preflight': preflight, 'scope': 'Engineering preflight' if preflight else 'Full saved prediction evaluation',
        'candidate_anomalies': dict(anomalies), 'metrics': averages, 'matrix_dtype': 'float64',
        'user_axis_sha256': uid_axis, 'global_user_axis_sha256': replay.AXIS_SHA,
        'cpu_manifest_sha256': audit['manifest_sha256'], 'official_cce_sha256': CCE_SHA,
        'frozen_normal_comparison': frozen_comparison, 'natural_identity': identity if preflight else None,
        'elapsed_seconds': time.monotonic() - started,
    }
    replay.dump(args.output / 'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('status', 'condition', 'users', 'elapsed_seconds')}))


def read_metric_matrix(root, condition, selected, users, preflight):
    import numpy as np
    summary = replay.load(root / 'summary.json')
    assert summary['condition'] == condition and summary['status'] == 'PASS_V2_EVALUATION'
    is_subset = condition == 'normal' and preflight
    axis = list(range(len(users))) if is_subset else selected
    expected_hash = replay.semantic_hash([users[idx]['user_id'] for idx in axis])
    path = root / 'primary/per_user_metrics.npz'
    with np.load(path, allow_pickle=False) as raw:
        assert raw['keys'].tolist() == metrics.KEYS and raw['values'].dtype == np.float64
        assert raw['values'].shape == (len(axis), len(metrics.KEYS))
        assert np.array_equal(raw['row_idx'], axis) and str(raw['user_axis_sha256']) == expected_hash
        values = raw['values'][selected] if is_subset else raw['values']
    assert np.isfinite(values).all() and summary['user_axis_sha256'] == expected_hash
    assert summary['preflight'] is (preflight and condition != 'normal')
    assert summary['global_user_axis_sha256'] == replay.AXIS_SHA and summary['official_cce_sha256'] == CCE_SHA
    return values, {'matrix_sha256': replay.frozen.sha(path),
                    'cpu_manifest_sha256': summary['cpu_manifest_sha256'],
                    'evaluation_summary_sha256': replay.frozen.sha(root / 'summary.json')}


def paired(args, config):
    import numpy as np
    started = time.monotonic()
    _, _, users, _ = replay.frozen.catalog_and_users(replay.repo_path(config['baseline']['data-root']))
    selected, _ = replay.frozen_samples(config) if args.preflight else (list(range(len(users))), None)
    axis = replay.semantic_hash([users[idx]['user_id'] for idx in selected])
    matrices, inputs = {}, {}
    for condition, root in zip(CONDITIONS, args.condition_roots, strict=True):
        matrices[condition], inputs[condition] = read_metric_matrix(root, condition, selected, users, args.preflight)
    assert len({row['cpu_manifest_sha256'] for row in inputs.values()}) == 1
    estimates = {condition: averages_for(values) for condition, values in matrices.items()}
    deltas = np.asarray([matrices[right] - matrices[left] for left, right in PAIRS])
    users_n = len(selected)
    rng = np.random.Generator(np.random.PCG64(20261113))
    draws = np.empty((len(PAIRS), 5000, len(metrics.KEYS)), dtype=np.float64)
    args.output.mkdir(parents=True, exist_ok=False)
    sample_path = args.output / 'bootstrap_indices.npy'
    samples = np.lib.format.open_memmap(sample_path, mode='w+', dtype='<i8', shape=(5000, users_n))
    draw_index_digest = hashlib.sha256()
    for start in range(0, 5000, 8):
        count = min(8, 5000 - start)
        sample = rng.integers(0, users_n, size=(count, users_n), dtype=np.int64)
        samples[start:start + count] = sample
        draw_index_digest.update(sample.astype('<i8', copy=False).tobytes(order='C'))
        for j, delta in enumerate(deltas):
            draws[j, start:start + count] = delta[sample].mean(axis=1)
    samples.flush()
    del samples
    limits = np.percentile(draws, [2.5, 97.5], axis=1, method='linear')
    comparisons = {}
    for n, (left, right) in enumerate(PAIRS):
        label = f'{right}-{left}'
        comparisons[label] = []
        for j, metric in enumerate(metrics.KEYS):
            a, b = estimates[left][metric], estimates[right][metric]
            delta = b - a
            comparisons[label].append({
                'metric': metric, 'left': a, 'right': b, 'absolute_difference': delta,
                'percentage_point_difference': 100 * delta, 'relative_change_fraction': delta / a if a else None,
                'difference_CI95_low': float(limits[0, n, j]), 'difference_CI95_high': float(limits[1, n, j]),
            })
    assert sum(map(len, comparisons.values())) == 80
    array_path = args.output / 'arrays.npz'
    np.savez_compressed(array_path, keys=np.asarray(metrics.KEYS), condition=np.asarray(CONDITIONS),
                        comparison=np.asarray(list(comparisons)), row_idx=np.asarray(selected),
                        user_axis_sha256=np.asarray(axis), per_user_values=np.asarray(list(matrices.values())),
                        per_user_differences=deltas, bootstrap_difference_means=draws)
    with (args.output / 'paired.csv').open('x', newline='', encoding='utf-8') as stream:
        fields = ('comparison', *next(iter(comparisons.values()))[0])
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(dict(comparison=name, **point) for name, records in comparisons.items() for point in records)
    summary = {
        'status': 'PASS_V2_PAIRED', 'preflight': args.preflight,
        'scope': 'Engineering only' if args.preflight else 'Fixed checkpoints, users and one Shuffle realization',
        'users': users_n, 'user_axis_sha256': axis, 'global_user_axis_sha256': replay.AXIS_SHA,
        'protocol': core.PROTOCOL, 'identity_policy': core.IDENTITY_POLICY, 'shuffle_seed': core.SEED,
        'metrics': estimates, 'comparisons': comparisons, 'inputs': inputs,
        'difference_units': {'absolute_difference': 'right minus left, on the original metric scale',
                             'percentage_point_difference': '100 times the absolute difference',
                             'relative_change_fraction': 'absolute difference divided by left, undefined at zero'},
        'bootstrap': {'rng': 'NumPy PCG64', 'seed': 20261113, 'replicates': 5000, 'replacement': True,
                      'percentile_method': 'linear', 'bounds': [2.5, 97.5], 'multiple_comparison_adjustment': False,
                      'draw_indices_sha256_little_endian': draw_index_digest.hexdigest(),
                      'draw_indices_file_sha256': replay.frozen.sha(sample_path),
                      'draw_order': 'Shared indices, replicate-major then user-major, int64 integers(0, users)'},
        'arrays_file_sha256': replay.frozen.sha(array_path), 'numpy_version': np.__version__,
        'elapsed_seconds': time.monotonic() - started,
    }
    replay.dump(args.output / 'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('status', 'preflight', 'users', 'elapsed_seconds')}))


def assess(args, config):
    started = time.monotonic()
    generation_report = args.preflight_dir / 'summary.json'
    generated = replay.load(generation_report)
    assert generated['status'] == 'PASS' and not generated['identity_discrepancies']
    assert list(generated['stages']) == list(core.MODES)
    run = generated['run_commit']
    assert len(run) == 40 and all(c in '0123456789abcdef' for c in run)
    for stage in generated['stages'].values():
        assert stage['users'] == 32 and stage['exit_code'] == 0
    out = args.output.resolve()
    for source in (args.normal, args.preflight_dir, args.cpu_manifest):
        source = source.resolve()
        assert out != source and source not in out.parents
    out.mkdir(parents=True, exist_ok=False)
    evaluation_roots = []
    for mode in core.MODES:
        root = out / mode
        request = SimpleNamespace(action='mode', mode=mode, preflight=True, test=args.preflight_dir / mode,
                                  output=root, cpu_manifest=args.cpu_manifest, run_commit=run)
        evaluate(request, config)
        evaluation_roots.append(root)
    paired(SimpleNamespace(output=out / 'paired', preflight=True, condition_roots=[args.normal, *evaluation_roots]),
           config)
    stages = {mode: replay.load((out / mode / 'summary.json')) for mode in core.MODES}
    assert sum(stage['users'] for stage in stages.values()) == 128
    assert sum(stage['candidate_anomalies']['candidate_count'] for stage in stages.values()) == 2560
    identity = {mode: stage['natural_identity'] for mode, stage in stages.items()}
    assert all(v['natural_whole_input_identity_users'] == v['candidate_exact_users'] for v in identity.values())
    summary = {'status': 'PASS_A1_V2_PREFLIGHT_ASSESSMENT', 'run_commit': run, 'users_per_mode': 32,
               'mode_user_records': 128, 'candidates': 2560, 'paired_records': 80,
               'scope': 'Engineering verification on the fixed preflight users',
               'generation_report_sha256': replay.frozen.sha(generation_report), 'natural_identity': identity,
               'condition_summaries': {mode: replay.frozen.sha(out / mode / 'summary.json') for mode in core.MODES},
               'paired_summary_sha256': replay.frozen.sha(out / 'paired/summary.json'),
               'elapsed_seconds': time.monotonic() - started}
    replay.dump(out / 'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('status', 'users_per_mode', 'candidates', 'paired_records', 'elapsed_seconds')}))


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    for action in ('normal', 'mode', 'paired'):
        sub = commands.add_parser(action, allow_abbrev=False)
        sub.add_argument('--output', type=Path, required=True)
        sub.add_argument('--config', type=Path, required=True)
        if action != 'paired':
            sub.add_argument('--cpu-manifest', type=Path, required=True)
        if action != 'normal':
            sub.add_argument('--preflight', action='store_true')
        if action == 'mode':
            sub.add_argument('--mode', choices=core.MODES, required=True)
            sub.add_argument('--test', type=Path, required=True)
            sub.add_argument('--run-commit')
        if action == 'paired':
            sub.add_argument('condition_roots', type=Path, nargs=len(CONDITIONS),
                             metavar='EVALUATION_DIR', help='Normal, S2, S3a, S3b, S4')
    assessment = commands.add_parser('assess', allow_abbrev=False)
    for key in ('config', 'cpu-manifest', 'output', 'normal', 'preflight-dir'):
        assessment.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    replay.cpu_only()
    config = replay.load(args.config)
    if args.action == 'paired':
        paired(args, config)
    elif args.action == 'assess':
        assess(args, config)
    else:
        evaluate(args, config)


if __name__ == '__main__':
    main()
