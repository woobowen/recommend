"""Check the 32-user Beauty preflight's scoring and pairing contracts on CPU."""
import argparse
from collections import Counter
import csv
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from lc_rec.shuffle_v2 import beauty, core, statistics
from lc_rec.shuffle_v2.beauty_evaluate import validate_input
from lc_rec.shuffle_v2.replay import dump, frozen, semantic_hash

STATUS = 'PASS_BEAUTY_PREFLIGHT_EVALUATION_INTERFACE'
SCOPE = 'Engineering interface verification on the fixed 32-user preflight'
N_USERS = 32
BEAMS = 20


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def load(path):
    return json.loads(path.read_bytes())


def asset(config, original, actual):
    expected = next(entry for entry in config['files'] if entry['path'] == str(original))
    digest = frozen.sha(actual)
    size = actual.stat().st_size
    require(expected['sha256'] == digest and expected.get('bytes', size) == size, f'Asset identity mismatch: {actual}')
    return {'path': str(actual), 'bytes': size, 'sha256': digest}


def paths(config, args):
    original = SimpleNamespace(data=Path(config['baseline']['data-root']) / 'Beauty',
                               normal=Path(config['baseline']['normal_test_dir']),
                               primary=Path(config['normal_primary']),
                               evaluator=Path(config['frozen_metrics']),
                               official=Path(config['official_cce']),
                               selection=Path(config['frozen_preflight']),
                               evaluator_config=Path(config['normal_evaluation_config']))
    if not args.archive_root:
        return original, original
    root = args.archive_root
    sources = root / 'archive/source/lc_rec'
    return original, SimpleNamespace(data=root / 'data',
        normal=root / 'conditions/normal/test', primary=root / 'conditions/normal/primary',
        evaluator=sources / 'a0_beauty/offline_metrics.py',
        official=sources / 'letter/review/final_audit_20261113/external/official_cce.py',
        selection=sources / 'a0_beauty/config/preflight_users_seed42.json',
        evaluator_config=root / 'archive/evaluation_config.json')


def replay_rows(cpu_path, single_path, config, fixed_user_ids):
    cpu = load(cpu_path)
    single = load(single_path)
    require(cpu['status'] == 'PASS_CPU_V2_FULL' and cpu['dataset'] == config['dataset'],
            'Full Beauty replay receipt required')
    require(cpu['protocol'] == core.PROTOCOL and cpu['user_axis_sha256'] == config['user_axis_sha256'],
            'Frozen replay protocol or global axis mismatch')
    require(single['status'] == 'PASS_SINGLE_SHARD_CPU' and single['users'] == cpu['users'] == config['users'],
            'Full single-shard Beauty axis required')
    manifest = cpu_path.with_name('inputs.jsonl')
    digest = frozen.sha(manifest)
    require(digest == cpu['manifest_sha256'] == single['manifest_sha256'], 'Replay manifest identity')
    assignment = single['preflight_assignment']
    require(assignment == fixed_user_ids and len(assignment) == len(set(assignment)) == N_USERS
            and all(type(row) is int and 0 <= row < config['users'] for row in assignment)
            and core.canonical_sha256(assignment) == single['preflight_assignment_sha256'],
            'Fixed ordered preflight assignment')
    wanted = set(assignment)
    rows = {}
    with manifest.open() as stream:
        for i, line in enumerate(stream):
            if i in wanted:
                row = json.loads(line)
                require(row['row_idx'] == i, 'Replay row position')
                rows[i] = row
    require(set(rows) == wanted, 'Replay sample coverage')
    return cpu, single, rows


def normal_subset(config, historical, actual, assignment):
    selected, receipts = {}, []
    wanted = set(assignment)
    for rank in (0, 1):
        filename = f'predictions.rank{rank}.jsonl'
        actual_path = actual / filename
        receipt = asset(config, historical / filename, actual_path)
        receipts.append(receipt)
        with actual_path.open() as stream:
            for line in stream:
                row = json.loads(line)
                row_id = row['row_idx']
                if row_id in wanted:
                    require(row_id not in selected and row['rank'] == rank, 'Normal subset duplicate or rank mismatch')
                    selected[row_id] = row
    require(set(selected) == wanted, 'Normal subset coverage')
    return selected, receipts


def measure_row(row, row_id, target_identity, sizes, metrics, official):
    user, item, sid = target_identity
    require((row['row_idx'], row['user_id'], row['target_item_id'], row['target'], row['prompt_id'])
            == (row_id, user, item, sid, 0), f'Row {row_id} identity or prompt mismatch')
    candidates = row['candidates']
    require(len(candidates) == BEAMS and row['original_callback_verified_all_tokens'] is True,
            f'Row {row_id} requires 20 candidates and verified generation constraints')
    scores, sids = [], []
    for generation_order, candidate in enumerate(candidates):
        score = candidate['sequence_score']
        normalized = metrics.normalize(candidate['raw_decoded'])
        tokens = candidate['generated_token_ids']
        require(candidate['generation_order'] == generation_order and normalized == candidate['sid']
                and type(score) in (int, float) and math.isfinite(score)
                and candidate['eos_position'] == 4 and len(tokens) >= 5 and tokens[4] == 2,
                f'Row {row_id}, beam {generation_order} failed parsing, score or EOS check')
        scores.append(score)
        sids.append(normalized)
    ranking = sorted(range(BEAMS), key=scores.__getitem__, reverse=True)
    sids = [sids[i] for i in ranking]
    require(row['sorted_generation_order'] == ranking and row['relevance'] == [int(s == sid) for s in sids],
            f'Row {row_id} stable ordering or relevance')
    values, _ = metrics.ranking(sids, sid)
    corrected = metrics.cce(sids, sid, sizes)
    expected = official.compute_cce_metrics(sids, sid, sizes, [5, 10, 20])
    require(corrected.keys() == expected.keys() and all(abs(corrected[key] - expected[key]) <= 1e-14 for key in expected),
            f'Row {row_id} differs from official CCE')
    values.update(corrected)
    return [values[key] for key in metrics.KEYS]


def write_csv(path, rows):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pair(matrices, metrics, independent, assignment, sample_axis, output):
    import numpy as np

    users = len(assignment)
    require(list(matrices) == list(statistics.CONDITIONS) and len(metrics) == 16 and all(
        matrix.shape == (users, len(metrics)) and matrix.dtype == np.float64 and np.isfinite(matrix).all()
        for matrix in matrices.values()), 'Condition order, metric axis or finite matrix contract')
    estimates = {mode: {key: math.fsum(values[:, j]) / users for j, key in enumerate(metrics)}
                 for mode, values in matrices.items()}
    deltas = [matrices[right] - matrices[left] for left, right in statistics.COMPARISONS]
    draws, indices_digest = statistics.paired_means(deltas, users)
    bounds = np.percentile(draws, [2.5, 97.5], axis=1, method='linear')
    checks, comparisons = {}, {}
    for index, (left, right) in enumerate(statistics.COMPARISONS):
        name = f'{right}-{left}'
        other_bounds, other_draws = independent.bootstrap_columns(deltas[index])
        checks[name] = {
            'max_draw_difference': float(np.max(np.abs(draws[index] - other_draws))),
            'max_ci_difference': float(np.max(np.abs(bounds[:, index] - other_bounds))),
        }
        require(max(checks[name].values()) <= 1e-13, f'{name} reference/bootstrap mismatch')
        comparisons[name] = []
        for j, metric in enumerate(metrics):
            before, after = estimates[left][metric], estimates[right][metric]
            change = after - before
            comparisons[name].append({'metric': metric, 'left': before, 'right': after,
                'absolute_difference': change, 'relative_decrease_fraction': -change / before if before else None,
                'difference_CI95_low': float(bounds[0, index, j]), 'difference_CI95_high': float(bounds[1, index, j]),
                'relative_change_fraction': change / before if before else None,
                'percentage_point_difference': 100 * change})
    normal_changes = []
    for mode, condition in estimates.items():
        for metric, value in condition.items():
            normal = estimates['normal'][metric]
            delta = value - normal
            normal_changes.append({'condition': mode, 'metric': metric, 'value': value, 'normal': normal,
                'delta': delta, 'percentage_point_difference': 100 * delta,
                'relative_change_fraction': delta / normal if normal else None})
    write_csv(output / 'metrics.csv', [dict(condition=mode, **values) for mode, values in estimates.items()])
    write_csv(output / 'paired.csv', [dict(comparison=name, **record)
              for name, records in comparisons.items() for record in records])
    write_csv(output / 'changes_from_normal.csv', normal_changes)
    np.savez_compressed(output / 'arrays.npz', keys=metrics, condition=statistics.CONDITIONS,
        comparison=list(comparisons), row_idx=np.asarray(assignment), user_axis_sha256=sample_axis,
        per_user_values=np.asarray(list(matrices.values())), per_user_differences=np.asarray(deltas),
        bootstrap_difference_means=draws)
    return {'metrics': estimates, 'comparisons': comparisons, 'changes_from_normal': normal_changes,
            'bootstrap': {'rng': 'NumPy PCG64', 'seed': 20261113, 'replacement': True, 'replicates': 5000,
                'draw_order': 'replicate-major then user-major; int64 integers(0, users)',
                'draw_indices_sha256_little_endian': indices_digest, 'percentile_method': 'linear',
                'bounds': [2.5, 97.5], 'multiple_comparison_adjustment': False, 'numeric_checks': checks}}


def evaluate(args):
    import numpy as np

    require(os.environ.get('CUDA_VISIBLE_DEVICES', '') == '' and not sys.flags.optimize,
            'Run with CUDA_VISIBLE_DEVICES empty and assertions enabled')
    require(len(args.run_commit) == 40 and all(c in '0123456789abcdef' for c in args.run_commit),
            'Preflight run commit must be a full lowercase commit SHA')
    config = load(args.config)
    require(config['dataset'] == 'Beauty' and config['users'] == 22363, 'Frozen Beauty dataset required')
    original, actual = paths(config, args)
    source_receipts = {key: asset(config, getattr(original, key), getattr(actual, key))
                       for key in ('evaluator', 'official', 'selection', 'evaluator_config')}
    metrics = beauty.frozen_module(actual.evaluator, source_receipts['evaluator']['sha256'])
    official = metrics.module(actual.official)
    metrics.verify_bootstrap_config(load(actual.evaluator_config)['bootstrap'])
    cce_check = metrics.selftest(official)
    for kind in ('index', 'inter', 'item'):
        filename = f'Beauty.{kind}.json'
        source_receipts[kind] = asset(config, original.data / filename, actual.data / filename)
        require(source_receipts[kind]['sha256'] == metrics.ASSETS[kind], f'Frozen {kind} identity')
    index = load(actual.data / 'Beauty.index.json')
    interactions = load(actual.data / 'Beauty.inter.json')
    users = list(interactions)
    sid_for = {item: ''.join(tokens) for item, tokens in index.items()}
    sizes = Counter(sid_for.values())
    require(len(users) == config['users'] and semantic_hash(users) == config['user_axis_sha256'],
            'Frozen full user axis')
    require(len(sid_for) == config['items'] and len(sizes) == config['sids']
            and sum(size > 1 for size in sizes.values()) == config['collision_groups'], 'Beauty SID catalog identity')
    selected = load(actual.selection)['ranks']
    require(set(selected) == {'0', '1'}, 'Frozen two-shard preflight selection')
    original_order = [row['row_idx'] for rank in ('0', '1') for row in selected[rank]]
    cpu, single, replay = replay_rows(args.cpu_audit, args.single_shard_audit, config, original_order)
    assignment = single['preflight_assignment']
    target_identities = {}
    for row_id in assignment:
        user = users[row_id]
        item = str(interactions[user][-1])
        target_identities[row_id] = (user, item, sid_for[item])
    sample_axis = semantic_hash([users[row_id] for row_id in assignment])
    normal, normal_receipts = normal_subset(config, original.normal, actual.normal, assignment)
    for row_id in assignment:
        require('shuffle' not in normal[row_id]
                and core.canonical_sha256(normal[row_id]['input_token_ids']) == replay[row_id]['original_sha256'],
                f'Frozen Normal input row {row_id}')
    matrices = {'normal': np.asarray([measure_row(normal[i], i, target_identities[i], sizes, metrics, official)
                                     for i in assignment], dtype=np.float64)}
    matrix_name = 'per_user_metrics.npz'
    primary_path = actual.primary / matrix_name
    source_receipts['normal_primary_matrix'] = asset(config, original.primary / matrix_name, primary_path)
    with np.load(primary_path, allow_pickle=False) as archive:
        require(archive['keys'].tolist() == metrics.KEYS
                and np.array_equal(archive['row_idx'], np.arange(config['users']))
                and str(archive['user_axis_sha256']) == config['user_axis_sha256']
                and np.array_equal(matrices['normal'], archive['values'][assignment]), 'Archived Normal sample values')
    gpu = load(args.preflight_dir / 'summary.json')
    require(gpu['status'] == 'PASS_V2_PREFLIGHT' and gpu['RUN_COMMIT'] == args.run_commit
            and [row['mode']['mode'] for row in gpu['stages']] == list(core.MODES),
            'Fingerprint, status or mode order of GPU preflight')
    reports = {}
    token_ids = set(cpu['token_ids'].values())
    for mode in core.MODES:
        folder = args.preflight_dir / mode
        report_path = folder / 'rank0.json'
        report = load(report_path)
        require(report['status'] == 'PREFLIGHT_INTERFACE_PASS' and report['RUN_COMMIT'] == args.run_commit
                and (report['world_size'], report['rank'], report['local_rank']) == (1, 0, 0)
                and report['completed_users'] == N_USERS
                and report['assignment_sha256'] == single['preflight_assignment_sha256']
                and report['allowed_callback_sha256'] == config['baseline']['allowed_callback_sha256']
                and report['beauty_normal_binding'] == config['baseline']['run-commit']
                and report['shuffle']['mode'] == mode
                and report['shuffle']['cpu_manifest_sha256'] == cpu['manifest_sha256'],
                f'{mode} rank, scope, source or callback')
        raw_path = folder / 'predictions.rank0.jsonl'
        values = []
        with raw_path.open() as stream:
            for row_id, line in zip(assignment, stream, strict=True):
                row = json.loads(line)
                require(row['rank'] == 0, f'{mode} row {row_id} rank')
                values.append(measure_row(row, row_id, target_identities[row_id], sizes, metrics, official))
                validate_input(row, replay[row_id], mode, cpu['manifest_sha256'], token_ids,
                               report['allowed_callback_sha256'])
                prefill = row['shuffle']['actual_prefill']
                require(prefill['lookup_exact'] is True
                        and prefill['input_ids'] == row['input_token_ids']
                        and prefill['attention_mask'] == [1] * len(prefill['input_ids'])
                        and prefill['position_ids'] == list(range(len(prefill['input_ids']))),
                        f'{mode} row {row_id} recorded prefill contents')
        matrices[mode] = np.asarray(values, dtype=np.float64)
        reports[mode] = {'users': len(values), 'candidates': len(values) * BEAMS,
                         'predictions_sha256': frozen.sha(raw_path), 'rank_sha256': frozen.sha(report_path)}

    independent_path = statistics.LC / 'letter/review/final_audit_20261113/statistics_verify.py'
    independent = statistics.module('frozen_statistics_reference', independent_path)
    args.output.mkdir(parents=True, exist_ok=False)
    result = pair(matrices, metrics.KEYS, independent, assignment, sample_axis, args.output)
    result['bootstrap']['nonauthor_reference_source_sha256'] = frozen.sha(independent_path)
    result['bootstrap']['scope'] = SCOPE
    result.update(status=STATUS, scope=SCOPE, RUN_COMMIT=args.run_commit, formal_results=False,
                  users=N_USERS, protocol=core.PROTOCOL, identity_policy=core.IDENTITY_POLICY, shuffle_seed=core.SEED,
                  condition_order=list(statistics.CONDITIONS), comparison_order=list(result['comparisons']),
                  global_user_axis_sha256=config['user_axis_sha256'], user_axis_sha256=sample_axis,
                  preflight_assignment=assignment, preflight_candidates=len(core.MODES) * N_USERS * BEAMS,
                  normal_sample_candidates=N_USERS * BEAMS, normal_sample_matches_archive=True,
                  preflight_summary_sha256=frozen.sha(args.preflight_dir / 'summary.json'),
                  cpu_manifest_sha256=cpu['manifest_sha256'], input_assets=source_receipts,
                  normal_predictions=normal_receipts, preflight=reports, cce_selftest=cce_check)
    dump(args.output / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', type=Path, default=beauty.CONFIG)
    for field in ('preflight-dir', 'cpu-audit', 'single-shard-audit', 'output'):
        parser.add_argument('--' + field, type=Path, required=True)
    parser.add_argument('--run-commit', required=True)
    parser.add_argument('--archive-root', type=Path, help='Use the frozen inputs from an extracted audit package.')
    result = evaluate(parser.parse_args())
    print(json.dumps({key: result[key] for key in ('status', 'users', 'preflight_candidates', 'comparison_order')}))


if __name__ == '__main__':
    main()
