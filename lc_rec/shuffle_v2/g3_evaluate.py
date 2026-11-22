"""CPU-only G3 Beam20 evaluation, frozen Normal replay and paired statistics."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

from lc_rec.a0_beauty import g3_identity
from lc_rec.a0_beauty.grid_sid_preparation.g3_conversion.production_preparation.cpu_closure import (
    predictions as independent,
)
from lc_rec.letter.review.final_audit_20261113 import statistics_verify
from lc_rec.letter.review.final_audit_20261113.external import official_cce
from lc_rec.shuffle_v2 import core, reference
from lc_rec.shuffle_v2.statistics import paired_means, COMPARISONS

LC = Path(__file__).resolve().parents[1]
FROZEN = LC / 'a0_beauty'
CLOSURE = Path(independent.__file__).resolve().parent
ARCHIVE = CLOSURE / 'results/compute/original/normal'
SELECTION = FROZEN / 'config/preflight_g3_users_seed42.json'
NORMAL_TEST = Path('/data/bwwu/lc-rec/g3-formal-20261118/21d495b0be5b/attempts/normal/a0/test')
TRAINING_COMMIT = '21d495b0be5b606853ec7cec68a3eaa1b58acbd7'
SELECTION_SHA = '8d72a6525b53be97fee4dc3283f75df68b6cba87fc08b1dfe6ce265ebcad77d0'
MODEL_SHA = '3315622866cd97833741608ca21d82b72a10184ab8bff4dc688892146d0bf915'
CUTOFFS = independent.CUTOFFS
KEYS = independent.KEYS
JSON = independent.write_json
require = independent.require


def pinned(path, digest):
    return independent.load_verified(path, {'sha256': digest})[0]


def normal_pins():
    path = CLOSURE / 'closure_config.json'
    fixed = json.loads(pinned(path, '020b712e76d201ef2cd501764a6d8d10719cc156da1336b78c41cbd3534cb7ea'))
    require(fixed['run_commit'] == TRAINING_COMMIT and fixed['adapter_identity_sha256'] == MODEL_SHA
            and fixed['asset_sha256'] == g3_identity.ASSETS, 'Frozen G3 source receipt')
    return fixed


def frozen_evaluator():
    path = FROZEN / 'offline_metrics.py'
    frozen = pinned(path, '2068f8d167f19189605f63d42067bd331738d137bb6ff5129dcad910b7eb6e21')
    pinned(Path(g3_identity.__file__), '0068f928c95f8cd2314da752c252faaa91f14eda7e18fe959bc43e21247d4ce8')
    pinned(FROZEN / 'uneven_sampler.py', '1f358c8c887b12acb06518fe89cf206a7be8eae595bb25501699874fc355e5da')
    pinned(Path(official_cce.__file__), independent.OFFICIAL_CCE_SHA256)
    pinned(Path(independent.__file__), '179f57ec097bcf98ff69f8e28e425139afc8b5d57d2ef2dee8bb0bd1db1f19b8')
    # Resolve the native script's sibling imports just as the frozen entry does.
    from types import SimpleNamespace
    namespace = {'__file__': str(path), '__name__': 'g3_frozen_offline_metrics'}
    sys.path.insert(0, str(FROZEN))
    try:
        exec(compile(frozen, str(path), 'exec'), namespace)
    finally:
        sys.path.pop(0)
    return SimpleNamespace(**namespace)


def preflight_selection(path, meta=None):
    manifest = json.loads(pinned(path, SELECTION_SHA))
    require(manifest['scope'] == 'PREFLIGHT' and manifest['asset_sha256'] == g3_identity.ASSETS
            and manifest['world_size'] == 1 and set(manifest['ranks']) == {'0'}, 'G3 frozen preflight identity')
    rows = manifest['ranks']['0']
    axis = [row['row_idx'] for row in rows]
    require(len(axis) == len(set(axis)) == 32 and all(type(i) is int and 0 <= i < independent.USERS for i in axis),
            'Frozen 32-user global row axis')
    if meta is not None:
        for row in rows:
            fields, _ = independent.identity_fields(meta, row['row_idx'])
            require((row['user_id'], row['target_item_id'], row['target']) == (fields[1], fields[3], fields[4]),
                    'Frozen preflight target identity')
    return axis


def metadata(data_root):
    paths = {key: data_root / 'Beauty' / f'Beauty.{key}.json' for key in g3_identity.ASSETS}
    return independent.input_metadata({'asset_sha256': g3_identity.ASSETS}, paths)


def bind_cpu(audit_path, meta):
    audit = json.loads(audit_path.read_bytes())
    path = audit_path.with_name('inputs.jsonl')
    file_bytes = pinned(path, audit['manifest_sha256'])
    require(audit['status'] == 'PASS_G3_CPU_V2_FULL' and audit['users'] == independent.USERS
            and audit['groups'] == 159966 and (audit['items'], audit['sids'], audit['collisions']) == (12101, 12101, 0)
            and audit['token_ids'] == meta['symbol_to_id'] and set(audit['modes']) == set(core.MODES)
            and audit['training_commit'] == TRAINING_COMMIT and audit['preflight_manifest_sha256'] == SELECTION_SHA
            and {k: v['sha256'] for k, v in audit['assets'].items()} == g3_identity.ASSETS
            and all(all(audit['modes'][mode][key] == value for key, value in core.mode_metadata(mode).items())
                    for mode in core.MODES)
            and audit['protocol'] == core.PROTOCOL and audit['identity_policy'] == core.IDENTITY_POLICY,
            'G3 full CPU manifest identity')
    rows = [json.loads(line) for line in file_bytes.splitlines()]
    fields = ('row_idx', 'user_id', 'target_item_id', 'target')
    axis = [{field: row[field] for field in fields} for row in rows]
    require(len(axis) == independent.USERS and core.canonical_sha256(axis) == audit['user_axis_sha256'],
            'G3 CPU manifest global row axis digest')
    for index, uid in enumerate(meta['users']):
        target = str(meta['interactions'][uid][-1])
        require(axis[index] == dict(zip(fields, (index, uid, target, meta['sids'][target]), strict=True)),
                f'Fixed G3 CPU global row: {index}')
    return audit, rows


def verify_v2_input(row, cpu_row, mode, manifest_digest, history, item_tokens):
    index = row['row_idx']
    original, actual, mask = (row[k] for k in ('original_input_token_ids', 'input_token_ids', 'attention_mask'))
    require(len(actual) == len(mask) == len(original) == row['input_nonpadding_tokens'] <= 2048
            and all(value == 1 for value in mask) and row['preflight_extra_left_padding'] == 0,
            'Real batch-one mask/length/padding mismatch')
    require(row['lookup_equivalence_exact'] is True and row['original_callback_verified_all_tokens'] is True,
            'Native embedding lookup or callback evidence missing')
    require((cpu_row['row_idx'], cpu_row['user_id'], cpu_row['target'], cpu_row['target_item_id'])
            == (index, row['user_id'], row['target'], row['target_item_id']), 'GPU / CPU user target binding')
    original_sha, actual_sha, mask_sha = map(core.canonical_sha256, (original, actual, mask))
    oracle = cpu_row['conditions'][mode]
    require((cpu_row['original_sha256'], oracle['input_sha256'], cpu_row['attention_mask_sha256'])
            == (original_sha, actual_sha, mask_sha), 'Actual GPU input differs from full CPU oracle')
    require(row['v2_cpu_binding'] == {
        'manifest_sha256': manifest_digest, 'row_sha256': core.canonical_sha256(cpu_row),
        'original_sha256': original_sha, 'attention_mask_sha256': mask_sha, 'input_sha256': actual_sha,
        'row_idx': index, 'mode': mode,
    }, 'Exact model-to-CPU row binding')
    prefill = row['actual_model_prefill']
    positions = list(range(len(actual)))
    beams = independent.BEAMS
    require((prefill['input_ids'], prefill['attention_mask'], prefill['position_ids'],
             prefill['initial_cache_length'], prefill['beam_expansion']) ==
            (actual, mask, positions, 0, beams), 'Native decoder prefill boundary')
    for key, values in (('input', actual), ('mask', mask), ('position', positions)):
        require(prefill[f'all_beam_{key}_sha256'] == core.canonical_sha256([values] * beams),
                f'Native decoder {key} differs across beams')
    lookup = prefill['embedding_lookup']
    require(lookup['all_beams_exact'] is True and lookup['dtype'] == 'torch.bfloat16'
            and lookup['shape'] == [beams, len(actual), 4096] and lookup['input_sha256'] == actual_sha
            and len(lookup['first_beam_bf16_sha256']) == 64, 'Real BF16 embedding lookup receipt')
    generation = row['effective_generation']
    expected_kwargs = {'max_new_tokens': 10, 'num_beams': beams, 'num_return_sequences': beams,
                       'output_scores': True, 'return_dict_in_generate': True, 'early_stopping': True}
    defaults = {'do_sample': False, 'use_cache': True, 'pad_token_id': 0, 'eos_token_id': 2}
    require(generation['kwargs'] == expected_kwargs and generation['effective'] == {**expected_kwargs, **defaults}
            and all(generation['generation_config'][key] == value for key, value in defaults.items()),
            'Actual generation kwargs / native config')
    shuffle = row['shuffle']
    require(all(shuffle[k] == v for k, v in core.mode_metadata(mode).items()) and
            (shuffle['row_idx'], shuffle['original_input_token_ids'], shuffle['attention_mask']) == (index, original, mask),
            'Stored V2 boundary metadata')
    evidence = shuffle['evidence']
    require((evidence['original_sha256'], evidence['shuffled_sha256'], evidence['attention_mask_sha256']) ==
            (original_sha, actual_sha, mask_sha), 'Stored V2 input fingerprints')
    require(evidence['history_group_count'] == len(history)
            and evidence['input_changed'] == oracle['input_changed'] == (original != actual), 'V2 input change')
    active = core.MODES[mode]
    expected = original.copy()
    locations = [i for i, token in enumerate(original) if token >= 32000]
    require(len(locations) == 4 * len(history), 'SID history group boundaries')
    for occ, (item, group, cpu_group, cpu_permutation) in enumerate(zip(
            history, evidence['groups'], cpu_row['groups'], oracle['permutation_ids'], strict=True)):
        positions = locations[occ * 4:occ * 4 + 4]
        tokens = item_tokens[item]
        require(positions == list(range(positions[0], positions[0] + 4))
                and [original[pos] for pos in positions] == tokens
                and cpu_group == [item, positions, tokens], 'History group tokens or boundaries')
        permutation_id, permutation, rejections = reference.relative(len(active), index, occ)
        shuffled = reference.remap(tokens, active, permutation)
        require(group['positions'] == positions and group['original_ids'] == tokens
                and group['relative_permutation'] == permutation
                and group['permutation_id'] == cpu_permutation == permutation_id
                and group['rejection_counter'] == rejections and group['shuffled_ids'] == shuffled,
                'GPU group differs from independent V2 permutation and CPU oracle')
        for pos, value in zip(positions, shuffled, strict=True):
            expected[pos] = value
    require(expected == actual, 'Actual decoder input differs outside independently verified positions')
    return original == actual


def rank_and_stage(args, fixed, assignment):
    normal = args.command == 'normal'
    archive_rank = args.archive / 'test/rank0.json' if normal else ARCHIVE / 'test/rank0.json'
    pins = {Path(record['path']).name: record for record in fixed['prior_metric_files']['normal']}
    original = json.loads(pinned(archive_rank, pins['rank0.json']['sha256']))
    rank = json.loads((args.test_dir / 'rank0.json').read_bytes())
    if normal:
        require(rank == original, 'Server Normal rank report differs from pinned historical archive')
        stage = json.loads(pinned(args.archive / 'stage_a.json',
                                  '73f2e2bff4ca26c9664f5ab655786ab22a8844ecd65c0810be46c958a64e9164'))
    else:
        stage = json.loads((args.test_dir / 'metrics.json').read_bytes())
        binding = json.loads((args.test_dir / 'v2-binding.rank0.json').read_bytes())
        require(binding['V2_RUN_COMMIT'] == args.run_commit and binding['mode'] == args.mode
                and binding['scope'] == args.scope and binding['adapter_scope'] == 'FORMAL'
                and binding['MODEL_TRAINING_COMMIT'] == TRAINING_COMMIT
                and binding['frozen_source_sha256'] == original['entry_sha256']
                and binding['cpu_manifest_sha256'] == rank['v2_cpu']['manifest_sha256']
                and hashlib.sha256((args.test_dir / 'derived.rank0.py').read_bytes()).hexdigest() == binding['derived_sha256'],
                'V2 immutable source / asset / CPU manifest binding')
        require(rank['MODEL_TRAINING_COMMIT'] == TRAINING_COMMIT and rank['conditions'] == [args.mode]
                and rank['normal_off_same_process_exact'] == [], 'V2 rank training identity or condition set')
        model = rank['v2_model']
        require(all(model[key] is True for key in ('all_parameters_on_logical_cuda_0', 'eval_mode', 'requires_grad_false',
                                                  'parameter_identity_storage_and_versions_unchanged_each_row'))
                and model['actual_parameter_tensor_counts']['LoRA'] == 448, 'Frozen parameters during rank generation')
    scope, mode = args.scope, args.mode
    count = len(assignment)
    expected_status = 'PREFLIGHT_INTERFACE_PASS' if scope == 'PREFLIGHT' else 'PASS'
    require((rank['status'], rank['scope'], rank['condition'], rank['rank'], rank['local_rank'], rank['world_size'],
             rank['completed_users'], rank['assigned_users'], rank['RUN_COMMIT']) ==
            (expected_status, scope, mode, 0, 0, 1, count, count, args.run_commit), 'G3 rank result scope')
    require(rank['assigned_row_indices'] == assignment
            and rank['assignment_sha256'] == core.canonical_sha256(assignment), 'G3 rank sampler order')
    original_fields = ('model_identity_sha256', 'adapter_files', 'tokenizer', 'tokenizer_identity', 'allowed_callback_sha256',
                       'execution_topology', 'execution_profile', 'experiment_id')
    require(all(rank[k] == original[k] for k in original_fields)
            and rank['model_identity_sha256'] == MODEL_SHA, 'G3 model, tokenizer, callback or topology')
    require(all(rank['generation'][k] == v for k, v in original['generation'].items()), 'Frozen G3 generation')
    require({k: v['sha256'] for k, v in rank['data_identity'].items()} == g3_identity.ASSETS, 'G3 rank data identity')
    require((stage['status'], stage['scope'], stage['world_size'], stage['actual_test_users'], stage['RUN_COMMIT'])
            == ('PASS', scope, 1, count, args.run_commit) and len(stage['prediction_files']) == 1
            and stage['all_row_indices_exactly_once'] is True, 'Single-rank stage coverage and source')
    if normal:
        receipt = stage['prediction_files'][0]
        require(all(receipt[k] == fixed['prediction_files']['normal'][k] for k in ('bytes', 'sha256')),
                'Frozen Normal stage prediction identity')
    else:
        require(stage['condition'] == mode and stage['model_identity_sha256'] == MODEL_SHA
                and stage['adapter_model_sha256'] == fixed['adapter_sha256'], 'V2 stage model/condition identity')
    return rank, stage


def evaluate(args):
    started = time.monotonic()
    normal = args.command == 'normal'
    fixed = normal_pins()
    meta = metadata(args.data_root)
    preflight = preflight_selection(args.preflight_users, meta)
    evaluator = frozen_evaluator()
    if normal:
        archived_rank = args.archive / 'test/rank0.json'
    else:
        archived_rank = ARCHIVE / 'test/rank0.json'
    assignment = json.loads(archived_rank.read_bytes())['assigned_row_indices']
    require(len(assignment) == independent.USERS and sorted(assignment) == list(range(independent.USERS))
            and core.canonical_sha256(assignment) == independent.ASSIGNMENT_SHA256, 'Full G3 assignment receipt')
    if args.scope == 'PREFLIGHT':
        assignment = preflight
    rank, stage = rank_and_stage(args, fixed, assignment)
    if not normal:
        audit, cpu_rows = bind_cpu(args.cpu_audit, meta)
        require(rank['v2_cpu']['manifest_sha256'] == audit['manifest_sha256']
                and rank['tokenizer_identity'] == audit['tokenizer_identity']
                and audit['normal_sha256'] == fixed['prediction_files']['normal']['sha256'],
                'V2 rank CPU oracle/tokenizer/Normal binding')
        require(audit['preflight_assignment'] == preflight and audit['assignment_sha256'] == independent.ASSIGNMENT_SHA256,
                'CPU frozen full / preflight assignments')
    out = args.output
    require(out.resolve() != args.test_dir.resolve() and args.test_dir.resolve() not in out.resolve().parents,
            'Evaluation output must be outside source directory')
    out.mkdir(parents=True)
    primary = out / 'primary'
    primary.mkdir()
    JSON(primary / 'unit_tests.json', evaluator.selftest(official_cce))

    checks = {k: 0.0 for k in KEYS}
    records = {}
    first_counts = Counter()
    tally = {k: Counter() for k in CUTOFFS}
    coverage = {k: set() for k in CUTOFFS}
    saved, natural = {}, []
    prediction = args.test_dir / 'predictions.rank0.jsonl'
    before = independent.signature(prediction.stat())
    digest = hashlib.sha256()
    full_receipt = stage['prediction_files'][0]
    require(full_receipt['bytes'] == before[2], 'Prediction size differs from stage receipt')
    with prediction.open('rb', buffering=1024 ** 2) as stream:
        require(independent.signature(os.fstat(stream.fileno())) == before, 'Prediction replaced before read')
        for expected_idx, raw in zip(assignment, stream, strict=True):
            digest.update(raw)
            row = json.loads(raw)
            idx = row['row_idx']
            require(type(idx) is int and idx == expected_idx and idx not in records, 'Prediction sampler order')
            fields, history = independent.identity_fields(meta, idx)
            require((row['user_id'], row['target_item_id'], row['target'], row['condition'], row['scope']) ==
                    (fields[1], fields[3], fields[4], args.mode, args.scope), f'Prediction user/target/scope: {idx}')
            require((row['rank'], row['world_size'], row['prompt_id'], row['model_identity_sha256'],
                     row['execution_topology'], row['execution_profile']) ==
                    (0, 1, 0, MODEL_SHA, rank['execution_topology'], rank['execution_profile']), f'Prediction identity: {idx}')
            if normal:
                independent.verify_input(row, history, meta['item_tokens'], None, None)
                if idx in preflight:
                    saved[idx] = raw
            else:
                require(row['V2_RUN_COMMIT'] == args.run_commit, f'Prediction V2 source: {idx}')
                identical = verify_v2_input(row, cpu_rows[idx], args.mode, audit['manifest_sha256'], history, meta['item_tokens'])
                if identical:
                    natural.append(idx)
                    if args.scope == 'PREFLIGHT':
                        saved[idx] = raw
            sids = independent.reconstruct_candidates(row, meta['symbol_by_id'], meta['allowed_ids'])
            values, first = evaluator.ranking(sids, row['target'])
            values.update(evaluator.cce(sids, row['target'], meta['sizes']))
            second, second_first = independent.score(sids, row['target'], meta['sizes'])
            official = official_cce.compute_cce_metrics(sids, row['target'], meta['sizes'], list(CUTOFFS))
            require(first == second_first, f'First matching rank: {idx}')
            for key, alternate in zip(KEYS, second, strict=True):
                checks[key] = max(checks[key], abs(values[key] - alternate))
                require(checks[key] <= 1e-14, f'Independent {key} differs: {idx}')
            require(all(abs(values[k] - v) <= 1e-14 for k, v in official.items()), f'Official CCE differs: {idx}')
            elapsed = row['generation_seconds']
            require(type(elapsed) in (float, int) and math.isfinite(elapsed) and elapsed >= 0, f'Invalid timing: {idx}')
            records[idx] = {**dict(zip(independent.PAIR_FIELDS[:-1], fields[:-1], strict=True)),
                            'first_target_rank': first, 'generation_seconds': elapsed,
                            independent.PAIR_FIELDS[-1]: fields[-1], **values}
            first_counts[str(first) if first else 'miss'] += 1
            for k in CUTOFFS:
                subset = sids[:k]
                valid = [sid for sid in subset if sid in meta['sizes']]
                invalid, dup = k - len(valid), k - len(set(subset))
                coll = sum(meta['sizes'][sid] > 1 for sid in valid)
                tally[k].update(candidates=k, valid=len(valid), invalid=invalid, duplicates=dup, colliding=coll,
                                users_invalid=int(bool(invalid)), users_duplicate=int(bool(dup)), users_colliding=int(bool(coll)))
                coverage[k].update(valid)
        require(independent.signature(os.fstat(stream.fileno())) == before, 'Prediction changed during read')
    require(independent.signature(prediction.stat()) == before and digest.hexdigest() == full_receipt['sha256'],
            'Prediction bytes differ from stage receipt')

    axis = sorted(records)
    count = len(axis)
    matrix = np.asarray([[records[i][key] for key in KEYS] for i in axis], dtype=np.float64)
    means = {key: math.fsum(matrix[:, j]) / count for j, key in enumerate(KEYS)}
    require(all(abs(means[k] - v) < 1e-13 for k, v in stage['metrics'].items()), 'Original G3 evaluator regression')
    sid_stats = {}
    for k, diag in tally.items():
        size = len(meta['sizes'])
        sid_stats[str(k)] = {
            **diag, 'valid_rate': diag['valid'] / (count * k), 'invalid_rate': diag['invalid'] / (count * k),
            'duplicate_rate': diag['duplicates'] / (count * k), 'collision_candidate_rate': diag['colliding'] / (count * k),
            'users_invalid_rate': diag['users_invalid'] / count, 'users_duplicate_rate': diag['users_duplicate'] / count,
            'users_colliding_rate': diag['users_colliding'] / count, 'unique_valid_sids': len(coverage[k]),
            'coverage_denominator': size, 'unique_valid_sid_coverage': len(coverage[k]) / size,
        }
    stat20 = tally[20]
    anomalies = stage['anomalies']
    anomaly_fields = {'candidates': 'candidate_count', 'invalid': 'invalid_complete_sid_candidates',
                      'valid': 'legal_complete_sid_candidates', 'duplicates': 'duplicate_candidate_occurrences',
                      'colliding': 'colliding_sid_candidate_occurrences', 'users_invalid': 'users_with_invalid_sid',
                      'users_duplicate': 'users_with_duplicate_sid', 'users_colliding': 'users_with_colliding_sid_candidate'}
    require(all(stat20[k] == anomalies[v] for k, v in anomaly_fields.items()), 'Original G3 diagnostic counts')
    histogram = {str(i): first_counts[str(i)] for i in range(1, 21)} | {'miss': first_counts['miss']}
    with (primary / 'per_user.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=independent.CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records[i] for i in axis)
    np.savez_compressed(primary / 'per_user_metrics.npz', values=matrix, keys=KEYS, row_idx=axis,
                        user_axis_sha256=independent.AXIS_SHA256)
    summary = {'status': 'PASS_G3_V2_EVALUATION', 'condition': args.mode, 'scope': args.scope,
               'run_commit': args.run_commit, 'model_training_commit': TRAINING_COMMIT,
               'model_identity_sha256': MODEL_SHA, 'users': count, 'user_axis_sha256': independent.AXIS_SHA256,
               'assignment_sha256': rank['assignment_sha256'], 'candidate_count': stat20['candidates'],
               'prediction_sha256': digest.hexdigest(), 'prediction_bytes': before[2],
               'assets': meta['identities'], 'items': len(meta['sids']), 'unique_sids': len(meta['sizes']),
               'collision_groups': sum(v > 1 for v in meta['sizes'].values()), 'official_cce_rows': count,
               'metrics': means, 'sid_statistics': sid_stats, 'first_target_rank_users': histogram,
               'independent_metric_max_abs_difference': checks,
               'generation_seconds': {'sum': math.fsum(records[i]['generation_seconds'] for i in axis),
                                      'mean': math.fsum(records[i]['generation_seconds'] for i in axis) / count},
               'natural_identity_rows': natural}
    if normal:
        result = {'matrix': matrix, 'per_user_fields': [tuple(str(records[i][k]) for k in independent.PAIR_FIELDS)
                                                        for i in axis],
                  'first_ranks': [records[i]['first_target_rank'] for i in axis],
                  'generation_seconds': [records[i]['generation_seconds'] for i in axis],
                  'sid_statistics': sid_stats, 'histogram': histogram}
        archive_primary = args.archive / 'primary'
        metrics_info = next(r for r in fixed['prior_metric_files']['normal'] if Path(r['path']).name == 'metrics.json')
        original = json.loads(pinned(archive_primary / 'metrics.json', metrics_info['sha256']))
        # Release archives can be moved. Compare native payload identity, using
        # the fixed artifact names under the caller's verified archive root.
        original['per_user_files'] = [{**row, 'path': str(archive_primary / Path(row['path']).name)}
                                      for row in original['per_user_files']]
        compare_frozen_archive(archive_primary, original, result, primary, means)
        require(set(saved) == set(preflight), 'Missing current preflight row in frozen Normal')
        summary['frozen_archive_csv_npz_and_means_exact'] = True
    else:
        summary['cpu_manifest_sha256'] = audit['manifest_sha256']
    if normal or args.scope == 'PREFLIGHT':
        evidence = out / ('normal_preflight.jsonl' if normal else 'natural_identity.jsonl')
        with evidence.open('xb') as stream:
            stream.writelines(saved[i] for i in sorted(saved))
        summary['identity_evidence_sha256'] = hashlib.sha256(evidence.read_bytes()).hexdigest()
    summary['elapsed_seconds'] = time.monotonic() - started
    JSON(out / 'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('status', 'scope', 'condition', 'users', 'candidate_count', 'elapsed_seconds')}),
          flush=True)


def compare_frozen_archive(archive, original, result, primary, means):
    require(means == original['metrics'] and result['sid_statistics'] == original['sid_statistics']
            and result['histogram'] == original['first_target_rank_users'], 'Archived full Normal means/diagnostics')
    receipts = {Path(r['path']).name: r['sha256'] for r in original['per_user_files']}
    require(pinned(archive / 'per_user.csv', receipts['per_user.csv']) == (primary / 'per_user.csv').read_bytes(),
            'Archived full Normal CSV differs')
    pinned(archive / 'per_user_metrics.npz', receipts['per_user_metrics.npz'])
    with np.load(archive / 'per_user_metrics.npz', allow_pickle=False) as old, np.load(
            primary / 'per_user_metrics.npz', allow_pickle=False) as new:
        require(set(old.files) == set(new.files) and all(np.array_equal(old[k], new[k]) for k in old.files),
                'Archived full Normal NPZ differs')


def verified_matrix(root, expected_mode, scope, axis):
    summary = json.loads((root / 'summary.json').read_bytes())
    require((summary['status'], summary['condition'], summary['scope'], summary['model_identity_sha256'],
             summary['user_axis_sha256']) ==
            ('PASS_G3_V2_EVALUATION', expected_mode, scope, MODEL_SHA, independent.AXIS_SHA256), 'Evaluation result identity')
    primary = root / 'primary'
    with np.load(primary / 'per_user_metrics.npz', allow_pickle=False) as archive:
        values = archive['values']
        full_axis = list(range(independent.USERS)) if scope == 'FORMAL' else axis
        require(archive['row_idx'].tolist() == full_axis and tuple(archive['keys']) == KEYS
                and str(archive['user_axis_sha256']) == independent.AXIS_SHA256 and values.dtype == np.float64
                and values.shape == (len(full_axis), len(KEYS)) and np.isfinite(values).all(), 'Metric matrix schema')
    with (primary / 'per_user.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    require([int(row['row_idx']) for row in rows] == full_axis and
            np.array_equal(values, np.asarray([[float(row[k]) for k in KEYS] for row in rows])), 'Evaluation CSV / NPZ')
    identity_fields = [tuple(row[k] for k in independent.PAIR_FIELDS) for row in rows]
    if expected_mode == 'normal':
        require(summary['frozen_archive_csv_npz_and_means_exact'] is True, 'Missing exact pinned Normal match')
        values = values[axis]
        identity_fields = [identity_fields[idx] for idx in axis]
    return values, identity_fields, summary


def compare_identity(normal_raw, condition_root, summary):
    path = condition_root / 'natural_identity.jsonl'
    raw = pinned(path, summary['identity_evidence_sha256'])
    rows = [json.loads(line) for line in raw.splitlines()]
    require([row['row_idx'] for row in rows] == sorted(summary['natural_identity_rows']), 'Natural identity evidence coverage')
    results = []
    fields = ('input_token_ids', 'original_input_token_ids', 'attention_mask', 'sorted_generation_order', 'relevance')
    for row in rows:
        archived = normal_raw[row['row_idx']]
        candidate_fields = ('generation_order', 'generated_token_ids', 'sid', 'eos_position', 'raw_decoded', 'sequence_score')
        differences = {k: [i for i, (left, right) in enumerate(zip(archived['candidates'], row['candidates'], strict=True))
                           if left[k] != right[k]] for k in candidate_fields}
        maximum_delta = max(abs(a['sequence_score'] - b['sequence_score']) for a, b in zip(archived['candidates'],
                                                                                          row['candidates'], strict=True))
        field_matches = {field: archived[field] == row[field] for field in fields}
        require(all(field_matches[k] for k in fields[:3]), 'Natural identity input differs from frozen Normal')
        results.append({'row_idx': row['row_idx'], 'row_field_matches': field_matches,
                        'candidate_indices_differing_by_field': differences, 'maximum_score_absolute_difference': maximum_delta,
                        'all_exact': all(field_matches.values()) and not any(differences.values())})
    return {'identity_users': len(rows), 'all_exact_users': sum(row['all_exact'] for row in results), 'rows': results}


def paired(args):
    started = time.monotonic()
    scope = args.scope
    axis = sorted(preflight_selection(args.preflight_users)) if scope == 'PREFLIGHT' else list(range(independent.USERS))
    normal, identities, normal_summary = verified_matrix(args.normal, 'normal', 'FORMAL', axis)
    if scope == 'PREFLIGHT':
        raw = pinned(args.normal / 'normal_preflight.jsonl', normal_summary['identity_evidence_sha256'])
        normal_raw = {row['row_idx']: row for row in map(json.loads, raw.splitlines())}
        require(sorted(normal_raw) == axis, 'Raw Normal preflight subset coverage')
    matrices = {'normal': normal}
    identities_vs_normal = {}
    run_commits, cpu_hashes = set(), set()
    for mode, root in zip(core.MODES, args.condition_roots, strict=True):
        matrix, condition_identities, summary = verified_matrix(root, mode, scope, axis)
        require(identities == condition_identities, 'Mode-specific user, target or grouping axis differs')
        matrices[mode] = matrix
        run_commits.add(summary['run_commit'])
        cpu_hashes.add(summary['cpu_manifest_sha256'])
        if scope == 'PREFLIGHT':
            identities_vs_normal[mode] = compare_identity(normal_raw, root, summary)
    require(len(run_commits) == len(cpu_hashes) == 1, 'V2 run source or CPU oracle differs between conditions')

    pinned(Path(statistics_verify.__file__), 'c17e47746751146eb4c50245e88616811981031dbd64501eb205e1615774ab96')
    deltas = [matrices[b] - matrices[a] for a, b in COMPARISONS]
    means, sample_hash = paired_means(deltas, len(axis))
    limits = np.percentile(means, [2.5, 97.5], axis=1, method='linear')
    summary = {'status': 'PASS_G3_V2_PAIRED', 'scope': scope, 'users': len(axis), 'row_idx': axis,
               'designation': '32_USER_PREFLIGHT_ENGINEERING_CHECK' if scope == 'PREFLIGHT' else 'FORMAL_PAIRED',
               'user_axis_sha256': independent.AXIS_SHA256, 'v2_run_commit': next(iter(run_commits)),
               'cpu_manifest_sha256': next(iter(cpu_hashes)), 'normal_prediction_sha256': normal_summary['prediction_sha256'],
               'bootstrap': {'rng': 'numpy Generator PCG64', 'seed': 20261113, 'replicates': 5000,
                             'replacement': True, 'shared_draws': True, 'percentiles': [2.5, 97.5],
                             'method': 'linear', 'adjustment': None, 'draw_indices_sha256_little_endian': sample_hash},
               'identity_vs_frozen_normal': identities_vs_normal}
    metric_table = {mode: {key: math.fsum(matrix[:, j]) / len(axis) for j, key in enumerate(KEYS)}
                    for mode, matrix in matrices.items()}
    rows, independent_checks = [], {}
    for i, (a, b) in enumerate(COMPARISONS):
        alternate_ci, alternate_draws = statistics_verify.bootstrap_columns(deltas[i])
        name = f'{b}-{a}'
        checks = {'max_draw_absolute_difference': float(np.max(np.abs(alternate_draws - means[i]))),
                  'max_ci_absolute_difference': float(np.max(np.abs(alternate_ci - limits[:, i])))}
        require(max(checks.values()) <= 1e-13, f'Independent paired statistics differ: {name}')
        independent_checks[name] = checks
        for j, key in enumerate(KEYS):
            left, right = metric_table[a][key], metric_table[b][key]
            difference = right - left
            require(abs(difference - math.fsum(deltas[i][:, j]) / len(axis)) < 1e-13,
                    f'Independent point estimate differs: {name} {key}')
            rows.append({'comparison': name, 'metric': key, 'left': left, 'right': right,
                         'difference': difference, 'relative_change': difference / left if left else None,
                         'ci95_low': float(limits[0, i, j]), 'ci95_high': float(limits[1, i, j])})
    require(len(rows) == 80, 'Five paired comparisons across 16 metrics')
    args.output.mkdir(parents=True)
    with (args.output / 'paired.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(args.output / 'arrays.npz', row_idx=axis, metric=KEYS, conditions=list(matrices),
                        comparison=list(independent_checks), per_user_values=list(matrices.values()),
                        per_user_differences=deltas, bootstrap_difference_means=means)
    summary.update(metrics=metric_table, comparisons=rows, independent_numeric_checks=independent_checks,
                   elapsed_seconds=time.monotonic() - started)
    JSON(args.output / 'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('status', 'scope', 'users', 'elapsed_seconds')}), flush=True)


def main():
    common = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    common.add_argument('--output', type=Path, required=True)
    common.add_argument('--preflight-users', type=Path, default=SELECTION)
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest='command', required=True)
    normal = commands.add_parser('normal', parents=[common], allow_abbrev=False)
    normal.add_argument('--data-root', type=Path, required=True)
    normal.add_argument('--test-dir', type=Path, default=NORMAL_TEST)
    normal.add_argument('--archive', type=Path, default=ARCHIVE)
    normal.set_defaults(scope='FORMAL', mode='normal', run_commit=TRAINING_COMMIT)
    condition = commands.add_parser('condition', parents=[common], allow_abbrev=False)
    for field in ('data-root', 'test-dir', 'cpu-audit'):
        condition.add_argument('--' + field, type=Path, required=True)
    condition.add_argument('--scope', choices=('PREFLIGHT', 'FORMAL'), required=True)
    condition.add_argument('--mode', choices=tuple(core.MODES), required=True)
    condition.add_argument('--run-commit', required=True)
    combination = commands.add_parser('paired', parents=[common], allow_abbrev=False)
    combination.add_argument('--scope', choices=('PREFLIGHT', 'FORMAL'), required=True)
    combination.add_argument('--normal', type=Path, required=True)
    combination.add_argument('--condition-roots', type=Path, nargs=4, required=True,
                             metavar=('S2', 'S3A', 'S3B', 'S4'))
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'The CPU request must hide every CUDA device')
    if args.command == 'paired':
        paired(args)
    else:
        require(len(args.run_commit) == 40 and all(c in '0123456789abcdef' for c in args.run_commit),
                'Full commit identity')
        evaluate(args)


if __name__ == '__main__':
    main()
