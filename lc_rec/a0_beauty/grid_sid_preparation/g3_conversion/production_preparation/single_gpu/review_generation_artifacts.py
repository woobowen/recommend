#!/usr/bin/env python3
"""Non-author, CPU-only recomputation of the frozen final G3 artifacts.

Reads original runner receipts and predictions. Does not load model weights,
start jobs, change production evidence, or authorize formal execution.
"""
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[6]
SCRIPTS = ROOT / 'lc_rec/a0_beauty'
sys.path.insert(0, str(SCRIPTS))
import formal_evaluate

COMMIT = '21d495b0be5b606853ec7cec68a3eaa1b58acbd7'
HERE = Path(__file__).resolve().parent
JOBS = HERE / 'jobs'
G = JOBS / 'g3-lcrec-single-generation-20261118-r3'
E = JOBS / 'g3-lcrec-single-evaluation-20261118-r3'
PROFILE = {'world_size': 1, 'per_device_train_batch_size': 8,
           'gradient_accumulation_steps': 8, 'global_batch': 64}


def load(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_row(row):
    relative = 'lc_rec/' + row['path'].split('/release/lc_rec/', 1)[1]
    original = subprocess.check_output(['git', 'show', COMMIT + ':' + relative], cwd=ROOT)
    assert hashlib.sha256(original).hexdigest() == row['sha256'], relative
    assert (ROOT / relative).read_bytes() == original, 'reviewed local source differs: ' + relative
    if 'bytes' in row:
        assert len(original) == row['bytes']
    return relative


def runner(job):
    request, accepted, status = [load(job / name) for name in
                                 ('request.json', 'accepted.json', 'status.json')]
    assert status['status'] == 'SUCCEEDED' and status['exit_code'] == 0
    assert status['cleanup']['verified'] and not status['worker_may_be_live']
    assert status['cleanup']['last_observation']['verified_empty']
    assert request['capture_version'] == COMMIT
    assert request['execution_profile'] == PROFILE and request['execution_topology'] == 'SINGLE_GPU'
    canonical = json.dumps(request, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
    assert accepted['request_sha256'] == hashlib.sha256(canonical).hexdigest()
    members = load(job / 'collection.json')
    assert len({row['relative'] for row in members}) == len(members)
    for row in members:
        local = job / row['relative']
        assert local.stat().st_size == row['bytes'] and sha(local) == row['sha256'], local
        transfer = load(local.with_name(local.name + '.receipt.json'))
        assert transfer['sha256'] == row['sha256'] and transfer['bytes'] == row['bytes']
    for row in status['artifacts']:
        assert sha(job / row['path']) == row['sha256']
    return {'job_id': request['job_id'], 'members_verified': len(members),
            'status': status['status'], 'exit_code': status['exit_code'],
            'cleanup_verified': True, 'request_sha256': accepted['request_sha256']}


def ranking(sids, target):
    """Independent literal definitions, preserving all repeated target positions."""
    positions = [i + 1 for i, sid in enumerate(sids) if sid == target]
    first = positions[0] if positions else 0
    result = {f'hit@{k}': float(0 < first <= k) for k in (1, 5, 10, 20)}
    for k in (5, 10, 20):
        result[f'ndcg@{k}'] = math.fsum(1 / math.log2(p + 1) for p in positions if p <= k)
        result[f'mrr@{k}'] = 1 / first if 0 < first <= k else 0.
        # Every actual catalog SID is unique; the original CCE uses first hit.
        result[f'ItemHit@{k}'] = float(0 < first <= k)
        result[f'ItemNDCG@{k}'] = 1 / math.log2(first + 1) if 0 < first <= k else 0.
    return result


def main():
    result = {'status': 'PASS', 'scope': 'INDEPENDENT_NONAUTHOR_ARTIFACT_REVIEW_ONLY',
              'run_commit': COMMIT, 'formal_launch_authorized': False,
              'script_sha256': sha(__file__), 'argv': sys.argv,
              'python': sys.executable, 'cwd': str(Path.cwd()),
              'jobs': [runner(G), runner(E)]}
    index_path = ROOT / 'lc_rec/a0_beauty/grid_sid_preparation/g3_conversion/Beauty.g3.index.json'
    assert sha(index_path) == '420fc060caba3a22b0f03900dfe26abc67da8753f3bac2803f7f3c8703a05c22'
    index = load(index_path)
    catalog = Counter(''.join(tokens) for tokens in index.values())
    assert len(index) == len(catalog) == 12101 and max(catalog.values()) == 1
    selected = load(SCRIPTS / 'config/preflight_g3_users_seed42.json')['ranks']['0']
    selected_ids = [row['row_idx'] for row in selected]
    reports, rows, sources = {}, {}, set()
    for condition in ('normal', 'off', 's3', 's4', 'reload'):
        path = G / ('model_evidence/generation_reload_normal' if condition == 'reload'
                    else 'model_evidence/generation/' + condition)
        report = load(path / 'rank0.json')
        predictions = [json.loads(line) for line in (path / 'predictions.rank0.jsonl').read_text().splitlines()]
        reports[condition], rows[condition] = report, predictions
        assert report['RUN_COMMIT'] == COMMIT and report['status'] == 'PREFLIGHT_INTERFACE_PASS'
        assert report['scope'] == 'PREFLIGHT' and report['preflight_only'] is True
        assert report['world_size'] == 1 and report['execution_profile'] == PROFILE
        assert [row['row_idx'] for row in predictions] == selected_ids
        assert report['tokenizer_identity'] == reports['normal']['tokenizer_identity']
        assert report['adapter_files'] == reports['normal']['adapter_files']
        assert report['model_identity_sha256'] == reports['normal']['model_identity_sha256']
        assert report['generation']['num_beams'] == report['generation']['num_return_sequences'] == 20
        assert report['generation']['max_new_tokens'] == 10 and report['generation']['early_stopping']
        assert report['generation']['filter_items'] is False
        strict = report['strict_adapter_reload']
        assert strict['keys'] == len(strict['tensor_checks']) == 450 and strict['lora_keys'] == 448
        assert strict['elements'] == 288571392 and strict['no_missing_keys_tolerated']
        expected = formal_evaluate.expected_adapter_shapes()
        assert set(strict['tensor_checks']) == set(expected)
        assert all(check['shape'] == list(expected[key]) and check['finite_and_exact_after_explicit_dtype_conversion']
                   for key, check in strict['tensor_checks'].items())
        assert sum(math.prod(shape) for shape in expected.values()) == strict['elements']
        for source in report['runtime_sources']:
            sources.add(source_row(source))
        for source in report['author_sources'].values():
            sources.add(source_row(source))
        token_to_id = report['tokenizer']['added_tokens']
        id_to_token = {value: key for key, value in token_to_id.items()}
        assert len(id_to_token) == 786
        allowed = [{i for token, i in token_to_id.items() if token.startswith('<' + position + '_')}
                   for position in 'abcd']
        assert list(map(len, allowed)) == [256, 256, 256, 18]
        for prediction, selected_row in zip(predictions, selected):
            for key in ('row_idx', 'user_id', 'target_item_id', 'target'):
                assert prediction[key] == selected_row[key]
            assert prediction['target'] == ''.join(index[prediction['target_item_id']])
            assert prediction['execution_profile'] == PROFILE and prediction['scope'] == 'PREFLIGHT'
            formal_evaluate.validate_condition_row(prediction, 'normal' if condition == 'reload' else condition)
            prefill = prediction['actual_model_prefill']
            assert prefill['initial_cache_length'] == 0 and prefill['beam_expansion'] == 20
            mask = prediction['attention_mask']
            positions, current = [], -1
            for visible in mask:
                current += visible
                positions.append(current if visible else 1)
            assert prefill['position_ids'] == positions and prediction['lookup_equivalence_exact']
            assert prediction['original_callback_verified_all_tokens']
            assert len(prediction['candidates']) == 20
            for order, candidate in enumerate(prediction['candidates']):
                assert candidate['generation_order'] == order and candidate['eos_position'] == 4
                tokens = candidate['generated_token_ids']
                assert all(tokens[i] in allowed[i] for i in range(4)) and tokens[4] == 2
                assert all(token in (0, 2) for token in tokens[5:])
                assert candidate['sid'] == ''.join(id_to_token[token] for token in tokens[:4])
                assert candidate['sid'] == candidate['raw_decoded'].split('Response:')[-1].strip().replace(' ', '')
                assert math.isfinite(candidate['sequence_score'])
            ordered = sorted(range(20), key=lambda i: prediction['candidates'][i]['sequence_score'], reverse=True)
            assert prediction['sorted_generation_order'] == ordered
            assert prediction['relevance'] == [int(prediction['candidates'][i]['sid'] == prediction['target']) for i in ordered]
    normal = reports['normal']
    adapter_path = G / 'model_evidence/continuous/preflight_adapter/g3_identity.json'
    adapter = load(adapter_path)
    assert sha(adapter_path) == normal['model_identity_sha256']
    assert adapter == normal['adapter_identity']
    assert adapter['scope'] == 'PREFLIGHT' and adapter['run_commit'] == COMMIT
    assert adapter['execution_profile'] == PROFILE and adapter['execution_topology'] == 'SINGLE_GPU'
    assert adapter['config_sha256'] == sha(SCRIPTS / 'config/formal_g3_beauty_seed42.json')
    assert adapter['upstream'] == {
        'g3': 'd41adcfb22a9a47d5e0424c068f4c74f33cc9fdbaf5db08540fbfa0e3d720e18',
        'bridge': '10a7ca66dba43b69232a7c96b48d3287df9314a70a9ec9083ff00acfcf2a8c4f'}
    assert adapter['tokenizer_identity'] == normal['tokenizer_identity']
    assert adapter['base_model_identity'] == normal['model']
    assert adapter['adapter_sha256'] == normal['adapter_files']['adapter_model.bin']['sha256']
    checkpoint = G / 'model_evidence/continuous/verified_checkpoint_manifests/checkpoint-10.complete.json'
    assert adapter['checkpoint_source']['complete_sha256'] == sha(checkpoint)
    assert adapter['checkpoint_source']['global_step'] == 10
    shared = ('host', 'pid', 'argv', 'RUN_COMMIT', 'adapter_files', 'generation', 'allowed_callback_sha256',
              'tokenizer_identity', 'model', 'installed_model_sources', 'runtime_sources')
    assert all(all(report[key] == normal[key] for key in shared) for name, report in reports.items() if name != 'reload')
    cross = ('RUN_COMMIT', 'adapter_files', 'generation', 'allowed_callback_sha256', 'tokenizer_identity',
             'model', 'installed_model_sources', 'runtime_sources', 'author_sources', 'assigned_row_indices', 'execution_profile')
    assert all(reports['reload'][key] == normal[key] for key in cross)
    assert reports['reload']['pid'] != normal['pid'] and reports['reload']['started_utc'] != normal['started_utc']
    max_score_delta = 0.
    changed = {'s3': {'groups': 0, 'users': 0}, 's4': {'groups': 0, 'users': 0}}
    exact_fields = ('row_idx', 'user_id', 'target_item_id', 'target', 'input_token_ids', 'original_input_token_ids',
                    'attention_mask', 'candidates', 'sorted_generation_order', 'relevance', 'actual_model_prefill')
    for i, left in enumerate(rows['normal']):
        assert all(left[key] == rows['off'][i][key] for key in exact_fields)
        reload = rows['reload'][i]
        for key in exact_fields:
            if key != 'candidates':
                assert left[key] == reload[key]
        for a, b in zip(left['candidates'], reload['candidates']):
            assert all(a[key] == b[key] for key in ('generation_order', 'generated_token_ids', 'eos_position', 'sid'))
        a = np.asarray([c['sequence_score'] for c in left['candidates']], dtype=np.float32)
        b = np.asarray([c['sequence_score'] for c in reload['candidates']], dtype=np.float32)
        assert np.array_equal(a, b)  # actual stronger outcome; production criterion remains inherited 1e-5.
        max_score_delta = max(max_score_delta, float(np.max(np.abs(a - b))))
        for name in ('s3', 's4'):
            shuffled = rows[name][i]
            assert all(shuffled[key] == left[key] for key in ('target', 'target_item_id', 'user_id', 'original_input_token_ids', 'attention_mask'))
            touched = set()
            for group in shuffled['shuffle']['evidence']['groups']:
                positions, permutation = group['positions'], group['permutation']
                assert sorted(permutation) == [0, 1, 2, 3] and permutation != [0, 1, 2, 3]
                assert positions == list(range(positions[0], positions[0] + 4))
                original = [left['input_token_ids'][position] for position in positions]
                actual = [shuffled['input_token_ids'][position] for position in positions]
                assert original == group['original_ids']
                assert actual == [original[j] for j in permutation] == group['shuffled_ids']
                assert name != 's3' or permutation[3] == 3
                touched.update(positions)
                changed[name]['groups'] += 1
            assert all(left['input_token_ids'][j] == shuffled['input_token_ids'][j]
                       for j in range(len(left['input_token_ids'])) if j not in touched)
            changed[name]['users'] += int(left['input_token_ids'] != shuffled['input_token_ids'])
    summary = load(E / 'output/evaluation/summary.json')
    assert summary['scope'] == 'PREFLIGHT' and summary['status'] == 'PASS'
    assert summary['formal_experiment_started'] is summary['formal_launch_authorized'] is False
    for source in summary['actual_import_files']:
        sources.add(source_row(source))
    metrics_result = {}
    matrices = {}
    for condition in ('normal', 'off', 's3', 's4'):
        primary = E / 'output/evaluation' / condition / 'primary'
        receipt = load(primary / 'metrics.json')
        assert receipt['users'] == 32 and receipt['scope'] == 'PREFLIGHT' and receipt['preflight_only']
        assert receipt['collision_catalog']['unique_sids'] == 12101 and receipt['collision_catalog']['groups'] == 0
        assert receipt['collision_catalog']['group_size_distribution'] == {}
        with np.load(primary / 'per_user_metrics.npz', allow_pickle=False) as saved:
            keys, matrix = saved['keys'].tolist(), saved['values'].copy()
            indices = saved['row_idx'].tolist()
        assert indices == sorted(selected_ids) and matrix.shape == (32, 16) and matrix.dtype == np.float64
        by_id = {row['row_idx']: row for row in rows[condition]}
        expected = []
        for idx in indices:
            row = by_id[idx]
            sids = [row['candidates'][j]['sid'] for j in row['sorted_generation_order']]
            expected.append([ranking(sids, row['target'])[key] for key in keys])
        expected = np.asarray(expected, dtype=np.float64)
        assert np.array_equal(matrix, expected)
        assert receipt['metrics'] == dict(zip(keys, expected.mean(axis=0).tolist())) == summary['metrics'][condition]
        with (primary / 'per_user.csv').open() as stream:
            csv_rows = list(csv.DictReader(stream))
        assert [int(row['row_idx']) for row in csv_rows] == indices
        assert all(float(row[key]) == matrix[i, j] for i, row in enumerate(csv_rows) for j, key in enumerate(keys))
        diagnostics = {}
        for cutoff in (5, 10, 20):
            lists = [[row['candidates'][j]['sid'] for j in row['sorted_generation_order'][:cutoff]] for row in rows[condition]]
            valid = sum(sid in catalog for sids in lists for sid in sids)
            unique_valid = set(sid for sids in lists for sid in sids if sid in catalog)
            duplicate = sum(len(sids) - len(set(sids)) for sids in lists)
            diag = receipt['sid_statistics'][str(cutoff)]
            assert diag['candidates'] == cutoff * 32 and diag['valid'] == valid
            assert diag['invalid'] == cutoff * 32 - valid and diag['duplicates'] == duplicate
            assert diag['colliding'] == 0 and diag['coverage_denominator'] == 12101
            assert diag['unique_valid_sids'] == len(unique_valid)
            assert diag['unique_valid_sid_coverage'] == len(unique_valid) / 12101
            diagnostics[str(cutoff)] = {'valid': valid, 'invalid': cutoff * 32 - valid, 'duplicates': duplicate}
        draws = np.load(primary / 'bootstrap_draw_means.npy', allow_pickle=False)
        assert draws.shape == (5000, 16) and np.array_equal(draws, np.zeros_like(draws))
        matrices[condition] = matrix
        metrics_result[condition] = {'users': 32, 'metrics': 16, 'independent_matrix_exact': True,
                                     'prefix_diagnostics': diagnostics, 'bootstrap_draws_exact': True}
    pairings = {}
    for condition in ('s3', 's4'):
        paired = E / 'output/evaluation/paired' / condition
        receipt = load(paired / 'paired_comparison.json')
        assert receipt['status'] == 'PASS' and receipt['scope'] == 'PREFLIGHT' and receipt['users'] == 32
        assert receipt['row_indices'] == sorted(selected_ids)
        delta = matrices[condition] - matrices['normal']
        draws = np.load(paired / 'paired_bootstrap_difference_means.npy', allow_pickle=False)
        assert draws.shape == (5000, 16) and np.array_equal(draws, np.zeros_like(draws)) and not delta.any()
        changed_top1 = 0
        for normal_row, shuffled in zip(rows['normal'], rows[condition]):
            def ordered(row):
                return [row['candidates'][j]['sid'] for j in row['sorted_generation_order']]
            changed_top1 += ordered(normal_row)[0] != ordered(shuffled)[0]
        assert receipt['changes']['top1_changed'] == changed_top1
        pairings[condition] = {'top1_changed': changed_top1, 'paired_users': 32,
                              'difference_matrix_and_bootstrap_exact': True}
    synthetic = load(G / 'model_evidence/generation/synthetic_repeat.rank0.json')
    assert synthetic['fixture'] == 'SYNTHETIC_REPEAT_HISTORY_BOUNDARY'
    assert synthetic['not_a_real_repeated_history_user'] and synthetic['target_unchanged']
    assert len(synthetic['fixture_history_item_ids']) > len(set(synthetic['fixture_history_item_ids']))
    assert synthetic['same_process_normal_off_exact']
    for name, prediction in synthetic['conditions'].items():
        formal_evaluate.validate_condition_row(prediction, name)
        assert prediction['lookup_equivalence_exact'] and prediction['actual_model_prefill']['beam_expansion'] == 20
        assert prediction['target'] == synthetic['conditions']['normal']['target']
        assert len(prediction['candidates']) == 20
        if name != 'normal':
            groups = prediction['shuffle']['evidence']['groups']
            assert groups[0]['original_ids'] == groups[1]['original_ids']
    formal_evaluate.require_normal_off_exact(synthetic['conditions']['normal'], synthetic['conditions']['off'])
    result.update(source_files_against_frozen_git=sorted(sources), normal_off_users_exact=32,
                  strict_reload_processes=2, strict_reload_keys=450, strict_reload_elements=288571392,
                  four_condition_pid=normal['pid'], independent_normal_pid=reports['reload']['pid'],
                  independent_normal_exact_scores_users=32, independent_normal_max_score_delta=max_score_delta,
                  shuffle_actual_prefill=changed, metrics=metrics_result, pairings=pairings,
                  synthetic_repeat_excluded_from_metrics=True,
                  synthetic_repeat_four_model_call_conditions_verified=True,
                  limitations='32 fixed users validate interfaces only. No full-population result, webpage second review, or user launch authorization. Reviewer authored training and boundary comparison, which are excluded from independent implementation scope.')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
