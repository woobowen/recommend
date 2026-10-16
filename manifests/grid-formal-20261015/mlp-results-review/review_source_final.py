import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

import yaml


BASE = Path('/home/addaswsw/recommend-support/grid-formal-20261015')
COLLECTED = BASE / 'collected-9a64fd31218d0c68'
DEST = BASE / 'mlp-results-review'
REPO = Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def record(p):
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)}


def jread(p):
    return json.loads(p.read_text())


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


manifest = jread(COLLECTED / 'COLLECTION_MANIFEST.json')
manifest_map = {x['path']: x for x in manifest['files']}
assert manifest['checkpoints_excluded'] and manifest['source_data_excluded']
startup_path = BASE / 'mlp-admission/formal-startup-review.json'
startup = jread(startup_path)
assert startup['status'] == 'FORMAL_STARTUP_REVIEW_PASS'
entities_path = DEST / 'checkpoint-entity-evidence.json'
entities = jread(entities_path)

for domain, expected_steps, expected_validations, expected_best, target_count, accumulation, cap, parameters in [
    ('toys', 5400, 54, 4400, 19412, 48, 512, 13291136),
    ('sports', 5900, 59, 4900, 35598, 16, 128, 13292160),
]:
    job_id = f'gf-20261015-{domain}-mlp-formal-v1'
    job = COLLECTED / job_id
    output = job / 'output'
    prior = startup['domains'][domain]
    checked_files = []
    for relative in ['request.json', 'accepted.json', 'status.json', 'cleanup.json', 'work.log'] + [
        'output/' + n for n in ['RESULT.json', 'training_result.json', 'domain_protocol.json', 'invocation.json',
            'used_config.json', 'effective_config.yaml', 'approved_formal_effective_config.yaml',
            'actual_trainer.yaml', 'actual_train_loader.json', 'structure.json', 'optimizer_updates.jsonl',
            'validations.jsonl', 'csv/version_0/metrics.csv', 'best_checkpoint_load.json',
            'latest_recovery.json', 'recovery/training-final-1/recovery.json',
            'recovery/training-final-1/python_numpy_rng.json', 'test/metrics.json', 'test/recomputed_metrics.json']
    ]:
        p = job / relative
        entry = manifest_map[job_id + '/' + relative]
        assert sha(p) == entry['sha256'] and p.stat().st_size == entry['bytes']
        checked_files.append(record(p))
    request = jread(job / 'request.json')
    accepted = jread(job / 'accepted.json')
    status = jread(job / 'status.json')
    cleanup = jread(job / 'cleanup.json')
    assert request['job_id'] == status['job_id'] == job_id
    fingerprint = hashlib.sha256(canonical(request)).hexdigest()
    assert accepted['request_sha256'] == fingerprint == prior['request_fingerprint_sha256']
    assert status['status'] == 'SUCCEEDED' and status['exit_code'] == 0 and status['worker_may_be_live'] is False
    assert cleanup == status['cleanup'] and cleanup['verified']
    assert cleanup['last_observation']['verified_empty']
    assert not cleanup['last_observation']['owned_live'] and not cleanup['last_observation']['unconfirmed_live']
    assert request['timeout_seconds'] == 43200
    started = dt.datetime.fromisoformat(status['started_at'])
    ended = dt.datetime.fromisoformat(status['ended_at'])
    total_seconds = (ended - started).total_seconds()
    assert 0 < total_seconds < 43200 - 180
    invocation = jread(output / 'invocation.json')
    assert invocation['code_identity']['source_commit'] == startup['source_commit'] == '4495c38459971aee36b00d881eb50f50fafa3159'
    assert invocation['code_identity']['snapshot_sha256'] == startup['snapshot_sha256'] == '77c012c07de3db19bc86108212ab292aabe704922f1f65d1d9c51da62b05a108'
    source_map = {r['path']: r['sha256'] for r in invocation['code_identity']['files'] if r['path'].endswith('.py')}
    gate = jread(BASE / 'mlp-admission' / (domain + '-gate-review.json'))
    assert source_map == gate['source_python_sha256'] and len(source_map) == 171
    assert sha(REPO / 'experiments/formal_20261015/train.py') == source_map['experiments/formal_20261015/train.py']
    assert sha(REPO / 'experiments/formal_20261015/run.py') == source_map['experiments/formal_20261015/run.py']
    for name in ['invocation.json', 'used_config.json', 'effective_config.yaml', 'approved_formal_effective_config.yaml',
                 'domain_protocol.json', 'actual_trainer.yaml', 'structure.json', 'actual_train_loader.json']:
        assert sha(output / name) == prior['raw_output_records'][name]['sha256']
    protocol = jread(output / 'domain_protocol.json')
    used = jread(output / 'used_config.json')
    structure = jread(output / 'structure.json')
    effective = yaml.safe_load((output / 'effective_config.yaml').read_text())
    assert effective['ckpt_path'] is None and structure['parameters_trainable'] == structure['parameters_total'] == parameters
    assert protocol == prior['actual_protocol']
    assert used['trainer']['accumulate_grad_batches'] == accumulation
    assert used['trainer']['val_check_interval'] == accumulation * 100
    assert used['trainer']['max_steps'] == 320000 and used['trainer']['max_epochs'] == 10
    assert used['model']['optimizer'] == {'_target_': 'torch.optim.Adam', '_partial_': True, 'lr': 0.001, 'weight_decay': 0.0001}
    assert used['model']['scheduler'] is None
    assert used['callbacks']['early_stopping']['patience'] == 10 and used['callbacks']['early_stopping']['min_delta'] == 0

    result = jread(output / 'RESULT.json')
    training = jread(output / 'training_result.json')
    for key in training:
        assert result[key] == training[key]
    assert training['status'] == 'COMPLETE' and training['training_complete'] and training['test_complete']
    assert training['fresh_initialization'] and training['full_parameter_training'] and not training['engineering_only']
    assert training['global_step'] == expected_steps and training['completed_full_validations'] == expected_validations
    assert training['actual_microbatches'] == expected_steps * accumulation
    assert training['termination'] == 'original_patience10_validation_early_stop'
    assert training['fit_wall_seconds'] < total_seconds

    validations = [json.loads(s) for s in (output / 'validations.jsonl').read_text().splitlines()]
    assert len(validations) == expected_validations
    running_best = -math.inf
    patience = 0
    first_patience_stop_step = None
    improvements = []
    full_val_seconds = 0
    for index, val in enumerate(validations, 1):
        assert val['validation_number'] == index and val['global_step'] == index * 100
        assert val['exact_key_target_multiset'] and val['targets'] == val['unique_users'] == target_count
        assert val['expected_index']['sha256'] == invocation['spec']['evaluation_index']['sha256']
        assert val['batch_size_histogram'] == {'8': target_count // 8, str(target_count % 8): 1}
        assert val['batches'] == target_count // 8 + 1
        assert all(math.isfinite(v) for v in val['metrics'].values())
        score = val['metrics']['val/recall@5']
        if score > running_best:
            running_best, patience = score, 0
            improvements.append(val['global_step'])
        else:
            patience += 1
        if patience >= 10 and first_patience_stop_step is None:
            first_patience_stop_step = val['global_step']
        full_val_seconds += val['hook_seconds']
    assert first_patience_stop_step == expected_steps and patience == 10
    assert improvements[-1] == expected_best
    best = training['best_checkpoint']
    assert best['score'] == running_best and best['monitor'] == 'val/recall@5' and best['mode'] == 'max'
    assert best['path'].endswith(f'checkpoint_epoch=000_step={expected_best:06d}.ckpt')
    assert training['early_stopping']['wait_count'] == training['early_stopping']['patience'] == 10
    assert training['early_stopping']['trainer_should_stop']

    csv_rows = list(csv.DictReader((output / 'csv/version_0/metrics.csv').read_text().splitlines()))
    csv_vals = [r for r in csv_rows if r.get('val/recall@5')]
    assert len(csv_vals) == len(validations)
    for row, val in zip(csv_vals, validations):
        assert int(row['step']) + 1 == val['global_step']
        for metric, value in val['metrics'].items():
            assert float(row[metric]) == value
    csv_tests = [r for r in csv_rows if r.get('test/recall@5')]
    assert len(csv_tests) == 1 and int(csv_tests[0]['step']) == expected_steps
    test_metrics = jread(output / 'test/metrics.json')
    for metric, value in test_metrics['metrics'][0].items():
        assert float(csv_tests[0][metric]) == value

    update_count = 0
    total_windows = 0
    gradient_min = {name: math.inf for name in ['0.weight', '0.bias', '3.weight', '3.bias']}
    gradient_max = {name: 0 for name in gradient_min}
    loss_min, loss_max = math.inf, -math.inf
    optimizer_wall = 0
    with (output / 'optimizer_updates.jsonl').open() as stream:
        for update_count, line in enumerate(stream, 1):
            update = json.loads(line)
            assert update['optimizer_update'] == update_count
            assert update['microbatches'] == accumulation
            assert update['cumulative_microbatches'] == update_count * accumulation
            windows = update['expanded_windows']
            losses = update['returned_training_losses']
            assert len(windows) == len(losses) == accumulation
            assert all(isinstance(v, int) and 0 < v <= cap for v in windows)
            assert all(math.isfinite(v) for v in losses)
            loss_min, loss_max = min(loss_min, *losses), max(loss_max, *losses)
            total_windows += sum(windows)
            assert update['cumulative_expanded_windows'] == total_windows
            opt = update['optimizer']
            assert opt['step_before'] == update_count - 1
            assert opt['optimizer_covers_every_trainable_parameter'] and opt['available_gradients_finite']
            assert opt['parameter_count'] == parameters
            assert set(opt['mlp_gradient_norms']) == set(gradient_min)
            for name, norm in opt['mlp_gradient_norms'].items():
                assert norm is not None and math.isfinite(norm) and norm >= 0
                gradient_min[name] = min(gradient_min[name], norm)
                gradient_max[name] = max(gradient_max[name], norm)
            optimizer_wall += update['wall_seconds_since_previous_update_including_intervening_validation']
    assert update_count == expected_steps and total_windows == training['expanded_windows']
    assert all(v > 0 for v in gradient_max.values())
    assert all(math.isfinite(v) and v > 0 for v in training['mlp_parameter_max_abs_updates'].values())

    strict = jread(output / 'best_checkpoint_load.json')
    assert strict['checkpoint'] == best and strict['best_global_step'] == expected_best
    assert strict['missing_keys'] == strict['unexpected_keys'] == []
    assert strict['all_keys_shapes_values_exact'] and strict['embedded_sid_exact'] and strict['optimizer_and_callbacks_present']
    assert strict['state_tensor_count'] == len(structure['state_dict_shapes']) == 180
    assert strict['selected_by'] == 'validation val/recall@5 only; original ModelCheckpoint'
    recovery = jread(output / 'latest_recovery.json')
    assert recovery == training['final_recovery'] == jread(output / 'recovery/training-final-1/recovery.json')
    assert recovery['global_step'] == expected_steps and recovery['microbatches_completed'] == expected_steps * accumulation
    assert recovery['microbatches_into_unfinished_accumulation'] == 0 and recovery['checkpoint_error'] is None
    assert recovery['datamodule_state_dict'] == {} and recovery['datamodule_state_is_empty']
    for flag in ['infinite_worker_cursor_saved', 'worker_process_RNG_and_prefetch_queue_saved',
                 'exact_uninterrupted_resume_claim', 'automatic_resume_allowed', 'partial_parameter_gradients_in_checkpoint']:
        assert recovery[flag] is False
    assert recovery['resume_requires_explicit_decision']
    assert recovery['weights_source'] == 'current training state'
    remote = entities['entities'][domain]
    for kind, expected in [('best', best), ('recovery', recovery['checkpoint'])]:
        assert len(remote[kind]) == 1
        actual = remote[kind][0]
        assert actual['path'] == expected['path'] and actual['bytes'] == expected['bytes'] and actual['sha256'] == expected['sha256']
        assert actual['regular_file'] and not actual['symlink'] and actual['unchanged_during_read'] and actual['header_hex'] == '504b0304'
    assert {r['path']: r['sha256'] for r in remote['rng']} == {r['path']: r['sha256'] for r in recovery['rng_files']}

    log_lines = (job / 'work.log').read_text().splitlines()
    stop_lines = [{'line': i, 'text': text} for i, text in enumerate(log_lines, 1) if 'did not improve in the last 10 records' in text]
    assert len(stop_lines) == 1 and 'Signaling Trainer to stop.' in stop_lines[0]['text']
    checkpoint_lines = [{'line': i, 'text': text} for i, text in enumerate(log_lines, 1) if best['path'] in text]
    assert checkpoint_lines and any('saving model' in r['text'] for r in checkpoint_lines)
    assert not any('Traceback (most recent call last)' in text for text in log_lines)
    assert test_metrics['complete_test_coverage'] and test_metrics['bounded_test_batches'] is None
    assert test_metrics['samples'] == training['test_target_count'] == target_count
    assert training['test_exact_key_target_multiset']

    review = {
        'schema': 1, 'status': 'SOURCE_FORMAL_FINAL_REVIEW_PASS', 'reviewer': '/root/source_conditions_review',
        'domain': domain, 'reviewed_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'job_id': job_id, 'job_status': status['status'], 'exit_code': status['exit_code'],
        'cleanup_verified': True, 'worker_may_be_live': False,
        'source_commit': invocation['code_identity']['source_commit'],
        'snapshot_sha256': invocation['code_identity']['snapshot_sha256'],
        'source_python_count': len(source_map), 'request_fingerprint_sha256': fingerprint,
        'inherited_startup_review': record(startup_path),
        'immutable_startup_configuration_files_unchanged': True,
        'approved_protocol': protocol,
        'research_identity': invocation['spec']['research_identity'],
        'data_sid_conditions': {'config': invocation['spec']['config'], 'sid': invocation['spec']['sid'],
            'data_dir': invocation['spec']['data_dir'], 'evaluation_index': invocation['spec']['evaluation_index'],
            'test_index': invocation['spec']['test_index'],
            'prior_domain_gate': record(BASE / 'mlp-admission' / (domain + '-gate-review.json'))},
        'full_validation': {'records': len(validations), 'global_steps': [v['global_step'] for v in validations],
            'targets_each': target_count, 'unique_users_each': target_count,
            'all_recorded_key_target_multisets_exact': True, 'all_CSV_metrics_equal': True,
            'tail_batch_size': target_count % 8, 'complete_validation_hook_seconds_sum': full_val_seconds,
            'evidence_boundary': 'Verified every original full-validation record plus CSV and fixed reviewed multiset-check producer. Validation raw predictions were not re-generated; prior full target index admission is inherited.'},
        'optimizer_training': {'updates': update_count, 'microbatches': expected_steps * accumulation,
            'expanded_windows': total_windows, 'all_updates_sequential': True,
            'all_accumulation_microbatches_exact': True, 'all_available_gradients_finite': True,
            'all_updates_cover_every_trainable_parameter': True, 'parameter_count': parameters,
            'MLP_gradient_norm_min': gradient_min, 'MLP_gradient_norm_max': gradient_max,
            'returned_batch_end_loss_min': loss_min, 'returned_batch_end_loss_max': loss_max,
            'loss_scaling_note': 'These batch-end losses are Lightning returned losses under gradient accumulation; original model train/loss is retained independently in CSV.',
            'final_MLP_parameter_max_abs_updates': training['mlp_parameter_max_abs_updates']},
        'selection_and_termination': {'monitor': 'val/recall@5', 'mode': 'max', 'min_delta': 0.0, 'patience': 10,
            'improvement_steps_replayed_from_all_records': improvements, 'best_step': expected_best,
            'best_score': best['score'], 'first_step_reaching_patience10': first_patience_stop_step,
            'last_ten_records': [{'step': v['global_step'], 'val_recall_at_5': v['metrics']['val/recall@5']} for v in validations[-10:]],
            'termination': training['termination'], 'original_work_log_stop': stop_lines,
            'original_work_log_best_save': checkpoint_lines, 'not_hard_timeout_or_source_max_steps': True},
        'strict_best_load': strict,
        'checkpoint_entities': remote,
        'entity_observation': record(entities_path),
        'no_reviewer_checkpoint_model_loading': True,
        'strict_load_evidence_boundary': 'Reused the actual saved strict loading report produced by unchanged reviewed train.py:521-536: strict=True, every key/shape/value exact, embedded SID exact, shared encoder/decoder table identity preserved. Reviewer independently rehashed actual server entity; did not deserialize or rerun model.',
        'recovery': {'record': recovery,
            'evidence_boundary': 'Producer train.py saves weights_only=False, optimizer/callback state and explicit main-process Python/NumPy/Torch CPU/CUDA RNG beside the checkpoint. Server entities and sidecar hashes verified. Final recovery binary was not independently deserialized in this review. Infinite data stream cursor, worker RNG and prefetch queue are absent, so exact uninterrupted/bitwise stream continuation is not established; automatic resume remains forbidden.'},
        'test_source_path': {'target_count': target_count, 'saved_original_evaluator_metrics': test_metrics['metrics'][0],
            'complete_test_coverage_recorded': True, 'bounded_test_batches': None,
            'independent_prediction_metric_recomputation': 'Assigned to separate results reviewer; this source receipt does not substitute for that independent result acceptance.'},
        'resource_budget': {'runner_started_at': status['started_at'], 'runner_ended_at': status['ended_at'],
            'total_seconds_including_complete_test_and_cleanup': total_seconds, 'total_hours': total_seconds / 3600,
            'fit_seconds': training['fit_wall_seconds'], 'hard_cap_seconds': 43200,
            'remaining_from_original_domain_allowance_seconds': 43200 - total_seconds,
            'safety_180_seconds_stop_not_reached': True, 'GPU_peak_allocated_bytes': training['gpu_peak_allocated_bytes'],
            'GPU_peak_reserved_bytes': training['gpu_peak_reserved_bytes'],
            'no_new_attention_approval_by_this_receipt': True},
        'scope': 'Non-author final review of source/approved training protocol/optimization/validation selection/natural stopping/strict best loading and recovery boundary. No training, GPU model loading, large-asset download, metadata regeneration, result modification or Git edit.',
        'correction_budget': {'original_rounds_used': 3, 'explicit_serialization_exception_rounds_used': 1, 'new_research_correction_rounds': 0},
        'known_provenance_limits_inherited': gate['provenance_limits_retained'],
        'checked_original_files': checked_files,
        'reviewer_program': record(Path(__file__)),
    }
    target = DEST / (domain + '-source-final-review.json')
    target.write_text(json.dumps(review, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({'domain': domain, 'status': review['status'], 'receipt': record(target), 'total_seconds': total_seconds, 'updates': update_count, 'validations': len(validations), 'best_step': expected_best}, ensure_ascii=False))
