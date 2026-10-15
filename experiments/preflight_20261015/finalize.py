"""Summarize existing, verified terminal evidence without loading any model.

This is delivery bookkeeping, not another repair round or experiment. Failed
attempts remain failed; research decisions and formal approval stay separate.
"""
import argparse
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil

from experiments.beauty_extensions.common import REPO, sha256, write_json
from .deploy import SUPPORT


STAGE = 'grid-shuffle-mlp-preflight-20261015'
DESTINATION = REPO / 'manifests' / STAGE
NEW_JOBS = {'gsp-20261015-toys-resume-r3', 'gsp-20261015-sports-resume-r1'}


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    args = parser.parse_args()
    evidence = args.evidence.resolve()
    manifest = read(evidence / 'COLLECTION_MANIFEST.json')
    assert manifest['model_payloads_excluded'] and manifest['source_data_excluded']
    assert len(manifest['jobs']) == 14
    for record in manifest['files']:
        path = evidence / record['path']
        assert not path.is_symlink() and path.stat().st_size == record['bytes']
        assert sha256(path) == record['sha256']
    previous = read(DESTINATION / 'EXECUTION_SUMMARY.json')
    jobs, outputs = [], {}
    for job in manifest['jobs']:
        base = evidence / job
        status, request = read(base / 'status.json'), read(base / 'request.json')
        cleanup = read(base / 'cleanup.json')
        assert status['status'] in ('SUCCEEDED', 'FAILED')
        assert cleanup['verified']
        invocation_path = base / 'output/invocation.json'
        invocation = read(invocation_path) if invocation_path.exists() else {}
        result_path = base / 'output/RESULT.json'
        result = read(result_path) if result_path.exists() else None
        phase = invocation.get('phase', request['argv'][4])
        identity = invocation.get('code_identity', {})
        began = dt.datetime.fromisoformat(status['started_at'])
        ended = dt.datetime.fromisoformat(status['ended_at'])
        wall = (ended - began).total_seconds()
        assert ended >= began
        if status['status'] == 'SUCCEEDED':
            assert result['status'] == 'PASS' and result['formal_approved'] is False
        jobs.append({
            'job_id': job, 'phase': phase, 'status': status['status'],
            'exit_code': status['exit_code'], 'cleanup_verified': True,
            'origin': 'continuation_new_execution' if job in NEW_JOBS else 'inherited_execution',
            'started_at': status['started_at'], 'ended_at': status['ended_at'],
            'request_wall_seconds_including_initialization': wall,
            'source_commit': identity.get('source_commit'),
            'snapshot_sha256': identity.get('snapshot_sha256'),
            'request_sha256': hashlib.sha256(json.dumps(request, sort_keys=True,
                ensure_ascii=False, separators=(',', ':')).encode()).hexdigest(),
            'deployment_root': request['cwd'],
            'physical_gpu_uuid': request['env']['CUDA_VISIBLE_DEVICES'] or None,
            'server_output': '/home/bwwu/grid-audits/sid-extensions-p0/jobs/' + job + '/output',
            'request_file': 'evidence/' + job + '/request.json',
            'log_file': 'evidence/' + job + '/work.log', 'result': result,
        })
        outputs[job] = base / 'output'
        if phase != 'prepare':
            # Small result records are readable on GitHub. Full logs,
            # predictions and sampling traces remain in the verified ZIP.
            selected = ['RESULT.json', 'loading.json', 'short_training.json',
                'strict_restore.json', 'restored_prediction_parity.json',
                'callback_restore.json', 'off_and_cache.json', 'shuffle_result.json',
                'post_test_off.json', 'exception_recovery.json', 'ordinary_validation.json',
                'train_off_forward.json', 'source_worker_lifecycle.json', 'resource-summary.json',
                'mlp_aggregation.json', 'empty_padding_probes.json', 'mask_contract.json']
            for name in selected:
                source = base / 'output' / name
                if source.exists():
                    target = DESTINATION / 'runs' / job / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
    counts = collections.Counter(j['status'] for j in jobs)
    assert counts == {'SUCCEEDED': 12, 'FAILED': 2}
    lines = []
    for name, job in [('A', 'gsp-20261015-a-shared-r1'), ('B', 'gsp-20261015-b-shared-r1')]:
        old = next(x for x in previous['lines'] if x['line'] == name)
        output = outputs[job]
        post = read(output / 'post_test_off.json')
        line = {k: old[k] for k in ('line', 'loading', 'sid_shape', 'sid_range', 'generation_k')}
        line.update({
            'ENGINEERING_PREFLIGHT_PASS': True, 'REAL_PREFLIGHT_PASS': True,
            'RESEARCH_CONDITIONS_CONFIRMED': True, 'READY_FOR_FORMAL_REVIEW': True,
            'technical_status': 'FORMAL_TECHNICALLY_READY_PENDING_USER_AUTHORIZATION',
            'formal_approved': False, 'formal_run_started': False,
            'bounded_preflight_job': job, 'test_rows_per_condition': 31,
            'test_batch_sizes': [8, 8, 8, 7], 'validation_rows': 11,
            'tested_conditions': ['Normal', 'Shuffle3', 'Shuffle4'],
            'source_worker_extra_scope': 'A only: original 8-worker datamodule, 4 batches / 32 unique users' if name == 'A' else 'source num_workers=0',
            'post_test_off': post, 'GPU_optimizer_updates': 0,
            'inherited_execution_code_validity': 'Current computation sources unchanged; run.py changes only MLP callback report serialization. Independent AST/byte comparison recorded.',
            'numerical_boundary': 'Same-shape OFF is exact. Original and adapter share observed cross-batch-size score drift; no universal batch-size equality claimed.',
            'maximum_tested_visible_history_items': 20,
            'independent_review_files': ['shuffle-a-execution-review.json' if name == 'A' else 'shuffle-b-review.json', 'continuation-independent-review.json'],
        })
        lines.append(line)
    for domain in ('toys', 'sports'):
        short_job = 'gsp-20261015-' + domain + '-short-r1'
        resume_job = 'gsp-20261015-' + domain + ('-resume-r3' if domain == 'toys' else '-resume-r1')
        short = read(outputs[short_job] / 'short_training.json')
        resume = read(outputs[resume_job] / 'short_training.json')
        restored = read(outputs[resume_job] / 'strict_restore.json')
        parity = read(outputs[resume_job] / 'restored_prediction_parity.json')
        assert short['new_optimizer_updates'] == 2 and resume['new_optimizer_updates'] == 1
        assert short['final_step'] == resume['initial_step'] == 2 and resume['final_step'] == 3
        assert restored['all_keys_shapes_values_equal'] and parity['maximum_score_absolute_error'] == 0
        assert all(v > 0 for v in short['aggregation_parameter_changes'].values())
        assert all(v > 0 for v in resume['aggregation_parameter_changes'].values())
        lines.append({
            'line': domain.title() + '-MLP', 'ENGINEERING_PREFLIGHT_PASS': True,
            'REAL_PREFLIGHT_PASS': True, 'RESEARCH_CONDITIONS_CONFIRMED': False,
            'READY_FOR_FORMAL_REVIEW': False,
            'technical_status': 'ENGINEERING_PREFLIGHT_PASS_RESEARCH_DECISION_PENDING',
            'formal_approved': False, 'formal_run_started': False,
            'short_job': short_job, 'independent_resume_job': resume_job,
            'optimizer_updates': {'scratch': 2, 'resume': 1, 'total': 3},
            'microbatches': {'scratch': 32, 'resume': 16}, 'gradient_accumulation': 16,
            'actual_collate_windows': {'scratch': short['actual_total_collate_windows'], 'resume': resume['actual_total_collate_windows']},
            'raw_microbatch': short['raw_batch_size'], 'validation_rows': 11,
            'test_rows': 19, 'test_batch_sizes': [8, 8, 3],
            'strict_restore': restored, 'restored_prediction_parity': parity,
            'all_four_mlp_parameters_updated_scratch_and_resume': True,
            'short_checkpoint': short['checkpoint'], 'resumed_checkpoint': resume['checkpoint'],
            'formal_state_reuse_allowed': False,
            'measured_training_validation_save_test_seconds': {'scratch': short['elapsed_seconds'], 'resume': resume['elapsed_seconds']},
            'engineering_metrics': short['saved_prediction_verification']['metrics'],
            'invalid_candidates': {'scratch': short['saved_prediction_verification']['invalid_candidate_count'], 'resume': resume['saved_prediction_verification']['invalid_candidate_count'], 'candidates_per_test': 190},
            'evaluation_boundary': 'Original evaluator unchanged, prefix constraint false, no candidate filtering; zero engineering metrics are not formal recommendation results.',
            'restore_boundary': 'Model/optimizer/global_step and EarlyStopping exact; changed-output-directory ModelCheckpoint ranking state resets per installed Lightning. RNG/data cursor are not uninterrupted continuations.',
            'dropout_evidence_boundary': 'Original Dropout0.1 and Lightning fit/eval lifecycle inspected; no per-phase Dropout.training field log was recorded.',
            'budget_estimation_boundary': 'Measured same-model peak memory and short fit/validation/save/test cost support bounded resource planning, not a reliable full-training ETA; formal protocol and budget remain unapproved.',
            'independent_review_files': ['source-conditions-review.json', 'continuation-independent-review.json'],
        })
    gpu_jobs = [j for j in jobs if j['phase'] != 'prepare']
    all_seconds = sum(j['request_wall_seconds_including_initialization'] for j in gpu_jobs)
    new_seconds = sum(j['request_wall_seconds_including_initialization'] for j in gpu_jobs if j['job_id'] in NEW_JOBS)
    summary = {
        'status': 'PARTIAL_PREFLIGHT', 'stage': STAGE,
        'conclusion': 'All four engineering preflight lines complete; future Toys/Sports formal research conditions/budgets require user decision.',
        'scope': 'ENGINEERING_PREFLIGHT_ONLY; no formal training, full evaluation or attention',
        'starting_commit': previous['starting_commit'],
        'round3_fixed_code_commit': '59c42a213f2c3f4b90d0c069d4d3b98a2d8a398b',
        'takeover': previous['takeover'], 'lines': lines, 'jobs': jobs,
        'job_counts': {**counts, 'accepted': 14, 'CPU': 7, 'GPU': 7, 'new_GPU_jobs': 2, 'formal_jobs': 0},
        'rejected_submissions': [{'job_id': 'gsp-20261015-toys-resume-r2', 'status': 'REJECTED_BEFORE_ACCEPTANCE', 'worker_started': False, 'GPU_seconds': 0, 'replacement_validation': 'gsp-20261015-toys-resume-r3', 'evidence': 'external-receipts/manifest-rejection-diagnosis.json'}],
        'correction_rounds': read(DESTINATION / 'rounds.json'),
        'gpu_budget': {'initial_target_hours': 2, 'all_corrections_max_hours': 6,
            'accepted_GPU_job_wall_hours_including_initialization_and_failed_attempt': all_seconds / 3600,
            'inherited_GPU_job_wall_hours': (all_seconds - new_seconds) / 3600,
            'new_GPU_job_wall_hours': new_seconds / 3600,
            'accounting': 'Conservative single-card request wall duration, not pure GPU compute time. CPU prepares and preacceptance rejections excluded.',
            'monitoring_boundary': 'Resource samples cover each actual GPU section at 10-second cadence with initial/final probes, not all import/initialization wall time.'},
        'bounded_plan_limits': previous['bounded_plan_limits'],
        'collection': read(SUPPORT / 'collection-receipt.json'),
        'active_stage_jobs': [], 'remaining_engineering_checks': [],
        'future_research_decisions': previous['future_research_decisions'],
        'prior_Beauty_delivery_status_unchanged': 'PARTIAL_BLOCKED',
        'prior_Beauty_results_or_checkpoints_modified': False,
        'formal_authorization_tokens_created': False,
        'new_dependencies': [], 'system_config_changes': [], 'cleanup_performed': False,
        'publication_verification': 'Post-commit publication.json and zip-receipt.json are external receipts; they bind the actual fixed GitHub ref/blobs and server export after publication.',
    }
    write_json(DESTINATION / 'EXECUTION_SUMMARY.json', summary)
    print(json.dumps({'status': summary['status'], 'accepted_jobs': 14,
        'all_four_engineering_lines_pass': True, 'GPU_request_wall_hours': all_seconds / 3600}))


if __name__ == '__main__':
    main()
