"""Write scoped nonauthor engineering admission from checked raw receipts."""
from __future__ import annotations

import ast
import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT = Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
OUT = Path(__file__).resolve().parent


def identity(path):
    value = path.read_bytes()
    return {'path': str(path), 'bytes': len(value), 'sha256': hashlib.sha256(value).hexdigest()}


def main():
    checked = []
    for name, expected in [
        ('baseline-inspection.json', 'BASELINE_EVIDENCE_VERIFIED_NEW_REPAIR_ADMISSION_PENDING'),
        ('static-repair-inspection.json', 'STATIC_REPAIR_REVIEW_PASS_RUNTIME_AND_REQUEST_REVIEW_PENDING'),
        ('cpu-raw-independent-inspection.json', 'INDEPENDENT_CPU_RAW_RECEIPTS_PASS'),
    ]:
        path = OUT / name
        row = json.loads(path.read_text())
        assert row['status'] == expected
        checked.append(identity(path))
    names = [str(p.relative_to(ROOT)) for p in sorted((ROOT / 'experiments/a2_fd_recovery_20261018').glob('*.py'))]
    names += ['experiments/launch_20261017/a2_upstream.py', 'experiments/launch_20261017/run.py']
    sources = []
    for name in names:
        path = ROOT / name
        ast.parse(path.read_text(), filename=str(path))
        sources.append({'relative_path': name, **identity(path)})
    value = {
        'reviewer': '/root/fd_repair_reviewer',
        'author_separation': 'Reviewer did not author or edit adapter, CPU test, controller, builder, research source, or job requests; wrote independent checks/review receipts only.',
        'reviewed_at': dt.datetime.now(dt.timezone.utc).isoformat(),
        'status': 'SOURCE_REPAIR_ADMITTED_PENDING_DEPLOYMENT_AND_FINAL_REQUEST_REVIEW',
        'fd_engineering_status': 'FD_ENGINEERING_EQUIVALENCE_PASS',
        'scientific_result_status': 'SCIENTIFIC_RESULT_NOT_YET_AVAILABLE',
        'future_results_approved': False,
        'formal_submission_approved_by_this_receipt': False,
        'source_files': sources, 'checked_evidence': checked,
        'findings_open_in_engineering_equivalence': [],
        'equivalence': {
            'real_sports_item_axis': 18357, 'fixture_batch_items': 2048,
            'ids_shape': [2048, 1], 'ids_dtype': 'torch.int32',
            'input_embedding_shape': [2048, 512], 'input_embedding_dtype': 'torch.float32',
            'ids_values_dtype_order_all_features_and_pairing_exact': True,
            'original_model_step_predict_step_writer_CPU_fixture_exact': True,
            'RNG_states_unchanged': True,
            'id_storages_before_after': [2048, 1], 'total_storages_before_after': [2049, 2],
            'bounded_512x2_IPC_original_FD': [527, 527, 1040, 527],
            'bounded_512x2_IPC_contiguous_FD': [16, 18, 16, 16],
            'native_2048x12_CPU_simulated_ranks': 2,
            'native_main_FD_range': [68, 75], 'native_worker_FD_range': [20, 96],
            'actual_PPID_receipts_checked': True,
            'limits_before': [1024, 524288], 'limits_after': [524288, 524288],
            'scope': 'One CPU main process simulated two ranks sequentially, using the actual preserved loaders and Lightning worker seed function. This is not real DDP2 execution.',
        },
        'source_control_and_method': {
            'author_frozen_sources_unchanged': True,
            'legacy_source_only_reviewed_scoped_engineering_diff': True,
            'new_collate': 'Call original collate, then torch.stack item_ids only; original feature tensors unchanged.',
            'no_worker_init_or_sharing_strategy_override': True,
            'quantizer_method_training_randomization_batch_workers_precision_seed_preserved': True,
            'checkpoint_gate': 'After original save, compare all saved state tensors exactly to trained model, then strict=True load those identical tensors into the same model. No new model or RNG consumption.',
            'checkpoint_gate_limit': 'Formal runtime receipt and payload current_layer/layers_initialized validation remain final result checks; this source receipt does not claim a quantizer checkpoint exists.',
        },
        'controller_builder_static_review': {
            'A2_only_stages': 11, 'no_encode_request': True,
            'old_controllers_and_34_stage_history_read_only': True,
            'new_job_ids_for_all_11_stages': True,
            'accepted_runner_GPU_locks_and_shared_resource_policy_inherited': True,
            'one_formal_quantizer_attempt_no_retry': True,
            'quantizer_outer_worker_wall_seconds': 3420,
            'quantizer_inner_callback_hard_seconds': 3300,
            'original_rkmeans_remaining_GPU_request_seconds': 6897.950524,
            'quantizer_reserved_GPU_request_seconds': 6840,
            'quantizer_remaining_margin_GPU_request_seconds': 57.950524,
            'original_absolute_deadline': '2026-10-19T05:40:30.012007+00:00',
            'deadline_per_stage_full_wall_plus_margin_gate': True,
            'historical_prior_engineering_usage_preserved': True,
            'scope': 'Read controller.py/build.py and accepted queue call chain. Compiled concrete specs/config, runner request hashes, frozen evidence set, snapshot bytes and live validate must be separately reviewed after deployment.',
        },
        'mandatory_before_formal_submission': [
            'Publish immutable source; verify GitHub exact ref and server deployment source/hash mapping.',
            'Review the concrete 11 specs, plan, launch-review, queue config, rendered requests, frozen lineage evidence and live validation output.',
            'Verify all new IDs unoccupied and both old controllers and received historical workers are terminal/clean; no SUBMITTING or identity ambiguity.',
            'Verify total/group remaining budgets, same shared GPU policies/locks, original embedding identity and unchanged absolute deadline.',
            'Issue separate concrete final launch/request admission; this source receipt alone does not authorize submission.',
        ],
        'mandatory_formal_startup_checks': [
            'Actual 2 GPU DDP ranks with expected UUIDs, each rank soft/hard values and 12 actual worker identities/limits.',
            'First batches returned, exact ID-embedding pairing and unchanged raw2048/worker12/prefetch2 conditions.',
            'No EMFILE/timeouts; FD below scoped min(8192,soft*0.5) guard.',
            'Quantizer initialization progresses, optimizer step increments, finite loss and fresh logs.',
        ],
        'old_failure_discrepancy': 'Original rank0 worker candidates have soft/hard524288; prompt table1024 conflicts with raw capture. Rank attribution remains argv-derived without saved PPID/worker_id.',
        'repair_round_boundary': 'Original task3/3 exhausted remains historical; this is the separately authorized one-set FD engineering recovery and one new formal quantizer attempt.',
        'dependencies_installed': [], 'global_config_changes': [],
        'GPT_UPLOAD_REQUIRED': 'YES',
    }
    path = OUT / 'source-repair-admission.json'
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'path': str(path), 'sha256': identity(path)['sha256'], 'fd_engineering_status': value['fd_engineering_status'], 'formal_submission_approved_by_this_receipt': False}))


if __name__ == '__main__':
    main()
