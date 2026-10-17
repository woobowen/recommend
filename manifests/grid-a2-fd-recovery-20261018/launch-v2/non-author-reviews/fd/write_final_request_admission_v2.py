"""Seal the nonauthor concrete submission admission without submitting a job."""
import datetime as dt
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


concrete_path = OUT / 'concrete-requests-independent-inspection-v2.json'
concrete = json.loads(concrete_path.read_text())
assert concrete['status'] == 'CONCRETE_REQUESTS_AND_CURRENT_LIVE_ADMISSION_PASS'
remote = concrete['remote_independent_check']
assert remote['supervisor_request_accepted_by_readonly_verify'] is True
assert remote['quantizer_prelaunch_resource_ok'] and remote['disk']['ok']
assert remote['existing_project_GPU_locks_available'] and not remote['new_ID_conflicts']
deployment = json.loads((BASE / 'queue-deployment.json').read_text())
root = Path(deployment['local_root'])
request_path = root / 'supervisor-request.json'
assert sha(request_path) == concrete['supervisor_request_file_sha256']
assert sha(root / 'queue-config.json') == deployment['config_sha256'] == concrete['queue_config_sha256']
assert sha(BASE / 'submit_recovery.py') == concrete['submit_script_static_review']['sha256']
request = json.loads(request_path.read_text())
receipt = {
    'reviewer': '/root/fd_repair_reviewer',
    'reviewed_at': dt.datetime.now(dt.timezone.utc).isoformat(),
    'status': 'FINAL_A2_ONLY_CONCRETE_SUBMISSION_ADMISSION_PASS',
    'formal_submission_admitted': True,
    'future_results_approved': False,
    'config_sha256': deployment['config_sha256'],
    'outer_request_file_sha256': sha(request_path),
    'outer_request_canonical_sha256': concrete['supervisor_canonical_request_sha256'],
    'fd_engineering_status': 'FD_ENGINEERING_EQUIVALENCE_PASS',
    'scientific_result_status': 'SCIENTIFIC_RESULT_NOT_YET_AVAILABLE',
    'admission_statement': '本次一次正式A2量化及成功后10阶段获准提交；未来科研结果未批准。',
    'scope': {
        'controller_job_id': 'ga-20261018-a2-fd-queue-v1',
        'quantizer_job_id': 'ga-20261018-sportsa2-quantize-fd-recovery-v1',
        'stages': concrete['stage_request_checks'],
        'single_formal_quantizer_attempt': True, 'automatic_retry_permitted': False,
        'only_A2_chain_no_encode_or_Toys_A1_recomputation': True,
    },
    'identity': {
        'source_commit': deployment['source_commit'], 'snapshot': deployment['snapshot'],
        'config_path': deployment['remote_config'],
        'outer_runner': deployment['outer_runner'], 'outer_runner_sha256': deployment['outer_runner_sha256'],
        'namespace_sidecar_sha256': 'e610be95a47894472c3f6b8396af8eb48fb1a998696a3a4914af2d886cc00877',
        'accepted_scientific_child_runner_sha256': '398768a732bbfebde4308486a0cedd924595157fda6cbff6a88f0bfaa512beff',
        'original_T5_embedding_sha256': 'b137cb8a006a8b950396524ed5be3bbc91d711b36068d671c01da3742d7cda6f',
        'launch_review_v2_file_sha256': sha(OUT / 'launch-review-v2.json'),
    },
    'direct_checks': {
        'all_eleven_final_specs_and_wrapped_requests_argv_cwd_env_namespace_input_hashes_checked': True,
        'old_status_original_failure_CPU_test_and_review_frozen_hashes_bound': True,
        'all_future_SID_and_best_checkpoint_links_use_new_A2_attempt': True,
        'every_dependency_requires_success_exit0_cleanup_and_capture_identity': True,
        'quantizer30step_DDP2_three_layers_SIDshape_and_strict_checkpoint_mechanical_gates': True,
        'actual_new_deployed_outer_full_verify_accepted_real_request': True,
        'actual_live_lineage_all_historical_received_jobs_clean_and_new_IDs_absent': True,
        'old_c31_candidate_and_original_runner_and_failed_work_log_unchanged': True,
        'accepted_shared_GPU_policies_GPU_candidate_sets_and_locks_exact_original': True,
        'original_total_group_remaining_budget_and_absolute_deadline_preserved': True,
        'three_current_quantizer_GPU_resource_samples_and_disk_guard_pass': True,
        'existing_project_GPU_locks_nonblocking_readonly_probe_pass': True,
        'submit_script_two_nonauthor_reviews_four_fields_fresh_validate_single_submit_no_retry_checked': True,
        'no_submission_or_GPU_computation_performed_by_reviewer': True,
    },
    'current_remote_check_at': remote['at'],
    'current_old_jobs': remote['selected_old_job_queries'],
    'resources': {
        'historical_stage_GPU_seconds': remote['historical_accounting']['used_gpu_request_seconds'],
        'prior_engineering_GPU_seconds': 58.650573,
        'exact_original_total_used_GPU_seconds': 41896.507458,
        'exact_original_total_remaining_GPU_seconds': 101203.492542,
        'conservative_new_queue_remaining_GPU_seconds': 101203.143115,
        'new_chain_max_GPU_request_seconds': concrete['GPU_reserved_seconds'],
        'new_chain_reservations': concrete['reserved_by_group'],
        'quantizer_remaining_group_seconds': 6897.950524,
        'quantizer_two_card_outer_wall_seconds': 3420,
        'quantizer_two_card_reserved_GPU_seconds': 6840,
        'quantizer_remaining_margin_GPU_seconds': 57.950524,
        'supervisor_wall_seconds': request['timeout_seconds'],
        'original_absolute_deadline': remote['deadline'],
        'deadline_remaining_at_check_seconds': remote['deadline_remaining_seconds'],
        'GPU_resource_sample_receipt': str(concrete_path),
        'peer_process_presence_allowed_and_observed': True,
        'disk': remote['disk'],
    },
    'required_startup_and_final_checks': [
        'Fresh controller live validation immediately before one submit; if identity, resource, namespace or request changes, this hash-bound admission does not cover it.',
        'Observe actual DDP2 and GPU UUIDs; each rank12worker IDs/PPID/limits, native2048 batch, prefetch2, exact ID/embedding pairing and finite optimizer progress.',
        'Confirm no EMFILE/timeouts and scoped FD guard healthy, not merely RUNNING/PIDs.',
        'Only actual30-step/three-initialized-layer checkpoint and full18357-item SID gate permits the original downstream fresh dual-table training and evaluations.',
        'If this one new formal quantizer attempt fails materially, preserve both attempts and stop A2 dependency chain without automatic rerun.',
        'Final quantization/training/Shuffle/attention/statistics and independent result review remain mandatory; source/launch admission is not final scientific PASS.',
    ],
    'findings_open_before_submission': [],
    'pre_submission_discovery_preserved': {'path': str(OUT / 'outer-request-readonly-verify-blocked.json'), 'sha256': sha(OUT / 'outer-request-readonly-verify-blocked.json'), 'meaning': 'c31/45fa old outer rejected long CPU controller before submit; resolved by narrow reviewed outer runner derivative. No new formal quantizer attempt was spent.'},
    'evidence': [{'path': str(OUT / name), 'bytes': (OUT / name).stat().st_size, 'sha256': sha(OUT / name)} for name in ['source-repair-admission.json', 'runner-adaptation-review.json', 'deployment-plan-independent-inspection-v2.json', 'launch-review-v2.json', 'concrete-requests-independent-inspection-v2.json']],
    'dependencies_installed': [], 'global_config_changes': [], 'GPT_UPLOAD_REQUIRED': 'YES',
}
path = OUT / 'final-request-admission-v2.json'
assert not path.exists(), 'Do not overwrite an admitted request identity'
path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path),status=receipt['status'],formal_submission_admitted=True,future_results_approved=False,config_sha256=receipt['config_sha256'],outer_request_file_sha256=receipt['outer_request_file_sha256'])))
