"""Seal scoped evidence-tool review; does not approve scientific completion."""
import ast
import datetime as dt
import hashlib
import json
from pathlib import Path
import tarfile

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1]


def identity(path):
    return dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())


names = ['collect_terminal_evidence.py', 'primary_a_route_acceptance.py', 'package_delivery.py', 'monitor_v2.py', 'watch_tick.py']
tools = []
for name in names:
    path = BASE / name
    ast.parse(path.read_text(), filename=str(path))
    tools.append(identity(path))
fixture_path = OUT / 'package-fixture-review-20261017T192451566122Z.json'
fixture = json.loads(fixture_path.read_text())
assert fixture['status'] == 'PACKAGE_TOOL_FIXTURES_PASS'
assert fixture['package_tool_source_sha256'] == identity(BASE / 'package_delivery.py')['sha256']
assert len(fixture['cases']) == 7
assert all(row['accepted'] is (row['case'] == 'normal_PARTIAL_package') for row in fixture['cases'])
capture_checks = []
for path in sorted((BASE / 'terminal-captures').iterdir()):
    receipt_path = path / 'capture-receipt.json'
    if not receipt_path.is_file():
        capture_checks.append(dict(directory=str(path),status='PRESERVED_FAILED_CAPTURE_NO_SUCCESS_RECEIPT',files=[identity(p) for p in path.iterdir() if p.is_file()],scientific_attempt=False))
        continue
    receipt = json.loads(receipt_path.read_text())
    raw = Path(receipt['raw_directory'])
    manifest = json.loads((raw / 'REMOTE_MANIFEST.json').read_text())
    assert len(manifest['files']) == receipt['verified_members'] == 40
    for row in manifest['files']:
        current = identity(raw / row['path'])
        assert current['bytes'] == row['bytes'] and current['sha256'] == row['sha256']
    with tarfile.open(receipt['archive'], 'r:gz') as archive:
        assert len(archive.getmembers()) == receipt['members'] == 41
    assert identity(Path(receipt['archive']))['sha256'] == receipt['sha256']
    capture_checks.append(dict(directory=str(path),status='EXISTING_TERMINAL_CAPTURE_INDEPENDENTLY_VERIFIED',receipt=identity(receipt_path),actual_payload_members=40,archive_members=41,terminal_jobs=receipt['jobs'],remote_mutations=0,scope='Existing metadata/GPU admission evidence, not a new capture or research run.'))
report = {
    'reviewer': '/root/fd_repair_reviewer', 'reviewed_at': dt.datetime.now(dt.timezone.utc).isoformat(),
    'status': 'DELIVERY_TOOLS_SCOPED_REVIEW_PASS',
    'scientific_task_final_review_pass': False,
    'future_scientific_results_approved': False,
    'author_separation': 'Reviewer did not edit the five production support tools. Directly read them, raised narrow defects; primary author fixed those. Reviewer wrote only local independent fixtures and this review.',
    'reviewed_tools': tools,
    'identified_and_resolved': [
        dict(id='path_alias',original_observation='Original safe_name inspected PurePosixPath.parts after normalization, permitting raw ./ and repeated-separator aliases in verify.',fix='Require raw name == as_posix plus NFC and casefold uniqueness; reject unsafe path components and credentials filenames.',evidence='Current canonical alias fixture is rejected.'),
        dict(id='suffix_limited_secret_scan',original_observation='Original build scanned only TEXT_SUFFIXES; extensionless or .env.local payloads could bypass supported-pattern scan, while verify did not independently scan bytes.',fix='Build scans every actual payload byte stream and verify separately scans every uncompressed member, including manifest; .env.* and extra credential filenames prohibited.',evidence='Build rejects extensionless synthetic token and .env.local, and verify rejects a hash-valid token-bearing ZIP.'),
        dict(id='silent_quantizer_rank_omission',original_observation='Original collector silently skipped .pt files outside narrow allowlist, including small rank0/1 quantizer prediction products.',fix='Small prediction-rank-0/1.pt explicitly allowed; all other tensor skips get assets_not_copied identity or explicit unknown-identity record.',evidence='Current collector source inspected; no remote recapture or model operation performed by reviewer.'),
    ],
    'test_evidence': identity(fixture_path),
    'tests': [{'case': r['case'], 'accepted': r['accepted'], 'error': r.get('error')} for r in fixture['cases']],
    'old_vs_new_test_scope': 'First dynamic fixture run already used the corrected source. Earlier defects were directly observed in original source, not falsely claimed as a pre-fix dynamic reproduction. Both corrected5-case and corrected7-case receipts remain preserved.',
    'collector_scope': {
        'remote_job_names_from_closed_A2_phase_allowlist': True,
        'terminal_raw_status_and_cleanup_last_verified_empty_required': True,
        'remote_no_write_no_model_no_GPU_no_submit_or_signal': True,
        'Windows_OpenSSH_binary_stdout_direct_to_local_archive': True,
        'member_byte_limits_and_total550MB_bound': True,
        'safe_regular_member_paths_no_duplicate_tar_names': True,
        'remote_per_member_SHA_and_runner_artifact_hashes_checked_when_available': True,
        'original_assets_omitted_not_deleted': True,
        'existing_capture_checks': capture_checks,
    },
    'arithmetic_scope': {
        'full_target_multiset_and_unique_user_key_coverage': True,
        'raw_prediction_file_runner_SHA_checked': True,
        'ten_original_candidates_retained_no_filter_completion_reorder': True,
        'all_four_Recall_NDCG_metrics_independent_float64_and_original_comparison': True,
        'invalid_duplicate_candidates_and_own_Normal_pairing_checked': True,
        'both_Toys_models_retained': True,
        'final_full_task_review_pass_explicit_false': True,
        'boundary': 'Only arithmetic/paired acceptance. Checkpoint/SID/shuffle implementation/attention identity remain separate nonauthor reviews. Relevant target score ties fail closed and would require inspecting native evaluator handling, not altering predictions.',
    },
    'package_scope': {
        'manifest_payload_member_size_SHA256_CRC32': True,
        'actual_ZIP_CRC_negative_fixture_checked': True,
        'whole_ZIP_SHA_and_verification_receipt_outside_archive': True,
        'manifest_self_hash_excluded_to_avoid_circularity': True,
        'raw_canonical_paths_NFC_casefold_and_credential_name_checks': True,
        'independent_actual_uncompressed_member_secret_pattern_scan': True,
        'large_checkpoints_T5_weights_TFRecord_environment_excluded': True,
        'integrity_PASS_not_scientific_PASS': True,
        'PARTIAL_fixture_remains_PARTIAL_after_build_verify': True,
    },
    'monitor_watch_scope': {
        'query_only_saved_exact_immutable_config_and_jobs': True,
        'owned_process_UID_start_ticks_proof_before_FD_read': True,
        'no_model_load_rehash_or_GPU_research': True,
        'no_submit_restart_signal_in_monitor_or_watch': True,
        'append_new_timestamped_local_observations_existing_latest_summary_only': True,
        'GPU_snapshot_and_peer_PS_metadata_only': True,
        'status_boundary': 'Mechanical RUNNING/SUCCEEDED and arithmetic summaries only; no automatic scientific final PASS. Numeric log error substring matching is a heuristic for operator review, not an automatic intervention.',
    },
    'findings_open_for_reviewed_authorized_tool_flow': [],
    'mandatory_final_delivery_checks': [
        'Do not place the fixtures subtree in the final staging; it intentionally contains invalid ZIP paths and synthetic token patterns. Include only the review summary/test source if needed.',
        'Review the actual final member manifest and all included source/result/review identities, source commit/server mapping, job states, resource totals, large-asset recovery entries and final status after real experiments terminate.',
        'Run this reviewed packager on finalized staging, independently verify the produced real ZIP, and report actual Windows/WSL paths, bytes, member count and full outer SHA.',
        'No FINAL_INTERNAL_REVIEW_PASS is issued by this tooling review; A2 training/evaluation/attention/statistics and final nonauthor result/delivery review remain mandatory.',
    ],
    'dependencies_installed': [], 'GPU_requested': 0, 'production_source_or_experiment_mutation_by_reviewer': False,
    'GPT_UPLOAD_REQUIRED': 'YES',
}
target = OUT / 'delivery-tool-review.json'
assert not target.exists()
target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(dict(path=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),status=report['status'],final_scientific_review_pass=False)))
