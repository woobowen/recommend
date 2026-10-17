"""Bind the nonauthor source/plan review for legacy admission, without submission."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT = Path('/home/addaswsw/recommend-support/grid-a2-fd-recovery-20261018')
OUT = Path(__file__).resolve().parent
COMPILED = ROOT / 'compiled-c31d2d3eaaafbd18'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = json.loads((OUT / 'source-repair-admission.json').read_text())
    evidence = json.loads((OUT / 'deployment-plan-independent-inspection.json').read_text())
    deployment = json.loads((ROOT / 'source-deployment.json').read_text())
    plan = json.loads((COMPILED / 'plan.json').read_text())
    assert source['fd_engineering_status'] == 'FD_ENGINEERING_EQUIVALENCE_PASS'
    assert evidence['status'] == 'IMMUTABLE_SOURCE_AND_ELEVEN_TEMPLATE_PLAN_PASS_CONCRETE_REQUEST_PENDING'
    assert evidence['source_commit'] == deployment['manifest']['source_commit']
    assert evidence['plan']['sha256'] == sha(COMPILED / 'plan.json')
    assert plan['source_python_sha256'] == {r['path']: r['sha256'] for r in deployment['manifest']['files'] if r['path'].endswith('.py')}
    references = []
    for name in ['source-repair-admission.json', 'source-repair-review.md', 'baseline-inspection.json', 'static-repair-inspection.json', 'cpu-raw-independent-inspection.json', 'deployment-plan-independent-inspection.json']:
        path = OUT / name
        references.append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)})
    result = {
        'status': 'SOURCE_AND_LAUNCH_ADMITTED_CONDITIONALLY',
        'fd_engineering_status': 'FD_ENGINEERING_EQUIVALENCE_PASS',
        'future_results_approved': False,
        'scientific_result_status': 'SCIENTIFIC_RESULT_NOT_YET_AVAILABLE',
        'reviewer': '/root/fd_repair_reviewer',
        'reviewed_at': dt.datetime.now(dt.timezone.utc).isoformat(),
        'plan': evidence['plan'],
        'source_python_sha256': plan['source_python_sha256'],
        'source_commit': evidence['source_commit'],
        'snapshot': evidence['snapshot'],
        'deployment_manifest_sha256': evidence['manifest_sha256'],
        'scope': 'A2-only source and exact eleven-template plan binding; author-preserved method and original T5 encode reused. For constructing immutable requests, not a substitute for concrete queue/request admission.',
        'concrete_request_admission_status': 'PENDING_POSTCOMPILE_LIVE_VALIDATION_AND_INDEPENDENT_REVIEW',
        'formal_submission_approved_by_this_receipt': False,
        'independent_review': {
            'non_author_of_research_and_recovery_code': True,
            'old_failure_raw_log_and_FD_capture_read': True,
            'original_collate_model_predict_writer_and_DDP_loader_call_chain_read': True,
            'actual_diff_and_effective_config_read': True,
            'raw_CPU_equivalence_and_IPC_artifacts_all_hashes_checked': True,
            'native_CPU_12workers_2048batch_pairing_and_FD_checked': True,
            'all_262_source_files_against_fixed_Git_and_independent_server_manifest_checked': True,
            'all_204_Python_hashes_exact_plan_deployment_mapping': True,
            'four_GitHub_fixed_commit_raw_file_contents_and_blob_hashes_checked': True,
            'all_11_template_fields_against_original_with_only_identity_and_authorized_engineering_changes': True,
            'reference_receipts': references,
        },
        'conditions_before_formal_submission': [
            'Review compiled queue configuration and all eleven rendered runner requests, including unique IDs, exact source/cwd/argv/env, spec/plan/review/frozen-evidence hashes and dynamic SID/checkpoint dependencies.',
            'Read actual server live validation for old-controller/old-job terminal-clean status, no SUBMITTING or new-ID conflict, runner/locks/shared GPU policy preservation, budgets and unchanged absolute deadline.',
            'Issue separate final concrete-request admission after those checks; do not treat this conditional source/plan receipt as that admission.',
        ],
        'conditions_after_formal_start': source['mandatory_formal_startup_checks'],
        'resource_limits': {
            'original_total_GPU_request_seconds': 143100,
            'old_stage_GPU_request_seconds': 41837.856885,
            'old_prior_engineering_GPU_request_seconds': 58.650573,
            'conservative_remaining_GPU_request_seconds': 101203.143115,
            'quantizer_original_remaining_GPU_request_seconds': 6897.950524,
            'quantizer_new_outer_wall_seconds': 3420,
            'quantizer_new_GPU_request_reservation_seconds': 6840,
            'quantizer_inner_callback_seconds': 3300,
            'generator_training_wall_cap_seconds': 43200,
            'eval_attention_GPU_request_seconds': 9000,
            'original_absolute_queue_deadline': plan['original_absolute_queue_deadline'],
            'new_GPU_budget_granted': False,
        },
        'limitations': [
            'CPU rank simulation is not real GPU DDP2. Real rank and 24-worker limits/FD/batches/optimizer steps must be checked at formal startup.',
            'CPU deterministic centroid fixture proves model/predict/writer interface equivalence, not trained quantizer or recommendation results.',
            'Tensor IPC FD amplification is strongly supported, not proven as the unique historical root cause.',
            'Original rank0 worker candidates soft=524288 conflicts with the prompt1024 table; original worker rank assignment remains argv-derived without saved PPID/worker_id.',
            'New A2 is a method reproduction with fresh generator training, not recovery of original historical Sports recommender weights.',
        ],
        'repair_round_boundary': source['repair_round_boundary'],
        'GPT_UPLOAD_REQUIRED': 'YES',
    }
    path = OUT / 'launch-review.json'
    assert not path.exists(), 'Do not overwrite frozen launch review identity'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path), 'status': result['status'], 'formal_submission_approved_by_this_receipt': False}))


if __name__ == '__main__':
    main()
