import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

REPO = Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
BASE = Path('/home/addaswsw/recommend-support/grid-formal-20261015')
COLLECTED = BASE / 'collected-9a64fd31218d0c68'
DEST = BASE / 'mlp-attention-admission'
sys.path.insert(0, str(REPO))
from experiments.formal_20261015.deploy import source_records
from experiments.beauty_extensions.authorization import canonical_plan


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def record(p):
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)}


def read(p):
    return json.loads(p.read_text())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


records = source_records()
code = {r['path']: r['sha256'] for r in records if r['path'].endswith('.py')}
assert len(code) == 171
combined_path = BASE / 'mlp-results-review/mlp-formal-combined-review.json'
combined = read(combined_path)
assert combined['status'] == 'TOYS_SPORTS_FORMAL_TRAINING_AND_FULL_TEST_ACCEPTED'
entities_path = BASE / 'mlp-results-review/checkpoint-entity-evidence.json'
entities = read(entities_path)
beauty_gate_path = BASE / 'attention-admission/beauty-a-0-gate-review.json'
beauty_gate = read(beauty_gate_path)
assert beauty_gate['status'] == 'PREFLIGHT_GATE_PASS'
for name, digest in beauty_gate['observer_and_path_source_review']['exact_current_sources'].items():
    assert code[name] == digest
attachment_path = Path('/home/addaswsw/recommend/CODEX_GRID_FORMAL_CONTINUATION_FINAL_20261016.txt')
attachment = attachment_path.read_text()
assert '### 4.2 MLP 注意力' in attachment and 'Toys MLP、Sports MLP' in attachment
source_hooks = {name: code[name] for name in [
    'experiments/formal_20261015/run.py', 'experiments/formal_20261015/deploy.py',
    'experiments/beauty_extensions/common.py', 'experiments/beauty_extensions/observation.py',
    'experiments/beauty_extensions/attention/capture.py', 'experiments/beauty_extensions/attention/records.py',
    'experiments/beauty_extensions/attention/plot.py', 'experiments/beauty_extensions/author/models.py',
    'experiments/beauty_extensions/mlp_attention.py',
]}
DEST.mkdir(exist_ok=True)

for domain in ['toys', 'sports']:
    path = BASE / 'mlp-attention-specs' / (domain + '.json')
    spec = read(path)
    plan = canonical_plan(spec)
    assert plan == {k: v for k, v in spec.items() if k not in ('nonauthor_gate', 'formal_approved')}
    digest = hashlib.sha256(canonical(plan)).hexdigest()
    prior_gate_path = BASE / 'mlp-admission' / (domain + '-gate-review.json')
    prior_gate = read(prior_gate_path)
    assert prior_gate['status'] == 'PREFLIGHT_GATE_PASS' and prior_gate['source_python_sha256'] == code
    source_review_path = BASE / 'mlp-results-review' / (domain + '-source-final-review.json')
    source_review = read(source_review_path)
    result_review_path = BASE / 'mlp-results-review' / (domain + '-formal-review.json')
    result_review = read(result_review_path)
    assert source_review['status'] == 'SOURCE_FORMAL_FINAL_REVIEW_PASS'
    assert result_review['status'] == 'FORMAL_EVALUATION_ACCEPTANCE_PASS'
    condition = next(c for c in combined['conditions'] if c['domain'] == domain)
    assert condition['status'] == 'FORMAL_TRAINING_AND_COMPLETE_TEST_INDEPENDENTLY_ACCEPTED'
    assert condition['source_training_selection_review'] == record(source_review_path)
    assert condition['complete_prediction_result_review'] == record(result_review_path)
    assert condition['same_job_source_snapshot_fingerprint_best_checkpoint_and_budget']
    job_id = f'gf-20261015-{domain}-mlp-formal-v1'
    job = COLLECTED / job_id
    result_path = job / 'output/RESULT.json'
    status_path = job / 'status.json'
    result, status = read(result_path), read(status_path)
    invocation = read(job / 'output/invocation.json')
    assert status['status'] == 'SUCCEEDED' and status['exit_code'] == 0 and status['cleanup']['verified'] and not status['worker_may_be_live']
    assert result['status'] == 'COMPLETE' and result['formal'] and result['training_complete'] and result['test_complete']
    assert spec['route'] == 'MLP' and spec['domain'] == domain and spec['shuffle_n'] == 0
    assert spec['phase'] == 'formal_attention_capture' and spec['generation_k'] == 10 and spec['seed'] == 42
    for key in ['config', 'sid', 'checkpoint']:
        assert spec[key] == result[key]
    for key in ['test_index', 'evaluation_index', 'attention_cohort', 'data_dir']:
        assert spec[key] == invocation['spec'][key]
    assert spec['checkpoint'] == source_review['strict_best_load']['checkpoint']
    entity = entities['entities'][domain]['best'][0]
    assert all(entity[k] == spec['checkpoint'][k] for k in ['path', 'bytes', 'sha256'])
    assert spec['evaluation_result']['sha256'] == sha(result_path)
    assert spec['evaluation_status']['sha256'] == sha(status_path)
    assert spec['evaluation_independent_review']['sha256'] == sha(combined_path)
    assert spec['training_protocol_reference']['job_id'] == job_id
    assert spec['training_protocol_reference']['approved_training_spec_sha256'] == invocation['spec_sha256']
    protocol = spec['attention_protocol']
    assert protocol['kind'] == 'cross-attention' and protocol['layer'] == 'last decoder layer'
    assert protocol['display_heads'] == 'mean' and protocol['retained_heads'] == 'all raw heads of last layer'
    assert protocol['paths'] == ['actual generation final Top-1 ancestry', 'local first parent per step']
    assert protocol['scientific_similarity_claim'] is None
    cohort_path = Path('/home/addaswsw/recommend-support/sid-extensions-p0/jobs/new-server') / f'gf-20261015-{domain}-metadata-v1/retrieved/output/fixed_attention_samples.json'
    cohort = read(cohort_path)
    assert record(cohort_path)['sha256'] == spec['attention_cohort']['sha256'] == prior_gate['cohort']['asset']['sha256']
    assert cohort_path.stat().st_size == spec['attention_cohort']['bytes']
    assert cohort['selected_users'] == cohort['metadata_rows_scanned'] == 32 and len(set(cohort['sample_keys'])) == 32
    assert not cohort['model_outputs_inspected'] and not cohort['raw_user_ids_exported']
    assert cohort['split'] == 'evaluation' and cohort['sid'] == spec['sid']
    assert len(cohort['input_ids']) == len(cohort['attention_mask']) == len(cohort['targets']) == 32
    assert all(len(r) == 120 for r in cohort['input_ids']) and all(len(r) == 120 for r in cohort['attention_mask'])
    assert all(len(r) == 4 for r in cohort['targets'])
    assert prior_gate['cohort']['exact_complete_evaluation_index_prefix']
    assert prior_gate['cohort']['complete_item_groups_and_target_hiding_verified']
    allowance = spec['resource_allowance']
    wall = (dt.datetime.fromisoformat(status['ended_at']) - dt.datetime.fromisoformat(status['started_at'])).total_seconds()
    assert wall == source_review['resource_budget']['total_seconds_including_complete_test_and_cleanup']
    assert wall == allowance['actual_training_validation_and_test_worker_wall_seconds']
    assert allowance['domain_original_hard_limit_seconds'] == 43200
    assert allowance['remaining_before_attention_seconds'] == 43200 - wall
    assert allowance['this_attention_request_max_seconds'] == spec['attention_wall_seconds_limit'] == 1200
    assert allowance['maximum_cumulative_worker_wall_with_capture_seconds'] == wall + 1200 < 43200

    review = {
        'schema': 'grid-formal-MLP-fixed32-attention-independent-admission-v1',
        'kind': 'INDEPENDENT_PREFLIGHT_GATE', 'status': 'PREFLIGHT_GATE_PASS',
        'reviewer': '/root/source_conditions_review', 'reviewed_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'domain': domain, 'route': 'MLP', 'shuffle_n': 0, 'phase': 'formal_attention_capture',
        'plan_sha256': digest, 'canonical_plan': plan,
        'canonical_plan_excluded_fields': ['nonauthor_gate', 'formal_approved'],
        'reviewed_spec': record(path), 'source_python_sha256': code, 'source_python_file_count': len(code),
        'source_python_map_sha256': hashlib.sha256(canonical(code)).hexdigest(),
        'source_directory_scope': 'Current experiments.formal_20261015.deploy.source_records all 171 .py records; exact to accepted MLP training and inherited observer reviews.',
        'assets': {name: spec[name] for name in ['config', 'sid', 'checkpoint', 'attention_cohort', 'evaluation_index', 'test_index', 'evaluation_result', 'evaluation_status', 'evaluation_independent_review']},
        'formal_training_result_dependency': {
            'job_id': job_id, 'status': 'SUCCEEDED', 'exit_code': 0, 'cleanup_verified': True,
            'result_file': record(result_path), 'status_file': record(status_path),
            'source_training_selection_review': record(source_review_path),
            'complete_prediction_result_review': record(result_review_path), 'combined_review': record(combined_path),
            'best_complete_dictionary_exact': True, 'training_source_commit': invocation['code_identity']['source_commit'],
            'training_snapshot_sha256': invocation['code_identity']['snapshot_sha256'],
            'server_best_entity_verification_inherited': record(entities_path),
            'no_repeated_checkpoint_hash_or_model_load': True,
        },
        'cohort_review': {
            'cohort_file': record(cohort_path), 'fixed_selected_users': 32, 'unique_opaque_keys': 32,
            'source': 'Original CPU metadata first32 evaluation users fixed before training/model outputs.',
            'source_order_and_complete_evaluation_prefix': prior_gate['cohort']['exact_complete_evaluation_index_prefix'],
            'raw_input_shape': [32, 120], 'raw_mask_shape': [32, 120], 'target_shape': [32, 4],
            'target_hiding_and_complete_item_groups_inherited': prior_gate['cohort']['complete_item_groups_and_target_hiding_verified'],
            'source_gate': record(prior_gate_path), 'no_test_or_hit_based_sample_selection': True,
        },
        'observer_and_path_source_review': {
            'source_sha256': source_hooks, 'inherited_unchanged_observer_review': record(beauty_gate_path),
            'MLP_specific_source_read': ['experiments/formal_20261015/run.py:76-109', 'experiments/beauty_extensions/common.py:141-178', 'experiments/beauty_extensions/observation.py', 'experiments/beauty_extensions/mlp_attention.py'],
            'capture_scope': 'Four saved batches of eight fixed evaluation samples. Each batch performs paired observer OFF/ON actual generation with identical initial RNG; exact output SID/score/RNG/parameter identity and cleanup checks are enforced. No Trainer.test/full-domain evaluation or training.',
            'expected_saved_bundles': 4, 'expected_saved_observed_samples': 32,
            'paths': protocol['paths'], 'rows': protocol['rows'], 'columns': protocol['columns'],
            'retention': 'Last decoder layer all six raw heads; display arithmetic head mean, full encoder item slots/masks, shared paired color scale, no row-normalizing/cropping.',
            'MLP_axis_boundary': 'Per-item aggregation token columns after original four-SID MLP. Cannot compare these columns pixel-wise to A/B four-code SID/SEP axes.',
            'strict_best_loading': 'Existing instantiate(checkpoint=True) strict keys/shapes/values/SID/shared-table loading; no alternate/random decoder embedding.',
        },
        'finite_resource_allowance': allowance,
        'approval_basis': {'user_attachment': record(attachment_path), 'section': '4.2 MLP attention; per-domain fixed32 only if original remaining approved resource allows', 'reviewer_does_not_issue_user_token': True},
        'admission_scope': 'Only this exact domain/SID/config/validation-best/cohort/source map and maximum1200s fixed32 formal attention inference. No training, larger cohort, full test repetition, alternate weight, cross-domain time transfer or scientific similarity acceptance.',
        'required_post_execution_acceptance': [
            'Accepted request binds exact plan/current source map/assets and timeout no larger than1200s; terminal exit0 and cleanup verified.',
            'Exactly four bundles cover all32 original opaque cohort keys once; SID input, grouped item positions, hidden targets and mask remain exact.',
            'Every paired observer OFF/ON output/score/RNG/parameter identity, hook restoration and cleanup assertion passes.',
            'Four generated query steps and finalTop1 beam ancestry reconstruct the actual final SID; local-first path remains separately labeled.',
            'Raw six-head matrices are finite, respect valid item/padding mask and save/reload hashes; all32 PNGs reopen and match their matrix/paired color scale.',
            'Only descriptive row relation statistics; preserve dissimilar samples, with no threshold or unsupported invariance claim.',
            'Actual cumulative original training/test wall plus attention job wall remains within this domain43200s; no cross-domain transfer.'
        ],
        'formal_authorization_issued_by_reviewer': False, 'formal_authorization_token_created': False,
        'attention_result_status': 'NOT_YET_EXECUTED_OR_ACCEPTED_BY_THIS_REVIEW', 'scientific_similarity_claim': None,
        'reviewer_model_or_GPU_runs': 0, 'reviewer_repository_edits': [], 'reviewer_Git_operations': [],
        'correction_budget': {'original_rounds_used': 3, 'explicit_configuration_exception_rounds_used': 1, 'new_research_corrections': 0},
        'reviewer_program': record(Path(__file__)),
    }
    target = DEST / (domain + '-gate-review.json')
    target.write_text(json.dumps(review, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({'domain': domain, 'status': review['status'], 'plan_sha256': digest, 'receipt': record(target)}, ensure_ascii=False))
