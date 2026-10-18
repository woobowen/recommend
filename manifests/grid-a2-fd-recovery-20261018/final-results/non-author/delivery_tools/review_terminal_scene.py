"""Independent local cross-check of actual terminal receipts; no remote/model action."""
import ast
import collections
import datetime as dt
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import subprocess

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1]
REPO = Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
COMMIT = '85b77f5e2a5b497fdc1f620649ba488fc783a245'
SCENE = BASE / 'final-scene/scene-20261018T043428336611Z.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()


scene = read(SCENE)
config_path = BASE / 'compiled-22286966cd544773/queue-config.json'
c = read(config_path)
deployment = read(BASE / 'source-deployment.json')
queue_deployment = read(BASE / 'queue-deployment.json')
initial = read(BASE / 'initial-scene.json')
original = initial['old_config']
manifest = deployment['manifest']
assert scene['status'] == 'TERMINAL_SCENE_AND_RESOURCE_IDENTITY_CAPTURED'
assert scene['scientific_final_pass'] is False
assert scene['config_sha256'] == sha(config_path) == queue_deployment['config_sha256']
assert scene['source_commit'] == c['snapshot']['commit'] == manifest['source_commit'] == COMMIT
assert scene['source_snapshot'] == c['snapshot']['root']
assert scene['source_snapshot_sha256'] == manifest['snapshot_sha256']
source_actual = {row['path']: row for row in scene['source_members_rehashed']}
assert len(source_actual) == len(manifest['files']) == 262
for row in manifest['files']:
    actual = source_actual[scene['source_snapshot'] + '/' + row['path']]
    data = subprocess.check_output(['git', 'show', COMMIT + ':' + row['path']], cwd=REPO)
    assert len(data) == row['bytes'] == actual['bytes']
    assert hashlib.sha256(data).hexdigest() == row['sha256'] == actual['sha256']
assert scene['frozen_evidence_unchanged'] == c['recovery']['frozen_evidence']
assert len(scene['frozen_evidence_unchanged']) == 17
tool = BASE / 'capture_final_scene.py'
tool_review = read(OUT / 'final-scene-tool-review.json')
assert sha(tool) == tool_review['tool']['sha256']
tree = ast.parse(tool.read_text())
remote = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == 'REMOTE' for target in node.targets))
actual_script = BASE / 'final-scene/remote-20261018T043428336611Z.py'
expected_script = remote.replace('CONFIG', repr(queue_deployment['remote_config'])).replace('EXPECTED_SOURCE', repr(manifest))
assert actual_script.read_bytes() == expected_script.encode()
assert sha(actual_script) == scene['remote_script_sha256']

names = {s['request']['job_id'] for s in original['stages'] + c['stages']}
names.update(c['recovery']['terminal_controller_ids'])
names.update(c['prior_gpu_request_usage']['jobs'])
names.update([c['queue_id'], 'ga-20261018-a2-fd-cpu-equivalence-v1',
              'ga-20261017-toys5300-metadata-v1', 'ga-20261017-toys5000-metadata-v1', 'ga-20261017-sportsa1-metadata-v1'])
assert set(scene['jobs']) == names and len(names) == 55
accepted = {}
not_submitted = []
terminal = {'SUCCEEDED', 'FAILED', 'INTERRUPTED', 'TIMED_OUT', 'WORKERS_TERMINATED'}
for name, doc in scene['jobs'].items():
    if 'query' not in doc:
        assert doc == dict(status='NOT_SUBMITTED', directory_exists=False)
        not_submitted.append(name)
        continue
    query, request, raw, receipt = (doc[key] for key in ['query', 'request', 'raw_status', 'accepted'])
    digest = hashlib.sha256(canonical(request)).hexdigest()
    assert query['request_sha256'] == receipt['request_sha256'] == digest
    assert query['job_id'] == request['job_id'] == raw['job_id'] == name
    assert query['status'] == raw['status'] and query['status'] in terminal
    assert query['cleanup_verified'] and doc['cleanup']['verified']
    assert not query['workgroup_live'] and not query['supervisor_alive'] and not query['worker_alive']
    observation = query['workgroup_observation']
    assert observation['verified_empty'] and observation['scan_complete']
    for key in ['scan_errors', 'owned_live', 'unconfirmed_live', 'unsupported_live']:
        assert not observation[key], (name, key)
    accepted[name] = doc
assert len(accepted) == 45 and len(not_submitted) == 10
old_names = [s['request']['job_id'] for s in original['stages']]
new_names = [s['request']['job_id'] for s in c['stages']]
assert len(old_names) == 34 and len(new_names) == 11 and not set(old_names) & set(new_names)
old_counts = collections.Counter(scene['jobs'][name].get('query', scene['jobs'][name])['status'] for name in old_names)
assert old_counts == dict(SUCCEEDED=23, INTERRUPTED=1, NOT_SUBMITTED=10)
assert all(accepted[name]['query']['status'] == 'SUCCEEDED' and accepted[name]['query']['exit_code'] == 0 for name in new_names)
assert accepted['ga-20261017-sportsa2-quantize-v1']['query']['status'] == 'INTERRUPTED'
assert accepted['ga-20261017-sportsa2-quantize-v1']['query']['exit_code'] == -9
assert accepted['ga-20261017-queue-v1']['query']['status'] == 'FAILED'
assert accepted['ga-20261018-shared-queue-v1']['query']['status'] == 'SUCCEEDED'
assert accepted[c['queue_id']]['query']['status'] == 'SUCCEEDED'
assert scene['queue_state']['status'] == 'TERMINAL_PENDING_RESULT_REVIEW'

wrapped = {r['job_id']: r for r in read(BASE / 'compiled-22286966cd544773/wrapped-requests.json')}
deadline = dt.datetime.fromisoformat(c['recovery']['original_absolute_queue_deadline'])
assert deadline.isoformat() == '2026-10-19T05:40:30.012007+00:00'
new_rows, groups = [], collections.defaultdict(Decimal)


def requested(name):
    doc = accepted[name]
    raw, request = doc['raw_status'], doc['request']
    start, end = (dt.datetime.fromisoformat(raw[key]) for key in ['started_at', 'ended_at'])
    span = end - start
    wall = Decimal(span.days * 86400 + span.seconds) + Decimal(span.microseconds) / Decimal(1000000)
    uuids = request['queue_stage']['gpu_uuids']
    assert wall >= 0 and len(set(uuids)) == len(uuids) and set(uuids) <= set(c['approved_gpu_uuids'])
    assert request['env']['CUDA_VISIBLE_DEVICES'] == ','.join(uuids)
    return wall * len(uuids), dict(job_id=name, wall_seconds=str(wall), gpu_count=len(uuids), gpu_uuids=uuids, gpu_requested_seconds=str(wall * len(uuids)))


for stage in c['stages']:
    name = stage['request']['job_id']
    actual = accepted[name]['request']
    expected = json.loads(json.dumps(wrapped[name]))
    uuids = actual['queue_stage']['gpu_uuids']
    if uuids != stage.get('gpu_uuids', []):
        assert len(uuids) == 1 and uuids[0] in stage.get('gpu_candidates', [])
    expected['queue_stage']['gpu_uuids'] = uuids
    expected['env']['CUDA_VISIBLE_DEVICES'] = ','.join(uuids)
    assert actual == expected, name
    assert dt.datetime.fromisoformat(accepted[name]['raw_status']['ended_at']) <= deadline
    used, row = requested(name)
    row['budget_group'] = stage['budget_group']
    assert Decimal(row['wall_seconds']) <= Decimal(actual['timeout_seconds']) + 1
    if uuids:
        groups[stage['budget_group']] += used
    new_rows.append(row)
for group, used in groups.items():
    assert used <= Decimal(str(c['budget_groups'][group]))
prior_names = c['prior_gpu_request_usage']['jobs']
historical = sum((requested(name)[0] for name in old_names if name in accepted), Decimal(0))
prior = sum((requested(name)[0] for name in prior_names), Decimal(0))
new = sum(groups.values(), Decimal(0))
total = historical + prior + new
assert historical == Decimal('41837.856885') and prior == Decimal('58.650573')
assert new == Decimal('33992.349971') and total == Decimal('75888.857429')
assert Decimal(143100) - total == Decimal('67211.142571')
for key, expected in [('new_gpu_requested_seconds', new), ('historical_stage_gpu_seconds', historical),
                      ('prior_gpu_requested_seconds', prior), ('total_gpu_requested_seconds', total)]:
    assert abs(Decimal(str(scene['resources'][key])) - expected) < Decimal('0.000001')
assert abs(Decimal(str(scene['queue_state']['accounting']['used_gpu_request_seconds'])) - new) < Decimal('0.000001')
assert all(row['available'] and row['nonblocking_existing_file_probe'] for row in scene['locks']) and len(scene['locks']) == 2

training = read(BASE / 'reviews/a2_training/final_training_review.json')
assert training['reviewer'] == '/root/sports_a1_result_audit' and training['status'] == 'A2_TRAINING_NONAUTHOR_REVIEW_PASS'
assert training['source_commit'] == COMMIT and training['scientific_test_results_approved'] is False
train_id = 'ga-20261018-sportsa2-train-fd-recovery-v1'
artifact_map = {row['path']: row for row in accepted[train_id]['raw_status']['artifacts']}
local_train = BASE / 'reviews/a2_training/terminal-originals'
train_metadata_hashes = []
for row in training['original_artifacts_verified']:
    assert artifact_map[row['path']] == row
    path = local_train / row['path']
    if path.exists():
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
        train_metadata_hashes.append(identity(path))
for ref in training['audit_evidence']:
    assert sha(ref['path']) == ref['sha256']
train_result = read(local_train / 'output/training_result.json')
assert train_result['training_complete'] and train_result['fresh_initialization'] and train_result['full_parameter_training'] and train_result['independent_sid_tables']
assert train_result['best_checkpoint'] == training['best']['checkpoint']
assert train_result['final_recovery']['checkpoint'] == training['final_checkpoint']
assert train_result['sid']['sha256'] == '06cfa6ba0f64d6c68e0e94c4111edd51d086796e8cdb474ac3fcd424a4a2adf0'
updates = [json.loads(line) for line in (local_train / 'output/optimizer_updates.jsonl').read_text().splitlines() if line]
validations = [json.loads(line) for line in (local_train / 'output/validations.jsonl').read_text().splitlines() if line]
assert len(updates) == 7900 and [row['optimizer_update'] for row in updates] == list(range(1, 7901))
windows = 0
losses = []
for i, row in enumerate(updates, 1):
    assert row['optimizer']['step_before'] == i - 1
    assert row['optimizer']['all_available_gradients_finite'] and row['optimizer']['optimizer_covers_every_trainable_parameter']
    assert row['cumulative_microbatches'] == i * 16 and len(row['returned_training_losses']) == len(row['expanded_windows']) == 16
    windows += sum(row['expanded_windows'])
    assert row['cumulative_expanded_windows'] == windows
    assert all(math.isfinite(v) and v > 0 for v in row['optimizer']['watched_gradient_norms'].values())
    assert all(math.isfinite(v) for v in row['returned_training_losses'])
    losses.extend(row['returned_training_losses'])
assert windows == 39426025 and (min(losses), max(losses)) == tuple(training['loss_range'])
assert len(validations) == 79
for i, row in enumerate(validations, 1):
    assert row['validation_number'] == i and row['global_step'] == i * 100
    assert row['exact_key_target_multiset'] and row['targets'] == 35598 and row['batches'] == 4450
    assert row['expected_index'] == train_result['evaluation_index']
best = max(validations, key=lambda row: row['metrics']['val/recall@5'])
assert best['global_step'] == 6900 and training['best_step'] == 6900 and training['final_step'] == 7900
assert all(row['metrics']['val/recall@5'] < best['metrics']['val/recall@5'] for row in validations[-10:])
assert train_result['early_stopping'] == dict(wait_count=10, patience=10)
quant_review = read(BASE / 'reviews/fd/quantization-result-review.json')
assert quant_review['status'] == 'QUANTIZATION_AND_SID_INDEPENDENT_REVIEW_PASS'

primary = read(BASE / 'primary-result-acceptance/all-four.json')
assert primary['status'] == 'PRIMARY_AVAILABLE_A_ROUTE_ARITHMETIC_ACCEPTANCE_PASS' and primary['a2_included']
assert primary['final_full_task_review_pass'] is False and len(primary['routes']) == 4
prediction_hashes = []
for route in primary['routes']:
    count = route['samples']
    assert count == (19412 if route['route'].startswith('Toys') else 35598)
    assert set(route['conditions']) == {'Normal', 'Shuffle3', 'Shuffle4'}
    for label, condition in route['conditions'].items():
        name = condition['job_id']
        artifact = next(row for row in accepted[name]['raw_status']['artifacts'] if row['path'] == 'output/test/predictions.jsonl')
        assert artifact['sha256'] == condition['raw_prediction_sha256']
        root = BASE / ('reviews/toys/raw' if route['route'].startswith('Toys') else 'reviews/sports_a1/raw' if route['route'] == 'Sports-A1' else 'reviews/a2_outputs/terminal_outputs/raw/jobs')
        path = root / name / artifact['path']
        assert path.stat().st_size == artifact['bytes'] and sha(path) == artifact['sha256']
        with path.open('rb') as stream:
            assert sum(1 for line in stream if line.strip()) == count
        assert condition['full_target_multiset_exact'] and condition['duplicate_candidates'] == 0
        prediction_hashes.append(identity(path))
    for label, paired in route['paired'].items():
        assert paired['samples'] == count
        for k in [5, 10]:
            assert sum(paired[f'hit{k}_{a}to{b}'] for a in [0, 1] for b in [0, 1]) == count
            normal_hits = paired[f'hit{k}_1to0'] + paired[f'hit{k}_1to1']
            changed_hits = paired[f'hit{k}_0to1'] + paired[f'hit{k}_1to1']
            assert abs(normal_hits / count - route['conditions']['Normal']['independent_float64_metrics'][f'recall@{k}']) < 1e-12
            assert abs(changed_hits / count - route['conditions'][label]['independent_float64_metrics'][f'recall@{k}']) < 1e-12
        assert sum(paired[key] for key in ['rank_improved_miss11', 'rank_worsened_miss11', 'rank_unchanged_miss11']) == count
        assert paired['top1_changed'] <= paired['top10_order_changed'] and paired['top10_set_changed'] <= paired['top10_order_changed']

result = dict(
    reviewer='/root/fd_repair_reviewer', reviewed_at=dt.datetime.now(dt.timezone.utc).isoformat(),
    status='TERMINAL_SCENE_RESOURCES_SOURCE_AND_TRAINING_CROSS_REVIEW_PASS',
    global_final_internal_review_pass=False, scope='Actual final scene, immutable source/request identities, budget arithmetic and training/prediction cross-bindings. No inference or second scientific evaluation.',
    actual_scene=identity(SCENE), actual_executed_capture_script=identity(actual_script),
    source_commit=COMMIT, source_snapshot=scene['source_snapshot'], source_snapshot_sha256=scene['source_snapshot_sha256'],
    remote_source_hashes_compared_to_independently_reread_Git_blobs=262,
    frozen_evidence_exactly_unchanged=17, accepted_request_raw_query_digests_crosschecked=45,
    old34=dict(old_counts), new11=dict(SUCCEEDED=11), known_accepted_jobs=45, known_not_submitted_templates=10,
    active_known_workers=0, active_known_supervisors=0, existing_GPU_locks_available=2,
    controller_states={name: accepted[name]['query']['status'] for name in c['recovery']['terminal_controller_ids'] + [c['queue_id']]},
    resources=dict(new_stage_details=new_rows, new_group_seconds={k: str(v) for k,v in groups.items()},
                   historical_seconds=str(historical), prior_three_gate_seconds=str(prior), new_seconds=str(new),
                   total_seconds=str(total), total_hours=str(total / 3600), remaining_seconds=str(Decimal(143100) - total),
                   original_total_seconds=143100, original_absolute_deadline=deadline.isoformat(),
                   all_actual_new_requests_within_original_budget_and_deadline=True),
    training_crosscheck=dict(raw_update_records=7900, raw_microbatches=126400, expanded_windows=windows, full_validation_records=79,
                            targets_each=35598, best_step=6900, final_step=7900, patience=10,
                            best_checkpoint=training['best']['checkpoint'], final_checkpoint=training['final_checkpoint'],
                            metadata_artifacts_independently_rehashed=train_metadata_hashes,
                            dedicated_nonauthor_training_review=identity(BASE / 'reviews/a2_training/final_training_review.json'),
                            fresh_CPU_checkpoint_load_inherited_from_named_nonauthor=True, additional_model_loads=0),
    all12_prediction_hash_and_count_crosschecks=prediction_hashes,
    primary_metrics_scope='Original metrics retained; paired hit migration/recall/count consistency checked. Dedicated raw prediction, actual shuffle, attention and statistics audit remains separate.',
    pending=['Actual A2 output/attention/statistics nonauthor report and final cross-domain index.', 'Final scientific report/README, final GitHub commit readback and execution-versus-delivery source mapping.', 'Actual final ZIP and independent full member/hash/CRC/path/secret verification.'],
    limitations=['Current activity statements refer to the explicit55 named task identities observed in the actual terminal capture, not a claim that the whole server has no processes.',
                 'Old INTERRUPTED/-9 remains preserved; it is not evidence of an original GPU OOM.',
                 'Source/frozen remote bytes were physically hashed by the reviewed capture; this reviewer compares that actual receipt to independently read immutable Git and frozen manifests without another server query.',
                 'Training full validation equality is checked against original complete validation events; no validation rerun or initial full random-state dump is claimed.',
                 'A1/A2 methods reproduce new Sports dual-table trainings; A1-versus-A2 differences do not isolate a single causal SID factor.'],
    new_remote_operations=0, GPU_calls=0, model_loads=0, dependencies_installed=[], GPT_UPLOAD_REQUIRED='YES')
target = OUT / 'terminal-scene-cross-review.json'
assert not target.exists()
target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(dict(path=str(target), sha256=sha(target), status=result['status'], total_gpu_requested_seconds=str(total),
                     remaining_gpu_seconds=str(Decimal(143100)-total), prediction_files=len(prediction_hashes), global_pass=False)))
