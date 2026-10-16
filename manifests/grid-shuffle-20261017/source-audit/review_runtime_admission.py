"""Independent read-only audit of raw preflight exports and exact formal plans.

All output is confined to this reviewer's support directory. No model is loaded.
"""
import argparse
from collections import Counter
import datetime
import hashlib
import json
import math
from pathlib import Path

REPO = Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
SUPPORT = Path(__file__).parent


def identity(path):
    data = path.read_bytes()
    return {'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def read(path):
    return json.loads(path.read_bytes())


def check_identity(record, path=None):
    path = Path(record['path']) if path is None else path
    actual = identity(path)
    assert actual['sha256'] == record['sha256'], path
    if 'bytes' in record:
        assert actual['bytes'] == record['bytes'], path
    return actual


def audit(directory, spec_path, asset_path):
    request, state, invocation = [read(directory/name) for name in
        ('request.json', 'status.json', 'output/invocation.json')]
    accepted = read(directory/'accepted.json')
    request_sha = hashlib.sha256(json.dumps(request, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
    assert accepted['request_sha256'] == request_sha
    exported = read(directory/'export-manifest.json')['files']
    names = {row['path'] for row in exported}
    assert len(names) == len(exported)
    assert {str(path.relative_to(directory)) for path in directory.rglob('*') if path.is_file()} == names | {'export-manifest.json'}
    for row in exported:
        path = Path(row['path'])
        assert not path.is_absolute() and '..' not in path.parts
        assert not (directory/path).is_symlink()
        check_identity(row, directory/path)
    assert request['job_id'] == state['job_id'] and request['job_id'].startswith('gs-20261017-')
    assert state['status'] == 'SUCCEEDED' and state['exit_code'] == 0
    assert state['cleanup']['verified'] and not state.get('worker_may_be_live', False)
    assert request['timeout_seconds'] <= 900
    assert invocation['engineering_only'] and invocation['interpreter'] == '/data/bwwu/envs/grid-20261009-e61baf3/bin/python'
    spec, original = read(spec_path), invocation['spec']
    domain, route = spec['domain'], spec['route']
    assert domain in ('toys', 'sports') and route == 'B'
    assert spec['formal_approved'] is True and spec['shuffle_n'] in (0, 3, 4) and spec['generation_k'] == 10
    assert original['engineering_only'] and original['shuffle_n'] == 0
    assert all(spec[k] == original[k] for k in ('domain', 'route', 'seed', 'generation_k', 'config', 'sid', 'checkpoint', 'data_dir'))
    asset_review = read(asset_path)
    assets = asset_review['B'][domain]['assets']
    assert all(spec[k] == assets[k] for k in ('config', 'sid', 'checkpoint', 'data_dir'))
    assert asset_review['B'][domain]['checkpoint_selection']['not_selected_by_largest_step_or_test_score']
    deployment = invocation['code_identity']
    assert request['capture_version'] == deployment['source_commit']+'/'+deployment['snapshot_sha256']
    digest = hashlib.sha256(json.dumps({'source_commit': deployment['source_commit'], 'files': deployment['files']}, sort_keys=True).encode()).hexdigest()
    assert digest == deployment['snapshot_sha256']
    python = {row['path']: row['sha256'] for row in deployment['files'] if row['path'].endswith('.py')}
    assert python and len(python) == sum(row['path'].endswith('.py') for row in deployment['files'])
    for name, digest in python.items():
        path = Path(name)
        assert not path.is_absolute() and '..' not in path.parts
        assert identity(REPO/path)['sha256'] == digest, name
    output = directory/'output'
    for row in state['artifacts']:
        check_identity(row, directory/row['path'])
    loading, result, off, data, cohort, coverage = [read(output/name) for name in
        ('loading.json', 'RESULT.json', 'off_and_cache.json', 'data_check.json',
         'fixed_attention_samples.json', 'test_coverage.json')]
    strict = loading['strict_load']
    assert strict['missing_keys'] == strict['unexpected_keys'] == []
    assert strict['all_parameter_keys_shapes_values_equal'] and strict['embedded_sid_exact']
    assert loading['shared_sid_table'] is True and off['shared_sid_object'] is True
    assert result['status'] == 'PASS' and result['formal'] is False and result['engineering_only']
    for field in ('checkpoint_parameters_unchanged', 'input_mask_labels_unchanged',
                  'effective_config_roundtrip_exact', 'separate_loss_generate_permutation_roles_verified'):
        assert result[field]
    assert off['off_rng_unchanged'] and off['off_cpu_rng_unchanged']
    for name in ('sid_exact', 'repeat_sid_exact', 'sample_order_sid_exact', 'batch_size_sid_exact',
                 'adapter_vs_original_single_sid_exact', 'batch_size_roundtrip_sid_exact', 'finite_loss'):
        assert off[name], name
    for name in ('decoder_hidden', 'loss', 'scores', 'repeat_scores', 'sample_order_scores',
                 'adapter_vs_original_single_scores', 'batch_size_roundtrip_scores'):
        assert off[name]['allclose'], name
    assert all(row['allclose'] for row in off['head_logits'])
    assert data['fourth_collision_rule_verified'] and data['source_data_modified'] is False
    assert data['sid_shape'][0] == 4 and data['sid_shape'][1] == coverage['sid_item_count']
    assert data['sid_shape'][1] == asset_review['B'][domain]['upstream']['shape'][0]
    assert coverage['sid'] == cohort['sid'] == spec['sid']
    assert coverage['original_label_targets_verified'] and coverage['complete_target_hidden_groups_verified']
    assert not coverage['model_execution'] and coverage['coverage_is_not_model_evaluation']
    assert cohort['selected_users'] == len(set(cohort['sample_keys'])) == 32
    assert not cohort['model_outputs_inspected'] and cohort['split'] == 'evaluation'
    assert all(len(values) == 32 for values in (cohort['targets'], cohort['input_ids'], cohort['attention_mask']))
    assert spec['test_index'] == result['test_index'] and spec['attention_cohort'] == result['attention_cohort']
    check_identity(spec['test_index'], output/'test_target_index.jsonl')
    check_identity(spec['attention_cohort'], output/'fixed_attention_samples.json')
    expected = Counter()
    for line in (output/'test_target_index.jsonl').read_text().splitlines():
        row = json.loads(line)
        assert len(row['target']) == 4 and all(type(value) is int and value >= 0 for value in row['target'])
        expected[(row['sample_key'], tuple(row['target']))] += 1
    assert sum(expected.values()) == coverage['targets'] == result['test_targets']
    assert len({key[0] for key in expected}) == coverage['unique_users']
    saved = []
    for n in (0, 3, 4):
        folder = output/f'trainer-test-{n}'
        trace, cleanup, metrics, recomputed = [read(folder/name) for name in
            ('intervention_trace.json', 'lifecycle_cleanup.json', 'metrics.json', 'recomputed_metrics.json')]
        batches = len(cleanup['real_batch_devices'])
        assert batches > 0 and trace['n'] == n and trace['state_cleared']
        assert trace['role_count_delta']['encoder_role_counts'] == {'loss': batches, 'generate': batches}
        shuffle = trace['role_count_delta']['shuffle_role_counts']
        assert shuffle.get('loss', 0) == shuffle.get('generate', 0) == (batches if n else 0)
        assert bool(trace['events']) == bool(n)
        if n:
            assert {row['role'] for row in trace['events']} == {'loss', 'generate'}
            assert all(row['multiset_exact'] and row['padding_unchanged'] and not row['replay'] for row in trace['events'])
        assert cleanup['exception'] is None and not cleanup['is_testing_after']
        assert all(cleanup[field] for field in ('encoder_mask_cleared', 'replay_cleared', 'evaluator_restored', 'author_replay_queue_cleared'))
        assert all(row['is_testing'] and row['input_device'] == 'cuda:0' and row['parameter_devices'] == ['cuda:0']
                   for row in cleanup['real_batch_devices'])
        assert metrics['formal'] is False and recomputed['status'] == 'PASS'
        assert recomputed['samples'] == metrics['samples'] and recomputed['samples'] > 0
        assert all(math.isfinite(value) and value <= 1e-7 for value in recomputed['trainer_absolute_differences'].values())
        check_identity({'path': str(folder/'predictions.jsonl'), 'sha256': recomputed['prediction_sha256']})
        pairs = Counter()
        for line in (folder/'predictions.jsonl').read_text().splitlines():
            row = json.loads(line); pairs[(row['sample_key'], tuple(row['target']))] += 1
            assert len(row['sids']) == len(row['scores']) == 10 and all(len(value) == 4 for value in row['sids'])
            assert all(math.isfinite(value) for value in row['scores'])
        saved.append(pairs)
    assert saved[0] == saved[1] == saved[2]
    identifiable = read(output/'identifiable_permutations.json')
    assert {row['n'] for row in identifiable} == {3, 4}
    assert all(row['exact_expected'] and row['padding_unchanged'] for row in identifiable)
    assert next(row for row in identifiable if row['n'] == 3)['fourth_unchanged']
    post = read(output/'post_test_off.json')
    assert post['sid_exact'] and not post['is_testing'] and post['model_eval'] and post['encoder_eval'] and post['decoder_eval']
    assert post['input_device'] == 'cuda:0' and post['parameter_devices'] == ['cuda:0'] and post['scores']['allclose']
    plan = {key: value for key, value in spec.items() if key not in ('nonauthor_gate', 'formal_approved')}
    digest = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
    input_files = [identity(path) for path in sorted(directory.rglob('*')) if path.is_file()]
    receipt = {
        'status': 'PREFLIGHT_GATE_PASS', 'reviewer_task': '/root/source_audit',
        'reviewed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'domain': domain, 'route': route, 'shuffle_n': spec['shuffle_n'],
        'plan_sha256': digest, 'source_python_sha256': python,
        'formal_spec': identity(spec_path), 'preflight_job_id': request['job_id'],
        'preflight_deployment': deployment, 'raw_evidence_files': input_files,
        'asset_review': identity(asset_path), 'coverage_targets': sum(expected.values()),
        'cohort_users': 32, 'conditions_actually_checked': [0, 3, 4],
        'checks': ['Exact deployment/current Python bytes', 'Original source and selection/asset pairing',
                   'Strict model key/shape/value load and embedded SID equality', 'Shared B embedding table',
                   'Normal vs original output/RNG equivalence', 'Native per-occurrence sampler and identifiable vector conservation',
                   'Three real test lifecycles with separate loss/generate sample roles', 'Input/mask/labels and parameters unchanged',
                   'Actual bounded predictions and metric recompute', 'Post-test CPU teardown/GPU restoration and normal flag cleanup',
                   'Route-specific trusted target index and prediction-independent fixed32 cohort'],
        'scope_limit': 'Formal evaluation authorization gate based on real engineering admission; full formal predictions, attention OFF/ON, final result and scientific phenomenon remain separate later checks.',
        'new_GPU_or_model_calls_by_reviewer': 0, 'program': identity(Path(__file__)),
    }
    destination = SUPPORT/f'{domain}-{route.lower()}-{spec["shuffle_n"]}-evaluation-independent-review.json'
    assert not destination.exists(), destination
    destination.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    return identity(destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', required=True, type=Path)
    parser.add_argument('--spec', required=True, type=Path)
    parser.add_argument('--asset-review', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.preflight, args.spec, args.asset_review)))
