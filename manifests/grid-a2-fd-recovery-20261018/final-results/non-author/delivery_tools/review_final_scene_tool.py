"""Independent bounded local contract tests; never run capture.main or SSH."""
import ast
import collections
import copy
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import types

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1]
REPO = Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
COMMIT = '85b77f5e2a5b497fdc1f620649ba488fc783a245'
tool = BASE / 'capture_final_scene.py'
source = tool.read_bytes()
tree = ast.parse(source, filename=str(tool))
remote = next(ast.literal_eval(node.value) for node in tree.body
              if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'REMOTE' for t in node.targets))
compile(remote.replace('CONFIG', repr('/not-executed/config.json')).replace('EXPECTED_SOURCE', repr({})), '<remote-code-syntax-only>', 'exec')
queue_source = subprocess.check_output(['git', 'show', COMMIT + ':experiments/launch_20261017/queue.py'], cwd=REPO)
qtree = ast.parse(queue_source)
qenv = {'json': json, 're': re}
for node in qtree.body:
    if isinstance(node, ast.FunctionDef) and node.name in {'canonical', 'gpus', 'allocation_stage'}:
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<preserved-queue-pure-helper>', 'exec'), qenv)
pure_q = types.SimpleNamespace(**{key: qenv[key] for key in ['canonical', 'gpus', 'allocation_stage']})
config_path = BASE / 'compiled-22286966cd544773/queue-config.json'
c_full = json.loads(config_path.read_text())
qid = 'ga-20261018-sportsa2-quantize-fd-recovery-v1'
mid = 'ga-20261018-sportsa2-metadata-fd-recovery-v1'
gid = 'ga-20261018-sportsa2-gpu-gate-fd-recovery-v1'
qroot = Path('/home/addaswsw/recommend-support/sid-extensions-p0/jobs/new-server') / qid / 'retrieved'
capture = BASE / 'terminal-captures/20261017T191745842345Z/raw'
roots = {qid: qroot, mid: capture / mid, gid: capture / gid}
docs = {name: {key: json.loads((root / (key + '.json')).read_text())
               for key in ['request', 'status', 'accepted', 'cleanup']}
        for name, root in roots.items()}
fixture_parent = OUT / 'fixtures'
fixture_parent.mkdir(exist_ok=True)
fixture = Path(tempfile.mkdtemp(prefix='final-scene-', dir=fixture_parent))
terminal = {'SUCCEEDED', 'FAILED', 'INTERRUPTED', 'TIMED_OUT', 'WORKERS_TERMINATED'}
job_fragment = remote[remote.index('jobs={}'):remote.index("assert jobs[c['queue_id']]")]


def exercise_job(name=qid, mutate=None, query_override=None, directory_only=False):
    sample = copy.deepcopy(docs[name])
    if mutate:
        mutate(sample)
    target = fixture / name
    target.mkdir(exist_ok=True)
    for key, value in sample.items():
        (target / (key + '.json')).write_text(json.dumps(value))
    if directory_only:
        (target / 'accepted.json').unlink()
    called = []

    def fake_query(config, mode, path):
        assert mode == 'query'
        called.append(str(path))
        value = copy.deepcopy(sample['status'])
        value.update(request_sha256=hashlib.sha256(pure_q.canonical(sample['request'])).hexdigest(),
                     cleanup_verified=sample['cleanup']['verified'], workgroup_live=False, supervisor_alive=False)
        if query_override:
            value.update(query_override)
        return value

    env = dict(c=c_full, q=types.SimpleNamespace(canonical=pure_q.canonical, runner_call=fake_query),
               root=fixture, names={name}, terminal=terminal, hashlib=hashlib,
               read=lambda path: json.loads(Path(path).read_text()))
    try:
        exec(compile(job_fragment, '<exact-repaired-job-query-fragment>', 'exec'), env)
        return dict(accepted=True, query_calls=len(called))
    except Exception as error:
        return dict(accepted=False, query_calls=len(called), error_type=type(error).__name__, error=str(error))


job_tests = {}
job_tests['real_quant_request_and_status_fields'] = exercise_job()
assert job_tests['real_quant_request_and_status_fields'] == dict(accepted=True, query_calls=1)
job_tests['raw_RUNNING_rejected_before_query'] = exercise_job(mutate=lambda doc: doc['status'].update(status='RUNNING'))
assert not job_tests['raw_RUNNING_rejected_before_query']['accepted'] and job_tests['raw_RUNNING_rejected_before_query']['query_calls'] == 0
job_tests['accepted_digest_mismatch_rejected_before_query'] = exercise_job(mutate=lambda doc: doc['accepted'].update(request_sha256='0' * 64))
assert not job_tests['accepted_digest_mismatch_rejected_before_query']['accepted'] and job_tests['accepted_digest_mismatch_rejected_before_query']['query_calls'] == 0
for field, value in [('request_sha256', '0' * 64), ('workgroup_live', True), ('supervisor_alive', True), ('cleanup_verified', False)]:
    result = exercise_job(query_override={field: value})
    assert not result['accepted'] and result['query_calls'] == 1
    job_tests['query_' + field + '_rejected'] = result
job_tests['unaccepted_directory_conflict_rejected'] = exercise_job(directory_only=True)
assert not job_tests['unaccepted_directory_conflict_rejected']['accepted'] and job_tests['unaccepted_directory_conflict_rejected']['query_calls'] == 0

account_fragment = remote[remote.index('new_usage=0.'):remote.index('def old_requested_usage')]


def exercise_account(names, mutate=None):
    config = copy.deepcopy(c_full)
    config['stages'] = [s for s in config['stages'] if s['request']['job_id'] in names]
    jobs = {name: dict(query={}, request=copy.deepcopy(docs[name]['request']), raw_status=copy.deepcopy(docs[name]['status'])) for name in names}
    if mutate:
        mutate(jobs, config)
    env = dict(c=config, jobs=jobs, q=pure_q, collections=collections, datetime=datetime,
               hashlib=hashlib, config_path=config_path)
    try:
        exec(compile(account_fragment, '<exact-repaired-new-accounting-fragment>', 'exec'), env)
        return dict(accepted=True, new_usage=env['new_usage'], details=env['details'])
    except Exception as error:
        return dict(accepted=False, error_type=type(error).__name__, error=str(error))


account_tests = {'real_quant_and_CPU_metadata': exercise_account([qid, mid])}
assert abs(account_tests['real_quant_and_CPU_metadata']['new_usage'] - 93.09782) < 1e-8
assert next(x for x in account_tests['real_quant_and_CPU_metadata']['details'] if x['job_id'] == mid)['gpu_requested_seconds'] == 0
mutations = {
    'duplicate_GPU_UUID': lambda jobs, config: jobs[qid]['request']['queue_stage'].update(gpu_uuids=[config['approved_gpu_uuids'][0]] * 2),
    'queue_stage_id': lambda jobs, config: jobs[qid]['request']['queue_stage'].update(id='wrong'),
    'queue_stage_workflow': lambda jobs, config: jobs[qid]['request']['queue_stage'].update(workflow='wrong'),
    'queue_stage_config_hash': lambda jobs, config: jobs[qid]['request']['queue_stage'].update(config_sha256='0' * 64),
    'queue_stage_source': lambda jobs, config: jobs[qid]['request']['queue_stage'].update(source_commit='0' * 40),
    'queue_stage_budget': lambda jobs, config: jobs[qid]['request']['queue_stage'].update(budget_group='engineering'),
    'queue_stage_command': lambda jobs, config: jobs[qid]['request']['queue_stage'].update(command=['wrong']),
    'CUDA_env_mismatch': lambda jobs, config: jobs[qid]['request']['env'].update(CUDA_VISIBLE_DEVICES=''),
    'CPU_GPU_env_forbidden': lambda jobs, config: jobs[mid]['request']['env'].update(CUDA_VISIBLE_DEVICES=config['approved_gpu_uuids'][0]),
    'cwd': lambda jobs, config: jobs[qid]['request'].update(cwd='/wrong'),
    'capture_identity': lambda jobs, config: jobs[qid]['request'].update(capture_version='wrong'),
    'wall_cap': lambda jobs, config: jobs[qid]['request'].update(timeout_seconds=3421),
    'absolute_deadline': lambda jobs, config: jobs[qid]['raw_status'].update(ended_at='2026-10-19T05:40:31+00:00'),
}
for label, mutation in mutations.items():
    result = exercise_account([qid, mid], mutation)
    assert not result['accepted'], label
    account_tests[label + '_rejected'] = result


def switch_single_GPU(jobs, config):
    request = jobs[gid]['request']
    other = next(gpu for gpu in config['approved_gpu_uuids'] if gpu not in request['queue_stage']['gpu_uuids'])
    request['queue_stage']['gpu_uuids'] = [other]
    request['env']['CUDA_VISIBLE_DEVICES'] = other


account_tests['approved_single_GPU_candidate_switch'] = exercise_account([gid], switch_single_GPU)
assert account_tests['approved_single_GPU_candidate_switch']['accepted']
old_function = remote[remote.index('def old_requested_usage'):remote.index('old_names=')]
old_jobs = {}
old_root = Path('/home/addaswsw/recommend-support/grid-a-launch-20261017/startup-evidence')
for name in c_full['prior_gpu_request_usage']['jobs']:
    old_jobs[name] = dict(query={}, request=json.loads((old_root / name / 'request.json').read_text()),
                          raw_status=json.loads((old_root / name / 'status.json').read_text()))
env = dict(c=c_full, jobs=old_jobs, datetime=datetime)
exec(compile(old_function, '<exact-repaired-old-accounting-helper>', 'exec'), env)
prior, prior_rows = env['old_requested_usage'](list(old_jobs))
assert abs(prior - 58.650573) < 1e-8

# Scope checks are source inspection, not execution of the final remote script.
rtree = ast.parse(remote)
remote_calls = [node for node in ast.walk(rtree) if isinstance(node, ast.Call)]
assert all(node.args[1].value == 'query' for node in remote_calls
           if isinstance(node.func, ast.Attribute) and node.func.attr == 'runner_call')
assert not any(isinstance(node.func, ast.Attribute) and node.func.attr in {'write_text', 'write_bytes', 'unlink', 'mkdir', 'rmdir', 'kill', 'killpg', 'submit'} for node in remote_calls)
assert "with path.open('r')" in remote and 'fcntl.LOCK_EX|fcntl.LOCK_NB' in remote
assert 'scientific_final_pass=False' in remote
assert "'ga-20261018-a2-fd-cpu-equivalence-v1'" in remote
transport = Path('/home/addaswsw/recommend-support/sid-extensions-p0/scripts/transport.py')
assert "self.ssh('python3 -I -S -B -'" in transport.read_text()

report = dict(
    reviewer='/root/fd_repair_reviewer', reviewed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    status='FINAL_SCENE_TOOL_SCOPED_STATIC_AND_LOCAL_CONTRACT_REVIEW_PASS', scientific_final_pass=False,
    actual_final_scene_captured=False, actual_jobs_current_terminal_verified=False,
    tool=dict(path=str(tool), bytes=len(source), sha256=hashlib.sha256(source).hexdigest()),
    remote_code_sha256=hashlib.sha256(remote.encode()).hexdigest(),
    pure_helper_source=dict(commit=COMMIT, path='experiments/launch_20261017/queue.py', sha256=hashlib.sha256(queue_source).hexdigest()),
    query_contract_tests=job_tests, new_stage_accounting_contract_tests=account_tests,
    actual_prior_three_recalculation=dict(gpu_requested_seconds=prior, rows=prior_rows),
    intended_final_scope=dict(original_templates=34, new_stages=11, frozen_small_evidence=17, new_execution_source_members=262,
        named_job_sources=['old34/new11 config IDs', 'old2/new1 controller IDs', 'prior3 GPU gates', 'CPU equivalence1', 'old3 CPU metadata'],
        unrelated_account_or_archive_scan=False),
    read_only_reason='Raw terminal status is required before original runner query, avoiding its control-loss diagnostic write branch; subprocess query and Transport use -B; lock files must already exist and are opened r with nonblocking advisory probes. No mutation, model, GPU research or runner submit method is invoked.',
    resource_scope='New11 accounting derives wall from raw start/end and GPUs from accepted-hash-bound actual request.queue_stage, validates approved allocations/UUID uniqueness/env/config/source/cwd/wall/deadline. Old34 and prior3 helpers independently sum accepted raw requests and compare frozen baselines; local tests directly rechecked all3 prior GPU gates, and new real quantization+metadata fields. No current server totals have been asserted.',
    initial_findings='Initial diagnostic-query write precondition, missing accepted/request identities and inherited-only historical totals are corrected. Original failed contract results remain in final-scene-tool-findings-initial.json.',
    fixture_scope='Standard-library local fixtures under reviewers/delivery_tools/fixtures, excluded from final evidence staging; altered requests are only malformed in-memory test inputs, never real requests.',
    limitations=['No SSH/Transport, final capture, active-state query or GPU action was run.', 'Actual terminal states, live ownership, 17 remote frozen hashes, 262 remote source hashes, final total/budget and lock availability remain for the real final capture and final result audit.', 'Successful capture is a resource/identity receipt and cannot approve scientific results.'],
    remote_operations=0, GPU_requested=0, model_loads=0, dependencies_installed=[], GPT_UPLOAD_REQUIRED='YES')
path = OUT / 'final-scene-tool-review.json'
assert not path.exists()
path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), tool_sha256=report['tool']['sha256'],
                     status=report['status'], query_cases=len(job_tests), accounting_cases=len(account_tests), prior_gpu_seconds=prior,
                     actual_final_capture_run=False)))
