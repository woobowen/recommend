"""Submit Beauty V2 through the existing supervisor with its own budget ledger."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from lc_rec.shuffle_v2 import beauty, control, core
from lc_rec.shuffle_v2.collect import hardware

NAMESPACE = 'letter-shuffle-v2-beauty'
PREFIX = 'letter-beauty-v2-'
SUPPORT = Path('/home/addaswsw/recommend-support') / NAMESPACE
ROOT = '/data/bwwu/lc-rec/' + NAMESPACE
CONFIG_PATH = 'lc_rec/shuffle_v2/a0_beauty_letter.json'
SOURCE_PATHS = ('lc_rec/shuffle_v2', 'lc_rec/shuffle/core.py', 'lc_rec/letter/scripts/formal_evaluate.py',
                'lc_rec/letter/review/final_audit_20261113/statistics_verify.py')
CPU_LIMITS = {'replay': 1200, 'evaluate': 600, 'statistics': 2100}
RUNNER_SOURCE = 'lc_rec/shuffle_v2/beauty_runner.py'


def connect(runner_sha=None):
    connection = control.connect()
    if runner_sha:
        connection.profile.update(
            runner=f'{control.BASE}/scripts/runner-beauty-{runner_sha[:16]}.py',
            runner_sha256=runner_sha)
    return connection


def runner_sha_at(commit):
    raw = subprocess.check_output(['git', 'show', f'{commit}:{RUNNER_SOURCE}'], cwd=control.REPO)
    return hashlib.sha256(raw).hexdigest()


def deploy_runner(connection, commit):
    raw = subprocess.check_output(['git', 'show', f'{commit}:{RUNNER_SOURCE}'], cwd=control.REPO)
    assert hashlib.sha256(raw).hexdigest() == connection.profile['runner_sha256']
    legacy = control.connect()
    original_deployment = legacy.profile['runner'] + '.deployment.json'
    info = json.loads(control.remote_python(connection, f'''
from pathlib import Path
import hashlib, json
p = Path({original_deployment!r})
raw = p.read_bytes()
print(json.dumps(dict(raw=raw.decode(), sha256=hashlib.sha256(raw).hexdigest())))
'''))
    assert info['sha256'] == connection.profile['deployment_sha256']
    runner = connection.profile['runner']
    staged = SUPPORT / f"runner-{connection.profile['runner_sha256'][:16]}"
    staged.mkdir(exist_ok=True)
    for name, contents in (('runner.py', raw), ('deployment.json', info['raw'].encode())):
        path = staged / name
        if path.exists():
            assert path.read_bytes() == contents
        else:
            path.write_bytes(contents)
    metadata = connection.put(staged / 'deployment.json', runner + '.deployment.json')
    actual = connection.put(staged / 'runner.py', runner)
    assert metadata['sha256'] == connection.profile['deployment_sha256']
    assert actual['sha256'] == connection.profile['runner_sha256']
    control.transport.atomic(staged / f'deployment-{commit}.json',
        {'previous_runner': legacy.profile['runner'], 'previous_runner_sha256': legacy.profile['runner_sha256'],
         'runner': actual, 'deployment': metadata, 'source_commit': commit, 'scope': PREFIX})


class NoIdleGpu(RuntimeError):
    pass


def attempt_id(value):
    digits = value[1:]
    if not value.startswith('r') or not digits.isascii() or not digits.isdecimal() or len(digits) > 6:
        raise argparse.ArgumentTypeError('Expected r followed by a nonnegative attempt number')
    if value != f'r{int(digits)}':
        raise argparse.ArgumentTypeError('Use canonical attempt numbers without leading zeros')
    return value


def commit_ref(ref):
    return subprocess.check_output(['git', 'rev-parse', '--verify', ref + '^{commit}'],
                                   cwd=control.REPO, text=True).strip()


def device(connection, config, preferred=None):
    snapshot = hardware(connection)
    allocations = {row.split(',')[0].strip() for row in snapshot['processes'].splitlines()[1:]}
    eligible = []
    for line, expected in zip(snapshot['gpus'].splitlines()[1:], config['gpus'], strict=True):
        _, uuid, model, used, total, _ = map(str.strip, line.split(','))
        assert uuid == expected and model == 'NVIDIA GeForce RTX 4090' and int(total.split()[0]) >= 24000
        if uuid not in allocations and int(used.split()[0]) < 100:
            eligible.append(uuid)
    if preferred in eligible:
        return preferred, snapshot
    if not eligible:
        raise NoIdleGpu('Neither approved card is idle')
    return eligible[0], snapshot


def wait_for_device(connection, config, preferred=None):
    deadline = time.monotonic() + config['budget']['gpu_wait_seconds']
    while True:
        try:
            return device(connection, config, preferred)
        except NoIdleGpu:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            print(f'GPU busy; {int(remaining)} s left in bounded wait', file=sys.stderr, flush=True)
            time.sleep(min(60, remaining))


def deployment(commit):
    record = json.loads((SUPPORT / (commit + '.deployment.json')).read_text())
    assert record['RUN_COMMIT'] == commit and record['runner_sha256'] == runner_sha_at(commit)
    return record['release'], [{'path': record['release'] + '/' + f['path'], 'sha256': f['sha256']} for f in record['files']]


def request(connection, config, commit, name, module, timeout, extra, inputs=(), card=''):
    release, files = deployment(commit)
    profile = connection.profile
    identity = {key: profile[key] for key in ('user', 'uid', 'namespace', 'deployment_sha256')}
    env = {'CUDA_VISIBLE_DEVICES': card, 'HF_HUB_OFFLINE': '1', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
           'OPENBLAS_NUM_THREADS': '4', 'PYTHONOPTIMIZE': '0', 'PYTHONDONTWRITEBYTECODE': '1',
           'PYTHONUNBUFFERED': '1', 'TOKENIZERS_PARALLELISM': 'false',
           'TORCHINDUCTOR_COMPILE_THREADS': '1', 'TRANSFORMERS_OFFLINE': '1', 'USE_FLAX': '0', 'USE_TF': '0'}
    assert timeout <= (config['budget']['cpu_attempt_wall_seconds'] if name.startswith('cpu-') else 345600)
    if card:
        assert card in config['gpus']
        env.update(config['gpu_environment'])
    return {**identity, 'schema': 1, 'host': profile['hostname'], 'job_id': PREFIX + name,
            'capture_version': commit, 'cwd': release, 'env': env, 'timeout_seconds': timeout, 'inputs': files + list(inputs),
            'argv': [config['baseline']['python'], '-B', '-m', 'lc_rec.shuffle_v2.' + module,
                     '--output', f'{ROOT}/{name}', *extra]}


def immutable(path, payload):
    if path.exists():
        assert json.loads(path.read_text()) == payload, path
    else:
        control.transport.atomic(path, payload)


def submit(connection, payload):
    path = SUPPORT / (payload['job_id'] + '.request.json')
    immutable(path, payload)
    return control.gridctl.submit(connection, path)


def require_success(states, name):
    row = states[PREFIX + name]
    assert row['status'] == 'SUCCEEDED' and row['exit_code'] == 0 and row['cleanup']['verified'], row


def audit_gates(connection, states, replay):
    require_success(states, replay)
    directory = f'{ROOT}/{replay}'
    paths = [f'{directory}/{name}' for name in ('audit.json', 'inputs.jsonl', 'assets.json', 'single-shard.json')]
    return ['--cpu-audit', paths[0], '--single-shard-audit', paths[3]], [
        {'path': path, 'sha256': control.remote_hash(connection, path)} for path in paths]


def series(connection, config, commit, states, args, *, build_only=False):
    assert args.review is not None
    require_success(states, args.preflight)
    require_success(states, args.normal)
    review = json.loads(args.review.read_text())
    assert review['status'] == 'PASS' and review['RUN_COMMIT'] == commit
    assert review['runner_sha256'] == connection.profile['runner_sha256']
    assert review['preflight_job_id'] == PREFIX + args.preflight
    preflight_summary = f'{ROOT}/{args.preflight}/summary.json'
    preflight_sha = control.remote_hash(connection, preflight_summary)
    assert preflight_sha == review['preflight_summary_sha256']
    parent = f'group-series-{args.attempt}'
    assert not any(name.startswith(PREFIX + 'group-series-') and row.get('ended_at') is None
                   for name, row in states.items()), 'An existing Beauty series still owns this task'
    local = SUPPORT / (PREFIX + parent + '.plan.json')
    profile = connection.profile
    if local.is_file() and not build_only:
        plan = json.loads(local.read_text())
        assert plan['formal_run_commit'] == commit and plan['preflight_review'] == review
        assert plan['runner'] == profile['runner'] and plan['runner_sha256'] == profile['runner_sha256']
        gate_path = f'{control.BASE}/requests/{PREFIX}{parent}.review.json'
        assert control.remote_hash(connection, gate_path) == hashlib.sha256(args.review.read_bytes()).hexdigest()
        caps = config['budget']
        timeout_by_stage = {
            stage: sum(item['timeout_seconds'] for phase in plan['modes'].values()
                       for key, item in phase.items() if key == stage and item['job_id'] not in states)
            for stage in ('gpu', 'cpu')}
        if plan['paired']['job_id'] not in states:
            timeout_by_stage['cpu'] += plan['paired']['timeout_seconds']
        assert control.budget_seconds(states, 'gpu-formal-', caps['formal_total_device_seconds'], prefix=PREFIX) >= timeout_by_stage['gpu'] + 15
        assert control.budget_seconds(states, 'cpu-', caps['cpu_total_wall_seconds'], prefix=PREFIX) >= timeout_by_stage['cpu'] + 15
        pending_gpus = {phase['gpu']['env']['CUDA_VISIBLE_DEVICES'] for phase in plan['modes'].values()
                        if phase['gpu']['job_id'] not in states}
        if pending_gpus:
            assert len(pending_gpus) == 1
            expected_gpu = next(iter(pending_gpus))
            assert device(connection, config, expected_gpu)[0] == expected_gpu
        return submit_plan(connection, config, commit, parent, local, plan['controller_timeout_seconds'])
    selected, snapshot = device(connection, config, review['device_uuid'])
    gates, files = audit_gates(connection, states, args.replay)
    files.append({'path': preflight_summary, 'sha256': preflight_sha})
    review_remote = f'{control.BASE}/requests/{PREFIX}{parent}.review.json'
    review_input = {'path': review_remote, 'sha256': hashlib.sha256(args.review.read_bytes()).hexdigest()}
    files.append(review_input)
    release, _ = deployment(commit)
    config_path = f'{release}/{CONFIG_PATH}'
    caps = config['budget']
    aggregate = min(caps['formal_total_wall_seconds'], caps['formal_total_device_seconds'])
    estimated = review['estimate_seconds']
    assert list(estimated) == list(core.MODES) and all(0 < n < caps['formal_attempt_wall_seconds'] for n in estimated.values())
    phases = {}
    completion = 0
    gpu_planned = 0
    cpu_planned = CPU_LIMITS['statistics']
    for mode in core.MODES:
        conditions = {}
        attempt = args.attempt
        for kind, stem, category in (('gpu', f'gpu-formal-{mode}', 'generate'),
                                     ('cpu', f'cpu-eval-{mode}', 'evaluate')):
            accepted = [name for name, state in states.items()
                        if name.startswith(f'{PREFIX}{stem}-') and state['status'] == 'SUCCEEDED']
            if accepted:
                assert len(accepted) == 1
                name = accepted[0]
                require_success(states, name.removeprefix(PREFIX))
                prior = SUPPORT / (name + '.request.json')
                if prior.exists():
                    payload = json.loads(prior.read_text())
                else:
                    remote_request = f"{connection.profile['job_root']}/{name}/request.json"
                    payload = json.loads(control.remote_python(connection, f'from pathlib import Path; print(Path({remote_request!r}).read_text())'))
                    immutable(prior, payload)
                assert payload['capture_version'] == commit
            else:
                name = f'{stem}-{attempt}'
                common = ['--config', config_path, *gates, '--mode', mode, '--run-commit', commit]
                if kind == 'gpu':
                    limit = caps['formal_attempt_wall_seconds'] - 30
                    gpu_planned += limit
                    payload = request(connection, config, commit, name, category, limit,
                                      [*common, '--seconds', str(limit - 55)], files, selected)
                else:
                    limit = CPU_LIMITS['evaluate']
                    cpu_planned += limit
                    test = conditions['gpu']['argv'][conditions['gpu']['argv'].index('--output') + 1] + '/test'
                    payload = request(connection, config, commit, name, 'beauty_evaluate', limit,
                                      [*common, '--evaluation-commit', commit, '--test-dir', test], files)
            conditions[kind] = payload
        completion += sum(1 for payload in conditions.values() if payload['job_id'] in states)
        phases[mode] = conditions
    assert control.budget_seconds(states, 'gpu-formal-', aggregate, prefix=PREFIX) >= gpu_planned + 15
    assert control.budget_seconds(states, 'cpu-', caps['cpu_total_wall_seconds'], prefix=PREFIX) >= cpu_planned + 15
    folders = [f'{ROOT}/{args.normal}'] + [p['cpu']['argv'][p['cpu']['argv'].index('--output') + 1] for p in phases.values()]
    paired = request(connection, config, commit, f'cpu-summary-{args.attempt}', 'statistics', CPU_LIMITS['statistics'],
                     ['--config', config_path, *folders])
    deadline = gpu_planned + cpu_planned + 1800
    plan = {'formal_run_commit': commit, 'orchestrator_commit': commit, 'modes': phases, 'paired': paired,
            'normal_job_id': PREFIX + args.normal, 'preflight_review': review, 'gpu_selection': snapshot,
            'prefix': PREFIX, 'runner': profile['runner'], 'runner_sha256': profile['runner_sha256'],
            'job_root': profile['job_root'], 'requests_root': control.BASE + '/requests', 'config': config_path,
            'adopted_completed_attempts': completion, 'controller_timeout_seconds': deadline}
    if build_only:
        assert args.plan_output is not None
        plan_path = f'{control.BASE}/requests/{local.name}'
        plan_sha = hashlib.sha256((json.dumps(plan, ensure_ascii=False, indent=2) + '\n').encode()).hexdigest()
        parent_request = request(connection, config, commit, parent, 'series', deadline,
                                 ['--plan', plan_path], [{'path': plan_path, 'sha256': plan_sha}])
        assert all(payload['env']['CUDA_VISIBLE_DEVICES'] == selected and
                   all(payload['env'][key] == value for key, value in config['gpu_environment'].items())
                   for phase in phases.values() for payload in (phase['gpu'],)
                   if payload['job_id'] not in states)
        result = {'status': 'FORMAL_PLAN_VALIDATED_NOT_SUBMITTED', 'formal_submitted': False,
                  'formal_gpu_device_seconds_delta': 0, 'plan': plan, 'parent_request': parent_request,
                  'launch_gate_upload': review_input, 'canonical_plan_sha256': plan_sha,
                  'plan_builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'future_requires_authorization_and_resource_recheck': True}
        immutable(args.plan_output, result)
        return result
    receipt = connection.put(args.review, review_remote)
    assert {'path': receipt['path'], 'sha256': receipt['sha256']} == review_input
    immutable(local, plan)
    return submit_plan(connection, config, commit, parent, local, deadline)


def submit_plan(connection, config, commit, parent, local, deadline):
    sent = connection.put(local, f'{control.BASE}/requests/{local.name}')
    payload = request(connection, config, commit, parent, 'series', deadline,
                      ['--plan', sent['path']], [{'path': sent['path'], 'sha256': sent['sha256']}])
    return submit(connection, payload)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('action', choices=('deploy', 'replay', 'normal', 'preflight', 'plan', 'series', 'status', 'pull'))
    parser.add_argument('--commit', default='HEAD')
    parser.add_argument('--attempt', default='r0', type=attempt_id)
    parser.add_argument('--replay', default='cpu-replay-r3')
    parser.add_argument('--normal', default='cpu-eval-normal-r0')
    parser.add_argument('--preflight', default='gpu-preflight-r1')
    parser.add_argument('--review', type=Path)
    parser.add_argument('--relative')
    parser.add_argument('--plan-output', type=Path)
    parser.add_argument('--confirm-formal-launch', action='store_true')
    args = parser.parse_args()
    SUPPORT.mkdir(parents=True, exist_ok=True)
    if args.action == 'series' and not args.confirm_formal_launch:
        parser.error('series submits formal jobs and requires future explicit launch authorization')
    if args.action not in ('status', 'pull'):
        commit = commit_ref(args.commit)
        connection = connect(runner_sha_at(commit))
    else:
        connection = connect()
    if args.action == 'pull':
        relative = Path(args.relative)
        assert relative.parts and not relative.is_absolute() and '..' not in relative.parts
        path = f'{ROOT}/{relative}'
        meta = json.loads(control.remote_python(connection, f'''from pathlib import Path
import hashlib,json
p=Path({path!r})
with p.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
print(json.dumps(dict(path=str(p),bytes=p.stat().st_size,sha256=digest)))''', timeout=120))
        destination = SUPPORT / 'evidence' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        result = connection.get_verified(path, destination, meta)
    elif args.action == 'status':
        result = control.job_states(connection, PREFIX)
    else:
        if args.action == 'deploy':
            deploy_runner(connection, commit)
            result = control.deploy(connection, commit, namespace=NAMESPACE, support=SUPPORT, paths=SOURCE_PATHS)
        else:
            config = json.loads(beauty.CONFIG.read_text())
            release, _ = deployment(commit)
            config_path = f'{release}/{CONFIG_PATH}'
            caps = config['budget']
            name = {'replay': 'cpu-replay', 'normal': 'cpu-eval-normal', 'preflight': 'gpu-preflight',
                    'series': 'group-series', 'plan': 'group-series'}[args.action] + '-' + args.attempt
            with (SUPPORT / 'submit.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                states = control.job_states(connection, PREFIX)
                if args.action == 'plan':
                    result = series(connection, config, commit, states, args, build_only=True)
                elif PREFIX + name in states:
                    result = control.gridctl.query(connection, PREFIX + name)
                elif args.action == 'series':
                    result = series(connection, config, commit, states, args)
                elif (SUPPORT / (PREFIX + name + '.request.json')).exists():
                    pending = json.loads((SUPPORT / (PREFIX + name + '.request.json')).read_text())
                    assert pending['capture_version'] == commit
                    if args.action == 'preflight':
                        expected = pending['env']['CUDA_VISIBLE_DEVICES']
                        assert wait_for_device(connection, config, expected)[0] == expected
                        refreshed = control.job_states(connection, PREFIX)
                        if PREFIX + name in refreshed:
                            print(json.dumps(control.gridctl.query(connection, PREFIX + name), indent=2))
                            return
                        assert pending['timeout_seconds'] <= caps['gpu_validation_attempt_seconds']
                        assert pending['timeout_seconds'] + 15 <= control.budget_seconds(
                            refreshed, 'gpu-preflight-', caps['preflight_device_seconds'], prefix=PREFIX)
                        assert all(pending['env'][k] == v for k, v in config['gpu_environment'].items())
                        _, inputs = audit_gates(connection, refreshed, args.replay)
                        assert all(item in pending['inputs'] for item in inputs)
                    result = submit(connection, pending)
                elif args.action == 'preflight':
                    gpu, _ = wait_for_device(connection, config)
                    states = control.job_states(connection, PREFIX)
                    assert PREFIX + name not in states
                    gate_args, inputs = audit_gates(connection, states, args.replay)
                    timeout = min(caps['gpu_validation_attempt_seconds'],
                                  control.budget_seconds(states, 'gpu-preflight-', caps['preflight_device_seconds'], prefix=PREFIX) - 30)
                    assert timeout > 90
                    result = submit(connection, request(connection, config, commit, name, 'generate', timeout,
                                    ['--config', config_path, *gate_args, '--mode', 'preflight',
                                     '--run-commit', commit, '--seconds', str(timeout - 55)], inputs, gpu))
                else:
                    evaluate = args.action == 'normal'
                    module = 'beauty_evaluate' if evaluate else 'replay'
                    timeout = CPU_LIMITS['evaluate' if evaluate else 'replay']
                    assert control.budget_seconds(states, 'cpu-', caps['cpu_total_wall_seconds'], prefix=PREFIX) >= timeout + 15
                    argv = ['--config', config_path]
                    if evaluate:
                        argv += ['--mode', 'normal', '--run-commit', config['baseline']['run-commit'], '--evaluation-commit', commit]
                    result = submit(connection, request(connection, config, commit, name, module, timeout, argv))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
