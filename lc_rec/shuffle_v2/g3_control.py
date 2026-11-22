"""Deploy G3 V2 and submit preparation jobs. Formal plans are local-only."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time

from lc_rec.shuffle_v2 import beauty_control, control, core, g3_assets

NAMESPACE = 'g3-beauty-v2'
PREFIX = NAMESPACE + '-'
SUPPORT = Path('/home/addaswsw/recommend-support') / NAMESPACE
ROOT = '/data/bwwu/lc-rec/' + NAMESPACE
RUNNER = 'lc_rec/shuffle_v2/beauty_runner.py'
RUNNER_SHA = 'cbb963fab68c34cc977653afc46f260e7ede962cb4a8f105458331a0e0e19a75'
CLOSURE = 'lc_rec/a0_beauty/grid_sid_preparation/g3_conversion/production_preparation/cpu_closure'
SOURCES = ('lc_rec/shuffle_v2', 'lc_rec/shuffle/core.py', 'lc_rec/letter/scripts/formal_evaluate.py',
           'lc_rec/a0_beauty/offline_metrics.py', 'lc_rec/a0_beauty/g3_identity.py',
           'lc_rec/a0_beauty/uneven_sampler.py', 'lc_rec/a0_beauty/config/preflight_g3_users_seed42.json',
           CLOSURE,
           'lc_rec/letter/review/final_audit_20261113/external/official_cce.py',
           'lc_rec/letter/review/final_audit_20261113/statistics_verify.py')
PREFLIGHT_MODE_TIMEOUT = 1100


def runner_source():
    frozen = subprocess.check_output(['git', 'show', '6212bb302ff4f8db87232ca1e7993f411961eac9:' + RUNNER],
                                     cwd=control.REPO)
    assert hashlib.sha256(frozen).hexdigest() == RUNNER_SHA
    original = b"BEAUTY_PREFIX = 'letter-beauty-v2-'"
    assert frozen.count(original) == 1
    return frozen.replace(original, f"BEAUTY_PREFIX = '{PREFIX}'".encode())


def connection():
    raw = runner_source()
    digest = hashlib.sha256(raw).hexdigest()
    target = control.connect()
    inherited = target.profile['runner']
    target.profile.update(runner=f'{control.BASE}/scripts/runner-{NAMESPACE}-{digest[:16]}.py',
                          runner_sha256=digest)
    return target, inherited, raw


def deploy(target, inherited, raw, commit):
    runner = target.profile['runner']
    source = SUPPORT / 'runner.py'
    if not source.exists():
        source.write_bytes(raw)
    assert source.read_bytes() == raw
    metadata = SUPPORT / 'runner.deployment.json'
    old = inherited + '.deployment.json'
    content = target.ssh('cat ' + old)
    if not metadata.exists():
        metadata.write_bytes(content.encode())
    assert g3_assets.sha(metadata) == target.profile['deployment_sha256']
    for local, destination in ((metadata, runner + '.deployment.json'), (source, runner)):
        wanted = g3_assets.sha(local)
        receipt = target.put(local, destination)
        assert receipt['sha256'] == wanted
    return control.deploy(target, commit, namespace=NAMESPACE, support=SUPPORT, paths=SOURCES)


def release_files(commit):
    record = g3_assets.read(SUPPORT / f'{commit}.deployment.json')
    assert record['RUN_COMMIT'] == commit
    release = record['release']
    return release, [{'path': release + '/' + row['path'], 'sha256': row['sha256']}
                     for row in record['files']]


def request(target, config, commit, name, timeout, module, args, *, card='', inputs=()):
    release, files = release_files(commit)
    identity = {key: target.profile[key] for key in ('user', 'uid', 'namespace', 'deployment_sha256')}
    argv = [config['python'], '-B', '-m']
    if card:
        assert card in config['gpus']
        argv += ['torch.distributed.run', '--standalone', '--nnodes=1', '--nproc_per_node=1', '--module']
    argv += ['lc_rec.shuffle_v2.' + module, *map(str, args)]
    env = {**config['environment'], 'CUDA_VISIBLE_DEVICES': card}
    if not card:
        env['DS_ACCELERATOR'] = 'cpu'
    return {**identity, 'schema': 1, 'host': target.profile['hostname'], 'job_id': PREFIX + name,
            'capture_version': commit, 'cwd': release,
            'env': env,
            'timeout_seconds': timeout, 'inputs': files + list(inputs), 'argv': argv}


def launch(target, payload):
    name = payload['job_id'].removeprefix(PREFIX)
    assert (name.startswith('cpu-') and 'formal' not in name) or name.startswith('gpu-preflight-')
    lock = SUPPORT / 'submission.lock'
    with lock.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        states = control.job_states(target, PREFIX)
        if payload['job_id'] in states:
            return control.gridctl.query(target, payload['job_id'])
        config = g3_assets.read(g3_assets.CONFIG)
        is_gpu = bool(payload['env']['CUDA_VISIBLE_DEVICES'])
        kind, cap = ('gpu-preflight-', 'preflight_device_seconds') if is_gpu else ('cpu-', 'cpu_wall_seconds')
        finished = {key: value for key, value in states.items() if value.get('ended_at')}
        active = [key for key, value in states.items() if key.startswith(PREFIX + kind) and not value.get('ended_at')]
        reserved = sum(g3_assets.read(SUPPORT / (key + '.request.json'))['timeout_seconds'] + 10 for key in active)
        remaining = config['budget'][cap] - usage(finished, kind) - reserved
        assert remaining >= payload['timeout_seconds'] + 10, (name, remaining)
        if is_gpu:
            assert not any(not val.get('ended_at') for key, val in states.items() if key.startswith(PREFIX + kind))
            pinned = payload['env']['CUDA_VISIBLE_DEVICES']
            selected, snapshot = beauty_control.device(target, config, pinned)
            assert selected == pinned
            beauty_control.immutable(SUPPORT / (payload['job_id'] + '.gpu-selection.json'), snapshot)
        path = SUPPORT / (payload['job_id'] + '.request.json')
        beauty_control.immutable(path, payload)
        return control.gridctl.submit(target, path)


def usage(states, kind):
    total = 0.0
    for name, row in states.items():
        if not name.startswith(PREFIX + kind):
            continue
        start = row.get('started_at')
        if row.get('ended_at'):
            if start:
                total += (datetime.fromisoformat(row['ended_at']) - datetime.fromisoformat(start)).total_seconds()
        elif start:
            total += max(0, (datetime.now(datetime.fromisoformat(start).tzinfo) - datetime.fromisoformat(start)).total_seconds())
    return total


def wait_terminal(target, name, timeout=1800):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        states = control.job_states(target, PREFIX)
        row = states.get(PREFIX + name, {})
        if row.get('ended_at'):
            return row
        time.sleep(5)
    raise TimeoutError('Query the existing job after a status timeout: ' + PREFIX + name)



def require_success(target, name):
    state = control.gridctl.query(target, PREFIX + name)
    assert state['status'] == 'SUCCEEDED' and state['exit_code'] == 0, state
    assert state['cleanup_verified'] and not state['workgroup_live'] and not state['worker_alive'], state
    return state


def cpu_gates(target, replay):
    require_success(target, replay)
    folder = f'{ROOT}/{replay}'
    files = [{'path': f'{folder}/{name}', 'sha256': control.remote_hash(target, f'{folder}/{name}')}
             for name in ('audit.json', 'inputs.jsonl')]
    return files


def generation_request(target, config, commit, scope, mode, attempt, card, cpu_files, authorization=None):
    assert scope in ('PREFLIGHT', 'FORMAL') and mode in core.MODES
    timeout = PREFLIGHT_MODE_TIMEOUT if scope == 'PREFLIGHT' else config['budget']['formal_condition_seconds'] - 15
    name = f'gpu-{scope.lower()}-{mode}-{attempt}'
    argv = ['--output', f'{ROOT}/{name}', '--cpu-audit', cpu_files[0]['path'], '--cpu-audit-sha256',
            cpu_files[0]['sha256'], '--run-commit', commit, '--mode', mode, '--scope', scope,
            '--max-seconds', str(timeout - 10)]
    files = list(cpu_files)
    if scope == 'FORMAL':
        assert authorization
        argv += ['--authorization', authorization['path'], '--authorization-sha256', authorization['sha256']]
        files.append(authorization)
    return request(target, config, commit, name, timeout, 'g3_generate', argv, card=card, inputs=files)


def evaluation_request(target, config, commit, scope, mode, attempt, cpu_files, dependency_files=()):
    phase = 'formal-' if scope == 'FORMAL' else ''
    name = f'cpu-{phase}eval-{mode}-{attempt}'
    test = f'{ROOT}/gpu-{scope.lower()}-{mode}-{attempt}'
    args = ['condition', '--data-root', config['data_root'], '--test-dir', test,
            '--cpu-audit', cpu_files[0]['path'], '--mode', mode, '--scope', scope,
            '--run-commit', commit, '--output', f'{ROOT}/{name}']
    return request(target, config, commit, name, 1200, 'g3_evaluate', args,
                   inputs=[*cpu_files, *dependency_files])


def paired_request(target, config, commit, scope, attempt, normal, dependency_files=()):
    phase = 'formal-' if scope == 'FORMAL' else ''
    name = f'cpu-{phase}paired-{attempt}'
    args = ['paired', '--scope', scope, '--normal', f'{ROOT}/{normal}', '--condition-roots',
            *(f'{ROOT}/cpu-{phase}eval-{mode}-{attempt}' for mode in core.MODES),
            '--output', f'{ROOT}/{name}']
    return request(target, config, commit, name, 2400, 'g3_evaluate', args, inputs=dependency_files)


def preflight(target, config, commit, attempt, replay):
    files = cpu_gates(target, replay)
    existing = sorted(SUPPORT.glob(f'{PREFIX}gpu-preflight-*-{attempt}.request.json'))
    if existing:
        cards = {g3_assets.read(path)['env']['CUDA_VISIBLE_DEVICES'] for path in existing}
        card, = cards
        assert card in config['gpus']
    else:
        card, _ = beauty_control.device(target, config)
    results = {}
    for mode in core.MODES:
        payload = generation_request(target, config, commit, 'PREFLIGHT', mode, attempt, card, files)
        name = payload['job_id'].removeprefix(PREFIX)
        launch(target, payload)
        state = wait_terminal(target, name)
        print(json.dumps({'job_id': state['job_id'], 'status': state['status'],
                          'exit_code': state['exit_code']}), flush=True)
        require_success(target, name)
        results[mode] = state['job_id']
    return results


def assess(target, config, commit, gpu_commit, attempt, replay, normal):
    files = cpu_gates(target, replay)
    require_success(target, normal)
    for mode in core.MODES:
        gpu = f'gpu-preflight-{mode}-{attempt}'
        require_success(target, gpu)
        test = f'{ROOT}/{gpu}'
        inputs = [{'path': f'{test}/{file}', 'sha256': control.remote_hash(target, f'{test}/{file}')}
                  for file in ('rank0.json', 'v2-binding.rank0.json', 'metrics.json', 'predictions.rank0.jsonl')]
        payload = evaluation_request(target, config, commit, 'PREFLIGHT', mode, attempt, files, inputs)
        payload['argv'][payload['argv'].index('--run-commit') + 1] = gpu_commit
        launch(target, payload)
    for mode in core.MODES:
        name = f'cpu-eval-{mode}-{attempt}'
        wait_terminal(target, name)
        require_success(target, name)
    payload = paired_request(target, config, commit, 'PREFLIGHT', attempt, normal)
    launch(target, payload)
    name = payload['job_id'].removeprefix(PREFIX)
    wait_terminal(target, name)
    require_success(target, name)
    return f'{ROOT}/{name}'

def boot(target, config, commit, action, attempt):
    if action == 'replay':
        name = 'cpu-replay-' + attempt
        return launch(target, request(target, config, commit, name, 1200, 'g3_replay',
                                      ['--output', f'{ROOT}/{name}']))
    if action == 'normal':
        name = 'cpu-normal-' + attempt
        return launch(target, request(target, config, commit, name, 900, 'g3_evaluate',
                                      ['normal', '--data-root', config['data_root'], '--output', f'{ROOT}/{name}']))
    if action == 'lifecycle':
        terminal = {}
        for case, timeout in (('decoy', 45), ('normal', 10), ('orphan', 12), ('timeout', 8)):
            name = f'cpu-lifecycle-{case}-{attempt}'
            launch(target, request(target, config, commit, name, timeout, 'beauty_lifecycle',
                                  ['--case', case, '--output', f'{ROOT}/{name}', '--job-prefix', PREFIX]))
            if case != 'decoy':
                terminal[case] = wait_terminal(target, name, 70)['status']
        terminal['decoy'] = wait_terminal(target, f'cpu-lifecycle-decoy-{attempt}', 70)['status']
        assert terminal == {'normal': 'SUCCEEDED', 'orphan': 'WORKERS_TERMINATED', 'timeout': 'TIMED_OUT', 'decoy': 'SUCCEEDED'}, terminal
        name = 'cpu-lifecycle-review-' + attempt
        launch(target, request(target, config, commit, name, 20, 'beauty_lifecycle',
                               ['--output', f'{ROOT}/{name}.json', '--job-root', target.profile['job_root'],
                                '--output-root', ROOT, '--attempt', attempt, '--job-prefix', PREFIX,
                                '--runner-sha', target.profile['runner_sha256']]))
        return wait_terminal(target, name, 70)['status']
    raise ValueError(action)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('action', choices=('deploy', 'replay', 'normal', 'lifecycle', 'preflight', 'assess', 'states'))
    parser.add_argument('--commit', default='HEAD')
    parser.add_argument('--attempt', default='r0', type=beauty_control.attempt_id)
    parser.add_argument('--replay', default='cpu-replay-r1')
    parser.add_argument('--normal', default='cpu-normal-r0')
    parser.add_argument('--gpu-commit')
    options = parser.parse_args()
    SUPPORT.mkdir(parents=True, exist_ok=True)
    target, inherited, raw = connection()
    config = g3_assets.read(g3_assets.CONFIG)
    commit = beauty_control.commit_ref(options.commit)
    if options.action == 'deploy':
        result = deploy(target, inherited, raw, commit)
    elif options.action == 'states':
        result = control.job_states(target, PREFIX)
    elif options.action == 'preflight':
        result = preflight(target, config, commit, options.attempt, options.replay)
    elif options.action == 'assess':
        result = assess(target, config, commit, options.gpu_commit or commit, options.attempt, options.replay, options.normal)
    else:
        result = boot(target, config, commit, options.action, options.attempt)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
