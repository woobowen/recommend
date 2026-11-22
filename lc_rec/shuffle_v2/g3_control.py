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
           'lc_rec/a0_beauty/offline_metrics.py', CLOSURE,
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
    return {**identity, 'schema': 1, 'host': target.profile['hostname'], 'job_id': PREFIX + name,
            'capture_version': commit, 'cwd': release,
            'env': {**config['environment'], 'CUDA_VISIBLE_DEVICES': card},
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
        remaining = config['budget'][cap] - usage(states, kind)
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
            assert row['cleanup']['verified']
            return row
        time.sleep(2)
    raise TimeoutError('Query the existing job after a status timeout: ' + PREFIX + name)


def boot(target, config, commit, action, attempt):
    if action == 'replay':
        name = 'cpu-replay-' + attempt
        return launch(target, request(target, config, commit, name, 1200, 'g3_replay',
                                      ['--output', f'{ROOT}/{name}']))
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
    parser.add_argument('action', choices=('deploy', 'replay', 'lifecycle', 'states'))
    parser.add_argument('--commit', default='HEAD')
    parser.add_argument('--attempt', default='r0', type=beauty_control.attempt_id)
    options = parser.parse_args()
    SUPPORT.mkdir(parents=True, exist_ok=True)
    target, inherited, raw = connection()
    config = g3_assets.read(g3_assets.CONFIG)
    commit = beauty_control.commit_ref(options.commit)
    if options.action == 'deploy':
        result = deploy(target, inherited, raw, commit)
    elif options.action == 'states':
        result = control.job_states(target, PREFIX)
    else:
        result = boot(target, config, commit, options.action, options.attempt)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
