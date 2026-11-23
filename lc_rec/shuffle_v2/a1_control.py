"""Deploy and submit bounded preparation jobs for A1 Instruments V2."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess

from lc_rec.shuffle_v2 import beauty_control, control, core

NAMESPACE = 'a1-instruments-v2'
PREFIX = NAMESPACE + '-'
ROOT = '/data/bwwu/lc-rec/' + NAMESPACE
SUPPORT = Path('/home/addaswsw/recommend-support') / NAMESPACE
CONFIG = 'lc_rec/shuffle_v2/a1_instruments.json'
RUNNER_SOURCE = 'lc_rec/shuffle_v2/beauty_runner.py'
RUNNER_BASE = '6212bb302ff4f8db87232ca1e7993f411961eac9'
RUNNER_BASE_HASH = 'cbb963fab68c34cc977653afc46f260e7ede962cb4a8f105458331a0e0e19a75'
CORE_FILES = (
    'lc_rec/shuffle/core.py', 'lc_rec/shuffle/entry.py', 'lc_rec/shuffle/paired_and_changes.py',
    'lc_rec/shuffle/evidence/cpu-bound-r0/a1/audit.json', 'lc_rec/shuffle/evidence/preflight-r1/a1/rank0.json',
    'lc_rec/shuffle/evidence/preflight-r1/a1/rank1.json',
    'lc_rec/a1_qwen3_06b/scripts/evaluate.py', 'lc_rec/a1_qwen3_06b/scripts/offline_metrics.py',
    'lc_rec/qwen_preflight_20261110/scripts/common.py',
    'lc_rec/letter/config/a0_evaluation_v1.json', 'lc_rec/letter/scripts/formal_evaluate.py',
    'lc_rec/letter/scripts/offline_metrics.py',
    'lc_rec/letter/review/final_audit_20261113/external/official_cce.py',
    'lc_rec/letter/review/final_audit_20261113/statistics_verify.py',
)
MODULES = ('a1_generate', 'a1_replay', 'a1_evaluate', 'a1_tasks',
           'core', 'reference', 'runtime', 'replay', 'generate', 'statistics')
STAGES = {
    'cpu-replay': 1800, 'cpu-normal': 900, 'gpu-preflight': 2200,
    'cpu-assess': 900, 'cpu-lifecycle-clean': 30, 'cpu-lifecycle-timeout': 8, 'cpu-export': 1800,
}


def read(path):
    return json.loads(Path(path).read_text())


def connection():
    old = subprocess.check_output(['git', 'show', RUNNER_BASE + ':' + RUNNER_SOURCE], cwd=control.REPO)
    assert hashlib.sha256(old).hexdigest() == RUNNER_BASE_HASH
    needle = b"BEAUTY_PREFIX = 'letter-beauty-v2-'"
    assert old.count(needle) == 1
    scoped = old.replace(needle, f"BEAUTY_PREFIX = '{PREFIX}'".encode())
    sha = hashlib.sha256(scoped).hexdigest()
    target = control.connect()
    ancestor = target.profile['runner']
    target.profile.update(runner=f'{control.BASE}/scripts/runner-{NAMESPACE}-{sha[:16]}.py', runner_sha256=sha)
    return target, ancestor, scoped


def deploy(commit):
    target, ancestor, scoped = connection()
    SUPPORT.mkdir(parents=True, exist_ok=True)
    runner = SUPPORT / 'runner.py'
    metadata = SUPPORT / 'runner.deployment.json'
    if not runner.exists():
        runner.write_bytes(scoped)
    assert runner.read_bytes() == scoped
    if not metadata.exists():
        metadata.write_bytes(target.ssh('cat ' + ancestor + '.deployment.json').encode())
    assert control.transport.sha(metadata) == target.profile['deployment_sha256']
    destination = target.profile['runner']
    for source, path in ((metadata, destination + '.deployment.json'), (runner, destination)):
        assert target.put(source, path)['sha256'] == control.transport.sha(source)
    files = (*CORE_FILES, CONFIG, *(f'lc_rec/shuffle_v2/{m}.py' for m in MODULES))
    available = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', commit, *files],
                                         cwd=control.REPO, text=True).splitlines()
    assert set(files) - set(available) <= {'lc_rec/shuffle_v2/a1_evaluate.py'}
    return control.deploy(target, commit, namespace=NAMESPACE, support=SUPPORT, paths=available)


def payload(target, commit, config, name, module, argv, timeout, *, card='', input_files=()):
    deploy_receipt = read(SUPPORT / (commit + '.deployment.json'))
    assert deploy_receipt['RUN_COMMIT'] == commit
    release = deploy_receipt['release']
    files = [{'path': release + '/' + row['path'], 'sha256': row['sha256']} for row in deploy_receipt['files']]
    fields = {key: target.profile[key] for key in ('user', 'uid', 'namespace', 'deployment_sha256')}
    env = {**config['gpu_environment'], 'CUDA_VISIBLE_DEVICES': card} if card else dict(config['cpu_environment'])
    return {**fields, 'schema': 1, 'host': target.profile['hostname'], 'job_id': PREFIX + name,
            'capture_version': commit, 'cwd': release, 'env': env, 'timeout_seconds': timeout,
            'inputs': files + list(input_files),
            'argv': [config['baseline']['python'], '-B', '-m', 'lc_rec.shuffle_v2.' + module, *argv]}


def outcome(state):
    return bool(state.get('status') == 'SUCCEEDED' and state.get('exit_code') == 0
                and state.get('cleanup', {}).get('verified'))


def successful_path(states, stage, attempt):
    assert re.fullmatch(r'r[0-3]', attempt)
    state = states[PREFIX + stage + '-' + attempt]
    assert outcome(state), (stage, attempt, state.get('status'))
    return f'{ROOT}/{stage}-{attempt}'


def bound_inputs(target, root, filenames):
    return [{'path': root + '/' + name, 'sha256': control.remote_hash(target, root + '/' + name)} for name in filenames]


def usage(states, kind):
    seconds = 0.0
    for name, state in states.items():
        if not name.startswith(PREFIX + kind + '-'):
            continue
        assert state.get('ended_at'), (name, 'nonterminal reserved request')
        if state.get('started_at'):
            elapsed = datetime.fromisoformat(state['ended_at']) - datetime.fromisoformat(state['started_at'])
            seconds += elapsed.total_seconds()
    return seconds


def launch(target, config, args):
    name = args.stage + '-' + args.attempt
    assert args.stage in STAGES and re.fullmatch(r'r[0-3]', args.attempt)
    out = f'{ROOT}/{name}'
    with (SUPPORT / 'submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        states = control.job_states(target, PREFIX)
        if PREFIX + name in states:
            return control.gridctl.query(target, PREFIX + name)
        is_gpu = args.stage == 'gpu-preflight'
        kind = 'gpu-preflight' if is_gpu else 'cpu'
        spent = usage(states, kind)
        cap = config['budget']['preflight_device_seconds' if is_gpu else 'cpu_worker_seconds']
        ttl = STAGES[args.stage]
        assert spent + ttl + 10 <= cap, (kind, cap, spent, ttl)
        commit = args.commit
        release = read(SUPPORT / (commit + '.deployment.json'))['release']
        conf = release + '/' + CONFIG
        card, inputs = '', []
        if args.stage.startswith('cpu-lifecycle-'):
            module = 'a1_tasks'
            argv = ['lifecycle', '--case', args.stage.split('-')[-1], '--output', out]
        elif args.stage == 'cpu-replay':
            module = 'a1_replay'
            argv = ['--config', conf, '--output', out]
        else:
            replay = successful_path(states, 'cpu-replay', args.replay)
            inputs = bound_inputs(target, replay, ('audit.json', 'assets.json', 'inputs.jsonl', 'single-shard.json',
                                                   'normal-sample.jsonl'))
            if is_gpu:
                module = 'a1_tasks'
                card, snapshot = beauty_control.device(target, config, args.card)
                beauty_control.immutable(SUPPORT / (PREFIX + name + '.gpu.json'), snapshot)
                argv = ['gpu', '--config', conf, '--cpu-manifest', replay, '--output', out,
                        '--mode', 'preflight', '--run-commit', commit, '--seconds', str(ttl - 30)]
            elif args.stage == 'cpu-normal':
                module = 'a1_evaluate'
                argv = ['normal', '--config', conf, '--cpu-manifest', replay, '--output', out]
            elif args.stage == 'cpu-assess':
                normal = successful_path(states, 'cpu-normal', args.normal)
                preflight = successful_path(states, 'gpu-preflight', args.gpu)
                module = 'a1_evaluate'
                argv = ['assess', '--config', conf, '--cpu-manifest', replay, '--output', out,
                        '--normal', normal, '--preflight-dir', preflight]
                inputs += bound_inputs(target, preflight, ('summary.json', *(
                    f'{mode}/predictions.rank0.jsonl' for mode in core.MODES)))
            else:
                raise ValueError('Export does not use the submission interface')
        request = payload(target, commit, config, name, module, argv, ttl, card=card, input_files=inputs)
        path = SUPPORT / (PREFIX + name + '.request.json')
        beauty_control.immutable(path, request)
        return control.gridctl.submit(target, path)


def zero_formal(target):
    return json.loads(control.remote_python(target, f'''from pathlib import Path
import datetime,json
jobroot=Path({control.BASE!r})/'jobs'
requestroot=Path({control.BASE!r})/'requests'
outroot=Path({ROOT!r})
jobs=sorted(str(p) for p in jobroot.glob({(PREFIX+'formal-*')!r}))
requests=sorted(str(p) for p in requestroot.glob({(PREFIX+'formal-*')!r}))
accepted=sorted(str(p/'accepted.json') for p in map(Path,jobs) if (p/'accepted.json').is_file())
formal_outputs=sorted(str(p) for p in outroot.glob('formal-*'))
formal_predictions=sorted(str(q) for p in map(Path,formal_outputs) for q in p.rglob('predictions.rank*.jsonl'))
print(json.dumps(dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 host=__import__('socket').gethostname(), job_root=str(jobroot), request_root=str(requestroot),
 output_root=str(outroot), formal_parent_or_child_jobs=jobs, formal_requests=requests,
 formal_accepted=accepted, formal_outputs=formal_outputs, formal_predictions=formal_predictions,
 formal_gpu_device_seconds_delta=0 if not any((jobs,requests,accepted,formal_outputs,formal_predictions)) else None)))'''))


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    subs = parser.add_subparsers(dest='action', required=True)
    deploy_parser = subs.add_parser('deploy')
    deploy_parser.add_argument('commit')
    job = subs.add_parser('launch')
    job.add_argument('stage', choices=tuple(stage for stage in STAGES if stage != 'cpu-export'))
    job.add_argument('attempt', choices=('r0', 'r1', 'r2', 'r3'))
    job.add_argument('--commit', required=True)
    for key in ('replay', 'normal', 'gpu'):
        job.add_argument('--' + key, default='r0')
    job.add_argument('--card')
    status = subs.add_parser('query')
    status.add_argument('name')
    subs.add_parser('status')
    subs.add_parser('zero-formal')
    args = parser.parse_args()
    if args.action == 'deploy':
        result = deploy(args.commit)
    else:
        target, _, _ = connection()
        if args.action == 'launch':
            config = read(control.REPO / CONFIG)
            result = launch(target, config, args)
        elif args.action == 'query':
            result = control.gridctl.query(target, PREFIX + args.name)
        elif args.action == 'zero-formal':
            result = zero_formal(target)
        else:
            states = control.job_states(target, PREFIX)
            result = {k: {field: v.get(field) for field in ('status', 'exit_code', 'started_at', 'ended_at')}
                      | {'cleanup_verified': v.get('cleanup', {}).get('verified')} for k, v in states.items()}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
