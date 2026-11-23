"""Build, validate and (only with separate approval) run the A1 formal plan."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

from lc_rec.shuffle_v2 import core

APPROVAL_PLACEHOLDER = 'AWAITING_WEB_REVIEW_AND_SEPARATE_USER_AUTHORIZATION'
GROUP = 'a1-instruments-v2-'
USERS, CANDIDATES = 24772, 495440
MODE_LIMIT, DEVICE_LIMIT = 28800, 129600
CPU_MODE_LIMIT, CPU_PAIRED_LIMIT = 1800, 4500
PARENT_LIMIT = 136800
PARAMS = ('summary.json', 'primary/per_user_metrics.npz')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, exclusive=False):
    if exclusive:
        with Path(path).open('x') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    else:
        temporary = path.with_suffix('.partial')
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        temporary.replace(path)


def output(request):
    argv = request['argv']
    return Path(argv[argv.index('--output') + 1])


def validate(plan):
    assert plan['schema'] == 1 and plan['formal_submitted'] is False
    assert plan['formal_gpu_device_seconds_delta'] == 0
    assert (plan['users_per_mode'], plan['candidates_per_mode']) == (USERS, CANDIDATES)
    assert plan['approvals_required'] == ['web_review_pass', 'separate_user_formal_authorization']
    assert re.fullmatch(r'[0-9a-f]{64}', plan['runner_sha256'])
    assert plan['card'].startswith('GPU-') and isinstance(plan['release_inputs'], dict)
    assert len(plan['phases']) == 9
    names, destinations, last = set(), set(), None
    for number, phase in enumerate(plan['phases']):
        request = phase['request']
        mode = tuple(core.MODES)[number // 2] if number < 8 else 'paired'
        kind = ('gpu' if number % 2 == 0 else 'cpu') if number < 8 else 'paired'
        assert (phase['mode'], phase['kind'], phase['depends_on']) == (mode, kind, last)
        expected = f'{GROUP}formal-{kind}-{mode}-{plan["attempt"]}'
        assert request['job_id'] == expected and expected not in names
        assert output(request) not in destinations
        assert request['capture_version'] in plan['release_inputs']
        assert request['env']['CUDA_VISIBLE_DEVICES'] == (plan['card'] if kind == 'gpu' else '')
        assert request['timeout_seconds'] == {'gpu': MODE_LIMIT - 10, 'cpu': CPU_MODE_LIMIT,
                                              'paired': CPU_PAIRED_LIMIT}[kind]
        assert last is None or phase['predecessor_members']
        if kind == 'gpu':
            assert request['capture_version'] == plan['preflight_gpu_commit']
            assert request['argv'][request['argv'].index('--mode') + 1] == mode
        else:
            assert request['capture_version'] == plan['evaluation_and_controller_commit']
        names.add(expected)
        destinations.add(output(request))
        last = expected
    gpu_seconds = sum(x['request']['timeout_seconds'] + 10 for x in plan['phases'] if x['kind'] == 'gpu')
    assert gpu_seconds == len(core.MODES) * MODE_LIMIT <= DEVICE_LIMIT
    assert plan['parent']['job_id'] == f'{GROUP}formal-parent-{plan["attempt"]}'
    assert plan['parent']['timeout_seconds'] == PARENT_LIMIT
    assert plan['parent']['env']['CUDA_VISIBLE_DEVICES'] == ''
    assert output(plan['parent']) not in destinations
    assert APPROVAL_PLACEHOLDER in plan['parent']['argv']


def build(args):
    from lc_rec.shuffle_v2 import a1_control as ac

    target, _, _ = ac.connection()
    cfg = ac.read(ac.control.REPO / ac.CONFIG)
    states = ac.control.job_states(target, ac.PREFIX)
    assert args.card in cfg['gpus'] and re.fullmatch(r'r[0-3]', args.attempt)
    prereq = [('cpu-replay', args.replay), ('cpu-normal', args.normal),
              ('gpu-preflight', args.gpu), ('cpu-assess', args.assess), ('cpu-lifecycle-clean', args.lifecycle)]
    outputs = {stage: ac.successful_path(states, stage, attempt) for stage, attempt in prereq}
    timeout_id = GROUP + 'cpu-lifecycle-timeout-' + args.lifecycle
    assert states[timeout_id]['status'] == 'TIMED_OUT' and states[timeout_id]['cleanup']['verified']
    zero = ac.zero_formal(target)
    assert zero['formal_gpu_device_seconds_delta'] == 0
    replay = outputs['cpu-replay']
    replay_inputs = ac.bound_inputs(target, replay, ('audit.json', 'assets.json', 'inputs.jsonl', 'single-shard.json',
                                                     'normal-sample.jsonl'))
    normal = outputs['cpu-normal']
    normal_inputs = ac.bound_inputs(target, normal, PARAMS)
    phases, release_inputs, evaluation_roots = [], {}, []

    def request(commit, name, module, tail, timeout, **kwargs):
        p = ac.payload(target, commit, cfg, 'formal-' + name + '-' + args.attempt, module, tail, timeout, **kwargs)
        common_len = len(ac.read(ac.SUPPORT / (commit + '.deployment.json'))['files'])
        release_inputs[commit] = p['inputs'][:common_len]
        p['inputs'] = p['inputs'][common_len:]
        return p

    def conf(commit):
        return ac.read(ac.SUPPORT / (commit + '.deployment.json'))['release'] + '/' + ac.CONFIG

    def add(mode, kind, req, files):
        phases.append({'mode': mode, 'kind': kind, 'request': req,
                       'depends_on': phases[-1]['request']['job_id'] if phases else None,
                       'predecessor_members': list(files)})

    for mode in core.MODES:
        gpu_output = f'{ac.ROOT}/formal-gpu-{mode}-{args.attempt}'
        cpu_output = f'{ac.ROOT}/formal-cpu-{mode}-{args.attempt}'
        previous_receipt = [str(output(phases[-1]['request']) / 'summary.json')] if phases else []
        req = request(args.gpu_commit, f'gpu-{mode}', 'a1_tasks',
                      ['gpu', '--config', conf(args.gpu_commit), '--cpu-manifest', replay, '--mode', mode,
                       '--output', gpu_output, '--run-commit', args.gpu_commit, '--seconds', str(MODE_LIMIT - 40)],
                      MODE_LIMIT - 10, card=args.card, input_files=replay_inputs)
        add(mode, 'gpu', req, previous_receipt)
        req = request(args.commit, f'cpu-{mode}', 'a1_evaluate',
                      ['mode', '--config', conf(args.commit), '--cpu-manifest', replay, '--mode', mode,
                       '--test', gpu_output + '/test', '--output', cpu_output, '--run-commit', args.gpu_commit],
                      CPU_MODE_LIMIT, input_files=replay_inputs)
        add(mode, 'cpu', req, [f'{gpu_output}/test/{name}' for name in
                              ('rank0.json', 'predictions.rank0.jsonl', 'derived.rank0.py', 'derived.rank0.diff')]
            + [gpu_output + '/summary.json'])
        evaluation_roots.append(cpu_output)
    req = request(args.commit, 'paired-paired', 'a1_evaluate',
                  ['paired', '--config', conf(args.commit), '--output', f'{ac.ROOT}/formal-paired-{args.attempt}',
                   normal, *evaluation_roots], CPU_PAIRED_LIMIT, input_files=normal_inputs)
    add('paired', 'paired', req, [folder + '/' + name for folder in evaluation_roots for name in PARAMS])
    requests = f'{ac.control.BASE}/requests'
    parent_id = GROUP + 'formal-parent-' + args.attempt
    plan_path = f'{requests}/{parent_id}.plan.json'
    approval = f'{requests}/{parent_id}.authorization.json'
    parent = request(args.commit, 'parent', 'a1_formal',
                     ['run', '--plan', plan_path, '--approval', approval, '--approval-sha256', APPROVAL_PLACEHOLDER,
                      '--output', f'{ac.ROOT}/formal-parent-{args.attempt}'], PARENT_LIMIT)
    plan = {'schema': 1, 'experiment': 'LETTER_INSTRUMENTS_QWEN3_06B_BASE_SHUFFLE_V2',
            'formal_submitted': False, 'formal_gpu_device_seconds_delta': 0, 'attempt': args.attempt,
            'approvals_required': ['web_review_pass', 'separate_user_formal_authorization'],
            'job_root': target.profile['job_root'], 'requests_root': requests,
            'runner': target.profile['runner'], 'runner_sha256': target.profile['runner_sha256'],
            'runner_python': target.profile['python'], 'card': args.card, 'users_per_mode': USERS,
            'candidates_per_mode': CANDIDATES, 'preflight_gpu_commit': args.gpu_commit,
            'evaluation_and_controller_commit': args.commit, 'release_inputs': release_inputs,
            'prerequisites': [GROUP + stage + '-' + attempt for stage, attempt in prereq],
            'timeout_fixture': timeout_id, 'phases': phases, 'parent': parent, 'zero_readback': zero,
            'server_plan_path': plan_path, 'approval_path': approval,
            'future_budget': {'gpu_seconds_per_mode': MODE_LIMIT, 'gpu_device_seconds': DEVICE_LIMIT,
                              'gpu_seconds_reserved_this_plan': len(core.MODES) * MODE_LIMIT,
                              'parent_wall_timeout_seconds': PARENT_LIMIT}}
    validate(plan)
    write(args.output, plan, exclusive=True)


def success(result):
    return (result['status'] == 'SUCCEEDED' and result['exit_code'] == 0 and result['cleanup_verified']
            and not result['worker_alive'] and not result['workgroup_live'])


def authorize(plan, args):
    assert str(args.plan) == plan['server_plan_path'] and str(args.approval) == plan['approval_path']
    assert re.fullmatch(r'[0-9a-f]{64}', args.approval_sha256) and sha(args.approval) == args.approval_sha256
    approval = read(args.approval)
    assert approval['experiment'] == plan['experiment'] and approval['plan_sha256'] == sha(args.plan)
    assert approval['web_review_pass'] is True and approval['user_formal_launch_authorized'] is True
    assert approval['gpu_commit'] == plan['preflight_gpu_commit'] and approval['attempt'] == plan['attempt']
    assert os.environ.get('GRID_JOB_ID') == plan['parent']['job_id'] and args.output == output(plan['parent'])
    assert sha(plan['runner']) == plan['runner_sha256']


def run(args):
    plan = read(args.plan)
    validate(plan)
    authorize(plan, args)
    jobs = Path(plan['job_root'])
    requests = Path(plan['requests_root'])
    child_env = {k: v for k, v in os.environ.items() if k not in ('GRID_JOB_ID', 'GRID_JOB_DIR')}

    def runner(*argv):
        return json.loads(subprocess.check_output([plan['runner_python'], '-I', '-S', '-B', plan['runner'],
                                                   *map(str, argv)], env=child_env, timeout=120))

    with (jobs / f'.{GROUP}formal.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert all(success(runner('query', jobs / name)) for name in plan['prerequisites'])
        fixture = runner('query', jobs / plan['timeout_fixture'])
        assert fixture['status'] == 'TIMED_OUT' and fixture['cleanup_verified'] and not fixture['workgroup_live']
        spent = 0.0
        for job in jobs.glob(GROUP + 'formal-gpu-*'):
            state = runner('query', job)
            assert state.get('ended_at') and state['cleanup_verified']
            if state.get('started_at'):
                spent += (datetime.fromisoformat(state['ended_at']) - datetime.fromisoformat(state['started_at'])).total_seconds()
        assert spent + len(core.MODES) * MODE_LIMIT <= DEVICE_LIMIT
        for phase in plan['phases']:
            req = phase['request']
            assert not (jobs / req['job_id']).exists()
            assert not (requests / (req['job_id'] + '.json')).exists()
            assert not output(req).exists()
        args.output.mkdir(parents=True, exist_ok=False)
        summary = {'status': 'RUNNING', 'phases': [], 'used_gpu_device_seconds_before': spent}
        dest = args.output / 'summary.json'
        write(dest, summary)
        try:
            for phase in plan['phases']:
                req = dict(phase['request'])
                name = req['job_id']
                req['inputs'] = (plan['release_inputs'][req['capture_version']] + req['inputs']
                                 + [{'path': str(args.approval), 'sha256': args.approval_sha256}]
                                 + [{'path': path, 'sha256': sha(path)} for path in phase['predecessor_members']])
                path = requests / (name + '.json')
                write(path, req, exclusive=True)
                stage = {'job_id': name, 'request': str(path), 'status': 'REQUEST_SAVED'}
                summary['phases'].append(stage)
                write(dest, summary)
                job = jobs / name
                try:
                    state = runner('submit', jobs, path)
                except subprocess.SubprocessError:
                    if not (job / 'accepted.json').is_file():
                        raise
                    state = runner('query', job)
                while state['status'] in ('ACCEPTED', 'RUNNING'):
                    time.sleep(25)
                    state = runner('query', job)
                stage['runner_state'] = state
                assert success(state)
                receipt = read(output(req) / 'summary.json')
                if phase['kind'] == 'gpu':
                    assert receipt['status'] == 'PASS' and receipt['stages'][phase['mode']]['users'] == USERS
                else:
                    assert receipt['users'] == USERS and receipt['preflight'] is False
                    assert receipt['status'].startswith('PASS')
                    if phase['kind'] == 'cpu':
                        assert receipt['candidate_anomalies']['candidate_count'] == CANDIDATES
                stage['status'] = 'PASS'
                write(dest, summary)
        except BaseException:
            summary['status'] = 'FAILED_OR_BLOCKED'
            write(dest, summary)
            raise
        summary['status'] = 'PASS_A1_INSTRUMENTS_V2_FORMAL'
        write(dest, summary)


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    commands = p.add_subparsers(dest='action', required=True)
    b = commands.add_parser('build')
    for name in ('commit', 'gpu-commit', 'card', 'replay', 'normal', 'gpu', 'assess', 'lifecycle'):
        b.add_argument('--' + name, required=True)
    b.add_argument('--attempt', default='r0')
    b.add_argument('--output', required=True, type=Path)
    for action in ('dry-run', 'run'):
        command = commands.add_parser(action)
        command.add_argument('--plan', required=True, type=Path)
        if action == 'run':
            command.add_argument('--approval', required=True, type=Path)
            command.add_argument('--approval-sha256', required=True)
            command.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    if args.action == 'build':
        build(args)
    elif args.action == 'run':
        run(args)
    else:
        plan = read(args.plan)
        validate(plan)
        print(json.dumps({'status': 'PASS_LOCAL_DRY_RUN', 'plan_sha256': sha(args.plan),
                          'stages': [phase['request']['job_id'] for phase in plan['phases']],
                          'remote_requests_sent': 0, 'formal_gpu_device_seconds_delta': 0}, indent=2))


if __name__ == '__main__':
    main()
