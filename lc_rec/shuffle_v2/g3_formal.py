"""Build a local formal plan. Future execution requires separate user authorization."""
import argparse
import fcntl
import json
import os
from datetime import datetime
from pathlib import Path
import subprocess
import time

from lc_rec.shuffle_v2 import core, g3_assets

APPROVAL_SHA = 'PIN_USER_AUTHORIZATION_SHA256_AFTER_WEB_REVIEW'
MODE_LIMIT = 28800
DEVICE_LIMIT = 129600
USERS = 22363
BEAMS = 20


def output_of(request):
    return Path(request['argv'][request['argv'].index('--output') + 1])


def validate(plan):
    modes = plan['modes']
    assert tuple(modes) == tuple(core.MODES)
    assert plan['formal_submitted'] is False and plan['formal_gpu_device_seconds_delta'] == 0
    assert plan['planned_users_per_condition'] == USERS and plan['planned_candidates_per_condition'] == USERS * BEAMS
    assert plan['card'] and plan['approvals_required'] == ['web_review_pass', 'user_formal_launch']
    job_ids, outputs = set(), set()
    for mode, pair in modes.items():
        gpu, cpu = pair['gpu'], pair['cpu']
        assert gpu['name'].startswith(f'g3-beauty-v2-gpu-formal-{mode}-')
        assert cpu['name'].startswith(f'g3-beauty-v2-cpu-formal-eval-{mode}-')
        assert gpu['request']['env']['CUDA_VISIBLE_DEVICES'] == plan['card']
        assert '--authorization' in gpu['request']['argv'] and APPROVAL_SHA in gpu['request']['argv']
        assert cpu['request']['env']['CUDA_VISIBLE_DEVICES'] == ''
        assert gpu['request']['timeout_seconds'] + 15 <= MODE_LIMIT
        for phase in (gpu, cpu):
            assert phase['request']['job_id'] == phase['name']
            assert phase['name'] not in job_ids and output_of(phase['request']) not in outputs
            job_ids.add(phase['name'])
            outputs.add(output_of(phase['request']))
    assert sum(pair['gpu']['request']['timeout_seconds'] + 15 for pair in modes.values()) <= DEVICE_LIMIT
    paired = plan['paired']
    assert paired['name'].startswith('g3-beauty-v2-cpu-formal-paired-') and paired['name'] not in job_ids
    assert output_of(paired['request']) not in outputs
    assert paired['request']['env']['CUDA_VISIBLE_DEVICES'] == ''


def build(args):
    from lc_rec.shuffle_v2 import g3_control as gc

    config = g3_assets.read(g3_assets.CONFIG)
    target, _, _ = gc.connection()
    cpu_files = gc.cpu_gates(target, args.replay)
    gc.require_success(target, args.normal)
    prereq_names = [f'gpu-preflight-{mode}-{args.preflight_attempt}' for mode in core.MODES]
    prereq_names += [f'cpu-paired-{args.preflight_attempt}', args.normal, args.replay, 'cpu-lifecycle-review-r0']
    for name in prereq_names:
        gc.require_success(target, name)
    prereq = [gc.PREFIX + name for name in prereq_names]
    known_states = gc.control.job_states(target, gc.PREFIX)
    card = args.card
    assert card in config['gpus']
    assert all(known_states[gc.PREFIX + name]['status'] == 'SUCCEEDED' for name in prereq_names)
    prior = [state for name, state in known_states.items() if name.startswith(gc.PREFIX + 'gpu-formal-')]
    spent = elapsed_formal_gpu(prior)
    assert spent + 4 * MODE_LIMIT <= DEVICE_LIMIT
    release, release_inputs = gc.release_files(args.commit)
    assert gc.release_files(args.gpu_commit)[0]
    runner = target.profile['runner']
    normal_root = f'{gc.ROOT}/{args.normal}'
    normal_inputs = [{'path': f'{normal_root}/{member}', 'sha256': gc.control.remote_hash(target, f'{normal_root}/{member}')}
                     for member in ('summary.json', 'primary/per_user.csv', 'primary/per_user_metrics.npz')]
    approval = {'path': f'{gc.control.BASE}/requests/{gc.PREFIX}formal-authorization-{args.attempt}.json',
                'sha256': APPROVAL_SHA}
    phases = {}
    common_inputs = {}

    def phase(request, predecessor_members=()):
        version = request['capture_version']
        _, inputs = gc.release_files(version)
        assert request['inputs'][:len(inputs)] == inputs
        common_inputs[version] = inputs
        request['inputs'] = request['inputs'][len(inputs):]
        return {'name': request['job_id'], 'request': request, 'predecessor_members': list(predecessor_members)}

    for mode in core.MODES:
        gpu = gc.generation_request(target, config, args.gpu_commit, 'FORMAL', mode, args.attempt, card, cpu_files, approval)
        cpu = gc.evaluation_request(target, config, args.commit, 'FORMAL', mode, args.attempt, cpu_files)
        cpu['argv'][cpu['argv'].index('--run-commit') + 1] = args.gpu_commit
        gpu_output = output_of(gpu)
        phases[mode] = {'gpu': phase(gpu), 'cpu': phase(cpu, [str(gpu_output / name) for name in
            ('rank0.json', 'metrics.json', 'v2-binding.rank0.json', 'derived.rank0.py', 'predictions.rank0.jsonl')])}
    paired = gc.paired_request(target, config, args.commit, 'FORMAL', args.attempt, args.normal, normal_inputs)
    paired_phase = phase(paired, [str(output_of(pair['cpu']['request']) / name) for pair in phases.values()
                                 for name in ('summary.json', 'primary/per_user.csv', 'primary/per_user_metrics.npz')])
    parent_id = gc.PREFIX + 'group-formal-' + args.attempt
    parent_output = f'{gc.ROOT}/group-formal-{args.attempt}'
    remote_plan = f'{gc.control.BASE}/requests/{parent_id}.plan.json'
    parent = gc.request(target, config, args.commit, 'group-formal-' + args.attempt, 126000, 'g3_formal',
        ['run', '--plan', remote_plan, '--approval', approval['path'], '--approval-sha256', APPROVAL_SHA,
         '--output', parent_output])
    parent = phase(parent)
    plan = {'schema': 1, 'formal_submitted': False, 'formal_gpu_device_seconds_delta': 0,
            'approvals_required': ['web_review_pass', 'user_formal_launch'], 'card': card,
            'planned_users_per_condition': USERS, 'planned_candidates_per_condition': USERS * BEAMS,
            'training_commit': config['training_commit'], 'gpu_commit': args.gpu_commit,
            'evaluation_and_controller_commit': args.commit,
            'prepared_model_identity': config['model_identity_sha256'], 'normal_prediction_sha256': config['normal_predictions_sha256'],
            'job_root': target.profile['job_root'], 'requests_root': str(Path(remote_plan).parent),
            'runner': runner, 'runner_sha256': target.profile['runner_sha256'],
            'parent': parent, 'server_plan_path': remote_plan, 'approval_path': approval['path'],
            'prerequisite_jobs': prereq, 'release_inputs': common_inputs,
            'observed_prior_formal_device_seconds': spent,
            'modes': phases, 'paired': paired_phase, 'resume_rule': 'Query existing IDs; retain failed attempts and allocate a new authorized attempt.'}
    assert release_inputs == common_inputs[args.commit] and parent['request']['cwd'] == release
    validate(plan)
    with args.output.open('x') as stream:
        json.dump(plan, stream, indent=2, allow_nan=False)
        stream.write('\n')


def elapsed_formal_gpu(prior):
    total = 0.0
    for state in prior:
        assert state.get('ended_at') and state['cleanup']['verified'], 'Active or unclean prior formal GPU job'
        if state.get('started_at'):
            total += (datetime.fromisoformat(state['ended_at']) - datetime.fromisoformat(state['started_at'])).total_seconds()
    return total


def success(state):
    return (state['status'] == 'SUCCEEDED' and state['exit_code'] == 0 and state['cleanup_verified']
            and not state['worker_alive'] and not state['workgroup_live'])


def run(args):
    plan = g3_assets.read(args.plan)
    validate(plan)
    assert str(args.approval) == plan['approval_path']
    approval = g3_assets.read(g3_assets.verified(args.approval, args.approval_sha256))
    assert approval['scope'] == 'G3_BEAUTY_SHUFFLE_V2_FORMAL' and approval['authorized_by'] == 'user'
    assert approval['web_review_pass'] is True and approval['run_commit'] == plan['gpu_commit']
    assert approval['plan_sha256'] == g3_assets.sha(args.plan)
    assert os.environ.get('GRID_JOB_ID') == plan['parent']['name']
    assert str(args.output) == str(output_of(plan['parent']['request']))
    g3_assets.verified(plan['runner'], plan['runner_sha256'])
    env = {key: value for key, value in os.environ.items() if key not in ('GRID_JOB_ID', 'GRID_JOB_DIR')}
    jobs = Path(plan['job_root'])
    requests = Path(plan['requests_root'])

    def runner(*argv):
        return json.loads(subprocess.check_output(
            ['/usr/bin/python3', '-I', '-S', '-B', plan['runner'], *map(str, argv)], env=env, timeout=120))

    def request(phase):
        raw = json.dumps(phase['request']).replace(APPROVAL_SHA, args.approval_sha256)
        payload = json.loads(raw)
        files = [{'path': path, 'sha256': g3_assets.sha(path)} for path in phase['predecessor_members']]
        payload['inputs'] = plan['release_inputs'][payload['capture_version']] + payload['inputs'] + files
        return payload

    def idle_card():
        query = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used', '--format=csv,noheader,nounits'],
                                        text=True)
        used = {row.split(',')[0].strip(): int(row.split(',')[1]) for row in query.splitlines()}
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True)
        assert plan['card'] in used and used[plan['card']] < 100 and plan['card'] not in apps

    with (jobs / '.g3-beauty-v2-formal.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert all(success(runner('query', jobs / job)) for job in plan['prerequisite_jobs'])
        prior = [g3_assets.read(path / 'status.json') for path in jobs.glob('g3-beauty-v2-gpu-formal-*')
                 if (path / 'status.json').is_file()]
        assert elapsed_formal_gpu(prior) + 4 * MODE_LIMIT <= DEVICE_LIMIT
        assert not any((jobs / p['name']).exists() or (requests / (p['name'] + '.json')).exists() or output_of(p['request']).exists()
                       for p in [*(p for pair in plan['modes'].values() for p in pair.values()), plan['paired']])
        assert not any(not g3_assets.read(path / 'status.json').get('ended_at')
                       for path in jobs.glob('g3-beauty-v2-gpu-*') if (path / 'status.json').is_file())
        args.output.mkdir(parents=True, exist_ok=False)
        stages = []

        def persist(status):
            pending = args.output / 'summary.partial'
            pending.write_text(json.dumps({'status': status, 'stages': stages}, indent=2, allow_nan=False) + '\n')
            pending.replace(args.output / 'summary.json')

        def execute(phase, category):
            payload = request(phase)
            if category == 'gpu':
                idle_card()
            path = requests / (payload['job_id'] + '.json')
            with path.open('x') as stream:
                json.dump(payload, stream, indent=2, allow_nan=False)
            stage = {'job_id': payload['job_id'], 'request_path': str(path), 'status': 'REQUEST_SAVED'}
            stages.append(stage)
            persist('RUNNING')
            job = jobs / payload['job_id']
            try:
                state = runner('submit', jobs, path)
            except subprocess.SubprocessError:
                if not (job / 'accepted.json').is_file():
                    raise
                state = runner('query', job)
            while state['status'] in ('ACCEPTED', 'RUNNING'):
                time.sleep(30)
                state = runner('query', job)
            stage['result'] = state
            assert success(state), state
            destination = output_of(payload)
            receipt = g3_assets.read(destination / ('metrics.json' if category == 'gpu' else 'summary.json'))
            if category == 'gpu':
                expected_mode = payload['argv'][payload['argv'].index('--mode') + 1]
                assert (receipt['status'], receipt['scope'], receipt['condition'], receipt['actual_test_users'],
                        receipt['all_row_indices_exactly_once']) == ('PASS', 'FORMAL', expected_mode, USERS, True)
                assert receipt['anomalies']['candidate_count'] == USERS * BEAMS
            elif category == 'cpu':
                assert (receipt['scope'], receipt['users'], receipt['status']) == ('FORMAL', USERS, 'PASS_G3_V2_EVALUATION')
            else:
                assert receipt['scope'] == 'FORMAL' and receipt['users'] == USERS and receipt['status'] == 'PASS_G3_V2_PAIRED'
            persist('RUNNING')

        try:
            for phase in plan['modes'].values():
                execute(phase['gpu'], 'gpu')
                execute(phase['cpu'], 'cpu')
            execute(plan['paired'], 'paired')
        except BaseException:
            persist('FAILED_GATE_OR_CHILD')
            raise
        persist('PASS_G3_BEAUTY_V2_FORMAL')


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    commands = parser.add_subparsers(dest='command', required=True)
    build_parser = commands.add_parser('build', allow_abbrev=False)
    for option in ('commit', 'gpu-commit', 'card', 'replay', 'normal', 'preflight-attempt'):
        build_parser.add_argument('--' + option, required=True)
    build_parser.add_argument('--attempt', default='r0')
    build_parser.add_argument('--output', required=True, type=Path)
    execute_parser = commands.add_parser('run', allow_abbrev=False)
    execute_parser.add_argument('--plan', required=True, type=Path)
    execute_parser.add_argument('--approval', required=True, type=Path)
    execute_parser.add_argument('--approval-sha256', required=True)
    execute_parser.add_argument('--output', required=True, type=Path)
    options = parser.parse_args()
    if options.command == 'build':
        build(options)
    else:
        run(options)


if __name__ == '__main__':
    main()
