"""Prepare immutable runner inputs for the reviewer-owned evaluator."""
import argparse
import json
from pathlib import Path
import subprocess

from lc_rec.shuffle_v2 import beauty, beauty_control, control, core

TIMEOUT = {'normal': 600, 'final': 1800}


def accepted_stage(states, stem):
    names = [name.removeprefix(beauty_control.PREFIX) for name, state in states.items()
             if name.startswith(beauty_control.PREFIX + stem + '-') and state['status'] == 'SUCCEEDED']
    assert len(names) == 1, names
    beauty_control.require_success(states, names[0])
    return names[0]


def launch(connection, config, control_commit, mode, attempt, run_commit):
    states = control.job_states(connection, beauty_control.PREFIX)
    root = beauty_control.ROOT
    job = f'cpu-independent-{mode}-{attempt}'
    if beauty_control.PREFIX + job in states:
        return control.gridctl.query(connection, beauty_control.PREFIX + job)
    assert control.budget_seconds(states, 'cpu-', config['budget']['cpu_total_wall_seconds'], prefix=beauty_control.PREFIX) >= TIMEOUT[mode] + 15
    base = config['baseline']
    normal_primary = config['normal_primary']
    options = ['--data-dir', base['data-root'] + '/Beauty',
               '--condition', 'normal', base['normal_test_dir'], normal_primary]
    if mode == 'final':
        paired = accepted_stage(states, 'cpu-summary')
        options.extend(('--v2-run-commit', run_commit, '--paired-dir', f'{root}/{paired}'))
        for shuffle in core.MODES:
            generated = accepted_stage(states, 'gpu-formal-' + shuffle)
            evaluated = accepted_stage(states, 'cpu-eval-' + shuffle)
            options.extend(('--condition', shuffle, f'{root}/{generated}/test', f'{root}/{evaluated}/primary'))
    payload = beauty_control.request(connection, config, control_commit, job, 'independent_beauty_review',
                                     TIMEOUT[mode], options)
    payload['env']['OPENBLAS_NUM_THREADS'] = '1'
    return beauty_control.submit(connection, payload)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('mode', choices=tuple(TIMEOUT))
    parser.add_argument('--attempt', default='r0', type=beauty_control.attempt_id)
    parser.add_argument('--control-commit', default='HEAD')
    parser.add_argument('--run-commit')
    args = parser.parse_args()
    commit = beauty_control.commit_ref(args.control_commit)
    if args.mode == 'final':
        assert args.run_commit and beauty_control.commit_ref(args.run_commit) == args.run_commit
        subprocess.run(['git', 'merge-base', '--is-ancestor', args.run_commit, commit], cwd=control.REPO, check=True)
    config = json.loads(beauty.CONFIG.read_text())
    print(json.dumps(launch(beauty_control.connect(beauty_control.runner_sha_at(commit)), config, commit, args.mode, args.attempt, args.run_commit), indent=2))


if __name__ == '__main__':
    main()
