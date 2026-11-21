"""Submit only bounded Beauty CPU preparation jobs and bind their receipts."""
import argparse
import fcntl
import json
from pathlib import Path

from lc_rec.shuffle_v2 import beauty, beauty_control as bc, control

LIFECYCLE_TIMEOUT = {'decoy': 45, 'normal': 10, 'orphan': 12, 'timeout': 8}


def launch(connection, config, commit, name, module, timeout, extra, inputs=()):
    states = control.job_states(connection, bc.PREFIX)
    if bc.PREFIX + name in states:
        return control.gridctl.query(connection, bc.PREFIX + name)
    payload = bc.request(connection, config, commit, name, module, timeout, extra, inputs)
    return bc.submit(connection, payload)


def remaining_cpu(states, config):
    from datetime import datetime
    used = 0.0
    held = 0
    for name, state in states.items():
        if not name.startswith(bc.PREFIX + 'cpu-'):
            continue
        if state.get('ended_at'):
            start = state.get('started_at') or state['received_at']
            used += (datetime.fromisoformat(state['ended_at']) - datetime.fromisoformat(start)).total_seconds()
        else:
            original = bc.SUPPORT / (name + '.request.json')
            held += json.loads(original.read_text())['timeout_seconds']
    return config['budget']['cpu_total_wall_seconds'] - used - held


def lifecycle(connection, config, commit, attempt):
    results = {}
    for case, timeout in LIFECYCLE_TIMEOUT.items():
        name = f'cpu-lifecycle-{case}-{attempt}'
        states = control.job_states(connection, bc.PREFIX)
        assert remaining_cpu(states, config) >= timeout + 15
        reply = launch(connection, config, commit, name, 'beauty_lifecycle', timeout, ['--case', case])
        results[case] = reply
        if case == 'decoy':
            assert reply['status'] in ('ACCEPTED', 'RUNNING', 'SUCCEEDED')
    return results


def evaluate(connection, config, commit, args):
    states = control.job_states(connection, bc.PREFIX)
    for job in (args.preflight, 'cpu-replay-r3'):
        bc.require_success(states, job)
    assert remaining_cpu(states, config) >= 315
    input_dir = f'{bc.ROOT}/{args.preflight}'
    cpu_dir = f'{bc.ROOT}/cpu-replay-r3'
    assert args.run_commit
    summary = json.loads(control.remote_python(connection,
        f'from pathlib import Path; print(Path({(input_dir + "/summary.json")!r}).read_text())'))
    assert summary['status'] == 'PASS_V2_PREFLIGHT' and summary['RUN_COMMIT'] == args.run_commit
    release, _ = bc.deployment(commit)
    files = [f'{input_dir}/summary.json', f'{cpu_dir}/audit.json', f'{cpu_dir}/single-shard.json']
    for stage in summary['stages']:
        folder = stage['output']
        files += [f'{folder}/rank0.json', f'{folder}/predictions.rank0.jsonl']
    inputs = [{'path': path, 'sha256': control.remote_hash(connection, path)} for path in files]
    return launch(connection, config, commit, f'cpu-preflight-eval-{args.attempt}', 'beauty_preflight_eval', 300,
                  ['--config', f'{release}/{bc.CONFIG_PATH}', '--preflight-dir', input_dir,
                   '--cpu-audit', files[1], '--single-shard-audit', files[2], '--run-commit', args.run_commit], inputs)


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('action', choices=('lifecycle', 'lifecycle-review', 'evaluate'))
    p.add_argument('--commit', default='HEAD')
    p.add_argument('--attempt', default='r0', type=bc.attempt_id)
    p.add_argument('--preflight', default='gpu-preflight-r1')
    p.add_argument('--run-commit')
    args = p.parse_args()
    commit = bc.commit_ref(args.commit)
    runner_sha = bc.runner_sha_at(commit)
    connection = bc.connect(runner_sha)
    release, _ = bc.deployment(commit)
    config = json.loads(beauty.CONFIG.read_text())
    with (bc.SUPPORT / 'submit.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        if args.action == 'lifecycle':
            receipt = lifecycle(connection, config, commit, args.attempt)
        elif args.action == 'evaluate':
            receipt = evaluate(connection, config, commit, args)
        else:
            import shlex
            receipt = json.loads(connection.ssh(shlex.join([
                config['baseline']['python'], '-B', '-c',
                'import json,sys; from pathlib import Path; sys.path.insert(0,sys.argv[1]); '
                'from lc_rec.shuffle_v2.beauty_lifecycle import review; '
                'print(json.dumps(review(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], sys.argv[5])))',
                release, connection.profile['job_root'], bc.ROOT, args.attempt, runner_sha])))
            bc.immutable(bc.SUPPORT / f'lifecycle-{args.attempt}.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
