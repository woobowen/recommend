"""Prepare and submit the fixed A0 sequence after the non-author GPU preflight gate."""
import argparse
import json
from pathlib import Path
import subprocess

from lc_rec.shuffle_v2 import control, core
from lc_rec.shuffle_v2.collect import hardware
from lc_rec.shuffle_v2.replay import CONFIG

RESEARCH = ['lc_rec/shuffle_v2/' + file for file in ('a0_instruments.json', 'core.py', 'entry.py', 'runtime.py',
                                                    'generate.py', 'evaluate.py', 'statistics.py', 'single_shard.py')]


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--run-commit', required=True)
    p.add_argument('--control-commit', required=True)
    p.add_argument('--review', type=Path, required=True)
    p.add_argument('--preflight', default='gpu-preflight-r0')
    a = p.parse_args()
    run = a.run_commit
    ops = a.control_commit
    assert run != ops and len(run) == len(ops) == 40
    control.readback(ops)
    subprocess.run(['git','merge-base','--is-ancestor',run,ops], cwd=control.REPO, check=True)
    subprocess.run(['git','diff','--exit-code',run,ops,'--',*RESEARCH], cwd=control.REPO, check=True)
    connection = control.connect()
    states = control.job_states(connection)
    job_id = control.PREFIX + 'group-series-r0'
    state = states.get(job_id)
    if state:
        print(json.dumps(control.gridctl.query(connection, job_id), indent=2))
        return
    config = json.loads(CONFIG.read_text())
    review = json.loads(a.review.read_text())
    assert review['status'] == 'PASS' and review['RUN_COMMIT'] == run
    assert review['preflight_job_id'] == control.PREFIX + a.preflight
    control.succeeded(states, a.preflight)
    snapshot = hardware(connection)
    allocations = {r.split(',')[0].strip() for r in snapshot['processes'].splitlines()[1:]}
    free = []
    for line, uuid in zip(snapshot['gpus'].splitlines()[1:], config['gpus'], strict=True):
        index, observed, model, used, total, _ = map(str.strip, line.split(','))
        assert observed == uuid and model == 'NVIDIA GeForce RTX 4090' and int(total.split()[0]) >= 24000
        if uuid not in allocations and int(used.split()[0]) < 100:
            free.append(uuid)
    assert free, 'No approved GPU is idle'
    preferred = review['device_uuid']
    device = preferred if preferred in free else free[0]
    profile = connection.profile
    state_root = control.SUPPORT
    def deployment(commit):
        proof = json.loads((state_root / f'{commit}.deployment.json').read_text())
        assert proof['RUN_COMMIT'] == commit and proof['runner_sha256'] == control.RUNNER_SHA
        return proof['release'], [{'path': proof['release'] + '/' + r['path'], 'sha256': r['sha256']} for r in proof['files']]
    release, source_inputs = deployment(run)
    orchestrator_release, orchestrator_inputs = deployment(ops)
    base = config['baseline']
    normal = control.stage_success(states, 'cpu-eval-normal')
    cpu = control.stage_success(states, 'cpu-replay')
    single = control.stage_success(states, 'cpu-single-shard')
    audit_dir = f'{control.ROOT}/{cpu}'
    shard = f'{control.ROOT}/{single}/audit.json'
    stage_receipt = f'{control.ROOT}/{a.preflight}/summary.json'
    assert control.remote_hash(connection, stage_receipt) == review['preflight_summary_sha256']
    frozen = [f'{audit_dir}/{name}' for name in ('audit.json','assets.json','inputs.jsonl')] + [shard, stage_receipt]
    gate = connection.put(a.review, f'{control.BASE}/requests/{job_id}.review.json')
    gates = [{'path': path, 'sha256': control.remote_hash(connection, path)} for path in frozen]
    gates.append({'path': gate['path'], 'sha256': gate['sha256']})
    original = json.loads((control.REPO / 'lc_rec/shuffle/results/letter-a0-20261112-shuffle3-instruments-seed42-20261114-r0/job/request.json').read_text())
    env = {**original['env'], 'CUDA_VISIBLE_DEVICES': ''}
    identity = {n: profile[n] for n in ('user','uid','namespace','deployment_sha256')}

    def request(job, module, timeout, directory, commit, inputs, extra, card=''):
        return {**identity, 'schema': 1, 'host': profile['hostname'], 'job_id': control.PREFIX + job,
                'capture_version': commit, 'cwd': directory, 'env': {**env, 'CUDA_VISIBLE_DEVICES': card},
                'timeout_seconds': timeout, 'inputs': inputs,
                'argv': [base['python'], '-B', '-m', 'lc_rec.shuffle_v2.' + module,
                         '--output', f'{control.ROOT}/{job}', *extra]}

    formal_limit = config['budget']['formal_attempt_wall_seconds']
    formal_total = min(config['budget']['formal_total_wall_seconds'], config['budget']['formal_total_device_seconds'])
    assert control.budget_seconds(states, 'gpu-formal-', formal_total) >= 4 * formal_limit
    assert control.budget_seconds(states, 'cpu-', config['budget']['cpu_total_wall_seconds']) >= 5 * 900
    phases = {}
    for mode in core.MODES:
        gpu = f'gpu-formal-{mode}-r0'
        cpu_job = f'cpu-eval-{mode}-r0'
        args = ['--mode', mode, '--run-commit', run, '--cpu-audit', f'{audit_dir}/audit.json', '--single-shard-audit', shard]
        phases[mode] = {
            'gpu': request(gpu, 'generate', formal_limit - 30, release, run, source_inputs + gates,
                           args + ['--seconds', str(formal_limit - 75)], device),
            'cpu': request(cpu_job, 'evaluate', 900, release, run, source_inputs + gates,
                           args + ['--evaluation-commit', run, '--test-dir', f'{control.ROOT}/{gpu}/test']),
        }
    comparison_dirs = [f'{control.ROOT}/{normal}', *(f'{control.ROOT}/cpu-eval-{mode}-r0' for mode in core.MODES)]
    paired = request('cpu-summary-r0', 'statistics', 900, release, run, source_inputs, comparison_dirs)
    plan = {'formal_run_commit': run, 'orchestrator_commit': ops, 'modes': phases, 'paired': paired,
            'normal_job_id': control.PREFIX + normal, 'preflight_review': review, 'gpu_selection': snapshot,
            'prefix': control.PREFIX, 'runner': profile['runner'], 'runner_sha256': control.RUNNER_SHA,
            'job_root': profile['job_root'], 'requests_root': control.BASE + '/requests'}
    plan_local = state_root / (job_id + '.plan.json')
    control.transport.atomic(plan_local, plan)
    sent = connection.put(plan_local, f'{control.BASE}/requests/{job_id}.plan.json')
    plan_input = {'path': sent['path'], 'sha256': sent['sha256']}
    parent = request('group-series-r0', 'series', formal_total + 5 * 900 + 1800,
                     orchestrator_release, ops, orchestrator_inputs + [plan_input], ['--plan', sent['path']])
    parent_path = state_root / (job_id + '.request.json')
    control.transport.atomic(parent_path, parent)
    print(json.dumps(control.gridctl.submit(connection, parent_path), indent=2))


if __name__ == '__main__':
    main()
