"""Queue the remaining seed 45 and 46 experiments, two sequential jobs per GPU."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
LANES = {
    0: [('T13', 'beauty', 'mlp', 45), ('T17', 'beauty', 'shuffle', 46)],
    1: [('T14', 'toys', 'shuffle', 45), ('T18', 'beauty', 'mlp', 46)],
    2: [('T15', 'toys', 'mlp', 45), ('T19', 'toys', 'shuffle', 46)],
    3: [('T16', 'beauty', 'shuffle', 45), ('T20', 'toys', 'mlp', 46)],
}


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    tmp.replace(path)


def verify_source():
    manifest = ROOT / 'experiments/sid_order_5seed_v1/beauty/shuffle/seed_42/source_manifest.json'
    hashes = json.loads(manifest.read_text())['files']
    for name, expected in hashes.items():
        if name.startswith(('src/', 'configs/', 'codex/sid_order_launch/')) or name in (
            'codex/plan_v1_1/run_task.py', 'codex/plan_v1_1/protocol.py',
            'codex/plan_v1_1/attention.py'):
            actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            assert actual == expected, f'Source changed: {name}'


def completed(status_path):
    status = json.loads(status_path.read_text())
    if status['state'] == 'failed':
        raise RuntimeError(f"Previous task failed: {status['task_id']}")
    if status['state'] != 'completed':
        return False
    assert status.get('exit_code') == 0, f"Previous task exit code: {status['task_id']}"
    result = json.loads((Path(status['output']) / 'stage_result.json').read_text())
    assert result.get('completed') is True, f"Previous task results incomplete: {status['task_id']}"
    return True


def run_lane(group, gpu):
    tasks = LANES[gpu]
    previous = ROOT / 'codex/third_round_20261104' / f'T{gpu + 9:02d}_status.json'
    for index, (task, dataset, variant, seed) in enumerate(tasks):
        output = ROOT / f'experiments/sid_order_5seed_v1/{dataset}/{variant}/seed_{seed}'
        status_path = group / f'{task}_status.json'
        status = dict(task_id=task, gpu=gpu, dataset=dataset, variant=variant, train_seed=seed,
                      supervisor_pid=os.getpid(), session_id=os.getsid(0), state='waiting_for_previous_task',
                      output=str(output), previous_task=str(previous), time=time.time())
        write(status_path, status)
        try:
            while not completed(previous):
                time.sleep(15)
            with (ROOT / f'stage_gpu_locks/gpu_{gpu}.lock').open('a') as lock:
                while True:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        time.sleep(15)
                        continue
                    if not completed(previous):
                        fcntl.flock(lock, fcntl.LOCK_UN)
                        time.sleep(15)
                        continue
                    used = int(subprocess.check_output(
                        ['nvidia-smi', f'--id={gpu}', '--query-gpu=memory.used',
                         '--format=csv,noheader,nounits'], text=True).strip())
                    if used < 512:
                        break
                    fcntl.flock(lock, fcntl.LOCK_UN)
                    time.sleep(15)
                verify_source()
                command = [sys.executable, '-u', str(ROOT / 'codex/plan_v1_1/run_task.py'),
                           '--dataset', dataset, '--variant', variant, '--gpu', str(gpu),
                           '--seed', str(seed), '--output', str(output), '--sample-dir',
                           str(ROOT / 'experiments/sid_order_5seed_v1/fixed_test_users')]
                with (group / f'{task}.log').open('w') as log:
                    proc = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                            stdout=log, stderr=subprocess.STDOUT, close_fds=True)
                    status.update(state='running', pid=proc.pid, argv=command, time=time.time())
                    write(status_path, status)
                    code = proc.wait()
                if code != 0:
                    raise RuntimeError(f'{task} exited with code {code}')
                result = json.loads((output / 'stage_result.json').read_text())
                assert result.get('completed') is True, f'{task} did not complete tests and attention'
                status.update(state='completed', exit_code=0, time=time.time())
                write(status_path, status)
            previous = status_path
        except BaseException as exc:
            status.update(state='failed', error=str(exc), time=time.time())
            write(status_path, status)
            for later_task, later_dataset, later_variant, later_seed in tasks[index + 1:]:
                later_path = group / f'{later_task}_status.json'
                write(later_path, dict(task_id=later_task, gpu=gpu, dataset=later_dataset,
                     variant=later_variant, train_seed=later_seed, state='blocked_previous_failed',
                     previous_task=str(status_path), time=time.time()))
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=Path, required=True)
    parser.add_argument('--lane', type=int, choices=range(4))
    args = parser.parse_args()
    group = args.group.resolve()
    verify_source()
    if args.lane is not None:
        run_lane(group, args.lane)
        return
    assert not group.exists(), 'Queue group already registered'
    for gpu in range(4):
        prior = json.loads((ROOT / 'codex/third_round_20261104' / f'T{gpu + 9:02d}_status.json').read_text())
        assert prior['gpu'] == gpu and prior['train_seed'] == 44
        assert prior['state'] == 'completed' and prior.get('exit_code') == 0
        assert completed(ROOT / 'codex/third_round_20261104' / f'T{gpu + 9:02d}_status.json')
    for lane in LANES.values():
        for task, dataset, variant, seed in lane:
            assert not (ROOT / f'experiments/sid_order_5seed_v1/{dataset}/{variant}/seed_{seed}').exists()
    group.mkdir(parents=True)
    supervisors = []
    for gpu, lane in LANES.items():
        task, dataset, variant, seed = lane[0]
        log_path = group / f'GPU{gpu}_supervisor.log'
        with log_path.open('w') as log:
            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--group',
                                     str(group), '--lane', str(gpu)], cwd=ROOT,
                                    stdin=subprocess.DEVNULL, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        supervisors.append(dict(gpu=gpu, first_task=task, first_dataset=dataset,
                                first_variant=variant, first_seed=seed, pid=proc.pid,
                                tasks=[entry[0] for entry in lane]))
    manifest = dict(plan_version='1.1', queue='remaining_seeds_45_46', time=time.time(),
                    lanes=supervisors, automatic_restart=False,
                    sequence='seed45 tasks first; each GPU starts its seed46 task after its seed45 task fully completes')
    write(group / 'launch_manifest.json', manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
