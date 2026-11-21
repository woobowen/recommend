"""Detach four GPU lanes, preserving existing work and recording every job."""
import argparse
from datetime import datetime
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lane', type=int)
    args = parser.parse_args()
    group = args.output.resolve()
    jobs = [[('sports', 'mlp')], [('sports', 'shuffle')],
            [('beauty', 'mlp'), ('toys', 'mlp')], [('beauty', 'shuffle'), ('toys', 'shuffle')]]
    if args.lane is None:
        if group.exists():
            parser.error('Use a fresh output directory')
        group.mkdir(parents=True)
        manifest = {'output': str(group), 'started_at': datetime.now().isoformat(),
                    'inputs': 'Own baseline Qwen/RQ-KMeans/SIDs; recommendation ckpt_path=null',
                    'train': True, 'test': True, 'lanes': []}
        for gpu, lane in enumerate(jobs):
            log_path = group / f'lane_{gpu}.log'
            command = [sys.executable, str(Path(__file__).resolve()), '--output', str(group), '--lane', str(gpu)]
            with log_path.open('w') as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL, start_new_session=True)
            manifest['lanes'].append({'gpu': gpu, 'pid': process.pid,
                                     'jobs': [f'{d}_{v}' for d, v in lane], 'log': str(log_path)})
        write_json(group / 'launch_manifest.json', manifest)
        print(json.dumps(manifest, indent=2), flush=True)
        return
    gpu = str(args.lane)
    for dataset, variant in jobs[args.lane]:
        name = f'{dataset}_{variant}'
        output = group / name
        status_file = group / f'{name}_job.json'
        command = [sys.executable, str(ROOT / 'codex/sid_order_launch/run_variant.py'),
                   '--dataset', dataset, '--variant', variant, '--gpu', gpu, '--output', str(output)]
        status = {'dataset': dataset, 'variant': variant, 'gpu': gpu, 'argv': command,
                  'state': 'waiting_for_gpu', 'train': True, 'test': True, 'ckpt_path': None}
        write_json(status_file, status)
        while True:
            stack = ExitStack()
            lock = stack.enter_context((ROOT / f'stage_gpu_locks/gpu_{gpu}.lock').open('a'))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                rows = subprocess.check_output(['nvidia-smi', f'--id={gpu}',
                    '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
                if int(rows.strip()) < 512:
                    break
            except BlockingIOError:
                pass
            stack.close()
            time.sleep(15)
        try:
            env = dict(os.environ, PYTHONUNBUFFERED='1')
            with (group / f'{name}.log').open('w') as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                status.update(state='running', pid=child.pid, output=str(output))
                write_json(status_file, status)
                print(f'{name} started pid={child.pid} gpu={gpu}', flush=True)
                code = child.wait()
            status.update(state='completed' if code == 0 else 'failed', exit_code=code)
            write_json(status_file, status)
            print(f'{name} exited {code}', flush=True)
        finally:
            stack.close()
        if code:
            raise RuntimeError(f'{name} failed; following jobs in this lane remain queued')


if __name__ == '__main__':
    main()
