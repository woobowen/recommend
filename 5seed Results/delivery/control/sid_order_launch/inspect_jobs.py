"""Read compact experiment status without changing running jobs."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    group = args.output
    manifest = json.loads((group / 'launch_manifest.json').read_text())
    snapshot = {'output': str(group), 'jobs': {}}
    for lane in manifest['lanes']:
        for name in lane['jobs']:
            folder = group / name
            entry = {'gpu': lane['gpu'], 'state': 'queued'}
            for file, key in [(group / f'{name}_job.json', 'job'),
                              (folder / 'model_verified.json', 'model'),
                              (folder / 'progress.json', 'progress'),
                              (folder / 'stage_result.json', 'result')]:
                if file.is_file():
                    entry[key] = json.loads(file.read_text())
            if 'job' in entry:
                entry['state'] = entry['job']['state']
                entry['pid'] = entry['job'].get('pid')
            log = group / f'{name}.log'
            if log.is_file():
                entry['log_tail'] = log.read_text(errors='replace').splitlines()[-4:]
            snapshot['jobs'][name] = entry
    snapshot['gpus'] = subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,memory.used,utilization.gpu', '--format=csv'], text=True)
    snapshot['processes'] = [line for line in subprocess.check_output(
        ['ps', '-u', 'jcyin', '-o', 'pid,ppid,args'], text=True).splitlines()
        if 'sid_order_launch/launch_jobs.py' in line or 'sid_order_launch/run_variant.py' in line]
    print(json.dumps(snapshot, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
