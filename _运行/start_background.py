"""Launch a fresh experiment independently of the SSH connection."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['beauty', 'sports', 'toys'], required=True)
    parser.add_argument('--method', choices=['rkmeans', 'rqvae'], default='rkmeans')
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--embedding-batch-size', type=int, default=8)
    args = parser.parse_args()
    model = ROOT/'models/Qwen2.5-7B-Instruct'
    if not (model/'transfer_verified.json').is_file():
        parser.error('Model transfer verification must be complete before launching')
    name = f'{args.dataset}_{args.method}_qwen_{datetime.datetime.now():%Y%m%d_%H%M%S}'
    run = ROOT/'experiments'/name
    launches = ROOT/'launches'
    launches.mkdir(exist_ok=True)
    command = [sys.executable, str(ROOT/'_运行/run_pipeline.py'), '--dataset', args.dataset,
        '--method', args.method, '--gpus', args.gpus, '--embedding-batch-size', str(args.embedding_batch_size),
        '--run-dir', str(run)]
    log = launches/(name+'.log')
    with log.open('w') as output:
        process = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, PYTHONUNBUFFERED='1'),
            stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    report = dict(pid=process.pid, command=command, run_dir=str(run), launch_log=str(log),
        dataset=args.dataset, method=args.method, gpus=args.gpus, train=True, test=True,
        started_at=datetime.datetime.now().isoformat())
    (launches/(name+'.json')).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
