"""Run an explicit TIGER command as a detached, stage-locked job."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from run_pipeline import reserve_stage_gpus

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text(encoding='utf-8'))
    run, stage = Path(job['run_dir']), Path(job['stage_dir'])
    gpu = str(job['physical_gpu'])
    status_path = run/'pipeline_status.json'
    status = dict(stage=stage.name, state='waiting_for_gpu', output=str(stage),
                  gpus=gpu, job_file=str(args.job), validation_interval_batches=1600)
    def write_status():
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(status, indent=2), encoding='utf-8')
        temporary.replace(status_path)
    def waiting(reason):
        status.update(state='waiting_for_gpu', reason=reason)
        write_status()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='4',
               PYTHONUNBUFFERED='1', TF_CPP_MIN_LOG_LEVEL='3', HYDRA_FULL_ERROR='1',
               TF_NUM_INTRAOP_THREADS='2', TF_NUM_INTEROP_THREADS='2')
    for key in ('PROJECT_ROOT', 'METADATA_DIR', 'LOCAL_RANK', 'RANK', 'WORLD_SIZE',
                'MASTER_ADDR', 'MASTER_PORT'):
        env.pop(key, None)
    stage.mkdir(parents=True, exist_ok=True)
    (stage/'command_val1600.json').write_text(json.dumps(job, indent=2), encoding='utf-8')
    write_status()
    with reserve_stage_gpus([gpu], waiting):
        status.update(state='running', started_at=time.strftime('%Y-%m-%d %H:%M:%S %z'))
        status.pop('reason', None)
        write_status()
        with (stage/job['console_filename']).open('w') as output:
            process = subprocess.Popen(job['argv'], cwd=ROOT, env=env,
                                       stdout=output, stderr=subprocess.STDOUT)
            status['training_pid'] = process.pid
            write_status()
            returncode = process.wait()
        status.update(state='completed' if returncode == 0 else 'failed', exit_code=returncode)
        write_status()
    if returncode:
        raise SystemExit(returncode)


if __name__ == '__main__':
    main()
