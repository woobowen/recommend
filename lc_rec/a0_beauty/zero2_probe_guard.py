#!/usr/bin/env python3
"""Bounded launch guard for the small ZeRO-2 observational diagnostic."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

probe = Path(sys.argv[1]).resolve()
assert probe.is_file() and os.environ['CUDA_VISIBLE_DEVICES'] == '0,1'
expected = ['GPU-dfff0e95-1db6-e2f4-3431-55bb852fa0ae', 'GPU-59cac3bd-3645-668b-79c8-839e0d878893']
gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used', '--format=csv,noheader,nounits'], text=True)
rows = [line.split(', ') for line in gpu.strip().splitlines()]
assert [r[0] for r in rows] == expected and all(int(r[1]) < 100 for r in rows)
active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
assert not active.strip(), 'GPU already in use'
mem = {line.split(':')[0]: int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if len(line.split()) >= 2}
assert mem['MemAvailable'] >= 40 * 1024**3
assert shutil.disk_usage('/data').free >= 100 * 1024**3
job = Path(os.environ['GRID_JOB_DIR'])
command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
           '--nproc_per_node=2', str(probe)]
record = {'job_id': os.environ['GRID_JOB_ID'], 'argv': command, 'gpu_start': gpu,
          'mem_available': mem['MemAvailable'], 'disk_free': shutil.disk_usage('/data').free,
          'budget_max_device_seconds': 360, 'scope':'small synthetic observation, not Beauty model acceptance'}
start = time.monotonic()
try:
    child = subprocess.Popen(command)
    try:
        record['exit_code'] = child.wait(timeout=150)
    except subprocess.TimeoutExpired:
        # torchrun workers can own separate process groups. Give torchrun's
        # signal handler time to shut down its workers before the runner's
        # outer deadline; the runner still verifies task-wide cleanup.
        record['timed_out'] = True
        child.terminate()
        try:
            record['exit_code'] = child.wait(timeout=8)
        except subprocess.TimeoutExpired:
            child.kill()
            record['exit_code'] = child.wait()
finally:
    record['wall_seconds'] = time.monotonic() - start
    record['charged_device_seconds_upper_bound'] = 2 * record['wall_seconds']
    (job/'output'/'guard.json').write_text(json.dumps(record, indent=2)+'\n')
assert record['exit_code'] == 0
