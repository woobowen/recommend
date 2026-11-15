#!/usr/bin/env python3
"""Three-stage, bounded, same-code Beauty model recovery preflight.

This is not the formal training runner and cannot launch formal training.
Existing checkpoints and unrelated processes are never removed or terminated.
"""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

release = Path(sys.argv[1]).resolve(strict=True)
commit = sys.argv[2]
assert len(commit) == 40 and all(c in '0123456789abcdef' for c in commit)
manifest=json.loads((release/'export-manifest.json').read_text())
assert manifest['git_commit']==commit
for row in manifest['files']:
    relative=Path(row['path'])
    assert not relative.is_absolute() and '..' not in relative.parts
    path=release/relative
    assert path.is_file() and not path.is_symlink() and path.stat().st_size==row['bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
job = Path(os.environ['GRID_JOB_DIR']).resolve(strict=True)
out = job / 'output'
large = Path('/data/bwwu/lc-rec/a0-beauty-20261115') / os.environ['GRID_JOB_ID']
source = release / 'lc_rec/letter/vendor/LETTER/LETTER-LC-Rec'
script = release / 'lc_rec/a0_beauty/formal_train.py'
config = release / 'lc_rec/a0_beauty/config/formal_beauty_seed42.json'
approval = release / 'lc_rec/letter/evidence/formal_20261111/precision_approval.json'
data = Path('/home/bwwu/grid-audits/sid-extensions-p0/transfers/letter-a0-beauty-author-8d0154e28de3')
model = Path('/data/bwwu/lc-rec/assets/llama-7b/4782ad278652c7c71b72204d462d6d01eaaf7549')
for p in (source, script, config, approval, data / 'Beauty', model): assert p.exists(), str(p)
for name, expected in {
    'index': 'be3fb890e59b5443824cdebc560ef349191b97f20d3aecad4727de2d6b212c82',
    'inter': '8809584b8b78fa6561771c88911ed65bd716fc9926fca1e581e6a1ab578ccfd5',
    'item': '74aeddf911548aadfcd1dab326c16678ed4ac74457e874da300d63a94bdb2330',
}.items():
    assert hashlib.sha256((data/'Beauty'/('Beauty.'+name+'.json')).read_bytes()).hexdigest() == expected
assert os.environ['CUDA_VISIBLE_DEVICES'] == '0,1'
gpus = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used',
                                '--format=csv,noheader,nounits'], text=True)
rows = [line.split(', ') for line in gpus.strip().splitlines()]
assert [r[0] for r in rows] == ['GPU-dfff0e95-1db6-e2f4-3431-55bb852fa0ae',
                               'GPU-59cac3bd-3645-668b-79c8-839e0d878893']
assert all(int(r[1]) < 100 for r in rows)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                                    '--format=csv,noheader'], text=True).strip()
mem = {line.split(':')[0]: int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if len(line.split()) >= 2}
assert mem['MemAvailable'] >= 40*1024**3
assert shutil.disk_usage('/data').free >= 100*1024**3
assert shutil.disk_usage(str(out)).free >= 100*1024**3
assert not large.exists(), 'attempt directory already exists; query existing job'
large.mkdir(parents=True,exist_ok=False)

# Earlier three probes charged 117.2274 device-seconds conservatively. This
# stage reserves at most 2*1650 seconds, keeping the total below 3600.
wall_cap = 1650
record = {'job_id':os.environ['GRID_JOB_ID'], 'run_commit':commit,
          'started_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
          'gpu_start':gpus,'memory_available':mem['MemAvailable'],
          'large_output_path':str(large),
          'disk_data_free':shutil.disk_usage('/data').free,
          'prior_charged_device_seconds_upper_bound':117.2274,
          'wall_cap_seconds':wall_cap,'stages':[],'status':'STARTED'}
start=time.monotonic()
try:
    for mode in ('continuous','early','resumed','generation_interface'):
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                                            '--format=csv,noheader'], text=True).strip()
        root=large/mode
        assert not root.exists()
        launcher=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2']
        if mode=='generation_interface':
            argv=launcher+[str(release/'lc_rec/a0_beauty/formal_evaluate.py'),
                 '--source-root',str(source),'--data-root',str(data),'--model',str(model),
                 '--adapter',str(large/'continuous/preflight_adapter'),'--output',str(root),
                 '--run-commit',commit,'--preflight-users-per-rank','8']
        else:
            argv=launcher+[str(script),'--source-root',str(source),
                  '--data-root',str(data),'--model',str(model),'--output',str(root),
                  '--config',str(config),'--run-commit',commit,'--precision-approval',str(approval),
                  '--boundary-preflight']
            if mode=='early': argv += ['--stop-after-epoch','1']
            if mode=='resumed': argv += ['--resume',str(large/'early/checkpoint-2')]
        stage={'mode':mode,'argv':argv,'start_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
        record['stages'].append(stage)
        remaining=wall_cap-(time.monotonic()-start)
        assert remaining>0
        child=subprocess.Popen(argv)
        try:
            while child.poll() is None:
                remaining=wall_cap-(time.monotonic()-start)
                current_mem={line.split(':')[0]: int(line.split()[1])*1024
                             for line in Path('/proc/meminfo').read_text().splitlines() if len(line.split())>=2}
                disk=shutil.disk_usage('/data')
                observed={'utc':dt.datetime.now(dt.timezone.utc).isoformat(),
                          'mode':mode,'memory_available':current_mem['MemAvailable'],
                          'data_free':disk.free,
                          'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,utilization.gpu',
                                                        '--format=csv,noheader,nounits'],text=True)}
                with (out/'boundary_resources.jsonl').open('a') as stream:stream.write(json.dumps(observed)+'\n')
                if remaining<=0:raise TimeoutError('preflight budget exhausted')
                if current_mem['MemAvailable']<40*1024**3:raise RuntimeError('host memory safety reserve reached')
                if disk.free<max(100*1024**3,disk.total//10):raise RuntimeError('data disk safety reserve reached')
                output_disk=shutil.disk_usage(str(out))
                if output_disk.free<max(100*1024**3,output_disk.total//10):raise RuntimeError('output disk safety reserve reached')
                try:child.wait(timeout=min(20,remaining))
                except subprocess.TimeoutExpired:pass
            stage['exit_code']=child.returncode
        except BaseException as error:
            stage['interrupted']=repr(error);child.terminate()
            try: stage['exit_code']=child.wait(timeout=8)
            except subprocess.TimeoutExpired:child.kill();stage['exit_code']=child.wait()
            raise
        stage['end_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
        assert stage['exit_code']==0 and not stage.get('timed_out'), mode
    record['status']='ALL_FOUR_EXECUTED_AWAITING_INDEPENDENT_COMPARISON'
finally:
    record['wall_seconds']=time.monotonic()-start
    record['charged_device_seconds_upper_bound']=2*record['wall_seconds']
    (out/'boundary_guard.json').write_text(json.dumps(record,indent=2)+'\n')
assert record['status']=='ALL_FOUR_EXECUTED_AWAITING_INDEPENDENT_COMPARISON'
