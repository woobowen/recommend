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
from preflight_budget import ledger
from preflight_subprocess import run

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

# Account for every existing Beauty job, including failed attempts. A separate
# cleanup reserve prevents a time-limit termination from spending the balance.
budget=ledger('/home/bwwu/grid-audits/sid-extensions-p0/jobs',os.environ['GRID_JOB_ID'])
wall_cap=budget['wall_cap_seconds']
record = {'job_id':os.environ['GRID_JOB_ID'], 'run_commit':commit,
          'started_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
          'gpu_start':gpus,'memory_available':mem['MemAvailable'],
          'large_output_path':str(large),
          'disk_data_free':shutil.disk_usage('/data').free,
          'budget':budget,
          'wall_cap_seconds':wall_cap,'stages':[],'status':'STARTED'}
start=time.monotonic()
try:
    for mode in ('continuous','early','resumed','compare','shuffle_bind','normal','shuffle_acceptance'):
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                                            '--format=csv,noheader'], text=True).strip()
        root=large/mode
        assert not root.exists()
        launcher=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2']
        user_manifest=release/'lc_rec/a0_beauty/config/preflight_users_seed42.json'
        if mode=='compare':
            argv=[sys.executable,str(release/'lc_rec/a0_beauty/boundary_compare.py'),
                  '--continuous',str(large/'continuous'),'--early',str(large/'early'),
                  '--resumed',str(large/'resumed'),'--output',str(out/'boundary_compare.json')]
        elif mode=='shuffle_bind':
            argv=[sys.executable,str(release/'lc_rec/a0_beauty/preflight_shuffle_bind.py'),
                  '--source-root',str(source),'--data-root',str(data),'--model',str(model),
                  '--adapter',str(large/'continuous/preflight_adapter'),'--manifest',str(user_manifest),
                  '--normal-test-dir',str(large/'normal'),'--output',str(root)]
        elif mode in ('normal','shuffle_acceptance'):
            entry=release/('lc_rec/a0_beauty/formal_evaluate.py' if mode=='normal' else 'lc_rec/a0_beauty/shuffle_entry.py')
            argv=launcher+[str(entry)]
            if mode=='shuffle_acceptance':
                argv+=['--model-key','a0','--shuffle-n','0','--acceptance',
                       '--cpu-audit',str(large/'shuffle_bind/audit.json')]
            argv += [
                 '--source-root',str(source),'--data-root',str(data),'--model',str(model),
                 '--adapter',str(large/'continuous/preflight_adapter'),'--output',str(root),
                 '--run-commit',commit,'--preflight-users-per-rank','16',
                 '--preflight-manifest',str(user_manifest)]
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
        def observe():
            remaining=wall_cap-(time.monotonic()-start)
            current_mem={line.split(':')[0]: int(line.split()[1])*1024
                         for line in Path('/proc/meminfo').read_text().splitlines() if len(line.split())>=2}
            disk=shutil.disk_usage('/data')
            observed={'utc':dt.datetime.now(dt.timezone.utc).isoformat(),
                      'mode':mode,'memory_available':current_mem['MemAvailable'],
                      'data_free':disk.free,
                      'charged_device_seconds_upper_bound':2*(time.monotonic()-start),
                      'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,utilization.gpu',
                                                    '--format=csv,noheader,nounits'],text=True)}
            with (out/'boundary_resources.jsonl').open('a') as stream:stream.write(json.dumps(observed)+'\n')
            if remaining<=0:raise TimeoutError('preflight budget exhausted')
            if current_mem['MemAvailable']<40*1024**3:raise RuntimeError('host memory safety reserve reached')
            if disk.free<max(100*1024**3,disk.total//10):raise RuntimeError('data disk safety reserve reached')
            output_disk=shutil.disk_usage(str(out))
            if output_disk.free<max(100*1024**3,output_disk.total//10):raise RuntimeError('output disk safety reserve reached')
        run(argv,start+wall_cap,stage,observe)
        stage['end_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
        assert stage['exit_code']==0 and not stage.get('timed_out'), mode
        if mode=='compare':
            assert json.loads((out/'boundary_compare.json').read_text())['status']=='PASS'
        if mode in ('normal','shuffle_acceptance'):
            for rank in (0,1):
                info=json.loads((root/('rank%d.json'%rank)).read_text())
                assert info['status']=='PREFLIGHT_INTERFACE_PASS' and info['completed_users']==16
                if mode=='shuffle_acceptance':
                    result=json.loads((root/('acceptance.rank%d.json'%rank)).read_text())
                    assert result['status']=='PASS' and result['users']==16
    record['status']='ALL_STAGES_EXECUTED_AWAITING_REVIEW'
except BaseException as error:
    record['status']='TIMED_OUT' if isinstance(error,TimeoutError) else 'FAILED'
    record['error']=repr(error)
    raise
finally:
    record['wall_seconds']=time.monotonic()-start
    record['charged_device_seconds_upper_bound']=2*record['wall_seconds']
    (out/'boundary_guard.json').write_text(json.dumps(record,indent=2)+'\n')
assert record['status']=='ALL_STAGES_EXECUTED_AWAITING_REVIEW'
