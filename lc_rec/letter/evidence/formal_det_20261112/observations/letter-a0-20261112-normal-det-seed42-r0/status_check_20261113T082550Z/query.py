"""One-time stdlib read-only query, invoked by existing Windows SSH transport.

No training/evaluation process is launched. No remote file is written.
Large checkpoint/model/prediction payloads are only stat'ed, never loaded.
"""
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys

RUN = '4c6a83d9ba42dc2fe737792759e410ef1874da76'
JOB_ID = 'letter-a0-20261112-normal-det-seed42-r0'
RELEASE = Path('/home/bwwu/lc-rec/letter-a0-normal-det-20261112/releases') / RUN
ROOT = Path('/data/bwwu/lc-rec/a0-normal-20261111')
JOB = Path('/home/bwwu/grid-audits/sid-extensions-p0/jobs') / JOB_ID
TRAIN = ROOT / 'formal_seed42/training'
TEST = ROOT / JOB_ID / 'test'
sys.path.insert(0, str(RELEASE / 'lc_rec/letter/scripts'))
from formal_train import verify_checkpoint


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def sha_small(path):
    assert path.stat().st_size <= 8 * 1024**2
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tail(path, count=65536):
    if not path.is_file():
        return None
    with path.open('rb') as stream:
        size = path.stat().st_size
        offset = max(0, size - count)
        stream.seek(offset)
        raw = stream.read()
    return {'path': str(path), 'total_bytes': size, 'offset': offset,
            'text': raw.decode('utf-8', 'replace')}


def file_meta(path):
    return {'path': str(path), 'bytes': path.stat().st_size,
            'mtime_utc': dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc).isoformat()}


def directory_size(root):
    files = [p for p in root.rglob('*') if p.is_file() and not p.is_symlink()] if root.exists() else []
    return {'path': str(root), 'files': len(files),
            'logical_bytes': sum(p.stat().st_size for p in files),
            'allocated_bytes': sum(p.stat().st_blocks * 512 for p in files)}


observed = dt.datetime.now(dt.timezone.utc).isoformat()
result = {'observed_utc': observed, 'job_id': JOB_ID, 'RUN_COMMIT': RUN,
          'server': {'hostname': socket.gethostname(), 'uid': os.getuid(),
                     'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()},
          'raw_json': {}, 'source_paths': {}, 'training': {}, 'checkpoints': [],
          'scope': 'One-time lightweight status cross-check; no GPU work, repair, model loading, full prediction read, metric recomputation or final scientific audit'}
for key, path in {
    'request': JOB / 'request.json', 'runner_status': JOB / 'status.json',
    'accepted': JOB / 'accepted.json', 'attempt_claim': JOB / 'attempt.json',
    'cleanup': JOB / 'cleanup.json', 'pipeline': JOB / 'output/pipeline.json',
    'ledger': ROOT / 'formal_seed42_ledger.json',
    'best_selection': TRAIN / 'best_selection.json', 'trainer_state': TRAIN / 'trainer_state.json',
    'metrics': TEST / 'metrics.json',
}.items():
    value = read(path)
    if key == 'trainer_state' and value:
        value = {**value, 'log_history': [v for v in value.get('log_history', [])
                 if 'eval_loss' in v] + value.get('log_history', [])[-6:]}
    result['raw_json'][key] = value
    result['source_paths'][key] = {'path': str(path),
                                  'sha256': sha_small(path) if path.is_file() else None,
                                  'capture': 'JSON parsed; trainer_state log_history excerpt only' if key == 'trainer_state' else 'complete JSON'}

request = result['raw_json']['request']
result['request_identity'] = {
    'file_sha256': sha_small(JOB / 'request.json'),
    'canonical_json_sha256': hashlib.sha256(json.dumps(request, sort_keys=True,
        ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()}
config = RELEASE / 'lc_rec/letter/config/formal_instruments_seed42.json'
result['config_sha256'] = sha_small(config)
for rank in (0, 1):
    path = TRAIN / ('rank%d.json' % rank)
    report = read(path)
    keys = ['status', 'run_commit', 'rank', 'argv', 'cwd', 'python', 'started_utc',
            'config_sha256', 'global_step', 'epoch', 'best_checkpoint', 'best_metric',
            'initial_global_step', 'elapsed_seconds', 'metrics', 'formal_attention',
            'data', 'data_identity', 'base_model_identity', 'error', 'traceback',
            'peak_allocated_bytes', 'peak_reserved_bytes']
    record = {'report_excerpt': {k: report[k] for k in keys if k in report} if report else None,
              'report_source': {'path': str(path), 'sha256': sha_small(path) if path.is_file() else None}}
    events_path = TRAIN / ('events.rank%d.jsonl' % rank)
    events = [json.loads(line) for line in events_path.read_text().splitlines()] if events_path.is_file() else []
    milestones = [v for v in events if v['event'] in
        ['epoch_end', 'checkpoint_complete', 'strict_ds_restored', 'train_begin',
         'formal_fa_backward_dispatch', 'formal_fa_installed', 'loader_replayed']
        or v['event'] == 'log' and 'eval_loss' in v.get('values', {})]
    updates = [v for v in events if v['event'] == 'optimizer_step']
    logs = [v for v in events if v['event'] == 'log' and 'loss' in v.get('values', {})]
    numeric = []
    for event in events:
        pairs = [('grad_norm', event.get('grad_norm'))]
        pairs += list(event.get('values', {}).items()) if event['event'] == 'log' else []
        numeric.extend({'event': event['event'], 'utc': event['utc'], 'key': key, 'value': str(value)}
                       for key, value in pairs if type(value) in (int, float) and not math.isfinite(value))
    record.update(event_source=str(events_path), event_count=len(events), raw_milestones=milestones,
                  last_optimizer_events=updates[-6:], last_train_loss_events=logs[-6:],
                  nonfinite_training_event_values=numeric)
    test_path = TEST / ('rank%d.json' % rank)
    test_report = read(test_path)
    result['raw_json']['test_rank%d' % rank] = test_report
    result['source_paths']['test_rank%d' % rank] = {
        'path': str(test_path), 'sha256': sha_small(test_path) if test_path.is_file() else None}
    result['training'][str(rank)] = record

for cp in sorted(TRAIN.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1])):
    row = {'path': str(cp), 'complete_marker_present': (cp / 'complete.json').is_file(),
           'validation': 'IN_PROGRESS' if not (cp / 'complete.json').is_file() else 'PENDING'}
    if row['complete_marker_present']:
        try:
            row['complete_manifest'] = verify_checkpoint(cp, RUN, result['config_sha256'])
            row['validation'] = 'PASS'
        except Exception as error:
            row.update(validation='FAIL', error=repr(error), complete_manifest=read(cp / 'complete.json'))
        state = read(cp / 'trainer_state.json')
        row['trainer_state_excerpt'] = {k: v for k, v in state.items() if k != 'log_history'} if state else None
        row['latest'] = (cp / 'latest').read_text().strip() if (cp / 'latest').is_file() else None
    result['checkpoints'].append(row)

best = TRAIN / 'best_adapter'
result['adapter_files'] = [file_meta(p) for p in sorted(best.glob('*')) if p.is_file()]
result['prediction_files_stat_only'] = [file_meta(p) for p in sorted(TEST.glob('predictions.rank*.jsonl'))]
result['failure_files'] = [str(p) for root in (TRAIN, TEST) for p in root.glob('failure*.json')]
result['work_log_tail'] = tail(JOB / 'work.log')
result['supervisor_log_tail'] = tail(JOB / 'supervisor.log', 8192)
result['gpu_csv'] = subprocess.check_output(['nvidia-smi',
    '--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,temperature.gpu',
    '--format=csv'], text=True)
result['compute_apps_csv'] = subprocess.check_output(['nvidia-smi',
    '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv'], text=True)
mem = {line.split(':')[0]: int(line.split()[1]) * 1024
       for line in Path('/proc/meminfo').read_text().splitlines() if len(line.split()) >= 2}
result['host_memory'] = {key: mem[key] for key in ['MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree']}
disk = os.statvfs(ROOT)
result['disk'] = {'available_bytes': disk.f_bavail * disk.f_frsize,
                  'total_bytes': disk.f_blocks * disk.f_frsize,
                  'safety_reserve_bytes': max(100 * 1024**3, disk.f_blocks * disk.f_frsize // 10)}
result['directory_sizes'] = [directory_size(path) for path in (TRAIN, TEST, JOB)]
resources_path = JOB / 'output/resources.jsonl'
resources = [json.loads(line) for line in resources_path.read_text().splitlines()] if resources_path.is_file() else []
result['resources'] = {
    'source': str(resources_path), 'samples': len(resources),
    'first': resources[0] if resources else None, 'last': resources[-1] if resources else None,
    'min_host_available_bytes': min((v['mem_available'] for v in resources), default=None),
    'max_swap_used_bytes': max((v['swap_used'] for v in resources), default=None),
    'min_disk_available_bytes': min((v['disk_available'] for v in resources), default=None),
    'max_seconds_without_worklog_progress': max((v['seconds_without_worklog_progress'] for v in resources), default=None)}
result['recorded_processes'] = []
status = result['raw_json']['runner_status'] or {}
identities = [status.get('supervisor'), status.get('worker')]
identities += status.get('worker_group', {}).get('known_members', [])
seen = set()
for identity in identities:
    if not identity or identity['pid'] in seen:
        continue
    seen.add(identity['pid'])
    result['recorded_processes'].append({'identity': identity,
                                       'proc_directory_exists': Path('/proc/%d' % identity['pid']).exists()})
result['query_finished_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
print(json.dumps(result, ensure_ascii=False, allow_nan=False))
