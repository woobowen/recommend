"""Capture a read-only formal Beauty observation and complete-line watermarks."""
import argparse
import json
from pathlib import Path

from lc_rec.shuffle_v2 import beauty_control as bc, control

PARENT = 'letter-beauty-v2-group-series-r0'
MODES = ('shuffle2', 'shuffle3a', 'shuffle3b', 'shuffle4')
USERS = 22363

REMOTE_OBSERVATION = r'''
from datetime import datetime, timezone
import hashlib,json,os,shutil,subprocess
from pathlib import Path

now = datetime.now(timezone.utc)
boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
parent_dir = jobs / parent_id
parent_request = json.loads((parent_dir/'request.json').read_text())
plan_path = Path(parent_request['argv'][parent_request['argv'].index('--plan')+1])
plan = json.loads(plan_path.read_text())
result = {'observed_utc': now.isoformat(), 'boot_id': boot, 'parent_id': parent_id, 'jobs': {},
          'outputs': {}, 'rank_reports': {}, 'log_tails': {}, 'gpu_records': [], 'gpu_compute': [],
          'disk_free_bytes': shutil.disk_usage(root).free}
series = root / parent_id.removeprefix(prefix) / 'summary.json'
if series.is_file():
    result['controller_summary'] = json.loads(series.read_text())

def alive(identity, job_path, record_markers=True):
    if not identity:
        return None
    pid = identity['pid']
    proc = Path('/proc') / str(pid)
    try:
        values = (proc/'stat').read_text().rsplit(')',1)[1].split()
        if (values[0] == 'Z' or values[19] != identity['start_ticks']
                or boot != identity['boot_id'] or proc.stat().st_uid != identity['uid']):
            return None
        argv = [os.fsdecode(x) for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
        evidence = {'identity': identity, 'argv': argv, 'state': values[0],
                    'pgid': int(values[2]), 'session_id': int(values[3])}
        if record_markers:
            env = (proc/'environ').read_bytes().split(b'\0')
            assert ('GRID_JOB_ID='+job_path.name).encode() in env
            assert ('GRID_JOB_DIR='+str(job_path)).encode() in env
            selected = {b'CUDA_VISIBLE_DEVICES', b'CUDA_HOME', b'LD_LIBRARY_PATH', b'WORLD_SIZE',
                        b'RANK', b'LOCAL_RANK', b'TORCHINDUCTOR_COMPILE_THREADS'}
            evidence['effective_env'] = {os.fsdecode(key):os.fsdecode(value) for item in env if b'=' in item
                for key,value in [item.split(b'=',1)] if key in selected}
        return evidence
    except (FileNotFoundError, ProcessLookupError):
        return None

def tail(path):
    with path.open('rb') as handle:
        end = os.fstat(handle.fileno()).st_size
        start = max(0, end-4096)
        handle.seek(start)
        raw = handle.read(end-start)
    return {'source':str(path),'start_byte':start,'end_byte_exclusive':end,
            'text_utf8_with_replacement':raw.decode('utf-8','replace')}

def count_output(path, previous):
    with path.open('rb') as handle:
        stat = os.fstat(handle.fileno())
        inode = [stat.st_dev,stat.st_ino]
        start = previous.get('complete_end_byte',0)
        if previous:
            assert previous['inode']==inode and start<=stat.st_size, path
        handle.seek(start)
        cursor, end, lines = start, start, previous.get('complete_lines',0)
        while cursor < stat.st_size:
            block = handle.read(min(2**20, stat.st_size-cursor))
            if not block:
                raise RuntimeError('append-only snapshot shortened: '+str(path))
            count = block.count(b'\n')
            if count:
                lines += count
                end = cursor + block.rfind(b'\n') + 1
            cursor += len(block)
        return {'source': str(path), 'inode': inode, 'snapshot_size':stat.st_size,
                'mtime_ns': stat.st_mtime_ns, 'counted_from_byte':start, 'complete_end_byte':end,
                'complete_lines':lines, 'candidate_count_by_enforced_20_per_row':20*lines,
                'trailing_bytes_outside_complete_rows':stat.st_size-end}

children = {entry['job_id']: entry for stage in plan['modes'].values() for entry in stage.values()}
children[plan['paired']['job_id']] = plan['paired']
handled = {parent_id: parent_request, **children}
matched_pids = {}
for job_id, request in handled.items():
    directory = jobs / job_id
    status_path = directory/'status.json'
    if not status_path.is_file():
        continue
    status = json.loads(status_path.read_text())
    group = status.get('worker_group') or {}
    running = not status.get('ended_at')
    live = [value for member in group.get('known_members',[]) if running and
            (value := alive(member,directory))]
    matched_pids.update({item['identity']['pid']:job_id for item in live})
    supervisor = alive(status.get('supervisor'),directory,False) if running else None
    entry = {key:status.get(key) for key in ('job_id','status','started_at','ended_at','received_at',
                                            'exit_code','cleanup','worker','supervisor')}
    entry.update(live_members=live,supervisor_alive=bool(supervisor),
                 request_sha256=hashlib.sha256(json.dumps(request,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest())
    result['jobs'][job_id] = entry
    folder = Path(request['argv'][request['argv'].index('--output')+1])
    for log in [directory/'work.log', *sorted(folder.glob('shuffle*.log'))]:
        if log.is_file() and (running or job_id==parent_id):
            result['log_tails'][str(log)] = tail(log)
    if job_id.startswith(prefix+'gpu-formal-') and folder.is_dir():
        report = folder/'test/rank0.json'
        if report.is_file():
            rank = json.loads(report.read_text())
            result['rank_reports'][job_id] = {name:rank.get(name) for name in ('status','pid','rank','world_size',
                'local_rank','assigned_users','completed_users','RUN_COMMIT','assignment_sha256',
                'elapsed_generation_seconds','gpu_peak_allocated_bytes','gpu_peak_reserved_bytes','error')}
        prediction = folder/'test/predictions.rank0.jsonl'
        if prediction.is_file():
            result['outputs'][job_id] = count_output(prediction, watermarks.get(job_id,{}))

for line in subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],
        text=True).splitlines():
    physical,uuid,used,total,util = [field.strip() for field in line.split(',')]
    result['gpu_records'].append({'physical_index':int(physical),'uuid':uuid,
        'memory_used_mib':int(used),'memory_total_mib':int(total),'utilization_percent':int(util)})
for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory',
        '--format=csv,noheader,nounits'],text=True).splitlines():
    if not line.strip(): continue
    uuid,pid,used = [field.strip() for field in line.split(',')]
    result['gpu_compute'].append({'uuid':uuid,'pid':int(pid),'memory_mib':int(used),
                                  'owned_formal_job_id':matched_pids.get(int(pid))})
known_paths = set(handled)
result['unexpected_active_formal'] = []
spend = {'cpu_worker_seconds':0.,'preflight_device_seconds':0.,'formal_device_seconds':0.}
for path in jobs.glob(prefix+'*'):
    status_path = path/'status.json'
    if not status_path.is_file(): continue
    row = json.loads(status_path.read_text())
    start = row.get('started_at')
    if row.get('ended_at') is None and path.name not in known_paths and path.name.startswith(
            (prefix+'gpu-formal-',prefix+'group-series-')):
        result['unexpected_active_formal'].append(path.name)
    if not start: continue
    elapsed = ((datetime.fromisoformat(row['ended_at']) if row.get('ended_at') else now)
               -datetime.fromisoformat(start)).total_seconds()
    kind = path.name.removeprefix(prefix)
    for tag,key in [('cpu-','cpu_worker_seconds'),('gpu-preflight-','preflight_device_seconds'),
                    ('gpu-formal-','formal_device_seconds')]:
        if kind.startswith(tag):
            spend[key]+=max(0.,elapsed)
            break
result['cumulative_budget_spend'] = spend
result['capture_finished_utc'] = datetime.now(timezone.utc).isoformat()
print(json.dumps(result))
'''


def observe(connection, parent, previous):
    variables = dict(jobs=Path(connection.profile['job_root']), root=Path(bc.ROOT),
                     parent_id=parent, prefix=bc.PREFIX, watermarks=previous)
    initializer = 'from pathlib import Path, PosixPath\n' + '\n'.join(
        f'{key} = {value!r}' for key, value in variables.items())
    return json.loads(control.remote_python(connection, initializer + '\n' + REMOTE_OBSERVATION, timeout=45))


def save_observation(conn, parent, output):
    if output.exists():
        return json.loads(output.read_text())
    older = sorted(output.parent.glob('t[0-9][0-9].json'))
    previous = json.loads(older[-1].read_text())['outputs'] if older else {}
    report = observe(conn, parent, previous)
    report['previous_observation'] = older[-1].name if older else None
    report['previous_output_watermarks'] = {name:row['complete_end_byte'] for name,row in previous.items()}
    bc.immutable(output, report)
    return report


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--parent', default=PARENT)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    r = save_observation(control.connect(), args.parent, args.output)
    print(json.dumps({'observed_utc':r['observed_utc'],'parent_status':r['jobs'][args.parent]['status'],
                      'rank_reports':r['rank_reports'],'outputs':r['outputs'],'gpu_compute':r['gpu_compute']},
                     indent=2))


if __name__ == '__main__':
    main()
