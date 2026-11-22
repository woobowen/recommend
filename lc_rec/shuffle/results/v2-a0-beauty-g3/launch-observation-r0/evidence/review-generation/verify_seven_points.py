"""Independent read-only consistency checks for seven saved G3 S2 observations."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

MODE = 'shuffle2'
TOTAL_USERS = 22363
MODEL_SHA = '3315622866cd97833741608ca21d82b72a10184ab8bff4dc688892146d0bf915'
ASSIGNMENT_SHA = '22ee58fe9f948f50e5cbc92160b6dc7df7e2f633cb9e8bf3121da5fb35bcb088'
GPU_COMMIT = '86660be5a62c8325caf74cf2810581850ca12803'
EASTERN = ZoneInfo('America/New_York')


def read(path):
    return json.loads(path.read_bytes())


def proc_key(row):
    return tuple(row[key] for key in ('boot_id', 'start_ticks', 'uid', 'pid'))


def check_budget(point, initial):
    now = datetime.fromisoformat(point['utc'])
    caps = {'formal_gpu': 129600, 'preflight_gpu': 10800, 'cpu': 28800, 'parent_wall': 126000}
    ledger = point['budget']
    assert set(ledger) == set(caps)
    for kind, cap in caps.items():
        info = ledger[kind]
        periods = info['jobs']
        seconds = []
        for row in periods.values():
            start, end = row['started_at'], row['ended_at']
            elapsed = max(0, ((datetime.fromisoformat(end) if end else now)
                              - datetime.fromisoformat(start)).total_seconds()) if start else 0
            assert math.isclose(elapsed, row['worker_seconds'], abs_tol=1e-5)
            seconds.append(elapsed)
        total = math.fsum(seconds)
        assert info['limit_seconds'] == cap
        assert math.isclose(total, info['used_seconds'], abs_tol=1e-5)
        assert math.isclose(cap - total, info['remaining_seconds'], abs_tol=1e-5)
        assert 0 < info['remaining_seconds'] <= cap
    assert math.isclose(ledger['preflight_gpu']['used_seconds'], initial['past']['preflight_gpu_seconds'], abs_tol=1e-5)
    assert math.isclose(ledger['cpu']['used_seconds'], initial['past']['cpu_worker_seconds'], abs_tol=1e-5)
    assert set(ledger['formal_gpu']['jobs']) == {point['child']['job_id']}
    assert set(ledger['parent_wall']['jobs']) == {point['parent']['job_id']}
    assert ledger['parent_wall']['used_seconds'] >= ledger['formal_gpu']['used_seconds']


def verify_points(work):
    plan = read(work/'formal-plan.json')
    first = plan['modes'][MODE]['gpu']
    parent_id, child_id = plan['parent']['name'], first['name']
    budget_before = read(work/'budget-before.json')
    members = [f'T{i:02d}' for i in range(0,31,5)]
    by_label, result = {}, []
    t0_ns, signature, file_identity = None, None, None
    preceding = {'complete_users': 0, 'complete_bytes': 0, 'bytes': 0}
    log_sizes = {}
    for ordinal, label in enumerate(members):
        path = work/'observations'/(label+'.json')
        point = read(path)
        by_label[label] = point
        assert point['label'] == label
        local = datetime.fromisoformat(point['america_new_york'])
        utc = datetime.fromisoformat(point['utc'])
        assert utc.tzinfo == timezone.utc and utc.astimezone(EASTERN).isoformat() == local.isoformat()
        assert abs(utc.timestamp() - point['unix_ns']/1e9) < 0.00001
        if t0_ns is None:
            t0_ns = point['unix_ns']
        elapsed = (point['unix_ns'] - t0_ns)/1e9
        assert math.isclose(elapsed, point['elapsed_since_t0_seconds'], abs_tol=1e-6)
        assert abs(elapsed - ordinal*300) <= 60, (label, elapsed)
        for state, job in ((point['parent'], parent_id), (point['child'], child_id)):
            assert state['job_id'] == job and state['status'] == 'RUNNING' and state['exit_code'] is None
            assert state['supervisor_alive'] and state['worker_alive'] and state['workgroup_live']
            assert state['workgroup_observation']['scan_complete']
            assert not state['workgroup_observation']['unconfirmed_live']
            assert not state['workgroup_observation']['scan_errors']
        controller = point['controller']
        assert controller['status'] == 'RUNNING' and [s['job_id'] for s in controller['stages']] == [child_id]
        rank, = [p for p in point['gpu_rank_processes'] if p['proc_parent_pid']==point['child']['worker']['pid']]
        assert any(proc_key(p) == proc_key(rank) for p in point['child']['workgroup_observation']['owned_live'])
        env = rank['rank_env']
        gpu = point['gpu']
        assert gpu['uuid'] == plan['card'] and gpu['name'] == 'NVIDIA GeForce RTX 4090'
        assert (env['CUDA_VISIBLE_DEVICES'], env['WORLD_SIZE'], env['RANK'], env['LOCAL_RANK'], env['GRID_JOB_ID']) == (
            plan['card'], '1', '0', '0', child_id)
        assert rank['cwd'].endswith(GPU_COMMIT) and env['DS_ACCELERATOR'] == 'cuda'
        assert env['TORCHINDUCTOR_COMPILE_THREADS'] == '1'
        this_signature = (proc_key(rank), proc_key(point['parent']['worker']), proc_key(point['parent']['supervisor']),
                          proc_key(point['child']['worker']), proc_key(point['child']['supervisor']),
                          point['parent']['request_sha256'], point['child']['request_sha256'], gpu['uuid'], gpu['physical_index'])
        if signature is None:
            signature = this_signature
        assert this_signature == signature
        rows = point['prediction']
        assert rows['complete_users'] == preceding['complete_users'] + rows['delta_users']
        assert rows['bytes'] == preceding['bytes'] + rows['delta_bytes']
        assert rows['bytes'] == rows['complete_bytes'] + rows['buffered_suffix_bytes']
        assert 0 <= rows['complete_users'] < TOTAL_USERS
        if 'delta_offset_start' in rows:
            assert rows['delta_offset_start'] == preceding['complete_bytes']
            assert rows['complete_bytes'] == rows['delta_offset_start'] + rows['delta_complete_bytes']
            assert len(bytes.fromhex(rows['delta_complete_sha256'])) == 32
            if file_identity is None:
                file_identity = (rows['device'], rows['inode'])
            assert (rows['device'], rows['inode']) == file_identity
        apps = [app for app in point['compute_apps'] if app['uuid'] == plan['card']]
        assert all(int(app['pid']) == rank['pid'] for app in apps)
        if ordinal > 0:
            assert rows['delta_users'] > 0 and rows['delta_bytes'] > 0 and len(apps) == 1
            runtime = point['rank']
            assert runtime['RUN_COMMIT'] == GPU_COMMIT and runtime['model_identity_sha256'] == MODEL_SHA
            assert runtime['scope'] == 'FORMAL' and runtime['assigned_users'] == TOTAL_USERS
            assert runtime['assignment_sha256'] == ASSIGNMENT_SHA and runtime['conditions'] == [MODE]
            assert (runtime['world_size'], runtime['local_rank'], runtime['rank']) == (1, 0, 0)
            assert preceding['complete_users'] <= runtime['completed_users'] <= rows['complete_users']
            # Production reports each 25 rows. Allow the snapshot's small opposite-order read lag.
            assert rows['complete_users'] - runtime['completed_users'] <= 30
        assert point['output_free_bytes'] >= 20*1024**3 and not point['error_matches']
        for name, log in point['logs'].items():
            if log is None:
                continue
            previous = log_sizes.get(name,0)
            assert log['offset_end'] == log['bytes'] >= log['offset_start'] >= previous
            assert log['delta_bytes'] == log['bytes'] - previous
            if '\ufffd' not in log['excerpt']:
                assert sha256(log['excerpt'].encode()).hexdigest() == log['excerpt_sha256']
            log_sizes[name] = log['bytes']
        check_budget(point, budget_before)
        result.append({'label':label,'utc':point['utc'],'eastern':local.isoformat(),'elapsed_seconds':elapsed,
          'offset_from_nominal_seconds':elapsed-ordinal*300,
          'complete_users':rows['complete_users'],'delta_users':rows['delta_users'],'remaining_users':TOTAL_USERS-rows['complete_users'],
          'rank_pid':rank['pid'],'prediction_bytes':rows['bytes'],'prediction_buffered_suffix_bytes':rows['buffered_suffix_bytes'],
          'gpu_index':int(gpu['physical_index']),'gpu_utilization_instant_percent':int(gpu['utilization_pct']),
          'gpu_memory_mib':int(gpu['memory_used_mib']),'gpu_device_seconds':point['budget']['formal_gpu']['used_seconds'],
          'captured_sha256':sha256(path.read_bytes()).hexdigest()})
        preceding = rows
    assert result[-1]['elapsed_seconds'] >= 1800
    bundle = read(work/'first-six-capture.json')
    observed = by_label[bundle['observation']]
    raw = (work/'first-six.jsonl').read_bytes()
    assert bundle['start_offset'] == 0 and bundle['end_offset_exclusive'] == len(raw) <= observed['prediction']['complete_bytes']
    assert raw.count(b'\n') == 6 and sha256(raw).hexdigest() == bundle['prediction_sha256']
    assert bundle['captured_utc'] == observed['utc']
    return {'status':'PASS_SEVEN_POINT_NON_AUTHOR','reviewed_utc':datetime.now(timezone.utc).isoformat(),
            'parent_id':parent_id,'child_id':child_id,'seven_points':result,'span_seconds':result[-1]['elapsed_seconds'],
            'first_six_capture':bundle['observation'],'stable_window_users_per_second':
              (result[-1]['complete_users']-result[1]['complete_users'])/(result[-1]['elapsed_seconds']-result[1]['elapsed_seconds']),
            'outcome_scope':'active prefix and ability to continue, not completed formal metrics'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    args=parser.parse_args()
    print(json.dumps(verify_points(args.root),indent=2))


if __name__=='__main__':
    main()
