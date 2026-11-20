"""Summarize seven saved live observations without contacting the server."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

from lc_rec.shuffle_v2 import control, core

OUT = Path(__file__).resolve().parent
JOB = control.PREFIX + 'gpu-formal-shuffle2-r0'
PARENT = control.PREFIX + 'group-series-r0'
DEV = 'GPU-59cac3bd-3645-668b-79c8-839e0d878893'


def load(file):
    return json.loads(file.read_text())


def elapsed(earlier, later):
    return (datetime.fromisoformat(later)-datetime.fromisoformat(earlier)).total_seconds()


def duration(seconds):
    mins = round(seconds / 60)
    hours, mins = divmod(mins, 60)
    return f'{hours} 小时 {mins:02d} 分钟' if hours else f'{mins} 分钟'


def main():
    offsets = range(0,31,5)
    raw = [load(OUT/f'T{i:02d}.json') for i in offsets]
    t0 = raw[0]['observed_utc']
    table, windows = [], []
    for offset, d in zip(offsets, raw, strict=True):
        assert abs(elapsed(t0,d['observed_utc'])-60*offset)<15, d['observed_eastern']
        j = d['stages'][JOB]
        assert d['stages'][PARENT]['worker_alive'] and j['worker_alive']
        assert {key for key in d['stages'] if 'gpu-formal' in key} == {JOB}
        assert {key for key in d['stages'] if 'cpu-eval-shuffle' in key or 'cpu-summary' in key} == set()
        assert j['status']=='RUNNING' and j['report']['status']=='RUNNING' and j['report']['error'] is None
        assert j['report']['completed_users'] > 0 and j['report']['assigned_users']==24772
        assert j['report']['world_size']==1 and j['report']['rank']==j['report']['local_rank']==0
        assert j['request_env_cuda']==DEV and not j['unexpected_rank1'] and not j['error_patterns']
        owned = [p for p in d['compute_processes'] if p['task_job']]
        assert len(owned)==1 and owned[0]['uuid']==DEV and owned[0]['task_job']==JOB
        assert owned[0]['runtime_env']['WORLD_SIZE']=='1' and owned[0]['runtime_env']['RANK']=='0'
        gpu = next(g for g in d['gpus'] if g[1]==DEV)
        progress = j['report']['completed_users']
        entry = {'label':d['observation_label'],'observed_eastern':d['observed_eastern'], 'observed_utc':d['observed_utc'],
                 'formal_job':JOB,'gpu_index':int(gpu[0]),'gpu_util_percent':int(gpu[5]),'gpu_total_mib':int(gpu[4]),
                 'gpu_process_memory_mib':owned[0]['memory_mib'],'gpu_pid':owned[0]['pid'],
                 'completed_users':progress,'prediction_bytes':j['prediction']['bytes'],'log_bytes':j['log']['bytes'],
                 'last_log':j['log_tail'][-1],'error_count':len(j['error_patterns']),'free_bytes':d['free_bytes'],
                 'input_snapshot':hashlib.sha256((OUT/f'T{offset:02d}.json').read_bytes()).hexdigest()}
        if table:
            last = table[-1]
            entry['delta_users'] = progress-last['completed_users']
            assert entry['delta_users']>0 and entry['prediction_bytes']>last['prediction_bytes'] and entry['log_bytes']>last['log_bytes']
            windows.append(entry['delta_users']/elapsed(last['observed_utc'],d['observed_utc']))
        else:
            entry['delta_users'] = None
        table.append(entry)
    final = table[-1]
    remaining = 24772-final['completed_users']
    recent_rate = (final['completed_users']-table[-3]['completed_users'])/elapsed(table[-3]['observed_utc'],final['observed_utc'])
    recent_low, recent_high = min(windows[-4:]), max(windows[-4:])
    s2_remaining = {'point':remaining/recent_rate,'low':remaining/recent_high,'high':remaining/recent_low}
    estimates = {}
    s2_preflight = load(control.SUPPORT/'verified-preflight/shuffle2/rank0.json')['elapsed_generation_seconds']
    for mode in tuple(core.MODES)[1:]:
        p = load(control.SUPPORT/'verified-preflight'/mode/'rank0.json')
        extrapolated = p['elapsed_generation_seconds']/32*24772+p['model_load_and_strict_verification_seconds']
        adjusted = p['elapsed_generation_seconds']/s2_preflight*24772/recent_rate+p['model_load_and_strict_verification_seconds']
        estimates[mode] = {'S2_throughput_scaled_by_preflight_s':adjusted,'direct_preflight_extrapolation_s':extrapolated,
                           'low':min(adjusted,extrapolated),'high':max(adjusted,extrapolated)}
    series = {key:s2_remaining[key]+sum(e[key] for e in estimates.values()) for key in ('low','high')}
    current = raw[-1]['stages'][JOB]
    formal_used = elapsed(current['start'],final['observed_utc'])
    prep = load(control.SUPPORT/'verified-prep/collection.json')['states']
    for suffix in ('cpu-single-shard-r1','cpu-evaluation-fixture-r0'):
        identity=control.PREFIX+suffix
        statement=Path('/home/addaswsw/recommend-support/sid-extensions-p0/jobs/new-server')/identity/'last-query.json'
        prep[identity] = load(statement)
    consumed = {name:elapsed(state['started_at'],state['ended_at']) for name,state in prep.items() if 'cpu-' in name}
    preflight = raw[-1]['preflight_job']
    preflight_used = elapsed(preflight['started_at'],preflight['ended_at'])
    record = {'status':'RUNNING_AFTER_30_MIN_OBSERVATION','observations':table,'model_load_and_strict_check_seconds':current['report']['model_load_and_strict_verification_seconds'],
              'generation_elapsed_to_last_report_seconds':current['report']['elapsed_generation_seconds'],
              'remaining_users':remaining,'recent_10_min_users_per_s':recent_rate,'observed_5_min_window_users_per_s':windows,
              's2_remaining_seconds':s2_remaining,'later_gpu_stage_seconds':estimates,'gpu_series_remaining_seconds':series,
              'budgets_seconds':{'formal_single_attempt_used':formal_used,'formal_single_attempt_remaining_from_hard_cap':28800-formal_used,
                                 'formal_runner_remaining_before_watchdog':current['timeout_seconds']-formal_used,
                                 'formal_total_used':formal_used,'formal_total_remaining':129600-formal_used,
                                 'gpu_preflight_used':preflight_used,'gpu_preflight_remaining':3600-preflight_used,
                                 'cpu_used':sum(consumed.values()),'cpu_remaining':14400-sum(consumed.values()),'cpu_attempts':consumed}}
    control.transport.atomic(OUT/'observation-summary.json',record)
    print(json.dumps(record,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
