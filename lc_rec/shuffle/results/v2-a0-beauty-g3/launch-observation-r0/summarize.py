"""Render exact observation counts and elapsed-time estimates."""
import argparse
from datetime import datetime
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_bytes())


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--evidence',type=Path,required=True)
    a=p.parse_args()
    root=a.evidence
    rows=[read(root/'observations'/f'T{i:02}.json') for i in range(0,31,5)]
    first,stable,last=rows[0],rows[1],rows[-1]
    delivery=read(root/'delivery-check.json')
    stable_seconds=(last['unix_ns']-stable['unix_ns'])/1e9
    completed=last['prediction']['complete_users']
    rate=(completed-stable['prediction']['complete_users'])/stable_seconds
    s2_remaining=(22363-completed)/rate/3600
    ratio=22363/rate/3600/4.71
    later={mode:hours*ratio for mode,hours in [('S3a',4.28),('S3b',4.72),('S4',4.32)]}
    result={
        'status':'THIRD_GROUP_FORMAL_RUNNING_30MIN_OBSERVATION_PASS',
        'active_monitoring_stopped':read(root/'window-complete.json')['seven_observations'],
        'full_group_results':'PENDING',
        'parent_id':last['parent']['job_id'],'S2_id':last['child']['job_id'],'GPU_rank_pid':last['gpu_rank_processes'][0]['pid'],
        'observed_span_seconds':(last['unix_ns']-first['unix_ns'])/1e9,
        'T0_ET':first['america_new_york'],'T30_ET':last['america_new_york'],
        'T30_complete_users':completed,'T30_latest_rank_log_count':last['rank']['completed_users'],
        'delivery_check_ET':delivery['america_new_york'],'delivery_check_complete_users':delivery['prediction']['complete_users'],
        'prediction_bytes_at_T30':last['prediction']['bytes'],'minimum_disk_free_bytes':min(r['output_free_bytes'] for r in rows),
        'strict_load_seconds':stable['rank']['model_load_and_strict_verification_seconds'],
        'stable_observed_interval_seconds':stable_seconds,'stable_users_per_second':rate,
        'S2_remaining_gpu_hours_at_T30':s2_remaining,'later_mode_gpu_hours_estimate':later,
        'all_remaining_gpu_hours_at_T30':s2_remaining+sum(later.values()),
        'estimate_basis':'T05 to T30 measured S2 records. Later ratios from reviewed G3 estimates, scaled to observed S2 full generation rate.',
        'GPU_uuid':last['gpu']['uuid'],
        'T30_budget':{k:{n:v[n] for n in ('used_seconds','limit_seconds','remaining_seconds')} for k,v in last['budget'].items()},
        'delivery_check_budget':{k:{n:v[n] for n in ('used_seconds','limit_seconds','remaining_seconds')} for k,v in delivery['budget'].items()},
    }
    assert result['observed_span_seconds']>=1800 and rate>0
    assert delivery['parent']['status']=='RUNNING' and delivery['child']['status']=='RUNNING'
    assert delivery['prediction']['complete_users']<22363
    (root/'stage-summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('| 点 | 纽约时间 | 阶段 | 完整用户 | 新增 | 剩余 | 预测文件字节 | 预测新增字节 | S2 日志新增字节 | MiB / 瞬时利用率 |')
    print('| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |')
    for r in rows:
        label=r['label']
        moment=datetime.fromisoformat(r['america_new_york']).strftime('%H:%M:%S')
        p=r['prediction']
        phase='生成' if p['complete_users'] else '启动／加载'
        log=r['logs']['gpu_worker']
        print(f'| [{label}](evidence/observations/{label}.json) | {moment} | {phase} | {p["complete_users"]:,} | {p["delta_users"]:,} | {22363-p["complete_users"]:,} | {p["bytes"]:,} | {p["delta_bytes"]:,} | {log["delta_bytes"] if log else 0:,} | {int(r["gpu"]["memory_used_mib"]):,} / {r["gpu"]["utilization_pct"]}% |')


if __name__=='__main__':
    main()
