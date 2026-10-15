"""Read-only online monitor; computation remains owned by the server runner."""
import argparse
import datetime as dt
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, '/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
from transport import Transport, atomic
import gridctl

SUPPORT = Path('/home/addaswsw/recommend-support/grid-formal-20261015')


def observe(jobs):
    t = Transport('new-server')
    states = {}
    for job in jobs:
        query = gridctl.query(t, job)
        states[job] = {key: query.get(key) for key in ('status', 'exit_code', 'started_at',
            'ended_at', 'worker_alive', 'workgroup_live', 'cleanup_verified')}
    code = '''import pathlib,json,subprocess,shutil
root=pathlib.Path('/home/bwwu/grid-audits/sid-extensions-p0/jobs')
r={'jobs':{},'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu','--format=csv,noheader'],text=True),'disk_free_bytes':shutil.disk_usage(root).free}
for job in JOBS:
 p=root/job
 d={}
 for name in ('progress.json','training_result.json','RESULT.json'):
  f=p/'output'/name
  if f.is_file() and f.stat().st_size<128*1024:
   try:d[name]=json.loads(f.read_text())
   except json.JSONDecodeError:d[name]={'being_written':True}
 f=p/'work.log'
 if f.is_file():
  with f.open('rb') as stream:
   stream.seek(max(0,f.stat().st_size-1800));d['log_tail']=stream.read().decode(errors='replace')
 r['jobs'][job]=d
print(json.dumps(r))
'''.replace('JOBS', repr(jobs))
    detail = json.loads(t.python(code))
    now = dt.datetime.now(dt.timezone.utc)
    record = {'observed_at_utc': now.isoformat(), 'states': states, 'detail': detail,
        'read_only': True, 'no_restart_or_submission': True}
    destination = SUPPORT/'observations'
    destination.mkdir(exist_ok=True)
    atomic(destination/(now.strftime('%Y%m%dT%H%M%SZ')+'.json'), record)
    print(json.dumps(record, ensure_ascii=False), flush=True)
    return all(s['status'] in ('SUCCEEDED', 'FAILED', 'TIMED_OUT', 'CANCELLED') for s in states.values())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('jobs', nargs='+')
    p.add_argument('--interval', type=int, default=600)
    p.add_argument('--watch-hours', type=float, default=0)
    a = p.parse_args()
    assert a.interval >= 600 and 0 <= a.watch_hours <= 13
    assert all(job.startswith('gf-20261015-') and '/' not in job for job in a.jobs)
    deadline = time.monotonic()+a.watch_hours*3600
    while True:
        terminal = observe(a.jobs)
        if terminal or time.monotonic() >= deadline:
            break
        time.sleep(min(a.interval, max(0, deadline-time.monotonic())))


if __name__ == '__main__':
    main()
