"""Record bounded observations of the Beauty series and its active child."""
import argparse
import json
from pathlib import Path

from lc_rec.shuffle_v2 import beauty_control, control


def observe(connection):
    return json.loads(control.remote_python(connection, f"""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
jobs = Path({(control.BASE+'/jobs')!r})
root = Path({beauty_control.ROOT!r})
report = dict(observed_utc=datetime.now(timezone.utc).isoformat(), jobs={{}})
for path in sorted(jobs.glob({(beauty_control.PREFIX+'*')!r})):
    status = path / 'status.json'
    if not status.is_file():
        continue
    row = json.loads(status.read_text())
    values = {{key: row.get(key) for key in ('status','started_at','ended_at','exit_code','cleanup','worker')}}
    if row.get('ended_at') is None:
        folder = root / path.name.removeprefix({beauty_control.PREFIX!r})
        worker = row.get('worker') or {{}}
        pid = worker.get('pid')
        if pid:
            proc = Path('/proc') / str(pid) / 'stat'
            try:
                observed = proc.read_text().rsplit(')', 1)[1].split()[19]
            except FileNotFoundError:
                observed = None
            values['worker_pid_identity_alive'] = observed == str(worker.get('start_ticks'))
        files = [path / 'work.log', *sorted(folder.glob('shuffle*.log'))]
        values['log_tails'] = {{}}
        for log in files:
            if log.is_file():
                with log.open('rb') as stream:
                    stream.seek(max(0, log.stat().st_size - 4096))
                    values['log_tails'][str(log)] = stream.read().decode(errors='replace').splitlines()[-12:]
        rank_report = folder / 'test/rank0.json'
        if rank_report.is_file():
            rank = json.loads(rank_report.read_text())
            values['progress'] = {{key: rank.get(key) for key in ('status','completed_users','assigned_users','elapsed_generation_seconds','pid','error')}}
            prediction = folder / 'test/predictions.rank0.jsonl'
            if prediction.is_file():
                values['prediction_bytes'] = prediction.stat().st_size
    report['jobs'][path.name] = values
for label, argv in {{
    'gpus': ['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],
    'gpu_processes': ['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],
    'disk': ['df','-hP','/data/bwwu'],
}}.items():
    report[label] = subprocess.check_output(argv, text=True)
print(json.dumps(report))
"""))


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--log', type=Path, default=beauty_control.SUPPORT / 'observations.jsonl')
    args = p.parse_args()
    observation = observe(control.connect())
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open('a') as stream:
        stream.write(json.dumps(observation, separators=(',', ':')) + '\n')
    print(json.dumps(observation, indent=2))


if __name__ == '__main__':
    main()
