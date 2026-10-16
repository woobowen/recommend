"""Lightweight original-ID checks through existing gridctl, optional terminal fetch."""
import argparse
import datetime as dt
import json
from pathlib import Path
import sys

SUPPORT=Path('/home/addaswsw/recommend-support/grid-shuffle-20261017')
sys.path.insert(0,'/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
import gridctl
from transport import Transport


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('jobs',nargs='+')
    p.add_argument('--fetch-terminal',action='store_true');args=p.parse_args()
    assert all(j.startswith('gs-20261017-') for j in args.jobs)
    t=Transport('new-server');records=[]
    for job in args.jobs:
        value=gridctl.query(t,job)
        record={k:value.get(k) for k in ('job_id','status','exit_code','started_at','ended_at','worker_alive','workgroup_live','cleanup_verified')}
        if args.fetch_terminal and value['status'] in ('SUCCEEDED','FAILED','TIMED_OUT','CANCELLED'):
            record['retrieval']=gridctl.fetch(t,job)
        records.append(record)
    code="""import json,subprocess,shutil
from pathlib import Path
root=Path('/home/bwwu/grid-audits/sid-extensions-p0/jobs');jobs=JOBS
r={}
for job in jobs:
 p=root/job;progress=p/'output/test/progress.json'
 r[job]={'progress':json.loads(progress.read_text()) if progress.exists() else None,
  'log_tail':(p/'work.log').read_text()[-1200:] if (p/'work.log').exists() else '',
  'output_bytes':sum(f.stat().st_size for f in (p/'output').rglob('*') if f.is_file())}
print(json.dumps({'jobs':r,'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv'],text=True),
 'compute_apps':subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv'],text=True),
 'disk_free_bytes':{p:shutil.disk_usage(p).free for p in ['/home/bwwu','/data/bwwu']}}))
""".replace('JOBS',repr(args.jobs))
    observation=json.loads(t.python(code));stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output=SUPPORT/'observations';output.mkdir(exist_ok=True)
    (output/(stamp+'.json')).write_text(json.dumps({'observed_at_utc':stamp,'records':records,'remote':observation},indent=2)+'\n')
    print(json.dumps({'records':records,'progress':{j:v['progress'] for j,v in observation['jobs'].items()},
        'gpu':observation['gpu'],'disk_free_bytes':observation['disk_free_bytes']},ensure_ascii=False))


if __name__=='__main__':main()
