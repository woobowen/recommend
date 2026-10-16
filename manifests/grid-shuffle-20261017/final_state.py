"""Final real original-ID process/status/resource checks; never submit/restart."""
import datetime as dt
import json
from pathlib import Path
import sys

SUPPORT=Path('/home/addaswsw/recommend-support/grid-shuffle-20261017')
JOBS=Path('/home/addaswsw/recommend-support/sid-extensions-p0/jobs/new-server')
sys.path.insert(0,'/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
import gridctl
from transport import Transport


def main():
    t=Transport('new-server');values={}
    for path in sorted(JOBS.glob('gs-20261017-*')):
        values[path.name]=gridctl.query(t,path.name)
    active=[name for name,row in values.items() if row['status'] not in ('SUCCEEDED','FAILED','TIMED_OUT','CANCELLED')
        or row['worker_alive'] or row['workgroup_live'] or row.get('worker_may_be_live',False) or not row['cleanup_verified']]
    resources=json.loads(t.python("import subprocess,json,shutil;print(json.dumps({'GPU':subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv'],text=True),'compute_apps':subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv'],text=True),'disk_free':{p:shutil.disk_usage(p).free for p in ['/home/bwwu','/data/bwwu']}}))"))
    result={'observed_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'jobs':values,'active_job_ids':active,
        'all_task_jobs_terminal_and_clean':not active,'resources':resources,'original_IDs_only':True,'new_submissions':0}
    (SUPPORT/'final-state-query.json').write_text(json.dumps(result,indent=2)+'\n')
    assert not active,active
    print(json.dumps({'all_task_jobs_terminal_and_clean':True,'jobs':len(values),'resources':resources}))


if __name__=='__main__':main()
