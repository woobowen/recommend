"""Read one accepted job's state/progress and bounded resource observations."""
import argparse
import datetime as dt
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job")
    parser.add_argument("--support", type=Path, default=Path("/home/addaswsw/recommend-support/sid-extensions-p0"))
    args = parser.parse_args()
    sys.path.insert(0, str(args.support / "scripts"))
    from transport import Transport, atomic
    import gridctl
    gridctl.valid(args.job)
    destination = args.support / "beauty-extensions/observations" / args.job
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    t = Transport("new-server")
    try:
        state = gridctl.query(t, args.job)
        root = t.profile["job_root"] + "/" + args.job
        code = """from pathlib import Path
import json,os,subprocess,datetime
p=Path(ROOT)
def read(name):
 q=p/'output'/name
 return json.loads(q.read_text()) if q.is_file() else None
q=p/'work.log'
tail=''
if q.is_file():
 with q.open('rb') as f:
  f.seek(max(0,q.stat().st_size-4096));tail=f.read(4096).decode('utf-8','replace')
disks={}
for name in ('/home/bwwu','/data/bwwu'):
 s=os.statvfs(name);disks[name]={'free_bytes':s.f_bavail*s.f_frsize,'total_bytes':s.f_blocks*s.f_frsize}
print(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'progress':read('progress.json'),'latest_validation':read('latest_validation.json'),
 'training_result':read('training_result.json'),'last_log_bytes_at_most4096':tail,
 'gpu':subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout,
 'gpu_processes':subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],capture_output=True,text=True).stdout,
 'disks':disks}))
""".replace("ROOT", repr(root))
        observation = json.loads(t.python(code))
        value = {"query_status": "PASS", "job_id": args.job, "state": state, "observation": observation}
        atomic(destination / (stamp + ".json"), value)
        print(json.dumps({"job": args.job, "status": state["status"], "worker_alive": state["worker_alive"],
            "progress": observation["progress"], "gpu": observation["gpu"], "disks": observation["disks"]}, ensure_ascii=False))
    except Exception as error:
        last = args.support / "jobs/new-server" / args.job / "last-query.json"
        value = {"query_status": "FAILED", "job_id": args.job, "error": repr(error),
            "last_known": json.loads(last.read_text()) if last.is_file() else None,
            "restart_submitted": False}
        atomic(destination / (stamp + "-failed.json"), value)
        raise


if __name__ == "__main__":
    main()
