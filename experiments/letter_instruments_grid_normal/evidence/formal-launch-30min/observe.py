"""Seven read-only samples. Deadlines follow the server worker's kernel start."""
import argparse
import datetime as dt
import json
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

parser = argparse.ArgumentParser()
parser.add_argument("--verify-now", action="store_true")
args = parser.parse_args()
here = Path(__file__).resolve().parent
request = json.loads((here / "request.json").read_text())
job = request["job_id"]
ssh = ["/mnt/c/Windows/System32/OpenSSH/ssh.exe", "-F", "C:/Users/lenovo/.ssh/config",
       "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=12",
       "-o", "ControlMaster=no", "-o", "ControlPath=none",
       "-o", "ClearAllForwardings=yes", "-T", "new-server"]
new_york = ZoneInfo("America/New_York")

def timepair(ts):
    return {"utc": dt.datetime.fromtimestamp(ts, dt.timezone.utc).isoformat(),
            "new_york": dt.datetime.fromtimestamp(ts, new_york).isoformat()}

def capture(label, target, prefix=False):
    script = (here / "capture.py").read_bytes()
    argv = ["/usr/bin/python3", "-I", "-S", "-B", "-", job]
    if prefix:
        argv.append("--include-prefixes")
    start = time.time()
    for attempt in range(2):
        try:
            proc = subprocess.run([*ssh, shlex.join(argv)], input=script, capture_output=True, timeout=32)
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr.decode(errors="replace").strip())
            end = time.time()
            data = json.loads(proc.stdout)
            if proc.stderr:
                data["ssh_stderr"] = proc.stderr.decode(errors="replace")
            break
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
            detail = {"started": timepair(start), "recorded": timepair(time.time()), "error": str(error)}
            (here / f"{label}-query-error-{attempt + 1}.json").write_text(json.dumps(detail, indent=2) + "\n")
            if attempt == 1:
                return None
    sample_epoch = data["server_epoch"]
    point = dict(label=label, local_query_start=timepair(start), local_query_end=timepair(end),
                 local_staging_free_bytes=shutil.disk_usage(here).free,
                 windows_downloads_free_bytes=shutil.disk_usage("/mnt/c/Users/lenovo/Downloads").free,
                 measured_time=timepair(sample_epoch))
    if target is None and data.get("process", {}).get("worker", {}).get("start_kernel_utc"):
        target = dt.datetime.fromisoformat(data["process"]["worker"]["start_kernel_utc"]).timestamp()
    if target is not None:
        point.update(target_time=timepair(target), deviation_seconds=sample_epoch - target)
    data["observation"] = point
    with (here / f"{label}.json").open("x") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(label, point, data["status"].get("status"), flush=True)
    return data

if args.verify_now:
    sys.exit(0 if capture("delivery-status", None) is not None else 1)

first = capture("T00", None)
if first is None or not (first.get("process", {}).get("worker") or {}).get("start_kernel_utc"):
    sys.exit("T0 did not establish a live worker start; inspect same job ID")
base = dt.datetime.fromisoformat(first["process"]["worker"]["start_kernel_utc"]).timestamp()
for minute in (5, 10, 15, 20, 25, 30):
    target = base + minute * 60
    while (remaining := target - time.time()) > 0:
        time.sleep(min(remaining, 10))
    snapshot = capture(f"T{minute:02d}", target, prefix=minute == 30)
    if snapshot is not None and snapshot["status"].get("status") not in {"RUNNING", "ACCEPTED"}:
        sys.exit("Job left running state during the observation window; inspect same job ID")
