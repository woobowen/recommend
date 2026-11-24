"""Read-only server snapshot for the selected formal job. Source is sent over SSH."""
import argparse
import base64
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("job_id")
parser.add_argument("--include-prefixes", action="store_true")
args = parser.parse_args()
start = time.time()
root = Path("/home/bwwu/grid-audits/sid-extensions-p0")
job = root / "jobs" / args.job_id
out = job / "output/run"
runner = root / "scripts/runner-398768a732bbfebd.py"

def run(argv):
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=12)
    return {"argv": argv, "exit": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}

def stamps(epoch):
    return dt.datetime.fromtimestamp(epoch, dt.timezone.utc).isoformat()

def record(p, tail=None, prefix=None):
    """Read to the byte boundary established through the open descriptor."""
    try:
        with p.open("rb") as file:
            st = os.fstat(file.fileno())
            result = {"path": str(p), "bytes": st.st_size, "mtime_utc": stamps(st.st_mtime),
                      "mtime_ns": st.st_mtime_ns, "inode": st.st_ino}
            if tail is not None:
                seek = max(0, st.st_size - tail)
                file.seek(seek)
                result["tail"] = {"offset": seek, "text": file.read(st.st_size - seek).decode(errors="replace")}
            if prefix is not None:
                file.seek(0)
                content = file.read(min(prefix, st.st_size))
                result["prefix"] = {"length": len(content), "complete": len(content) == st.st_size,
                                    "base64": base64.b64encode(content).decode("ascii")}
            return result
    except FileNotFoundError:
        return None

def meta(p):
    return json.loads(p.read_text()) if p.is_file() else None

def proc_info(saved):
    if not saved:
        return None
    pid = saved["pid"]
    p = Path(f"/proc/{pid}")
    try:
        fields = (p / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[19] != saved["start_ticks"]:
            return {"pid": pid, "identity_mismatch": True}
        uptime = time.clock_gettime(time.CLOCK_BOOTTIME)
        process_observed = time.time()
        elapsed = uptime - int(fields[19]) / os.sysconf("SC_CLK_TCK")
        links = {}
        for name in ("fd/0", "fd/1", "fd/2", "cwd", "exe"):
            try:
                links[name] = os.readlink(p / name)
            except FileNotFoundError:
                links[name] = None
        allow = {"CUDA_VISIBLE_DEVICES", "GRID_JOB_ID", "GRID_JOB_DIR", "WORLD_SIZE", "RANK", "LOCAL_RANK"}
        filtered = {}
        for entry in (p / "environ").read_bytes().decode(errors="replace").split("\0"):
            key, _, value = entry.partition("=")
            if key in allow:
                filtered[key] = value
        return dict(pid=pid, state=fields[0], ppid=int(fields[1]), pgid=int(fields[2]), sid=int(fields[3]),
                    tty_number=int(fields[4]), start_ticks=fields[19], start_kernel_utc=stamps(
                        process_observed - elapsed),
                    elapsed_uptime_seconds=elapsed, process_observed_utc=stamps(process_observed),
                    fd_and_paths=links, initial_exec_environment=filtered)
    except (ProcessLookupError, FileNotFoundError):
        return {"pid": pid, "gone": True}

# Persisted status and real identities are checked by the deployed runner.
query = run(["/usr/bin/python3", "-I", "-S", "-B", str(runner), "query", str(job)])
status = json.loads(query["stdout"]) if query["exit"] == 0 else {}
if status:
    query["stdout"] = ""
process = {key: proc_info(status.get(key)) for key in ("supervisor", "worker")}
gpu = run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
           "--format=csv,noheader,nounits"])
apps = run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"])
files = {}
paths = [job / n for n in ("work.log", "supervisor.log", "status.json")]
paths += [out / n for n in ("events.jsonl", "startup.json", "model.json", "effective_config.yaml",
                             "overrides.json", "candidates.jsonl", "stage_result.json", "offline_evaluation.json")]
paths += sorted(out.glob("**/metrics.csv"))
for path in paths:
    tail = 6000 if path.name in ("work.log", "supervisor.log", "events.jsonl", "metrics.csv") else None
    prefix = None
    if args.include_prefixes and path.name in (
        "work.log", "supervisor.log", "events.jsonl", "metrics.csv", "effective_config.yaml",
        "startup.json", "model.json", "overrides.json",
    ):
        prefix = 8 * 1024 * 1024
    files[str(path.relative_to(job))] = record(path, tail, prefix)
checkpoints = []
for path in sorted(out.glob("checkpoints/*.ckpt")):
    info = record(path)
    if info:
        checkpoints.append(info)
queued = sorted(str(p.relative_to(root)) for p in (root / "requests").glob(
    "grid-letter-instruments-normal-20261124-formal-*.json"))
accepted = sorted(str(p.relative_to(root)) for p in (root / "jobs").glob(
    "grid-letter-instruments-normal-20261124-formal-*/accepted.json"))
print(json.dumps(dict(
    job_id=args.job_id, host=socket.gethostname(), server_epoch=start, server_utc=stamps(start),
    server_end_utc=stamps(time.time()), boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    status=status, query=query, process=process, gpu=gpu, gpu_compute_apps=apps, files=files,
    checkpoints=checkpoints, accepted=meta(job / "accepted.json"), attempt=meta(job / "attempt.json"),
    queued_formal_requests=queued, accepted_formal_requests=accepted,
    free_bytes=shutil.disk_usage(job.parent).free,
)))
