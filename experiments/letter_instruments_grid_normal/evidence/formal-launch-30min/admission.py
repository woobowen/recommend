"""Read-only, task-local pre-submission admission. Request is passed over stdin."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import shutil
import subprocess
import sys

request = json.load(sys.stdin)
cwd = Path(request["cwd"])
job_id = request["job_id"]
job = Path(request["argv"][-1]).parents[1]
root = job.parents[1]
plan = cwd.parent / f"formal-config-{job_id}-{request['capture_version'][:12]}"
runner = root / "scripts/runner-398768a732bbfebd.py"

def sha(path):
    with path.open("rb") as file:
        digest = hashlib.sha256()
        for chunk in iter(lambda: file.read(1048576), b""):
            digest.update(chunk)
        return digest.hexdigest()

assert socket.gethostname() == request["host"] and os.getuid() == request["uid"]
assert cwd.with_suffix(".commit").read_text().strip() == request["capture_version"]
assert not cwd.stat().st_mode & 0o222
assert os.access(request["argv"][0], os.X_OK)
assert not job.exists()
assert not (root / "requests" / f"{job_id}.json").exists()
assert sha(Path(str(runner) + ".deployment.json")) == request["deployment_sha256"]
with (cwd.parent / "gpu.lock").open("rb") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

matches = []
for entry in Path("/proc").iterdir():
    if not entry.name.isdecimal():
        continue
    try:
        if entry.stat().st_uid != request["uid"]:
            continue
        args = (entry / "cmdline").read_bytes().decode(errors="replace").split("\0")
        if any(arg.endswith("/run_normal.py") for arg in args) and "formal" in args:
            matches.append(int(entry.name))
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        continue
assert not matches, matches

inputs = []
for info in request["inputs"]:
    path = Path(info["path"])
    digest = sha(path)
    assert digest == info["sha256"], str(path)
    inputs.append(dict(path=str(path), sha256=digest, size=path.stat().st_size))
free = {str(p): shutil.disk_usage(p).free for p in (job.parent, cwd.parent)}
assert min(free.values()) > 2 * 1024**3
device = request["env"]["CUDA_VISIBLE_DEVICES"]
assert device.startswith("GPU-") and "," not in device
gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
                               "--format=csv,noheader,nounits"], text=True).strip().splitlines()
selected = [row.split(", ") for row in gpu if f", {device}, " in row]
assert len(selected) == 1 and "4090" in selected[0][2] and int(selected[0][3]) <= 256, selected
apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                                "--format=csv,noheader,nounits"], text=True).strip().splitlines()
assert not any(device in row for row in apps)
print(json.dumps({
    "server_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "host": socket.gethostname(), "uid": os.getuid(), "job_id": job_id,
    "job_exists": False, "request_exists": False, "matching_formal_worker_pids": matches,
    "experiment_gpu_lock": "available at admission", "capture_version": request["capture_version"],
    "inputs": inputs, "runner_sha256": sha(runner), "deployment_sha256": request["deployment_sha256"],
    "runner_path": str(runner), "config_plan_path": str(plan),
    "plan_sha256": {name: sha(plan / name) for name in ("effective_config.yaml", "overrides.json", "dry_run.json")},
    "python": subprocess.check_output([request["argv"][0], "-V"], text=True).strip(),
    "free_bytes": free, "gpus": gpu, "selected_gpu": selected[0], "compute_apps": apps,
}, indent=2))
