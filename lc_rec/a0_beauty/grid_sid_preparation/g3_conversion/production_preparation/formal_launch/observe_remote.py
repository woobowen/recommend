"""Collect bounded, read-only evidence for the original formal train attempt."""
import base64
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys


def utc(seconds=None):
    if seconds is None:
        return dt.datetime.now(dt.timezone.utc).isoformat()
    return dt.datetime.fromtimestamp(seconds, dt.timezone.utc).isoformat()


def run(*args):
    return subprocess.check_output(args, text=True, timeout=45)


def snapshot(path, limit):
    try:
        with path.open("rb") as stream:
            now = utc()
            stat = os.fstat(stream.fileno())
            begin = max(0, stat.st_size - limit)
            stream.seek(begin)
            data = stream.read(stat.st_size - begin)
    except FileNotFoundError:
        return {"path": str(path), "read_started_utc": utc(), "exists": False}
    return {
        "path": str(path), "read_started_utc": now, "exists": True,
        "source_mtime_utc": utc(stat.st_mtime),
        "source_size_at_open": stat.st_size, "offset_bytes": begin,
        "copied_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
        "content_base64": base64.b64encode(data).decode("ascii"),
    }


def load_snapshot(row):
    if row["exists"]:
        assert row["offset_bytes"] == 0
        return json.loads(base64.b64decode(row["content_base64"]))
    return None


def nul_fields(path):
    return path.read_bytes().rstrip(b"\0").decode(errors="replace").split("\0")


def readlink(path):
    try:
        return os.readlink(path)
    except (FileNotFoundError, PermissionError):
        return None


def proc(pid):
    path = Path(f"/proc/{pid}")
    try:
        if path.stat().st_uid != os.getuid():
            return None
        fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        argv = nul_fields(path / "cmdline")
        begin = uptime_epoch + int(fields[19]) / ticks_per_second
        keys = {"CUDA_VISIBLE_DEVICES", "GRID_JOB_DIR", "GRID_JOB_ID", "LOCAL_RANK",
                "RANK", "WORLD_SIZE", "G3_FORMAL_AUTHORIZATION_SHA256",
                "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"}
        env = {key: value for key, _, value in
               (item.partition("=") for item in nul_fields(path / "environ")) if key in keys}
        return {
            "pid": pid, "ppid": int(fields[1]), "pgid": int(fields[2]),
            "session_id": int(fields[3]), "tty_nr": int(fields[4]),
            "start_ticks": fields[19], "boot_id": boot_id, "uid": os.getuid(),
            "started_at_utc_from_proc": utc(begin), "argv": argv, "env": env,
            "cwd": readlink(path / "cwd"),
            "fds": {str(fd): readlink(path / f"fd/{fd}") for fd in range(3)},
        }
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


started = utc()
root = Path("/data/bwwu/lc-rec/g3-formal-20261118/21d495b0be5b")
plan = json.loads((root / "plan.json").read_bytes())
job_id = "letter-a0-beauty-formal-g3-beauty-seed42-train-a0"
job = Path(plan["job_root"]) / job_id
training = root / "training"
files = {
    "plan.identity.json": (root / "plan.identity.json", 16_384),
    "state.json": (root / "state.json", 524_288),
    "controller.log": (root / "controller.log", 16_384),
    "controller-request.json": (root / "requests/train-a0.json", 524_288),
    "runner-request.json": (job / "request.json", 524_288),
    "runner-accepted.json": (job / "accepted.json", 32_768),
    "runner-status.json": (job / "status.json", 524_288),
    "work.log": (job / "work.log", 32_768),
    "compute_argv.json": (root / "attempts/train/a0/compute_argv.json", 32_768),
    "rank0.json": (training / "rank0.json", 1_048_576),
    "events.rank0.jsonl": (training / "events.rank0.jsonl", 131_072),
    "failure.rank0.json": (training / "failure.rank0.json", 32_768),
    "launch.exit": (root / "launch-20261118/launch.exit", 4096),
}
if len(sys.argv) > 2 and sys.argv[2] == "full":
    files.update({
        "server-launch.stdout": (root / "launch-20261118/launch.stdout", 32_768),
        "server-launch.stderr": (root / "launch-20261118/launch.stderr", 32_768),
        "server-authorization.json": (root / "launch-20261118/authorization.json", 32_768),
        "experiment_registry.json": (
            Path("/data/bwwu/lc-rec/a0-beauty-20261115/formal-experiment-registry"
                 "/3aa649784aa154e1274b7ddc9ede10e6ccf3f47cee36b787e2361ce69b646726.json"),
            16_384),
    })

runner = None
runner_query = None
if (job / "request.json").is_file():
    runner_args = ["/usr/bin/python3", "-I", "-S", "-B", plan["runner"], "query", str(job)]
    query = subprocess.run(runner_args, text=True, capture_output=True, timeout=45)
    runner_query = {
        "argv": runner_args, "read_utc": utc(), "exit_code": query.returncode,
        "stdout": query.stdout, "stderr": query.stderr,
    }
    if query.returncode == 0:
        runner = json.loads(query.stdout)

boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
uptime_epoch = next(int(line.split()[1]) for line in Path("/proc/stat").read_text().splitlines()
                    if line.startswith("btime "))
ticks_per_second = os.sysconf("SC_CLK_TCK")
controller = int(sys.argv[1])
pids = {controller}
if runner:
    pids.update(runner[key]["pid"] for key in ("supervisor", "worker") if runner.get(key))
    pids.update(row["pid"] for row in runner.get("workgroup_observation", {}).get("owned_live", [])
                if isinstance(row, dict) and "pid" in row)
# Include the torchrun launcher and rank processes once their exact experiment
# markers occur in this UID's process list. The process data itself comes from /proc.
for line in run("ps", "-u", str(os.getuid()), "-o", "pid=,args=").splitlines():
    if str(root) in line and ("torch.distributed.run" in line or "formal_train.py" in line):
        pids.add(int(line.split(None, 1)[0]))
processes = {pid: proc(pid) for pid in sorted(pids)}
samples = {name: snapshot(path, limit) for name, (path, limit) in files.items()}
state = load_snapshot(samples["state.json"])
budget = {k: 0.0 for k in ("train_normal", "s3", "s4", "cpu", "total")}
stage_charge = {k: 0.0 for k in plan["stage_seconds"]}
if state:
    for attempt in state["attempts"]:
        stage = attempt["stage"]
        seconds = attempt.get("charged_seconds", 0)
        stage_charge[stage] += seconds
        group = "train_normal" if stage in ("train", "normal") else (
            stage if stage in ("s3", "s4") else "cpu")
        budget[group] += seconds
        budget["total"] += seconds
disk = os.statvfs(root)
memory = next(int(line.split()[1]) * 1024
              for line in Path("/proc/meminfo").read_text().splitlines()
              if line.startswith("MemAvailable:"))
outputs = []
if training.is_dir():
    for item in sorted(training.iterdir()):
        if item.is_symlink():
            continue
        row = {"name": item.name, "size_bytes": item.stat().st_size, "directory": item.is_dir()}
        if item.is_dir() and item.name.startswith("checkpoint-"):
            row["has_complete_json"] = (item / "complete.json").is_file()
        outputs.append(row)
gpu = run("nvidia-smi", "--id=1",
          "--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu",
          "--format=csv,noheader").splitlines()
gpu_compute = run("nvidia-smi", "--id=1", "--query-compute-apps=pid,used_memory",
                  "--format=csv,noheader").splitlines()
print(json.dumps({
    "collection_started_utc": started, "collection_finished_utc": utc(),
    "host": socket.gethostname(), "expected_job_id": job_id,
    "expected_controller_pid": controller, "processes": processes,
    "runner_query": runner_query, "files": samples,
    "host_memory_available_bytes": memory,
    "filesystem_available_bytes": disk.f_bavail * disk.f_frsize,
    "filesystem_total_bytes": disk.f_blocks * disk.f_frsize,
    "gpu1": gpu, "gpu1_compute_processes": gpu_compute, "training_outputs": outputs,
    "budget_at_state_mtime_utc": samples["state.json"].get("source_mtime_utc"),
    "charged_seconds": budget, "stage_charge_seconds": stage_charge,
    "remaining_seconds": {key: plan["budget"][key] - value for key, value in budget.items()},
    "cleanup_reserve_seconds": plan["budget"]["cleanup_reserve"],
}, indent=2))
