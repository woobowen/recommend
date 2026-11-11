"""GPU and host guard, charged over complete child lifetimes across attempts."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", required=True, choices=["q06", "q17"])
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.command[0] == "--" and os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert os.environ["GRID_JOB_ID"] == args.job_id and args.job_id.startswith("lc-qwen6-20261111-")
    import psutil
    root = Path("/home/bwwu/lc-rec/qwen-six-task-20261111")
    data = Path("/data/bwwu/lc-rec/qwen-six-task-20261111")
    root.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for variable in ["TMPDIR", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "TORCH_EXTENSIONS_DIR", "XDG_CACHE_HOME", "HF_HOME", "CUDA_CACHE_PATH"]:
        if variable in os.environ:
            folder = Path(os.environ[variable])
            assert folder.resolve().is_relative_to(data.resolve())
            folder.mkdir(parents=True, exist_ok=True)
    ledger_path = root / "gpu-budget.json"
    with (root / "gpu-budget.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {
            "limits": {"task_device_seconds": 14400, "model_device_seconds": 7200, "job_wall_seconds": 2700}, "runs": []}
        assert ledger["limits"] == {"task_device_seconds": 14400, "model_device_seconds": 7200, "job_wall_seconds": 2700}
        assert all(row.get("end_utc") for row in ledger["runs"]), "Original job unresolved; budget cannot be reset"
        assert not any(row["job_id"] == args.job_id for row in ledger["runs"])
        used = sum(row["charged_gpu_device_seconds"] for row in ledger["runs"])
        model_used = sum(row["charged_gpu_device_seconds"] for row in ledger["runs"] if row["key"] == args.key)
        limit = min(2660, (14400 - used) / 2 - 30, (7200 - model_used) / 2 - 30)
        assert limit > 0
        assert not subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip(), "GPU occupied"
        gpu_identity = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total", "--format=csv"], text=True)
        assert gpu_identity.count("RTX 4090") == 2

        def allocated():
            total = 0
            jobs = Path(os.environ["GRID_JOB_DIR"]).parent
            locations = [root, data] + list(jobs.glob("lc-qwen6-20261111-*"))
            for location in locations:
                for directory, directories, files in os.walk(location, followlinks=False):
                    directories[:] = [name for name in directories if not (Path(directory) / name).is_symlink()]
                    for name in files:
                        path = Path(directory) / name
                        try:
                            if not path.is_symlink():
                                total += path.stat().st_blocks * 512
                        except FileNotFoundError:
                            pass
            for folder, pattern in [("transfers", "lc-qwen6-*"), ("requests", "lc-qwen6-20261111-*")]:
                for path in (jobs.parent / folder).glob(pattern):
                    if path.is_file() and not path.is_symlink():
                        total += path.stat().st_blocks * 512
            return total

        assert psutil.virtual_memory().available >= 40 * 1024 ** 3
        assert allocated() < 20 * 1024 ** 3
        for location in ["/home/bwwu", "/data/bwwu"]:
            disk = shutil.disk_usage(location)
            assert disk.free >= max(30 * 1024 ** 3, disk.total // 10)
        started = time.monotonic()
        entry = {"job_id": args.job_id, "key": args.key, "argv": args.command[1:],
                 "gpu_identity": gpu_identity, "start_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 "limit_seconds": limit, "charging": "two devices times entire child lifetime including imports/load/save/failure/inference"}
        ledger["runs"].append(entry)
        write(ledger_path, ledger)
        child = subprocess.Popen(args.command[1:], stdin=subprocess.DEVNULL)
        owner = psutil.Process(child.pid)
        tracked = {(owner.pid, owner.create_time()): owner}
        reason = None
        minimum_mem = psutil.virtual_memory().available
        rss_peak = 0
        disk_peak = 0
        count = 0
        last_progress = time.monotonic()
        latest_event_mtime = 0
        try:
            with (output / "resource-samples.jsonl").open("a", buffering=1) as stream:
                while True:
                    try:
                        for process in [owner] + owner.children(recursive=True):
                            tracked.setdefault((process.pid, process.create_time()), process)
                    except psutil.Error:
                        pass
                    alive = []
                    rss = 0
                    swap = 0
                    for identity, process in tracked.items():
                        try:
                            if process.create_time() != identity[1] or not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
                                continue
                            alive.append(process)
                            rss += process.memory_info().rss
                            for line in Path("/proc", str(process.pid), "status").read_text().splitlines():
                                if line.startswith("VmSwap:"):
                                    swap += int(line.split()[1]) * 1024
                        except (psutil.Error, OSError):
                            pass
                    elapsed = time.monotonic() - started
                    available = psutil.virtual_memory().available
                    minimum_mem = min(minimum_mem, available)
                    rss_peak = max(rss_peak, rss)
                    row = {"elapsed_seconds": elapsed, "mem_available_bytes": available, "task_tree_rss_bytes": rss,
                           "task_swap_bytes": swap, "pids": [process.pid for process in alive]}
                    if count % 3 == 0:
                        row["nvidia_smi"] = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
                        occupants = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
                        foreign = [int(value.strip()) for value in occupants.splitlines() if value.strip().isdigit() and int(value.strip()) not in {process.pid for process in alive}]
                        row["foreign_gpu_pids"] = foreign
                        if foreign:
                            reason = "FOREIGN_GPU_PROCESS"
                        size = allocated()
                        disk_peak = max(disk_peak, size)
                        row["new_disk_allocated_bytes"] = size
                        if size >= 20 * 1024 ** 3:
                            reason = "TASK_20GIB_DISK_BOUND"
                        for location in ["/data/bwwu", "/home/bwwu"]:
                            disk = shutil.disk_usage(location)
                            if disk.free < max(30 * 1024 ** 3, disk.total // 10):
                                reason = "DISK_RESERVE_BOUND"
                    stream.write(json.dumps(row) + "\n")
                    count += 1
                    for event in Path(os.environ["GRID_JOB_DIR"]).glob("output/probe/*rank*.jsonl"):
                        current = event.stat().st_mtime_ns
                        if current > latest_event_mtime:
                            latest_event_mtime = current
                            last_progress = time.monotonic()
                    if time.monotonic() - last_progress > 600:
                        reason = "NO_PROGRESS_600S"
                    if available < 40 * 1024 ** 3:
                        reason = "CPU_40GIB_RESERVE_BOUND"
                    if swap > 64 * 1024 ** 2:
                        reason = "TASK_SWAP_BOUND"
                    if elapsed >= limit:
                        reason = "GPU_TIME_BOUND"
                    if reason or child.poll() is not None:
                        break
                    time.sleep(5)
        finally:
            def still_owned(identity, process):
                try:
                    return process.create_time() == identity[1] and process.is_running() and process.status() != psutil.STATUS_ZOMBIE and process.uids().real == os.getuid()
                except psutil.Error:
                    return False

            remaining = [process for identity, process in tracked.items() if still_owned(identity, process)]
            for identity, process in reversed(list(tracked.items())):
                if still_owned(identity, process):
                    try:
                        process.terminate()
                    except psutil.Error:
                        pass
            if remaining:
                psutil.wait_procs(remaining, timeout=3)
            for identity, process in reversed(list(tracked.items())):
                if still_owned(identity, process):
                    try:
                        process.kill()
                    except psutil.Error:
                        pass
            code = child.wait(timeout=10)
            live = [process.pid for identity, process in tracked.items() if still_owned(identity, process)]
            entry.update(end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started,
                         child_exit_code=code, stop_reason=reason, mem_available_min_bytes=minimum_mem,
                         task_tree_rss_sample_peak_bytes=rss_peak, new_disk_allocated_peak_bytes=disk_peak,
                         tracked_live_pids_after_cleanup=live, cleanup_verified=not live)
            entry["charged_gpu_device_seconds"] = 2 * entry["elapsed_seconds"]
            write(ledger_path, ledger)
            write(output / "resource-summary.json", entry)
        sys.exit(125 if live else 124 if reason else code if code >= 0 else 128 - code)


if __name__ == "__main__":
    main()
