"""Read-only, job-scoped observations and tar capture for formal-r0."""
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tarfile

JOB = Path("/home/bwwu/grid-audits/sid-extensions-p0/jobs/grid-letter-rerun-20261125-formal-r0")
RUN = JOB / "output/run"
BASE = Path("/data/bwwu/grid-letter-rerun-20261125")
INPUTS = BASE / "inputs"


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path):
    return json.loads(path.read_bytes())


def stamp(info):
    return {"inode": info.st_ino, "device": info.st_dev, "bytes": info.st_size,
            "mtime_ns": info.st_mtime_ns,
            "mtime_utc": dt.datetime.fromtimestamp(info.st_mtime, dt.timezone.utc).isoformat()}


def read_prefix(path, tail=None):
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Not a regular file: {path}")
        start = max(0, before.st_size - tail) if tail is not None else 0
        stream.seek(start)
        data = stream.read(before.st_size - start)
        after = os.fstat(stream.fileno())
    if len(data) != before.st_size - start:
        raise IOError(f"Short read: {path}")
    return data, {"source": str(path), "start_byte": start, "end_byte_exclusive": start + len(data),
                  "before": stamp(before), "after": stamp(after), "captured_at": utc()}


def active_paths():
    fixed = [JOB / name for name in ("work.log", "supervisor.log", "status.json", "cleanup.json")]
    fixed += [RUN / name for name in ("startup.json", "training/validation.jsonl",
              "training/model.json", "training/effective_config.yaml", "training/overrides.json",
              "training/best.json", "training/training.json")]
    fixed += sorted((RUN / "stages").glob("*.json"))
    fixed += sorted((RUN / "training/csv").glob("version_*/*"))
    fixed += sorted(p for p in (RUN / "evaluation").rglob("*") if p.is_file())
    fixed += sorted((RUN / "upstream").glob("*.log"))
    fixed += [RUN / "upstream/quantizer_train/fit_batches.jsonl"]
    return [p for p in fixed if p.is_file() and not p.is_symlink()]


def process(saved):
    proc = Path("/proc") / str(saved["pid"])
    try:
        rec = (proc / "stat").read_text().split(") ", 1)[1].split()
        if rec[19] != saved["start_ticks"]:
            return {"pid": saved["pid"], "start_ticks_match": False}
        return {"pid": saved["pid"], "start_ticks_match": True, "start_ticks": rec[19],
                "state": rec[0], "ppid": int(rec[1]), "pgid": int(rec[2]), "sid": int(rec[3]),
                "tty_nr": int(rec[4]), "utime_ticks": int(rec[11]), "stime_ticks": int(rec[12]),
                "argv": (proc / "cmdline").read_bytes().strip(b"\0").decode().split("\0")}
    except FileNotFoundError:
        return {"pid": saved["pid"], "exited_during_sample": True}


def observe():
    start = utc()
    s = read_json(JOB / "status.json")
    members = (s.get("worker_group") or {}).get("known_members", [])
    if isinstance(members, dict):
        members = list(members.values())
    primary = [s[name] for name in ("worker", "supervisor") if s.get(name)]
    processes = {m["pid"]: process(m) for m in members + primary}
    files = {}
    for path in active_paths():
        name = str(path.relative_to(JOB))
        if path.name in {"work.log", "supervisor.log", "metrics.csv", "validation.jsonl",
                         "fit_batches.jsonl", "quantizer_train.log", "sid_generation.log"}:
            data, bounds = read_prefix(path, 8192)
            files[name] = bounds | {"tail": data.decode("utf-8", "replace")}
        else:
            files[name] = stamp(path.stat())
    gpus = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
                                   "--format=csv"], text=True)
    gpu_apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                                       "--format=csv"], text=True)
    checkpoints = {}
    for p in sorted((RUN / "training/checkpoints").glob("*.ckpt")):
        try:
            checkpoints[p.name] = stamp(p.stat())
        except FileNotFoundError:
            continue
    return {"sample_started_utc": start, "sample_ended_utc": utc(), "status": s,
            "processes": processes, "files": files, "checkpoints": checkpoints,
            "gpu": gpus, "gpu_apps": gpu_apps,
            "free_bytes": {str(p): shutil.disk_usage(p).free for p in (JOB, BASE)}}


def archive(selection):
    started = utc()
    entries = {}
    with tarfile.open(mode="w|", fileobj=sys.stdout.buffer) as dest:
        def add(name, data, meta):
            info = tarfile.TarInfo(name)
            info.mode = 0o600
            info.size = len(data)
            dest.addfile(info, io.BytesIO(data))
            entries[name] = meta | {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}

        if selection == "closed":
            for f in ("quantizer", "sid", "bound"):
                assert read_json(RUN / "stages" / (f + ".json"))["exit_code"] == 0
            paths = [p for folder in ("upstream", "bound") for p in sorted((RUN / folder).rglob("*"))
                     if p.is_file() and not p.is_symlink() and p.suffix != ".ckpt"]
            paths += [RUN / "startup.json", *(RUN / "stages" / (f + ".json") for f in ("quantizer", "sid", "bound"))]
            paths += sorted(p for p in INPUTS.iterdir() if p.is_file() and not p.is_symlink())
        elif selection == "cutoff":
            sample = observe()
            add("sample.json", (json.dumps(sample, indent=2) + "\n").encode(), {"kind": "observation"})
            paths = active_paths()
        else:
            raise ValueError(selection)
        for path in paths:
            if path.is_relative_to(JOB):
                name = str(path.relative_to(JOB))
            else:
                assert path.is_relative_to(INPUTS)
                name = "inputs/" + path.name
            data, meta = read_prefix(path)
            add(name, data, meta)
        recorded = {"selection": selection, "started_utc": started, "ended_utc": utc(),
                    "range": "Per-file prefixes bounded by size of each opened FD. Reads are sequential, not a filesystem snapshot.",
                    "files": entries}
        data = (json.dumps(recorded, indent=2) + "\n").encode()
        info = tarfile.TarInfo("capture.json")
        info.mode = 0o600
        info.size = len(data)
        dest.addfile(info, io.BytesIO(data))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "observe":
        print(json.dumps(observe(), indent=2))
    else:
        archive(mode)
