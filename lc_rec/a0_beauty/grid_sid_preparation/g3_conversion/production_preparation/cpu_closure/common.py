"""Shared identity and ledger primitives for the supplemental CPU task."""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

CONDITIONS = ("normal", "s3", "s4")
STAGES = ("train", "verify_best_adapter", "normal", "normal_eval", "s3", "s3_eval", "s4", "s4_eval", "paired")
RUNNER_PYTHON = ("/usr/bin/python3", "-I", "-S", "-B")
TERMINAL = {"SUCCEEDED", "FAILED", "TIMED_OUT", "INTERRUPTED", "WORKERS_TERMINATED"}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_bytes())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    with Path(path).open("rb") as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
        return h.hexdigest()


def pin(path, expected, size=None):
    path = Path(path)
    need(path.is_file(), f"required input missing: {path}")
    if size is not None:
        need(path.stat().st_size == size, f"size mismatch: {path}")
    need(sha(path) == expected, f"SHA-256 mismatch: {path}")
    return path


def facts(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def write(path, value, *, exclusive=False):
    path = Path(path)
    raw = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode() + b"\n"
    if exclusive:
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    else:
        fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def runner(path, *args):
    return json.loads(subprocess.check_output([*RUNNER_PYTHON, str(path), *map(str, args)]))


def read_bound_release(directory):
    directory = Path(directory).resolve(strict=True)
    stamp = read(directory / "deployment.json")
    need(stamp["implementation_commit"] == directory.name and len(directory.name) == 40,
         "deployed directory must be named by the full code commit")
    rows = stamp["files"]
    need(set(rows) | {"deployment.json"} == {p.name for p in directory.iterdir()},
         "deployment member set mismatch")
    for name, row in rows.items():
        target = directory / name
        need(target.parent == directory and not target.is_symlink(), f"unsafe release member: {target}")
        pin(target, row["sha256"], row["bytes"])
    cfg = read(directory / "closure_config.json")
    need(directory == Path(cfg["closure_root"]) / "releases" / stamp["implementation_commit"],
         "unexpected deployment root")
    return cfg, stamp


def original_snapshot(cfg):
    """Read the preserved ledger; the original controller query writes it."""
    root = Path(cfg["original_root"])
    plan_path = pin(root / "plan.json", cfg["plan_raw_sha256"])
    plan = read(plan_path)
    state = read(root / "state.json")
    identity = read(root / "plan.identity.json")
    need(digest(plan) == cfg["plan_canonical_sha256"] == identity["sha256"] and identity["path"] == str(plan_path),
         "canonical plan identity mismatch")
    need(plan["task_id"] == cfg["task_id"] == identity["task_id"] and plan["root"] == str(root),
         "task/root mismatch")
    for key in ("run_commit", "experiment_id", "release", "python", "asset_sha256", "runner_sha256"):
        need(plan[key] == cfg[key], f"original plan binding: {key}")
    need(plan["config_sha256"] == cfg["original_config_sha256"] and plan["runner"] == cfg["original_runner"],
         "original config/runner mismatch")
    need(plan["execution_topology"] == "SINGLE_GPU" and plan["budget"]["total"] == cfg["budget"]["combined_max_seconds"],
         "original topology/budget mismatch")
    need([a["stage"] for a in state["attempts"]] == list(STAGES) and set(state["stages"]) == set(STAGES),
         "original stage ledger changed")
    for idx, attempt in enumerate(state["attempts"]):
        name = attempt["stage"]
        row = state["stages"][name]
        expected = cfg["original_stages"][name]
        observed = attempt["runner"]
        need(row["attempt_index"] == idx and attempt["attempt"] == expected["attempt"] == 0,
             f"original attempt mismatch: {name}")
        need(attempt["job_id"] == expected["job_id"] == observed["job_id"] and
             attempt["request_sha256"] == expected["request_sha256"] == observed["request_sha256"],
             f"original request mismatch: {name}")
        need(attempt["charged_seconds"] == expected["seconds_charged"] and
             attempt["status"] == expected["runner"] == observed["status"] and
             observed["exit_code"] == expected["exit_code"] and observed["cleanup_verified"] is True,
             f"original runtime mismatch: {name}")
        need(row["status"] == ("SUBMITTING" if name == "paired" else "COMPLETE"),
             f"original stage outcome mismatch: {name}")
        need(Path(row["output"]).resolve(strict=True).is_relative_to(root / "attempts" / name),
             f"original output path escapes stage: {name}")
    return plan, state, math.fsum(a["charged_seconds"] for a in state["attempts"])


def query_original_pair(cfg, plan):
    pin(cfg["original_runner"], cfg["runner_sha256"])
    expected = cfg["original_stages"]["paired"]
    receipt = runner(cfg["original_runner"], "query", Path(plan["job_root"]) / expected["job_id"])
    need(receipt["status"] == "TIMED_OUT" and receipt["exit_code"] == -15 and
         receipt["request_sha256"] == expected["request_sha256"] and receipt["cleanup_verified"] and
         receipt["workgroup_observation"]["verified_empty"] and not any(
             receipt[key] for key in ("worker_alive", "supervisor_alive", "workgroup_live")),
         "original paired is not confirmed terminal and cleared")
    argv_marker = f"production_control.py run --plan {cfg['original_root']}/plan.json"
    processes = subprocess.check_output(["ps", "-u", str(os.getuid()), "-o", "args="], text=True)
    need(not any(argv_marker in line for line in processes.splitlines()), "original controller still active")
    return receipt
