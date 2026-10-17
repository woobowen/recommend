"""Finite, deterministic dependency queue around the accepted GRID runner.

This CPU supervisor never chooses research settings, retries a failed job, or
requires a connection to the submitting laptop. Each stage is an immutable
runner request. Its worker holds the GPU locks until the actual command exits.
Future-result review is deliberately outside this mechanical scheduler.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid


WORKFLOWS = {"toys-a-5300", "toys-a-5000", "sports-a1", "sports-a2"}
GROUP_CAPS = {
    "toys-5300": 9000, "toys-5000": 9000,
    "sports-a1-train": 43200, "sports-a2-encode": 10800,
    "sports-a2-rkmeans": 7200, "sports-a2-train": 43200,
    "sports-a1-eval-attention": 9000, "sports-a2-eval-attention": 9000,
    "engineering": 2700,
}
KIND_CAPS = {"admission": 2700, "evaluate": 1800, "attention": 1200,
             "train": 43200, "encode": 10800, "rkmeans": 7200, "cpu": 21600}
ACTIVE = {"ACCEPTED", "RUNNING"}
FINAL_BAD = {"FAILED", "TIMED_OUT", "INTERRUPTED", "WORKERS_TERMINATED",
             "SUPERVISOR_ERROR", "CLEANUP_UNCONFIRMED", "CONTROL_LOST", "UNCONFIRMED"}
FINAL_QUEUE = {"SUCCEEDED", "BLOCKED_NO_RETRY", "BLOCKED_DEPENDENCY", "BLOCKED_GATE"}


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with tmp.open("wb") as f:
        f.write(canonical(value) + b"\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,90}", value):
        raise ValueError("invalid ID: " + repr(value))
    return value


def gpus(stage):
    result = stage.get("gpu_uuids", [])
    if stage.get("gpu_uuid"):
        if result:
            raise ValueError("use gpu_uuid or gpu_uuids, never both")
        result = [stage["gpu_uuid"]]
    if not isinstance(result, list) or len(set(result)) != len(result):
        raise ValueError("invalid GPU list")
    for value in result:
        if not isinstance(value, str) or not re.fullmatch(r"GPU-[A-Fa-f0-9-]+", value):
            raise ValueError("GPU must be a physical UUID")
    return result


def lookup(value, key):
    for part in key.split("."):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def validate(config, path, require_ready=True):
    if config.get("schema") != 1 or (require_ready and not config.get("ready")):
        raise ValueError("queue is not bound and approved")
    safe_id(config["queue_id"])
    root = Path(config["snapshot"]["root"])
    if not root.is_absolute() or not re.fullmatch(r"[0-9a-f]{40}", config["snapshot"]["commit"]):
        raise ValueError("fixed snapshot identity required")
    if not Path(path).is_absolute():
        raise ValueError("absolute immutable queue configuration path required")
    for key in ["state_dir", "gpu_locks_dir"]:
        if not Path(config[key]).is_absolute():
            raise ValueError("absolute " + key + " required")
    if not 1 <= config["poll_seconds"] <= 60 or not 0 < config["max_queue_seconds"] <= 172800:
        raise ValueError("queue timing outside approved bounds")
    if not 0 < config["gpu_budget_seconds"] <= 143100:
        raise ValueError("total GPU request budget exceeds 39.75 hours")
    if not 0 < config["max_storage_bytes"] <= 50 * 1024 ** 3:
        raise ValueError("new persistent output budget exceeds 50GiB")
    if not config["free_space"] or not config["storage_roots"]:
        raise ValueError("disk protection is mandatory")
    for item in config["free_space"]:
        if not Path(item["path"]).is_absolute() or item["min_free_bytes"] <= 0:
            raise ValueError("invalid free-space guard")
    for item in config["storage_roots"]:
        if not Path(item).is_absolute():
            raise ValueError("output roots must be absolute")
    caps = config["budget_groups"]
    for group, cap in caps.items():
        if group not in GROUP_CAPS or not 0 < cap <= GROUP_CAPS[group]:
            raise ValueError("unapproved budget group: " + group)
    stages = config["stages"]
    by_id = {safe_id(s["id"]): s for s in stages}
    if len(by_id) != len(stages):
        raise ValueError("duplicate stage IDs")
    if set(config.get("blocked_workflows", {})) - WORKFLOWS:
        raise ValueError("unknown blocked workflow")
    jobs, total, a2_cpu, groups = set(), 0, 0, {key: 0 for key in caps}
    for stage in stages:
        if stage["workflow"] not in WORKFLOWS or stage["kind"] not in KIND_CAPS:
            raise ValueError("unapproved workflow or stage kind")
        devices = gpus(stage)
        policy = stage.get("resource_policy", {})
        if set(policy) - {"allow_shared", "min_free_mib", "max_peer_memory_mib",
                          "estimated_peak_mib", "safety_margin_mib", "min_runtime_free_mib",
                          "max_prelaunch_utilization_percent", "sample_seconds", "required_samples"}:
            raise ValueError("unsupported resource policy; no implied duty-cycle throttling")
        if policy.get("allow_shared"):
            if "estimated_peak_mib" in policy:
                if "max_peer_memory_mib" in policy or "min_free_mib" in policy:
                    raise ValueError("demand-based sharing cannot impose a fixed peer-memory ceiling")
                if not 0 < policy["estimated_peak_mib"] <= 24564 or not 512 <= policy.get("safety_margin_mib", 0) <= 8192:
                    raise ValueError("stage demand and explicit safety margin required")
                if not 256 <= policy.get("min_runtime_free_mib", 0) <= policy["safety_margin_mib"]:
                    raise ValueError("invalid runtime headroom")
            elif policy.get("min_free_mib", 0) < 8192 or not 0 < policy.get("max_peer_memory_mib", 0) <= 2048:
                raise ValueError("invalid legacy encoding-only resource policy")
            if not 0 <= policy.get("max_prelaunch_utilization_percent", 100) <= 60:
                raise ValueError("bounded sustained prelaunch load is required")
            if not 2 <= policy.get("required_samples", 2) <= 3 or not 1 <= policy.get("sample_seconds", 2) <= 5:
                raise ValueError("shared resource policy requires two or three bounded samples")
        if stage.get("gpu_candidates"):
            candidates = stage["gpu_candidates"]
            if len(devices) != 1 or len(candidates) != len(set(candidates)) or devices[0] not in candidates:
                raise ValueError("only single-card stages may choose approved alternative devices")
            if not set(candidates) <= set(config.get("approved_gpu_uuids", [])):
                raise ValueError("candidate device outside approved physical GPUs")
        r = stage["request"]
        job = safe_id(r["job_id"])
        if job in jobs:
            raise ValueError("duplicate runner job ID")
        jobs.add(job)
        if not isinstance(r["argv"], list) or not r["argv"] or not all(isinstance(v, str) for v in r["argv"]):
            raise ValueError("literal argv required")
        if not Path(r["argv"][0]).is_absolute() or not Path(r["cwd"]).is_absolute():
            raise ValueError("absolute executable/cwd required")
        if Path(r["cwd"]).resolve() != root.resolve():
            raise ValueError("stage cwd must be fixed source snapshot")
        if not 10 < r["timeout_seconds"] <= KIND_CAPS[stage["kind"]]:
            raise ValueError("per-stage wall cap exceeded: " + stage["id"])
        if stage["kind"] == "cpu" and devices:
            raise ValueError("CPU stages cannot request GPUs")
        if stage["kind"] == "cpu" and stage["workflow"] == "sports-a2":
            a2_cpu += r["timeout_seconds"]
        if stage["kind"] != "cpu" and not devices:
            raise ValueError("GPU stage must bind physical devices")
        if any(k in r.get("env", {}) for k in ["GRID_JOB_ID", "GRID_JOB_DIR"]):
            raise ValueError("runner ownership markers cannot be supplied")
        for key, value in config["namespace_identity"].items():
            if r.get(key) != value:
                raise ValueError("stage namespace mismatch: " + key)
        if r.get("schema") != 1 or r.get("deployment_sha256") != config["runner"]["deployment_sha256"]:
            raise ValueError("runner deployment mismatch")
        for dep in stage["depends_on"]:
            if dep not in by_id or dep == stage["id"]:
                raise ValueError("invalid dependency")
        reserved = r["timeout_seconds"] * len(devices)
        if reserved:
            group = stage["budget_group"]
            if group not in caps:
                raise ValueError("missing GPU group cap")
            if stage["kind"] == "rkmeans" and reserved > 7200:
                raise ValueError("RKmeans exceeds two GPU-hours including every device")
            groups[group] += reserved
            total += reserved
        for gate in stage.get("gates", []):
            if not Path(gate["path"]).is_absolute():
                raise ValueError("gate must address an absolute file")
            if set(gate) - {"path", "sha256", "sha256_from", "json_equals", "min_bytes"}:
                raise ValueError("unsupported mechanical gate")
    def visit(name, ancestors):
        if name in ancestors:
            raise ValueError("dependency cycle")
        for dep in by_id[name]["depends_on"]:
            visit(dep, ancestors | {name})
    for name in by_id:
        visit(name, set())
    if any(groups[g] > caps[g] for g in groups) or total > config["gpu_budget_seconds"]:
        raise ValueError("immutable requests exceed approved GPU budget")
    if a2_cpu > 21600:
        raise ValueError("A2 non-GPU requests exceed six CPU-stage wall-hours")
    present = {s["workflow"] for s in stages} | set(config.get("blocked_workflows", {}))
    if require_ready and present != WORKFLOWS:
        raise ValueError("all four workflows need explicit stage or blocking identity")
    return {"reserved_gpu_seconds": total, "groups": groups,
            "sports_a2_reserved_cpu_seconds": a2_cpu, "stages": len(stages)}


def wrapped_request(config, config_path, stage):
    r = json.loads(json.dumps(stage["request"]))
    script = str(Path(__file__).resolve())
    r["argv"] = [r["argv"][0], "-u", script, "run-stage", "--config",
                 str(Path(config_path).resolve()), "--stage", stage["id"]]
    r.setdefault("env", {})["CUDA_VISIBLE_DEVICES"] = ",".join(gpus(stage))
    inputs = {item["path"]: item for item in r["inputs"]}
    for path in [script, str(Path(config_path).resolve())]:
        inputs[path] = {"path": path, "sha256": sha(path)}
    r["inputs"] = list(inputs.values())
    r["queue_stage"] = {"id": stage["id"], "workflow": stage["workflow"],
                        "config_sha256": sha(config_path), "command": stage["request"]["argv"],
                        "budget_group": stage.get("budget_group"), "gpu_uuids": gpus(stage),
                        "source_commit": config["snapshot"]["commit"]}
    return r


def runner_call(config, mode, argument):
    runner = config["runner"]
    if sha(runner["script"]) != runner["script_sha256"]:
        raise ValueError("accepted runner bytes changed")
    args = [runner["python"], "-I", "-S", "-B", runner["script"], mode]
    args += [runner["job_root"], str(argument)] if mode == "submit" else [str(argument)]
    # Nested independent supervisors must not inherit the outer queue markers.
    # Otherwise the old runner correctly flags them as escaped task members.
    env = os.environ.copy()
    env.pop("GRID_JOB_ID", None)
    env.pop("GRID_JOB_DIR", None)
    p = subprocess.run(args, stdin=subprocess.DEVNULL, capture_output=True,
                       text=True, timeout=45, env=env)
    if p.returncode:
        raise RuntimeError("runner call failed; query original ID, never resubmit blindly: " + p.stderr[-2000:])
    return json.loads(p.stdout)


def gpu_snapshot():
    args = ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits"]
    q = subprocess.run(args, capture_output=True, text=True, timeout=15, check=True)
    cards = {}
    for row in csv.reader(q.stdout.splitlines()):
        index, uid, name, total, used, free, util = [v.strip() for v in row]
        cards[uid] = dict(index=int(index), name=name, total_mib=int(total),
                          used_mib=int(used), free_mib=int(free), utilization=int(util), processes=[])
    q = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                        "--format=csv,noheader,nounits"], capture_output=True,
                       text=True, timeout=15, check=True)
    for row in csv.reader(q.stdout.splitlines()):
        uid, pid, name, memory = [v.strip() for v in row]
        if uid not in cards:
            raise RuntimeError("compute process reported on unknown GPU")
        cards[uid]["processes"].append(dict(pid=int(pid), process_name=name, used_mib=int(memory)))
    return cards


def resource_assessment(stage, cards, running=False):
    """Judge remaining stage demand; peer presence alone is never a shared veto."""
    policy = stage.get("resource_policy", {})
    result = {}
    for device in gpus(stage):
        if device not in cards:
            result[device] = {"ok": False, "reason": "physical device absent"}
            continue
        card = cards[device]
        peers, owned_mib = list(card["processes"]), 0
        if running:
            foreign = []
            for process in peers:
                try:
                    if os.getpgid(process["pid"]) == os.getpgrp():
                        owned_mib += process["used_mib"]
                    else:
                        foreign.append(process)
                except ProcessLookupError:
                    continue
            peers = foreign
        free = card["free_mib"]
        if not policy.get("allow_shared"):
            result[device] = {"ok": not peers, "reason": "legacy unshared stage", "free_mib": free}
            continue
        if "estimated_peak_mib" in policy:
            remaining = max(0, policy["estimated_peak_mib"] - owned_mib) if running else policy["estimated_peak_mib"]
            required = max(policy["min_runtime_free_mib"], remaining + policy["safety_margin_mib"])
            load_ok = running or card["utilization"] <= policy["max_prelaunch_utilization_percent"]
            result[device] = {"ok": free >= required and load_ok, "free_mib": free,
                "owned_mib": owned_mib, "remaining_estimated_peak_mib": remaining,
                "required_free_mib": required, "peer_memory_mib": sum(p["used_mib"] for p in peers),
                "utilization_percent": card["utilization"], "load_ok": load_ok,
                "reason": "stage peak plus headroom and sampled load; no fixed peer-memory cap"}
        else:
            # Historical completed encoding request compatibility only.
            result[device] = {"ok": free >= policy["min_free_mib"] and
                sum(p["used_mib"] for p in peers) <= policy["max_peer_memory_mib"] and
                (running or card["utilization"] <= policy["max_prelaunch_utilization_percent"]),
                "free_mib": free, "reason": "legacy encoding policy"}
    return result


def resource_ok(stage, cards, running=False):
    return all(v["ok"] for v in resource_assessment(stage, cards, running).values())


def prelaunch_samples(stage, first=None):
    samples = [{"at": utc(), "gpus": gpu_snapshot() if first is None else first}]
    policy = stage.get("resource_policy", {})
    count = policy.get("required_samples", 2) if policy.get("allow_shared") else 1
    for _ in range(1, count):
        time.sleep(policy.get("sample_seconds", 2))
        samples.append({"at": utc(), "gpus": gpu_snapshot()})
    okay = all(resource_ok(stage, v["gpus"]) for v in samples)
    for sample in samples:
        sample["resource_assessment"] = resource_assessment(stage, sample["gpus"])
    if "estimated_peak_mib" in policy:
        for device in gpus(stage):
            if any(device not in s["gpus"] for s in samples):
                okay = False
                continue
            used = [s["gpus"][device]["used_mib"] for s in samples]
            observed_swing = max(used) - min(used)
            required = policy["estimated_peak_mib"] + policy["safety_margin_mib"] + observed_swing
            for sample in samples:
                assessment = sample["resource_assessment"][device]
                assessment.update(observed_memory_swing_mib=observed_swing, sampled_required_free_mib=required)
                if assessment["free_mib"] < required:
                    assessment["ok"] = False
                    okay = False
    return okay, samples


def allocation_stage(stage, devices):
    """Freeze one approved physical allocation without changing research argv."""
    if devices == gpus(stage):
        return stage
    if len(devices) != 1 or devices[0] not in stage.get("gpu_candidates", []):
        raise ValueError("unapproved stage GPU allocation")
    value = json.loads(json.dumps(stage))
    value.pop("gpu_uuid", None)
    value["gpu_uuids"] = list(devices)
    return value


def existing_stage(stage, job):
    actual = read(Path(job) / "request.json")
    return allocation_stage(stage, actual["queue_stage"]["gpu_uuids"])


def initial_stage_states(config):
    return {s["id"]: dict(job_id=s["request"]["job_id"], status="QUEUED") for s in config["stages"]}


def queue_should_stop(config, started):
    reserve = min(60, config["max_queue_seconds"] * 0.1)
    return time.monotonic() - started >= config["max_queue_seconds"] - reserve


@contextlib.contextmanager
def gpu_locks(config, devices):
    handles = []
    root = Path(config["gpu_locks_dir"])
    root.mkdir(parents=True, exist_ok=True)
    try:
        for device in sorted(devices):
            handle = (root / (device + ".lock")).open("a")
            handles.append(handle)
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        for handle in handles:
            handle.close()


def disk_snapshot(config):
    free = [{"path": v["path"], "free_bytes": shutil.disk_usage(v["path"]).free,
             "min_free_bytes": v["min_free_bytes"]} for v in config["free_space"]]
    total = 0
    for root in config["storage_roots"]:
        for directory, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [d for d in dirs if not (Path(directory) / d).is_symlink()]
            for name in files:
                path = Path(directory) / name
                try:
                    if not path.is_symlink():
                        total += path.stat().st_size
                except FileNotFoundError:
                    pass
    ok = all(v["free_bytes"] >= v["min_free_bytes"] for v in free) and total <= config["max_storage_bytes"]
    return dict(ok=ok, files_logical_bytes=total, max_bytes=config["max_storage_bytes"], free=free)


def check_gates(stage):
    receipts = []
    for gate in stage.get("gates", []):
        path = Path(gate["path"])
        if not path.is_file() or path.is_symlink() or path.stat().st_size < gate.get("min_bytes", 1):
            raise ValueError("gate file absent/empty/symlink: " + str(path))
        expected = gate.get("sha256")
        sources = []
        if "sha256_from" in gate:
            source = gate["sha256_from"]
            expected = lookup(read(source["path"]), source["field"])
            sources.append(dict(path=source["path"], sha256=sha(source["path"])))
        actual = sha(path)
        if expected and actual != expected:
            raise ValueError("gate content hash mismatch: " + str(path))
        if "json_equals" in gate:
            data = read(path)
            for key, value in gate["json_equals"].items():
                if lookup(data, key) != value:
                    raise ValueError("gate JSON field mismatch: " + str(path) + ":" + key)
        receipts.append(dict(path=str(path), bytes=path.stat().st_size, sha256=actual, hash_sources=sources))
    return receipts


def elapsed(status):
    if not status.get("started_at"):
        return 0.0
    begin = dt.datetime.fromisoformat(status["started_at"])
    end = dt.datetime.fromisoformat(status["ended_at"]) if status.get("ended_at") else dt.datetime.now(dt.timezone.utc)
    return max(0.0, (end - begin).total_seconds())


def accounting(config, state):
    groups = {k: 0.0 for k in config["budget_groups"]}
    for stage in config["stages"]:
        if gpus(stage):
            groups[stage["budget_group"]] += elapsed(state["stages"][stage["id"]].get("runner_status", {})) * len(gpus(stage))
    return {"used_gpu_request_seconds": sum(groups.values()), "groups": groups,
            "remaining_gpu_request_seconds": config["gpu_budget_seconds"] - sum(groups.values())}


def supervise(config, config_path):
    summary = validate(config, config_path)
    root = Path(config["state_dir"])
    root.mkdir(parents=True, exist_ok=True)
    config_hash = sha(config_path)
    with (root / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "queue-status.json").exists():
            raise RuntimeError("queue already claimed; query saved status, no automatic restart")
        state = dict(queue_id=config["queue_id"], config_sha256=config_hash, started_at=utc(),
                     supervisor_pid=os.getpid(), status="RUNNING", reservation=summary,
                     blocked_workflows=config.get("blocked_workflows", {}),
                     stages=initial_stage_states(config))
        atomic(root / "queue-status.json", state)
        started = time.monotonic()
        stop = False
        def stop_after_signal(signum, frame):
            nonlocal stop
            stop = True
        signal.signal(signal.SIGTERM, stop_after_signal)
        signal.signal(signal.SIGINT, stop_after_signal)
        while True:
            if sha(config_path) != config_hash:
                raise RuntimeError("immutable queue configuration changed")
            state.update(observed_at=utc(), disk=disk_snapshot(config))
            cards = gpu_snapshot()
            state["gpus"] = cards
            reserved = set()
            for stage in config["stages"]:
                item = state["stages"][stage["id"]]
                job = Path(config["runner"]["job_root"]) / stage["request"]["job_id"]
                if (job / "accepted.json").exists():
                    stage = existing_stage(stage, job)
                    actual = runner_call(config, "query", job)
                    expected = hashlib.sha256(canonical(wrapped_request(config, config_path, stage))).hexdigest()
                    if actual["request_sha256"] != expected:
                        raise RuntimeError("existing job ID differs from approved immutable request")
                    item["runner_status"] = actual
                    if actual["status"] == "SUCCEEDED" and actual["exit_code"] == 0 and actual["cleanup_verified"]:
                        item["status"] = "SUCCEEDED"
                    elif actual["status"] in ACTIVE:
                        item["status"] = actual["status"]
                        reserved.update(gpus(stage))
                    else:
                        item.update(status="BLOCKED_NO_RETRY", reason="terminal or unconfirmed predecessor: " + actual["status"])
            state["accounting"] = accounting(config, state)
            if state["accounting"]["used_gpu_request_seconds"] > config["gpu_budget_seconds"] or any(
                    state["accounting"]["groups"][g] > cap for g, cap in config["budget_groups"].items()):
                raise RuntimeError("GPU accounting exceeds approved hard limit; no further submission")
            for stage in config["stages"]:
                if stop or queue_should_stop(config, started):
                    break
                item = state["stages"][stage["id"]]
                if item["status"] in FINAL_QUEUE | ACTIVE:
                    continue
                deps = [state["stages"][name]["status"] for name in stage["depends_on"]]
                if any(v.startswith("BLOCKED") for v in deps):
                    item.update(status="BLOCKED_DEPENDENCY", reason="predecessor failed; no retry")
                    continue
                if any(v != "SUCCEEDED" for v in deps):
                    item["status"] = "WAITING_DEPENDENCY"
                    continue
                if not state["disk"]["ok"]:
                    item["status"] = "WAITING_DISK"
                    continue
                try:
                    item["gate_receipts"] = check_gates(stage)
                except (ValueError, KeyError, FileNotFoundError) as error:
                    item.update(status="BLOCKED_GATE", reason=str(error))
                    continue
                devices = gpus(stage)
                if any(d not in cards for d in devices):
                    item.update(status="BLOCKED_GATE", reason="physical GPU UUID absent")
                    continue
                options = [[d] for d in stage.get("gpu_candidates", [])] or [devices]
                available, samples = False, []
                selected = stage
                for option in options:
                    if any(d in reserved for d in option):
                        continue
                    selected = allocation_stage(stage, option)
                    available, samples = prelaunch_samples(selected) if option else (True, [])
                    if available:
                        break
                stage = selected
                devices = gpus(stage)
                item["prelaunch_resource_samples"] = samples
                if not available:
                    item["status"] = "WAITING_GPU"
                    continue
                try:
                    with gpu_locks(config, devices):
                        pass
                except BlockingIOError:
                    item["status"] = "WAITING_GPU_LOCK"
                    continue
                request = wrapped_request(config, config_path, stage)
                request_path = root / "requests" / (request["job_id"] + ".json")
                if request_path.exists() and read(request_path) != request:
                    raise RuntimeError("immutable request conflict")
                atomic(request_path, request)
                if stop or queue_should_stop(config, started):
                    break
                # Persist intent before submission. An ambiguous failure stops the
                # queue; the next human queries this exact ID before any action.
                item.update(status="SUBMITTING", request_path=str(request_path), submission_intent_at=utc())
                atomic(root / "queue-status.json", state)
                actual = runner_call(config, "submit", request_path)
                item.update(status=actual["status"], runner_status=actual, accepted_at=utc())
                atomic(root / "receipts" / (request["job_id"] + ".json"), actual)
                reserved.update(devices)
            atomic(root / "queue-status.json", state)
            with (root / "health.jsonl").open("ab") as log:
                log.write(canonical(dict(observed_at=state["observed_at"], disk=state["disk"], gpus=cards,
                                         accounting=state["accounting"],
                                         stages={k: v["status"] for k, v in state["stages"].items()})) + b"\n")
            if all(v["status"] in FINAL_QUEUE for v in state["stages"].values()):
                state.update(status="TERMINAL_PENDING_RESULT_REVIEW", ended_at=utc())
                atomic(root / "queue-status.json", state)
                return 0
            if stop or queue_should_stop(config, started):
                state.update(status="STOPPED_NO_NEW_SUBMISSIONS", ended_at=utc(),
                             reason="signal or queue wall limit; accepted independent children continue")
                atomic(root / "queue-status.json", state)
                return 2
            time.sleep(config["poll_seconds"])


def run_stage(config, config_path, name):
    validate(config, config_path)
    stage = next(s for s in config["stages"] if s["id"] == name)
    request = stage["request"]
    job_dir = Path(os.environ["GRID_JOB_DIR"])
    stage = existing_stage(stage, job_dir)
    worker_entered = time.monotonic()
    if os.environ["GRID_JOB_ID"] != request["job_id"]:
        raise ValueError("stage/runner job identity mismatch")
    expected = wrapped_request(config, config_path, stage)
    if read(job_dir / "request.json") != expected:
        raise ValueError("actual runner request differs from immutable approved wrapper")
    with gpu_locks(config, gpus(stage)):
        available, samples = prelaunch_samples(stage) if gpus(stage) else (True, [])
        cards = samples[-1]["gpus"] if samples else {}
        if not available:
            raise RuntimeError("GPU changed after scheduling; refuse occupied device; no automatic retry")
        disk = disk_snapshot(config)
        if not disk["ok"]:
            raise RuntimeError("insufficient free space or output limit")
        receipts = check_gates(stage)
        atomic(job_dir / "output" / "stage-entry.json", dict(stage=name, entered_at=utc(),
               source_commit=config["snapshot"]["commit"], config_sha256=sha(config_path),
               command=request["argv"], gate_receipts=receipts, disk=disk, gpus=cards,
               resource_policy=stage.get("resource_policy", {}), prelaunch_resource_samples=samples))
        interrupted = False
        def mark_interrupt(signum, frame):
            nonlocal interrupted
            interrupted = True
        signal.signal(signal.SIGTERM, mark_interrupt)
        signal.signal(signal.SIGINT, mark_interrupt)
        env = os.environ.copy()
        env.update(request.get("env", {}))
        env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus(stage))
        start = time.monotonic()
        consumed = max(elapsed(read(job_dir / "status.json")), start - worker_entered)
        available = request["timeout_seconds"] - consumed - 10
        if available <= 0:
            raise TimeoutError("entry checks consumed the stage wall allowance; command not launched")
        deadline = start + available
        child = subprocess.Popen(request["argv"], cwd=request["cwd"], env=env,
                                 stdin=subprocess.DEVNULL, close_fds=True)
        reason = None
        while child.poll() is None:
            if interrupted:
                reason = "INTERRUPTED"
            elif time.monotonic() >= deadline:
                reason = "TIME_BUDGET"
            elif not disk_snapshot(config)["ok"]:
                reason = "SPACE_GUARD"
            elif stage.get("resource_policy", {}).get("allow_shared"):
                cards_now = gpu_snapshot()
                if not resource_ok(stage, cards_now, running=True):
                    atomic(job_dir / "output" / "resource-guard.json", {"at": utc(), "gpus": cards_now,
                        "assessment": resource_assessment(stage, cards_now, running=True)})
                    reason = "SHARED_RESOURCE_GUARD"
            if reason:
                child.terminate()
                try:
                    child.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=2)
                break
            try:
                child.wait(timeout=max(0.01, min(10, deadline - time.monotonic())))
            except subprocess.TimeoutExpired:
                pass
        rc = child.returncode
        atomic(job_dir / "output" / "stage-exit.json", dict(stage=name, ended_at=utc(),
               command_exit_code=rc, stop_reason=reason, command_wall_seconds=time.monotonic()-start,
               result_review="NOT_PERFORMED_BY_MECHANICAL_QUEUE"))
        return 124 if reason else rc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["validate", "render-requests", "supervise", "run-stage", "status"])
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--stage")
    p.add_argument("--allow-unbound", action="store_true")
    args = p.parse_args()
    config = read(args.config)
    if args.mode == "status":
        print(json.dumps(read(Path(config["state_dir"]) / "queue-status.json"), indent=2))
        return 0
    summary = validate(config, args.config, not args.allow_unbound)
    if args.mode == "validate":
        print(json.dumps(summary, indent=2))
        return 0
    if args.mode == "render-requests":
        print(json.dumps([wrapped_request(config, args.config, s) for s in config["stages"]], indent=2))
        return 0
    if args.mode == "run-stage":
        return run_stage(config, args.config, args.stage)
    return supervise(config, args.config)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BaseException as error:
        if isinstance(error, SystemExit):
            raise
        print(json.dumps(dict(at=utc(), error=repr(error), action="STOP_NO_AUTOMATIC_RETRY")), flush=True)
        sys.exit(2)
