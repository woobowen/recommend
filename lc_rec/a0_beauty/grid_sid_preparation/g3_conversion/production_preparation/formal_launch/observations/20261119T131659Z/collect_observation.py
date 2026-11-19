import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path("/data/bwwu/lc-rec/g3-formal-20261118/21d495b0be5b")
CONTROLLER_PID = 3334030
UTC = dt.timezone.utc


def utc(timestamp=None):
    if timestamp is None:
        return dt.datetime.now(UTC).isoformat()
    return dt.datetime.fromtimestamp(timestamp, UTC).isoformat()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode()


def load(path):
    return json.loads(path.read_bytes())


def select(value, keys):
    return {key: value[key] for key in keys.split() if key in value}


def identity(path, hash_bytes=True):
    info = {"path": str(path), "bytes": path.stat().st_size}
    if hash_bytes:
        info["sha256"] = sha(path.read_bytes())
    return info


def file_piece(path, last_bytes=None):
    with path.open("rb") as stream:
        stat = os.fstat(stream.fileno())
        offset = max(0, stat.st_size - last_bytes) if last_bytes else 0
        stream.seek(offset)
        data = stream.read(stat.st_size - offset)
    return {"path": str(path), "source_mtime_utc": utc(stat.st_mtime),
            "source_size_at_open": stat.st_size, "offset_bytes": offset,
            "copied_bytes": len(data), "sha256": sha(data)}, data.decode("utf-8")


def proc_identity(pid, boot):
    folder = Path("/proc") / str(pid)
    fields = (folder / "stat").read_text().rsplit(") ", 1)[1].split()
    return {"pid": pid, "state": fields[0], "ppid": int(fields[1]),
            "pgid": int(fields[2]), "session_id": int(fields[3]), "tty_nr": int(fields[4]),
            "start_ticks": fields[19], "boot_id": boot, "uid": folder.stat().st_uid,
            "argv": (folder / "cmdline").read_bytes().decode().rstrip("\0").split("\0"),
            "cwd": os.readlink(folder / "cwd"),
            "stdin": os.readlink(folder / "fd/0")}


def checked(argv):
    return subprocess.check_output(argv, text=True).strip()


started = utc()
files = {}
ranges = {}


def capture(label, path, last_bytes=None):
    ranges[label], files[label] = file_piece(path, last_bytes)
    return files[label]


state = json.loads(capture("state.json", ROOT / "state.json"))
seal = json.loads(capture("plan.identity.json", ROOT / "plan.identity.json"))
plan_raw = (ROOT / "plan.json").read_bytes()
plan = json.loads(plan_raw)
boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
controller = proc_identity(CONTROLLER_PID, boot)
stage = next((name for name, record in state["stages"].items()
              if record["status"] != "COMPLETE"), None)
if stage != "normal":
    raise RuntimeError(f"Stage changed after discovery, observe {stage!r} from the same ledger")
active = state["attempts"][state["stages"][stage]["attempt_index"]]
job = Path(plan["job_root"]) / active["job_id"]
runner_argv = ["/usr/bin/python3", "-I", "-S", "-B", plan["runner"], "query", str(job)]
query = subprocess.run(runner_argv, text=True, capture_output=True, timeout=40)
if query.returncode:
    raise RuntimeError(query.stderr)
runner_at = utc()
files["runner-query.json"] = query.stdout
runner = json.loads(query.stdout)
supervisor = proc_identity(runner["supervisor"]["pid"], boot)
worker = proc_identity(runner["worker"]["pid"], boot)
training = ROOT / "training"
output = Path(state["stages"][stage]["output"])
rank_path = output / "test/rank0.json"
rank_meta, rank_text = file_piece(rank_path)
rank = json.loads(rank_text)
rank_proc = proc_identity(rank["pid"], boot)
normal_argv = json.loads(capture("normal-compute-argv.json", output / "compute_argv.json"))
capture("normal.work.tail.log", job / "work.log", 6 * 1024)
capture("normal.supervisor.log", job / "supervisor.log", 4 * 1024)
capture("controller.log", ROOT / "controller.log", 4 * 1024)
gpu = checked(["nvidia-smi", "--id=1",
               "--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu",
               "--format=csv,noheader"])
gpu_apps = checked(["nvidia-smi", "--id=1", "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader"])
memory = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
              if line.startswith("MemAvailable:"))
disk = os.statvfs(ROOT)

# Some rank records become visible just before the log line. The committed
# user counter is the newest value present in both saved evidence streams.
def log_progress(line):
    if line.startswith('{"stage": "generation"'):
        return json.loads(line)
    return None


progress = [v for line in files["normal.work.tail.log"].splitlines()
            if (v := log_progress(line)) is not None]
confirmed = next(v for v in reversed(progress) if v["completed_users"] <= rank["completed_users"])
pred_path = output / "test/predictions.rank0.jsonl"
with pred_path.open("rb") as stream:
    pred_st = os.fstat(stream.fileno())
    stream.seek(max(0, pred_st.st_size - 128 * 1024))
    tail = stream.read(pred_st.st_size - stream.tell())
rows = tail.split(b"\n")[1:-1]
last = json.loads(rows[-1])
pred_info = {"path": str(pred_path), "bytes": pred_st.st_size, "mtime_utc": utc(pred_st.st_mtime),
             "last_complete_line_identity": select(last, "row_idx condition scope model_identity_sha256"),
             "last_complete_line_candidates": len(last["candidates"]),
             "last_complete_line_assignment_offset_zero_based": rank["assigned_row_indices"].index(last["row_idx"]),
             "last_complete_line_sha256": sha(rows[-1] + b"\n")}
rank_keys = ("status rank scope condition RUN_COMMIT world_size execution_topology execution_profile "
             "pid argv cwd interpreter completed_users generation data_identity adapter_files "
             "model_identity_sha256 tokenizer_identity strict_adapter_reload assigned_users "
             "assignment_sha256 elapsed_generation_seconds")
files["normal-rank0-extract.json"] = select(rank, rank_keys)

train_report = load(training / "rank0.json")
train_keys = ("status run_commit config_sha256 initial_global_step global_step epoch best_checkpoint "
              "best_metric metrics execution_topology execution_profile data data_identity sample_flow "
              "tokenizer_identity optimizer_membership_exact started_utc elapsed_seconds")
files["train-rank0-extract.json"] = select(train_report, train_keys)
event_path = training / "events.rank0.jsonl"
ev = [(line, json.loads(line)) for line in event_path.read_text().splitlines()]
val = [item for _, item in ev if "eval_loss" in item.get("values", {})]
last_logs = [item for _, item in ev if "loss" in item.get("values", {})][-4:]
steps = [item for _, item in ev if item["event"] == "optimizer_step"]
wanted = {"epoch_end", "checkpoint_complete", "strict_ds_restored", "train_begin", "before_best_restore"}
files["train-events.extract.jsonl"] = "".join(
    line + "\n" for line, item in ev if item["event"] in wanted or item in val + last_logs + steps[-4:]
    or "train_loss" in item.get("values", {}))
capture("train-compute-argv.json", Path(state["stages"]["train"]["output"]) / "compute_argv.json")
train_attempt = state["attempts"][state["stages"]["train"]["attempt_index"]]
train_job = Path(plan["job_root"]) / train_attempt["job_id"]
capture("train.work.tail.log", train_job / "work.log", 5 * 1024)
captured_requests = {}
for attempt in state["attempts"]:
    name = attempt["stage"]
    attempt_job = Path(plan["job_root"]) / attempt["job_id"]
    a = json.loads(capture(f"{name}-accepted.json", attempt_job / "accepted.json"))
    request = load(attempt_job / "request.json")
    request_copy = load(Path(attempt["request_path"]))
    fields = "job_id argv beauty_stage capture_version beauty_config_sha256 beauty_input_identity timeout_seconds cwd"
    captured_requests[name] = {"request_fields": select(request, fields),
                               "canonical_sha256": sha(canonical(request)),
                               "controller_copy_sha256": sha(canonical(request_copy)),
                               "accepted_sha256": a["request_sha256"],
                               "runner_status_sha256": attempt["runner"]["request_sha256"]}
    if name != stage:
        capture(f"{name}-stage_result.json", Path(state["stages"][name]["output"]) / "stage_result.json")

selection = json.loads(capture("best_selection.json", training / "best_selection.json"))
adapter = training / "best_adapter"
adapter_id = json.loads(capture("best-adapter-g3-identity.json", adapter / "g3_identity.json"))
capture("best-adapter-config.json", adapter / "adapter_config.json")
proof = json.loads(files["verify_best_adapter-stage_result.json"])["proof"]
checkpoint_records = {}
for receipt_path in sorted((training / "checkpoint_receipts").glob("*.json")):
    receipt = load(receipt_path)
    cp = Path(receipt["checkpoint_path"])
    retained = cp.is_dir()
    receipt_name = f"{cp.name}-receipt.json"
    capture(receipt_name, receipt_path)
    on_disk = []
    if retained:
        complete_path = cp / "complete.json"
        capture(f"{cp.name}-complete.json", complete_path)
        for entry in receipt["manifest"]["files"]:
            member = cp / entry["path"]
            on_disk.append(dict(entry, actual_bytes=member.stat().st_size,
                                actual_sha256=sha(member.read_bytes()) if entry["bytes"] <= 2 * 1024**2 else None))
    checkpoint_records[cp.name] = {
        "receipt": identity(receipt_path), "manifest_sha256": receipt["manifest_sha256"],
        "manifest_sha_from_receipt": sha(receipt["manifest_json"].encode()),
        "loss": receipt["eval_loss"], "validation_records": receipt["validation_records"],
        "at_step": receipt["global_step"], "epoch": receipt["epoch"], "retained": retained,
        "retained_manifest_sha256": identity(cp / "complete.json")["sha256"] if retained else None,
        "on_disk_members": on_disk}
adapter_files = {name: identity(adapter / name, hash_bytes=row["bytes"] <= 2 * 1024**2)
                 for name, row in proof["adapter_files"].items()}
source_paths = ("lc_rec/a0_beauty/production_control.py", "lc_rec/a0_beauty/production_pipeline.py",
                "lc_rec/a0_beauty/production_identity.py", "lc_rec/a0_beauty/formal_train.py",
                "lc_rec/a0_beauty/formal_evaluate.py", "lc_rec/a0_beauty/single_gpu_training.py",
                "lc_rec/a0_beauty/config/formal_g3_beauty_seed42.json")
release = Path(plan["release"])
sources = {name: identity(release / name) for name in source_paths}
data = {key: identity(Path(plan["data_root"]) / "Beauty" / f"Beauty.{key}.json")
        for key in plan["asset_sha256"]}
charged = {"train_normal": 0.0, "s3": 0.0, "s4": 0.0, "cpu": 0.0}
for a in state["attempts"]:
    name = a["stage"]
    group = "train_normal" if name in {"train", "normal"} else name if name in {"s3", "s4"} else "cpu"
    charged[group] += a["charged_seconds"]
charged["total"] = sum(charged.values())
remaining = {name: plan["budget"][name] - amount for name, amount in charged.items()}
reserved = plan["budget"]["cleanup_reserve"]
limits = captured_requests[stage]["request_fields"]["timeout_seconds"]
red_flags = ("Traceback (most recent call last)", "CUDA out of memory", "ChildFailedError",
             "AssertionError", "FloatingPointError")
summary = {
    "started_utc": started, "observation_utc": utc(), "host": os.uname().nodename,
    "previous_observation": {"commit": "f109fef12b4fa8a27cd67f869c42e69947312c8d",
                             "train_step": 3790, "utc": "2026-11-19T01:53:46Z"},
    "plan": select(plan, "task_id experiment_id experiment_series run_commit config_sha256 execution_topology "
                   "execution_profile budget stage_seconds asset_sha256 release root python gpu_uuids runner_sha256"),
    "plan_sha256": sha(plan_raw), "canonical_plan_sha256": sha(canonical(plan)), "seal": seal,
    "source_files": sources, "runner_file": identity(Path(plan["runner"])), "data_files": data,
    "controller_status": state["status"], "current_stage": stage, "stage_record_status": state["stages"][stage]["status"],
    "state_mtime_utc": ranges["state.json"]["source_mtime_utc"],
    "request_checks": captured_requests, "runner_query": {"argv": runner_argv, "sample_utc": runner_at,
                    "exit_code": query.returncode, "stderr": query.stderr},
    "processes": {"controller": controller, "runner_supervisor": supervisor, "pipeline_worker": worker,
                  "generation_rank": rank_proc},
    "training": {"validation": val, "last_training_logs": last_logs, "sampled_optimizer_event_count": len(steps),
                 "last_sampled_optimizer_events": steps[-4:], "events_file": identity(event_path),
                 "train_rank_file": identity(training / "rank0.json"), "resume_in_compute_argv":
                 "--resume" in json.loads(files.get("train-compute-argv.json", "{}")).get("argv", []),
                 "failure_rank_file_present": (training / "failure.rank0.json").exists(),
                 "checkpoints": checkpoint_records, "best_selection": selection,
                 "best_adapter_checkpoint_source": adapter_id["checkpoint_source"],
                 "best_adapter_members": adapter_files,
                 "last_update_delta": train_report["global_step"] - 3790},
    "generation": {"rank_report": rank_meta, "last_log_progress": progress[-5:],
                   "confirmed_log_and_rank_progress": confirmed, "predictions": pred_info,
                   "normal_off_verification_requested": "--verify-normal-off" in normal_argv["argv"],
                   "stage_result_present": (output / "stage_result.json").exists(),
                   "normal_eval_submitted": "normal_eval" in state["stages"],
                   "rank_failure_in_report": rank["status"] not in {"RUNNING", "PASS"}},
    "resources": {"gpu": gpu, "gpu_compute": gpu_apps, "host_memory_available_bytes": memory,
                  "filesystem_available_bytes": disk.f_bavail * disk.f_frsize,
                  "filesystem_total_bytes": disk.f_blocks * disk.f_frsize},
    "budget": {"charged_seconds": charged, "remaining_seconds": remaining,
               "runner_timeout_seconds_at_admission": limits,
               "normal_time_until_runner_timeout_from_ledger_seconds": limits - active["charged_seconds"],
               "normal_stage_nominal_remaining_seconds": plan["stage_seconds"][stage] - active["charged_seconds"],
               "cleanup_reserve_seconds": reserved},
    "log_error_flags": {name: [flag for flag in red_flags if flag in content]
                        for name, content in files.items() if name.endswith(".log")},
    "raw_file_ranges": ranges,
}
print(json.dumps({"observation": summary, "files": files}, ensure_ascii=False, allow_nan=False))
