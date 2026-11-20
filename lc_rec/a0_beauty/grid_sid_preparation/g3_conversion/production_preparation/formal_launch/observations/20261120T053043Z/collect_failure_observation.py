import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path("/data/bwwu/lc-rec/g3-formal-20261118/21d495b0be5b")
STAGES = ("train", "verify_best_adapter", "normal", "normal_eval", "s3", "s3_eval",
          "s4", "s4_eval", "paired", "collect")
EXPECTED_USERS = 22363
UTC = dt.timezone.utc


def utc(timestamp=None):
    return (dt.datetime.now(UTC) if timestamp is None else dt.datetime.fromtimestamp(timestamp, UTC)).isoformat()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(data):
    return sha(json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode())


def load(path):
    return json.loads(path.read_bytes())


def identity(path, hash_file=True):
    st = path.stat()
    row = {"path": str(path), "bytes": st.st_size, "mtime_utc": utc(st.st_mtime)}
    if hash_file:
        row["sha256"] = sha(path.read_bytes())
    return row


def select(data, keys):
    return {k: data[k] for k in keys.split() if k in data}


def query(args):
    return subprocess.check_output(args, text=True).strip()


started = utc()
files = {}


def capture(name, path, tail=None):
    with path.open("rb") as stream:
        if tail:
            stream.seek(max(0, os.fstat(stream.fileno()).st_size - tail))
        raw = stream.read()
    files[name] = raw.decode("utf-8")
    return raw


state = json.loads(capture("state.json", ROOT / "state.json"))
seal = json.loads(capture("plan.identity.json", ROOT / "plan.identity.json"))
plan_raw = (ROOT / "plan.json").read_bytes()
plan = json.loads(plan_raw)
capture("controller.log", ROOT / "controller.log")
boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
all_processes = subprocess.check_output(["ps", "-u", str(os.getuid()), "-o", "pid=,ppid=,lstart=,args="], text=True)
controller_lines = []
for line in all_processes.splitlines():
    # Match only the controller command for the sealed plan, excluding this
    # sampler and unrelated user activity.
    if "production_control.py run --plan " + str(ROOT / "plan.json") in line:
        controller_lines.append(line.strip())
state_stage = next(s for s in STAGES if state["stages"].get(s, {}).get("status") != "COMPLETE")
if state_stage != "paired":
    raise RuntimeError(f"Unexpected phase {state_stage}")
active = state["attempts"][state["stages"][state_stage]["attempt_index"]]
runners = Path(plan["job_root"])
paired_job = runners / active["job_id"]
paired_out = Path(state["stages"]["paired"]["output"])
capture("paired.work.log", paired_job / "work.log", 8192)
capture("paired.supervisor.log", paired_job / "supervisor.log", 8192)
capture("paired.attempt.json", paired_job / "attempt.json")
capture("paired.accepted.json", paired_job / "accepted.json")
capture("paired.runner-status.json", paired_job / "status.json")

requests = {}
for attempt in state["attempts"]:
    stage = attempt["stage"]
    job = runners / attempt["job_id"]
    request = load(job / "request.json")
    accepted = load(job / "accepted.json")
    copy = load(Path(attempt["request_path"]))
    requests[stage] = {
        "request": select(request, "job_id argv cwd capture_version beauty_stage beauty_input_identity beauty_config_sha256 timeout_seconds"),
        "runner_request_canonical_sha256": digest(request),
        "controller_request_canonical_sha256": digest(copy),
        "runner_accepted_sha256": accepted["request_sha256"],
    }
    prior_out = Path(state["stages"][stage]["output"])
    argv_file = prior_out / "compute_argv.json"
    if argv_file.is_file():
        capture(f"{stage}-compute-argv.json", argv_file)
    if stage != "paired":
        capture(f"{stage}-stage-result.json", prior_out / "stage_result.json")
request_env_keys = sorted(load(paired_job / "request.json")["env"])
training = ROOT / "training"
selection = json.loads(capture("best-selection.json", training / "best_selection.json"))
adapter = training / "best_adapter"
adapter_info = identity(adapter / "g3_identity.json")
checkpoint_storage = {}
for step in (6160, 8214):
    cp_dir = training / f"checkpoint-{step}"
    path = cp_dir / "complete.json"
    files_in = load(path)["files"]
    members_present = [int((cp_dir / row["path"]).stat().st_size) == row["bytes"] for row in files_in]
    checkpoint_storage[str(step)] = {"manifest": identity(path), "members": len(files_in),
                                     "member_sizes_match": all(members_present),
                                     "manifest_total_bytes": sum(row["bytes"] for row in files_in)}
train = load(training / "rank0.json")
training_summary = select(train, "status run_commit config_sha256 global_step initial_global_step epoch data data_identity best_checkpoint best_metric metrics")
training_summary.update(best_selection=selection, best_adapter_model=identity(adapter / "adapter_model.bin", hash_file=False),
                        best_adapter_identity=adapter_info, checkpoint_storage=checkpoint_storage)
conditions = {}
for condition in ("normal", "s3", "s4"):
    gen = Path(state["stages"][condition]["output"])
    ev = Path(state["stages"][condition+"_eval"]["output"])
    gen_result = json.loads(files[f"{condition}-stage-result.json"])
    ev_result = json.loads(files[f"{condition}_eval-stage-result.json"])
    rank_path = gen / "test/rank0.json"
    rank = load(rank_path)
    pred = gen / "test/predictions.rank0.jsonl"
    metrics_path = ev / "primary/metrics.json"
    metrics = json.loads(capture(f"{condition}-offline-metrics.json", metrics_path))
    capture(f"{condition}-evaluation-config.json", ev / "evaluation_config.json")
    capture(f"{condition}-stage-a.json", ev / "stage_a.json")
    rank_record = select(rank, "status scope experiment_id execution_topology execution_profile condition world_size RUN_COMMIT generation completed_users assigned_users assignment_sha256 model_identity_sha256 tokenizer_identity adapter_files data_identity shuffle")
    indices = rank["assigned_row_indices"]
    reload = rank["strict_adapter_reload"]
    rank_record["strict_reload_summary"] = select(reload, "keys lora_keys all_active_embedding_head_and_sid_rows_checked no_missing_keys_tolerated")
    rank_record["strict_reload_embedding_and_head"] = {k: v["shape"] for k, v in reload["tensor_checks"].items() if k.endswith(("embed_tokens.weight", "lm_head.weight"))}
    rank_record["user_assignment_full_range_unique"] = len(indices) == EXPECTED_USERS and set(indices) == set(range(EXPECTED_USERS))
    if condition == "normal":
        checks = rank["normal_off_same_process_exact"]
        rank_record["normal_off"] = {"checks": len(checks),
                                     "all_exact": all(row["same_process_exact"] is True for row in checks),
                                     "full_unique_range": len(checks) == EXPECTED_USERS and {row["row_idx"] for row in checks} == set(range(EXPECTED_USERS)),
                                     "checks_canonical_sha256": digest(checks)}
    rank_record["rank_report"] = identity(rank_path)
    conditions[condition] = {"rank": rank_record, "prediction": identity(pred, hash_file=False),
                             "generation_receipt_prediction": gen_result["proof"]["prediction_files"][0],
                             "evaluation_receipt_prediction": ev_result["proof"]["prediction_proof"]["prediction_files"][0],
                             "evaluation_prediction_identity": metrics["prediction_files"][0],
                             "offline_metrics": identity(metrics_path),
                             "per_user_files": [identity(ev / "primary" / Path(row["path"]).name) for row in metrics["per_user_files"]]}
s3_pair = paired_out / "s3/paired_comparison.json"
s3_report = json.loads(capture("paired-s3-partial-comparison.json", s3_pair))
paired_files = sorted(paired_out.rglob("*"))
paired_result = {
    "output": str(paired_out),
    "entries": [{"relative_path": str(p.relative_to(paired_out)), "directory": p.is_dir(),
                 "file_identity": identity(p) if p.is_file() else None} for p in paired_files],
    "s3": select(s3_report, "scope status users bootstrap elapsed_seconds independent_verification"),
    "s4_directory_exists": (paired_out / "s4").is_dir(),
    "formal_paired_stage_result_exists": (paired_out / "stage_result.json").exists(),
    "collect_attempt_present": "collect" in state["stages"],
    "collect_output_parent_exists": (ROOT / "attempts/collect").exists(),
}
source_root = Path(plan["release"])
sources = ("production_control.py", "production_pipeline.py", "production_identity.py", "formal_train.py",
           "formal_evaluate.py", "offline_metrics.py", "paired_beauty.py", "shuffle_entry.py",
           "shuffle_runtime.py", "config/formal_g3_beauty_seed42.json")
source_files = {f"lc_rec/a0_beauty/{p}": identity(source_root / "lc_rec/a0_beauty" / p) for p in sources}
asset_files = {k: identity(Path(plan["data_root"]) / "Beauty" / f"Beauty.{k}.json") for k in plan["asset_sha256"]}
charged = {k: 0. for k in ("train_normal", "s3", "s4", "cpu")}
for attempt in state["attempts"]:
    stage = attempt["stage"]
    group = "train_normal" if stage in ("train", "normal") else stage if stage in ("s3", "s4") else "cpu"
    charged[group] += attempt["charged_seconds"]
charged["total"] = sum(charged.values())
remaining = {k: plan["budget"][k] - val for k, val in charged.items()}
effective_paired_retry = min(plan["stage_seconds"]["paired"] - active["charged_seconds"], remaining["cpu"],
                             remaining["total"]) - plan["budget"]["cleanup_reserve"]
disk = os.statvfs(ROOT)
mem = next(int(x.split()[1]) * 1024 for x in Path("/proc/meminfo").read_text().splitlines()
           if x.startswith("MemAvailable:"))
gpu = query(["nvidia-smi", "--id=1", "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"])
reported = {
    "collection_started_utc": started, "observed_at_utc": utc(), "hostname": os.uname().nodename,
    "plan": select(plan, "root release python task_id experiment_id experiment_series run_commit config_sha256 runner_sha256 execution_topology execution_profile asset_sha256 gpu_uuids budget stage_seconds"),
    "plan_raw_sha256": sha(plan_raw), "plan_canonical_sha256": digest(plan), "sealed_identity": seal,
    "state_mtime_utc": utc((ROOT/"state.json").stat().st_mtime), "controller_raw_status": state["status"],
    "controller_matching_argv_processes": controller_lines, "original_controller_pid_exists": Path("/proc/3334030").exists(),
    "boot_id": boot, "requests": requests, "source_files": source_files,
    "runner_file": identity(Path(plan["runner"])), "assets": asset_files, "training": training_summary,
    "condition_results": conditions, "paired": paired_result, "paired_request_env_keys": request_env_keys,
    "resources_at_observation": {"gpu1": gpu, "mem_available_bytes": mem,
                                 "disk_available_bytes": disk.f_frsize*disk.f_bavail},
    "budget": {"used": charged, "remaining": remaining, "paired_retry_usable_under_current_caps_seconds": effective_paired_retry},
}
print(json.dumps({"observation": reported, "files": files}, ensure_ascii=False, allow_nan=False))
