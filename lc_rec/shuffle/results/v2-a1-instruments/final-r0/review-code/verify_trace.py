#!/usr/bin/env python3
"""Verify immutable formal requests, cleanup, timing and predecessor bindings."""
import argparse
from datetime import datetime
from hashlib import file_digest, sha256
import json
from pathlib import Path
import re


PREFIX = "a1-instruments-v2-"
MODES = ("shuffle2", "shuffle3a", "shuffle3b", "shuffle4")
STAGES = tuple((lane, mode) for mode in MODES for lane in ("gpu", "cpu")) + (("paired", "paired"),)
RUN_ROOT = Path("/data/bwwu/lc-rec/a1-instruments-v2")
PENDING = "AWAITING_WEB_REVIEW_AND_SEPARATE_USER_AUTHORIZATION"
ERROR = re.compile(r"Traceback \(most recent|\b(?:Runtime|Assertion|Value|Timeout|Import|OSError|Memory)Error\b|"
                   r"CUDA out of memory|ChildFailedError|I/O error|No space left|non-finite|NCCL error", re.I)
PLAN_SHA = "f37fbe7ac295480a181daf2fed098810bd5b2d3c9499e19a7eec67d160603c7b"
AUTH_SHA = "d4c86662ccfbd46fe59bb03d1945ab8862783b2ecd3b955f4d100b6eadc36a94"


def need(condition, label):
    if not condition:
        raise ValueError(label)


def read(path):
    return json.loads(path.read_bytes())


def file_sha(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def digest(value):
    canonical = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return sha256(canonical).hexdigest()


def time(iso):
    return datetime.fromisoformat(iso)


def output(request):
    argv = request["argv"]
    return Path(argv[argv.index("--output") + 1])


def exact_empty(observation):
    return (observation["scan_complete"] and observation["verified_empty"] and not observation["boot_changed"]
            and not any(observation[key] for key in ("owned_live", "unconfirmed_live", "unsupported_live", "scan_errors")))


def validate_job(local, live_query, expected_request, plan):
    name = expected_request["job_id"]
    request, status, accepted, attempt, cleanup = (
        read(local / (member + ".json")) for member in ("request", "status", "accepted", "attempt", "cleanup"))
    request_sha = digest(expected_request)
    need(request == expected_request, f"{name} mutable request template or wrong dependency bytes")
    need(accepted["request_sha256"] == attempt["request_sha256"] == live_query["request_sha256"] == request_sha,
         f"{name} accepted request fingerprint")
    need(accepted["runner_sha256"] == plan["runner_sha256"] and accepted["runner_path"] == plan["runner"],
         f"{name} accepted runner identity")
    need(status == {key: live_query[key] for key in status}, f"{name} terminal persisted and fresh query states differ")
    need(status["status"] == "SUCCEEDED" and status["exit_code"] == 0 and status["cleanup"] == cleanup and
         status["worker_may_be_live"] is False, f"{name} terminal outcome")
    need(live_query["cleanup_verified"] and not any(live_query[key] for key in ("worker_alive", "supervisor_alive", "workgroup_live")),
         f"{name} remains active or uncleared")
    need(exact_empty(live_query["workgroup_observation"]) and
         cleanup["verified"] and cleanup["reason"] == "NORMAL_EXIT_CHECK" and not cleanup["events"] and
         exact_empty(cleanup["initial_observation"]) and exact_empty(cleanup["last_observation"]),
         f"{name} workgroup cleanup receipts")
    need(status["worker"]["argv"] == request["argv"] and status["worker_group"]["job_id"] == name,
         f"{name} captured worker argv/ownership")
    begin, end = time(status["started_at"]), time(status["ended_at"])
    need(time(accepted["accepted_at"]) <= time(attempt["claimed_at"]) <= begin <
         time(cleanup["started_at"]) <= time(cleanup["ended_at"]) <= end <= time(live_query["workgroup_observation"]["observed_at"]),
         f"{name} checkout timestamp ordering")
    seconds = (end-begin).total_seconds()
    need(0 < seconds < request["timeout_seconds"], f"{name} runner time bound")
    for logname in ("work.log", "supervisor.log"):
        path = local / logname
        need(not ERROR.search(path.read_text()), f"{name} {logname} contains an error")
    return {"job_id": name, "request_sha256": request_sha, "accepted_at": accepted["accepted_at"],
            "started_at": status["started_at"], "ended_at": status["ended_at"], "worker_elapsed_seconds": seconds,
            "configured_timeout_seconds": request["timeout_seconds"], "cleanup_verified": True}


def gpu_output(evidence, step, request, plan):
    name = request["job_id"]
    receipt = read(evidence / "summary.json")
    mode = step["mode"]
    rank = read(evidence / "test/rank0.json")
    launch = read(evidence / f"{mode}.request.json")
    exit_receipt = read(evidence / f"{mode}.exit.json")
    resources = [json.loads(line) for line in (evidence / f"{mode}.resources.jsonl").read_text().splitlines()]
    device = launch["device"]
    need(receipt["status"] == "PASS" and receipt["job_id"] == name and receipt["run_commit"] == request["capture_version"],
         f"{name} GPU stage result identity")
    need(device == receipt["device"] and device["uuid"] == plan["card"] and device["physical_index"] == 1 and
         device["visible_device"] == 0 and device["world_size"] == 1 and not device["has_compute_process"],
         f"{name} single-device checkout")
    need(launch["run_commit"] == request["capture_version"] == rank["RUN_COMMIT"] and "--preflight" not in launch["argv"],
         f"{name} actual GPU launch identity")
    need((rank["status"], rank["rank"], rank["local_rank"], rank["world_size"], rank["assigned_users"],
          rank["completed_users"], rank["preflight_only"], rank["gpu"]["visible_devices"]) ==
         ("PASS", 0, 0, 1, plan["users_per_mode"], plan["users_per_mode"], False, plan["card"]), f"{name} rank coverage")
    worker = receipt["stages"][mode]
    need(exit_receipt["exit_code"] == worker["exit_code"] == 0 and worker["seconds"] == exit_receipt["seconds"] and
         worker["users"] == plan["users_per_mode"], f"{name} GPU worker exit and full coverage")
    need(len(resources) >= 2 and all(a["seconds"] < b["seconds"] for a,b in zip(resources, resources[1:])),
         f"{name} resource monitor timeline")
    device_stats = [r["gpu"].split(", ") for r in resources]
    need(all(row[0] == plan["card"] and int(row[1]) >= 0 and 0 <= int(row[2]) <= 100 for row in device_stats),
         f"{name} GPU usage attribution")
    disk_min = min(r["disk_free_bytes"] for r in resources)
    ram_min = min(r["host_mem_available_bytes"] for r in resources)
    need(disk_min >= 20*1024**3 and ram_min >= 40*1024**3, f"{name} resource safety bounds")
    progress, notes = [], []
    for line in (evidence / f"{mode}.log").read_text().splitlines():
        need(not ERROR.search(line), f"{name} generator log contains an error")
        if line.startswith("{"):
            progress.append(json.loads(line))
        elif line.strip():
            notes.append(line)
    expected = list(range(25, plan["users_per_mode"], 25))
    need([p["completed_users"] for p in progress] == expected and
         all((p["stage"], p["rank"], p["assigned_users"]) == ("test", 0, plan["users_per_mode"]) for p in progress),
         f"{name} complete ordered generator log history")
    need(all(a["elapsed_seconds"] < b["elapsed_seconds"] for a,b in zip(progress, progress[1:])),
         f"{name} progress log timeline")
    return {"rank_completed_users": rank["completed_users"], "model_load_seconds": worker["model_load_seconds"],
            "generation_seconds": worker["generation_seconds"], "resource_log_points": len(resources),
            "progress_checkpoints": len(progress), "minimum_available_disk_bytes": disk_min,
            "minimum_available_cpu_ram_bytes": ram_min, "monitor_memory_used_mib_range": [
                min(int(row[1]) for row in device_stats), max(int(row[1]) for row in device_stats)],
            "startup_log_notes": notes}


def verify(raw):
    control, evidence, jobs = (raw / folder for folder in ("control", "evidence", "jobs"))
    plan_path, auth_path = control / "formal-plan.json", control / "authorization.json"
    plan, auth = read(plan_path), read(auth_path)
    need(file_sha(plan_path) == PLAN_SHA and file_sha(auth_path) == AUTH_SHA, "Pinned plan/authorization SHA-256")
    need(auth["plan_sha256"] == PLAN_SHA and auth["experiment"] == plan["experiment"] and
         auth["gpu_commit"] == plan["preflight_gpu_commit"] and auth["controller_commit"] == plan["evaluation_and_controller_commit"] and
         auth["user_formal_launch_authorized"] and auth["web_review_pass"] and auth["attempt"] == plan["attempt"],
         "Formal authorization semantics")
    need(tuple((p["kind"], p["mode"]) for p in plan["phases"]) == STAGES, "Reviewed nine-phase dependency plan")
    releases = read(control / "releases.json")
    for version, expected in plan["release_inputs"].items():
        observed = releases["releases"][version]
        need([{"path": row["path"], "sha256": row["sha256"]} for row in observed] == expected,
             f"{version} server fixed source readback")
    need(releases["host"] == plan["parent"]["host"] and releases["runner_sha256"] == plan["runner_sha256"] and
         releases["runner_deployment_sha256"] == plan["parent"]["deployment_sha256"], "Runner export and host")
    inventory = read(control / "inventory.json")
    jobs_fresh = read(control / "job-queries.json")
    expected_templates = [plan["parent"], *[p["request"] for p in plan["phases"]]]
    identifiers = {template["job_id"] for template in expected_templates}
    need(identifiers == {name for name in jobs_fresh if name.startswith(PREFIX + "formal-")} ==
         {Path(name).name for name in inventory["jobs"]}, "Exact formal job population")
    need(set(inventory["outputs"]) == {str(output(template)) for template in expected_templates}, "Exact formal output tree")
    need(set(inventory["request_paths"]) == {plan["server_plan_path"], plan["approval_path"], *(
        str(Path(plan["requests_root"]) / (job + ".json")) for job in identifiers)}, "Exact formal request inventory")
    tagged = evidence / "formal-parent-r0/summary.json"
    parent_report = read(tagged)
    need(parent_report["status"] == "PASS_A1_INSTRUMENTS_V2_FORMAL" and parent_report["used_gpu_device_seconds_before"] == 0,
         "Formal controller success and prior GPU usage")
    observed = parent_report["phases"]
    need(len(observed) == len(STAGES), "Controller nine steps persisted")
    auth_input = {"path": plan["approval_path"], "sha256": AUTH_SHA}
    template = plan["parent"]
    request = {**template, "argv": [AUTH_SHA if value == PENDING else value for value in template["argv"]],
               "inputs": plan["release_inputs"][template["capture_version"]] + template["inputs"] +
               [{"path":plan["server_plan_path"], "sha256":PLAN_SHA}, auth_input]}
    parent = validate_job(jobs / template["job_id"].removeprefix(PREFIX), jobs_fresh[template["job_id"]], request, plan)
    need(time(auth["recorded_at_utc"]) < time(parent["accepted_at"]), "Authorization predates acceptance")
    timings = []
    previous_id, previous_done = None, time(parent["started_at"])
    sha_cache = {}

    def bound_asset(path):
        local = evidence / Path(path).relative_to(RUN_ROOT)
        if local not in sha_cache:
            sha_cache[local] = file_sha(local)
        return sha_cache[local]

    for step, trail in zip(plan["phases"], observed, strict=True):
        template = step["request"]
        name = template["job_id"]
        need(step["depends_on"] == previous_id and trail["job_id"] == name and
             trail["request"] == str(Path(plan["requests_root"]) / (name + ".json")) and trail["status"] == "PASS",
             f"{name} recorded dependency chain")
        need(all(bound_asset(asset["path"]) == asset["sha256"] for asset in template["inputs"]),
             f"{name} base CPU replay or frozen matrix actual bytes")
        requested = [{ "path": path, "sha256": bound_asset(path)} for path in step["predecessor_members"]]
        request = {**template, "inputs": [*plan["release_inputs"][template["capture_version"]], *template["inputs"],
                                          auth_input, *requested]}
        timing = validate_job(jobs / name.removeprefix(PREFIX), jobs_fresh[name], request, plan)
        audit_keys = ("job_id", "status", "started_at", "ended_at", "request_sha256", "exit_code", "cleanup_verified")
        need(all(trail["runner_state"][key] == jobs_fresh[name][key] for key in audit_keys),
             f"{name} controller terminal query")
        need(time(timing["accepted_at"]) >= previous_done and time(timing["ended_at"]) <= time(parent["ended_at"]),
             f"{name} serial predecessor actually completed before submission")
        previous_id, previous_done = name, time(timing["ended_at"])
        stage_evidence = evidence / output(template).relative_to(RUN_ROOT)
        if step["kind"] == "gpu":
            need(request["env"]["CUDA_VISIBLE_DEVICES"] == plan["card"], f"{name} GPU request visibility")
            timing.update(gpu_output(stage_evidence, step, request, plan))
        else:
            need(request["env"]["CUDA_VISIBLE_DEVICES"] == "" and request["env"]["DS_ACCELERATOR"] == "cpu",
                 f"{name} CPU-only request visibility")
            receipt = read(stage_evidence / "summary.json")
            need(receipt["preflight"] is False and receipt["users"] == plan["users_per_mode"],
                 f"{name} full CPU coverage")
            if step["kind"] == "cpu":
                need(receipt["status"] == "PASS_V2_EVALUATION" and receipt["condition"] == step["mode"] and
                     receipt["candidate_anomalies"]["candidate_count"] == plan["candidates_per_mode"], f"{name} CPU evaluation")
            else:
                need(receipt["status"] == "PASS_V2_PAIRED" and sum(map(len, receipt["comparisons"].values())) == 80,
                     f"{name} paired evaluation")
        timings.append({"kind":step["kind"], "mode":step["mode"], **timing})
    gpu_time = sum(s["worker_elapsed_seconds"] for s in timings if s["kind"] == "gpu")
    cpu_time = sum(s["worker_elapsed_seconds"] for s in timings if s["kind"] != "gpu")
    limits = plan["future_budget"]
    need(gpu_time <= limits["gpu_device_seconds"] == 36*3600 and
         limits["gpu_seconds_per_mode"] == 8*3600 and limits["gpu_seconds_reserved_this_plan"] == 32*3600,
         "Whole-plan GPU device-hour budget")
    used_preflight_gpu = used_preflight_cpu = 0
    for name, status in jobs_fresh.items():
        if name in identifiers:
            continue
        if name == plan["timeout_fixture"]:
            need(status["status"] == "TIMED_OUT" and status["exit_code"] == -15 and status["cleanup_verified"],
                 "Deliberate preflight timeout cleanup")
        else:
            need(status["status"] == "SUCCEEDED" and status["exit_code"] == 0 and status["cleanup_verified"],
                 f"Preflight dependency terminal state {name}")
        seconds = (time(status["ended_at"]) - time(status["started_at"])).total_seconds()
        if "-gpu-preflight-" in name:
            used_preflight_gpu += seconds
        else:
            used_preflight_cpu += seconds
    frozen = read(control / "frozen-normal-readback.json")
    for name, row in frozen["members"].items():
        path = raw / name
        need(path.stat().st_size == row["bytes"] and file_sha(path) == row["sha256"], f"{name} frozen Normal readback")
    return {"status":"PASS_NON_AUTHOR_B_FORMAL_EXECUTION_TRACE", "plan_sha256":PLAN_SHA, "authorization_sha256":AUTH_SHA,
            "observed_utc":inventory["observed_utc"], "parent":parent, "phases":timings,
            "formal_gpu_device_seconds_runner_bound":gpu_time, "formal_cpu_worker_wall_seconds":cpu_time,
            "engineering_gpu_device_seconds_runner_bound":used_preflight_gpu, "engineering_cpu_worker_wall_seconds":used_preflight_cpu,
            "formal_gpu_device_seconds_remaining":limits["gpu_device_seconds"]-gpu_time, "future_budget":limits,
            "release_files":{key:len(value) for key,value in releases["releases"].items()},
            "frozen_normal_members":len(frozen["members"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    need(not args.output.exists(), "Preserve earlier audit receipt")
    result = verify(args.raw)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key:result[key] for key in ("status", "observed_utc", "formal_gpu_device_seconds_runner_bound",
                                                "formal_cpu_worker_wall_seconds")} ))
