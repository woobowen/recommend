#!/usr/bin/env python3
"""Server-side submission and cumulative accounting over the existing CPU runner."""
import argparse
from datetime import datetime
import fcntl
import math
from pathlib import Path

from common import (TERMINAL, digest, facts, need, original_snapshot, pin,
                    query_original_pair, read, read_bound_release, runner, utc, write)

HERE = Path(__file__).resolve().parent


def job_name(cfg, kind, correction):
    return f"{cfg['runner_job_prefix']}-{kind}-r{correction}"


def elapsed(row):
    return (datetime.fromisoformat(row["ended_at"]) - datetime.fromisoformat(row["started_at"])).total_seconds()


def ledger(cfg, intents, original_charge):
    statuses = []
    root = Path(cfg["cpu_job_root"])
    files = sorted(intents.glob("*.json"))
    recorded = {f.stem for f in files}
    existing = {p.name for p in root.glob(f"{cfg['runner_job_prefix']}-*")}
    need(existing <= recorded, "unaccounted existing CPU job in the stable namespace")
    for intent in files:
        request = read(intent)
        job = request["job_id"]
        kind = request["closure_kind"]
        correction = request["closure_correction_round"]
        need(intent.stem == job == job_name(cfg, kind, correction) and
             request["closure_id"] == cfg["closure_id"] and request["original_plan_sha256"] == cfg["plan_raw_sha256"],
             f"conflicting ledger intent: {intent}")
        row = {"job_id": job, "kind": kind, "correction_round": correction, "request_sha256": digest(request),
               "timeout_seconds": request["timeout_seconds"], "request_path": str(intent)}
        if job not in existing or not (root / job / "request.json").is_file():
            row.update(status="RESERVED_NO_CONFIRMED_RUNNER_REQUEST", charged_seconds=None)
        else:
            status = runner(cfg["cpu_runner"], "query", root / job)
            need(status["request_sha256"] == row["request_sha256"], f"runner request mismatch: {job}")
            row.update(status=status["status"], runner=status, charged_seconds=None)
            if (status["status"] in TERMINAL and status.get("cleanup_verified")
                    and not any(status[k] for k in ("worker_alive", "supervisor_alive", "workgroup_live"))
                    and status.get("started_at") and status.get("ended_at")):
                row["charged_seconds"] = elapsed(status)
                need(row["charged_seconds"] >= 0, f"runner clock moved backwards: {job}")
        statuses.append(row)
    used = math.fsum(row["charged_seconds"] for row in statuses if row["charged_seconds"] is not None)
    reserved = sum(row["timeout_seconds"] for row in statuses if row["charged_seconds"] is None)
    caps = cfg["budget"]
    return {
        "closure_id": cfg["closure_id"], "observed_utc": utc(), "attempts": statuses,
        "original_worker_wall_seconds": original_charge, "closure_used_seconds": used,
        "closure_reserved_seconds_unsettled": reserved, "combined_used_seconds": original_charge + used,
        "remaining_closure_seconds": caps["cpu_worker_wall_max_seconds"] - used - reserved,
        "remaining_combined_seconds": caps["combined_max_seconds"] - original_charge - used - reserved,
        "cleanup_reserve_seconds": caps["cleanup_reserve_seconds"],
        "highest_correction_round": max((row["correction_round"] for row in statuses), default=0),
        "new_gpu_jobs": 0, "new_gpu_device_minutes": 0,
    }


def build_request(cfg, stamp, args, timeout, ledger_before):
    deployment = Path(cfg["cpu_runner"] + ".deployment.json")
    sidecar = read(deployment)
    job = job_name(cfg, args.kind, args.round)
    output = Path(cfg["cpu_job_root"]) / job / "output"
    member_paths = [HERE / name for name in stamp["files"]]
    member_paths += [HERE / "deployment.json", deployment, Path(cfg["cpu_runner"])]
    if args.kind == "review":
        completion = Path(args.completion).resolve(strict=True)
        success = next((row for row in ledger_before["attempts"] if row["kind"] == "compute"
                       and row["status"] == "SUCCEEDED" and row["charged_seconds"] is not None
                       and completion == Path(row["runner"]["result_path"])), None)
        need(success is not None, "review requires a cleared successful compute in this ledger")
        member_paths.extend((completion / "completion.json", completion / "manifest.json"))
    else:
        need(args.completion is None, "compute does not accept a prior output")
        completion = None
    request = {
        "schema": 1, "job_id": job, "host": sidecar["hostname"], "user": sidecar["user"], "uid": sidecar["uid"],
        "namespace": sidecar["namespace"], "deployment_sha256": cfg["cpu_deployment_sha256"],
        "capture_version": stamp["implementation_commit"], "cwd": str(HERE), "timeout_seconds": timeout,
        "argv": [cfg["python"], "-B", "-u", str(HERE / "closure.py"), "--kind", args.kind,
                 "--output", str(output)] + (["--completion", str(completion)] if completion else []),
        "env": {"CUDA_VISIBLE_DEVICES": "", "NVIDIA_VISIBLE_DEVICES": "none", "DS_ACCELERATOR": "cpu",
                "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": "", "PYTHONOPTIMIZE": "0",
                "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        "inputs": [facts(path) for path in member_paths],
        "closure_id": cfg["closure_id"], "closure_kind": args.kind, "closure_correction_round": args.round,
        "original_plan_sha256": cfg["plan_raw_sha256"], "original_run_commit": cfg["run_commit"],
        "original_paired_job": cfg["original_stages"]["paired"]["job_id"],
        "original_paired_request_sha256": cfg["original_stages"]["paired"]["request_sha256"],
        "closure_config_sha256": stamp["files"]["closure_config.json"]["sha256"],
        "budget_at_admission": {key: ledger_before[key] for key in (
            "original_worker_wall_seconds", "closure_used_seconds", "remaining_closure_seconds",
            "remaining_combined_seconds", "cleanup_reserve_seconds")},
    }
    return request


def main():
    import json
    import os
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    commands.add_parser("query")
    submit = commands.add_parser("submit")
    submit.add_argument("--kind", choices=("compute", "review"), required=True)
    submit.add_argument("--round", type=int, required=True)
    submit.add_argument("--completion")
    submit.add_argument("--reconcile-intent", action="store_true",
                        help="retry only the identical durable intent when the runner never recorded an accepted request")
    args = parser.parse_args()
    os.umask(0o077)
    cfg, stamp = read_bound_release(HERE)
    root = Path(cfg["closure_root"])
    pin(cfg["cpu_runner"], cfg["runner_sha256"])
    sidecar = pin(cfg["cpu_runner"] + ".deployment.json", cfg["cpu_deployment_sha256"])
    deployment = read(sidecar)
    need(deployment["namespace"] == cfg["runner_namespace"] and deployment["job_root"] == cfg["cpu_job_root"],
         "CPU runner sidecar scope mismatch")
    intents = root / "requests"
    intents.mkdir(exist_ok=True)
    with (root / ".budget.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        plan, _, charge = original_snapshot(cfg)
        before = ledger(cfg, intents, charge)
        write(root / "budget.json", before)
        if args.mode == "query":
            print(json.dumps(before, indent=2))
            return
        caps = cfg["budget"]
        need(0 <= args.round <= caps["maximum_correction_rounds"], "correction budget exhausted")
        job = job_name(cfg, args.kind, args.round)
        previous_attempt = next((row for row in before["attempts"] if row["job_id"] == job), None)
        if previous_attempt:
            if not args.reconcile_intent:
                print(json.dumps(before, indent=2))
                return
            runner_job = Path(cfg["cpu_job_root"]) / job
            need(previous_attempt["status"] == "RESERVED_NO_CONFIRMED_RUNNER_REQUEST"
                 and not any((runner_job / name).exists() for name in ("accepted.json", "status.json", "attempt.json")),
                 "reconciliation requires evidence that the existing intent was never accepted")
            path = Path(previous_attempt["request_path"])
            req = read(path)
            need(req["capture_version"] == stamp["implementation_commit"] and req["cwd"] == str(HERE),
                 "reconcile the original source release and same intent")
            query_original_pair(cfg, plan)
            accepted = runner(cfg["cpu_runner"], "submit", cfg["cpu_job_root"], path)
            need(accepted["request_sha256"] == digest(req), "reconciled request identity changed")
            after = ledger(cfg, intents, charge)
            write(root / "budget.json", after)
            print(json.dumps(after, indent=2))
            return
        need(not args.reconcile_intent, "there is no previously recorded intent to reconcile")
        need(args.round >= before["highest_correction_round"], "CPU closure correction rounds cannot decrease")
        need(before["closure_reserved_seconds_unsettled"] == 0,
             "prior CPU attempt is active or lacks cleared terminal evidence")
        usable = min(before["remaining_closure_seconds"], before["remaining_combined_seconds"])
        timeout = min(caps["preferred_attempt_timeout_seconds"][args.kind],
                      math.floor(usable-caps["cleanup_reserve_seconds"]))
        need(timeout > 0, "no CPU budget after cleanup reserve")
        query_original_pair(cfg, plan)
        req = build_request(cfg, stamp, args, timeout, before)
        path = intents / f"{job}.json"
        write(path, req, exclusive=True)
        accepted = runner(cfg["cpu_runner"], "submit", cfg["cpu_job_root"], path)
        need(accepted["request_sha256"] == digest(req), "submitted request identity changed")
        after = ledger(cfg, intents, charge)
        write(root / "budget.json", after)
        print(json.dumps(after, indent=2))


if __name__ == "__main__":
    main()
