"""Validate captured bytes and verify progress for the original a0 attempt."""
import argparse
import base64
import datetime as dt
import hashlib
import json
from pathlib import Path


def check(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def captured(observation):
    files = {}
    for name, row in observation["files"].items():
        if not row["exists"]:
            continue
        data = base64.b64decode(row["content_base64"], validate=True)
        check(len(data) == row["copied_bytes"] and sha(data) == row["sha256"],
              "Captured bytes mismatch: " + name)
        check(row["offset_bytes"] + len(data) == row["source_size_at_open"],
              "Byte range does not match captured file size: " + name)
        files[name] = data
    return files


def process_for(value, filename):
    return next((entry for entry in value["processes"].values()
                 if entry and len(entry["argv"]) > 2
                 and entry["argv"][1] == "-u" and Path(entry["argv"][2]).name == filename), None)


def events_from(files):
    data = files.get("events.rank0.jsonl", b"")
    return [json.loads(line) for line in data.split(b"\n")[:-1]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--complete", action="store_true")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    receipt_bytes = (here / "authorization.json").read_bytes()
    receipt = json.loads(receipt_bytes)
    check(sha(receipt_bytes) == (here / "authorization.sha256").read_text().split()[0],
          "Fixed authorization file mismatch")
    observations = sorted((here / "observations").glob("*.json"))
    check(bool(observations), "No live evidence")
    expected_job = "letter-a0-beauty-formal-g3-beauty-seed42-train-a0"
    progress = []
    for path in observations:
        value = json.loads(path.read_bytes())
        files = captured(value)
        runner = json.loads(value["runner_query"]["stdout"])
        state = json.loads(files["state.json"])
        accepted = json.loads(files["runner-accepted.json"])
        request_bytes = files["runner-request.json"]
        request = json.loads(request_bytes)
        digest = sha(canonical(request))
        seal = json.loads(files["plan.identity.json"])
        check(seal["sha256"] == receipt["plan_sha256"], "Sealed plan mismatch")
        check(request_bytes == files["controller-request.json"], "Runner request copy mismatch")
        check(digest == accepted["request_sha256"] == runner["request_sha256"],
              "Request canonical identity mismatch")
        last_attempt = state["attempts"][-1]
        check(last_attempt["request_sha256"] == digest and last_attempt["stage"] == "train"
              and last_attempt["job_id"] == runner["job_id"] == request["job_id"] == expected_job
              and last_attempt["attempt"] == 0, "Attempt mismatch")
        check(request["capture_version"] == receipt["run_commit"]
              and request["beauty_config_sha256"] == receipt["config_sha256"]
              and request["beauty_input_identity"] == receipt["asset_sha256"]
              and request["env"]["CUDA_VISIBLE_DEVICES"] == "1"
              and request["timeout_seconds"] == 107880, "Formal request identity mismatch")
        check(request["argv"][-1] == sha(receipt_bytes)
              and any(row["sha256"] == sha(receipt_bytes) and row["path"] == request["argv"][-3]
                      for row in request["inputs"]), "Request authorization mismatch")
        check(runner["status"] == "RUNNING" and runner["exit_code"] is None
              and runner["supervisor_alive"] and runner["worker_alive"]
              and runner["workgroup_live"], "Original runner did not confirm live a0")
        control = value["processes"][str(value["expected_controller_pid"])]
        check(control["fds"]["0"] == "/dev/null"
              and control["fds"]["1"].endswith("/controller.log")
              and control["session_id"] == control["pid"] and control["tty_nr"] == 0,
              "Controller attachment or identity mismatch")
        for key in ("worker", "supervisor"):
            saved = runner[key]
            live = value["processes"][str(saved["pid"])]
            check(all(live[field] == saved[field] for field in
                      ("pid", "start_ticks", "boot_id", "uid", "argv"))
                  and live["fds"]["0"] == "/dev/null" and live["tty_nr"] == 0,
                  "Live identity mismatch: " + key)
        if "server-authorization.json" in files:
            check(files["server-authorization.json"] == receipt_bytes, "Server receipt bytes mismatch")
        if "experiment_registry.json" in files:
            check(json.loads(files["experiment_registry.json"])["plan_sha256"] == seal["sha256"],
                  "Registry does not bind the sealed plan")
        rank = process_for(value, "formal_train.py")
        if rank:
            compute = json.loads(files["compute_argv.json"])
            argv = compute["argv"]
            check(compute["scope"] == "FORMAL" and "--nproc_per_node=1" in argv
                  and not any(arg in argv for arg in ("--resume", "--probe-stop", "--boundary-preflight",
                                                     "--diagnosis")),
                  "Unexpected compute scope or invocation")
            entry = next(i for i, arg in enumerate(argv) if Path(arg).name == "formal_train.py")
            check(any(p and p["argv"] == argv for p in value["processes"].values()),
                  "Actual torchrun invocation does not match saved argv")
            check(rank["argv"][2:] == argv[entry:] and rank["argv"][0] == argv[0]
                  and rank["env"]["CUDA_VISIBLE_DEVICES"] == "1"
                  and rank["env"]["WORLD_SIZE"] == "1" and rank["env"]["LOCAL_RANK"] == "0"
                  and rank["env"]["GRID_JOB_ID"] == expected_job,
                  "Actual compute rank does not match saved argv or environment")
        events = events_from(files)
        check(all(row["attempt_job_id"] == expected_job for row in events),
              "Events from wrong attempt")
        updates = [row for row in events if row["event"] == "optimizer_step"]
        logs = [row for row in events if row["event"] == "log" and "loss" in row["values"]]
        progress.append({
            "file": path.name, "utc": value["collection_started_utc"],
            "rank_pid": rank["pid"] if rank else None,
            "optimizer_step": updates[-1] if updates else None,
            "loss_log": logs[-1] if logs else None,
            "gpu": value["gpu1"], "budget": value["charged_seconds"],
        })
    if args.complete:
        check(rank is not None, "Actual formal_train worker missing from final snapshot")
        end = dt.datetime.fromisoformat(value["collection_started_utc"])
        start = dt.datetime.fromisoformat(rank["started_at_utc_from_proc"])
        elapsed = (end - start).total_seconds()
        check(elapsed >= 1801, "Window missing 30 minutes after rank startup (plus btime precision)")
        considered = [entry["optimizer_step"]["step"] for entry in progress
                      if entry["utc"] >= (start + dt.timedelta(minutes=9)).isoformat()
                      and entry["optimizer_step"]]
        check(len(considered) >= 3 and all(a < b for a, b in zip(considered, considered[1:])),
              "Insufficient progressing observations")
        check(any(row["event"] == "train_begin" and row["max_steps"] == 8214
                  and row["global_step"] == 0 for row in events), "Fresh 8,214-step train_begin missing")
        check(any(row.get("stage") == "after_first_optimizer_update_this_process" for row in events),
              "First actual optimizer update evidence missing")
        report = json.loads(files["rank0.json"])
        check(report["scope"] == "FORMAL" and not report["probe"] and
              report["run_commit"] == receipt["run_commit"] and
              report["config_sha256"] == receipt["config_sha256"] and
              report["execution_profile"] == receipt["execution_profile"] and
              report["optimizer_membership_exact"] and report["optimizer_parameter_count"] == 288571392
              and sum(p["requires_grad"] for p in report["parameter_inventory"].values()) == 450,
              "Actual optimizer/profile evidence mismatch")
        print(json.dumps({"window_start_utc": start.isoformat(), "window_end_utc": end.isoformat(),
                          "elapsed_seconds": elapsed, "samples": progress}, indent=2))
    else:
        print(json.dumps(progress, indent=2))


if __name__ == "__main__":
    main()
