"""Bounded continuation of the original GRID dependency queue.

Only scheduling wrappers change. Every scientific stage request, original job,
asset, budget and absolute queue deadline remains bound to its original source.
This module sends no signals and never retries a terminal or ambiguous job.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import time

# Scientific requests deliberately retain the original C2 PYTHONPATH/cwd. Load
# the controller's own queue by absolute path instead of accidentally importing
# C2 through that inherited environment.
_CONTROLLER_ROOT = Path(__file__).resolve().parents[2]
_QUEUE_SPEC = importlib.util.spec_from_file_location(
    "grid_shared_immutable_queue", _CONTROLLER_ROOT / "experiments/launch_20261017/queue.py")
queue = importlib.util.module_from_spec(_QUEUE_SPEC)
_QUEUE_SPEC.loader.exec_module(queue)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _date(value):
    result = dt.datetime.fromisoformat(value)
    _require(result.utcoffset() is not None, "queue timestamps must include timezone")
    return result


def _file(path, digest):
    path = Path(path)
    _require(path.is_absolute() and path.is_file() and not path.is_symlink(),
             "missing or unsafe continuation source: " + str(path))
    _require(queue.sha(path) == digest, "continuation evidence changed: " + str(path))
    return queue.read(path)


def _canonical_sha(value):
    return hashlib.sha256(queue.canonical(value)).hexdigest()


def _old_module(old, old_request):
    root = Path(old["snapshot"]["root"])
    source = root / "experiments/launch_20261017/queue.py"
    deployment = root / "DEPLOYMENT.json"
    bound = {item["path"]: item["sha256"] for item in old_request["inputs"]}
    _require(str(deployment) in bound, "original deployment is not request-bound")
    manifest = _file(deployment, bound[str(deployment)])
    _require(manifest["source_commit"] == old["snapshot"]["commit"],
             "original source commit mismatch")
    entries = {item["path"]: item for item in manifest["files"]}
    relative = source.relative_to(root).as_posix()
    _require(relative in entries and queue.sha(source) == entries[relative]["sha256"],
             "original queue implementation differs from deployment")
    spec = importlib.util.spec_from_file_location("grid_original_immutable_queue", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bind(config, config_path, *, live=True):
    """Validate immutable lineage and return inherited wrappers/stage states.

    ``live`` is required before a new controller starts. Stage workers only
    recheck frozen evidence and their own immutable request, avoiding repeated
    all-job queries inside GPU wall budgets.
    """
    continuation = config["continuation"]
    old_path = Path(continuation["old_config_path"])
    old = _file(old_path, continuation["old_config_sha256"])
    old_state_path = Path(continuation["old_queue_state_path"])
    _require(old_state_path == Path(old["state_dir"]) / "queue-status.json",
             "old state path is not the original queue state")
    old_state = _file(old_state_path, continuation["old_queue_state_sha256"])
    old_job = continuation["old_queue_job_id"]
    _require(old_job == old["queue_id"] == old_state["queue_id"], "old queue identity mismatch")
    request_path = Path(continuation["old_queue_request_path"])
    _require(request_path == Path(old["runner"]["job_root"]) / old_job / "request.json",
             "old outer request path mismatch")
    old_request = _file(request_path, continuation["old_queue_request_sha256"])
    _require(old_request["job_id"] == old_job, "original queue request ID mismatch")
    original_inputs = {item["path"]: item["sha256"] for item in old_request["inputs"]}
    _require(original_inputs.get(str(old_path)) == continuation["old_config_sha256"],
             "old queue request did not bind this exact original configuration")
    _require(old_state["config_sha256"] == continuation["old_config_sha256"],
             "old state/config identity mismatch")
    _require(old_state["status"] in {"STOPPED_NO_NEW_SUBMISSIONS", "TERMINAL_PENDING_RESULT_REVIEW"},
             "old controller has not stopped future submission")
    _require(all(v["status"] != "SUBMITTING" for v in old_state["stages"].values()),
             "unresolved SUBMITTING intent: inspect original ID before continuation")
    deadline = _date(old_state["started_at"]) + dt.timedelta(seconds=old["max_queue_seconds"])
    _require(_date(continuation["original_absolute_queue_deadline"]) == deadline,
             "original absolute queue deadline was changed")
    _require(config["queue_id"] != old_job and config["state_dir"] != old["state_dir"],
             "continuation needs independent controller and state identities")
    _require(0 < config["max_queue_seconds"] <= old["max_queue_seconds"],
             "continuation cannot extend original queue horizon")
    for key in ["snapshot", "runner", "namespace_identity", "gpu_locks_dir", "gpu_budget_seconds",
                "budget_groups", "prior_gpu_request_usage", "max_storage_bytes", "free_space",
                "blocked_workflows"]:
        _require(config.get(key) == old.get(key), "continuation changed original bound setting: " + key)
    _require(set(old["storage_roots"]) <= set(config["storage_roots"]),
             "continuation cannot omit original storage roots")
    approved_gpus = {device for stage in old["stages"] for device in queue.gpus(stage)}
    _require(set(config["approved_gpu_uuids"]) == approved_gpus
             and len(config["approved_gpu_uuids"]) == len(approved_gpus),
             "continuation physical GPU approval differs from original devices")
    controller = config["controller_snapshot"]
    _require(Path(controller["root"]).is_absolute() and
             re.fullmatch(r"[0-9a-f]{40}", controller["commit"]) is not None,
             "new controller snapshot identity is missing")
    _require(Path(controller["root"]) in Path(__file__).resolve().parents,
             "controller is not executing from its bound snapshot")

    old_stages = {s["id"]: s for s in old["stages"]}
    new_stages = {s["id"]: s for s in config["stages"]}
    _require(len(new_stages) == len(config["stages"]) and old_stages.keys() == new_stages.keys(),
             "continuation must preserve every original stage exactly once")
    for name, stage in new_stages.items():
        def science(value):
            return {k: v for k, v in value.items()
                    if k not in {"resource_policy", "priority", "gpu_candidates"}}
        _require(science(stage) == science(old_stages[name]),
                 "scientific request/dependency/gate changed: " + name)
    _require(set(old_state["stages"]) == set(old_stages), "old state stage set mismatch")
    submitted = continuation["submitted_stage_ids"]
    _require(len(submitted) == len(set(submitted)) and set(submitted) <= set(old_stages),
             "invalid original submitted-stage inventory")
    submitted = set(submitted)
    legacy = _old_module(old, old_request)
    expected_requests = {name: legacy.wrapped_request(old, old_path, old_stages[name])
                         for name in submitted}
    seed = {}
    with (Path(old["state_dir"]) / "supervisor.lock").open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _require(queue.sha(old_state_path) == continuation["old_queue_state_sha256"],
                 "old queue state changed while taking its lock")
        if live:
            outer = queue.runner_call(old, "query", request_path.parent)
            _require(outer["request_sha256"] == _canonical_sha(old_request),
                     "old runner request identity changed")
            _require(outer["status"] not in queue.ACTIVE | {"CONTROL_LOST", "UNCONFIRMED"}
                     and outer.get("cleanup_verified") is True
                     and not any(outer.get(k) for k in ["supervisor_alive", "worker_alive", "workgroup_live",
                                                       "worker_may_be_live"]),
                     "old outer supervisor/worker is not verified terminal and empty")
        for name, stage in old_stages.items():
            job = Path(old["runner"]["job_root"]) / stage["request"]["job_id"]
            item = copy.deepcopy(old_state["stages"][name])
            _require(item["job_id"] == stage["request"]["job_id"], "old stage/job identity mismatch")
            if name in submitted:
                _require((job / "accepted.json").is_file(), "inherited accepted receipt missing: " + name)
                expected = expected_requests[name]
                _require(queue.read(job / "request.json") == expected,
                         "inherited request differs from original C2 wrapper: " + name)
                digest = _canonical_sha(expected)
                _require(queue.read(job / "accepted.json")["request_sha256"] == digest,
                         "inherited acceptance identity mismatch: " + name)
                if live:
                    actual = queue.runner_call(old, "query", job)
                    _require(actual["request_sha256"] == digest, "inherited runner identity mismatch: " + name)
                    item["runner_status"] = actual
                    if actual["status"] == "SUCCEEDED" and actual.get("exit_code") == 0 and actual.get("cleanup_verified"):
                        item["status"] = "SUCCEEDED"
                    elif actual["status"] in queue.ACTIVE:
                        item["status"] = actual["status"]
                    else:
                        item.update(status="BLOCKED_NO_RETRY", reason="inherited original terminal/unconfirmed: " + actual["status"])
            else:
                if live:
                    _require(not job.exists(),
                             "unlisted original job or unresolved request identity: " + name)
                _require(not item.get("runner_status"), "unsubmitted stage contains unexplained runner status")
                without_execution = ((queue.FINAL_QUEUE - {"SUCCEEDED"}) |
                    {"QUEUED", "WAITING_DEPENDENCY", "WAITING_GPU", "WAITING_GPU_LOCK", "WAITING_DISK"})
                _require(item["status"] in without_execution,
                         "unsubmitted stage carries unexplained execution status: " + name)
                if item["status"] not in queue.FINAL_QUEUE:
                    item = dict(job_id=stage["request"]["job_id"], status="QUEUED")
            seed[name] = item
    return dict(old_config=old, old_state=old_state, inherited_requests=expected_requests,
                stage_states=seed, original_absolute_queue_deadline=deadline.isoformat())


def install(config, config_path, *, live=True):
    """Install three explicit scheduler hooks, keeping the finite queue loop."""
    binding = bind(config, config_path, live=live)
    base_wrapped = queue.wrapped_request
    base_accounting = queue.accounting

    def wrapped(current, path, stage):
        if stage["id"] in binding["inherited_requests"]:
            return copy.deepcopy(binding["inherited_requests"][stage["id"]])
        result = base_wrapped(current, path, stage)
        script = str(Path(__file__).resolve())
        result["argv"] = [result["argv"][0], "-u", script, "run-stage", "--config",
                          str(Path(path).resolve()), "--stage", stage["id"]]
        inputs = {item["path"]: item for item in result["inputs"]}
        inputs[script] = dict(path=script, sha256=queue.sha(script))
        result["inputs"] = list(inputs.values())
        result["queue_stage"]["controller_source_commit"] = current["controller_snapshot"]["commit"]
        return result

    def should_stop(current, started):
        # The old horizon and its shutdown margin survive controller migration.
        absolute = _date(binding["original_absolute_queue_deadline"])
        margin = min(60, binding["old_config"]["max_queue_seconds"] * 0.1)
        return (dt.datetime.now(dt.timezone.utc) >= absolute - dt.timedelta(seconds=margin)
                or time.monotonic() - started >= current["max_queue_seconds"] - margin)

    def accounting(current, state):
        result = base_accounting(current, state)
        prior = current["prior_gpu_request_usage"]["seconds"]
        result.update(prior_gpu_request_seconds=prior,
                      total_used_gpu_request_seconds_including_prior=result["used_gpu_request_seconds"] + prior,
                      original_total_gpu_budget_seconds=143100,
                      original_absolute_queue_deadline=binding["original_absolute_queue_deadline"])
        return result

    queue.wrapped_request = wrapped
    queue.initial_stage_states = lambda current: copy.deepcopy(binding["stage_states"])
    queue.queue_should_stop = should_stop
    queue.accounting = accounting
    return binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["validate", "render-requests", "supervise", "run-stage", "status"])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage")
    args = parser.parse_args()
    config = queue.read(args.config)
    if args.mode == "status":
        print(json.dumps(queue.read(Path(config["state_dir"]) / "queue-status.json"), indent=2))
        return 0
    queue.validate(config, args.config)
    binding = install(config, args.config, live=args.mode != "run-stage")
    if args.mode == "validate":
        print(json.dumps(dict(inherited_stages=list(binding["inherited_requests"]),
                              original_absolute_queue_deadline=binding["original_absolute_queue_deadline"],
                              stage_statuses={k: v["status"] for k, v in binding["stage_states"].items()}), indent=2))
        return 0
    if args.mode == "render-requests":
        print(json.dumps([queue.wrapped_request(config, args.config, s) for s in config["stages"]], indent=2))
        return 0
    if args.mode == "run-stage":
        return queue.run_stage(config, args.config, args.stage)
    return queue.supervise(config, args.config)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BaseException as error:
        if isinstance(error, SystemExit):
            raise
        print(json.dumps(dict(at=queue.utc(), error=repr(error), action="STOP_NO_AUTOMATIC_RETRY")), flush=True)
        sys.exit(2)
