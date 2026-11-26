"""Verify terminal, complete validation, best selection, and test-loading evidence."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re

import yaml

CONDITIONS = ("Normal", "Shuffle3a", "Shuffle4")
STAGES = ("quantizer", "sid", "bound", "training", *CONDITIONS, "summary")
METRICS = ("recall@5", "recall@10", "ndcg@5", "ndcg@10")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as fp:
        return hashlib.file_digest(fp, "sha256").hexdigest()


def normalized_config(path, replacements):
    def relocate(value):
        if isinstance(value, str):
            for source, destination in replacements:
                value = value.replace(source, destination)
        if isinstance(value, list):
            return [relocate(v) for v in value]
        if isinstance(value, dict):
            return {key: relocate(v) for key, v in value.items()}
        return value

    cfg = yaml.safe_load(path.read_bytes())
    cfg.pop("id")
    return relocate(cfg)


def rows(path):
    with path.open(newline="") as stream:
        data = list(csv.DictReader(stream))
    for row in data:
        assert None not in row and all(v is not None and (v == "" or math.isfinite(float(v))) for v in row.values())
    return data


def verify(terminal, query_file, checkpoint_file, quantizer_file, runtime_plan):
    q = read(query_file)
    raw = read(terminal / "status.json")
    assert all(q[key] == raw[key] for key in ("job_id", "status", "started_at", "ended_at", "exit_code", "artifacts"))
    assert q["status"] == "SUCCEEDED" and q["exit_code"] == 0 and q["cleanup_verified"]
    assert not any(q[key] for key in ("supervisor_alive", "worker_alive", "workgroup_live", "worker_may_be_live"))
    assert read(terminal / "cleanup.json") == raw["cleanup"]
    assert raw["cleanup"]["reason"] == "NORMAL_EXIT_CHECK" and raw["cleanup"]["verified"]
    req = read(terminal / "request.json")
    canon = hashlib.sha256(json.dumps(req, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    assert req["job_id"] == q["job_id"] and req["argv"][-3:] == ["execute", "--mode", "formal"]
    assert req["timeout_seconds"] == 86400 and req["scope"]["reuse_job"] is None
    assert canon == q["request_sha256"] == read(terminal / "accepted.json")["request_sha256"]
    assert canon == read(terminal / "attempt.json")["request_sha256"]
    run = terminal / "output/run"
    startup = read(run / "startup.json")
    assert startup["mode"] == "formal" and startup["version"] == req["capture_version"]
    assert startup["scope"] == req["scope"] and startup["cwd"] == req["cwd"]
    assert startup["resources"]["gpu"]["uuid"] == req["env"]["CUDA_VISIBLE_DEVICES"]

    quantity = read(quantizer_file)
    expected_artifacts = {record["path"]: record for record in q["artifacts"]}
    retained = [name for name in expected_artifacts if not (terminal / name).is_file()]
    assert len(retained) == 1 and retained[0].startswith("output/run/upstream/") and retained[0].endswith(".ckpt")
    assert expected_artifacts[retained[0]]["sha256"] == quantity["checkpoint_sha256"]
    for label, record in expected_artifacts.items():
        if label in retained:
            continue
        path = terminal / label
        assert path.stat().st_size == record["bytes"] and sha(path) == record["sha256"], label
    stage_seconds = {}
    folder_names = {"quantizer": "upstream", "sid": "upstream", "bound": "bound",
                    "training": "training", "summary": "evaluation"} | {mode: f"evaluation/{mode}" for mode in CONDITIONS}
    for stage in STAGES:
        receipt = read(run / "stages" / (stage + ".json"))
        assert receipt["stage"] == stage and receipt["mode"] == "formal" and receipt["exit_code"] == 0
        assert receipt["reused_from"] is None and receipt["argv"][:2] == [req["argv"][0], "-B"]
        assert receipt["argv"][2].startswith(req["cwd"] + "/")
        assert all(expected_artifacts[f"output/run/{folder_names[stage]}/{name}"]["sha256"] == value
                   for name, value in receipt["artifacts"].items())
        stage_seconds[stage] = receipt["seconds"]

    work = (terminal / "work.log").read_text()
    assert re.findall(r"^Starting (\S+) \(formal\)$", work, re.MULTILINE) == list(STAGES)
    assert not re.search(r"Traceback \(most recent call last\)|OutOfMemoryError|CUDA out of memory", work)
    stop, = [line for line in work.splitlines() if line.startswith("Monitored metric ") and "Trainer to stop" in line]

    training_dir = run / "training"
    train = read(training_dir / "training.json")
    best = read(training_dir / "best.json")
    val = [json.loads(line) for line in (training_dir / "validation.jsonl").read_text().splitlines()]
    assert train["validation"] == val and train["train_seed"] == 42 and train["legacy_shuffle_calls"] == 0
    cfg_train = training_dir / "effective_config.yaml"
    cfg = yaml.safe_load(cfg_train.read_bytes())
    plan_root = str(Path(req["cwd"]).parent / "formal-runtime-plan")
    remote_run = q["result_path"] + "/run"
    assert normalized_config(cfg_train, []) == normalized_config(runtime_plan / "downstream/effective_config.yaml",
        [(plan_root + "/downstream", remote_run + "/training"), (plan_root + "/bound", remote_run + "/bound")])
    assert cfg["ckpt_path"] is None and cfg["seed"] == 42
    assert cfg["trainer"]["precision"] == "32-true" and cfg["trainer"]["max_steps"] == 320000
    assert cfg["trainer"]["accumulate_grad_batches"] == 16 and cfg["trainer"]["val_check_interval"] == 1600
    early = cfg["callbacks"]["early_stopping"]
    select = cfg["callbacks"]["model_checkpoint"]
    assert early["monitor"] == select["monitor"] == "val/recall@5"
    assert early["mode"] == select["mode"] == "max" and select["save_top_k"] == 1
    assert early["min_delta"] == 0.0 and early["patience"] == 10
    assert "did not improve in the last 10 records" in stop
    assert [v["step"] for v in val] == list(range(100, train["updates"] + 1, 100))
    assert all(v["batches"] == 3097 and math.isfinite(v["recall_at_5"]) for v in val)
    assert sum(train["expanded_microbatch"].values()) == train["updates"] * 16
    assert all(0 < int(batch) <= 128 for batch in train["expanded_microbatch"])
    best_score = max(v["recall_at_5"] for v in val)
    winners = [v["step"] for v in val if v["recall_at_5"] == best_score]
    assert best["best_step"] == winners[0] and best["score"] == best_score
    assert len([v for v in val if v["step"] > best["best_step"]]) == early["patience"]
    csv_train = rows(training_dir / "csv/version_0/metrics.csv")
    csv_val = [line for line in csv_train if line["val/recall@5"]]
    assert [float(line["val/recall@5"]) for line in csv_val] == [v["recall_at_5"] for v in val]
    checkpoint = read(checkpoint_file)
    path = training_dir / best["file"]
    assert path.is_file() and sha(path) == best["sha256"] == checkpoint["checkpoint_sha256"]
    assert checkpoint["state_sha256"] == best["state_sha256"] and checkpoint["global_step"] == best["best_step"]
    assert checkpoint["optimizer"]["step_values"] == [best["best_step"]]
    assert checkpoint["selection"]["best_model_score"] == best["score"]
    assert checkpoint["selection"]["best_k_models"] == {checkpoint["source"]: best["score"]}
    assert best["binding_sha256"] == sha(run / "bound/binding.json")
    assert best["sid_sha256"] == sha(run / "upstream/semantic_ids.pt")
    assert best["sid_sha256"] == read(run / "bound/binding.json")["sid_sha256"]
    assert work.count("Loaded model weights from the checkpoint at " + checkpoint["source"]) == 3

    best_val = csv_val[best["best_step"] // 100 - 1]
    tests = {}
    for mode in CONDITIONS:
        ev = run / "evaluation" / mode
        assert normalized_config(ev / "effective_config.yaml", []) == normalized_config(cfg_train, [
            (remote_run + "/training", remote_run + "/evaluation/" + mode)])
        result = read(ev / "native_evaluation.json")
        assert result["reload"] == {"strict": True, "keys": checkpoint["state_keys"],
                                    "checkpoint_sha256": best["sha256"], "state_sha256": best["state_sha256"]}
        assert result["sid_sha256"] == best["sid_sha256"] and result["eval_seed"] == 2026
        assert result["checks"]["batch_size_8"] == 3096 and result["checks"]["batch_size_4"] == 1
        assert result["checks"]["history_groups"] == 172929
        test_row, = rows(ev / "csv/version_0/metrics.csv")
        assert {key: float(test_row[key]) for key in result["metrics"]} == result["metrics"]
        tests[mode] = {"metrics_runtime_fp32": result["metrics"], "loaded": result["reload"],
                       "batches": result["checks"]["batch_size_8"] + 1,
                       "candidate_sha256": result["candidate_sha256"], "checks": result["checks"]}
    seconds = (datetime.fromisoformat(q["ended_at"]) - datetime.fromisoformat(q["started_at"])).total_seconds()
    assert seconds < req["timeout_seconds"]
    return {"job": q["job_id"], "started_at": q["started_at"], "ended_at": q["ended_at"], "observed_utc": q["workgroup_observation"]["observed_at"],
            "wall_seconds": seconds, "runner_timeout_seconds": req["timeout_seconds"], "exit_code": 0,
            "execution_commit": req["capture_version"], "retained_quantizer_original": expected_artifacts[retained[0]],
            "artifact_bytes_verified": len(expected_artifacts) - 1, "stage_seconds": stage_seconds,
            "stop_reason_original": stop, "patience": early["patience"], "optimizer_updates": train["updates"],
            "microbatches": sum(train["expanded_microbatch"].values()), "full_validations": len(val),
            "last_validation": val[-1], "latest_validation_csv_row": csv_val[-1],
            "training_reported_seconds": train["seconds"], "best": best, "best_validation_csv_row": best_val,
            "checkpoint": checkpoint, "tests": tests}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for label in ("terminal", "query", "checkpoint", "quantizer", "runtime-plan"):
        cli.add_argument("--" + label, required=True, type=Path)
    args = cli.parse_args()
    print(json.dumps(verify(args.terminal, args.query, args.checkpoint, args.quantizer, args.runtime_plan),
                     indent=2, allow_nan=False))
