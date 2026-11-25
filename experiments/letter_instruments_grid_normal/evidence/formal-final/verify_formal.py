"""Verify the immutable formal export and replay its metrics on CPU."""
import argparse
from collections import Counter
import csv
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics
import sys
import time
from zoneinfo import ZoneInfo

METRICS = ("val/recall@5", "val/recall@10", "val/ndcg@5", "val/ndcg@10", "val/loss")
NY = ZoneInfo("America/New_York")

def load(path):
    return json.loads(path.read_bytes())

def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def iso(timestamp):
    t = dt.datetime.fromtimestamp(timestamp, dt.timezone.utc)
    return {"utc": t.isoformat(), "new_york": t.astimezone(NY).isoformat()}

def runner(job, query, fixed):
    request = load(job / "request.json")
    digest = hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False,
                                      separators=(",", ":")).encode()).hexdigest()
    assert digest == query["request_sha256"] == load(job / "accepted.json")["request_sha256"]
    assert load(job / "attempt.json")["request_sha256"] == digest
    assert request == load(fixed / "request.json")
    assert request["capture_version"] == "390df9f508b7dc11eaa57b79a5a91fb5a7f99385"
    assert request["argv"][3] == "formal" and request["timeout_seconds"] == 86400
    assert request["scope"]["max_gpu_devices"] == 1 and request["scope"]["test_users"] == 24772
    status = load(job / "status.json")
    assert all(status[k] == query[k] for k in status)
    assert status["status"] == "SUCCEEDED" and status["exit_code"] == 0
    assert not any(query[k] for k in ("worker_alive", "supervisor_alive", "workgroup_live", "worker_may_be_live"))
    assert query["cleanup_verified"] and query["workgroup_observation"]["verified_empty"]
    cleanup = load(job / "cleanup.json")
    assert cleanup == status["cleanup"] and cleanup["verified"] and not cleanup["events"]
    assert cleanup["reason"] == "NORMAL_EXIT_CHECK"
    assert cleanup["initial_observation"]["verified_empty"] and cleanup["last_observation"]["verified_empty"]
    files = load(job / "export-manifest.json")["files"]
    expected = {f["path"] for f in files}
    assert len(expected) == len(files)
    actual = {str(p.relative_to(job)) for p in job.rglob("*") if p.is_file()}
    assert actual == expected | {"export-manifest.json"}
    for f in files:
        p = job / f["path"]
        assert not p.is_symlink() and job.resolve() in p.resolve().parents
        assert (p.stat().st_size, sha(p)) == (f["bytes"], f["sha256"])
    assert [f for f in files if f["path"].startswith("output/")] == status["artifacts"]
    last_observed_live = load(fixed / "snapshots/T30.json")["status"]
    assert status["worker"] == last_observed_live["worker"] and status["supervisor"] == last_observed_live["supervisor"]
    return request, status, digest

def verify_bound_files(request, runtime, assets, job, fixed):
    for item in request["inputs"]:
        source = item["path"]
        relative = source.removeprefix(request["cwd"] + "/")
        target = runtime / relative if relative != source else assets / source.split("/assets/", 1)[1]
        assert (sha(target) == item["sha256"]), source
    admission = load(fixed / "admission.json")
    for name in ("effective_config.yaml", "overrides.json"):
        assert sha(job / "output/run" / name) == admission["plan_sha256"][name]
    startup = load(job / "output/run/startup.json")
    model = load(job / "output/run/model.json")
    assert startup["argv"] == request["argv"][2:] and startup["cwd"] == request["cwd"]
    assert startup["asset_coverage"]["rows_exact"] == dict(training=24772, evaluation=24772, testing=24772)
    assert startup["asset_coverage"]["items"] == 9922 and startup["asset_coverage"]["unique_sids"] == 9897
    assert startup["gpu"]["uuid"] == request["env"]["CUDA_VISIBLE_DEVICES"]
    assert model["parameters"] == model["trainable_parameters"] == 13128064
    assert model["deterministic"] and model["matmul_precision"] == "medium" and model["shared"]
    assert model["bos"] == model["sep"] == [1, 128] and model["heads"] == [[256, 128]] * 4
    assert model["beam"] == 10 and model["sid_matches_official"] and not model["prefix_filter"]
    cfg = (job / "output/run" / "effective_config.yaml").read_text()
    for setting in ("ckpt_path: null", "seed: 42", "devices: 1", "precision: 32-true",
                    "batch_size_per_device: 16", "accumulate_grad_batches: 16", "val_check_interval: 1600",
                    "lr: 0.001", "weight_decay: 0.0001", "monitor: val/recall@5", "patience: 10",
                    "save_top_k: 1", "save_weights_only: false", "max_steps: 320000", "max_epochs: 10",
                    "top_k_for_generation: 10", "eval_seed: 2026", "shuffle_embedding_on_test: false"):
        assert setting in cfg, setting
    assert not any(key in cfg for key in ("limit_train_batches", "limit_val_batches", "limit_test_batches", "preflight_testing"))
    return startup

def training(run, stage, log, status):
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    with (run / "csv/version_0/metrics.csv").open(newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert all(math.isfinite(float(value)) for row in csv_rows for value in row.values() if value)
    assert all(a["time"] <= b["time"] for a, b in zip(events, events[1:]))
    assert not any(e["event"] == "error" for e in events)
    assert not re.search(r"Traceback|out of memory|CUDA error|non.finite|\b(?:NaN|Inf)\b", log, flags=re.I)
    (start,) = [r for r in events if r["event"] == "train_start"]
    assert start["step"] == start["optimizer_state_entries"] == 0
    assert (start["adam"]["lr"],start["adam"]["weight_decay"],start["adam"]["betas"],start["adam"]["eps"]) == (.001,.0001,[.9,.999],1e-8)
    fit_step = stage["global_step"]
    batches = [e for e in events if e["event"] == "train_batch"]
    assert [e["step"] for e in batches] == list(range(0, fit_step + 1, 10))
    assert [e["microbatch"] for e in batches] == [1, *range(160, fit_step * 16 + 1, 160)]
    assert set(e["shuffle_calls"] for e in batches) == {0}
    starts = [e for e in events if e["event"] == "validation_start"]
    firsts = [e for e in events if e["event"] == "validation_contract"]
    ends = [e for e in events if e["event"] == "validation_end"]
    steps = list(range(100, fit_step + 1, 100))
    assert all([e["step"] for e in series] == steps for series in (starts, firsts, ends))
    assert [e["seconds"] for e in ends] == stage["validation_seconds"]
    vals = [r for r in csv_rows if r["val/recall@5"]]
    assert len(vals) == len(ends)
    curve = []
    best_score = -1
    saves = []
    for a, first, b, row in zip(starts, firsts, ends, vals, strict=True):
        assert a["time"] <= first["time"] <= b["time"] and first["size"] == 8
        assert b["batches"] == 3097 and 0 <= b["monitor"] <= 1 and b["monitor"] == float(row["val/recall@5"])
        assert int(row["step"]) == b["step"] - 1
        current_best = b["monitor"] > best_score
        if current_best:
            saves.append(b["step"])
            best_score = b["monitor"]
        curve.append(dict(step=b["step"], csv_step=int(row["step"]), end_utc=iso(b["time"])["utc"],
                          batches=b["batches"], validation_seconds=b["seconds"],
                          train_window_seconds=a["train_seconds_since_validation"], best_updated=current_best,
                          **{key:float(row[key]) for key in METRICS}))
    logfile_saves = [int(i) for i in re.findall(r"global step (\d+).*saving model", log)]
    assert logfile_saves == saves
    ckpt = stage["checkpoint"]
    assert saves[-1] == ckpt["step"] and ckpt["score"] == best_score
    assert len([r for r in curve if r["step"] > saves[-1] and r["val/recall@5"] <= best_score]) == 10
    assert "did not improve in the last 10 records" in log and "Signaling Trainer to stop." in log
    assert "max_steps=" not in log and "max_epochs=" not in log
    assert "Seed set to 2026" in log and log.count("Restoring states from the checkpoint path") == 1
    assert log.count("Loaded model weights from the checkpoint") == 1 and log.count("Seed set to 2026") == 1
    assert ckpt["path"] in log and stage["mode"] == "formal" and stage["test"] == "Normal"
    assert stage["train_seed"] == 42 and stage["eval_seed"] == 2026 and stage["shuffle_calls"] == 0
    (test_start,) = [e for e in events if e["event"] == "test_start"]
    (test_end,) = [e for e in events if e["event"] == "test_end"]
    assert test_start["evaluation"] == test_end["evaluation"] == 1
    assert test_start["time"] > ends[-1]["time"] and test_end["time"] > test_start["time"]
    assert test_end["shuffle_calls"] == 0 and test_end["step"] == fit_step
    assert test_end["batch_sizes"] == {"8": 3096, "4": 1}
    assert set(test_end["metric_denominators"].values()) == {24772} and len(test_end["metric_denominators"]) == 4
    checkpoint_path = run / ckpt["path"]
    assert [p.name for p in (run / "checkpoints").iterdir()] == [checkpoint_path.name]
    assert checkpoint_path.stat().st_size == ckpt["bytes"] and sha(checkpoint_path) == ckpt["sha256"]
    receipt = load(run / "reload_enabled.json")
    assert receipt["strict"] is True and receipt["missing"] == receipt["unexpected"] == []
    assert receipt["checkpoint_sha256"] == ckpt["sha256"] and receipt["model_state_sha256"] == ckpt["model_state_sha256"]
    begin = dt.datetime.fromisoformat(status["started_at"])
    end = dt.datetime.fromisoformat(status["ended_at"])
    spent = (end - begin).total_seconds()
    loss = [(int(row["step"]), float(row["train/loss_step"])) for row in csv_rows if row["train/loss_step"]]
    assert [t for t, _ in loss] == list(range(9, fit_step, 10))
    return dict(
        stop="patience_10_no_improvement", start=iso(begin.timestamp()), end=iso(end.timestamp()),
        train_start=iso(start["time"]), fit_end_estimate_from_recorded_fit_seconds=iso(start["time"] + stage["fit_seconds"]),
        final_validation_end=iso(ends[-1]["time"]), test_start=iso(test_start["time"]), test_end=iso(test_end["time"]),
        optimizer_step=fit_step, microbatches=batches[-1]["microbatch"], validation_count=len(curve),
        best_updates=saves, best=ckpt, strict_reload=receipt, test_batch_size_counts=test_end["batch_sizes"],
        test_denominators=test_end["metric_denominators"],
        gpu_device_seconds=spent, remaining_gpu_device_seconds=86400-spent,
        fit_seconds=stage["fit_seconds"], validation_seconds=math.fsum(stage["validation_seconds"]),
        fit_outside_validation_seconds=stage["fit_seconds"]-math.fsum(stage["validation_seconds"]),
        validation_cycle_median=statistics.median(stage["validation_seconds"]), startup_to_fit_seconds=start["time"]-begin.timestamp(),
        test_seconds=stage["test_seconds"], candidate_write_seconds=stage["candidate_write_seconds"],
        offline_seconds=stage["offline_seconds"], peak_allocated_bytes=stage["peak_gpu_bytes"],
        peak_reserved_bytes=stage["peak_gpu_reserved_bytes"],
        loss_points=len(loss), loss_median_first_1000_updates=statistics.median([v for s,v in loss if s < 1000]),
        loss_median_last_1000_updates=statistics.median([v for s,v in loss if s >= fit_step-1000]),
        loss_first=dict(csv_step=loss[0][0], value=loss[0][1]), loss_last=dict(csv_step=loss[-1][0], value=loss[-1][1]),
        val_first=curve[0], val_last=curve[-1],
    ), curve

def replay(assets, run):
    from candidates import rows, scores
    from data_assets import official
    from evaluate import FP32_TOLERANCE, fixtures, verify
    (native,) = load(run / "native_metrics.json")
    assert all(math.isfinite(v) for v in native.values()) and len(native) == 5
    replay_rows = io.StringIO()
    received = verify(assets, run / "candidates.jsonl", "formal", native, replay_rows)
    assert received == load(run / "offline_evaluation.json")["replay"]
    assert received["coverage"] == 24772 and received["statistics"]["candidates"] == 247720
    assert len(received["metrics"]["native"]) == 4 and len(received["metrics"]["unified"]) == 11
    original = (run / "per_user_metrics.jsonl").read_text()
    assert replay_rows.getvalue() == original
    codes, sequences, groups = official(assets)
    extra = Counter()
    out = []
    candidates = run / "candidates.jsonl"
    for record in rows(candidates):
        uid = record["global_row"]
        assert record["target_sid"] == list(codes[sequences[uid][-1]])
        sids = [tuple(s) for s in record["ids"]]
        probabilities = scores(record)
        extra["catalog_colliding_candidates"] += sum(groups[s] > 1 for s in sids)
        extra["rows_with_catalog_colliding_candidate"] += any(groups[s] > 1 for s in sids)
        for ix, sid in enumerate(sids):
            if sid not in groups:
                out.append({"global_row": uid, "original_position": ix, "sid": sid,
                            "score_f32le": record["scores_f32le"][8 * ix:8 * (ix + 1)]})
        extra["target_hits_with_colliding_target"] += groups[tuple(record["target_sid"])] > 1 and tuple(record["target_sid"]) in sids
        extra["zero_score_candidates"] += sum(p == 0 for p in probabilities)
    assert len(out) == received["statistics"]["out_of_catalog"]
    stats = dict(extra, catalog_unique_sids=len(groups), catalog_collision_groups=sum(n > 1 for n in groups.values()),
                 catalog_items_in_collision_groups=sum(n for n in groups.values() if n > 1),
                 out_of_catalog_rows=len({r["global_row"] for r in out}), out_of_catalog_records=out)
    return dict(native_runtime=native, replay=received, fp32_abs_tolerance=FP32_TOLERANCE,
                all_per_user_serialized_rows_exact=True, per_user_metrics_sha256=sha(run/"per_user_metrics.jsonl"),
                candidate_sha256=sha(candidates), additional_candidate_statistics=stats, fixtures=fixtures())

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("runtime", "assets", "job", "query", "launch", "output"):
        parser.add_argument("--"+flag, type=Path, required=True)
    args = parser.parse_args()
    t0 = time.monotonic()
    query = load(args.query)
    request, status, digest = runner(args.job, query, args.launch)
    startup = verify_bound_files(request, args.runtime, args.assets, args.job, args.launch)
    sys.path.insert(0, str(args.runtime.resolve()))
    from data_assets import verify as verify_assets
    asset_axes = verify_assets(args.assets)
    run = args.job / "output/run"
    execution, curve = training(run, load(run / "stage_result.json"), (args.job / "work.log").read_text(), status)
    metrics = replay(args.assets, run)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "validation-trajectory.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curve[0]))
        writer.writeheader()
        writer.writerows(curve)
    report = dict(job_id=request["job_id"], capture_version=request["capture_version"],
                  request_sha256=digest, checked_asset_axes=asset_axes, gpu_admission=startup["gpu"],
                  runner_cleanup=status["cleanup"], execution=execution, evaluation=metrics,
                  argv=sys.argv, verification_wall_seconds=time.monotonic()-t0)
    with (args.output / "verification.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(stop=execution["stop"], step=execution["optimizer_step"], best=execution["best"]["step"],
                          coverage=metrics["replay"]["coverage"], verification_wall_seconds=report["verification_wall_seconds"])))

if __name__ == "__main__":
    main()
