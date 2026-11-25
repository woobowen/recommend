"""Tie source counters, bounded raw evidence, and training observations to each timestamp."""
import csv
from datetime import datetime
from io import StringIO
import json
import math
from pathlib import Path
import re
import sys
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
LOCAL = ZoneInfo("America/New_York")
STAGES = ["quantizer", "sid", "bound", "training", "Normal", "Shuffle3a", "Shuffle4", "summary"]


def load(path):
    return json.loads(path.read_text())


def complete_lines(text, *, omit_first=False):
    lines = text.splitlines(keepends=True)
    return [line for line in lines[int(omit_first):] if line.endswith(("\n", "\r"))]


def records(sample, filename, full_root=None):
    paths = [p for p in sample["files"] if filename(p)]
    rows = []
    for p in sorted(paths):
        if full_root is None:
            doc = sample["files"][p]
            content = doc["tail"]
        else:
            path = full_root / p
            if not path.is_file():
                continue
            doc = {"start_byte": 0}
            content = path.read_text()
        if p.endswith(".csv"):
            if doc["start_byte"] != 0:
                # Each sample can use the same captured CSV schema verified against the full prefix.
                header = (full_root or HERE / "cutoff").joinpath(p).open().readline()
                content = header + "".join(complete_lines(content, omit_first=True))
            items = list(csv.DictReader(complete_lines(content)))
            for row in items:
                assert None not in row and all(value is not None for value in row.values()), p
                for value in row.values():
                    if value:
                        assert math.isfinite(float(value)), (p, row)
        else:
            items = [json.loads(line) for line in complete_lines(content, omit_first=doc["start_byte"] != 0)]
        rows.extend({"file": p, "row": row} for row in items)
    return rows


def parse_gpu(sample, uuid):
    rows = csv.DictReader(StringIO(sample["gpu"]), skipinitialspace=True)
    return next(dict(r) for r in rows if r["uuid"] == uuid)


def decode_point(label, path, req, full_root=None):
    q = load(path / "query.json")
    sample = load(path / "sample.json")
    stamp = datetime.fromisoformat(sample["sample_started_utc"])
    files = sample["files"]
    finished = [s for s in STAGES if "output/run/stages/" + s + ".json" in files]
    failing = [s for s in STAGES if "output/run/stages/" + s + ".failed.json" in files]
    stage = next((s for s in STAGES if s not in finished), "summary complete")
    if failing:
        stage = failing[0] + " failed"
    train = records(sample, lambda p: "/training/csv/" in p and p.endswith("/metrics.csv"), full_root)
    steps = [r for r in train if r["row"].get("train/loss_step")]
    epochs = [r for r in train if r["row"].get("train/loss_epoch")]
    val = records(sample, lambda p: p.endswith("/training/validation.jsonl"), full_root)
    metric = [r for r in train if r["row"].get("val/recall@5")]
    gpu = parse_gpu(sample, req["env"]["CUDA_VISIBLE_DEVICES"])
    exceptions = re.findall(r"Traceback \(most recent call last\)|OutOfMemoryError|CUDA out of memory",
                            files.get("work.log", {}).get("tail", ""), re.I)
    assert q["request_sha256"] == load(HERE / "accepted.json")["request_sha256"]
    return {"label": label, "utc": stamp.isoformat(), "ny": stamp.astimezone(LOCAL).isoformat(),
            "stage": stage, "status": q["status"], "completed_stages": finished, "failed_stages": failing,
            "supervisor_alive": q["supervisor_alive"], "worker_alive": q["worker_alive"],
            "training_csv_rows_read": len(train), "last_train_step_row": steps[-1] if steps else None,
            "last_train_epoch_row": epochs[-1] if epochs else None, "validations": val,
            "last_validation_csv_row": metric[-1] if metric else None, "file_progress": {
                p: {k: v for k, v in row.get("before", row).items() if k in ("bytes", "mtime_utc")}
                for p, row in files.items() if p.endswith(("work.log", "/metrics.csv", "validation.jsonl"))
            },
            "checkpoints": sample["checkpoints"], "gpu": gpu, "free_bytes": sample["free_bytes"],
            "log_exception_markers": exceptions, "runner_exit_code": q.get("exit_code")}


def summarize():
    req = load(HERE / "request.json")
    accepted = load(HERE / "accepted.json")
    initial = load(HERE / "start-confirm.json")
    cutoff = HERE / "cutoff"
    start = datetime.fromisoformat(initial["time"])
    points = []
    for label in ("T00", "upstream-transition", "T10", "T20"):
        path = HERE / "observations" / label
        if (path / "sample.json").exists():
            points.append(decode_point(label, path, req))
    end = decode_point("T30", cutoff, req, cutoff)
    points.append(end)
    delivery = HERE / "observations/delivery"
    if (delivery / "sample.json").exists():
        points.append(decode_point("delivery", delivery, req))
    e = datetime.fromisoformat(end["utc"])
    run = HERE / "closed/output/run"
    fit = load(run / "upstream/fit_gate.json")
    sid = load(run / "upstream/sid_gate.json")
    binding = load(run / "bound/binding.json")
    stages = {name: load(run / "stages" / (name + ".json"))["seconds"] for name in ("quantizer", "sid", "bound")}
    best_path = HERE / "best-checkpoint/checkpoint_extraction.json"
    best = load(best_path) if best_path.is_file() else None
    val = [r["row"] for r in end["validations"]]
    if best:
        assert val and all(item["batches"] == 3097 for item in val)
        callback = best["callbacks"][0]
        assert callback["best_model_score"] == max(row["recall_at_5"] for row in val)
        assert any(row["step"] == best["global_step"] and row["recall_at_5"] == callback["best_model_score"]
                   for row in val)
        assert best["optimizer"][0]["step_values"] == [best["global_step"]]
    return {"job": req["job_id"], "canonical_request_sha256": accepted["request_sha256"],
            "snapshot_version": req["capture_version"], "worker_started_at_utc": initial["status"]["started_at"],
            "start_utc": start.isoformat(), "start_ny": start.astimezone(LOCAL).isoformat(),
            "cutoff_utc": e.isoformat(), "cutoff_ny": e.astimezone(LOCAL).isoformat(),
            "observed_seconds": (e - start).total_seconds(), "points": points,
            "count_note": "CSV step is the logged index. validation.jsonl step and checkpoint global_step come from Lightning. No offset inferred.",
            "stage_seconds": stages, "fit": {key: fit[key] for key in ("step", "raw_batches", "rows_processed_with_repeats", "distinct_rows_observed", "per_layer")},
            "sid": {key: sid[key] for key in ("unique_sids", "unique_prefixes", "prefix_group_size_histogram", "digit_ranges", "sid_sha256")},
            "bound_readback": binding["native_readback"], "checkpoint_metadata": best}


if __name__ == "__main__":
    result = summarize()
    out = Path(sys.argv[1])
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
