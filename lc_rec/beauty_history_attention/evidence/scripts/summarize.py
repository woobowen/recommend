#!/usr/bin/env python3
"""Read collected raw receipts and calculate the report's scope and resource totals."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import statistics


PREFIX = "letter-beauty-v2-history-attn-"


def load(path):
    return json.loads(path.read_text())


def attempt(root, path):
    state = load(path / "status.json")
    data = {k: state.get(k) for k in ("job_id", "status", "exit_code", "started_at", "ended_at")}
    data["source"] = str(path.relative_to(root) / "status.json")
    return data


def summarize(root):
    attempts = {path.name.removeprefix(PREFIX): attempt(root, path)
                for path in sorted(root.glob(PREFIX+"*"))}
    report = {"attempts": attempts, "formal_run_started": False}
    training = root / "train-r1/training"
    if training.is_dir():
        final = load(training / "rank0.json")
        rows = [json.loads(line) for line in (training / "events.rank0.jsonl").read_text().splitlines()]
        history = report["training"] = {
            "status": final["status"], "completed_updates": final.get("global_step"),
            "trainable_tensor_count": len(final["optimizer_parameter_names"]),
            "trainable_elements": final["optimizer_parameter_count"],
            "updates": [{k:r.get(k) for k in ("step", "lr", "grad_norm", "attempt_job_id")}
                        for r in rows if r["event"] == "optimizer_step"],
            "resume_events": [r for r in rows if r["event"] in ("partial_restore_exact", "loader_replayed")],
            "validation": [{"update": r["step"], **r["values"]} for r in rows if r["event"] == "log"
                           and "eval_loss" in r["values"]],
            "epoch_boundary_updates": [r["step"] for r in rows if r["event"] == "epoch_end"],
            "peak_allocated_bytes": max(r["lifecycle_peak_allocated_bytes"] for r in rows if r["event"] == "gpu_memory"),
            "peak_reserved_bytes": max(r["lifecycle_peak_reserved_bytes"] for r in rows if r["event"] == "gpu_memory"),
            "attention": final["attention"],
        }
        if (training / "best_selection.json").is_file():
            history["selection"] = load(training / "best_selection.json")
        per_step = [r for r in rows if r["event"] == "micro_loss"]
        pure_batch = []
        for prev, nxt in zip(per_step, per_step[1:]):
            if (prev["attempt_job_id"] != nxt["attempt_job_id"] or prev["epoch_index"] != nxt["epoch_index"]
                    or prev["ds_accumulation_boundary"] or nxt["ds_accumulation_boundary"]):
                continue
            pure_batch.append((datetime.fromisoformat(nxt["utc"]) - datetime.fromisoformat(prev["utc"])).total_seconds())
        if pure_batch:
            median_microbatch = statistics.median(pure_batch)
            eval_rate = statistics.median(row["eval_samples_per_second"] for row in history["validation"])
            checkpoint_seconds = statistics.median(r["elapsed_seconds_checkpoint"] for r in rows
                                                    if r["event"] == "checkpoint_complete")
            history["microbatch_timings"] = {
                "count": len(pure_batch), "median_seconds": median_microbatch,
                "mean_seconds": statistics.mean(pure_batch), "maximum_seconds": max(pure_batch),
                "scope": "adjacent micro_loss timestamps within the same epoch/attempt and no optimizer boundary; excludes startup and validation",
            }
            history["rough_single_gpu_formal_hours_before_buffer"] = (
                median_microbatch * 65_712 + (44_726 / eval_rate) * 4 + 4 * checkpoint_seconds
            ) / 3600
    for directory in (root / "generate-r1", root / "generate-r2", root / "generate-r3"):
        state = directory / "predictions/generation.json"
        if state.is_file():
            generation = load(state)
            result = report["evaluation"] = {k: generation.get(k) for k in (
                "status", "preflight_assignment", "conditions", "gpu_peak_reserved_bytes", "gpu_peak_allocated_bytes",
                "attention_runtime", "elapsed_wall_seconds")}
            conditions = generation.get("conditions")
            if conditions:
                result["rough_single_gpu_formal_five_condition_hours_before_buffer"] = sum(
                    data["generation_seconds"] / data["completed_users"] * 22_363
                    for data in conditions.values()) / 3600
    for folder in sorted(root.glob("metrics-*")):
        outcome = folder / "metrics/summary.json"
        if outcome.is_file():
            report["metrics"] = {k:v for k,v in load(outcome).items()
                                 if k in ("status", "users_per_condition", "condition_count", "run_commit",
                                          "evaluation_commit", "generation_sha256")}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collected", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.collected), indent=2, allow_nan=False) + "\n")
