"""Estimate full-run resources from the successful single-card preflight."""
import argparse
import json
import math
from pathlib import Path
import statistics


def estimate(path):
    result = json.loads((path / "stage_result.json").read_bytes())
    startup = json.loads((path / "startup.json").read_bytes())
    events = [json.loads(line) for line in (path / "events.jsonl").read_bytes().splitlines()]
    batches = {e["microbatch"]: e["time"] for e in events if e["event"] == "train_batch"}
    assert set(batches) == {1, *range(160, 1601, 160)}
    # Eight full intervals after the first 320 microbatches avoid initial input/setup.
    update_seconds = [(batches[n + 160] - batches[n]) / 10 for n in range(320, 1600, 160)]
    validation, = [e for e in events if e["event"] == "validation_end"]
    steady_val = (validation["last_batch_seconds"] - validation["first_batch_seconds"]) / 23
    full_validation = (validation["first_batch_seconds"] + 3096 * steady_val
                       + validation["seconds"] - validation["last_batch_seconds"])
    test = [e["loop_elapsed_seconds"] for e in events if e["event"] == "test_batch" and e["evaluation"] == 1]
    warm_test_batch = test[1] - test[0]
    full_test = result["test_seconds"] + 3094 * warm_test_batch
    expected_update = statistics.median(update_seconds)
    seconds_per_period = 100 * expected_update + full_validation
    evaluation_upper = (24772 / 20) * result["offline_seconds"]
    initial = startup["initial_checks_seconds"] + result["startup_and_instance_seconds"]
    estimates = {n: (initial + (n / 100) * seconds_per_period + full_test + evaluation_upper) / 3600
                 for n in (4000, 6400, 9000)}
    # Permit 25% above the 9,000-update scenario, plus up to 15 minutes for
    # startup, artifact handoff and moderate measurement variance.
    hours = min(24, math.ceil(1.25 * estimates[9000] + .25))
    return {
        "basis": "actual 100-updates preflight; no full-val/test timings",
        "estimated_initialization_seconds": initial,
        "observed_stable_seconds_per_update": update_seconds,
        "median_seconds_per_update": expected_update,
        "observed_validation_seconds_24_batches": validation["seconds"],
        "estimated_validation_seconds_3097_batches": full_validation,
        "estimated_validation_fraction_per_period": full_validation / seconds_per_period,
        "observed_normal_test_seconds_20_users": result["test_seconds"],
        "observed_candidate_serialization_seconds_20_users": result["candidate_write_seconds"],
        "estimated_full_candidate_serialization_seconds": result["candidate_write_seconds"] * 24772 / 20,
        "estimated_normal_test_seconds_24772_users": full_test,
        "observed_offline_seconds_20_users": result["offline_seconds"],
        "upper_extrapolation_offline_seconds": evaluation_upper,
        "measured_peak_gpu_allocated_bytes": result["peak_gpu_bytes"],
        "measured_peak_gpu_reserved_bytes": result["peak_gpu_reserved_bytes"],
        "measured_checkpoint_bytes": result["checkpoint"]["bytes"],
        "estimated_full_candidate_bytes": math.ceil((path / "candidates.jsonl").stat().st_size * 24772 / 20),
        "estimated_full_per_user_bytes": math.ceil((path / "per_user_metrics.jsonl").stat().st_size * 24772 / 20),
        "wall_hours_by_update_scenario": estimates,
        "proposed_timeout_seconds": hours * 3600,
        "proposed_gpu_device_hours": hours,
        "min_server_and_staging_free_bytes_each": 2 * 1024**3,
        "historical_anchor_updates": 6400,
        "interpretation": "Forecast scales warm fixed-shape batches, not achieved full coverage. "
                          "The 4k/6.4k/9k scenarios only bracket elapsed time. Formal remains unsubmitted. "
                          "The proposed wall cap is bounded by the runner's 24-hour limit; validation coverage, "
                          "data timing and natural early stopping remain uncertain."
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(json.dumps(estimate(args.run), indent=2))
