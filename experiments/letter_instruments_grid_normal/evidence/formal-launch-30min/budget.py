"""Compute one-card accounting and timing scenarios from the recorded formal events."""
import argparse
import datetime as dt
import json
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo

from summarize import records

parser = argparse.ArgumentParser()
parser.add_argument("stage", type=Path)
parser.add_argument("preparation_ledger", type=Path)
parser.add_argument("preflight_forecast", type=Path)
args = parser.parse_args()

def read(path):
    return json.loads(path.read_text())

request = read(args.stage / "request.json")
prep = read(args.preparation_ledger)
forecast = read(args.preflight_forecast)
points = read(args.stage / "samples-derived.json")
terminal = read(args.stage / "delivery-status.json")
events = records((args.stage / "captured/output/run/events.jsonl").read_text())
started = dt.datetime.fromisoformat(points[0]["point"]["target_time"]["utc"])
budget = request["timeout_seconds"]
delivery_elapsed = terminal["server_epoch"] - started.timestamp()
checks_begin = [e for e in events if e["event"] == "validation_start"]
checks_end = [e for e in events if e["event"] == "validation_end"]
updates = [e["train_seconds_since_validation"] for e in checks_begin]
validation = [e["seconds"] for e in checks_end]
train_per_100 = statistics.median(updates)
complete_validation = statistics.median(validation)
period = train_per_100 + complete_validation
initialization = next(e["time"] for e in events if e["event"] == "train_start") - started.timestamp()
test = forecast["estimated_normal_test_seconds_24772_users"]
offline = forecast["upper_extrapolation_offline_seconds"]
serialization = forecast["estimated_full_candidate_serialization_seconds"]
ny = ZoneInfo("America/New_York")
scenarios = {}
for stop_step in (4000, 6400, 9000):
    wall = initialization + stop_step / 100 * period + test + offline + serialization
    eta = started + dt.timedelta(seconds=wall)
    scenarios[str(stop_step)] = dict(total_seconds=wall, total_hours=wall / 3600,
        indicative_finish_utc=eta.isoformat(), indicative_finish_new_york=eta.astimezone(ny).isoformat())

print(json.dumps(dict(
    approved_gpu_devices=1, formal_budget_gpu_device_seconds=budget,
    attempt_count=1, formal_failed_attempt_gpu_device_seconds=0, formal_additional_cpu_runner_seconds=0,
    accounting="One assigned device times server worker wall time from kernel process start; included from initialization through validation, testing, and offline stages. Not nvidia-smi utilization time.",
    retained_preparation_gpu_device_seconds=prep["preflight_gpu_assigned_worker_seconds"],
    retained_preparation_cpu_runner_seconds=prep["cpu_runner_worker_seconds"],
    retained_remaining_cpu_runner_seconds=prep["cpu_runner_remaining_seconds"],
    observations=[{
        "label":point["point"]["label"],
        "sample":point["point"]["measured_time"],
        "consumed_gpu_device_seconds":point["consumed_gpu_device_seconds"],
        "remaining_gpu_device_seconds":point["remaining_gpu_device_seconds"],
        "server_free_bytes":point["server_free_bytes"],
        "local_staging_free_bytes":point["point"]["local_staging_free_bytes"],
        "windows_downloads_free_bytes":point["point"]["windows_downloads_free_bytes"],
    } for point in points],
    delivery_query=dict(
        sample=terminal["observation"]["measured_time"], gpu_device_seconds=delivery_elapsed,
        remaining_gpu_device_seconds=budget-delivery_elapsed, server_free_bytes=terminal["free_bytes"],
        local_staging_free_bytes=terminal["observation"]["local_staging_free_bytes"],
        windows_downloads_free_bytes=terminal["observation"]["windows_downloads_free_bytes"],
        status=terminal["status"]["status"],
    ),
    timing=dict(
        initialization_to_train_start_seconds=initialization,
        train_cycles=[dict(step=e["step"], seconds=e["train_seconds_since_validation"]) for e in checks_begin],
        completed_validations=[dict(step=e["step"], batches=e["batches"], seconds=e["seconds"]) for e in checks_end],
        median_train_seconds_per_100_updates=train_per_100, median_full_validation_seconds=complete_validation,
        median_cycle_model_seconds=period, projected_full_test_seconds=test,
        projected_candidate_serialization_seconds=serialization, upper_extrapolation_offline_seconds=offline,
        update_scenarios=scenarios,
        caveat="Few observed cycles; future early stopping, I/O and GPU speeds and full-test/offline times remain uncertain. Step scenarios are elapsed-time illustrations. The runner's 86400s limit is an execution abort followed by standard bounded cleanup."
    ),
), indent=2))
