"""Verify saved historical GRID Normal aggregates and compare to the formal run."""
import argparse
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(), parse_float=Decimal)


def validation_axes(events, history):
    if history:
        return [(row["global_step"], row["metrics"]["val/recall@5"])
                for row in events if row["state"] == "validation_completed"]
    return [(row["step"], row["monitor"])
            for row in events if row["event"] == "validation_end"]


def verify_csv(path, validations, native, final_step):
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    monitor = [(int(row["step"]) + 1, Decimal(row["val/recall@5"]))
               for row in rows if row["val/recall@5"]]
    assert monitor == validations
    first_test = next(row for row in rows if row["test/loss"])
    assert int(first_test["step"]) == final_step
    assert all(Decimal(first_test[key]) == value for key, value in native.items())


def summarize_validation(data, expected_step, expected_value):
    best_step, best_score = max(data, key=lambda row: row[1])
    assert (best_step, best_score) == (expected_step, expected_value)
    assert [step for step, _ in data] == list(range(100, data[-1][0] + 1, 100))
    trailing = [(step, score) for step, score in data if step > best_step]
    assert len(trailing) == 10 and all(score < best_score for _, score in trailing)
    return {"best_step": best_step, "val_recall_at_5": str(best_score),
            "terminal_step": data[-1][0], "validation_count": len(data),
            "checks_without_improvement_after_best": len(trailing)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("historical_original", type=Path, help="fixed historical events/CSV and source.json")
    parser.add_argument("runtime_historical", type=Path, help="historical directory in fixed runtime")
    parser.add_argument("formal_run", type=Path, help="saved formal run directory")
    args = parser.parse_args()
    source = read(args.historical_original / "source.json")
    for name, identity in source["files"].items():
        content = (args.historical_original / name).read_bytes()
        assert len(content) == identity["bytes"]
        assert hashlib.sha256(content).hexdigest() == identity["sha256"]
    past_stage = read(args.runtime_historical / "stage_result.json")
    past = read(args.runtime_historical / "test_unshuffle.json")
    assert past_stage["test_metrics"]["unshuffle"] == past
    assert past["evaluated_users"] == 24772 and past["full_test_set"]
    assert not past["shuffle_enabled"] and past["shuffle_calls"] == 0
    assert past["checkpoint_sha256"] == past_stage["checkpoint_sha256"]
    past_native = past["metrics"][0]
    now_native, = read(args.formal_run / "native_metrics.json")
    now_stage = read(args.formal_run / "stage_result.json")
    assert now_stage["test"] == "Normal" and now_stage["shuffle_calls"] == 0
    assert now_stage["eval_seed"] == past["eval_seed"] == 2026
    assert now_stage["train_seed"] == past_stage["train_seed"] == 42
    assert past_stage["trained_from_scratch"]
    hist_events = [json.loads(line, parse_float=Decimal)
                   for line in (args.historical_original / "events.jsonl").read_text().splitlines()]
    now_events = [json.loads(line, parse_float=Decimal)
                  for line in (args.formal_run / "events.jsonl").read_text().splitlines()]
    old_val = validation_axes(hist_events, history=True)
    new_val = validation_axes(now_events, history=False)
    verify_csv(args.historical_original / "metrics.csv", old_val, past_native, past_stage["global_step"])
    verify_csv(args.formal_run / "csv/version_0/metrics.csv", new_val, now_native, now_stage["global_step"])
    best_past = int(Path(past["checkpoint"]).stem.split("step=")[-1])
    old = summarize_validation(old_val, best_past, past_stage["best_validation_recall5"])
    new = summarize_validation(new_val, now_stage["checkpoint"]["step"], now_stage["checkpoint"]["score"])
    comparison = {}
    for field in sorted(past_native):
        before = past_native[field]
        after = now_native[field]
        comparison[field] = {"historical": str(before), "formal": str(after),
                             "formal_minus_historical": str(after - before)}
    print(json.dumps(dict(historical_commit=source["commit"], historical=old, formal=new,
                          native_comparison=comparison), indent=2))


if __name__ == "__main__":
    main()
