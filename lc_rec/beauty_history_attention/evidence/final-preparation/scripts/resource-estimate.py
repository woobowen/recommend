"""Estimate formal B costs from stored real timings and the frozen length distribution."""

from collections import Counter
from datetime import datetime
import json
from pathlib import Path
from statistics import mean, median
import zipfile


HERE = Path("/home/addaswsw/recommend-support/beauty-history-attention-final-20261127")
OLD = HERE.parent / "beauty-history-attention-20261127"
ARCHIVE = Path("/home/addaswsw/recommend/LETTER_Beauty_LLaMA_history_attention_preflight_20261127.zip")
ACTUAL = HERE / "collected"
PILOT = ACTUAL / "pilot"
MODES = ("normal", "shuffle2", "shuffle3a", "shuffle3b", "shuffle4")


def read(path):
    return json.loads(path.read_text())


def run_seconds(state):
    delta = datetime.fromisoformat(state["ended_at"]) - datetime.fromisoformat(state["started_at"])
    return delta.total_seconds()


def main():
    source = read(OLD / "summary-current.json")
    training = source["training"]
    micro_s = training["microbatch_timings"]["median_seconds"]
    val_rate = median([r["eval_samples_per_second"] for r in training["validation"]])
    events = OLD / "collected/train-r1/training/events.rank0.jsonl"
    rows = [json.loads(line) for line in events.read_text().splitlines()]
    save_s = median([r["elapsed_seconds_checkpoint"] for r in rows if r["event"] == "checkpoint_complete"])
    archives = [read(f) for f in (OLD / "collected/train-r1/training").glob("checkpoint-*/complete.json")]
    maximum_checkpoint_bytes = max(sum(p["bytes"] for p in r["files"]) for r in archives)
    train = {
        "source": "Previously measured real B preflight, reused default training implementation",
        "training_median_seconds_per_microbatch": micro_s,
        "validation_median_rows_per_second": val_rate, "checkpoint_median_seconds": save_s,
        "training_compute_hours": 65712 * micro_s / 3600,
        "four_validation_hours": 4 * 44726 / val_rate / 3600,
        "four_checkpoint_hours": 4 * save_s / 3600,
        "largest_observed_complete_checkpoint_bytes": maximum_checkpoint_bytes,
        "training_peak_reserved_bytes": training["peak_reserved_bytes"],
        "scope": "Old bounded run, includes separate optimizer and resume stages, excludes fixed startup and long-run variation",
    }
    train["total_hours"] = sum(train[k] for k in ("training_compute_hours", "four_validation_hours",
                                               "four_checkpoint_hours"))

    with zipfile.ZipFile(ARCHIVE) as z:
        manifest = json.loads(z.read("MEMBERS.json"))["files"]
        item = "historical/server-originals/inputs.jsonl"
        doc = z.open(manifest[item].get("duplicate_of", item))
        counts = Counter(json.loads(line)["encoded_length"] for line in doc)
        item = "historical/actual-V2/formal-run.json"
        old_full_cpu_seconds = json.loads(z.read(manifest[item].get("duplicate_of", item)))["formal_cpu_worker_seconds"]
    population = sum(counts.values())
    assert population == 22363
    lengths = sorted(counts.elements())
    edges = sorted(set(lengths[int(i * (population - 1) / 4)] for i in range(1, 4)))

    def bucket(length):
        return sum(length > edge for edge in edges)

    serial = read(PILOT / "final/generation.json")
    assignment = serial["preflight_assignment"]
    attempts = [read(p) for p in (PILOT / "attempts").glob("*/attempt.json")]
    boundaries = {r.get("reused_users") for r in attempts if r.get("load_and_strict_verification_seconds")}
    assert boundaries == {0, 8}, boundaries
    warm_rows = [assignment[i] for i in sorted(boundaries)]
    generation = {}
    raw_byte_estimate = 0
    for mode in MODES:
        raw = PILOT / f"final/{mode}/predictions.rank0.jsonl"
        sample = [json.loads(line) for line in raw.read_text().splitlines()]
        assert [row["row_idx"] for row in sample] == assignment
        stable = [row for row in sample if row["row_idx"] not in warm_rows]
        strata = {
            i: [row["generation_seconds"] for row in stable if bucket(row["input_nonpadding_tokens"]) == i]
            for i in range(len(edges) + 1)
        }
        assert all(strata.values())
        point = 0.0
        for length, count in counts.items():
            point += count * mean(strata[bucket(length)])
        raw_byte_estimate += round(raw.stat().st_size * population / len(sample))
        generation[mode] = {
            "hours_reweighted": point / 3600,
            "raw_observed_seconds": sum(row["generation_seconds"] for row in sample),
            "warmup_seconds": [row["generation_seconds"] for row in sample if row["row_idx"] in warm_rows],
            "stable_median_seconds": median(row["generation_seconds"] for row in stable),
            "stable_bins": [dict(samples=len(strata[i]), mean_s=mean(strata[i]),
                                 users=sum(count for length, count in counts.items() if bucket(length) == i))
                            for i in sorted(strata)],
        }
    evidence = {
        "fit_scope": "Observed short pilot by encoded-length quartile and boundary cases. Excludes first user in each process from stable estimate.",
        "eager_generation_by_condition": generation, "frozen_length_quartile_edges": edges,
        "frozen_lengths_min_max": [lengths[0], lengths[-1]],
        "sample_lengths_min_max": [min(row["input_nonpadding_tokens"] for row in sample),
                                   max(row["input_nonpadding_tokens"] for row in sample)],
        "preflight_users_each_condition": len(assignment),
        "excluded_first_user_per_process": warm_rows,
        "loading_and_strict_reload_seconds": sorted(r["load_and_strict_verification_seconds"] for r in attempts),
        "observed_peak_reserved_bytes": max(r["gpu_peak_reserved_bytes"] for r in attempts),
        "estimated_final_raw_prediction_bytes_from_pilot_payload": raw_byte_estimate,
        "generation_total_hours": sum(r["hours_reweighted"] for r in generation.values()),
    }
    cpu = read(ACTUAL / "metrics-r0/metrics/summary.json")
    gpu_stages = ("old-causal-r0", "b-mask-r0", "generate-stop-r0", "generate-resume-r0",
                  "generate-resume-duplicate-r0")
    gpu_used = {stage: run_seconds(read(ACTUAL / f"letter-beauty-v2-history-attn-final-{stage}/status.json"))
                for stage in gpu_stages}
    result = {
        "training": train, "generation": evidence,
        "cpu_bounded_stage_seconds": cpu["stage_seconds"],
        "cpu_bounded_elapsed_seconds": cpu["elapsed_seconds"],
        "old_full_v2_four_modes_and_pairing_worker_seconds": old_full_cpu_seconds,
        "formal_cpu_metrics_io_and_closeout_minutes_provision": [10, 25],
        "new_preflight_gpu_reserved_wall_seconds": gpu_used,
        "new_preflight_gpu_reserved_wall_total_seconds": sum(gpu_used.values()),
        "rough_train_plus_generation_hours": train["total_hours"] + evidence["generation_total_hours"],
        "formal_launch_free_disk_reserve_gib": 72,
    }
    (HERE / "evidence/resources.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
