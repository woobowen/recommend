#!/usr/bin/env python3
"""Describe padding and original token-mean CE grouping; never change a loss.

The mixed plans contain diagnostic safe training-period records. Full-epoch
workload uses every stored fixed-realization record, including mismatched and
zero-supervision records. Its explicit NumPy permutation is a cost diagnostic,
not a claim about the actual Trainer/Accelerate sampling or epoch randomness.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time


TASKS = ["seqrec", "item2index", "index2item", "fusionseqrec", "itemsearch", "preferenceobtain"]
MICROS = [1, 2, 4, 8]


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def distribution(values, np):
    a = np.asarray(values, dtype=np.float64)
    if not len(a):
        return {"count": 0}
    q = np.percentile(a, [10, 50, 90, 99], method="linear")
    return {"count": len(a), "min": float(a.min()), "p10": float(q[0]), "p50": float(q[1]),
            "p90": float(q[2]), "p99": float(q[3]), "max": float(a.max()),
            "mean": float(a.mean()), "sum": float(a.sum())}


def layout_workload(lengths, label_counts, micro, np, include_array=False):
    """Consecutive per-rank microbatches, world2; repeat prefix for final pair."""
    n = len(lengths)
    pair_size = 2 * micro
    padded_n = math.ceil(n / pair_size) * pair_size
    duplicate_n = padded_n - n
    expanded_lens = list(lengths) + list(lengths[:duplicate_n])
    expanded_labels = list(label_counts) + list(label_counts[:duplicate_n])
    if len(expanded_lens) != padded_n:
        raise ValueError("Diagnostic requires at least 2*micro records")
    a = np.asarray(expanded_lens, dtype=np.int64).reshape(-1, 2, micro)
    labels = np.asarray(expanded_labels, dtype=np.int64).reshape(-1, 2, micro)
    rank_max = a.max(axis=2)
    rank_useful = a.sum(axis=2)
    rank_slots = rank_max * micro
    rank_quadratic = rank_max * rank_max * micro
    pair_slots = rank_slots.sum(axis=1)
    pair_useful = rank_useful.sum(axis=1)
    rank_label_sums = labels.sum(axis=2)
    total_slots = int(rank_slots.sum())
    useful = int(a.sum())
    result = {"world_size": 2, "micro_batch_per_rank": micro, "target_accumulation_for_global128": 64 // micro,
              "original_records": n, "presented_records_after_diagnostic_pair_padding": padded_n,
              "tail_duplicate_records": duplicate_n, "rank_micro_steps": a.shape[0],
              "global128_complete_groups_by_this_layout": a.shape[0] // (64 // micro),
              "remaining_rank_micro_steps_after_complete_groups": a.shape[0] % (64 // micro),
              "useful_token_sum_including_tail_duplicates": useful,
              "gpu_token_slots_sum_both_ranks": total_slots,
              "gpu_attention_quadratic_sum_both_ranks": int(rank_quadratic.sum()),
              "synchronized_wall_token_slots_max_rank_sum": int(rank_slots.max(axis=1).sum()),
              "synchronized_wall_attention_quadratic_max_rank_sum": int(rank_quadratic.max(axis=1).sum()),
              "padding_token_sum": total_slots - useful,
              "padding_ratio": 1 - useful / total_slots if total_slots else 0.,
              "per_pair_padding_ratio": distribution(1 - pair_useful / pair_slots, np),
              "per_rank_micro_max_length": distribution(rank_max.reshape(-1), np),
              "per_pair_slowest_rank_max_length": distribution(rank_max.max(axis=1), np),
              "zero_shift_label_rank_microbatches": int((rank_label_sums == 0).sum()),
              "label_mean_loss_nan_risk": bool((rank_label_sums == 0).any()),
              "timing_boundary": "Quadratic/linear work units are cost covariates, not elapsed seconds or measured kernel FLOPs."}
    if include_array:
        result["rank_micro_shapes"] = [[int(x) for x in row] for row in rank_max]
        result["rank_micro_shift_label_counts"] = [[int(x) for x in row] for row in rank_label_sums]
    return result


def ce_grouping(records, ids, micro, np):
    assert len(ids) == 128
    accumulation = 64 // micro
    sample_weights, token_weights = [], []
    by_task = collections.defaultdict(list)
    per_sample = []
    label_total = sum(records[sample_id]["metrics"]["shift_effective_label_tokens"] for sample_id in ids)
    for pair_step in range(accumulation):
        for rank in range(2):
            start = (pair_step * 2 + rank) * micro
            sample_ids = ids[start:start + micro]
            label_counts = [records[sample_id]["metrics"]["shift_effective_label_tokens"] for sample_id in sample_ids]
            denominator = sum(label_counts)
            if denominator <= 0:
                return {"status": "UNDEFINED_ZERO_LABEL_MICROBATCH", "rank": rank, "accumulation_step": pair_step,
                        "sample_ids": sample_ids, "boundary": "Original mean CrossEntropy over all ignored labels is undefined; no epsilon or replacement was used."}
            coefficient = 1.0 / (2 * accumulation * denominator)
            for sample_id, labels in zip(sample_ids, label_counts):
                task = records[sample_id]["task"]
                sample_weight = labels * coefficient
                sample_weights.append(sample_weight)
                token_weights.append(coefficient)
                by_task[task].append({"sample_weight": sample_weight, "per_token_weight": coefficient,
                                      "shift_label_tokens": labels})
                per_sample.append({"sample_id": sample_id, "rank": rank, "accumulation_step": pair_step,
                                   "micro_shift_label_tokens": denominator, "sample_shift_label_tokens": labels,
                                   "sample_loss_weight": sample_weight, "per_label_token_loss_weight": coefficient})
    task_summary = {}
    for task in TASKS:
        rows = by_task[task]
        count = len(rows)
        labels = sum(row["shift_label_tokens"] for row in rows)
        task_summary[task] = {"samples": count, "shift_label_tokens": labels,
                              "original_grouped_ce_task_weight": sum(row["sample_weight"] for row in rows),
                              "reference_global_token_mean_weight": labels / label_total,
                              "reference_global_sample_mean_weight": count / len(ids),
                              "sample_loss_weight_distribution": distribution([row["sample_weight"] for row in rows], np),
                              "per_token_loss_weight_distribution": distribution([row["per_token_weight"] for row in rows], np)}
    total = sum(sample_weights)
    assert abs(total - 1.) < 1e-12, total
    return {"status": "QUANTIFIED_DIAGNOSTIC", "world_size": 2, "micro_batch_per_rank": micro,
            "accumulation": accumulation, "effective_batch": 128,
            "sum_sample_loss_weights": total, "total_shift_label_tokens": label_total,
            "sample_loss_weights": distribution(sample_weights, np),
            "per_label_token_loss_weights": distribution(token_weights, np),
            "maximum_to_minimum_per_label_token_weight_ratio": max(token_weights) / min(token_weights),
            "per_task": task_summary, "per_sample": per_sample,
            "formula": "loss=(1/(world_size*accumulation))*sum_rank,micro(mean_nonignored_shift_label_token_CE). token_weight=1/(2*A*micro_label_count). No alternative loss was executed."}


def run(records_path, samples_path):
    import numpy as np

    lengths, labels, tasks = [], [], []
    with gzip.open(records_path, "rt", encoding="utf-8") as stream:
        for i, line in enumerate(stream):
            row = json.loads(line)
            assert row["concat_index"] == i
            lengths.append(row["full_tokens"])
            labels.append(row["shift_effective_label_tokens"])
            tasks.append(row["task"])
    assert len(lengths) == 221225
    samples = json.loads(samples_path.read_text())
    for sample_id, row in samples["records"].items():
        i = row["concat_index"]
        assert row["sample_id"] == sample_id
        assert row["metrics"]["full_tokens"] == lengths[i]
        assert row["metrics"]["shift_effective_label_tokens"] == labels[i]
        assert row["task"] == tasks[i]
    permutation = np.random.RandomState(42).permutation(len(lengths)).tolist()
    permuted_lengths = [lengths[i] for i in permutation]
    permuted_labels = [labels[i] for i in permutation]
    full = {str(micro): layout_workload(permuted_lengths, permuted_labels, micro, np) for micro in MICROS}
    mixed = {}
    for plan_name in ["mixed_global128_0", "mixed_global128_1"]:
        ids = samples["plans"][plan_name]
        assert len(ids) == 128
        assert all(samples["records"][sample_id]["technical_update_eligible"] for sample_id in ids)
        local_lengths = [samples["records"][sample_id]["metrics"]["full_tokens"] for sample_id in ids]
        local_labels = [samples["records"][sample_id]["metrics"]["shift_effective_label_tokens"] for sample_id in ids]
        mixed[plan_name] = {str(micro): {"workload": layout_workload(local_lengths, local_labels, micro, np, include_array=True),
                                        "ce_weighting": ce_grouping(samples["records"], ids, micro, np)} for micro in MICROS}
    return {"status": "QUANTIFIED_DIAGNOSTIC", "seed": 42, "numpy_version": np.__version__,
            "source_files": {"records": {"path": str(records_path), "bytes": records_path.stat().st_size, "sha256": sha256_file(records_path)},
                             "samples": {"path": str(samples_path), "bytes": samples_path.stat().st_size, "sha256": sha256_file(samples_path)}},
            "full_records": len(lengths), "task_counts": dict(collections.Counter(tasks)),
            "full_dataset_zero_shift_label_records": sum(value == 0 for value in labels),
            "full_epoch_fixed_realization": full, "mixed_global128": mixed,
            "permutation": "np.random.RandomState(42).permutation(221225); original fixed-realization records; consecutive rank0 micro then rank1 micro; final pair repeats the permutation prefix to 2*micro only",
            "permutation_sha256_int64_little_endian": hashlib.sha256(np.asarray(permutation, dtype="<i8").tobytes()).hexdigest(),
            "boundary": ["Not an implementation or verification of actual Trainer/Accelerate sampler/tail behavior",
                         "One saved stochastic Dataset prompt/query/preference realization, not all four epochs or all possible prompts",
                         "All raw records, mismatches, truncation and zero-label records remain in the full workload",
                         "Safe technical mixed plans can have a different auxiliary within-task length distribution",
                         "CE weights quantify the original microbatch token-mean objective; no loss normalization was changed",
                         "Linear/quadratic units need measured representative kernels to infer time; neither is elapsed time"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an existing output")
    started = time.monotonic()
    report = run(args.records.resolve(), args.samples.resolve())
    report.update(argv=sys.argv, wall_seconds=time.monotonic() - started,
                  created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve()),
                      "wall_seconds": report["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
