"""Conditional full-seed cost from the four bounded LC-Rec Qwen3 routes."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import statistics
import sys
import time
import traceback

import numpy

from common import TASKS, sha, write


ROUTES = ["q06-L", "q06-F", "q17-L", "q17-F"]
TRAIN_ROWS = 221225
TEST_USERS = 24772
GLOBAL_BATCH = 128
WORLD = 2
EPOCHS = 4
SCENARIO_FACTOR = 1.25


class Inputs:
    def __init__(self):
        self.files = {}

    def register(self, path):
        path = Path(path).resolve()
        self.files[str(path)] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}
        return path

    def read(self, path):
        return json.loads(self.register(path).read_text())


def file_bytes(records):
    unique = {}
    for record in records:
        key = record.get("path", record.get("name"))
        assert key is not None and int(record["bytes"]) >= 0
        if key in unique:
            assert unique[key] == int(record["bytes"])
        unique[key] = int(record["bytes"])
    return sum(unique.values())


def scenario(components):
    missing = [name for name, seconds in components.items() if seconds is None]
    known = sum(seconds for seconds in components.values() if seconds is not None)
    assert all(seconds is None or math.isfinite(seconds) and seconds >= 0 for seconds in components.values())
    total = None if missing else known
    return {
        "component_seconds": components,
        "component_hours": {name: None if seconds is None else seconds / 3600 for name, seconds in components.items()},
        "known_subtotal_hours": known / 3600,
        "conditional_total_hours": None if total is None else total / 3600,
        "conditional_total_days": None if total is None else total / 86400,
        "margin_to_96h": None if total is None else 96 - total / 3600,
        "missing_components": missing,
        "is_statistical_confidence_interval": False,
    }


def nonnegative_fit(design, timings):
    design = numpy.asarray(design, dtype=float)
    timings = numpy.asarray(timings, dtype=float)
    assert design.shape == (len(timings), 3) and len(timings) >= 3
    assert numpy.all(design >= 0) and numpy.all(timings >= 0)
    best = None
    for count in range(1, 4):
        for active in itertools.combinations(range(3), count):
            values = numpy.linalg.lstsq(design[:, active], timings, rcond=None)[0]
            if numpy.any(values < -1e-12):
                continue
            coefficients = numpy.zeros(3)
            coefficients[list(active)] = numpy.maximum(values, 0)
            residual = design @ coefficients - timings
            squared = float(residual @ residual)
            if best is None or squared < best[0]:
                best = (squared, coefficients, residual)
    assert best is not None
    squared, coefficients, residual = best
    condition = float(numpy.linalg.cond(design))
    return {
        "coefficients_seconds": coefficients.tolist(),
        "features": ["one_per_synchronized_microstep", "max_rank_padded_slots/1000", "max_rank_micro_times_length_squared/1e6"],
        "active_feature_indices": numpy.flatnonzero(coefficients).tolist(),
        "rmse_seconds": math.sqrt(squared / len(timings)),
        "max_absolute_residual_seconds": float(numpy.abs(residual).max()),
        "design_condition_number": condition if math.isfinite(condition) else None,
        "raw_design": design.tolist(), "raw_forward_backward_seconds": timings.tolist(),
        "observations": len(timings),
        "method": "Nonnegative least squares by all seven active subsets; elapsed-time predictor, not FLOPs or a confidence interval",
    }


def paired_updates(ranks):
    indexed = [{row["index"]: row for row in rank.get("updates", [])} for rank in ranks]
    result = []
    for update_index in sorted(set(indexed[0]).intersection(indexed[1])):
        pair = [rows[update_index] for rows in indexed]
        if any(row.get("status") != "PASS" for row in pair):
            continue
        assert pair[0]["sample_ids"] == pair[1]["sample_ids"]
        assert pair[0]["global_counts"] == pair[1]["global_counts"]
        assert pair[0]["global_counts"]["samples"] == GLOBAL_BATCH
        assert pair[0]["kind"] == pair[1]["kind"]
        assert len(pair[0]["microsteps"]) == len(pair[1]["microsteps"])
        design, timings = [], []
        for micros in zip(pair[0]["microsteps"], pair[1]["microsteps"]):
            shape = [micro["input_shape"] for micro in micros]
            slots = max(dimensions[0] * dimensions[1] for dimensions in shape)
            quadratic = max(dimensions[0] * dimensions[1] ** 2 for dimensions in shape)
            design.append([1., slots / 1000, quadratic / 1000000])
            timings.append(max(micro["forward_seconds"] + micro["backward_seconds"] for micro in micros))
        rank_compute = [sum(micro["forward_seconds"] + micro["backward_seconds"] for micro in row["microsteps"]) for row in pair]
        compute = max(rank_compute)
        wall = max(row["seconds"] for row in pair)
        assert wall >= compute - 1e-6, (wall, compute)
        result.append({
            "index": update_index, "kind": pair[0]["kind"], "plan": pair[0]["plan"],
            "nonzero_lr": all(row["nonzero_lr"] for row in pair),
            "optimizer_states_resident_at_start": all(row.get("optimizer_states_resident_at_start", False) for row in pair),
            "seconds": wall, "samples_per_second": GLOBAL_BATCH / wall,
            "effective_tokens_per_second": pair[0]["global_counts"]["input_tokens"] / wall,
            "supervised_tokens_per_second": pair[0]["global_counts"]["shift_label_tokens"] / wall,
            "global_counts": pair[0]["global_counts"], "padding_ratio": pair[0]["padding_ratio"],
            "rank_forward_backward_seconds": rank_compute, "whole_update_compute_seconds": compute,
            "whole_update_other_seconds": max(0., wall - compute),
            "paired_max_micro_forward_backward_sum": sum(timings),
            "design": design, "timings": timings,
            "measured_phase_seconds_max_rank": {
                name: max(sum(micro[name] for micro in row["microsteps"]) for row in pair)
                for name in ["cpu_collator_seconds", "transfer_seconds", "forward_seconds", "backward_seconds", "engine_step_seconds"]
            },
            "micro_loss_weighting": pair[0].get("original_micro_loss_weighting"),
        })
    return result


def full_workload(arrays, micro, epoch_presentations, epoch_trace):
    lengths = arrays["full_tokens"].astype(numpy.int64)
    assert len(lengths) == TRAIN_ROWS and numpy.all((lengths > 0) & (lengths <= 2048))
    assert epoch_presentations >= len(lengths) and epoch_presentations % (WORLD * micro) == 0
    features = numpy.zeros(3)
    epoch_rows = []
    seeds = [42, 43, 44, 45]
    for epoch_index, seed in enumerate(seeds):
        order = numpy.random.RandomState(seed).permutation(len(lengths))
        if epoch_presentations > len(order):
            order = numpy.concatenate((order, numpy.resize(order, epoch_presentations - len(order))))
        batches = lengths[order].reshape(-1, WORLD, micro)
        if epoch_trace:
            if epoch_index >= len(epoch_trace):
                break
            consumed = epoch_trace[epoch_index]["microbatches_per_rank"]
            assert consumed <= len(batches)
            batches = batches[:consumed]
        widths = batches.max(axis=2)
        synchronized_widths = widths.max(axis=1)
        target = numpy.asarray([
            len(batches), micro * synchronized_widths.sum() / 1000,
            micro * (synchronized_widths ** 2).sum() / 1000000,
        ], dtype=float)
        features += target
        useful = int(batches.sum())
        slots = int(micro * widths.sum())
        epoch_rows.append({
            "epoch": epoch_index, "numpy_layout_seed": seed, "rank_microsteps": len(batches),
            "compute_presentations": int(batches.size), "useful_tokens": useful,
            "allocated_slots_across_ranks": slots, "padding_ratio": 1 - useful / slots,
            "synchronized_max_rank_feature_totals": target.tolist(),
        })
    return {
        "feature_totals_four_epochs": features.tolist(), "epochs": epoch_rows,
        "effective_presentations": sum(row["compute_presentations"] for row in epoch_rows),
        "layout_method": "Four fixed NumPy permutations of every original-realization row, padded with unchanged repeated original indices to observed sampler presentation count; paired-rank MAX per microstep",
        "actual_trainer_sampler_order_replayed": False,
        "boundary": "Exact full token/label distribution and sampler counts; random padding layout is a deterministic cost scenario, not the saved actual rank sampler hash",
    }


def train_budget(ranks, arrays, protocol_layout):
    updates = paired_updates(ranks)
    steady = [update for update in updates if update["kind"] == "mixed" and update["nonzero_lr"] and update["optimizer_states_resident_at_start"]]
    report = {"measured_updates": updates, "artificial_tail_used_as_average": False}
    if not steady:
        report["missing"] = "No post-warmup complete mixed global128 update with resident Adam states on both ranks"
        return None, None, report
    micro = ranks[0]["training_definition"]["micro"]
    loaders = [rank["formal_loader"] for rank in ranks]
    assert loaders[0]["formal_steps"] == loaders[1]["formal_steps"]
    assert loaders[0]["epoch_sampler_presentations_across_ranks"] == loaders[1]["epoch_sampler_presentations_across_ranks"]
    steps = loaders[0]["formal_steps"]
    if protocol_layout:
        assert protocol_layout["trainer_set_initial_training_values"]["max_steps"] == steps
        trace = protocol_layout["epoch_loop_control"]
    else:
        trace = None
    workload = full_workload(arrays, micro, loaders[0]["epoch_sampler_presentations_across_ranks"], trace)
    if protocol_layout:
        assert workload["effective_presentations"] == protocol_layout["actual_sampler_presentations_four_epochs"]
    target = numpy.asarray(workload["feature_totals_four_epochs"])
    design = [row for update in steady for row in update["design"]]
    timings = [seconds for update in steady for seconds in update["timings"]]
    fit = nonnegative_fit(design, timings)
    normalizer = sum(update["whole_update_compute_seconds"] for update in steady) / sum(timings)
    assert 0 < normalizer <= 1 + 1e-6
    unnormalized_compute = float(target @ numpy.asarray(fit["coefficients_seconds"]))
    compute = unnormalized_compute * normalizer
    sensitivity = []
    for update in steady:
        observed_features = numpy.asarray(update["design"]).sum(axis=0)
        sensitivity.append({
            "source_update": update["index"],
            "constant_only_seconds": update["whole_update_compute_seconds"] * target[0] / observed_features[0],
            "linear_only_seconds": update["whole_update_compute_seconds"] * target[1] / observed_features[1],
            "quadratic_only_seconds": update["whole_update_compute_seconds"] * target[2] / observed_features[2],
        })
    fit_plus_residual = compute + 2 * fit["rmse_seconds"] * target[0] * normalizer
    unnormalized_plus_residual = unnormalized_compute + 2 * fit["rmse_seconds"] * target[0]
    candidates = [compute, fit_plus_residual, unnormalized_compute, unnormalized_plus_residual]
    candidates.extend(value for row in sensitivity for name, value in row.items() if name.endswith("_seconds"))
    other_center = steps * statistics.mean(update["whole_update_other_seconds"] for update in steady)
    other_slow = steps * max(update["whole_update_other_seconds"] for update in steady)
    report.update({
        "steady_source_update_indices": [update["index"] for update in steady],
        "formal_steps": steps, "micro": micro, "accumulation": ranks[0]["training_definition"]["accumulation"],
        "full_workload": workload, "nonnegative_forward_backward_fit": fit,
        "paired_micro_max_to_whole_rank_normalizer": normalizer,
        "normalizer_center_assumption": "The observed ratio between MAX(rank sum F+B) and sum(microstep MAX F+B) is assumed stable only for the center scenario. Rank-alternation skew can change with full-data layouts; the conservative envelope retains the unnormalized predictions",
        "fit_compute_seconds": compute, "fit_plus_two_rmse_per_micro_seconds": fit_plus_residual,
        "unnormalized_fit_compute_seconds": unnormalized_compute,
        "unnormalized_fit_plus_two_rmse_per_micro_seconds": unnormalized_plus_residual,
        "covariate_sensitivities": sensitivity,
        "center_measured_other_seconds": other_center,
        "conservative_measured_other_seconds_before_factor": other_slow,
        "center_seconds": compute + other_center,
        "conservative_seconds": SCENARIO_FACTOR * (max(candidates) + other_slow),
        "timing_denominator": "Two synchronized ranks, MAX rank wall; never sum rank wall seconds",
        "other_cost": "Whole update wall minus its slow-rank F+B includes collator, transfer, optimizer, synchronization, guard and probe instrumentation exactly once",
        "communication": "ZeRO communication inside measured backward/step is included; separate NCCL kernel time is not profiled",
        "conservative_method": "Maximum normalized and unnormalized NNLS/residual/constant/linear/quadratic full-distribution sensitivity plus worst steady non-F+B overhead, multiplied by1.25",
        "model_speed_claim_limit": "Gradient hooks, guard, data replay, logging and other probe instrumentation remain included. These are measured harness costs; this is not a model-intrinsic speed or recommendation-quality comparison",
    })
    return report["center_seconds"], report["conservative_seconds"], report


def micro_loss_weights(samples):
    if samples is None:
        return None
    records = samples["records"]
    result = {}
    for plan_name in ["mixed_global128_0", "mixed_global128_1"]:
        sample_ids = samples["plans"][plan_name]
        labels = [records[sample_id]["metrics"]["shift_effective_label_tokens"] for sample_id in sample_ids]
        token_weights = {task: sum(count for sample_id, count in zip(sample_ids, labels) if records[sample_id]["task"] == task) / sum(labels) for task in TASKS}
        result[plan_name] = {"global_token_mean_task_weights": token_weights, "micro_mean_layouts": {}}
        for micro in [1, 2, 4, 8]:
            weights = {task: 0. for task in TASKS}
            number_of_micro_means = GLOBAL_BATCH // micro
            for begin in range(0, GLOBAL_BATCH, micro):
                current_ids = sample_ids[begin:begin + micro]
                denominator = sum(labels[begin:begin + micro])
                assert denominator > 0
                for sample_id, count in zip(current_ids, labels[begin:begin + micro]):
                    weights[records[sample_id]["task"]] += count / denominator / number_of_micro_means
            assert abs(sum(weights.values()) - 1.) < 1e-9
            result[plan_name]["micro_mean_layouts"][str(micro)] = weights
    return {
        "plans": result,
        "meaning": "Actual original rank/micro CE averaging changes task and token weighting with micro layout even at global128; different selected micro layouts are not identical optimization trajectories",
        "formal_epoch_tail_numerical_equivalence": "NOT TESTED by the full-global128 bounded updates",
    }


def validation_budget(ranks, calls):
    cases = {}
    for rank in ranks:
        for case in rank.get("validation", {}).get("cases", []):
            key = case["validation_index"]
            duration = case["forward_seconds_max_rank"] + case["cpu_collator_seconds"] + case["transfer_seconds"]
            cases[key] = max(cases.get(key, 0.), duration)
    if not cases or calls is None:
        return None, None, {"missing": "No real bounded validation timing or no verified callback count"}
    rows = ranks[0]["validation"]["formal_rows_per_evaluation"]
    multiplier = rows * calls / WORLD
    values = list(cases.values())
    return multiplier * statistics.mean(values), multiplier * max(values) * SCENARIO_FACTOR, {
        "distinct_bounded_records": len(cases), "batch_size_per_gpu": 1,
        "measured_seconds_per_record_by_index": cases, "formal_rows_per_call": rows,
        "effective_calls": calls, "dual_rank_divisor": WORLD,
        "formal_micro_batch_throughput_measured": False,
        "formula": "49544 * effective_validation_calls /2 * mean_or_max(batch1 forward+collator+transfer)",
        "conservative_factor": SCENARIO_FACTOR,
        "full_validation_and_model_selection_executed": False,
    }


def generation_budget(ranks):
    cases = {}
    walls = {}
    for rank in ranks:
        generation = rank.get("generation", {})
        for case in generation.get("cases", []):
            key = (case["prompt_id"], case["row_idx"], case["history_bucket"])
            if key in cases:
                assert cases[key]["status"] == case["status"]
            cases[key] = case
        for timing in generation.get("prompt_timings", []):
            walls[timing["prompt_id"]] = max(walls.get(timing["prompt_id"], 0.), timing["wall_seconds_max_rank"])
    prompt_rows = []
    for prompt_id in range(17):
        selected = sorted([case for key, case in cases.items() if key[0] == prompt_id], key=lambda case: (case["row_idx"], case["history_bucket"]))
        if len(selected) != 3 or any(case["status"] != "PASS" for case in selected):
            return None, None, None, None, {"missing": "All17 templates need three successful actual real-user beam20 cases", "successful_cases": sum(case["status"] == "PASS" for case in cases.values()), "distinct_cases": len(cases)}
        seconds = [case["generation_seconds"] for case in selected]
        critical_generation = max(sum(seconds[rank::WORLD]) for rank in range(WORLD))
        residual = None
        if prompt_id in walls:
            assert walls[prompt_id] >= critical_generation - 1e-6
            residual = max(0., walls[prompt_id] - critical_generation) / math.ceil(len(selected) / WORLD)
        prompt_rows.append({
            "prompt_id": prompt_id, "users": 3, "generation_seconds": seconds,
            "input_token_lengths": [case["input_tokens"] for case in selected],
            "mean_seconds": statistics.mean(seconds), "max_seconds": max(seconds),
            "template_wall_seconds": walls.get(prompt_id), "critical_rank_generation_sum_seconds": critical_generation,
            "non_generation_wall_proxy_per_critical_rank_user_seconds": residual,
        })
    multiplier = TEST_USERS / WORLD
    center = multiplier * sum(row["mean_seconds"] for row in prompt_rows)
    slow = multiplier * sum(row["max_seconds"] for row in prompt_rows) * SCENARIO_FACTOR
    complete_residuals = all(row["non_generation_wall_proxy_per_critical_rank_user_seconds"] is not None for row in prompt_rows)
    residual_center = multiplier * sum(row["non_generation_wall_proxy_per_critical_rank_user_seconds"] for row in prompt_rows) if complete_residuals else None
    residual_slow = multiplier * 17 * max(row["non_generation_wall_proxy_per_critical_rank_user_seconds"] for row in prompt_rows) * SCENARIO_FACTOR if complete_residuals else None
    return center, slow, None, None, {
        "per_prompt": prompt_rows, "test_users": TEST_USERS, "templates": 17, "beam": 20,
        "return_sequences": 20, "max_new_tokens": 10, "dual_rank_divisor": WORLD,
        "formula": "Sum over17 prompts of24772/2 * mean_or_max(actual per-user beam20 generation seconds)",
        "conservative_factor": SCENARIO_FACTOR,
        "diagnostic_harness_upper_bound": {
            "full_test_center_residual_seconds": residual_center,
            "full_test_conservative_residual_seconds": residual_slow,
            "formal_cost_component": False,
            "cause": "GPU probe constructs Allowed anew for each case and re-encodes the complete9922-item four-position catalog. Official LC-Rec constructs the callback once and caches BaseDataset.allowed_tokens. This repeated harness initialization must not be extrapolated as required per-user formal CPU processing",
        },
        "processing_cost_policy": "Use only the independent CPU replay benchmark with a cached callback and exact original preprocessing/postprocessing. Without it the required test CPU processing component remains unknown",
        "remaining_unknown": "Full-test result aggregation/durable output flush and trained-output variation are unmeasured and covered only by explicit planning reserves",
        "formal_recall_ndcg_computed": False,
    }


def cached_processing_budget(benchmark):
    if benchmark is None:
        return None, None, {"missing": "No successful cached-callback original CPU replay benchmark; old GPU harness residual is diagnostic only"}
    assert benchmark["status"] == "PASS"
    assert benchmark["callback"]["cached_across_cases"] is True
    initialization = benchmark["callback"]["initialization_seconds"]
    assert math.isfinite(initialization) and initialization >= 0
    cases = benchmark["cases"]
    assert len(cases) == 51 and len({(case["prompt_id"], case["row_idx"], case["history_bucket"]) for case in cases}) == 51
    rows = []
    for prompt_id in range(17):
        selected = [case for case in cases if case["prompt_id"] == prompt_id]
        assert len(selected) == 3
        assert all(case["status"] == "PASS" and case["exact_reference_outputs"] is True for case in selected)
        means = [case["total_per_user_seconds"] for case in selected]
        maxima = [case["conservative_per_user_seconds"] for case in selected]
        assert all(math.isfinite(mean) and 0 <= mean <= maximum for mean, maximum in zip(means, maxima))
        rows.append({
            "prompt_id": prompt_id, "mean_seconds": statistics.mean(means), "conservative_seconds": max(maxima),
            "per_case_mean_seconds": means, "per_case_max_seconds": maxima,
        })
    center = initialization + TEST_USERS / WORLD * sum(row["mean_seconds"] for row in rows)
    slow = 17 * initialization + TEST_USERS / WORLD * sum(row["conservative_seconds"] for row in rows)
    slow *= SCENARIO_FACTOR
    return center, slow, {
        "status": "PASS_CACHED_ORIGINAL_CPU_REPLAY", "route_key": benchmark["route_key"],
        "benchmark_run_commit": benchmark["run_commit"],
        "callback_initialization_seconds": initialization,
        "callback_initialization_count_center_per_rank": 1,
        "callback_initialization_count_conservative_per_rank": 17,
        "per_prompt": rows, "distinct_actual_replayed_cases": 51,
        "formula_center": "One cached callback initialization plus24772/2 times sum of17 template mean CPU per-user times",
        "formula_conservative": "(17 callback initializations plus24772/2 times sum of17 template maximum per-user repetition times)*1.25",
        "timing_policy": "Use measured total_per_user_seconds only; phase encode/collator times may overlap and are not summed again",
        "callback_generation_runtime": "Already included in GPU generate timing; CPU replay does not add or simulate it",
        "parallelism_assumption": "Two independent formal CPU data/output workers can supply the two GPUs; CPU processing is added without assuming overlap with GPU work",
        "remaining_unknown": "Complete-test output aggregation, durable flush and process contention remain bounded only by planning reserves",
    }


def semantic_boundary(audit, arrays):
    counts = audit["joint"]["counts"]
    boundary_count = audit["concat_rows"] - counts["prompt_full_token_prefix_match"]
    nonempty_eos = int((arrays["eos_only_supervision"] & ~arrays["answer_text_empty"]).sum()) if arrays is not None else None
    return {
        "semantic_status": "PARTIAL_BOUNDARY" if boundary_count else "NO_OBSERVED_BOUNDARY_MISMATCH",
        "records": audit["concat_rows"], "boundary_mismatch_records": boundary_count,
        "raw_empty_answer_records": counts["answer_text_empty"],
        "eos_only_supervision_records": counts["eos_only_supervision"],
        "nonempty_answer_eos_only_records": nonempty_eos,
        "zero_effective_supervision_records": counts["zero_supervision"],
        "cost_scope": "Computational budget of the currently retained original independently-tokenized prompt-length mask with Qwen genuine-EOS adaptation",
        "not_claimed": "Complete original answer-text supervision, a validated corrected-boundary method, trusted formal baseline or paper-matched recommendation quality",
        "future_corrected_mask_method": "NOT APPROVED OR TESTED; this budget does not validate its loss, numerical trajectory or runtime",
        "compute_budget_status_is_separate": True,
    }


def state_measurement(ranks):
    states = [rank.get("training_state_restore", {}) for rank in ranks]
    if not all(state.get("status") == "PASS" for state in states):
        return None
    size = file_bytes(record for state in states for record in state.get("files", []))
    if size <= 0:
        return None
    return {"bytes": size, "save_seconds": max(state["save_seconds"] for state in states),
            "reload_seconds": max(state["reload_seconds"] for state in states),
            "scope": "Real disk save/load into the existing ZeRO engine; a new-process resume and actual Trainer best-model load are NOT TESTED"}


def checkpoint_budget(ranks, cpu, protocol_layout, donor):
    saves = [rank.get("save_reload", {}) for rank in ranks]
    model_valid = all(save.get("status") == "PASS" for save in saves)
    checkpoint_calls = None if protocol_layout is None else protocol_layout["scheduled_full_checkpoint_calls"]
    final_calls = None if protocol_layout is None else protocol_layout["final_explicit_model_save_calls"]
    best_calls = None if protocol_layout is None else protocol_layout["best_model_restore_calls"]
    report = {"full_checkpoint_calls": checkpoint_calls, "final_model_save_calls": final_calls, "best_restore_calls": best_calls}
    if not model_valid or checkpoint_calls is None:
        report["missing"] = "Real model/adapter save+strict reload or source-verified checkpoint count missing"
        return None, None, None, None, report
    model_save = max(save["save_seconds"] + save.get("training_model_to_cpu_seconds", 0.) for save in saves)
    model_restore = max(save["reload_seconds"] for save in saves)
    model_bytes = file_bytes(record for save in saves for record in save.get("saved_files", []))
    route = ranks[0]["route"]
    if route == "F":
        physical = cpu["model"]["expanded_unique_parameters"]
        trainable = physical
    else:
        physical = cpu["lora_static"]["physical_unique_parameters"]
        trainable = cpu["lora_static"]["trainable_parameters"]
    static_bytes = 2 * physical + 12 * trainable
    measured = state_measurement(ranks)
    if measured:
        state = {**measured, "source": "This exact route's real bounded full-state disk save/load", "proxy": False}
    elif donor:
        ratio = static_bytes / donor["static_bytes"]
        state = {
            "bytes": static_bytes, "save_seconds": donor["measurement"]["save_seconds"] * ratio,
            "reload_seconds": donor["measurement"]["reload_seconds"] * ratio,
            "source": donor["route_key"], "source_run_commit": donor["run_commit"],
            "static_byte_ratio_to_donor": ratio, "proxy": True,
            "assumption": "Linear byte scaling from the measured Q06-F state serialization/restore; not a measured save or verified resume of this route",
        }
    else:
        state = None
    report.update({
        "model_save_with_transfer_seconds_each": model_save, "model_fresh_reload_seconds": model_restore,
        "model_or_adapter_serialized_bytes": model_bytes, "state_static_bytes": static_bytes,
        "state_static_formula": "2*physical_unique_parameters +12*trainable_parameters: BF16 model plus global FP32 master and Adam moments; metadata/layout/storage sharing can differ",
        "state": state, "full_state_save_and_model_save_are_distinct": True,
        "model_save_calls_total": checkpoint_calls + final_calls,
        "formal_retention_payload_scenario_bytes": checkpoint_calls * (static_bytes + model_bytes) + final_calls * model_bytes,
        "formal_retention_is_not_this_preflight_disk_usage": True,
    })
    if state is None:
        report["missing"] = "Full training-state I/O has no successful real donor; never substitute adapter-only save or zero"
        return None, None, None, None, report
    checkpoint = checkpoint_calls * state["save_seconds"] + (checkpoint_calls + final_calls) * model_save
    best = best_calls * max(model_restore, state["reload_seconds"])
    report["best_restore_proxy"] = "Maximum fresh strict model/adapter reload and full-state restore proxy; actual Trainer DeepSpeed best-model path is not executed"
    return checkpoint, checkpoint * SCENARIO_FACTOR, best, best * SCENARIO_FACTOR, report


def event_initialization(paths, ranks, inputs):
    starts = []
    for path in paths:
        event_path = path.parent / ("events_rank" + str(len(starts)) + ".jsonl")
        if event_path.exists():
            inputs.register(event_path)
            entries = [json.loads(line) for line in event_path.read_text().splitlines() if line.strip()]
            initial = next((entry["elapsed_seconds"] for entry in entries if entry.get("stage") == "load_model"), None)
            starts.append(initial)
        else:
            starts.append(None)
    if not all("model_load_resize_wrap_seconds" in rank and "deepspeed_initialization_seconds" in rank for rank in ranks):
        return None, None, {"missing": "Model or ZeRO initialization timing missing"}
    bootstrap = max(starts) if all(value is not None for value in starts) else 0.
    measured = bootstrap + max(rank["model_load_resize_wrap_seconds"] for rank in ranks) + max(rank["deepspeed_initialization_seconds"] for rank in ranks)
    cold = max((update["seconds"] for update in paired_updates(ranks) if update["kind"] == "mixed" and not update["optimizer_states_resident_at_start"]), default=0.)
    return measured + cold + 300, (measured + cold) * SCENARIO_FACTOR + 600, {
        "measured_bootstrap_seconds": bootstrap if all(value is not None for value in starts) else None,
        "measured_model_and_zero_seconds": measured - bootstrap,
        "cold_update_allowance_seconds": cold,
        "cold_update_is_extra_planning_reserve_not_additional_measured_initialization": True,
        "unmeasured_full_cpu_dataset_setup_allowance_seconds_center": 300,
        "unmeasured_full_cpu_dataset_setup_allowance_seconds_conservative": 600,
        "boundary": "Only named bootstrap/model/ZeRO fields are measured initialization. A duplicated cold update and five/ten minute CPU setup are explicit planning reserves, not measurements of a complete formal initializer. Full formal CPU preparation and new-process recovery are not executed",
    }


def classify(row):
    if not row.get("has_nonzero_update"):
        return "BLOCKED", "No verified complete nonzero-LR dual-rank update"
    conservative = row["conservative"]
    if conservative["known_subtotal_hours"] > 96 and row["center"]["component_seconds"]["training"] is not None and row["center"]["component_seconds"]["test_generation"] is not None:
        return "UNLIKELY_WITHIN_96H", "The conservative scenario for the selected micro, retained original mask and measured instrumented implementation exceeds96h before missing costs. Model-intrinsic speed and a lean unmeasured formal loop are not established by this result"
    required = row.get("core_training_status") == "PASS" and row.get("long_tail_status") == "PASS" and row.get("save_reload_status") == "PASS" and row.get("generation_status") == "PASS"
    if required and conservative["conditional_total_hours"] is not None and conservative["conditional_total_hours"] <= 72:
        return "LIKELY_WITHIN_96H", "All essential bounded chains and full-state I/O measurement/proxy present; conservative conditional total leaves at least24h margin"
    return "INCONCLUSIVE", "Missing required chain/cost evidence, or conservative margin below24h; estimates are conditional engineering scenarios"


def discover(evidence, inputs):
    cpu = {}
    cpu_attempts = []
    for path in sorted(evidence.glob("jobs/*/output/audit/audit.json")):
        report = inputs.read(path)
        key = report.get("model_identity", {}).get("key")
        if key not in ["q06", "q17"]:
            continue
        cpu_attempts.append({"path": str(path), "key": key, "status": report.get("status"), "run_commit": report.get("run_commit")})
        if str(report.get("status", "")).startswith("PASS"):
            candidate = (report.get("finished_utc", ""), str(path))
            if key not in cpu or candidate > cpu[key][0]:
                cpu[key] = (candidate, path, report)
    attempts = {route: [] for route in ROUTES}
    for path in sorted(evidence.glob("jobs/*/output/probe/rank0.json")):
        first = inputs.read(path)
        key = first.get("route_key")
        if key not in attempts:
            continue
        second_path = path.with_name("rank1.json")
        second = inputs.read(second_path) if second_path.exists() else None
        attempts[key].append({"paths": [path, second_path], "ranks": [first, second], "time": first.get("finished_utc", ""), "job_id": first.get("job_id")})
    selected = {}
    for key, values in attempts.items():
        complete = [value for value in values if all(value["ranks"])]
        if complete:
            selected[key] = max(complete, key=lambda value: (value["time"], str(value["paths"][0])))
    protocols = []
    for path in sorted(evidence.glob("jobs/*/output/protocol/protocol.json")):
        report = inputs.read(path)
        if report.get("status") == "PASS_CPU_PROTOCOL_ONLY":
            protocols.append((report.get("observed_utc", ""), str(path), report))
    protocol = max(protocols, default=(None, None, None))[2]
    benchmarks = {}
    for route_key in ROUTES:
        for path in sorted(evidence.rglob(route_key + ".json")):
            benchmark = inputs.read(path)
            if benchmark.get("route_key") != route_key or benchmark.get("status") != "PASS" or "callback" not in benchmark or "cases" not in benchmark:
                continue
            candidate = (benchmark.get("finished_utc", benchmark.get("observed_utc", "")), str(path))
            if route_key not in benchmarks or candidate > benchmarks[route_key][0]:
                benchmarks[route_key] = (candidate, path, benchmark)
    return cpu, cpu_attempts, attempts, selected, protocol, benchmarks


def estimate(evidence, run_commit):
    inputs = Inputs()
    cpu, cpu_attempts, attempts, selected, protocol, benchmarks = discover(evidence, inputs)
    donor = None
    if "q06-F" in selected and "q06" in cpu:
        ranks = selected["q06-F"]["ranks"]
        measured = state_measurement(ranks)
        if measured:
            parameters = cpu["q06"][2]["model"]["expanded_unique_parameters"]
            donor = {"route_key": "q06-F", "run_commit": ranks[0]["run_commit"], "measurement": measured, "static_bytes": 14 * parameters}
    result = {
        "calculation_run_commit": run_commit, "calculation_script": {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)},
        "calculation_environment": {
            "executable": sys.executable, "python": sys.version, "numpy": numpy.__version__,
            "cwd": str(Path.cwd()), "argv": sys.argv,
        },
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "One Instruments seed, original six tasks and930 LC-Rec SID,4epochs/global128, sampled SeqRec validation,24772 users times17 prompts beam20; no formal run launched",
        "cpu_attempts": cpu_attempts, "routes": {}, "full_state_donor": donor,
        "semantic_status_all_routes": "PARTIAL_BOUNDARY",
        "supervision_scope": "Current retained original prompt-length mask only:29574 boundary mismatches and979 EOS-only records, including474 originally nonempty answers. No corrected-supervision method or complete six-task semantic PASS is established",
        "compute_budget_judgment_is_not_semantic_or_baseline_approval": True,
        "compute_budget_scope": "Current selected per-route micro/accumulation and preserved original prompt-length mask, including measured gradient-hook, guard, replay and logging instrumentation. Lean formal-kernel throughput and future boundary-corrected supervision are unmeasured",
        "method_boundary": [
            "Center/conservative are explicit engineering scenarios, not statistical confidence intervals or runtime guarantees",
            "Current full-realization arrays include every original auxiliary mismatch, empty answer and EOS-only record; technical update eligibility does not redefine a formal training corpus",
            "Only post-warmup true mixed global128 updates calibrate average training; artificial mixed-tail updates are separate capacity/latency evidence",
            "MAX of synchronized ranks is wall time; shared generation cases are deduplicated, and formal distinct users divide by two GPUs exactly once",
            "Full-state proxies require a successful measured Q06-F save/restore; missing required costs remain null",
            "Cold update, CPU setup, callback/output handling and explicit operating reserves are included; no unmeasured required cost is silently set to zero",
            "Per-case full-catalog Allowed initialization in the GPU harness is diagnostic overhead. Formal CPU test cost uses only separately measured cached-callback replay; missing replay evidence keeps that component null",
        ],
        "unquantified_costs": [
            "Actual future full-epoch prompt/query/preference random realizations and long-run padding variation beyond the fixed audited realization",
            "Exact short-tail Trainer plus DeepSpeed gradient normalization and new-process checkpoint resume",
            "Durable fsync, full-test cross-rank aggregation and output storage at421124 user-template pairs",
            "Actual lowest-eval-loss checkpoint selection and Trainer DeepSpeed best-model loading",
            "Thermal, allocator fragmentation, shared CPU/I/O variation across a complete seed; capacity and downtime are not guaranteed",
        ],
        "formal_data_status": "BLOCKED: preexisting12233 training/7473 testing auxiliary mismatches and unresolved preference-generation provenance; technical feasibility is not formal recommendation reproduction",
    }
    for route_key in ROUTES:
        label = route_key.upper()
        all_attempts = attempts[route_key]
        attempt_rows = [{"job_id": value["job_id"], "paths": [str(path) for path in value["paths"]], "statuses": [rank.get("status") if rank else "MISSING_RANK" for rank in value["ranks"]], "run_commits": [rank.get("run_commit") if rank else None for rank in value["ranks"]]} for value in all_attempts]
        row = {"route": label, "attempts": attempt_rows, "semantic_status": "PARTIAL_BOUNDARY"}
        result["routes"][label] = row
        if route_key not in selected or route_key.split("-")[0] not in cpu:
            row.update(status="BLOCKED", reason="Missing completed dual-rank GPU evidence or successful actual-model CPU tokenization", center=scenario({"full_seed": None}), conservative=scenario({"full_seed": None}))
            continue
        try:
            chosen = selected[route_key]
            ranks = chosen["ranks"]
            key = route_key.split("-")[0]
            audit_path, audit = cpu[key][1:]
            samples_path = audit_path.parent / "samples.json"
            samples = inputs.read(samples_path) if samples_path.exists() else None
            arrays_path = audit_path.parent / "workload.npz"
            arrays = numpy.load(inputs.register(arrays_path), allow_pickle=False) if arrays_path.exists() else None
            definition = ranks[0]["training_definition"]
            assert all(rank["training_definition"] == definition for rank in ranks)
            assert definition["global_batch"] == 128 and definition["epochs_for_estimate_only"] == 4 and definition["temperature"] == 1
            assert ranks[0]["run_commit"] == ranks[1]["run_commit"]
            layout = next((value for value in protocol["layouts"] if value["micro"] == definition["micro"]), None) if protocol else None
            calls = layout["effective_validation_calls"] if layout else None
            components = {name: None for name in ["training", "validation", "checkpoint", "best_restore", "test_generation", "test_processing_proxy", "initialization", "operating_reserve"]}
            slow_components = dict(components)
            if arrays is not None:
                components["training"], slow_components["training"], row["training"] = train_budget(ranks, arrays, layout)
            else:
                row["training"] = {"measured_updates": paired_updates(ranks), "missing": "Full workload.npz not collected; full-distribution calibrated training cost remains unknown"}
            components["validation"], slow_components["validation"], row["validation"] = validation_budget(ranks, calls)
            components["test_generation"], slow_components["test_generation"], components["test_processing_proxy"], slow_components["test_processing_proxy"], row["test"] = generation_budget(ranks)
            benchmark = benchmarks.get(route_key)
            if benchmark:
                assert benchmark[2]["probe_run_commit"] == ranks[0]["run_commit"], "CPU replay benchmark belongs to a different GPU source run"
            components["test_processing_proxy"], slow_components["test_processing_proxy"], row["test_cpu_processing"] = cached_processing_budget(benchmark[2] if benchmark else None)
            if benchmark:
                row["test_cpu_processing"]["benchmark_path"] = str(benchmark[1])
            components["checkpoint"], slow_components["checkpoint"], components["best_restore"], slow_components["best_restore"], row["checkpoint"] = checkpoint_budget(ranks, audit, layout, donor)
            components["initialization"], slow_components["initialization"], row["initialization"] = event_initialization(chosen["paths"], ranks, inputs)
            subtotal = sum(value for value in components.values() if value is not None)
            slow_subtotal = sum(value for value in slow_components.values() if value is not None)
            components["operating_reserve"] = .05 * subtotal + 600
            slow_components["operating_reserve"] = .10 * slow_subtotal + 1800
            row.update({
                "selected_job_id": chosen["job_id"], "run_commit": ranks[0]["run_commit"],
                "compute_budget_scope": "This exact selected micro/accumulation and retained-mask instrumented implementation; no model-intrinsic speed or formal baseline adoption conclusion",
                "selection_rule": "Latest timestamped attempt with both rank reports; no fastest-attempt selection; prior failures retained",
                "training_definition": definition, "formal_loader": ranks[0].get("formal_loader"),
                "rank_statuses": [rank.get("status") for rank in ranks],
                "core_training_status": "PASS" if all(rank.get("core_training_status") == "PASS" for rank in ranks) else "PARTIAL",
                "long_tail_status": "PASS" if all(rank.get("long_tail_zero_training", {}).get("status") == "PASS" for rank in ranks) else "NOT_FULLY_VERIFIED",
                "long_tail_evidence": ranks[0].get("long_tail_zero_training"),
                "save_reload_status": "PASS" if all(rank.get("save_reload", {}).get("status") == "PASS" for rank in ranks) else "PARTIAL",
                "generation_status": "PASS" if all(rank.get("generation", {}).get("status") == "PASS" for rank in ranks) else "PARTIAL",
                "has_nonzero_update": any(update["nonzero_lr"] for update in paired_updates(ranks)),
                "measured_gpu_and_host_peaks": [rank.get("training_peak", {}) for rank in ranks],
                "token_lengths": {"per_task": audit["per_task"], "joint": audit["joint"]},
                "supervision_semantics": semantic_boundary(audit, arrays),
                "micro_loss_weighting": micro_loss_weights(samples),
                "center": scenario(components), "conservative": scenario(slow_components),
                "reserves": {"center": "5% of quantified subtotal plus10min", "conservative": "1.25 factor within measured component scenarios, plus10% subtotal and30min operating reserve", "unmeasured_cpu_setup": "Additional5min center/10min conservative initialization allowances"},
                "confidence": "MODERATE_CONDITIONAL" if all(value is not None for value in components.values()) else "LOW_INCOMPLETE",
            })
            row["status"], row["reason"] = classify(row)
            row["compute_budget_status"] = row["status"]
            if arrays is not None:
                arrays.close()
        except Exception:
            row.update(status="INCONCLUSIVE", reason="Cost calculation encountered an evidence/schema inconsistency; other route calculations continue", calculation_error=traceback.format_exc())
            row.setdefault("center", scenario({"full_seed": None}))
            row.setdefault("conservative", scenario({"full_seed": None}))
    result["inputs"] = list(inputs.files.values())
    return result


def self_test():
    design = [[1., slots, slots * slots] for slots in [1., 2., 3., 4., 5., 6.]]
    coefficients = numpy.asarray([.01, .02, .003])
    timings = numpy.asarray(design) @ coefficients
    fitted = nonnegative_fit(design, timings)
    assert numpy.allclose(fitted["coefficients_seconds"], coefficients, rtol=1e-8, atol=1e-10)
    assert file_bytes([{"path": "same", "bytes": 9}, {"path": "same", "bytes": 9}, {"path": "other", "bytes": 4}]) == 13
    assert scenario({"measured": 3600, "unknown": None})["conditional_total_hours"] is None
    assert scenario({"measured": 3600, "unknown": None})["known_subtotal_hours"] == 1
    assert math.ceil(TRAIN_ROWS / GLOBAL_BATCH) * EPOCHS == 6916
    measured_ranks = []
    for rank_index in range(2):
        micros = [{"input_shape": [1, 10 + rank_index], "forward_seconds": 1. + rank_index, "backward_seconds": 1., "cpu_collator_seconds": .1, "transfer_seconds": .1, "engine_step_seconds": .1}]
        measured_ranks.append({"updates": [{"index": 1, "sample_ids": ["fixed"], "global_counts": {"samples": 128, "input_tokens": 1280, "shift_label_tokens": 640}, "status": "PASS", "kind": "mixed", "plan": "fixed", "nonzero_lr": True, "optimizer_states_resident_at_start": True, "seconds": 4., "padding_ratio": 0., "microsteps": micros}]})
    paired = paired_updates(measured_ranks)
    assert paired[0]["seconds"] == 4. and paired[0]["whole_update_compute_seconds"] == 3.
    assert paired[0]["whole_update_other_seconds"] == 1.
    assert paired[0]["samples_per_second"] == 32.
    generation_cases = [
        {"prompt_id": prompt_id, "row_idx": row_index, "history_bucket": str(row_index),
         "status": "PASS", "generation_seconds": float(row_index + 1), "input_tokens": 100 + row_index}
        for prompt_id in range(17) for row_index in range(3)
    ]
    generation_ranks = [{"generation": {"cases": generation_cases,
                                       "prompt_timings": [{"prompt_id": prompt_id, "wall_seconds_max_rank": 5.} for prompt_id in range(17)]}}] * 2
    generation_center, generation_slow, processing_center, processing_slow, generation = generation_budget(generation_ranks)
    assert generation_center == TEST_USERS / 2 * 17 * 2
    assert generation_slow == TEST_USERS / 2 * 17 * 3 * SCENARIO_FACTOR
    assert processing_center is None and processing_slow is None
    assert generation["diagnostic_harness_upper_bound"]["full_test_center_residual_seconds"] == TEST_USERS / 2 * 17 * .5
    assert cached_processing_budget(None)[:2] == (None, None)
    processing_cases = [
        {"prompt_id": prompt_id, "row_idx": row_index, "history_bucket": str(row_index),
         "status": "PASS", "exact_reference_outputs": True,
         "total_per_user_seconds": .02, "conservative_per_user_seconds": .03}
        for prompt_id in range(17) for row_index in range(3)
    ]
    benchmark = {
        "route_key": "q06-L", "status": "PASS", "run_commit": "0" * 40,
        "callback": {"initialization_seconds": 1., "cached_across_cases": True},
        "cases": processing_cases,
    }
    processing_center, processing_slow, processing = cached_processing_budget(benchmark)
    assert math.isclose(processing_center, 1. + TEST_USERS / 2 * 17 * .02)
    assert math.isclose(processing_slow, (17. + TEST_USERS / 2 * 17 * .03) * SCENARIO_FACTOR)
    presentations = math.ceil(TRAIN_ROWS / 4) * 4
    workload = full_workload({"full_tokens": numpy.full(TRAIN_ROWS, 10)}, 2, presentations, None)
    assert workload["effective_presentations"] == presentations * 4
    assert workload["feature_totals_four_epochs"][0] == presentations / 4 * 4
    fake_ids = ["sample" + str(position) for position in range(128)]
    fake_samples = {
        "plans": {"mixed_global128_0": fake_ids, "mixed_global128_1": fake_ids},
        "records": {sample_id: {"task": "seqrec" if position % 2 == 0 else "preferenceobtain",
                                "metrics": {"shift_effective_label_tokens": 5 if position % 2 == 0 else 50}}
                    for position, sample_id in enumerate(fake_ids)},
    }
    weights = micro_loss_weights(fake_samples)["plans"]["mixed_global128_0"]
    assert abs(weights["micro_mean_layouts"]["1"]["seqrec"] - .5) < 1e-9
    assert abs(weights["micro_mean_layouts"]["2"]["seqrec"] - 1 / 11) < 1e-9
    print(json.dumps({"status": "PASS_SELF_TEST", "nonnegative_fit": True, "missing_cost_retained": True, "duplicate_file_bytes_not_double_counted": True, "two_rank_wall_max_not_sum": True, "ceil_schedule_four_epochs": 6916, "all17_template_dual_rank_generation_denominator": True, "full_random_layout_preserves_sampler_presentations": True, "micro_mean_differs_from_global_token_mean": True, "uncached_harness_residual_not_formal_cost": True, "cached_callback_once_or17_not_per_user": True, "missing_cached_benchmark_retains_unknown": True}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-commit")
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_test:
        self_test()
        return 0
    if arguments.evidence is None or arguments.output is None or arguments.run_commit is None:
        parser.error("--evidence, --output and --run-commit are required unless --self-test")
    assert len(arguments.run_commit) == 40 and all(character in "0123456789abcdef" for character in arguments.run_commit)
    assert not arguments.output.exists(), "Refusing to overwrite existing cost evidence"
    result = estimate(arguments.evidence.resolve(), arguments.run_commit)
    result["argv"] = sys.argv
    write(arguments.output, result)
    print(json.dumps({route: {"status": report["status"], "center_hours": report["center"]["conditional_total_hours"], "conservative_hours": report["conservative"]["conditional_total_hours"], "missing": report["center"]["missing_components"]} for route, report in result["routes"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
