#!/usr/bin/env python3
"""CPU acceptance comparison of uninterrupted versus cross-process DS resume.

All stored state is compared recursively. Discrete state, scheduler, counters
and every RNG stream must match exactly. Floating model/optimizer tensors are
first checked for exact equality, then against the fixed, predeclared criterion
abs(resumed-reference) <= 1e-6 + 1e-3 * abs(reference). The tolerance is deliberately
stricter than a typical single BF16 ULP at ordinary magnitudes; nondeterminism
may therefore fail this acceptance. It is never widened after seeing results.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import socket
import sys
import time
import traceback


ATOL, RTOL = 1e-6, 1e-3
CHUNK_ELEMENTS = 1024 * 1024
RESUME_INTERVAL_START = 2
EXPECTED_TRAINABLE_TENSORS = 450
MASTER_KEYS = {
    "single_partition_of_fp32_groups", "fp32_flat_groups", "fp32_groups_flat",
    "fp32_partitioned_groups_flat", "fp32_groups", "master_flat", "fp32_master",
    "master_params", "master_parameters", "master_weights",
}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def state_root(directory):
    root = Path(directory).resolve()
    direct = (root / "acceptance_state_0.pt").is_file()
    nested = (root / "training/acceptance_state_0.pt").is_file()
    if direct and nested:
        raise ValueError(f"Ambiguous direct and training/ acceptance state: {root}")
    if not direct and not nested:
        raise FileNotFoundError(f"No acceptance_state_0.pt in {root} or its training/ directory")
    return root if direct else root / "training"


def key_path(path, key):
    return f"{path}[{key!r}]"


def tensor_leaves(value, torch, path="state"):
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from tensor_leaves(child, torch, key_path(path, key))
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            yield from tensor_leaves(child, torch, key_path(path, index))


def named_nodes(value, path="state"):
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = key_path(path, key)
            yield str(key).lower(), child_path, child
            yield from named_nodes(child, child_path)
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            yield from named_nodes(child, key_path(path, index))


class Compare:
    def __init__(self, torch, numpy):
        self.torch, self.np = torch, numpy
        self.failures, self.tensors, self.arrays = [], [], []
        self.exact_scalar_leaves = 0

    def fail(self, path, reason, **details):
        self.failures.append({"path": path, "reason": reason, **details})

    def tensor(self, reference, resumed, path, exact):
        torch = self.torch
        row = {"path": path, "reference_shape": list(reference.shape), "resumed_shape": list(resumed.shape),
               "reference_dtype": str(reference.dtype), "resumed_dtype": str(resumed.dtype),
               "elements_reference": reference.numel(), "elements_resumed": resumed.numel(),
               "requires_exact": exact or not reference.is_floating_point()}
        self.tensors.append(row)
        if reference.device.type != "cpu" or resumed.device.type != "cpu":
            self.fail(path, "Comparator must load every tensor on CPU")
            return
        if reference.layout != torch.strided or resumed.layout != torch.strided:
            self.fail(path, "Unsupported tensor layout; cannot silently omit state")
            return
        reference, resumed = reference.detach(), resumed.detach()
        # Finite checks are required even if shapes or dtypes disagree.
        finite = []
        for tensor in (reference, resumed):
            flattened = tensor.reshape(-1)
            finite.append(all(bool(torch.isfinite(flattened[i:i + CHUNK_ELEMENTS]).all())
                              for i in range(0, tensor.numel(), CHUNK_ELEMENTS)))
        row["reference_finite"], row["resumed_finite"] = finite
        if not all(finite):
            self.fail(path, "Nonfinite model/optimizer/state tensor", reference_finite=finite[0], resumed_finite=finite[1])
        if reference.shape != resumed.shape or reference.dtype != resumed.dtype:
            self.fail(path, "Tensor shape or dtype mismatch")
            return
        if reference.is_complex() or resumed.is_complex():
            self.fail(path, "Unexpected complex tensor state")
            return
        reference, resumed = reference.reshape(-1), resumed.reshape(-1)
        exact_equal, unequal, violations = True, 0, 0
        maximum_abs, maximum_relative, sum_squares = 0.0, 0.0, 0.0
        for begin in range(0, reference.numel(), CHUNK_ELEMENTS):
            left, right = reference[begin:begin + CHUNK_ELEMENTS], resumed[begin:begin + CHUNK_ELEMENTS]
            same = torch.eq(left, right)
            different = int((~same).sum())
            unequal += different
            exact_equal = exact_equal and different == 0
            if not reference.is_floating_point():
                violations += different
                continue
            if not all(finite):
                continue
            left, right = left.to(torch.float64), right.to(torch.float64)
            difference = (right - left).abs()
            maximum_abs = max(maximum_abs, float(difference.max()) if difference.numel() else 0.0)
            # A stated denominator floor yields finite diagnostics at reference zero.
            relative = difference / left.abs().clamp_min(ATOL)
            maximum_relative = max(maximum_relative, float(relative.max()) if relative.numel() else 0.0)
            sum_squares += float((difference * difference).sum())
            violations += int((difference > ATOL + RTOL * left.abs()).sum())
        row.update(exact_equal=exact_equal, differing_elements=unequal,
                   outside_fixed_tolerance_elements=violations)
        if reference.is_floating_point() and all(finite):
            rms = math.sqrt(sum_squares / reference.numel()) if reference.numel() else 0.0
            if not all(math.isfinite(value) for value in (maximum_abs, maximum_relative, rms)):
                self.fail(path, "Nonfinite difference diagnostic; cannot certify comparison")
                row["difference_diagnostic_overflow"] = True
            else:
                row.update(max_abs=maximum_abs, max_relative_with_atol_denominator_floor=maximum_relative, rms=rms)
        if row["requires_exact"] and not exact_equal:
            self.fail(path, "Discrete/scheduler/RNG/counter state differs; numerical tolerance is not applicable",
                      differing_elements=unequal)
        elif violations:
            self.fail(path, "Floating tensor exceeds fixed tolerance", violating_elements=violations,
                      differing_elements=unequal)

    def compare(self, reference, resumed, path="state", exact=False):
        torch, np = self.torch, self.np
        if isinstance(reference, torch.Tensor) or isinstance(resumed, torch.Tensor):
            if not isinstance(reference, torch.Tensor) or not isinstance(resumed, torch.Tensor):
                self.fail(path, "Tensor versus non-tensor type mismatch")
            else:
                self.tensor(reference, resumed, path, exact)
            return
        if isinstance(reference, np.ndarray) or isinstance(resumed, np.ndarray):
            if not isinstance(reference, np.ndarray) or not isinstance(resumed, np.ndarray):
                self.fail(path, "NumPy array versus non-array mismatch")
                return
            finite = bool(np.isfinite(reference).all()) and bool(np.isfinite(resumed).all())
            equal = reference.shape == resumed.shape and reference.dtype == resumed.dtype and np.array_equal(reference, resumed)
            self.arrays.append({"path": path, "reference_shape": list(reference.shape), "resumed_shape": list(resumed.shape),
                                "reference_dtype": str(reference.dtype), "resumed_dtype": str(resumed.dtype),
                                "finite": finite, "exact_equal": equal})
            if not finite or not equal:
                self.fail(path, "NumPy RNG/state array differs or is nonfinite; exact match required")
            return
        if type(reference) is not type(resumed):
            self.fail(path, "State value types differ", reference_type=type(reference).__name__, resumed_type=type(resumed).__name__)
            return
        if isinstance(reference, dict):
            if reference.keys() != resumed.keys():
                self.fail(path, "Dictionary key sets differ",
                          missing_in_resumed=[repr(key) for key in reference.keys() - resumed.keys()],
                          unexpected_in_resumed=[repr(key) for key in resumed.keys() - reference.keys()])
            for key in reference.keys() & resumed.keys():
                semantic_key = str(key).lower()
                force_exact = exact or semantic_key in {"step", "global_step", "global_steps", "scheduler", "rng", "loader_generator"}
                self.compare(reference[key], resumed[key], key_path(path, key), force_exact)
            return
        if isinstance(reference, (list, tuple)):
            if len(reference) != len(resumed):
                self.fail(path, "Sequence lengths differ", reference_length=len(reference), resumed_length=len(resumed))
            for index, (left, right) in enumerate(zip(reference, resumed)):
                self.compare(left, right, key_path(path, index), exact)
            return
        # DeepSpeed optimizer state may include a concrete LossScaler object.
        # Its class must be identical and every stored field is compared exactly.
        if type(reference).__module__.startswith("deepspeed.") and hasattr(reference, "__dict__"):
            self.compare(vars(reference), vars(resumed), path + ".__dict__", exact=True)
            return
        if isinstance(reference, np.generic):
            equal = reference.dtype == resumed.dtype and bool(np.isfinite(reference)) and bool(np.isfinite(resumed)) and bool(reference == resumed)
        elif isinstance(reference, (int, float, bool, str, bytes, type(None))):
            equal = reference == resumed
            if isinstance(reference, float):
                equal = equal and math.isfinite(reference) and math.isfinite(resumed)
        else:
            self.fail(path, "Unsupported state value type; no state may be silently ignored", value_type=type(reference).__name__)
            return
        self.exact_scalar_leaves += 1
        if not equal:
            self.fail(path, "Scalar/discrete state differs; exact match required", reference=repr(reference), resumed=repr(resumed))


def inspect_schema(state, torch, label):
    required = {"trainable", "optimizer", "scheduler", "global_step", "rng"}
    if not isinstance(state, dict) or not required <= state.keys():
        raise AssertionError(label + " acceptance payload lacks required state sections")
    if type(state["global_step"]) is not int or state["global_step"] <= RESUME_INTERVAL_START:
        raise AssertionError(label + " did not perform a real optimizer update after the resume point")
    if not isinstance(state["trainable"], dict) or len(state["trainable"]) != EXPECTED_TRAINABLE_TENSORS:
        raise AssertionError(label + " does not include all 450 trainable tensors")
    if any(not isinstance(tensor, torch.Tensor) for tensor in state["trainable"].values()):
        raise AssertionError(label + " trainable state contains a non-tensor value")
    rng = state["rng"]
    if not isinstance(rng, dict) or not {"python", "numpy", "cpu", "cuda", "loader_generator"} <= rng.keys():
        raise AssertionError(label + " lacks Python/NumPy/Torch/CUDA/loader-generator RNG state")
    if not isinstance(rng["cuda"], (list, tuple)) or len(rng["cuda"]) != 2:
        raise AssertionError(label + " must capture RNG state for both visible CUDA devices")
    optimizer_sections = {key: value for key, value in state.items() if "optim" in str(key).lower() or "master" in str(key).lower()}
    masters, moments = [], {"exp_avg": 0, "exp_avg_sq": 0}
    for key, path, value in named_nodes(optimizer_sections):
        if key in MASTER_KEYS:
            for tensor_path, tensor in tensor_leaves(value, torch, path):
                if tensor.is_floating_point() and tensor.numel():
                    masters.append({"path": tensor_path, "elements": tensor.numel(), "dtype": str(tensor.dtype)})
        if key in moments and isinstance(value, torch.Tensor):
            moments[key] += value.numel()
    if not masters:
        raise AssertionError(label + " lacks explicit DeepSpeed FP32 master-flat partition tensors; underlying Adam state_dict alone is insufficient")
    if any(item["dtype"] != "torch.float32" for item in masters):
        raise AssertionError(label + " master partition tensors are not FP32")
    master_elements = sum(item["elements"] for item in masters)
    if moments["exp_avg"] != master_elements or moments["exp_avg_sq"] != master_elements:
        raise AssertionError(label + " optimizer moments do not completely cover its FP32 master partition")
    return {"global_step": state["global_step"], "trainable_tensors": len(state["trainable"]),
            "trainable_elements": sum(tensor.numel() for tensor in state["trainable"].values()),
            "master_partitions": masters, "master_partition_elements": master_elements,
            "optimizer_moment_elements": moments, "rng_keys": list(rng)}


def read_events(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream]


def compare_events(continuous_path, resumed_path, final_step):
    continuous, resumed = read_events(continuous_path), read_events(resumed_path)
    restored = [row for row in resumed if row.get("event") == "strict_ds_restored"]
    replayed = [row for row in resumed if row.get("event") == "loader_replayed"]
    if len(restored) != 1 or restored[0]["global_step"] != RESUME_INTERVAL_START:
        raise AssertionError("Resumed events must prove one strict DeepSpeed restore at global_step=2")
    if len(replayed) != 1 or replayed[0]["checkpoint_step"] != RESUME_INTERVAL_START:
        raise AssertionError("Resumed events must prove loader/RNG replay from checkpoint step 2")
    fields = ("step", "loader_yielded", "input_sha256", "label_sha256", "shape")

    def batches(events):
        return [{key: row[key] for key in fields} for row in events
                if row.get("event") == "batch" and row["step"] >= RESUME_INTERVAL_START]

    left, right = batches(continuous), batches(resumed)
    expected_steps = set(range(RESUME_INTERVAL_START, final_step))
    for label, rows in (("continuous", left), ("resumed", right)):
        if not rows or set(row["step"] for row in rows) != expected_steps:
            raise AssertionError(label + " is missing a batch in the complete restoration interval")
        counts = Counter(row["step"] for row in rows)
        if any(count != 4 for count in counts.values()):
            raise AssertionError(label + " does not record all four accumulation microbatches per optimizer update")
        for row in rows:
            for key in ("input_sha256", "label_sha256"):
                if len(row[key]) != 64 or any(char not in "0123456789abcdef" for char in row[key]):
                    raise AssertionError("Malformed input/label hash evidence")
    if left != right:
        index = next((i for i, pair in enumerate(zip(left, right)) if pair[0] != pair[1]), min(len(left), len(right)))
        raise AssertionError(f"Restored batch/input/label/loader position sequence differs at interval microbatch {index}")

    def updates(events):
        return [{"step": row["step"], "lr": row["lr"]} for row in events
                if row.get("event") == "optimizer_step" and row["step"] > RESUME_INTERVAL_START]

    left_updates, right_updates = updates(continuous), updates(resumed)
    if left_updates != right_updates or [row["step"] for row in right_updates] != list(range(3, final_step + 1)):
        raise AssertionError("Restored optimizer-step and learning-rate sequence does not exactly match continuous execution")
    if not any(any(math.isfinite(lr) and lr > 0 for lr in row["lr"]) for row in right_updates):
        raise AssertionError("No resumed optimizer update has a nonzero finite learning rate")

    def floating_event_sequence(kind, field, expected_sequence):
        def select(events):
            return [{"step": row["step"], field: row[field]} for row in events
                    if row.get("event") == kind and row["step"] >= RESUME_INTERVAL_START
                    and (kind != "optimizer_step" or row["step"] > RESUME_INTERVAL_START)]

        reference, restored_values = select(continuous), select(resumed)
        if ([row["step"] for row in reference] != expected_sequence or
                [row["step"] for row in restored_values] != expected_sequence):
            raise AssertionError(f"{kind}.{field} sequence is missing, reordered, or has unexpected events")
        rows = []
        for index, (reference_row, restored_row) in enumerate(zip(reference, restored_values)):
            ref, actual = reference_row[field], restored_row[field]
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                   for value in (ref, actual)):
                raise AssertionError(f"{kind}.{field} must be finite at interval position {index}")
            difference = abs(actual - ref)
            limit = ATOL + RTOL * abs(ref)
            rows.append({"step": reference_row["step"], "interval_position": index,
                         "continuous": ref, "resumed": actual, "abs_diff": difference,
                         "fixed_tolerance_limit": limit, "within_fixed_tolerance": difference <= limit})
        maximum = max(row["abs_diff"] for row in rows)
        violations = sum(not row["within_fixed_tolerance"] for row in rows)
        if violations:
            raise AssertionError(f"{kind}.{field} exceeds fixed atol={ATOL}/rtol={RTOL}: "
                                 f"violations={violations}, max_abs_diff={maximum}")
        return {"count": len(rows), "all_finite": True, "all_within_fixed_tolerance": True,
                "exact_equal": all(row["abs_diff"] == 0 for row in rows),
                "max_abs_diff": maximum,
                "max_relative_diff_with_atol_denominator_floor": max(
                    row["abs_diff"] / max(abs(row["continuous"]), ATOL) for row in rows),
                "comparisons": rows}

    losses = floating_event_sequence("micro_loss", "loss", [row["step"] for row in left])
    losses["meaning"] = "Trainer.training_step return value: microbatch loss already divided by gradient_accumulation_steps=4"
    gradient_norms = floating_event_sequence("optimizer_step", "grad_norm", list(range(3, final_step + 1)))
    sequence_hash = hashlib.sha256(json.dumps(left, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"status": "PASS", "restore_global_step": 2, "final_global_step": final_step,
            "complete_interval_microbatches": len(left), "batch_and_label_hash_sequence_exact": True,
            "loader_yielded_and_shapes_exact": True, "batch_sequence_sha256": sequence_hash,
            "optimizer_step_and_lr_sequence_exact": True, "resumed_optimizer_updates": len(right_updates),
            "nonzero_learning_rate_update_verified": True,
            "micro_loss_comparison": losses, "optimizer_grad_norm_comparison": gradient_norms,
            "continuous_events": {"path": str(continuous_path), "sha256": sha(continuous_path)},
            "resumed_events": {"path": str(resumed_path), "sha256": sha(resumed_path)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuous", required=True)
    parser.add_argument("--resumed", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    started = time.monotonic()
    report = {"schema": "letter_a0_full_resume_compare_v1", "status": "FAIL", "argv": sys.argv,
              "host": socket.gethostname(), "python": sys.executable,
              "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "execution": "CPU only: torch.load(map_location='cpu', weights_only=False)",
              "tolerance": {"atol": ATOL, "rtol": RTOL, "reference": "continuous",
                            "criterion": "Every floating tensor element satisfies abs(resumed-continuous) <= atol + rtol*abs(continuous)",
                            "exact_required": ["scheduler", "global_step", "optimizer step/counters", "all RNG including loader generator", "all discrete state"],
                            "relative_diagnostic_denominator": "max(abs(continuous), atol)",
                            "fixed_before_observation": True,
                            "bf16_note": "rtol=1e-3 is stricter than a typical single BF16 ULP; nondeterministic BF16 differences may fail. No result-dependent widening is permitted."},
              "ranks": [], "errors": []}
    try:
        import torch
        import numpy as np
        torch.set_num_threads(4)
        report["versions"] = {"torch": torch.__version__, "numpy": np.__version__}
        continuous_root, resumed_root = state_root(args.continuous), state_root(args.resumed)
        if continuous_root == resumed_root:
            raise ValueError("Continuous and resumed paths must identify distinct executions")
        for root in (continuous_root, resumed_root):
            names = {path.name for path in root.glob("acceptance_state_*.pt")}
            if names != {"acceptance_state_0.pt", "acceptance_state_1.pt"}:
                raise AssertionError("Acceptance must include exactly the two expected rank-state files")
        for rank in range(2):
            row = {"rank": rank, "status": "FAIL"}
            report["ranks"].append(row)
            try:
                paths = [root / f"acceptance_state_{rank}.pt" for root in (continuous_root, resumed_root)]
                row["files"] = [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)} for path in paths]
                left, right = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
                row["continuous_schema"] = inspect_schema(left, torch, "continuous")
                row["resumed_schema"] = inspect_schema(right, torch, "resumed")
                compare = Compare(torch, np)
                compare.compare(left, right)
                row.update(tensor_comparisons=compare.tensors, numpy_array_comparisons=compare.arrays,
                           exact_scalar_leaves=compare.exact_scalar_leaves, state_failures=compare.failures,
                           all_model_and_optimizer_tensors_finite=all(item.get("reference_finite") and item.get("resumed_finite") for item in compare.tensors))
                row["events"] = compare_events(continuous_root / f"events.rank{rank}.jsonl",
                                                resumed_root / f"events.rank{rank}.jsonl", left["global_step"])
                row["status"] = "PASS" if not compare.failures else "FAIL"
                del left, right
            except BaseException as error:
                row["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
            write(args.output, report)
            print(json.dumps({"rank": rank, "status": row["status"], "error": row.get("error", {}).get("message"),
                              "state_failures": len(row.get("state_failures", []))}), flush=True)
        report["status"] = "PASS" if len(report["ranks"]) == 2 and all(row["status"] == "PASS" for row in report["ranks"]) else "FAIL"
    except BaseException as error:
        report["errors"].append({"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write(args.output, report)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
