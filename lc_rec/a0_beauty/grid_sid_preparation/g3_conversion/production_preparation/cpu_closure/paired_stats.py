"""Pair fixed Beauty evaluations using prediction results from this closure job."""

import csv
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from predictions import KEYS, PAIR_FIELDS


POPULATION = 22363
REPETITIONS = 5000
SEED = 20261113
INTERVAL_TOLERANCE = 1e-13
ARCHIVED_S3_JSON_SHA256 = "6899948326c5ef5ab10aca80481bad6c7d1bdbfa6b708e3e7745aefe4260d29c"
ARCHIVED_S3_DRAW_SHA256 = "b176dfda7bc00244b332a46ccb8e34920681be761de9c55c5e56bf83c827f297"
COMPARISON_FIELDS = (
    "Normal", "Shuffle", "absolute_difference", "relative_difference",
    "difference_CI95_low", "difference_CI95_high",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def file_identity(path):
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": digest(raw)}


def module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bootstrap_sources(release):
    training_src = release / "lc_rec" / "a0_beauty"
    independent_src = release / "lc_rec" / "letter" / "review" / "final_audit_20261113" / "statistics_verify.py"
    sys.path.insert(0, str(training_src))
    try:
        original = module_from_path("frozen_g3_paired_beauty", training_src / "paired_beauty.py")
    finally:
        sys.path.pop(0)
    independent = module_from_path("frozen_g3_statistics_verify", independent_src)
    require(tuple(original.KEYS) == KEYS == tuple(independent.METRICS), "Frozen metric order differs")
    return original, independent, file_identity(independent_src)


def validate_inputs(conditions):
    require(set(conditions) == {"normal", "s3", "s4"}, "Expected three formal closure conditions")
    reference = conditions["normal"]
    reference_fields = reference["per_user_fields"]
    reference_adapter = reference["audit"]["adapter_files"]
    reference_commit = reference["audit"]["run_commit"]
    for name, result in conditions.items():
        matrix = result["original_matrix"]
        require(matrix.shape == (POPULATION, len(KEYS)) and matrix.dtype == np.float64,
                f"{name} production matrix shape/dtype")
        require(np.isfinite(matrix).all(), f"{name} production matrix is not finite")
        fields = result["per_user_fields"]
        require(len(fields) == POPULATION and all(len(row) == len(PAIR_FIELDS) for row in fields),
                f"{name} paired user fields are incomplete")
        for idx, (ref, actual) in enumerate(zip(reference_fields, fields, strict=True)):
            require(ref == actual and str(actual[0]) == str(idx), f"{name} paired user mismatch at row {idx}")
        require(len(result["ranked"]) == POPULATION and all(len(r) == 20 for r in result["ranked"]),
                f"{name} candidate axis")
        require(result["audit"]["adapter_files"] == reference_adapter and
                result["audit"]["run_commit"] == reference_commit, f"{name} model provenance differs")
        require(len(result["paired_input_identities"]) == 1, f"{name} primary output identity is ambiguous")
        per_user = next(iter(result["paired_input_identities"].values()))
        require(set(per_user) == {"per_user.csv", "per_user_metrics.npz"}, f"{name} per-user inputs missing")


def means(matrix):
    return {key: math.fsum(matrix[:, col]) / POPULATION for col, key in enumerate(KEYS)}


def behavior_changes(normal, shuffle, normal_means, shuffle_means):
    sizes = dict(top1_changed=0, top20_ordered_changed=0, top20_set_changed=0)
    for expected, actual in zip(normal["ranked"], shuffle["ranked"], strict=True):
        sizes["top1_changed"] += expected[0] != actual[0]
        sizes["top20_ordered_changed"] += expected != actual
        sizes["top20_set_changed"] += set(expected) != set(actual)
    transitions = {}
    left_matrix, right_matrix = normal["original_matrix"], shuffle["original_matrix"]
    for col, key in enumerate(KEYS):
        if not key.startswith("hit@"):
            continue
        left_values, right_values = left_matrix[:, col], right_matrix[:, col]
        cells = {f"{left}->{right}": int(np.count_nonzero(
                    (left_values == left) & (right_values == right)))
                 for left in (0, 1) for right in (0, 1)}
        require(sum(cells.values()) == POPULATION, f"{key} transition user count differs")
        require((cells["1->0"] + cells["1->1"]) / POPULATION == normal_means[key],
                f"{key} Normal transition margin differs")
        require((cells["0->1"] + cells["1->1"]) / POPULATION == shuffle_means[key],
                f"{key} Shuffle transition margin differs")
        transitions[key] = cells
    fractions = {f"{name}_fraction": value / POPULATION for name, value in sizes.items()}
    return {"users": POPULATION, **sizes, "hit_transitions": transitions, **fractions}


def compare_archived_s3(archive_json, archive_draws, report, draws):
    raw_json = archive_json.read_bytes()
    raw_draw = archive_draws.read_bytes()
    require(digest(raw_json) == ARCHIVED_S3_JSON_SHA256, "Archived S3 report SHA-256 mismatch")
    require(len(raw_draw) == 640128 and digest(raw_draw) == ARCHIVED_S3_DRAW_SHA256,
            "Archived S3 bootstrap size or SHA-256 mismatch")
    archived = json.loads(raw_json)
    old_draws = np.load(io.BytesIO(raw_draw), allow_pickle=False)
    require(old_draws.shape == draws.shape and old_draws.dtype == draws.dtype and np.isfinite(old_draws).all(),
            "Archived S3 bootstrap shape/dtype")
    maximum_draw = float(np.max(np.abs(old_draws - draws)))
    require(np.array_equal(old_draws, draws), f"Archived S3 bootstrap differs, max delta {maximum_draw}")
    for key in ("scope", "status", "users", "world_size", "execution_topology", "execution_profile",
                "preflight_only", "row_indices", "identity_fields_exact", "bootstrap", "inputs",
                "prediction_provenance", "changes"):
        require(archived[key] == report[key], f"Archived S3 {key} differs")
    require(archived["independent_verification"]["source_sha256"] ==
            report["independent_verification"]["source_sha256"], "Archived independent bootstrap source differs")
    require(tuple(archived["metrics"]) == KEYS, "Archived S3 metric axis")
    maximum_by_field = dict.fromkeys(COMPARISON_FIELDS, 0.0)
    for key in KEYS:
        for field in COMPARISON_FIELDS:
            previous, current = archived["metrics"][key][field], report["metrics"][key][field]
            require(previous == current, f"Archived S3 {key}/{field} differs: {previous} vs {current}")
            if previous is not None:
                maximum_by_field[field] = max(maximum_by_field[field], abs(previous - current))
    return {
        "status": "PASS", "source_report": file_identity(archive_json), "source_draws": file_identity(archive_draws),
        "values_compared": len(KEYS) * len(COMPARISON_FIELDS),
        "draws_compared": int(draws.size), "maximum_absolute_draw_difference": maximum_draw,
        "maximum_absolute_difference_by_field": maximum_by_field,
        "row_axis_exact": True, "prediction_provenance_exact": True, "behavior_and_transitions_exact": True,
    }


def calculate_pair(normal, shuffle, frozen_module, independent_module, independent_source, execution_profile):
    started = time.monotonic()
    normal_means = means(normal["original_matrix"])
    shuffle_means = means(shuffle["original_matrix"])
    difference = shuffle["original_matrix"] - normal["original_matrix"]
    bootstrap_started = time.monotonic()
    draws, lower, upper = frozen_module.paired_bootstrap(np, difference, "FORMAL")
    bootstrap_seconds = time.monotonic() - bootstrap_started
    independent_started = time.monotonic()
    independent_intervals, independent_draws = independent_module.bootstrap_columns(difference)
    independent_seconds = time.monotonic() - independent_started
    require(draws.shape == independent_draws.shape == (REPETITIONS, len(KEYS)) and
            draws.dtype == independent_draws.dtype == np.float64 and
            np.isfinite(draws).all() and np.isfinite(independent_draws).all(), "Paired bootstrap array shape or dtype")
    max_draw = float(np.max(np.abs(draws - independent_draws)))
    max_interval = float(np.max(np.abs(np.asarray([lower, upper]) - independent_intervals)))
    require(max_draw <= INTERVAL_TOLERANCE and max_interval <= INTERVAL_TOLERANCE,
            f"Independent paired bootstrap differs, replicates {max_draw}, intervals {max_interval}")
    metrics = {}
    for col, key in enumerate(KEYS):
        normal_mean, shuffle_mean = normal_means[key], shuffle_means[key]
        delta = shuffle_mean - normal_mean
        metrics[key] = {
            "Normal": normal_mean, "Shuffle": shuffle_mean, "absolute_difference": delta,
            "relative_difference": delta / normal_mean if normal_mean else None,
            "difference_CI95_low": float(lower[col]), "difference_CI95_high": float(upper[col]),
        }

    changes_started = time.monotonic()
    changes = behavior_changes(normal, shuffle, normal_means, shuffle_means)
    changes_seconds = time.monotonic() - changes_started
    input_identities = normal["paired_input_identities"] | shuffle["paired_input_identities"]
    require(len(input_identities) == 2, "Pair inputs refer to the same evaluation directory")
    provenance = [
        {"run_commit": result["audit"]["run_commit"], "adapter_files": result["audit"]["adapter_files"]}
        for result in (normal, shuffle)
    ]
    report = {
        "scope": "FORMAL", "status": "PASS", "users": POPULATION, "world_size": 1,
        "execution_topology": "SINGLE_GPU", "execution_profile": execution_profile, "preflight_only": False,
        "row_indices": list(range(POPULATION)), "identity_fields_exact": list(PAIR_FIELDS), "metrics": metrics,
        "bootstrap": {
            "repetitions": REPETITIONS, "seed": SEED, "rng": "PCG64", "replacement": True,
            "percentile": "95% linear",
            "scope": "User uncertainty conditional on fixed trained model and fixed shuffle realization; not training-seed or permutation variance.",
        },
        "independent_verification": {
            "source": independent_source["path"], "source_sha256": independent_source["sha256"],
            "maximum_draw_difference": max_draw, "maximum_interval_difference": max_interval,
            "tolerance": INTERVAL_TOLERANCE,
        },
        "inputs": input_identities, "prediction_provenance": provenance, "changes": changes,
        "elapsed_seconds": time.monotonic() - started,
        "closure_execution_topology": "CPU_ONLY",
        "timings": {"paired_bootstrap_seconds": bootstrap_seconds, "independent_column_bootstrap_seconds": independent_seconds,
                    "behavior_statistics_seconds": changes_seconds},
    }
    return report, draws


def write_metrics(path, pairs):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow((
            "metric", "Normal", "S3", "S4",
            "S3_minus_Normal", "S3_relative_to_Normal", "S3_CI95_low", "S3_CI95_high",
            "S4_minus_Normal", "S4_relative_to_Normal", "S4_CI95_low", "S4_CI95_high",
        ))
        for key in KEYS:
            s3, s4 = (pairs[c]["metrics"][key] for c in ("s3", "s4"))
            writer.writerow((
                key, s3["Normal"], s3["Shuffle"], s4["Shuffle"],
                *(s3[field] for field in COMPARISON_FIELDS[2:]),
                *(s4[field] for field in COMPARISON_FIELDS[2:]),
            ))


def run_pairing(output, conditions, original_s3_json, original_s3_draws, frozen_release):
    """Use the verified, immutable prediction context without rescanning its inputs."""
    started = time.monotonic()
    validate_inputs(conditions)
    normal = conditions["normal"]
    original, independent, independent_source = bootstrap_sources(Path(frozen_release))
    profile = dict(original.EXECUTION_PROFILE)
    output = Path(output)
    paired_dir = output / "paired"
    paired_dir.mkdir()
    reports = {}
    artifacts = {}
    comparison = None
    for condition in ("s3", "s4"):
        report, draws = calculate_pair(normal, conditions[condition], original, independent,
                                       independent_source, profile)
        if condition == "s3":
            archived_started = time.monotonic()
            comparison = compare_archived_s3(Path(original_s3_json), Path(original_s3_draws), report, draws)
            report["timings"]["archived_comparison_seconds"] = time.monotonic() - archived_started
            report["archived_s3_comparison"] = comparison
        out = paired_dir / condition
        out.mkdir()
        draw_path = out / "paired_bootstrap_difference_means.npy"
        with draw_path.open("xb") as stream:
            np.save(stream, draws, allow_pickle=False)
        report_path = out / "paired_comparison.json"
        with report_path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        reports[condition] = report
        artifacts[condition] = {"report": file_identity(report_path), "bootstrap_draws": file_identity(draw_path)}
        print(json.dumps({"paired_condition": condition, "users": POPULATION, "status": "PASS",
                          "timings": report["timings"]}), flush=True)
    metrics_path = output / "condition_metrics.csv"
    write_metrics(metrics_path, reports)
    return {"pairs": reports, "artifacts": artifacts, "original_s3_comparison": comparison,
            "condition_metrics_csv": file_identity(metrics_path),
            "timings": {"paired_total_seconds": time.monotonic() - started}}
