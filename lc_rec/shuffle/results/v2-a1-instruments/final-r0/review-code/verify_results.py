#!/usr/bin/env python3
"""Independently reconstruct final A1 metrics and paired inference from original candidates."""
import argparse
from collections import Counter
import csv
from hashlib import file_digest, sha256
import json
import math
from pathlib import Path
from time import monotonic
from types import SimpleNamespace

import numpy as np

import preflight_independent_evidence as oracle


ROWS = oracle.NORMAL
BEAMS = oracle.BEAMS
REPLICATES = 5000
SEED = 20261113
CONDITIONS = oracle.CONDITIONS
PAIRS = oracle.PAIRS
COMPARISONS = tuple(f"{right}-{left}" for left, right in PAIRS)
METRICS = oracle.METRICS
A1_GPU = "c487f7e7176bc223445d3deecfab87e0bcc30d1a"
A1_EVAL = "3cc1110097c4764ebfd93bb38ee2913c8afb6958"
A1_NORMAL = "0e3e2c6f5c1a76152ea0228390d94ffc1ff75ba5"
FROZEN_MEANS_SHA = "f92aaff15786fb6e71835ba876e98e4b59c15ad599d6384cb4a1c031c4af02aa"
CCE_SHA = "173c257a8711a885e8f6d5abcdb0ad2510ad4fab01c1700631b5ec261f50cd2d"
need = oracle.need


def sha_file(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def rebuild_condition(shards, targets, levels, catalog, *, raw_root, expected_order):
    values = np.empty((ROWS, len(METRICS)), dtype=np.float64)
    covered = np.zeros(ROWS, dtype=bool)
    anomalies = Counter()
    receipts = []
    for rank, shard in enumerate(shards):
        digest = sha256()
        assignment = []
        byte_count = 0
        with shard.open("rb") as stream:
            for line in stream:
                need(line.endswith(b"\n"), f"Missing LF boundary in {shard.name}")
                digest.update(line)
                byte_count += len(line)
                row = json.loads(line)
                number, metrics, data = oracle.verify_row(row, rank, targets, levels, catalog)
                need(not covered[number], f"Duplicate user {number} in {shard.name}")
                covered[number] = True
                values[number] = metrics
                assignment.append(number)
                invalid = data["invalid_complete_sids"]
                duplicate = data["duplicate_complete_sids"]
                collisions = data["collision_candidate_occurrences"]
                anomalies.update(candidate_count=BEAMS, legal_complete_sid_candidates=BEAMS-invalid,
                                 invalid_complete_sid_candidates=invalid, duplicate_candidate_occurrences=duplicate,
                                 colliding_sid_candidate_occurrences=collisions, users_with_invalid_sid=int(invalid>0),
                                 users_with_duplicate_sid=int(duplicate>0), users_with_colliding_sid_candidate=int(collisions>0),
                                 targets_with_colliding_sid=int(len(catalog[targets[number][2]])>1))
        need(assignment == expected_order[rank], f"{shard.name} recorded user order")
        receipts.append({"path": str(shard.relative_to(raw_root)), "rows": len(assignment), "bytes": byte_count,
                         "sha256": digest.hexdigest(), "assignment_sha256": oracle.canonical_digest(assignment)})
    need(covered.all() and np.isfinite(values).all(), "Complete finite metric matrix")
    return values, receipts, dict(anomalies)


def check_summary(folder, mode, means, audit_sha):
    path = folder / "summary.json"
    summary = oracle.read_json(path)
    need(summary["status"] == "PASS_V2_EVALUATION" and summary["condition"] == mode
         and summary["preflight"] is False, f"{mode} full-run evaluation summary")
    need(summary["users"] == ROWS and summary["candidate_anomalies"]["candidate_count"] == ROWS * BEAMS,
         f"{mode} evaluation coverage")
    need(summary["metrics"] == means, f"{mode} sixteen means differ")
    axis = oracle.EXPECTED_HASH["user_axis"]
    need(summary["user_axis_sha256"] == summary["global_user_axis_sha256"] == axis,
         f"{mode} evaluation user axis")
    need(summary["cpu_manifest_sha256"] == audit_sha and summary["official_cce_sha256"] == CCE_SHA,
         f"{mode} manifest or CCE identity")
    return summary


def check_matrix(folder, matrix, targets):
    path = folder / "primary/per_user_metrics.npz"
    with np.load(path, allow_pickle=False) as data:
        axis = data["row_idx"]
        keys = data["keys"].tolist()
        need(set(keys) == set(METRICS) and len(keys) == len(METRICS), f"{folder.name} metric axis")
        need(np.array_equal(axis, np.arange(ROWS)) and axis.dtype == np.int64, f"{folder.name} index axis")
        need(str(data["user_axis_sha256"]) == oracle.EXPECTED_HASH["user_axis"], f"{folder.name} user axis")
        saved = data["values"]
        need(np.array_equal(saved[:, [keys.index(k) for k in METRICS]], matrix) and saved.dtype == np.float64,
             f"{folder.name} every metric value must match exactly")
    table = folder / "primary/per_user.csv"
    csv_cells = 0
    if table.is_file():
        with table.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            need(set(reader.fieldnames) == {"row_idx", "user_id", "target_item_id", "target", *METRICS},
                 f"{folder.name} per-user CSV header")
            for index, row in enumerate(reader):
                identity = (row["user_id"], row["target_item_id"], row["target"])
                values = np.asarray([float(row[key]) for key in METRICS], dtype=np.float64)
                need(int(row["row_idx"]) == index and identity == targets[index], f"{folder.name} CSV identity {index}")
                need(np.array_equal(values, matrix[index]), f"{folder.name} CSV values {index}")
                csv_cells += len(METRICS)
        need(csv_cells == matrix.size, f"{folder.name} per-user CSV coverage")
    return {"matrix_sha256": sha_file(path), "matrix_cells_exact": matrix.size, "csv_cells_exact": csv_cells}


def verify_prediction_manifest(folder, receipts, commit):
    path = folder / "prediction_manifest.json"
    rows = oracle.read_json(path)
    need(isinstance(rows, list) and len(rows) == len(receipts), f"{folder.name} prediction manifest shape")
    for rank, (manifest, receipt) in enumerate(zip(rows, receipts, strict=True)):
        need(manifest["rank"] == rank and manifest["run_commit"] == commit,
             f"{folder.name} prediction version or rank")
        need(all(manifest[name] == receipt[name] for name in ("bytes", "sha256", "rows")),
             f"{folder.name} prediction manifest does not identify the audited raw bytes")
    return sha_file(path)


def statistical_reference(matrices, indices_path, settings):
    differences = np.stack([matrices[CONDITIONS.index(right)] - matrices[CONDITIONS.index(left)]
                            for left, right in PAIRS])
    # The producer gathers values in draw order. Independently aggregate
    # multiplicities in original user-axis order and use bounded batches.
    by_user = differences.transpose(1, 0, 2).reshape(ROWS, -1)
    rng = np.random.Generator(np.random.PCG64(SEED))
    index_sha = sha256()
    draws = np.empty((REPLICATES, len(PAIRS) * len(METRICS)), dtype=np.float64)
    archived = np.load(indices_path, mmap_mode="r", allow_pickle=False)
    need(archived.shape == (REPLICATES, ROWS) and archived.dtype.str == "<i8", "Raw bootstrap indices shape/dtype")
    for start in range(0, REPLICATES, 32):
        stop = min(REPLICATES, start + 32)
        batch = rng.integers(0, ROWS, size=(stop-start, ROWS), dtype=np.int64)
        need(np.array_equal(archived[start:stop], batch), f"Bootstrap index values in range [{start},{stop})")
        index_sha.update(batch.astype("<i8", copy=False).tobytes(order="C"))
        counts = np.stack([np.bincount(row, minlength=ROWS) for row in batch]).astype(np.float64)
        need(np.all(counts.sum(axis=1) == ROWS), "Draw multiplicity sum")
        draws[start:stop] = counts @ by_user / ROWS
    indices_hash = index_sha.hexdigest()
    need(indices_hash == settings["draw_indices_sha256_little_endian"], "Canonical shared bootstrap index SHA-256")
    file_sha = sha_file(indices_path)
    need(file_sha == settings["draw_indices_file_sha256"], "Original bootstrap .npy disk identity")
    draws = draws.reshape(REPLICATES, len(PAIRS), len(METRICS)).transpose(1, 0, 2)
    ordered = np.sort(draws, axis=1)
    confidence = []
    for percentile in (0.025, 0.975):
        location = (REPLICATES - 1) * percentile
        floor, ceiling = math.floor(location), math.ceil(location)
        fraction = location - floor
        confidence.append(ordered[:, floor] + (ordered[:, ceiling] - ordered[:, floor]) * fraction)
    return differences, draws, confidence, indices_hash, file_sha


def check_paired_csv(path, comparison_map):
    checked = set()
    with path.open(newline="", encoding="utf-8") as csv_stream:
        for row in csv.DictReader(csv_stream):
            name, key = row.pop("comparison"), row.pop("metric")
            lookup = comparison_map[name][key]
            need((name, key) not in checked and set(row) == (lookup.keys() - {"metric"}), "Paired CSV unique identity and schema")
            for label, value in row.items():
                output = float(value) if value != "" else None
                need(output == lookup[label], f"Paired CSV {name}/{key}/{label} differs")
            checked.add((name, key))
    need(len(checked) == len(PAIRS) * len(METRICS), "Paired CSV must include eighty exact records")
    return len(checked)


def audit_paired(folder, matrices, evaluations, estimates, indices_path):
    summary_path = folder / "summary.json"
    arrays_path = folder / "arrays.npz"
    summary = oracle.read_json(summary_path)
    need((summary["status"], summary["preflight"], summary["users"], summary["protocol"],
          summary["identity_policy"], summary["shuffle_seed"]) ==
         ("PASS_V2_PAIRED", False, ROWS, "GRID-LETTER-SHUFFLE-V2", "allow_identity", 42), "Final paired identity")
    need(summary["user_axis_sha256"] == summary["global_user_axis_sha256"] == oracle.EXPECTED_HASH["user_axis"],
         "Paired user identity")
    need(summary["metrics"] == estimates, "All eighty paired means must match raw inputs exactly")
    need(summary["inputs"] == evaluations, "Paired input identities do not match the audited CPU stages")
    settings = summary["bootstrap"]
    need((settings["rng"], settings["seed"], settings["replicates"], settings["replacement"],
          settings["percentile_method"], settings["bounds"], settings["multiple_comparison_adjustment"]) ==
         ("NumPy PCG64", SEED, REPLICATES, True, "linear", [2.5, 97.5], False), "Execution used the approved bootstrap")
    differences, draws, confidence, index_sha, original_file_sha = statistical_reference(matrices, indices_path, settings)
    arrays_hash = sha_file(arrays_path)
    need(arrays_hash == summary["arrays_file_sha256"], "Saved data NPZ identity")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        keys = arrays["keys"].tolist()
        order = [keys.index(key) for key in METRICS]
        need(set(keys) == set(METRICS) and len(keys) == len(METRICS), "Paired metric keys")
        need(arrays["condition"].tolist() == list(CONDITIONS) and arrays["comparison"].tolist() == list(COMPARISONS),
             "Paired condition order")
        need(np.array_equal(arrays["row_idx"], np.arange(ROWS)) and
             str(arrays["user_axis_sha256"]) == oracle.EXPECTED_HASH["user_axis"], "Paired output axis")
        for key, expected in (("per_user_values", matrices), ("per_user_differences", differences)):
            oracle.array_difference(arrays[key][..., order], expected, key, exact=True)
        saved_draws = arrays["bootstrap_difference_means"][..., order]
        max_draw = oracle.array_difference(saved_draws, draws, "All paired draws")
        need(np.isfinite(saved_draws).all(), "Paired draws contain nonfinite data")
        saved_confidence = np.percentile(saved_draws, [2.5, 97.5], axis=1, method="linear")
    comparisons = summary["comparisons"]
    need(set(comparisons) == set(COMPARISONS), "Summary paired identifiers")
    compare_by_name = {}
    exact_intervals, max_interval = 0, 0.0
    qualitative = {}
    for n, ((left, right), name) in enumerate(zip(PAIRS, COMPARISONS, strict=True)):
        records = comparisons[name]
        need(len(records) == len(METRICS) and {r["metric"] for r in records} == set(METRICS), "Summary paired records")
        compare_by_name[name] = records_by_metric = {r["metric"]: r for r in records}
        direction = Counter()
        for i, metric in enumerate(METRICS):
            row = records_by_metric[metric]
            a, b = estimates[left][metric], estimates[right][metric]
            diff = b - a
            need((row["left"], row["right"], row["absolute_difference"],
                  row["percentage_point_difference"], row["relative_change_fraction"]) ==
                 (a, b, diff, 100 * diff, (diff / a if a else None)), f"{name}/{metric} point estimates")
            for endpoint, key in enumerate(("difference_CI95_low", "difference_CI95_high")):
                observed = row[key]
                need(observed == saved_confidence[endpoint, n, i], f"{name}/{metric} saved CI vs archived draws")
                max_interval = max(max_interval, abs(observed - confidence[endpoint][n, i]))
                exact_intervals += int(observed == confidence[endpoint][n, i])
            low, high = row["difference_CI95_low"], row["difference_CI95_high"]
            direction["CI entirely below zero" if high < 0 else "CI entirely above zero" if low > 0 else "CI includes zero"] += 1
        qualitative[name] = dict(direction)
    need(max_interval <= oracle.OFFLINE_TOLERANCE, "Independent CI interpolation exceeded numerical tolerance")
    inspected_csv = check_paired_csv(folder / "paired.csv", compare_by_name)
    return {"rows": inspected_csv, "paired_npz_sha256": arrays_hash, "paired_summary_sha256": sha_file(summary_path),
            "per_user_values_exact": matrices.size, "per_user_differences_exact": differences.size,
            "bootstrap_index_entries_exact": REPLICATES*ROWS, "bootstrap_index_data_sha256": index_sha,
            "bootstrap_original_npy_sha256": original_file_sha, "bootstrap_draws_checked": draws.size,
            "bootstrap_draws_bitwise_exact": int(np.count_nonzero(saved_draws.view(np.uint64) == draws.view(np.uint64))),
            "bootstrap_max_abs_draw_difference": max_draw, "CI_endpoints_checked": 2*len(PAIRS)*len(METRICS),
            "CI_endpoints_exact_vs_saved_draw_percentiles": 2*len(PAIRS)*len(METRICS),
            "CI_endpoints_exact_vs_independent_multiplicity": exact_intervals,
            "CI_max_abs_difference_vs_independent_multiplicity": max_interval,
            "floating_reduction_comparison_bound": oracle.OFFLINE_TOLERANCE,
            "unadjusted_CI_direction_counts": qualitative}


def verify(args):
    evidence, frozen = args.raw / "evidence", args.raw / "frozen"
    normal_folder = evidence / "cpu-normal-r0"
    replay = evidence / "cpu-replay-r0"
    mode_config = SimpleNamespace(
        data_dir=frozen / "data/Instruments", tokenizer=frozen / "adapter/tokenizer.json",
        normal=[frozen / f"normal/predictions.rank{rank}.jsonl" for rank in range(2)],
        archived_matrix=normal_folder / "primary/per_user_metrics.npz",
        archived_metrics=frozen / "primary/metrics.json",
    )
    need(sha_file(mode_config.archived_metrics) == FROZEN_MEANS_SHA, "Frozen Normal metric receipt identity")
    users, targets, catalog, levels = oracle.catalog(mode_config)
    need(sha_file(mode_config.archived_matrix) == oracle.EXPECTED_HASH["archived_matrix"], "Frozen Normal matrix identity")
    single = oracle.read_json(replay / "single-shard.json")
    audit = oracle.read_json(replay / "audit.json")
    audit_sha = audit["manifest_sha256"]
    need(single["manifest_sha256"] == audit_sha and audit["users"] == ROWS and
         audit["user_axis_sha256"] == oracle.EXPECTED_HASH["user_axis"], "Frozen CPU manifest receipts")
    expected_order = single["assignment"]
    need(len(expected_order) == ROWS and set(expected_order) == set(range(ROWS)), "Single rank full assignment")
    need(single["assignment_sha256"] == oracle.canonical_digest(expected_order), "Single rank assignment digest")
    interleaved = [row for pair in zip(*single["normal_rank_assignments"], strict=True) for row in pair]
    need(expected_order == interleaved, "Single rank order is the interleaving of frozen Normal")
    for rank, order in enumerate(single["normal_rank_assignments"]):
        expected = oracle.EXPECTED_HASH[f"assignment{rank}"]
        need(oracle.canonical_digest(order) == expected, f"Frozen Normal rank {rank} assignment identity")
    evaluations, matrices, receipts, estimates = {}, [], {}, {}
    for mode in CONDITIONS:
        folder = evidence / f"formal-cpu-{mode}-r0" if mode != "normal" else normal_folder
        normal = mode == "normal"
        shards = mode_config.normal if normal else [
            evidence / f"formal-gpu-{mode}-r0" / "test/predictions.rank0.jsonl"]
        order = single["normal_rank_assignments"] if normal else [expected_order]
        matrix, prediction_receipts, diagnostics = rebuild_condition(
            shards, targets, levels, catalog, raw_root=args.raw, expected_order=order)
        if normal:
            need(all(receipt["sha256"] == oracle.EXPECTED_HASH[f"normal{rank}"]
                     for rank, receipt in enumerate(prediction_receipts)), "Frozen Normal shard content identity")
            need(oracle.means(matrix) == oracle.read_json(mode_config.archived_metrics)["metrics"],
                 "Frozen historical Normal means reproduced exactly")
        summary = check_summary(folder, mode, oracle.means(matrix), audit_sha)
        need(diagnostics == summary["candidate_anomalies"], f"{mode} candidate diagnostics match raw occurrence counts")
        estimates[mode] = summary["metrics"]
        matrix_check = check_matrix(folder, matrix, targets)
        commit = A1_NORMAL if mode == "normal" else A1_GPU
        prediction_sha = verify_prediction_manifest(folder, prediction_receipts, commit)
        evaluations[mode] = {"matrix_sha256": matrix_check["matrix_sha256"],
                             "cpu_manifest_sha256": audit_sha,
                             "evaluation_summary_sha256": sha_file(folder / "summary.json")}
        receipts[mode] = {"metrics": estimates[mode], "raw_predictions": prediction_receipts,
                          "prediction_manifest_sha256": prediction_sha, **matrix_check,
                          "candidate_anomalies": summary["candidate_anomalies"]}
        matrices.append(matrix)
    pairs = audit_paired(evidence / "formal-paired-r0", np.stack(matrices), evaluations, estimates, args.indices)
    return {"status": "PASS_NON_AUTHOR_B_FORMAL_ALL_METRICS_RECOMPUTED", "users_per_condition": len(users),
            "candidate_records_checked": ROWS*BEAMS*len(CONDITIONS), "frozen_items": 9922, "distinct_sids": len(catalog),
            "sid_collision_groups": sum(len(group) > 1 for group in catalog.values()),
            "frozen_user_axis_sha256": oracle.EXPECTED_HASH["user_axis"],
            "assignment_sha256": single["assignment_sha256"], "cpu_manifest_sha256": audit_sha,
            "formal_gpu_commit": A1_GPU, "formal_evaluation_commit": A1_EVAL,
            "per_condition": receipts, "paired": pairs,
            "numerical_method": "Independent expanded-item collision credit; independent bootstrap via user multiplicities"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--indices", type=Path, help="Use a losslessly reconstructed original int64 NPY")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.indices is None:
        args.indices = args.raw / "evidence/formal-paired-r0/bootstrap_indices.npy"
    need(not args.output.exists(), "Preserve existing audit receipt")
    need(args.indices.is_file(), "Materialize the exact original bootstrap NPY before the audit")
    started = monotonic()
    report = verify(args)
    report.update(elapsed_seconds=monotonic()-started, numpy=np.__version__,
                  verifier_sha256=sha_file(Path(__file__)), preflight_oracle_sha256=sha_file(Path(oracle.__file__)))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "per_condition"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
