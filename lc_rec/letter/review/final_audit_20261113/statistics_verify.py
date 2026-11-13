#!/usr/bin/env python3
"""Independent CPU-only checks of A0 grouping and user-level bootstrap.

This reviewer-owned module imports no training or production evaluator module.
Data-only mode deliberately does not inspect prediction performance.
"""
import argparse
from collections import Counter
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time

HASHES = {
    "Instruments.index.json": "33c87e28c52ae367985bdf494cb49d676d181e13a39497b4ad035f46111546dd",
    "Instruments.inter.json": "0a11d47791d90483005da93c36d826679e20e4592d1d2b86518d5f56f218e743",
    "Instruments.item.json": "1d9938e576a8d964e6b977a2b63aa56402840a788eaa6685f26f0d910a7133f2",
}
METRICS = ([f"hit@{k}" for k in (1, 5, 10, 20)] +
           [f"{name}@{k}" for name in ("ndcg", "mrr") for k in (5, 10, 20)] +
           [f"{name}@{k}" for name in ("ItemHit", "ItemNDCG") for k in (5, 10, 20)])


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def digest_json(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                                     sort_keys=True).encode()).hexdigest()


def file_identity(path):
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def history_group(length):
    require(type(length) is int and 0 <= length <= 20, "Visible history length")
    return "0-5" if length <= 5 else "6-10" if length <= 10 else "11-20"


def thresholds(frequencies):
    """Positive catalog item frequencies; inverse empirical CDF at thirds."""
    ordered = sorted(value for value in frequencies if value > 0)
    require(ordered, "No positive training frequencies")
    n = len(ordered)
    return ordered[(n + 2) // 3 - 1], ordered[(2 * n + 2) // 3 - 1]


def popularity_group(frequency, q1, q2):
    require(type(frequency) is int and frequency >= 0, "Negative/nonintegral count")
    if frequency == 0:
        return "UNSEEN"
    if frequency <= q1:
        return "LONG_TAIL"
    return "MID" if frequency <= q2 else "POPULAR"


def data_census(root):
    loaded, identities = {}, {}
    for name, expected in HASHES.items():
        path = root / name
        raw = path.read_bytes()
        found = hashlib.sha256(raw).hexdigest()
        require(found == expected, "Asset SHA: " + name)
        loaded[name] = json.loads(raw)
        identities[name] = {"bytes": len(raw), "sha256": found}
    index, inter, items = (loaded["Instruments." + suffix + ".json"]
                           for suffix in ("index", "inter", "item"))
    require(set(index) == set(items) and len(items) == 9922, "Catalog axis")
    require(len(inter) == 24772, "User count")
    frequencies = Counter({item: 0 for item in index})
    for sequence in inter.values():
        require(sequence and all(str(item) in index for item in sequence), "Interaction item")
        frequencies.update(str(item) for item in sequence[:-2])
    q1, q2 = thresholds(frequencies.values())
    item_groups = Counter(popularity_group(f, q1, q2) for f in frequencies.values())
    history_counts, target_counts = Counter(), Counter()
    rows = []
    for row_idx, (user, sequence) in enumerate(inter.items()):
        target_item = str(sequence[-1])
        visible = min(20, len(sequence[:-1]))
        hg = history_group(visible)
        pg = popularity_group(frequencies[target_item], q1, q2)
        history_counts[hg] += 1
        target_counts[pg] += 1
        rows.append({"row_idx": row_idx, "user_id": str(user),
                     "target_item_id": target_item,
                     "target_sid": "".join(index[target_item]),
                     "original_interaction_length": len(sequence),
                     "visible_history_length": visible, "history_group": hg,
                     "target_training_frequency": frequencies[target_item],
                     "popularity_group": pg})
    report = {"status": "PASS", "asset_identity": identities,
              "training_visible_interaction_count": sum(frequencies.values()),
              "catalog_items": len(index), "positive_frequency_items":
              sum(f > 0 for f in frequencies.values()),
              "unseen_catalog_items": sum(f == 0 for f in frequencies.values()),
              "quantile_rule": "positive-frequency catalog, nearest rank ceil(n*p), ties not split",
              "q1": q1, "q2": q2, "catalog_item_groups": dict(item_groups),
              "history_group_users": dict(history_counts),
              "popularity_group_users": dict(target_counts),
              "user_axis_sha256": digest_json([r["user_id"] for r in rows]),
              "derived_rows_sha256": digest_json(rows)}
    require(sum(history_counts.values()) == 24772 == sum(target_counts.values()), "Group total")
    return rows, report


def bootstrap_columns(matrix, repetitions=5000, seed=20261113, block=8):
    """Independent column-wise gather implementation; fixed PCG64 stream.

    Sampling order is replicate-major, then row-major. Column-wise gather avoids
    materializing (block, users, metrics) and intentionally differs from a joint
    multivariate gather. Changing block leaves the consumed integer stream equal.
    """
    import numpy as np
    a = np.asarray(matrix, dtype=np.float64)
    require(a.ndim == 2 and a.shape[0] > 0 and np.isfinite(a).all(), "Metric matrix")
    rng = np.random.Generator(np.random.PCG64(seed))
    result = np.empty((repetitions, a.shape[1]), dtype=np.float64)
    for begin in range(0, repetitions, block):
        count = min(block, repetitions - begin)
        selection = rng.integers(0, a.shape[0], size=(count, a.shape[0]), dtype=np.int64)
        for column in range(a.shape[1]):
            result[begin:begin + count, column] = np.take(a[:, column], selection).mean(axis=1)
    return np.percentile(result, [2.5, 97.5], axis=0, method="linear"), result


def self_test():
    import numpy as np
    require([history_group(x) for x in (0, 5, 6, 10, 11, 20)] ==
            ["0-5", "0-5", "6-10", "6-10", "11-20", "11-20"], "History boundaries")
    require(thresholds([0, 1, 1, 2, 3, 4, 5]) == (1, 3), "Nearest-rank thresholds")
    require(thresholds([0, 2, 2, 2, 2]) == (2, 2), "Unsplit tied frequencies")
    require([popularity_group(x, 2, 5) for x in (0, 1, 2, 3, 5, 6)] ==
            ["UNSEEN", "LONG_TAIL", "LONG_TAIL", "MID", "MID", "POPULAR"], "Popularity boundaries")
    # Numerical percentile fixture: exhaustive two-user sample means are 0, .5, .5, 1.
    exact = np.array([0.0, 0.5, 0.5, 1.0])
    require(np.allclose(np.percentile(exact, [2.5, 97.5], method="linear"),
                        [0.0375, 0.9625], atol=1e-15), "Percentile interpolation")
    fixture = np.array([[0.0, 1.0, 1.0], [1.0, 1.0, 0.5], [0.0, 1.0, 0.0]])
    ci1, draws1 = bootstrap_columns(fixture, repetitions=101, block=1)
    ci7, draws7 = bootstrap_columns(fixture, repetitions=101, block=7)
    require(np.array_equal(draws1, draws7) and np.array_equal(ci1, ci7), "Block independence")
    require(np.array_equal(ci1[:, 1], [1.0, 1.0]), "Constant metric CI")
    require(np.isfinite(ci1).all() and (ci1[0] <= ci1[1]).all(), "CI finite and ordered")
    return {"status": "PASS", "tests": 7, "numpy_version": np.__version__,
            "synthetic_ci": ci1.tolist(), "interpretation":
            "User sampling conditional on one frozen trained model; not between-seed variance"}


def verify_full_statistics(expected_rows, census, per_user, primary_bootstrap, primary_draws):
    """Check all derived metadata and independently resample the full fixed user axis."""
    import numpy as np
    started = time.monotonic()
    n = len(expected_rows)
    identities = {"per_user": file_identity(per_user),
                  "primary_bootstrap": file_identity(primary_bootstrap),
                  "primary_draws": file_identity(primary_draws)}
    with per_user.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames and len(reader.fieldnames) == len(set(reader.fieldnames)),
                "Duplicate/missing CSV columns")
        require(set(METRICS).issubset(reader.fieldnames), "Missing 16 metric columns")
        matrix = np.empty((n, len(METRICS)), dtype=np.float64)
        counts = {"history_group": Counter(), "popularity_group": Counter()}
        parsed_users = 0
        for i, row in enumerate(reader):
            require(i < n, "Extra user row")
            expected = expected_rows[i]
            for key, value in expected.items():
                require(key in row, "Missing metadata: " + key)
                got = int(row[key]) if type(value) is int else row[key]
                require(got == value, f"Metadata differs at row {i}: {key}")
            require(row["user_axis_sha256"] == census["user_axis_sha256"], "User axis hash")
            require(row["protocol_version"] == "letter_instruments_beam20_v1", "Protocol version")
            require(0 <= int(row["first_target_rank"]) <= 20, "First rank range")
            elapsed = float(row["generation_seconds"])
            require(math.isfinite(elapsed) and elapsed >= 0, "Generation seconds")
            for j, name in enumerate(METRICS):
                value = float(row[name])
                require(math.isfinite(value) and value >= 0, "Invalid per-user metric")
                matrix[i, j] = value
            for field in counts:
                counts[field][row[field]] += 1
            parsed_users += 1
    require(parsed_users == n == 24772, "Not all users inspected")

    # The companion NPZ is optional for this CLI but checked when present. It is
    # the primary manifest's claimed metric-array identity, not merely a CSV label.
    npz_path = per_user.parent / "per_user_metrics.npz"
    require(npz_path.is_file(), "Missing companion frozen per-user NPZ")
    identities["primary_metric_array"] = file_identity(npz_path)
    with np.load(npz_path, allow_pickle=False) as saved:
        require(saved["values"].dtype == np.dtype("float64"), "NPZ must preserve float64")
        require(saved["keys"].tolist() == METRICS, "NPZ metric column axis")
        require(np.array_equal(saved["row_idx"], np.arange(n)), "NPZ row axis")
        require(str(saved["user_axis_sha256"]) == census["user_axis_sha256"], "NPZ user hash")
        require(np.array_equal(saved["values"], matrix), "CSV and NPZ metrics differ")

    groups_path = per_user.parent / "groups.json"
    require(groups_path.is_file(), "Missing subgroup report")
    identities["primary_groups"] = file_identity(groups_path)
    groups = json.loads(groups_path.read_bytes())
    require(groups["designation"] == "SUPPLEMENTARY_ANALYSIS", "Subgroup designation")
    require((groups["q1"], groups["q2"]) == (census["q1"], census["q2"]), "Popularity cutpoints")
    require(groups["training_visible_interactions"] == census["training_visible_interaction_count"],
            "Training-visible frequency count")
    require(groups["catalog_groups"] == census["catalog_item_groups"], "Catalog popularity groups")
    group_max_difference = 0.0
    for field, observed in counts.items():
        for label, saved in groups["groups"][field].items():
            indices = [i for i, row in enumerate(expected_rows) if row[field] == label]
            require(saved["users"] == observed[label] == len(indices), "Subgroup denominator")
            for j, name in enumerate(METRICS):
                estimate = math.fsum(float(matrix[i, j]) for i in indices) / len(indices) if indices else None
                if estimate is None:
                    require(saved["metrics"][name] is None, "Empty-group estimate must be null")
                else:
                    difference = abs(saved["metrics"][name] - estimate)
                    group_max_difference = max(group_max_difference, difference)
                    require(difference <= 1e-13, "Subgroup metric differs")

    primary = json.loads(primary_bootstrap.read_bytes())
    require(primary["unit"] == "user" and primary["replacement"] is True, "Bootstrap unit")
    require(primary["repetitions"] == 5000 and primary["seed"] == 20261113, "Bootstrap repetitions/seed")
    require(primary["users"] == n and primary["percentiles"] == [2.5, 97.5]
            and primary["percentile_method"] == "linear", "Bootstrap interval definition")
    require(primary["rng"] == "numpy Generator PCG64", "Bootstrap RNG")
    require(primary["metric_array_sha256"] == identities["primary_metric_array"]["sha256"],
            "Bootstrap input matrix SHA differs")
    boot_started = time.monotonic()
    independent_ci, independent_draws = bootstrap_columns(matrix, repetitions=5000,
                                                        seed=20261113, block=8)
    bootstrap_seconds = time.monotonic() - boot_started
    original_draws = np.load(primary_draws, allow_pickle=False)
    require(original_draws.dtype == np.dtype("float64") and original_draws.shape == (5000, 16)
            and np.isfinite(original_draws).all(), "Primary draw means shape/type")
    max_draw_difference = float(np.max(np.abs(original_draws - independent_draws)))
    require(max_draw_difference <= 1e-13, "Independent bootstrap replicate means differ")
    max_ci_difference, max_point_difference = 0.0, 0.0
    intervals = {}
    for j, name in enumerate(METRICS):
        point = math.fsum(float(x) for x in matrix[:, j]) / n
        saved = primary["intervals"][name]
        point_difference = abs(saved["estimate"] - point)
        differences = [abs(saved["low"] - independent_ci[0, j]),
                       abs(saved["high"] - independent_ci[1, j])]
        max_point_difference = max(max_point_difference, point_difference)
        max_ci_difference = max(max_ci_difference, *differences)
        require(point_difference <= 1e-13 and max(differences) <= 1e-13,
                "Independent bootstrap interval differs: " + name)
        intervals[name] = {"estimate": point, "low": float(independent_ci[0, j]),
                           "high": float(independent_ci[1, j])}
    return {"status": "PASS", "users_compared": n, "metric_columns_compared": METRICS,
            "source_identity": file_identity(Path(__file__)), "input_identity": identities,
            "metadata_checks": "all users against independently reconstructed original interactions",
            "group_users": {key: dict(value) for key, value in counts.items()},
            "subgroup_max_absolute_difference": group_max_difference,
            "replicate_means_compared": 5000 * 16,
            "bootstrap_max_draw_absolute_difference": max_draw_difference,
            "bootstrap_max_interval_absolute_difference": max_ci_difference,
            "bootstrap_max_point_absolute_difference": max_point_difference,
            "absolute_tolerance": 1e-13, "bootstrap_elapsed_seconds": bootstrap_seconds,
            "elapsed_seconds": time.monotonic() - started, "numpy_version": np.__version__,
            "intervals": intervals,
            "limits": "No raw prediction rescan. Recomputed metadata, grouping and user bootstrap from derived rows; ranking/CCE independently audited separately. Overall CI is not a subgroup CI or training-seed variability."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--per-user", type=Path)
    parser.add_argument("--primary-bootstrap", type=Path)
    parser.add_argument("--primary-draws", type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    result = {"reviewer": "statistics_review", "scope": "CPU, independent implementation"}
    if args.self_test:
        result["self_test"] = self_test()
    if args.data_root:
        expected_rows, result["data_census"] = data_census(args.data_root)
    if args.per_user or args.primary_bootstrap or args.primary_draws:
        require(args.data_root and args.per_user and args.primary_bootstrap and args.primary_draws,
                "Full verification needs data-root, per-user, primary-bootstrap and primary-draws")
        result["full_statistics"] = verify_full_statistics(expected_rows, result["data_census"],
                                                          args.per_user, args.primary_bootstrap,
                                                          args.primary_draws)
    require(args.self_test or args.data_root, "No requested check")
    result["status"] = "PASS"
    result["created_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    result["elapsed_seconds"] = time.monotonic() - started
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result["cpu_user_seconds"] = usage.ru_utime
    result["cpu_system_seconds"] = usage.ru_stime
    result["cpu_peak_rss_bytes"] = usage.ru_maxrss * 1024
    result["python"] = sys.executable
    result["argv"] = sys.argv
    result["gpu_used"] = False
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as out:
            json.dump(result, out, ensure_ascii=False, indent=2, sort_keys=True)
            out.write("\n")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
