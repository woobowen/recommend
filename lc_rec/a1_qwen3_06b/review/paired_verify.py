#!/usr/bin/env python3
"""Nonauthor CPU verification of frozen A0/A1 paired user bootstrap.

No import of the production paired/evaluation modules. The independent
column-wise gather follows the previously reviewed statistics verifier, while
CSV/NPZ metadata and value equality bind every user before subtraction.
"""
import argparse
import csv
import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import resource
import sys
import time

N = 24772
REPETITIONS = 5000
SEED = 20261113
TOLERANCE = 1e-13
PROTOCOL = "letter_instruments_beam20_v1"
AXIS = "1daf63ce4c77b0265777a46b61939006aafab0276b9106c742de602c8c7e867a"
A0_RUN = "4c6a83d9ba42dc2fe737792759e410ef1874da76"
A0_HASHES = {
    "per_user.csv": "423309c6765395e42fc6a18a43706a4653d71692b184c65fe5755f78a34df6fb",
    "per_user_metrics.npz": "d46590f04dcd0f8438b67ccea636b3310a61ff081e146305032c614f444a1689",
}
KEYS = ([f"hit@{k}" for k in (1, 5, 10, 20)] +
        [f"{m}@{k}" for m in ("ndcg", "mrr") for k in (5, 10, 20)] +
        [f"{m}@{k}" for m in ("ItemHit", "ItemNDCG") for k in (5, 10, 20)])
FIELDS = ["row_idx", "user_id", "user_axis_sha256", "target_item_id", "target_sid",
          "target_group_size", "original_interaction_length", "visible_history_length",
          "history_group", "target_training_frequency", "popularity_group", "protocol_version"]


def require(value, message):
    if not value:
        raise ValueError(message)


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + str(key))
        result[key] = value
    return result


def read_bound(path, protected):
    path = Path(path).resolve()
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    stamp = lambda value: (value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    require(stamp(before) == stamp(after), "Input changed during read: " + str(path))
    identity = {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    protected[str(path)] = {"identity": identity, "stat": stamp(after)}
    return raw, identity


def parse_json(raw):
    return json.loads(raw, object_pairs_hook=unique)


def bind_values(rows, values, keys, expected_users):
    """Bind metric values to the CSV order before any bootstrap calculation."""
    import numpy as np
    require(keys == KEYS, "Frozen sixteen metric columns/order")
    require(values.dtype == np.dtype("float64") and values.shape == (expected_users, 16)
            and np.isfinite(values).all(), "Finite float64 metric matrix")
    require(len(rows) == expected_users, "CSV user count")
    csv_values = np.asarray([[float(row[key]) for key in KEYS] for row in rows], dtype=np.float64)
    require(np.array_equal(csv_values, values), "CSV/NPZ full per-user metric values differ")


def pair_rows(left, right, expected_users):
    require(len(left) == len(right) == expected_users, "Paired user count")
    for i, (a, b) in enumerate(zip(left, right)):
        require(a["row_idx"] == b["row_idx"] == str(i), "Paired row_idx order")
        require(all(a[field] == b[field] for field in FIELDS), "Paired user/target/group mismatch")


def load_primary(root, protected, baseline=False):
    import numpy as np
    root = Path(root).resolve()
    raw, identities = {}, {}
    for name in ("per_user.csv", "per_user_metrics.npz", "metrics.json"):
        raw[name], identities[name] = read_bound(root / name, protected)
    if baseline:
        for name, expected in A0_HASHES.items():
            require(identities[name]["sha256"] == expected, "Frozen A0 file SHA: " + name)
    summary = parse_json(raw["metrics.json"])
    require(summary["status"] == "PASS" and summary["protocol_version"] == PROTOCOL
            and summary["users"] == N and summary["user_axis_sha256"] == AXIS,
            "Primary frozen protocol/user axis")
    run = summary["run_commit"]
    require(isinstance(run, str) and len(run) == 40
            and all(c in "0123456789abcdef" for c in run), "Primary full GPU RUN SHA")
    require(run == A0_RUN if baseline else run != A0_RUN, "Distinct correct A0/A1 RUN identities")
    claimed = summary["per_user_files"]
    require(isinstance(claimed, list) and len(claimed) == 2, "Primary per-user file manifest")
    claims = {Path(row["path"]).name: row for row in claimed}
    require(set(claims) == set(A0_HASHES), "Primary unique per-user filenames")
    for name, record in claims.items():
        identity = identities[name]
        require(Path(record["path"]).resolve() == root / name
                and record["bytes"] == identity["bytes"] and record["sha256"] == identity["sha256"],
                "Primary manifest differs from actual CSV/NPZ: " + name)
    reader = csv.DictReader(io.StringIO(raw["per_user.csv"].decode("utf-8"), newline=""))
    require(reader.fieldnames and len(reader.fieldnames) == len(set(reader.fieldnames)), "Duplicate CSV columns")
    require(set(FIELDS + KEYS) <= set(reader.fieldnames), "Missing CSV metadata/metrics")
    rows = list(reader)
    require(len(rows) == N, "Complete CSV user population")
    users = []
    for i, row in enumerate(rows):
        require(row["row_idx"] == str(i) and row["user_axis_sha256"] == AXIS
                and row["protocol_version"] == PROTOCOL, "CSV row/protocol axis")
        users.append(row["user_id"])
    require(len(set(users)) == N, "Unique CSV users")
    actual_axis = hashlib.sha256(json.dumps(users, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    require(actual_axis == AXIS, "Reconstructed ordered user axis SHA")
    with np.load(io.BytesIO(raw["per_user_metrics.npz"]), allow_pickle=False) as saved:
        values = saved["values"].copy()
        keys = saved["keys"].tolist()
        require(saved["row_idx"].dtype.kind in "iu"
                and np.array_equal(saved["row_idx"], np.arange(N)), "NPZ integer row axis")
        require(str(saved["user_axis_sha256"]) == AXIS, "NPZ canonical user-axis SHA")
    bind_values(rows, values, keys, N)
    require(set(summary["metrics"]) == set(KEYS), "Primary complete metric summary")
    for j, key in enumerate(KEYS):
        mean = math.fsum(float(value) for value in values[:, j]) / N
        require(abs(mean - summary["metrics"][key]) <= TOLERANCE, "Primary metric mean: " + key)
    return rows, values, identities, run


def bootstrap_columns(difference, repetitions=REPETITIONS, seed=SEED, block=8):
    """Independent column-wise gather; same user draws across all columns."""
    import numpy as np
    a = np.asarray(difference, dtype=np.float64)
    require(a.ndim == 2 and a.shape[0] > 0 and np.isfinite(a).all(), "Paired difference matrix")
    rng = np.random.Generator(np.random.PCG64(seed))
    means = np.empty((repetitions, a.shape[1]), dtype=np.float64)
    for begin in range(0, repetitions, block):
        count = min(block, repetitions - begin)
        selected = rng.integers(0, a.shape[0], size=(count, a.shape[0]), dtype=np.int64)
        for column in range(a.shape[1]):
            means[begin:begin + count, column] = np.take(a[:, column], selected).mean(axis=1)
    return np.percentile(means, [2.5, 97.5], axis=0, method="linear"), means


def verify(a0, a1, paired):
    import numpy as np
    protected = {}
    left, baseline, left_files, left_run = load_primary(a0, protected, baseline=True)
    right, candidate, right_files, right_run = load_primary(a1, protected)
    pair_rows(left, right, N)
    result_raw, result_file = read_bound(Path(paired) / "paired_comparison.json", protected)
    draw_raw, draw_file = read_bound(Path(paired) / "paired_bootstrap_difference_means.npy", protected)
    original = parse_json(result_raw)
    require(original["status"] == "PASS_PAIRED_USER_AXIS" and original["users"] == N
            and original["user_axis_sha256"] == AXIS, "Main paired population/axis")
    boot = original["bootstrap"]
    require(boot["repetitions"] == REPETITIONS and boot["seed"] == SEED
            and boot["replacement"] is True and boot["rng"] == "numpy PCG64"
            and boot["percentile_method"] == "linear", "Main paired bootstrap definition")
    require(set(original["metrics"]) == set(KEYS), "Main paired complete sixteen columns")
    inputs = {str(Path(key).resolve()): value for key, value in original["inputs"].items()}
    require(len(inputs) == 2 and set(inputs) == {str(Path(a0).resolve()), str(Path(a1).resolve())}, "Paired input roots")
    for root, identities in ((a0, left_files), (a1, right_files)):
        hashes = inputs[str(Path(root).resolve())]
        require(set(hashes) == set(A0_HASHES), "Paired claimed per-user files")
        require(all(hashes[name] == identities[name]["sha256"] for name in A0_HASHES), "Paired input hashes")
    reference = np.load(io.BytesIO(draw_raw), allow_pickle=False)
    require(reference.dtype == np.dtype("float64") and reference.shape == (REPETITIONS, 16)
            and np.isfinite(reference).all(), "Main paired replicate means")
    tick = time.monotonic()
    intervals, draws = bootstrap_columns(candidate - baseline)
    bootstrap_seconds = time.monotonic() - tick
    max_draw_delta = float(np.max(np.abs(draws - reference)))
    require(max_draw_delta <= TOLERANCE, "Independent paired replicate means differ")
    max_point_delta = max_interval_delta = 0.0
    metrics = {}
    for j, key in enumerate(KEYS):
        base = math.fsum(float(value) for value in baseline[:, j]) / N
        new = math.fsum(float(value) for value in candidate[:, j]) / N
        delta = new - base
        expected = original["metrics"][key]
        points = {"A0": base, "A1": new, "absolute_A1_minus_A0": delta,
                  "relative_difference": delta / base if base else None}
        for field, value in points.items():
            if value is None:
                require(expected[field] is None, "Undefined relative difference must be null")
            else:
                require(type(expected[field]) in (int, float) and math.isfinite(expected[field]), "Finite paired point")
                difference = abs(value - expected[field])
                max_point_delta = max(max_point_delta, difference)
                require(difference <= TOLERANCE, "Paired point estimate differs: " + key + ":" + field)
        for endpoint, row in (("difference_CI95_low", 0), ("difference_CI95_high", 1)):
            difference = abs(float(intervals[row, j]) - expected[endpoint])
            max_interval_delta = max(max_interval_delta, difference)
            require(difference <= TOLERANCE, "Independent paired CI differs: " + key)
        metrics[key] = {**points, "difference_CI95_low": float(intervals[0, j]),
                        "difference_CI95_high": float(intervals[1, j])}
    for name, value in protected.items():
        state = Path(name).stat()
        require((state.st_size, state.st_mtime_ns, state.st_ctime_ns) == value["stat"], "Protected input changed: " + name)
    return {"status": "PASS", "audit": "NONAUTHOR_INDEPENDENT_PAIRED_BOOTSTRAP",
            "A0_GPU_RUN_COMMIT": left_run, "A1_GPU_RUN_COMMIT": right_run,
            "users": N, "user_axis_sha256": AXIS, "identity_fields_exact": FIELDS,
            "metric_columns": KEYS, "CSV_NPZ_metric_values_compared": 2 * N * 16,
            "replicate_means_compared": REPETITIONS * 16, "bootstrap_max_draw_absolute_difference": max_draw_delta,
            "bootstrap_max_interval_absolute_difference": max_interval_delta,
            "point_estimate_max_absolute_difference": max_point_delta,
            "absolute_tolerance": TOLERANCE, "bootstrap_elapsed_seconds": bootstrap_seconds,
            "bootstrap": {"repetitions": REPETITIONS, "seed": SEED, "rng": "numpy Generator PCG64",
                "replacement": True, "draw_order": "replicate-major int64, same user indices across both models",
                "percentiles": [2.5, 97.5], "percentile_method": "linear"},
            "metrics": metrics, "input_files": {"A0": left_files, "A1": right_files,
                "paired_comparison": result_file, "paired_draws": draw_file},
            "protected_inputs_stat_unchanged": True, "gpu_used": False,
            "numpy_version": np.__version__,
            "limits": "Derived metrics only; raw ranking/CCE and single-model grouping/statistics require separate audits. Conditional on fixed trained models; not seed variance, a subgroup CI, or an isolated SID causal effect. Controller bridge execution does not imply independent SSH/GPU reproduction."}


def self_test():
    import numpy as np
    a0 = np.array([[0., .1, .4], [1., .3, .4], [0., .7, .4], [1., .9, .4]])
    a1 = np.array([[1., .3, .65], [0., .5, .65], [1., .2, .65], [1., .7, .65]])
    ci, draws = bootstrap_columns(a1 - a0, repetitions=137, block=7)
    rng = np.random.Generator(np.random.PCG64(SEED))
    scalar = np.empty_like(draws)
    for i in range(len(draws)):
        selected = rng.integers(0, len(a0), size=len(a0), dtype=np.int64)
        for j in range(a0.shape[1]):
            scalar[i, j] = math.fsum(float(a1[r, j] - a0[r, j]) for r in selected) / len(a0)
    require(np.max(np.abs(draws - scalar)) <= TOLERANCE, "Synthetic scalar paired replicate formula")
    ci1, draws1 = bootstrap_columns(a1 - a0, repetitions=137, block=1)
    require(np.array_equal(draws, draws1) and np.array_equal(ci, ci1), "Synthetic block invariance")
    require(np.array_equal(ci[:, 2], [.25, .25]), "Constant paired shift CI")
    reversed_ci, reversed_draws = bootstrap_columns(a0 - a1, repetitions=137, block=7)
    require(np.array_equal(reversed_draws, -draws)
            and np.max(np.abs(reversed_ci - np.stack([-ci[1], -ci[0]]))) <= TOLERANCE,
            "Paired swap antisymmetry")
    zero_ci, zero_draws = bootstrap_columns(a0 - a0, repetitions=137)
    require(np.count_nonzero(zero_ci) == np.count_nonzero(zero_draws) == 0, "Identical models zero difference")
    fixture = np.arange(64, dtype=np.float64).reshape(4, 16) / 100
    rows = [{key: str(fixture[i, j]) for j, key in enumerate(KEYS)} for i in range(4)]
    bind_values(rows, fixture, KEYS, 4)
    checks = 6
    for changed, keys in ((fixture[[1, 0, 2, 3]], KEYS), (fixture.copy(), list(reversed(KEYS))),
                          (np.full_like(fixture, np.nan), KEYS)):
        try:
            bind_values(rows, changed, keys, 4)
        except ValueError:
            checks += 1
        else:
            raise AssertionError("Bad synthetic CSV/NPZ fixture accepted")
    paired_rows = [{field: str(i) if field == "row_idx" else "same" for field in FIELDS} for i in range(4)]
    pair_rows(paired_rows, paired_rows, 4)
    checks += 1
    bad_rows = [dict(row) for row in paired_rows]
    bad_rows[2]["target_item_id"] = "wrong"
    try:
        pair_rows(paired_rows, bad_rows, 4)
    except ValueError:
        checks += 1
    else:
        raise AssertionError("Bad synthetic paired target accepted")
    return {"status": "PASS_SYNTHETIC_PAIRED_VERIFIER_ONLY", "directed_checks": checks,
            "synthetic_users": 4, "synthetic_repetitions": 137, "numpy_version": np.__version__,
            "scalar_max_absolute_difference": float(np.max(np.abs(draws - scalar))),
            "scope": "Synthetic formula/negative input fixtures only; no A1 prediction or formal result was loaded."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("a0", "a1", "paired", "output"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("Preserve prior reviewer output")
    if not args.self_test and not all((args.a0, args.a1, args.paired, args.output)):
        parser.error("Full verification requires --a0 --a1 --paired --output")
    if args.output and not args.self_test:
        for root in (args.a0, args.a1):
            require(Path(root).resolve() not in args.output.resolve().parents, "Do not write inside primary input roots")
    started = time.monotonic()
    try:
        report = self_test() if args.self_test else verify(args.a0, args.a1, args.paired)
        code = 0
    except Exception as error:
        report = {"status": "FAIL", "error": repr(error), "synthetic_only": args.self_test}
        code = 1
    report.update(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), argv=sys.argv,
                  created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), exit_code=code,
                  elapsed_seconds=time.monotonic() - started,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(text)
    print(text, end="")
    return code


if __name__ == "__main__":
    sys.exit(main())
