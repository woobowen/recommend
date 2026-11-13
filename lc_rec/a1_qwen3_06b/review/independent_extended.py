#!/usr/bin/env python3
"""Nonauthor, stdlib-only second-pass audit of manifest-bound A1 saved predictions.

Does not import the primary offline evaluator, Torch, NumPy, or vendor metrics.
The separate Stage-A script supplies the original-metric/identity gate. This
independent pass checks identities, raw parsing, per-row relevance, ranking
metrics, explicit expanded-item CCE ranks, and Top-K SID diagnostics. It checks
SHA during its own necessary complete stream against the frozen A1 manifest.
"""
import argparse
import collections
import csv
import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import resource
import sys
import time

N, B = 24772, 20
INPUTS = {
    "index": "33c87e28c52ae367985bdf494cb49d676d181e13a39497b4ad035f46111546dd",
    "inter": "0a11d47791d90483005da93c36d826679e20e4592d1d2b86518d5f56f218e743",
    "item": "1d9938e576a8d964e6b977a2b63aa56402840a788eaa6685f26f0d910a7133f2",
}


def require(test, message):
    if not test:
        raise ValueError(message)


def no_duplicate_keys(pairs):
    answer = {}
    for key, value in pairs:
        require(key not in answer, "Duplicate JSON key: " + str(key))
        answer[key] = value
    return answer


def read_json(path):
    return json.loads(Path(path).read_bytes(), object_pairs_hook=no_duplicate_keys)


def normalize(raw):
    require(isinstance(raw, str), "Non-string decoded prediction")
    return raw.split("Response:")[-1].strip().replace(" ", "")


def sorted_sids(candidates):
    require(isinstance(candidates, list) and len(candidates) == B, "Not Beam20")
    pairs = []
    for j, candidate in enumerate(candidates):
        require(candidate.get("generation_order") == j, "Generation order changed")
        score = candidate.get("sequence_score")
        require(type(score) in (int, float) and math.isfinite(score), "Bad beam score")
        sid = normalize(candidate.get("raw_decoded"))
        require(sid == candidate.get("sid"), "Saved SID parsing mismatch")
        pairs.append((float(score), j, sid))
    # Sorting only by score intentionally preserves original order on ties.
    pairs.sort(key=lambda pair: pair[0], reverse=True)
    return [pair[2] for pair in pairs], [pair[1] for pair in pairs]


def ranking(sids, target):
    hits = [i + 1 for i, sid in enumerate(sids) if sid == target]
    first = hits[0] if hits else None
    result = {}
    for k in (1, 5, 10, 20):
        result[f"hit@{k}"] = float(first is not None and first <= k)
    for k in (5, 10, 20):
        result[f"ndcg@{k}"] = math.fsum(1 / math.log2(r + 1) for r in hits if r <= k)
        result[f"mrr@{k}"] = 1 / first if first is not None and first <= k else 0.0
    return result, first


def expanded_cce(sids, target, sizes):
    """Explicitly assign item ranks before averaging target-group credit.

    Repeated preceding SIDs each occupy a fresh group; invalid SIDs occupy one
    slot, matching fixed official code. Only first occurrence of target counts.
    """
    target_positions = []
    cursor = 0
    target_size = sizes.get(target, 1)
    require(type(target_size) is int and target_size > 0, "Invalid target group")
    for sid in sids:
        width = sizes.get(sid, 1)
        require(type(width) is int and width > 0, "Invalid SID group width")
        if sid == target:
            target_positions = list(range(cursor + 1, cursor + width + 1))
            break
        cursor += width
    result = {}
    for k in (5, 10, 20):
        positions = [p for p in target_positions if p <= k]
        result[f"ItemHit@{k}"] = len(positions) / target_size
        result[f"ItemNDCG@{k}"] = math.fsum(1 / math.log2(p + 1) for p in positions) / target_size
    return result


def compare_derived_row(derived, row_idx, user, item, target, first, values):
    require(derived.get("row_idx") == str(row_idx)
            and derived.get("user_id") == user
            and derived.get("target_item_id") == item
            and derived.get("target_sid") == target, "Derived per-user identity mismatch")
    require(derived.get("first_target_rank") == str(first or 0), "Derived first target rank mismatch")
    deltas = {}
    for key, expected in values.items():
        actual = float(derived[key])
        require(math.isfinite(actual), "Non-finite derived per-user metric")
        deltas[key] = abs(actual - expected)
        require(deltas[key] <= 1e-14, "Derived per-user metric mismatch: " + key)
    return deltas


def self_test(official_path=None, official_sha=None):
    checks = []

    def close(got, expected, label):
        require(math.isclose(got, expected, rel_tol=0, abs_tol=1e-14), label)
        checks.append(label)

    for r in (1, 5, 6, 10, 11, 20, None):
        sids = ["miss"] * B
        if r is not None:
            sids[r - 1] = "target"
        values, first = ranking(sids, "target")
        require(first == r, "First target position")
        for k in (1, 5, 10, 20):
            close(values[f"hit@{k}"], float(r is not None and r <= k), f"rank{r}:Hit{k}")
        for k in (5, 10, 20):
            present = r is not None and r <= k
            close(values[f"ndcg@{k}"], 1 / math.log2(r + 1) if present else 0, f"rank{r}:NDCG{k}")
            close(values[f"mrr@{k}"], 1 / r if present else 0, f"rank{r}:MRR{k}")
    duplicate, _ = ranking(["t", "t"] + ["x"] * 18, "t")
    close(duplicate["ndcg@5"], 1 + 1 / math.log2(3), "Duplicate target NDCG additive")
    close(duplicate["mrr@5"], 1, "Duplicate target MRR first")
    tied = [{"generation_order": i, "sequence_score": 0., "raw_decoded": f"Response: s{i}", "sid": f"s{i}"} for i in range(B)]
    require(sorted_sids(tied)[1] == list(range(B)), "Stable ties")
    checks.append("Stable tied sequence scores")
    require(normalize("old Response: prompt Response: <a_1> <b_2>\n") == "<a_1><b_2>", "Response parsing")
    checks.append("Last Response separator and exact space normalization")
    fixture = {"row_idx": "0", "user_id": "u", "target_item_id": "i", "target_sid": "t",
               "first_target_rank": "0", "hit@5": "0"}
    require(compare_derived_row(fixture, 0, "u", "i", "t", None, {"hit@5": 0}) == {"hit@5": 0},
            "Derived no-hit row equality")
    for field, bad in (("row_idx", "1"), ("target_sid", "wrong"), ("first_target_rank", "1"), ("hit@5", "1")):
        changed = dict(fixture, **{field: bad})
        try:
            compare_derived_row(changed, 0, "u", "i", "t", None, {"hit@5": 0})
        except ValueError:
            checks.append("Derived wrong " + field + " rejected")
        else:
            raise AssertionError("Bad derived fixture was accepted: " + field)

    cases = [
        ("no_collision", ["a", "b", "c", "t"], {"a": 1, "b": 1, "c": 1, "t": 1}, 1., 1 / math.log2(5)),
        ("target_size2", ["a", "b", "c", "t"], {"a": 1, "b": 1, "c": 1, "t": 2}, 1., (1 / math.log2(5) + 1 / math.log2(6)) / 2),
        ("figure2_size3_cross_K", ["a", "b", "c", "t"], {"a": 1, "b": 1, "c": 1, "t": 3}, 2 / 3, (1 / math.log2(5) + 1 / math.log2(6)) / 3),
        ("group_outside_K", ["a", "t"], {"a": 5, "t": 3}, 0., 0.),
        ("target_absent", ["a", "b"], {"a": 1, "b": 1, "t": 2}, 0., 0.),
        ("previous_collision", ["a", "b", "t"], {"a": 3, "b": 1, "t": 1}, 1., 1 / math.log2(6)),
        ("invalid_preceding", ["invalid", "t"], {"t": 1}, 1., 1 / math.log2(3)),
        ("duplicate_preceding", ["a", "a", "t"], {"a": 2, "t": 1}, 1., 1 / math.log2(6)),
        ("duplicate_target_first", ["t", "a", "t"], {"a": 1, "t": 2}, 1., (1 + 1 / math.log2(3)) / 2),
        ("unique_target_pushed_out", ["a", "b", "t"], {"a": 4, "b": 2, "t": 1}, 0., 0.),
    ]
    for label, sids, sizes, hit, ndcg in cases:
        result = expanded_cce(sids, "t", sizes)
        close(result["ItemHit@5"], hit, label + ":ItemHit5")
        close(result["ItemNDCG@5"], ndcg, label + ":ItemNDCG5")
    official_receipt = None
    if official_path is not None:
        require(official_sha is not None, "Official CCE SHA must be pinned")
        raw = Path(official_path).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == official_sha, "Official CCE bytes differ")
        spec = importlib.util.spec_from_file_location("fixed_official_cce_for_independent_test", official_path)
        official = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(official)
        compared = 0
        random_source = random.Random(20261113)
        extra = []
        for _ in range(256):
            names = ["a", "b", "c", "d", "t", "invalid"]
            sizes = {name: random_source.randint(1, 4) for name in names[:-1]}
            extra.append(("random", [random_source.choice(names) for _ in range(20)], sizes, 0, 0))
        for label, sids, sizes, _, _ in cases + extra:
            mine = expanded_cce(sids, "t", sizes)
            theirs = official.compute_cce_metrics(sids, "t", sizes, [5, 10, 20])
            for key in mine:
                close(mine[key], theirs[key], label + ":official:" + key)
            compared += 1
        official_receipt = {"path": str(official_path), "bytes": len(raw), "sha256": official_sha,
                            "cases_compared": compared, "per_metric_absolute_tolerance": 1e-14}
    return {"status": "PASS", "numerical_assertions": len(checks),
            "named_cce_cases": [case[0] for case in cases], "official_comparison": official_receipt}


def frozen_identity(manifest_path, test_dir):
    """Bind only this A1 run's immutable prediction identity, never A0 constants."""
    manifest = read_json(manifest_path)
    require(manifest.get("schema") == "a1_frozen_prediction_manifest_v1", "Frozen manifest schema")
    run = manifest.get("gpu_run_commit")
    require(isinstance(run, str) and len(run) == 40
            and all(c in "0123456789abcdef" for c in run), "Complete GPU RUN SHA")
    records = manifest.get("predictions")
    require(isinstance(records, list) and len(records) == 2, "Two frozen prediction shards")
    predictions = {}
    for record in records:
        require(isinstance(record, dict), "Prediction manifest record")
        rank = record.get("rank")
        require(type(rank) is int and rank in (0, 1) and rank not in predictions, "Unique prediction rank")
        size, digest, name = record.get("bytes"), record.get("sha256"), record.get("path")
        require(type(size) is int and size > 0, "Prediction byte size")
        require(isinstance(digest, str) and len(digest) == 64
                and all(c in "0123456789abcdef" for c in digest), "Prediction SHA-256")
        require(isinstance(name, str) and Path(name).is_absolute(), "Absolute prediction path")
        require(Path(name).resolve() == (Path(test_dir) / f"predictions.rank{rank}.jsonl").resolve(),
                "Frozen prediction path differs from test directory")
        predictions[rank] = record
    require(set(predictions) == {0, 1}, "Frozen rank axis")
    return run, predictions


def aggregate(data_root, test_dir, stage_a_path, manifest_path, per_user_path=None):
    run, predictions = frozen_identity(manifest_path, test_dir)
    stage_a = read_json(stage_a_path)
    require(stage_a.get("status") == "PASS" and stage_a.get("run_commit") == run
            and stage_a.get("actual_test_users") == N, "Stage-A audit not manifest-bound full PASS")
    loaded, asset_receipts = {}, {}
    for kind, expected_sha in INPUTS.items():
        path = Path(data_root) / "Instruments" / f"Instruments.{kind}.json"
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        require(actual == expected_sha, "Frozen data mismatch: " + kind)
        loaded[kind] = json.loads(raw, object_pairs_hook=no_duplicate_keys)
        asset_receipts[kind] = {"path": str(path), "bytes": len(raw), "sha256": actual}
    index, inter = loaded["index"], loaded["inter"]
    require(len(index) == 9922 and len(inter) == N and set(index) == set(loaded["item"]), "Asset axes")
    user_axis = list(inter)
    derived_rows, per_user_receipt = None, None
    if per_user_path is not None:
        derived_rows = {}
        with Path(per_user_path).open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            require(reader.fieldnames and len(reader.fieldnames) == len(set(reader.fieldnames)), "Duplicate CSV fields")
            for row in reader:
                i = int(row["row_idx"])
                require(0 <= i < N and i not in derived_rows, "Derived CSV row coverage")
                derived_rows[i] = row
        require(set(derived_rows) == set(range(N)), "Derived CSV population incomplete")
        per_user_receipt = {"path": str(per_user_path), "bytes": Path(per_user_path).stat().st_size,
                            "sha256": hashlib.sha256(Path(per_user_path).read_bytes()).hexdigest(),
                            "checked_rows": 0, "checked_metric_values": 0,
                            "maximum_absolute_delta_by_metric": {}, "absolute_tolerance": 1e-14}
    sizes = collections.Counter("".join(tokens) for tokens in index.values())
    require(len(sizes) == 9897 and sum(g > 1 for g in sizes.values()) == 23, "Collision catalog")
    stage_records = stage_a.get("prediction_files")
    require(isinstance(stage_records, list) and len(stage_records) == 2, "Stage-A prediction records")
    require(all(type(r.get("rank")) is int for r in stage_records), "Stage-A rank type")
    stage_manifests = {r["rank"]: r for r in stage_records}
    require(set(stage_manifests) == {0, 1}, "Stage-A unique rank axis")
    all_values = collections.defaultdict(list)
    first_ranks = collections.Counter()
    targets_collision = 0
    topk_counts = {k: collections.Counter() for k in (5, 10, 20)}
    valid_seen = {k: set() for k in (5, 10, 20)}
    seen, manifests, examples = set(), [], {"invalid": [], "target_collision": [], "preceding_collision_changes_cce": []}
    for rank in (0, 1):
        path = Path(test_dir) / f"predictions.rank{rank}.jsonl"
        before = path.stat()
        frozen = predictions[rank]
        expected_bytes, expected_sha = frozen["bytes"], frozen["sha256"]
        inherited = stage_manifests[rank]
        require(Path(inherited["path"]).resolve() == path.resolve(), "Stage-A prediction path")
        require(before.st_size == expected_bytes and inherited["bytes"] == expected_bytes
                and inherited["sha256"] == expected_sha and inherited["users"] == N // 2,
                "Stage-A full-stream prediction identity")
        count, byte_count, actual_sha = 0, 0, hashlib.sha256()
        with path.open("rb") as stream:
            for raw in stream:
                actual_sha.update(raw)
                byte_count += len(raw)
                row = json.loads(raw, object_pairs_hook=no_duplicate_keys)
                i = row.get("row_idx")
                require(type(i) is int and 0 <= i < N and i not in seen, "User row coverage")
                history = inter[user_axis[i]]
                item = str(history[-1])
                target = "".join(index[item])
                require(row.get("rank") == rank and row.get("prompt_id") == 0, "Rank or prompt")
                require(row.get("user_id") == user_axis[i] and row.get("target_item_id") == item
                        and row.get("target") == target, "User and target identity")
                sids, order = sorted_sids(row.get("candidates"))
                relevance = [int(sid == target) for sid in sids]
                require(row.get("sorted_generation_order") == order and row.get("relevance") == relevance,
                        "Stored rank relevance does not match raw independent parsing")
                metrics, first = ranking(sids, target)
                cce = expanded_cce(sids, target, sizes)
                if derived_rows is not None:
                    differences = compare_derived_row(derived_rows[i], i, user_axis[i], item, target,
                                                      first, {**metrics, **cce})
                    per_user_receipt["checked_rows"] += 1
                    per_user_receipt["checked_metric_values"] += len(differences)
                    maxima = per_user_receipt["maximum_absolute_delta_by_metric"]
                    for key, delta in differences.items():
                        maxima[key] = max(maxima.get(key, 0), delta)
                for key, value in {**metrics, **cce}.items():
                    all_values[key].append(value)
                first_ranks[str(first) if first is not None else "not_in_top20"] += 1
                target_g = sizes[target]
                targets_collision += int(target_g > 1)
                for k in (5, 10, 20):
                    chosen = sids[:k]
                    valid = [sid for sid in chosen if sid in sizes]
                    invalid_n = k - len(valid)
                    dup_n = k - len(set(chosen))
                    coll_n = sum(sizes.get(sid, 0) > 1 for sid in chosen)
                    valid_seen[k].update(valid)
                    topk_counts[k].update({"candidates": k, "valid": len(valid), "invalid": invalid_n,
                                          "duplicate_occurrences": dup_n, "collision_candidates": coll_n,
                                          "users_with_invalid": int(invalid_n > 0),
                                          "users_with_duplicate": int(dup_n > 0),
                                          "users_with_collision": int(coll_n > 0)})
                sample = {"row_idx": i, "target_sid": target, "target_group_size": target_g,
                          "first_target_rank": first, "sid_metrics": metrics, "cce": cce}
                if any(sid not in sizes for sid in sids) and len(examples["invalid"]) < 3:
                    examples["invalid"].append({**sample, "invalid_ranks": [j + 1 for j, sid in enumerate(sids) if sid not in sizes]})
                if target_g > 1 and len(examples["target_collision"]) < 3:
                    examples["target_collision"].append(sample)
                if (first is not None and target_g == 1
                        and any(sizes.get(sid, 1) > 1 for sid in sids[:first - 1])
                        and len(examples["preceding_collision_changes_cce"]) < 3):
                    examples["preceding_collision_changes_cce"].append(sample)
                seen.add(i)
                count += 1
        after = path.stat()
        require(count == N // 2 and (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                "Prediction file changed or count mismatch")
        require(byte_count == expected_bytes and actual_sha.hexdigest() == expected_sha,
                "Independent full-stream SHA differs from frozen A1 manifest")
        manifests.append({"rank": rank, "path": str(path), "users": count, "bytes": after.st_size,
                          "mtime_ns": after.st_mtime_ns, "sha256": expected_sha,
                          "hash_scope": "Independent SHA computed during this necessary complete prediction stream"})
    require(seen == set(range(N)), "Incomplete full user axis")
    sums = {key: math.fsum(values) for key, values in all_values.items()}
    means = {key: value / N for key, value in sums.items()}
    for key in ("hit@5", "hit@10", "ndcg@5", "ndcg@10"):
        require(abs(means[key] - stage_a["metrics"][key]) < 1e-13, "Original metric mismatch")
    diagnostics = {}
    for k, counts in topk_counts.items():
        diagnostics[str(k)] = {**counts, "users": N,
            "valid_sid_rate": counts["valid"] / counts["candidates"],
            "invalid_sid_rate": counts["invalid"] / counts["candidates"],
            "duplicate_sid_rate": counts["duplicate_occurrences"] / counts["candidates"],
            "collision_sid_candidate_rate": counts["collision_candidates"] / counts["candidates"],
            "users_with_invalid_rate": counts["users_with_invalid"] / N,
            "users_with_duplicate_rate": counts["users_with_duplicate"] / N,
            "users_with_collision_rate": counts["users_with_collision"] / N,
            "unique_valid_sids": len(valid_seen[k]), "sid_coverage_denominator": len(sizes),
            "unique_valid_sid_coverage": len(valid_seen[k]) / len(sizes)}
    require(sum(first_ranks.values()) == N, "First-rank partition")
    if diagnostics["20"]["duplicate_occurrences"] == 0:
        for k in (5, 10, 20):
            require(sum(first_ranks[str(r)] for r in range(1, k + 1)) == sums[f"hit@{k}"], "First rank hit count")
    return {"status": "PASS", "run_commit": run, "actual_test_users": len(seen),
            "frozen_manifest": {"path": str(manifest_path),
                "sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()},
            "metrics": means, "metric_sums": sums, "sid_diagnostics": diagnostics,
            "first_target_rank_counts": {str(r): first_ranks[str(r)] for r in range(1, 21)}
                 | {"not_in_top20": first_ranks["not_in_top20"]},
            "collision_assets": {"items": len(index), "unique_sids": len(sizes),
                "collision_groups": sum(g > 1 for g in sizes.values()),
                "collision_items": sum(g for g in sizes.values() if g > 1),
                "max_group_size": max(sizes.values()),
                "group_size_distribution": dict(sorted(collections.Counter(sizes.values()).items())),
                "target_collision_users": targets_collision},
            "assets": asset_receipts, "prediction_files": manifests, "bounded_examples": examples,
            "per_user_comparison": per_user_receipt,
            "stage_a_receipt": {"path": str(stage_a_path), "sha256": hashlib.sha256(Path(stage_a_path).read_bytes()).hexdigest()},
            "policy": {"paper_sid_metrics": "Original raw normalization, stable descending sequence score, no filtering/deduplication, all target matches additive NDCG",
                       "mrr": "First target match reciprocal rank",
                       "cce": "First target group, explicit expanded positions, invalid SID consumes one, repeated preceding SID expands repeatedly",
                       "sid_coverage": "Distinct valid complete SID / 9897 catalog SIDs"},
            "verification_scope": "Independent implementation and second complete CPU stream; no GPU/model loads, no original file writes"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--test-dir", type=Path)
    parser.add_argument("--stage-a-report", type=Path)
    parser.add_argument("--manifest", type=Path, help="Required frozen A1 RUN/prediction manifest for full audit")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--per-user", type=Path,
                        help="Optional primary CSV: independently compare each raw-prediction user and all 16 values")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--official-cce", type=Path)
    parser.add_argument("--official-sha256")
    args = parser.parse_args()
    started, usage = time.perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
    if args.output and args.output.exists():
        parser.error("Preserve prior output; --output already exists")
    tests = self_test(args.official_cce, args.official_sha256)
    if args.self_test:
        report = {"status": "PASS", "self_test": tests}
        code = 0
    else:
        if not all((args.data_root, args.test_dir, args.stage_a_report, args.manifest, args.output, args.official_cce, args.official_sha256)):
            parser.error("Full audit needs data, test, Stage-A, frozen A1 manifest, output, and pinned official CCE")
        try:
            report = aggregate(args.data_root, args.test_dir, args.stage_a_report, args.manifest, args.per_user)
            report["self_test"] = tests
            code = 0
        except Exception as error:
            report, code = {"status": "FAIL", "error": repr(error), "self_test": tests}, 1
    final = resource.getrusage(resource.RUSAGE_SELF)
    report.update(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  audit="NONAUTHOR_INDEPENDENT_EXTENDED_RECOMPUTE", argv=sys.argv,
                  created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), exit_code=code,
                  elapsed_seconds=time.perf_counter() - started,
                  cpu_user_seconds=final.ru_utime - usage.ru_utime,
                  cpu_system_seconds=final.ru_stime - usage.ru_stime, peak_rss_kib=final.ru_maxrss)
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    print(text, end="")
    return code


if __name__ == "__main__":
    sys.exit(main())
