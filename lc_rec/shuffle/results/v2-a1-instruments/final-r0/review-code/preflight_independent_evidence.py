#!/usr/bin/env python3
"""Rebuild the frozen A1 and V2 32-user metrics from raw candidates.

No experiment evaluator or statistics module is imported. CCE is reconstructed
with explicit expanded item positions and a single unknown placeholder for each
directory-external candidate. Bootstrap means use user multiplicities.
"""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
from struct import pack, unpack
from time import monotonic

import numpy as np


NORMAL = 24772
BEAMS = 20
EOS = 151643
MODES = ("shuffle2", "shuffle3a", "shuffle3b", "shuffle4")
CONDITIONS = ("normal", *MODES)
PAIRS = tuple(("normal", mode) for mode in MODES) + (("shuffle3a", "shuffle3b"),)
CUTOFFS = (5, 10, 20)
METRICS = (
    *(f"hit@{k}" for k in (1, *CUTOFFS)),
    *(f"{key}@{k}" for k in CUTOFFS for key in ("ndcg", "mrr")),
    *(f"{key}@{k}" for k in CUTOFFS for key in ("ItemHit", "ItemNDCG")),
)
EXPECTED_HASH = {
    "index": "33c87e28c52ae367985bdf494cb49d676d181e13a39497b4ad035f46111546dd",
    "inter": "0a11d47791d90483005da93c36d826679e20e4592d1d2b86518d5f56f218e743",
    "item": "1d9938e576a8d964e6b977a2b63aa56402840a788eaa6685f26f0d910a7133f2",
    "tokenizer": "a65b6872ff61bcfba1e617d065654eba58d678e876ada9547a4c2cc78ba619c4",
    "archived_matrix": "6141f21905aa4544625c62b51dcc5a1bc275214a839b200b6e741409b9c2c197",
    "user_axis": "1daf63ce4c77b0265777a46b61939006aafab0276b9106c742de602c8c7e867a",
    "normal0": "790506cf834bca27f4bb01d35d7d4048d494872f74ff4d9a8bcdac516c475b4d",
    "normal1": "028a74dbab9ba2afaa018994b1518c465d20d58ce05218720634e5c42a4a2a45",
    "assignment0": "b02fc00a77981c4d80c45516785ad2e827991a4455f0319df553a7b8b03102e8",
    "assignment1": "f5884584684e17c33d5e6857203fc8ba72d2e7da681f412429dea3c17c6e845f",
}
OFFLINE_TOLERANCE = 1e-13


def need(ok, detail):
    if not ok:
        raise ValueError(detail)


def read_json(path, digest=None):
    raw = path.read_bytes()
    if digest is not None:
        need(sha256(raw).hexdigest() == digest, f"Frozen file SHA-256 mismatch: {path}")
    return json.loads(raw)


def canonical_digest(items):
    return sha256(json.dumps(items, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def catalog(args):
    documents = {label: read_json(args.data_dir / f"Instruments.{label}.json", EXPECTED_HASH[label])
                 for label in ("index", "inter", "item")}
    index, inter = documents["index"], documents["inter"]
    users = list(inter)
    need(len(index) == 9922 and set(index) == set(documents["item"]), "Item identity")
    need(len(users) == NORMAL and canonical_digest(users) == EXPECTED_HASH["user_axis"], "Global user axis identity")
    need(all(len(parts) == 4 for parts in index.values()), "Four code tokens per item")
    sid_by_item = {item: "".join(parts) for item, parts in index.items()}
    members = defaultdict(list)
    for item, sid in sid_by_item.items():
        members[sid].append(item)
    need(len(members) == 9897 and sum(len(group) > 1 for group in members.values()) == 23, "SID collision identity")
    additions = {row["content"]: row["id"] for row in read_json(args.tokenizer, EXPECTED_HASH["tokenizer"])["added_tokens"]}
    levels = [{additions[parts[p]]: parts[p] for parts in index.values()} for p in range(4)]
    need([len(level) for level in levels] == [150, 256, 256, 256], "Four allowed candidate code sets")
    all_code_ids = set().union(*levels)
    need(len(all_code_ids) == 918 and EOS not in all_code_ids, "Disjoint SID and EOS token IDs")
    targets = [(user, str(inter[user][-1]), sid_by_item[str(inter[user][-1])]) for user in users]
    return users, targets, members, levels


def metric_row(ranked, target, members):
    matching_beams = [rank for rank, sid in enumerate(ranked, 1) if sid == target]
    values = {f"hit@{k}": float(any(rank <= k for rank in matching_beams))
              for k in (1, *CUTOFFS)}
    expanded = []
    for sid in ranked:
        start = len(expanded) + 1
        group = members.get(sid, [None])
        expanded.extend(group)
        if sid == target:
            target_positions = range(start, len(expanded) + 1)
            break
    else:
        target_positions = range(0)
    size = len(members[target])
    for k in CUTOFFS:
        matches = [rank for rank in matching_beams if rank <= k]
        visible_positions = [rank for rank in target_positions if rank <= k]
        values[f"ndcg@{k}"] = math.fsum(1 / math.log2(rank + 1) for rank in matches)
        values[f"mrr@{k}"] = 1 / min(matches) if matches else 0.0
        values[f"ItemHit@{k}"] = len(visible_positions) / size
        values[f"ItemNDCG@{k}"] = math.fsum(1 / math.log2(rank + 1) for rank in visible_positions) / size
    return [values[key] for key in METRICS]


def verify_row(row, rank, targets, levels, members):
    number = row["row_idx"]
    need(type(number) is int and 0 <= number < len(targets), "Global row index")
    uid, item, target = targets[number]
    need((row["user_id"], row["target_item_id"], row["target"], row["prompt_id"], row["rank"]) ==
         (uid, item, target, 0, rank), f"Target, user, prompt, or rank mismatch at row {number}")
    need(row["original_callback_verified_all_tokens"] is True, f"Missing Allowed receipt at row {number}")
    need(len(row["input_token_ids"]) == row["input_nonpadding_tokens"], "Batch-1 input count")
    candidates = row["candidates"]
    need(len(candidates) == BEAMS, f"Wrong candidate count at row {number}")
    sids, scores = [], []
    for generation, candidate in enumerate(candidates):
        sid = candidate["raw_decoded"].rsplit("Response:", 1)[-1].strip().replace(" ", "")
        score = candidate["sequence_score"]
        tokens = candidate["generated_token_ids"]
        need(candidate["generation_order"] == generation and sid == candidate["sid"],
             f"Candidate order or normalization at row {number} beam {generation}")
        need(type(score) is float and math.isfinite(score) and unpack("<f", pack("<f", score))[0] == score,
             f"Candidate score not a finite round-trippable FP32 at row {number} beam {generation}")
        need(candidate["eos_position"] == 4 and len(tokens) >= 5 and all(token == EOS for token in tokens[4:]),
             f"Candidate EOS or padding at row {number} beam {generation}")
        need(all(tokens[p] in levels[p] for p in range(4)), f"Invalid per-code token at row {number} beam {generation}")
        need("".join(levels[p][tokens[p]] for p in range(4)) == sid,
             f"Decoded tokens do not reproduce SID at row {number} beam {generation}")
        sids.append(sid)
        scores.append(score)
    order = sorted(range(BEAMS), key=lambda generation: (-scores[generation], generation))
    ranked = [sids[generation] for generation in order]
    need(order == row["sorted_generation_order"] and row["relevance"] == [int(sid == target) for sid in ranked],
         f"Stored stable rank or relevance differs at row {number}")
    counts = {
        "invalid_complete_sids": sum(sid not in members for sid in ranked),
        "duplicate_complete_sids": len(ranked) - len(set(ranked)),
        "collision_candidate_occurrences": sum(len(members.get(sid, ())) > 1 for sid in ranked),
    }
    return number, metric_row(ranked, target, members), counts


def read_shard(path, rank, targets, levels, members, *, retained=(), expected_digest=None):
    digest, bytes_read = sha256(), 0
    diagnostics, matrix, selected, assignment = Counter(), {}, {}, []
    retained = set(retained)
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            bytes_read += len(line)
            row = json.loads(line)
            number, values, counts = verify_row(row, rank, targets, levels, members)
            need(number not in matrix, f"Duplicate global user row {number} in {path}")
            matrix[number] = values
            assignment.append(number)
            diagnostics.update(counts)
            if number in retained:
                selected[number] = row
    payload_hash = digest.hexdigest()
    if expected_digest is not None:
        need(payload_hash == expected_digest, f"Frozen prediction SHA-256 mismatch: {path}")
    receipt = {"path": str(path), "bytes": bytes_read, "sha256": payload_hash, "rows": len(assignment),
               "candidates": BEAMS * len(assignment), "assignment_sha256": canonical_digest(assignment),
               **diagnostics}
    return matrix, selected, assignment, receipt


def means(matrix):
    return {name: math.fsum(matrix[:, i]) / len(matrix) for i, name in enumerate(METRICS)}


def frozen_normal(args, targets, levels, members, sample):
    normal, subset, receipts = {}, {}, []
    for rank, path in enumerate(args.normal):
        data, picked, assignment, receipt = read_shard(
            path, rank, targets, levels, members, retained=sample,
            expected_digest=EXPECTED_HASH[f"normal{rank}"])
        need(len(assignment) == NORMAL // 2 and not normal.keys() & data.keys(), f"Frozen rank {rank} coverage")
        need(receipt["assignment_sha256"] == EXPECTED_HASH[f"assignment{rank}"], f"Frozen rank {rank} assignment")
        normal.update(data)
        subset.update(picked)
        receipts.append(receipt)
    need(normal.keys() == set(range(NORMAL)), "Incomplete global Normal axis")
    matrix = np.asarray([normal[i] for i in range(NORMAL)], dtype=np.float64)
    need(sha256(args.archived_matrix.read_bytes()).hexdigest() == EXPECTED_HASH["archived_matrix"],
         "Wrong frozen NPZ")

    with np.load(args.archived_matrix, allow_pickle=False) as archived:
        keys = archived["keys"].tolist()
        need(sorted(keys) == sorted(METRICS) and archived["values"].dtype == np.float64, "Frozen metric axis/dtype")
        need(np.array_equal(archived["row_idx"], np.arange(NORMAL)), "Frozen metric row axis")
        need(str(archived["user_axis_sha256"]) == EXPECTED_HASH["user_axis"], "Frozen metric user-axis identity")
        frozen = archived["values"][:, [keys.index(key) for key in METRICS]]
        need(np.array_equal(matrix, frozen),
             f"Frozen matrix differs in {np.count_nonzero(matrix != frozen)} cells, max {np.max(np.abs(matrix-frozen))}")
    overall = means(matrix)
    reference = read_json(args.archived_metrics)["metrics"]
    need(overall == reference, "Frozen full-precision means mismatch")
    receipt = {"users": NORMAL, "candidates": NORMAL * BEAMS, "matrix_shape": list(matrix.shape),
               "matrix_exact": True, "means_exact": True, "metric_means": overall,
               "frozen_shards": receipts, "frozen_user_axis_sha256": EXPECTED_HASH["user_axis"],
               "collision_groups": sum(len(group) > 1 for group in members.values()),
               "target_collision_users": sum(len(members[sid]) > 1 for _, _, sid in targets)}
    return matrix, subset, receipt


def read_sample(path):
    source = read_json(path)
    if isinstance(source, list):
        sample = source
    elif "preflight_assignment" in source:
        sample = source["preflight_assignment"]
    else:
        sample = [row["row_idx"] for rank in ("0", "1") for row in source["ranks"][rank]]
    need(len(sample) == len(set(sample)) == 32, "Expected frozen 32-user sample")
    need(all(type(i) is int and 0 <= i < NORMAL for i in sample), "Sample row was renumbered or not on frozen axis")
    return sample


def array_difference(observed, expected, label, exact=False):
    need(observed.shape == expected.shape and observed.dtype == expected.dtype, f"{label}: array shape/dtype")
    if exact:
        need(np.array_equal(observed, expected), f"{label}: exact comparison")
        return 0.0
    error = float(np.max(np.abs(observed - expected)))
    need(error <= OFFLINE_TOLERANCE, f"{label}: difference {error} exceeds {OFFLINE_TOLERANCE}")
    return error


def bootstrap(matrices):
    rng = np.random.Generator(np.random.PCG64(20261113))
    n = matrices.shape[1]
    sample = rng.integers(0, n, size=(5000, n), dtype=np.int64)
    weights = np.stack([np.bincount(row, minlength=n) for row in sample])
    differences = np.asarray([matrices[CONDITIONS.index(right)] - matrices[CONDITIONS.index(left)]
                              for left, right in PAIRS])
    draws = np.stack([weights @ comparison / n for comparison in differences])
    ordered = draws.copy()
    ordered.sort(axis=1)
    limits = []
    for percentile in (0.025, 0.975):
        location = percentile * (len(sample) - 1)
        low, high = math.floor(location), math.ceil(location)
        limits.append(ordered[:, low] * (high - location) + ordered[:, high] * (location - low))
    return differences, sample, draws, limits


def natural_control(shuffled, original):
    same = [i for i in shuffled if shuffled[i]["input_token_ids"] == original[i]["input_token_ids"]]
    difference_users = []
    for i in same:
        row, control = shuffled[i], original[i]
        mutable = ("candidates", "sorted_generation_order", "relevance")
        mismatches = [key for key in mutable if row[key] != control[key]]
        if mismatches:
            scores = [abs(a["sequence_score"] - b["sequence_score"])
                      for a, b in zip(row["candidates"], control["candidates"], strict=True)]
            difference_users.append({"row_idx": i, "fields": mismatches, "max_generation_order_score_difference": max(scores)})
    return {"unchanged_input_users": same, "candidate_exact_users": len(same) - len(difference_users),
            "differences": difference_users}


def check_paired(args, users, targets, levels, members, sample, normal_matrix, original):
    matrices = [normal_matrix[sample]]
    receipts = {}
    for mode, path in zip(MODES, args.preflight, strict=True):
        rows, raw, assignment, receipt = read_shard(path, 0, targets, levels, members, retained=sample)
        need(assignment == sample, f"{mode} preflight does not follow the ordered frozen 32-user axis")
        matrices.append(np.asarray([rows[i] for i in sample], dtype=np.float64))
        receipts[mode] = {**receipt, "identity_controls": natural_control(raw, original)}
    matrices = np.stack(matrices)
    estimates = {mode: means(matrix) for mode, matrix in zip(CONDITIONS, matrices, strict=True)}
    differences, indices, draws, (low, high) = bootstrap(matrices)
    digest = sha256(indices.astype("<i8", copy=False).tobytes(order="C")).hexdigest()
    summary = read_json(args.paired_summary)
    norm = summary["bootstrap"]
    need((norm["rng"], norm["seed"], norm["replicates"], norm["replacement"], norm["percentile_method"],
          norm["bounds"], norm["multiple_comparison_adjustment"], norm["draw_indices_sha256_little_endian"]) ==
         ("NumPy PCG64", 20261113, 5000, True, "linear", [2.5, 97.5], False, digest),
         "Bootstrap RNG, seed, repetitions, replacement, percentile, adjustment, or shared draw identity")
    need(summary["metrics"] == estimates, "All 80 condition means must match the raw-input results exactly")
    comparison_names = [f"{right}-{left}" for left, right in PAIRS]
    need(set(summary["comparisons"]) == set(comparison_names), "Paired comparison identities")
    axis = canonical_digest([users[i] for i in sample])
    checks = {}
    with np.load(args.paired_arrays, allow_pickle=False) as raw:
        keys = raw["keys"].tolist()
        need(sorted(keys) == sorted(METRICS), "Paired metric columns")
        map_metrics = [keys.index(key) for key in METRICS]
        need(raw["condition"].tolist() == list(CONDITIONS), "Paired condition object order")
        need(raw["comparison"].tolist() == comparison_names, "Paired comparison array order")
        need(np.array_equal(raw["row_idx"], sample) and str(raw["user_axis_sha256"]) == axis,
             "Preflight sample row and user-axis identity")
        checks["per_user_values"] = array_difference(raw["per_user_values"][..., map_metrics], matrices, "Per-user matrices", exact=True)
        checks["per_user_differences"] = array_difference(raw["per_user_differences"][..., map_metrics], differences,
                                                          "Per-user differences", exact=True)
        checks["bootstrap_difference_means"] = array_difference(
            raw["bootstrap_difference_means"][..., map_metrics], draws, "Bootstrap draws")
        if args.paired_indices:
            shared_indices = np.load(args.paired_indices, allow_pickle=False)
        else:
            need("bootstrap_indices" in raw.files, "Raw bootstrap index array required")
            shared_indices = raw["bootstrap_indices"]
        checks["bootstrap_indices"] = array_difference(shared_indices, indices, "Shared bootstrap indices", exact=True)
    max_ci_difference = 0.0
    for pair_num, (left, right) in enumerate(PAIRS):
        records = summary["comparisons"][comparison_names[pair_num]]
        need(len(records) == 16 and {v["metric"] for v in records} == set(METRICS), "80 unique comparison-by-metric records")
        for record in records:
            key = record["metric"]
            col = METRICS.index(key)
            a, b = estimates[left][key], estimates[right][key]
            need((a, b) == (record["left"], record["right"]), f"{left}/{right}/{key} point estimates")
            need(b - a == record["absolute_difference"], f"{left}/{right}/{key} proportional difference")
            need(100 * (b - a) == record["percentage_point_difference"], f"{left}/{right}/{key} percentage-point difference")
            need(((b - a) / a if a else None) == record["relative_change_fraction"], f"{left}/{right}/{key} relative change")
            tolerance = max(abs(record["difference_CI95_low"] - low[pair_num, col]),
                            abs(record["difference_CI95_high"] - high[pair_num, col]))
            need(tolerance <= OFFLINE_TOLERANCE, f"{left}/{right}/{key} interval differs by {tolerance}")
            max_ci_difference = max(max_ci_difference, tolerance)
    return {"users_per_condition": len(sample), "sample_user_axis_sha256": axis, "global_rows": sample,
            "preflight_candidates": BEAMS * len(sample) * len(MODES), "comparisons_checked": len(PAIRS) * len(METRICS),
            "max_ci_difference": max_ci_difference, "array_comparisons_max_differences": checks,
            "bootstrap_index_sha256_little_endian": digest, "condition_receipts": receipts}


def main():
    start = monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory with three frozen Instruments JSON files")
    parser.add_argument("--tokenizer", type=Path, required=True, help="Adapter tokenizer.json")
    parser.add_argument("--normal", nargs=2, type=Path, required=True, help="Normal raw rank 0 then rank 1 JSONL")
    parser.add_argument("--archived-matrix", type=Path, required=True, help="Frozen Normal per_user_metrics.npz")
    parser.add_argument("--archived-metrics", type=Path, required=True, help="Frozen Normal primary metrics.json")
    parser.add_argument("--samples", type=Path, help="Four-mode ordered preflight sample JSON")
    parser.add_argument("--preflight", type=Path, nargs=4, help="Four V2 raw prediction shards in S2 S3a S3b S4 order")
    parser.add_argument("--paired-summary", type=Path)
    parser.add_argument("--paired-arrays", type=Path)
    parser.add_argument("--paired-indices", type=Path, help="Alternative separate raw bootstrap indices .npy")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    need(not args.output.exists(), "Preserve earlier audit receipts")
    optional = tuple(map(bool, (args.samples, args.preflight, args.paired_summary, args.paired_arrays)))
    need(not any(optional) or all(optional), "Full paired audit requires samples, four raw shards, summary and arrays")
    users, targets, members, levels = catalog(args)
    sample = read_sample(args.samples) if args.samples else ()
    matrix, original, normal = frozen_normal(args, targets, levels, members, sample)
    paired = check_paired(args, users, targets, levels, members, sample, matrix, original) if args.preflight else None
    report = {"status": "PASS", "audit": "B independent raw candidate and paired metric audit",
              "normal": normal, "paired": paired, "elapsed_seconds": monotonic() - start,
              "numpy": np.__version__, "program_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "normal_users": normal["users"],
                      "normal_candidates": normal["candidates"], "paired_rows": paired["comparisons_checked"] if paired else None,
                      "elapsed_seconds": report["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
