#!/usr/bin/env python3
"""Independently audit the bounded GRID Instruments candidates using the Python standard library."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys


RAW = Path("third_party/LETTER-8d0154e/data/Instruments")
EXPECTED_HASHES = {
    "index": "33c87e28c52ae367985bdf494cb49d676d181e13a39497b4ad035f46111546dd",
    "inter": "0a11d47791d90483005da93c36d826679e20e4592d1d2b86518d5f56f218e743",
    "item": "1d9938e576a8d964e6b977a2b63aa56402840a788eaa6685f26f0d910a7133f2",
}
CUTOFFS = (5, 10)
NATIVE_KEYS = {f"{metric}@{k}" for metric in ("recall", "ndcg") for k in CUTOFFS}
DISCOUNTS = {rank: math.log(2) / math.log(rank + 1) for rank in range(1, 11)}
NATIVE_FP32_TOLERANCE = 24 * 2**-24
FP64_TOLERANCE = 1e-14  # Python log versus log2 and summation association.


def digest(path):
    with path.open("rb") as stream:
        if hasattr(hashlib, "file_digest"):
            return hashlib.file_digest(stream, "sha256").hexdigest()
        sha = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
        return sha.hexdigest()


def read_json(path):
    return json.loads(path.read_bytes())


def read_jsonl(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def load_axes(assets):
    raw = {kind: assets / RAW / f"Instruments.{kind}.json" for kind in EXPECTED_HASHES}
    assert {kind: digest(path) for kind, path in raw.items()} == EXPECTED_HASHES
    index, inter, items = (read_json(raw[kind]) for kind in ("index", "inter", "item"))
    codes = {}
    for raw_item, tokens in index.items():
        assert len(tokens) == 4
        code = tuple(int(re.fullmatch(fr"<{symbol}_(\d+)>", token)[1])
                     for symbol, token in zip("abcd", tokens, strict=True))
        assert all(0 <= value < 256 for value in code)
        codes[int(raw_item)] = code
    assert set(codes) == set(map(int, items)) == set(range(9922))
    history = {int(user): seq for user, seq in inter.items()}
    assert set(history) == set(range(24772))
    groups = Counter(codes.values())
    assert len(groups) == 9897 and sum(size > 1 for size in groups.values()) == 23
    assert sum(len(seq) for seq in history.values()) == 206153
    assert sum(groups[codes[seq[-1]]] > 1 for seq in history.values()) == 122
    # The sample was already ordered and frozen in the previous delivery.
    axis = [*range(19), 24771]
    prep = assets / "codex/letter_official_downstream_20261116/inputs/instruments/preparation.json"
    assert read_json(prep)["preflight_test_users"] == axis
    return codes, history, groups, axis


def score_values(raw_hex):
    raw = bytes.fromhex(raw_hex)
    assert raw.hex() == raw_hex and len(raw) == 40
    decoded = struct.unpack("<10f", raw)
    assert all(math.isfinite(x) and 0 <= x <= 1 for x in decoded)
    return decoded


def validate_native_selection(scores, indices, k):
    if len(indices) != k or len(set(indices)) != k or any(type(i) is not int or not 0 <= i < 10 for i in indices):
        raise ValueError(f"invalid {k} candidate indices: {indices}")
    selected = [scores[i] for i in indices]
    expected = sorted(scores)[-k:]
    if Counter(selected) != Counter(expected) or selected != sorted(selected, reverse=True):
        raise ValueError(f"invalid native top-{k} selection: {indices}")


def native_metrics(candidates, target, scores, rankings):
    assert set(rankings) == NATIVE_KEYS
    correct = {i for i, item in enumerate(candidates) if item == target}
    answer = {}
    for name, indices in rankings.items():
        measure, cut = name.split("@")
        k = int(cut)
        validate_native_selection(scores, indices, k)
        ideal_count = min(len(correct), k)
        hits = [rank for rank, i in enumerate(indices, 1) if i in correct]
        if measure == "recall":
            answer[name] = len(hits) / max(1, ideal_count)
        else:
            ideal_gain = sum(DISCOUNTS[rank] for rank in range(1, ideal_count + 1))
            answer[name] = sum(DISCOUNTS[rank] for rank in hits) / (ideal_gain or 1)
    return answer


def unified_metrics(candidates, target, scores, groups):
    # The index breaks equal-score ties in original emission order.
    order = sorted(range(10), key=lambda i: (-scores[i], i))
    sids = [candidates[i] for i in order]
    positions = [rank for rank, sid in enumerate(sids, 1) if sid == target]
    first = min(positions, default=0)
    # Expand each ranked SID's collision group into occupied slots, including
    # one slot for an unknown SID. The first target group supplies the credits.
    target_slots = []
    for sid in sids:
        weight = groups.get(sid, 1)
        target_slots.extend([sid == target] * weight)
        if sid == target:
            break
    result = {"hit@1": float(first == 1)}
    for k in CUTOFFS:
        visible = [rank for rank in positions if rank <= k]
        result[f"hit@{k}"] = float(bool(visible))
        result[f"ndcg@{k}"] = sum(DISCOUNTS[rank] for rank in visible)
        result[f"mrr@{k}"] = 1 / visible[0] if visible else 0.0
        expanded = [rank for rank, hit in enumerate(target_slots[:k], 1) if hit]
        result[f"ItemHit@{k}"] = len(expanded) / groups[target]
        result[f"ItemNDCG@{k}"] = sum(DISCOUNTS[rank] for rank in expanded) / groups[target]
    return result, order, first


def compare_metrics(actual, expected, tolerance=FP64_TOLERANCE):
    assert set(actual) == set(expected), (set(actual), set(expected))
    differences = {k: abs(actual[k] - value) for k, value in expected.items()}
    assert all(math.isfinite(delta) and delta <= tolerance for delta in differences.values()), differences
    return differences


def verify_fixtures(experiment=None):
    scores = [.9, .8, .7, .6, .5, .5, .4, .3, .2, .1]
    legal_indices, bad_indices = [0, 1, 2, 3, 5], [0, 2, 3, 4, 5]
    validate_native_selection(scores, legal_indices, 5)
    try:
        validate_native_selection(scores, bad_indices, 5)
    except ValueError:
        pass
    else:
        raise AssertionError("independent verifier accepted missing rank 2")
    if experiment:
        sys.path.insert(0, str(experiment.resolve()))
        from evaluate import native

        targets = [1, 1, 1, 1]
        preds = [[0, 0, 0, i] for i in range(10)]
        preds[5] = targets
        selections = {name: (legal_indices if name.endswith("@5") else list(range(10))) for name in NATIVE_KEYS}
        row = {"target_sid": targets, "ids": preds, "native_topk": selections}
        assert math.isclose(native(row, scores)["ndcg@5"], DISCOUNTS[5], abs_tol=FP64_TOLERANCE)
        row["native_topk"]["recall@5"] = bad_indices
        try:
            native(row, scores)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("published evaluator accepted missing rank 2")
    return {"scores": scores, "legal_top5": legal_indices, "illegal_top5": bad_indices,
            "independent_positive": "PASS", "independent_negative": "PASS",
            "published_adapter_verified": experiment is not None}


def replay(assets, work):
    codes, history, groups, axis = load_axes(assets)
    raw = read_jsonl(work / "candidates.jsonl")
    per_user = read_jsonl(work / "per_user_metrics.jsonl")
    assert [r["global_row"] for r in raw] == [r["global_row"] for r in per_user] == axis
    independent = {"native": [], "unified": []}
    diagnostics = Counter(dict.fromkeys(("candidates", "out_of_catalog", "duplicate_sids", "tied_score_rows",
                                        "native_stable_order_differences", "collision_target_rows"), 0))
    max_per_user_diff = 0.0
    for uid, row, report in zip(axis, raw, per_user, strict=True):
        seq = history[uid][-21:]
        target = codes[seq[-1]]
        flattened = [word for item in seq[:-1] for word in codes[item]]
        pad = 120 - len(flattened)
        assert row["user_id"] == str(uid) and row["target_item_id"] == seq[-1]
        assert row["target_sid"] == list(target) and row["history_item_ids"] == seq[:-1]
        assert row["input_ids"] == flattened + [-1] * pad and row["input_mask"] == [1] * len(flattened) + [0] * pad
        candidates = [tuple(values) for values in row["ids"]]
        assert len(candidates) == 10
        assert all(len(sid) == 4 and all(type(i) is int and 0 <= i < 256 for i in sid) for sid in candidates)
        scores = score_values(row["scores_f32le"])
        native = native_metrics(candidates, target, scores, row["native_topk"])
        unified, order, first = unified_metrics(candidates, target, scores, groups)
        assert report["first_sid_rank"] == first and report["unified_order"] == order
        expected = {key: value for key, value in report.items() if key not in
                    {"global_row", "native", "first_sid_rank", "unified_order"}}
        errors = compare_metrics(native, report["native"]) | compare_metrics(unified, expected)
        max_per_user_diff = max(max_per_user_diff, *errors.values())
        independent["native"].append(native)
        independent["unified"].append(unified)
        diagnostics.update(candidates=10, out_of_catalog=sum(sid not in groups for sid in candidates),
                           duplicate_sids=10 - len(set(candidates)), tied_score_rows=len(set(scores)) < 10,
                           native_stable_order_differences=any(
                               indices != order[:len(indices)] for indices in row["native_topk"].values()),
                           collision_target_rows=groups[target] > 1)
    average = {key: {metric: math.fsum(row[metric] for row in rows) / len(axis) for metric in rows[0]}
               for key, rows in independent.items()}
    offline = read_json(work / "offline_evaluation.json")["replay"]
    assert offline["coverage"] == len(axis)
    assert offline["statistics"] == dict(diagnostics)
    for key in average:
        compare_metrics(average[key], offline["metrics"][key])
    (native_full,) = read_json(work / "native_metrics.json")
    assert set(native_full) == {f"test/{k}" for k in NATIVE_KEYS} | {"test/loss"}
    assert math.isfinite(native_full["test/loss"])
    differences = compare_metrics(average["native"], {k: native_full[f"test/{k}"] for k in NATIVE_KEYS},
                                  NATIVE_FP32_TOLERANCE)
    compare_metrics(differences, offline["metrics"]["native_abs_differences"])
    return raw, {"coverage": len(axis), "user_axis": axis, "statistics": dict(diagnostics),
                 "independent_metrics": average, "max_per_user_abs_error": max_per_user_diff,
                 "native_absolute_errors": differences, "native_max_abs_error": max(differences.values()),
                 "native_fp32_tolerance": NATIVE_FP32_TOLERANCE, "offline_fp64_tolerance": FP64_TOLERANCE}


def verify_observers(work, candidates, export_manifest=None):
    enabled = work / "observer_enabled.json"
    disabled = work / "observer_disabled.json"
    payload = enabled.read_bytes()
    sha = digest(enabled)
    if disabled.is_file():
        assert payload == disabled.read_bytes()
        evidence = "byte-for-byte comparison of both runtime files"
    else:
        assert export_manifest is not None, "missing disabled observer and runner export manifest"
        exported = {entry["path"]: entry for entry in read_json(export_manifest)["files"]}
        for name in ("observer_enabled.json", "observer_disabled.json"):
            entry = exported[f"output/run/{name}"]
            assert entry["sha256"] == sha and entry["bytes"] == len(payload)
        evidence = "runtime export manifest binds both files to the delivered bytes"
    observation = json.loads(payload)
    assert set(observation) == {"batches", "metrics", "rng"}
    assert observation["metrics"] == read_json(work / "native_metrics.json")
    batches = observation["batches"]
    counts = [len(batch["user_ids"]) for batch in batches]
    assert counts == [8, 8, 4]
    start = 0
    for count, batch in zip(counts, batches, strict=True):
        rows = candidates[start:start + count]
        assert batch["user_ids"] == [r["global_row"] for r in rows]
        for key in ("input_ids", "input_mask", "ids", "scores_f32le"):
            assert batch[key] == [r[key] for r in rows], (start, key)
        trace = batch["decoder"]
        assert [frame["prefix_width"] for frame in trace] == [0, 1, 2, 3]
        assert trace[0]["cache_layers_before"] == [0, 0]
        for frame in trace:
            assert len(frame["cache_layers_before"]) == 2
            assert all(type(x) is int and x >= 0 for x in frame["cache_layers_before"])
        start += count

    rng = observation["rng"]
    assert set(rng) == {"torch_cpu", "torch_cuda", "python", "numpy"}
    for key in ("torch_cpu", "torch_cuda"):
        words = rng[key]
        assert words and all(type(i) is int and 0 <= i <= 255 for i in words)
    version, python_words, _ = rng["python"]
    assert version == 3 and len(python_words) == 625
    algorithm, numpy_words, offset, _, _ = rng["numpy"]
    assert algorithm == "MT19937" and len(numpy_words) == 624 and 0 <= offset <= 624
    state_hashes = {name: hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()
                    for name, value in rng.items()}

    events = read_jsonl(work / "events.jsonl")
    endings = [event for event in events if event["event"] == "test_end"]
    assert [event["evaluation"] for event in endings] == [1, 2]
    for n, ending in enumerate(endings, 1):
        batches = [e for e in events if e["event"] == "test_batch" and e["evaluation"] == n]
        assert [e["index"] for e in batches] == [0, 1, 2] and [e["size"] for e in batches] == counts
        assert ending["batch_sizes"] == {"8": 2, "4": 1}
        assert ending["shuffle_calls"] == 0
        assert ending["metric_denominators"] == dict.fromkeys(NATIVE_KEYS, 20)

    stage = read_json(work / "stage_result.json")
    assert stage["mode"] == "preflight" and stage["test"] == "Normal" and stage["eval_seed"] == 2026
    assert stage["shuffle_calls"] == 0
    assert stage["observer_equivalence"]["candidate_values_float32_bits_metrics_rng_equal"] is True
    checkpoint = stage["checkpoint"]
    best = (work / checkpoint["path"]).resolve()
    assert best.is_relative_to(work.resolve()) and best.stat().st_size == checkpoint["bytes"]
    assert digest(best) == checkpoint["sha256"]
    for phase in ("enabled", "disabled"):
        load = read_json(work / f"reload_{phase}.json")
        assert load["strict"] is True and load["keys"] > 0
        assert load["missing"] == [] and load["unexpected"] == []
        assert load["model_state_sha256"] == checkpoint["model_state_sha256"]
        assert load["checkpoint_sha256"] == checkpoint["sha256"]
    return {"comparison": evidence, "both_sha256": sha, "batch_sizes": counts,
            "generation_calls_per_pass": len(counts), "decoder_calls_per_pass": 4 * len(counts),
            "rng_sha256": state_hashes, "checkpoint_sha256": checkpoint["sha256"],
            "model_state_sha256": checkpoint["model_state_sha256"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--export-manifest", type=Path)
    parser.add_argument("--fixtures-only", action="store_true")
    args = parser.parse_args()
    output = {"top5_fixture": verify_fixtures(args.experiment)}
    if not args.fixtures_only:
        if not (args.assets and args.run):
            parser.error("--assets and --run are required for the real replay")
        rows, report = replay(args.assets, args.run)
        output["real_candidate_replay"] = report
        output["observers"] = verify_observers(args.run, rows, args.export_manifest)
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
