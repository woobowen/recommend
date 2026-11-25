"""Independent standard-library replay of the entire formal candidate export."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import time

USERS = 24_772
ITEMS = 9_922
K_VALUES = (5, 10)
NATIVE_KEYS = {f"{name}@{k}" for name in ("recall", "ndcg") for k in K_VALUES}
INPUT_ROOT = Path("third_party/LETTER-8d0154e/data/Instruments")
FP64_TOLERANCE = 1e-14
NATIVE_FP32_TOLERANCE = 24 * 2**-24
DISCOUNT = {rank: 1 / math.log2(rank + 1) for rank in range(1, 11)}


def read_json(path):
    return json.loads(path.read_bytes())


def read_axes(root, request):
    argv = request["argv"]
    server_root = argv[argv.index("--assets") + 1] + "/"
    inputs = {i["path"].removeprefix(server_root): i["sha256"]
              for i in request["inputs"] if i["path"].startswith(server_root)}
    assert len(inputs) == 9
    for path, expected in inputs.items():
        with (root / path).open("rb") as file:
            assert hashlib.file_digest(file, "sha256").hexdigest() == expected, path
    index, sequences, catalog = [read_json(root / INPUT_ROOT / f"Instruments.{kind}.json")
                                 for kind in ("index", "inter", "item")]
    codes = {}
    for item, words in index.items():
        codes[int(item)] = tuple(int(re.fullmatch(fr"<{symbol}_(\d+)>", word)[1])
                                 for word, symbol in zip(words, "abcd", strict=True))
    assert set(codes) == set(map(int, catalog)) == set(range(ITEMS))
    assert all(all(0 <= code < 256 for code in sid) for sid in codes.values())
    groups = Counter(codes.values())
    assert len(groups) == 9897 and sum(n > 1 for n in groups.values()) == 23
    histories = {int(key): seq for key, seq in sequences.items()}
    assert set(histories) == set(range(USERS)) and sum(map(len, histories.values())) == 206153
    assert sum(groups[codes[seq[-1]]] > 1 for seq in histories.values()) == 122
    return codes, histories, groups, len(inputs)


def mean_gain(ranks):
    return math.fsum(DISCOUNT[rank] for rank in ranks)


def native_rank_metrics(scores, raw_sids, target, observed):
    assert set(observed) == NATIVE_KEYS
    score_order = sorted(scores, reverse=True)
    positive = set(i for i, sid in enumerate(raw_sids) if sid == target)
    result = {}
    for metric, positions in observed.items():
        name, cutoff = metric.split("@")
        k = int(cutoff)
        assert len(positions) == len(set(positions)) == k
        assert all(type(i) is int and 0 <= i < len(scores) for i in positions)
        assert [scores[i] for i in positions] == score_order[:k], (metric, positions)
        observed_hits = [rank for rank, slot in enumerate(positions, 1) if slot in positive]
        ideal_count = min(len(positive), k)
        if name == "recall":
            result[metric] = len(observed_hits) / (ideal_count or 1)
        else:
            result[metric] = mean_gain(observed_hits) / (mean_gain(range(1, ideal_count + 1)) or 1)
    return result


def stable_rank_metrics(scores, sids, target, groups):
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    ranked = [sids[i] for i in order]
    all_hit_ranks = [rank for rank, sid in enumerate(ranked, 1) if sid == target]
    first = next(iter(all_hit_ranks), 0)
    if first:
        occupied = sum(groups.get(sid, 1) for sid in ranked[:first - 1])
        group_slots = range(occupied + 1, occupied + groups[target] + 1)
    else:
        group_slots = ()
    result = {"hit@1": float(first == 1)}
    for k in K_VALUES:
        ranks = [r for r in all_hit_ranks if r <= k]
        visible_group_slots = [r for r in group_slots if r <= k]
        result[f"hit@{k}"] = float(bool(ranks))
        result[f"mrr@{k}"] = 1 / ranks[0] if ranks else 0.0
        result[f"ndcg@{k}"] = mean_gain(ranks)
        result[f"ItemHit@{k}"] = len(visible_group_slots) / groups[target]
        result[f"ItemNDCG@{k}"] = mean_gain(visible_group_slots) / groups[target]
    return order, first, result


def differences(a, b, tolerance=FP64_TOLERANCE):
    assert a.keys() == b.keys()
    deltas = {key: abs(a[key] - b[key]) for key in a}
    assert all(math.isfinite(d) and d <= tolerance for d in deltas.values()), deltas
    return deltas


def check_cutoffs():
    # SID target first appears at 5, tied with position 6's duplicate target.
    # Expanded CCE reaches only one of three target slots at cutoff 10.
    scores = (.9, .8, .7, .6, .5, .5, .4, .3, .2, .1)
    sids = ("group", "group", "unknown", "other", "target", "target", "other", "other", "other", "other")
    order, first, result = stable_rank_metrics(scores, sids, "target", {"group": 3, "target": 3, "other": 2})
    assert order == list(range(10)) and first == 5
    assert result["hit@5"] == 1 and result["mrr@5"] == .2 and result["ItemHit@5"] == 0
    assert result["ItemHit@10"] == 1 / 3 and result["ItemNDCG@10"] == DISCOUNT[10] / 3
    assert result["ndcg@10"] == math.fsum([DISCOUNT[5], DISCOUNT[6]])
    native = native_rank_metrics(
        scores, sids, "target",
        {key: [0, 1, 2, 3, 5] if key.endswith("@5") else order for key in NATIVE_KEYS},
    )
    assert native["recall@5"] == .5 and native["recall@10"] == 1
    assert native["ndcg@5"] == DISCOUNT[5] / math.fsum([DISCOUNT[1], DISCOUNT[2]])


def replay(assets, job):
    started = time.monotonic()
    request = read_json(job / "request.json")
    assert request["scope"]["mode"] == "formal" and request["scope"]["test_users"] == USERS
    codes, histories, groups, bound_assets = read_axes(assets, request)
    run = job / "output/run"
    distributions = defaultdict(list)
    diagnostics = Counter(dict.fromkeys((
        "candidates", "out_of_catalog", "duplicate_sids", "tied_score_rows",
        "native_stable_order_differences", "collision_target_rows"), 0))
    max_user_error = 0.0
    repeated_target_rows = 0
    compared = 0
    with (run / "candidates.jsonl").open() as candidates, (run / "per_user_metrics.jsonl").open() as recorded:
        for uid, (raw, user_metrics) in enumerate(zip(candidates, recorded, strict=True)):
            record, reference = json.loads(raw), json.loads(user_metrics)
            assert record["global_row"] == reference["global_row"] == uid
            assert record["user_id"] == str(uid) and uid < USERS
            history = histories[uid][-21:]
            target = codes[history[-1]]
            assert record["history_item_ids"] == history[:-1] and record["target_item_id"] == history[-1]
            assert record["target_sid"] == list(target)
            flattened = [code for item in history[:-1] for code in codes[item]]
            n = len(flattened)
            assert record["input_ids"] == flattened + [-1] * (120 - n)
            assert record["input_mask"] == [1] * n + [0] * (120 - n)
            sids = [tuple(values) for values in record["ids"]]
            assert len(sids) == 10
            assert all(len(sid) == 4 and all(type(v) is int and 0 <= v < 256 for v in sid) for sid in sids)
            word = bytes.fromhex(record["scores_f32le"])
            assert len(word) == 40 and word.hex() == record["scores_f32le"]
            scores = struct.unpack("<10f", word)
            assert all(math.isfinite(score) and 0 <= score <= 1 for score in scores)
            order, first, unified = stable_rank_metrics(scores, sids, target, groups)
            assert reference["unified_order"] == order and reference["first_sid_rank"] == first
            native = native_rank_metrics(scores, sids, target, record["native_topk"])
            actual = {key: v for key, v in reference.items()
                      if key not in ("global_row", "unified_order", "first_sid_rank", "native")}
            errors = differences(native, reference["native"]) | differences(unified, actual)
            max_user_error = max(max_user_error, *errors.values())
            for group, metrics in (("native", native), ("unified", unified)):
                for key, value in metrics.items():
                    distributions[group, key].append(value)
            repeated_target_rows += sids.count(target) > 1
            diagnostics.update(candidates=10, out_of_catalog=sum(sid not in groups for sid in sids),
                               duplicate_sids=10 - len(set(sids)), tied_score_rows=len(set(scores)) < 10,
                               native_stable_order_differences=any(
                                   ranks != order[:len(ranks)] for ranks in record["native_topk"].values()),
                               collision_target_rows=groups[target] > 1)
            compared += 1
    assert compared == USERS
    replayed = {group: {key: math.fsum(values) / USERS for (g, key), values in distributions.items() if g == group}
                for group in ("native", "unified")}
    published = read_json(run / "offline_evaluation.json")["replay"]
    assert published["coverage"] == USERS and published["statistics"] == dict(diagnostics)
    for group, metrics in replayed.items():
        differences(metrics, published["metrics"][group])
    [native_gpu] = read_json(run / "native_metrics.json")
    assert set(native_gpu) == {"test/loss", *(f"test/{key}" for key in NATIVE_KEYS)}
    assert math.isfinite(native_gpu["test/loss"])
    deltas = differences(replayed["native"], {key: native_gpu[f"test/{key}"] for key in NATIVE_KEYS},
                         NATIVE_FP32_TOLERANCE)
    differences(deltas, published["metrics"]["native_abs_differences"])
    return {
        "users": compared, "candidates": diagnostics["candidates"], "asset_hashes_checked": bound_assets,
        "per_user_metrics_checked": compared * 15, "statistics": dict(diagnostics),
        "repeated_target_rows": repeated_target_rows, "metrics": replayed,
        "max_per_user_error": max_user_error, "native_gpu_replay_absolute_differences": deltas,
        "replay_only_cpu_seconds": time.monotonic() - started,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    parser.add_argument("job", type=Path)
    args = parser.parse_args()
    check_cutoffs()
    print(json.dumps(replay(args.assets, args.job), indent=2, sort_keys=True))
