"""Beam-10 input adapter for the frozen SID and CCE kernels, plus native replay."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from candidates import rows, scores
from data_assets import official, sample_users
from frozen_eval.offline_metrics import ranking, cce
from frozen_eval.official_cce import compute_cce_metrics

NATIVE = {"recall@5", "recall@10", "ndcg@5", "ndcg@10"}
# Float32 uses u=2**-24. Up to 10 positive terms, their ratio and the final sum/divide.
FP32_TOLERANCE = 24 * 2**-24


def coverage(got, expected):
    if got != expected:
        counts = Counter(got)
        raise ValueError({"missing": sorted(set(expected) - set(got)),
                          "unexpected": sorted(set(got) - set(expected)),
                          "duplicates": sorted(uid for uid, n in counts.items() if n != 1),
                          "order_matches": False})


def stable_order(values):
    # Frozen Beam20 evaluator uses Python stable descending score sort.
    return sorted(range(len(values)), key=values.__getitem__, reverse=True)


def native(record, values):
    flags = [sid == record["target_sid"] for sid in record["ids"]]
    result = {}
    assert set(record["native_topk"]) == NATIVE
    for name, order in record["native_topk"].items():
        k = int(name.split("@")[1])
        assert len(order) == len(set(order)) == k and all(0 <= i < 10 for i in order)
        assert all(values[a] >= values[b] for a, b in zip(order, order[1:]))
        assert values[order[-1]] == sorted(values, reverse=True)[k - 1]
        total = min(k, sum(flags))
        relevance = [flags[i] for i in order]
        if name.startswith("recall"):
            result[name] = sum(relevance) / total if total else 0.0
        else:
            gain = math.fsum(1 / math.log2(i + 2) for i, hit in enumerate(relevance) if hit)
            ideal = math.fsum(1 / math.log2(i + 2) for i in range(total))
            result[name] = gain / ideal if ideal else 0.0
    return result


def unified(sids, target, sizes):
    sid, first = ranking(sids, target)
    item = compute_cce_metrics(sids, target, sizes, [5, 10])
    secondary = cce(sids, target, sizes)
    assert all(abs(value - secondary[key]) <= 1e-14 for key, value in item.items())
    return {key: value for key, value in sid.items() if not key.endswith("@20")} | item, first


def verify(root, path, mode, native_metrics=None, per_user=None):
    codes, seqs, sizes = official(root)
    expected = sample_users(codes, seqs, sizes) if mode == "preflight" else list(range(len(seqs)))
    coverage([row["global_row"] for row in rows(path)], expected)
    statistics = Counter({key: 0 for key in ("candidates", "out_of_catalog", "duplicate_sids",
                                              "tied_score_rows", "native_stable_order_differences",
                                              "collision_target_rows")})
    metrics = {"native": [], "unified": []}
    for record in rows(path):
        uid = record["global_row"]
        sequence = seqs[uid][-21:]
        target = codes[sequence[-1]]
        history = [value for item in sequence[:-1] for value in codes[item]]
        assert record["user_id"] == str(uid) and record["target_item_id"] == sequence[-1]
        assert record["target_sid"] == list(target) and record["history_item_ids"] == sequence[:-1]
        assert record["input_ids"] == history + [-1] * (120 - len(history))
        assert record["input_mask"] == [1] * len(history) + [0] * (120 - len(history))
        values = scores(record)
        assert len(record["ids"]) == 10 and all(math.isfinite(x) and 0 <= x <= 1 for x in values)
        sids = [tuple(sid) for sid in record["ids"]]
        assert all(len(sid) == 4 and all(type(v) is int and 0 <= v < 256 for v in sid) for sid in sids)
        order = stable_order(values)
        ranked = [sids[i] for i in order]
        native_user = native(record, values)
        unified_user, first = unified(ranked, target, sizes)
        metrics["native"].append(native_user)
        metrics["unified"].append(unified_user)
        if per_user:
            per_user.write(json.dumps({"global_row": uid, "unified_order": order, "first_sid_rank": first,
                                      "native": native_user, **unified_user}, allow_nan=False) + "\n")
        statistics.update(candidates=10, out_of_catalog=sum(sid not in sizes for sid in sids),
                          duplicate_sids=10 - len(set(sids)), tied_score_rows=len(set(values)) != 10,
                          native_stable_order_differences=any(
                              indexes != order[:len(indexes)] for indexes in record["native_topk"].values()),
                          collision_target_rows=sizes[target] > 1)
    average = {group: {key: math.fsum(row[key] for row in records) / len(expected)
                       for key in records[0]} for group, records in metrics.items()}
    if native_metrics is not None:
        differences = {k: abs(v - native_metrics["test/" + k]) for k, v in average["native"].items()}
        assert max(differences.values()) <= FP32_TOLERANCE, differences
        average["native_abs_differences"] = differences
    return {"coverage": len(expected), "statistics": dict(statistics), "metrics": average}


def fixtures():
    target, other, invalid = (1, 2, 3, 4), (4, 5, 6, 7), (0, 0, 0, 0)
    scores_ = [1.0, .5, .5, .25, .125, .12, .11, .1, .01, 0.0]
    assert stable_order(scores_) == list(range(10))
    for rank in (0, 1, 5, 6, 10):
        cases = [invalid] * 10
        if rank:
            cases[rank - 1] = target
        result, first = unified(cases, target, {target: 1})
        assert first == rank and result["hit@5"] == (0 < rank <= 5)
        assert result["mrr@10"] == (1 / rank if rank else 0)
        assert result["ItemHit@10"] == bool(rank)
    result, _ = unified([other, invalid, target] + [invalid] * 7, target, {target: 3, other: 3})
    assert result["ItemHit@5"] == 1 / 3 and result["ItemNDCG@5"] == 1 / (3 * math.log2(6))
    result, _ = unified([target] * 2 + [other] * 8, target, {target: 2, other: 2})
    assert result["ndcg@5"] == 1 + 1 / math.log2(3) and result["ItemHit@5"] == 1
    record = {"ids": [list(other)] * 10, "target_sid": list(target)}
    record["ids"][2] = record["target_sid"]
    record["ids"][8] = record["target_sid"]
    record["native_topk"] = {key: list(range(int(key.split("@")[1]))) for key in NATIVE}
    assert native(record, scores_)["recall@5"] == 0.5
    record["native_topk"]["recall@5"] = [0, 2, 1, 3, 4]
    assert native(record, scores_)["recall@5"] == 0.5
    coverage(list(range(20)), list(range(20)))
    for got in (list(range(16)), list(range(19)), list(range(19)) + [18]):
        try:
            coverage(got, list(range(20)))
        except ValueError:
            continue
        raise AssertionError("Incomplete or duplicated axis was accepted")
    return {"cases": ["miss", "ranks 1/5/6/10", "out-of-catalog", "stable tie and allowed native tie-order",
                      "target collision", "earlier collision rank shift", "duplicate target",
                      "four-row tail", "missing/duplicate users"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?")
    parser.add_argument("candidates", type=Path, nargs="?")
    parser.add_argument("--mode", choices=("preflight", "formal"), default="preflight")
    parser.add_argument("--native-metrics", type=Path)
    args = parser.parse_args()
    result = {"fixtures": fixtures()}
    if args.root is not None:
        result["replay"] = verify(args.root, args.candidates, args.mode,
                                  json.loads(args.native_metrics.read_bytes())[0] if args.native_metrics else None)
    print(json.dumps(result, indent=2))
