#!/usr/bin/env python3
"""Independent stdlib-only audit of completed LETTER Instruments predictions."""
import argparse
import collections
import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

N, B = 24772, 20
M = ("hit@5", "hit@10", "ndcg@5", "ndcg@10")
HASHES = {
    "Instruments.index.json":
        "33c87e28c52ae367985bdf494cb49d676d181e13a39497b4ad035f46111546dd",
    "Instruments.inter.json":
        "0a11d47791d90483005da93c36d826679e20e4592d1d2b86518d5f56f218e743",
    "Instruments.item.json":
        "1d9938e576a8d964e6b977a2b63aa56402840a788eaa6685f26f0d910a7133f2",
}
AKEYS = (
    "candidate_count legal_complete_sid_candidates invalid_complete_sid_candidates "
    "duplicate_candidate_occurrences colliding_sid_candidate_occurrences "
    "users_with_invalid_sid users_with_duplicate_sid "
    "users_with_colliding_sid_candidate targets_with_colliding_sid"
).split()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def unique(pairs):
    result = {}
    for key, value in pairs:
        check(key not in result, "Duplicate JSON key: " + str(key))
        result[key] = value
    return result


def read(path, expected=None):
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    check(expected is None or digest == expected, "SHA mismatch: " + str(path))
    return json.loads(raw, object_pairs_hook=unique), {
        "path": str(path), "bytes": len(raw), "sha256": digest
    }


def number(value):
    check(type(value) in (int, float), "Non-numeric value")
    value = float(value)
    check(math.isfinite(value), "Non-finite value")
    return value


def run_id(obj):
    values = [obj[k] for k in ("run_commit", "RUN_COMMIT") if k in obj]
    check(values and all(v == values[0] for v in values), "Missing/conflicting RUN")
    value = values[0]
    check(isinstance(value, str) and len(value) == 40
          and all(c in "0123456789abcdef" for c in value), "Invalid full RUN SHA")
    return value


def normalize(text):
    check(isinstance(text, str), "Non-string raw_decoded")
    return text.split("Response:")[-1].strip().replace(" ", "")


def calculate(relevance):
    result = {}
    for k in (5, 10):
        result["hit@" + str(k)] = float(any(relevance[:k]))
        result["ndcg@" + str(k)] = math.fsum(
            hit / math.log2(i + 2) for i, hit in enumerate(relevance[:k])
        )
    return result


def selftest():
    check(all(v == 1 for v in calculate([1] + [0] * 19).values()), "First hit")
    sixth = calculate([0] * 5 + [1] + [0] * 14)
    check(sixth["hit@5"] == 0 and sixth["hit@10"] == 1, "Sixth hit")
    check(abs(sixth["ndcg@10"] - 1 / math.log2(7)) < 1e-15, "Discount")
    check(calculate([1, 1] + [0] * 18)["ndcg@5"] > 1, "Duplicate target")
    check(sorted(range(3), key=[0, 0, 1].__getitem__, reverse=True) == [2, 0, 1],
          "Stable sorting")
    check(normalize("prefix Response: <a_1> <b_2>\n") == "<a_1><b_2>", "Parsing")


def audit(data_root, test_dir):
    loaded, assets = {}, {}
    for name, digest in HASHES.items():
        loaded[name], assets[name] = read(data_root / "Instruments" / name, digest)
    index, inter, items = [loaded["Instruments." + s + ".json"]
                           for s in ("index", "inter", "item")]
    check(isinstance(index, dict) and len(index) == 9922, "SID item count")
    check(isinstance(items, dict) and set(items) == set(index), "Item/SID keys")
    check(isinstance(inter, dict) and len(inter) == N, "Interaction user count")
    catalog, item_sid = collections.defaultdict(list), {}
    for item, tokens in index.items():
        check(isinstance(tokens, list) and len(tokens) == 4
              and all(isinstance(t, str) for t in tokens), "Four-position SID")
        item_sid[item] = "".join(tokens)
        catalog[item_sid[item]].append(item)
    collisions = {sid for sid, ids in catalog.items() if len(ids) > 1}
    check(len(catalog) == 9897 and len(collisions) == 23
          and sum(len(catalog[s]) for s in collisions) == 48, "Collision identity")
    users, target_items, targets = list(inter), [], []
    for user in users:
        history = inter[user]
        check(isinstance(history, list) and history
              and type(history[-1]) is int, "Invalid interaction target")
        item = str(history[-1])
        check(item in item_sid, "Unknown target item")
        target_items.append(item)
        targets.append(item_sid[item])
    original, original_file = read(test_dir / "metrics.json")
    check(original.get("status") == "PASS" and original.get("test_complete") is True,
          "Original formal test is not PASS")
    check(all(original.get(k) == N for k in ("actual_test_users", "valid_result_users"))
          and all(original.get(k) == 0 for k in ("failed_users", "skipped_users"))
          and original.get("all_row_indices_exactly_once") is True, "Completion gate")
    identity, states, state_files = run_id(original), [], []
    for rank in (0, 1):
        state, receipt = read(test_dir / ("rank%d.json" % rank))
        check(state.get("status") == "PASS"
              and state.get("completed_users") == N // 2
              and state.get("assigned_users") == N // 2, "Rank formal completion")
        check(run_id(state) == identity, "Rank RUN mismatch")
        states.append(state)
        state_files.append(receipt)
    for key in ("adapter_files", "data_identity", "author_sources"):
        check(key in states[0] and key in states[1]
              and states[0][key] == states[1][key], "Rank identity: " + key)
    if "adapter_files" in original:
        check(original["adapter_files"] == states[0]["adapter_files"], "Metric adapter")
    if any("generation" in state for state in states):
        check(all("generation" in state for state in states)
              and states[0]["generation"] == states[1]["generation"], "Rank generation")
    values, anomalies = {m: [] for m in M}, dict.fromkeys(AKEYS, 0)
    seen, prediction_files = set(), []
    for rank in (0, 1):
        path = test_dir / ("predictions.rank%d.jsonl" % rank)
        before, digest, count, byte_count = path.stat(), hashlib.sha256(), 0, 0
        with path.open("rb") as stream:
            for raw in stream:
                digest.update(raw)
                byte_count += len(raw)
                row = json.loads(raw, object_pairs_hook=unique)
                i = row.get("row_idx")
                check(type(i) is int and 0 <= i < N and i not in seen, "Row coverage")
                check(type(row.get("rank")) is int and row["rank"] == rank,
                      "Prediction rank")
                check(type(row.get("prompt_id")) is int and row["prompt_id"] == 0,
                      "Prompt0 required")
                check(row.get("user_id") == users[i]
                      and row.get("target_item_id") == target_items[i]
                      and row.get("target") == targets[i], "User/target pairing")
                check(row.get("original_callback_verified_all_tokens") is True,
                      "Recorded callback verification")
                candidates = row.get("candidates")
                check(isinstance(candidates, list) and len(candidates) == B, "Beam20")
                sids, scores = [], []
                for order, candidate in enumerate(candidates):
                    check(type(candidate.get("generation_order")) is int
                          and candidate["generation_order"] == order, "Beam order")
                    sid = normalize(candidate.get("raw_decoded"))
                    check(candidate.get("sid") == sid, "Independent SID parsing")
                    sids.append(sid)
                    scores.append(number(candidate.get("sequence_score")))
                order = sorted(range(B), key=scores.__getitem__, reverse=True)
                relevance = [int(sids[j] == targets[i]) for j in order]
                check(row.get("sorted_generation_order") == order, "Stored sort")
                check(row.get("relevance") == relevance, "Stored relevance")
                result = calculate(relevance)
                for metric in M:
                    values[metric].append(result[metric])
                legal = sum(sid in catalog for sid in sids)
                duplicate, colliding = B - len(set(sids)), sum(s in collisions for s in sids)
                increments = (B, legal, B - legal, duplicate, colliding,
                              int(legal < B), int(duplicate > 0), int(colliding > 0),
                              int(targets[i] in collisions))
                for key, increment in zip(AKEYS, increments):
                    anomalies[key] += increment
                seen.add(i)
                count += 1
        after = path.stat()
        check((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
              and byte_count == after.st_size, "Prediction file changed during audit")
        check(count == N // 2, "Rank prediction count")
        prediction_files.append({"rank": rank, "path": str(path), "users": count,
                                 "bytes": byte_count, "sha256": digest.hexdigest()})
    check(seen == set(range(N)), "Incomplete or duplicate user coverage")
    check(anomalies["candidate_count"] == N * B, "Total beam count")
    anomalies["complete_sid_legal_rate"] = anomalies["legal_complete_sid_candidates"] / (N * B)
    claims = original.get("prediction_files")
    check(isinstance(claims, list) and len(claims) == 2, "Prediction manifest")
    check(all(type(c.get("rank")) is int for c in claims), "Manifest rank type")
    claims = {c["rank"]: c for c in claims}
    check(set(claims) == {0, 1}, "Manifest ranks")
    for actual in prediction_files:
        claim = claims[actual["rank"]]
        check(all(claim.get(k) == actual[k] for k in ("users", "bytes", "sha256")),
              "Prediction manifest values")
        check(isinstance(claim.get("path"), str)
              and Path(claim["path"]).name == Path(actual["path"]).name, "Manifest path")
    sums = {m: math.fsum(values[m]) for m in M}
    means, deltas = {m: sums[m] / N for m in M}, {}
    for metric in M:
        ds = abs(number(original["metric_sums"][metric]) - sums[metric])
        dm = abs(number(original["metrics"][metric]) - means[metric])
        check(ds <= 1e-9 and dm <= 1e-13, "Independent metric mismatch: " + metric)
        deltas[metric] = {"sum_absolute_delta": ds, "mean_absolute_delta": dm}
    for key, value in anomalies.items():
        claimed = original["anomalies"].get(key)
        if key == "complete_sid_legal_rate":
            check(abs(number(claimed) - value) <= 1e-15, "Legal SID rate")
        else:
            check(type(claimed) is int and claimed == value, "Anomaly count: " + key)
    return {"status": "PASS", "run_commit": identity, "actual_test_users": len(seen),
            "beam_candidates_per_user": B, "skipped_users": 0,
            "metric_sums": sums, "metrics": means, "metric_deltas": deltas,
            "anomalies": anomalies, "assets": assets, "rank_metadata": state_files,
            "prediction_files": prediction_files, "original_metrics": original_file,
            "policy": "Stable score sort; complete SID equality; no filtering, "
                      "deduplication or collision changes; duplicate target NDCG additive",
            "tolerances": {"metric_mean": 1e-13, "metric_sum": 1e-9},
            "limits": ["Recorded predictions only; no GPU/model/training rerun",
                       "No independent token decoding or allowed-token callback execution",
                       "Adapter manifests compared; model tensors not reloaded",
                       "Controller bridge execution; reviewer did not directly SSH"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("data-root", "test-dir", "output"):
        parser.add_argument("--" + flag, type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    selftest()
    if args.self_test:
        print('{"status":"PASS","self_test":true}')
        return 0
    if not all((args.data_root, args.test_dir, args.output)):
        parser.error("--data-root, --test-dir and --output are required")
    if args.output.exists():
        parser.error("--output exists; preserve it and choose another path")
    try:
        report, code = audit(args.data_root, args.test_dir), 0
    except Exception as error:
        report, code = {"status": "FAIL", "error": repr(error)}, 1
    report.update(audit="INDEPENDENT_STDLIB_RECOMPUTE", exit_code=code, argv=sys.argv,
                  created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(text)
    print(text, end="")
    return code


if __name__ == "__main__":
    sys.exit(main())
