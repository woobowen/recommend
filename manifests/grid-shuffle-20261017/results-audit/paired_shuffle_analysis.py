"""Within-domain/route pairing of accepted complete original Shuffle predictions.

Mechanically carries forward the Beauty paired-analysis definitions. Domain,
route, paths and coverage come from an explicit plan and exact trusted target
index, rather than fixed Beauty constants. No model or GPU execution occurs.
"""
import argparse
from collections import Counter
import datetime as dt
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


def identity(path):
    path = Path(path)
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    assert not path.exists(), f"Preserve previous analysis: {path}"
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def prediction_rows(path, accepted, expected):
    assert identity(path)["sha256"] == accepted["prediction_sha256"]
    result, actual = {}, Counter()
    for line in Path(path).read_bytes().splitlines():
        row = json.loads(line)
        assert row["sample_key"] not in result
        assert len(row["sids"]) == len(row["scores"]) == 10
        assert all(math.isfinite(v) for v in row["scores"])
        assert all(a >= b for a, b in zip(row["scores"], row["scores"][1:]))
        assert len(set(map(tuple, row["sids"]))) == 10
        result[row["sample_key"]] = row
        actual[(row["sample_key"], tuple(row["target"]))] += 1
    assert actual == expected
    return result


def target_rank(row):
    found = [i + 1 for i, sid in enumerate(row["sids"]) if sid == row["target"]]
    assert len(found) <= 1
    return found[0] if found else None


def per_user_metrics(rank):
    return {f"{name}@{k}": float(rank is not None and rank <= k) * (1 / math.log2(rank + 1) if name == "ndcg" and rank is not None else 1)
            for name in ("recall", "ndcg") for k in (5, 10)}


def distribution(values):
    assert values and all(math.isfinite(v) for v in values)
    return {"n": len(values), "min": min(values), "median": statistics.median(values), "mean": math.fsum(values) / len(values), "max": max(values),
            "negative": sum(v < 0 for v in values), "zero": sum(v == 0 for v in values), "positive": sum(v > 0 for v in values)}


def analyze_route(plan, output):
    domain, route = plan["domain"], plan["route"]
    assert domain in ("toys", "sports") and route in ("A", "B")
    expected = Counter()
    for line in Path(plan["targets"]).read_bytes().splitlines():
        row = json.loads(line); expected[(row["sample_key"], tuple(row["target"]))] += 1
    count = sum(expected.values())
    assert count > 0 and len(expected) == len({key[0] for key in expected}) == count
    legal = set(map(tuple, read(plan["legal_sids"])["legal_raw_sids"]))
    inputs = [identity(plan["targets"]), identity(plan["legal_sids"])]
    rows, accepted = {}, {}
    assert set(plan["conditions"]) == {"0", "3", "4"}
    for n in (0, 3, 4):
        condition = plan["conditions"][str(n)]
        acceptance = read(condition["review"])
        assert acceptance["status"] == "FORMAL_EVALUATION_ACCEPTANCE_PASS"
        assert (acceptance["domain"], acceptance["route"], acceptance["shuffle_n"]) == (domain, route, n)
        assert acceptance["samples"] == count and acceptance["targets_sha256"] == identity(plan["targets"])["sha256"]
        assert acceptance["legal_vocabulary"]["sha256"] == identity(plan["legal_sids"])["sha256"]
        accepted[n] = acceptance
        rows[n] = prediction_rows(condition["predictions"], acceptance, expected)
        inputs.extend([identity(condition["review"]), identity(condition["predictions"])])
    assert set(rows[0]) == set(rows[3]) == set(rows[4])
    for name in ("config", "sid", "checkpoint", "test_index", "source_commit", "snapshot_sha256"):
        assert accepted[0][name] == accepted[3][name] == accepted[4][name], name
    comparisons = []
    for n in (3, 4):
        same_top1 = same_ordered = same_set = 0
        transitions = {5: Counter(), 10: Counter()}
        ranks = [[0] * 11 for _ in range(11)]
        rank_changes, common_hist, changed_positions, invalid_changes = (Counter() for _ in range(4))
        invalid_totals, retained_rank_delta = [0, 0], []
        total_common = total_abs_common_rank_shift = 0
        sum_jaccard = 0.0
        metric_deltas = {name: [] for name in ("recall@5", "ndcg@5", "recall@10", "ndcg@10")}
        most_changed, lost_first, gained_first = [], [], []
        pair_path = output / f"{domain}-{route.lower()}-normal-to-shuffle{n}-paired-users.jsonl.gz"
        assert not pair_path.exists()
        with pair_path.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as paired:
            for key in sorted(rows[0]):
                normal, changed = rows[0][key], rows[n][key]
                assert normal["target"] == changed["target"]
                normal_ids, changed_ids = list(map(tuple, normal["sids"])), list(map(tuple, changed["sids"]))
                normal_rank, changed_rank = target_rank(normal), target_rank(changed)
                ranks[normal_rank or 0][changed_rank or 0] += 1
                if normal_rank is None and changed_rank is None:
                    rank_changes["absent_both_top10"] += 1
                elif normal_rank is None:
                    rank_changes["entered_top10"] += 1
                elif changed_rank is None:
                    rank_changes["left_top10"] += 1
                else:
                    rank_changes["same_rank" if changed_rank == normal_rank else "improved_rank" if changed_rank < normal_rank else "worsened_rank"] += 1
                    retained_rank_delta.append(changed_rank - normal_rank)
                hits = {}
                for k in (5, 10):
                    before, after = normal_rank is not None and normal_rank <= k, changed_rank is not None and changed_rank <= k
                    transitions[k]["both_hit" if before and after else "lost_hit" if before else "gained_hit" if after else "both_miss"] += 1
                    hits[f"normal_hit{k}"], hits[f"shuffle_hit{k}"] = before, after
                top1_same, ordered_same, set_same = normal_ids[0] == changed_ids[0], normal_ids == changed_ids, set(normal_ids) == set(changed_ids)
                common = set(normal_ids) & set(changed_ids)
                positions_changed = sum(x != y for x, y in zip(normal_ids, changed_ids))
                same_top1 += top1_same; same_ordered += ordered_same; same_set += set_same
                common_hist[len(common)] += 1; changed_positions[positions_changed] += 1
                total_common += len(common)
                total_abs_common_rank_shift += sum(abs(normal_ids.index(sid) - changed_ids.index(sid)) for sid in common)
                sum_jaccard += len(common) / (20 - len(common))
                invalid_normal, invalid_changed = sum(sid not in legal for sid in normal_ids), sum(sid not in legal for sid in changed_ids)
                invalid_totals[0] += invalid_normal; invalid_totals[1] += invalid_changed
                invalid_changes[invalid_changed - invalid_normal] += 1
                before_metrics, after_metrics = per_user_metrics(normal_rank), per_user_metrics(changed_rank)
                differences = {name: after_metrics[name] - before_metrics[name] for name in metric_deltas}
                for name in metric_deltas:
                    metric_deltas[name].append(differences[name])
                row = {"sample_key": key, "target_unchanged": True, "normal_target_rank": normal_rank, "shuffle_target_rank": changed_rank,
                       **hits, "top1_same": top1_same, "ordered_top10_same": ordered_same, "top10_set_same": set_same,
                       "common_candidates_at10": len(common), "changed_positions_at10": positions_changed,
                       "normal_invalid_count": invalid_normal, "shuffle_invalid_count": invalid_changed,
                       "normal_duplicate_count": 0, "shuffle_duplicate_count": 0,
                       "independent_metric_deltas": differences}
                paired.write((json.dumps(row, separators=(",", ":")) + "\n").encode())
                case = {**row, "target": normal["target"], "normal_sids": normal["sids"], "shuffle_sids": changed["sids"]}
                most_changed.append(case)
                most_changed.sort(key=lambda r: (-r["changed_positions_at10"], r["common_candidates_at10"], r["sample_key"]))
                del most_changed[3:]
                if hits["normal_hit10"] and not hits["shuffle_hit10"] and len(lost_first) < 3:
                    lost_first.append(case)
                if hits["shuffle_hit10"] and not hits["normal_hit10"] and len(gained_first) < 3:
                    gained_first.append(case)
        assert invalid_totals == [accepted[0]["invalid_candidate_count"], accepted[n]["invalid_candidate_count"]]
        differences = {}
        for metric, baseline in accepted[0]["metrics"].items():
            value = accepted[n]["metrics"][metric]; difference = value - baseline
            differences[metric] = {"Normal": baseline, "Shuffle": value, "absolute_difference": difference,
                                   "absolute_difference_percentage_points": 100 * difference,
                                   "relative_difference": difference / baseline if baseline else None,
                                   "relative_difference_percent": 100 * difference / baseline if baseline else None}
            assert abs(math.fsum(metric_deltas[metric]) / count - difference) <= 1e-7
        for k in (5, 10):
            transition = transitions[k]
            assert sum(transition.values()) == count
            assert abs((transition["gained_hit"] - transition["lost_hit"]) / count - differences[f"recall@{k}"]["absolute_difference"]) <= 1e-7
        comparisons.append({"domain": domain, "route": route, "Normal_to_Shuffle_n": n, "paired_users": count,
            "target_multiset_preserved": True, "metric_differences_from_accepted_original_float32_metrics": differences,
            "paired_hits": {str(k): {name: v[name] for name in ("both_hit", "both_miss", "lost_hit", "gained_hit")} for k, v in transitions.items()},
            "target_rank_transition_axes": "row Normal rank, column Shuffle rank; index0 means absent from Top10, indexes1..10 are actual original model-returned ranks",
            "target_rank_transition_counts": ranks, "target_rank_change_counts": dict(rank_changes),
            "mean_rank_delta_when_target_present_in_both": math.fsum(retained_rank_delta) / len(retained_rank_delta) if retained_rank_delta else None,
            "rank_delta_sign": "Shuffle rank minus Normal rank; negative is improved; excludes target absent in either list",
            "recommendation_list_changes": {"top1_same_users": same_top1, "top1_changed_users": count - same_top1, "top1_changed_fraction": (count - same_top1) / count,
                "ordered_top10_same_users": same_ordered, "ordered_top10_changed_users": count - same_ordered,
                "same_top10_set_users": same_set, "top10_set_changed_users": count - same_set, "same_set_but_changed_order_users": same_set - same_ordered,
                "intersection_size_at10_histogram": dict(sorted(common_hist.items())), "mean_common_candidates_at10": total_common / count,
                "mean_jaccard_at10": sum_jaccard / count, "changed_original_candidate_positions_histogram": dict(sorted(changed_positions.items())),
                "mean_abs_common_candidate_rank_shift": total_abs_common_rank_shift / total_common if total_common else None},
            "invalid_and_duplicate_behavior": {"invalid_candidates_Normal": invalid_totals[0], "invalid_candidates_Shuffle": invalid_totals[1],
                "per_user_invalid_count_delta_histogram": dict(sorted(invalid_changes.items())), "duplicate_candidates_Normal": 0, "duplicate_candidates_Shuffle": 0,
                "candidate_filtering_deduplication_or_rescoring_applied": False},
            "independent_per_user_metric_difference_distribution": {name: distribution(v) for name, v in metric_deltas.items()},
            "prespecified_examples": {"greatest_original_list_change_first_three": most_changed, "sample_key_first_three_lost_hit10": lost_first, "sample_key_first_three_gained_hit10": gained_first},
            "all_user_paired_observations": identity(pair_path)})
    return comparisons, inputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output-directory", type=Path, required=True)
    a = p.parse_args()
    assert a.output_directory.is_dir()
    plan = read(a.plan); comparisons, inputs = [], [identity(a.plan)]
    assert len({(v["domain"], v["route"]) for v in plan["routes"]}) == len(plan["routes"])
    for route in plan["routes"]:
        current, sources = analyze_route(route, a.output_directory); comparisons.extend(current); inputs.extend(sources)
    result = {"schema": "grid-domain-within-route-paired-shuffle-analysis-v1", "status": "DESCRIPTIVE_PAIRED_ANALYSIS_COMPLETE",
        "author": "/root/results_audit", "computed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "comparisons": comparisons,
        "method": {"global_four_metrics": "Inherited accepted original float32 evaluator values; no redundant evaluator or model execution",
            "absolute_difference": "Shuffle minus route-specific Normal, native metric units; percentage points multiply by100",
            "relative_difference": "(Shuffle-Normal)/Normal; null when Normal=0", "user_pairing": "Exact trusted sample_key+target Counter, exact same set and four-code targets within each domain/route",
            "ranking": "Original saved model-returned order; scores checked nonincreasing and duplicate candidates absent; no candidate rescoring/reordering",
            "list_comparison": "Top1 equality, ordered10 equality, set equality, intersection/Jaccard, positional and common-candidate rank changes; sets are descriptive only",
            "target_absent": "Rank null/transition index0 when target is absent from original Top10",
            "prespecified_examples_rule": "Before new results: largest number of changed original list positions, then smallest Top10 intersection, then sample_key ascending; retain first three. Separately first three lost/gained@10 by sample_key. No support-for-hypothesis selection.",
            "scope": "Within-domain/route Normal→Shuffle only; no A/B same-model, multi-seed, significance or strict permutation-invariance claim"},
        "inputs": inputs, "program": identity(Path(__file__)), "argv": sys.argv,
        "new_model_GPU_or_evaluator_calls": 0, "original_predictions_or_accepted_reviews_modified": False,
        "independent_review_of_this_new_analysis": "Required from a different actual agent; this author does not self-sign nonauthor acceptance."}
    path = a.output_directory / "toys-sports-paired-shuffle-analysis.json"
    write(path, result)
    print(json.dumps({"report": identity(path), "comparisons": len(comparisons), "routes": len(plan["routes"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
