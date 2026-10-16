"""Descriptive within-route pairing of existing, already accepted predictions.

No model, torch, evaluator rerun, selection, rescore, filtering, or new seed.
Original model-returned candidate order is retained; input files must match
the SHA256 recorded by the six existing independent acceptance receipts.
"""
import argparse
from collections import Counter
import datetime as dt
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path):
    p = Path(path)
    return {"path": str(p), "bytes": p.stat().st_size, "sha256": sha(p)}


def load(path):
    return json.loads(Path(path).read_text())


def prediction_rows(path, expected_sha):
    assert sha(path) == expected_sha
    result = {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        assert row["sample_key"] not in result
        assert len(row["sids"]) == len(row["scores"]) == 10
        assert all(math.isfinite(v) for v in row["scores"])
        assert all(a >= b for a, b in zip(row["scores"], row["scores"][1:]))
        assert len(set(map(tuple, row["sids"]))) == 10
        result[row["sample_key"]] = row
    assert len(result) == 22363
    return result


def target_rank(row):
    found = [i + 1 for i, sid in enumerate(row["sids"]) if sid == row["target"]]
    assert len(found) <= 1
    return found[0] if found else None


def write_json(path, value):
    assert not path.exists(), f"Preserve prior analysis: {path}"
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--support", type=Path, required=True)
    p.add_argument("--jobs", type=Path, required=True)
    a = p.parse_args()
    output = a.support / "results-review-r4"
    summary_path = output / "beauty-six-formal-results-independent-summary.json"
    inherited = load(summary_path)
    assert inherited["status"] == "BEAUTY_SIX_FORMAL_EVALUATIONS_INDEPENDENTLY_ACCEPTED"
    accepted = {(v["route"], v["shuffle_n"]): v for v in inherited["conditions"]}
    comparisons = []
    inputs = [record(summary_path)]
    expected_legal_sha = {
        "A": "c84e1ac4107c77675a9c646f5294b852beccb7a86ccd31156dcb79b3cf4d2f58",
        "B": "b9a42b21d038c7e8dfe0e8f69a68964e6ee05cbf8d4fdf5c55c76e60e55fd4a5",
    }
    for route in ("A", "B"):
        legal_path = a.support / f"beauty-{route.lower()}-legal-sids.json"
        assert sha(legal_path) == expected_legal_sha[route]
        legal = set(map(tuple, load(legal_path)["legal_raw_sids"]))
        inputs.append(record(legal_path))
        rows = {}
        for n in (0, 3, 4):
            acceptance = accepted[(route, n)]
            path = a.jobs / acceptance["job_id"] / "retrieved/output/test/predictions.jsonl"
            rows[n] = prediction_rows(path, acceptance["prediction_sha256"])
            inputs.append(record(path))
        assert set(rows[0]) == set(rows[3]) == set(rows[4])
        for n in (3, 4):
            count = len(rows[0])
            same_top1 = same_ordered = same_set = 0
            transitions = {5: Counter(), 10: Counter()}
            ranks = [[0] * 11 for _ in range(11)]
            rank_changes = Counter()
            common_hist = Counter()
            changed_positions = Counter()
            invalid_changes = Counter()
            invalid_totals = [0, 0]
            retained_rank_delta = []
            total_common = total_abs_common_rank_shift = 0
            sum_jaccard = 0.0
            pair_path = output / f"beauty-{route.lower()}-normal-to-shuffle{n}-paired-users.jsonl.gz"
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
                        hits[f"normal_hit{k}"] = before
                        hits[f"shuffle_hit{k}"] = after
                    top1_same = normal_ids[0] == changed_ids[0]
                    ordered_same = normal_ids == changed_ids
                    set_same = set(normal_ids) == set(changed_ids)
                    common = set(normal_ids) & set(changed_ids)
                    positions_changed = sum(x != y for x, y in zip(normal_ids, changed_ids))
                    same_top1 += top1_same
                    same_ordered += ordered_same
                    same_set += set_same
                    common_hist[len(common)] += 1
                    changed_positions[positions_changed] += 1
                    total_common += len(common)
                    total_abs_common_rank_shift += sum(abs(normal_ids.index(sid) - changed_ids.index(sid)) for sid in common)
                    sum_jaccard += len(common) / (20 - len(common))
                    invalid_normal = sum(sid not in legal for sid in normal_ids)
                    invalid_changed = sum(sid not in legal for sid in changed_ids)
                    invalid_totals[0] += invalid_normal
                    invalid_totals[1] += invalid_changed
                    invalid_changes[invalid_changed - invalid_normal] += 1
                    row = {"sample_key": key, "target_unchanged": True,
                           "normal_target_rank": normal_rank, "shuffle_target_rank": changed_rank,
                           **hits, "top1_same": top1_same, "ordered_top10_same": ordered_same,
                           "top10_set_same": set_same, "common_candidates_at10": len(common),
                           "changed_positions_at10": positions_changed,
                           "normal_invalid_count": invalid_normal, "shuffle_invalid_count": invalid_changed,
                           "normal_duplicate_count": 0, "shuffle_duplicate_count": 0}
                    paired.write((json.dumps(row, separators=(",", ":")) + "\n").encode())
            assert invalid_totals == [accepted[(route, 0)]["invalid_candidate_count"], accepted[(route, n)]["invalid_candidate_count"]]
            differences = {}
            for metric, baseline in accepted[(route, 0)]["metrics"].items():
                value = accepted[(route, n)]["metrics"][metric]
                difference = value - baseline
                differences[metric] = {"Normal": baseline, "Shuffle": value,
                                       "absolute_difference": difference,
                                       "absolute_difference_percentage_points": 100 * difference,
                                       "relative_difference": difference / baseline if baseline else None,
                                       "relative_difference_percent": 100 * difference / baseline if baseline else None}
            for k in (5, 10):
                transition = transitions[k]
                assert sum(transition.values()) == count
                net = (transition["gained_hit"] - transition["lost_hit"]) / count
                assert abs(net - differences[f"recall@{k}"]["absolute_difference"]) <= 1e-7
            comparisons.append({
                "route": route, "Normal_to_Shuffle_n": n, "paired_users": count,
                "target_multiset_preserved": True, "metric_differences_from_accepted_original_float32_metrics": differences,
                "paired_hits": {str(k): dict(v) for k, v in transitions.items()},
                "target_rank_transition_axes": "row Normal rank, column Shuffle rank; index0 means absent from Top10, indexes1..10 are actual original model-returned ranks",
                "target_rank_transition_counts": ranks, "target_rank_change_counts": dict(rank_changes),
                "mean_rank_delta_when_target_present_in_both": sum(retained_rank_delta) / len(retained_rank_delta) if retained_rank_delta else None,
                "rank_delta_sign": "Shuffle rank minus Normal rank; negative is improved; excludes target absent in either list",
                "recommendation_list_changes": {
                    "top1_same_users": same_top1, "top1_changed_users": count - same_top1,
                    "top1_changed_fraction": (count - same_top1) / count,
                    "ordered_top10_same_users": same_ordered, "ordered_top10_changed_users": count - same_ordered,
                    "same_top10_set_users": same_set, "same_set_but_changed_order_users": same_set - same_ordered,
                    "intersection_size_at10_histogram": dict(sorted(common_hist.items())),
                    "mean_common_candidates_at10": total_common / count,
                    "mean_jaccard_at10": sum_jaccard / count,
                    "changed_original_candidate_positions_histogram": dict(sorted(changed_positions.items())),
                    "mean_abs_common_candidate_rank_shift": total_abs_common_rank_shift / total_common if total_common else None},
                "invalid_and_duplicate_behavior": {
                    "invalid_candidates_Normal": invalid_totals[0], "invalid_candidates_Shuffle": invalid_totals[1],
                    "per_user_invalid_count_delta_histogram": dict(sorted(invalid_changes.items())),
                    "duplicate_candidates_Normal": 0, "duplicate_candidates_Shuffle": 0,
                    "candidate_filtering_deduplication_or_rescoring_applied": False},
                "all_user_paired_observations": record(pair_path)})
    result = {"schema": "grid-beauty-within-route-paired-shuffle-analysis-v1",
              "status": "DESCRIPTIVE_PAIRED_ANALYSIS_COMPLETE", "reviewer": "/root/results_publication_review",
              "computed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "comparisons": comparisons, "method": {
                  "global_four_metrics": "Inherited accepted original evaluator values; no redundant model or evaluator execution",
                  "absolute_difference": "Shuffle minus route-specific Normal, native metric units; percentage points multiply by100",
                  "relative_difference": "(Shuffle-Normal)/Normal; null when Normal=0",
                  "user_pairing": "Exact same opaque sample_key set and exact target four-code tuple within each route; sorted keys only determine output observation order",
                  "ranking": "Original saved model-returned order; scores checked nonincreasing; no rescore or candidate reorder. No duplicate candidates occur in any input.",
                  "list_comparison": "Top1 exact SID equality, ordered10 equality, set equality, intersection/Jaccard, positional and common-candidate rank changes; sets used only for description, never to change evaluator inputs.",
                  "target_absent": "Rank null/transition index0 when target is absent from original Top10",
                  "scope": "Four within-route Normal→Shuffle comparisons; no A/B same-model equivalence claim, no multi-seed or statistical significance claim, no similarity threshold",
                  "interpretation": "Small aggregate metric differences do not imply every user's list is unchanged or prove strict mathematical permutation invariance."},
              "inputs": inputs, "script": record(Path(__file__).resolve()), "argv": sys.argv,
              "research_execution": {"models_loaded": 0, "GPU_calls": 0, "new_seeds_or_test_runs": 0,
                                     "original_predictions_or_accepted_reviews_modified": False}}
    report_path = output / "beauty-paired-shuffle-analysis.json"
    write_json(report_path, result)
    invocation = {"argv": sys.argv, "interpreter": sys.executable, "script": record(Path(__file__).resolve()),
                  "inputs": inputs, "report": record(report_path),
                  "paired_outputs": [v["all_user_paired_observations"] for v in comparisons],
                  "exit_status": "COMPLETED", "model_or_GPU_execution": False}
    write_json(output / "beauty-paired-shuffle-analysis-invocation.json", invocation)
    print(json.dumps({"report": record(report_path), "comparisons": [
        {"route": v["route"], "shuffle_n": v["Normal_to_Shuffle_n"],
         "top1_changed_users": v["recommendation_list_changes"]["top1_changed_users"],
         "ordered_top10_changed_users": v["recommendation_list_changes"]["ordered_top10_changed_users"],
         "paired_hits": v["paired_hits"]} for v in comparisons]}, indent=2))


if __name__ == "__main__":
    main()
