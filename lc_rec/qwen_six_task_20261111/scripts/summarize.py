"""Summarize actual terminal evidence without promoting compute to semantic PASS."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


def read(path):
    return json.loads(path.read_text())


def source(path):
    content = path.read_bytes()
    return {"path": str(path), "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    jobs = args.evidence / "jobs"
    results = {"scope": "Actual LC-Rec930 six-task bounded compute; full natural-language supervision remains PARTIAL_BOUNDARY",
               "overall_status": "PARTIAL", "semantic_status": "PARTIAL_BOUNDARY", "routes": {}, "all_jobs": [], "argv": sys.argv}
    for directory in sorted(jobs.iterdir()):
        if not directory.is_dir() or not (directory / "status.json").exists():
            continue
        terminal = read(directory / "controller-query.json") if (directory / "controller-query.json").exists() else read(directory / "status.json")
        results["all_jobs"].append({"job_id": directory.name, "status": terminal.get("status"), "exit_code": terminal.get("exit_code"),
                                    "cleanup_verified": terminal.get("cleanup_verified"), "source": source(directory / "status.json")})
    for key, route in (("q06", "L"), ("q06", "F"), ("q17", "L"), ("q17", "F")):
        pattern = re.compile("lc-qwen6-20261111-" + key + "-" + route.lower() + "-r([0-3])$")
        attempts = sorted((directory for directory in jobs.iterdir() if directory.is_dir() and pattern.fullmatch(directory.name)), key=lambda directory: int(pattern.fullmatch(directory.name).group(1)))
        candidate = {"route_status": "BLOCKED", "compute_status": "BLOCKED", "semantic_status": "PARTIAL_BOUNDARY", "attempts": []}
        for directory in attempts:
            ranks = [read(directory / "output/probe" / ("rank" + str(rank) + ".json")) for rank in (0, 1) if (directory / "output/probe" / ("rank" + str(rank) + ".json")).exists()]
            updates = [update for rank in ranks for update in rank.get("updates", []) if update.get("status") == "PASS"]
            both_rank = len(ranks) == 2
            nonzero = [sum(update.get("nonzero_lr", False) for update in rank.get("updates", [])) for rank in ranks]
            core = both_rank and all(rank.get("core_training_status") == "PASS" for rank in ranks)
            long_tail = both_rank and all(rank.get("long_tail_zero_training", {}).get("status") == "PASS" for rank in ranks)
            saved = both_rank and all(rank.get("save_reload", {}).get("status") == "PASS" for rank in ranks)
            generated = both_rank and all(rank.get("generation", {}).get("status") == "PASS" for rank in ranks)
            diagnostic = both_rank and all(len(rank.get("task_diagnostics", [])) == 24 and all(item.get("status") == "PASS" for item in rank.get("task_diagnostics", [])) for rank in ranks)
            compute = "PASS" if core and long_tail and saved and generated and diagnostic else "PARTIAL" if updates or any(rank.get("model_identity") for rank in ranks) else "BLOCKED"
            ordinary = [update for rank in ranks for update in rank.get("updates", []) if update.get("status") == "PASS" and update.get("nonzero_lr") and update.get("kind") == "mixed"]
            ordinary_by_index = {}
            for update in ordinary:
                ordinary_by_index.setdefault(update["index"], []).append(update)
            timed = [{"index": index, "wall_seconds_max_rank": max(update["seconds"] for update in group), "samples": 128,
                      "samples_per_second": 128 / max(update["seconds"] for update in group), "global_counts": group[0].get("global_counts")}
                     for index, group in sorted(ordinary_by_index.items()) if len(group) == 2]
            state = [rank.get("training_state_restore", {"status": "NOT_TESTED"}) for rank in ranks]
            diagnostics = []
            if both_rank:
                for diagnostic_index, record in enumerate(ranks[0].get("task_diagnostics", [])):
                    pair = [rank["task_diagnostics"][diagnostic_index] for rank in ranks]
                    assert all((item["task"], item["quantile"], item["sample_id"]) == (record["task"], record["quantile"], record["sample_id"]) for item in pair)
                    measured_seconds = max(sum(item[name] for name in ("cpu_collator_seconds", "transfer_seconds", "forward_seconds", "backward_seconds")) for item in pair)
                    diagnostics.append({"task": record["task"], "quantile": record["quantile"], "sample_id": record["sample_id"],
                                        "distinct_records": 1, "compute_presentations": 2, "communication": "NONE", "optimizer_updates": 0,
                                        "counts_per_rank": record["counts_per_rank"], "measured_seconds_max_rank": measured_seconds,
                                        "forward_backward_seconds_max_rank": max(item["forward_seconds"] + item["backward_seconds"] for item in pair),
                                        "compute_presentations_per_second": 2 / measured_seconds,
                                        "effective_input_tokens_per_second": 2 * record["counts_per_rank"]["input_tokens"] / measured_seconds,
                                        "phase_seconds_max_rank": {name: max(item[name] for item in pair) for name in ("cpu_collator_seconds", "transfer_seconds", "forward_seconds", "backward_seconds")},
                                        "phase_maxima_are_not_additive_wall": True})
            resources_path = directory / "output/resources/resource-summary.json"
            resources = read(resources_path) if resources_path.exists() else {}
            attempt = {"job_id": directory.name, "run_commits": sorted({rank.get("run_commit", "UNKNOWN") for rank in ranks}),
                       "compute_status": compute, "two_rank_reports": both_rank,
                       "load_observed": [bool(rank.get("model_identity")) for rank in ranks],
                       "successful_global_updates_per_rank": [len(rank.get("updates", [])) for rank in ranks], "nonzero_lr_updates_per_rank": nonzero,
                       "core_training_status": [rank.get("core_training_status", "NOT_REACHED") for rank in ranks],
                       "long_tail_full_zero_status": "PASS" if long_tail else "NOT_PASSED",
                       "model_or_adapter_save_reload": "PASS" if saved else "NOT_PASSED",
                       "generation17_beam20_status": "PASS" if generated else "NOT_PASSED",
                       "task24_no_communication_diagnostics_status": "PASS" if diagnostic else "NOT_PASSED",
                       "training_update_peak_allocated_gib": max((rank.get("training_peak", {}).get("peak_allocated_gib", 0) for rank in ranks), default=0),
                       "training_update_peak_reserved_gib": max((rank.get("training_peak", {}).get("peak_reserved_gib", 0) for rank in ranks), default=0),
                       "ordinary_mixed_timing": timed, "training_state_restore": state,
                       "ordinary_joint_samples_per_second": sum(item["samples"] for item in timed) / sum(item["wall_seconds_max_rank"] for item in timed) if timed else None,
                       "task_diagnostics_no_communication": diagnostics,
                       "resource_guard": resources,
                       "generation_summary": [{"case_count": len(rank.get("generation", {}).get("cases", [])),
                                               "returned_sequences": sum(case["returned_sequences"] for case in rank.get("generation", {}).get("cases", [])),
                                               "canonical_sequences": sum(case["canonical_count"] for case in rank.get("generation", {}).get("cases", [])),
                                               "all_position_legal": all(case.get("all_position_legal", False) for case in rank.get("generation", {}).get("cases", []))} for rank in ranks],
                       "parameter_scope": [rank.get("parameter_scope") for rank in ranks],
                       "rank_reports": [source(directory / "output/probe" / ("rank" + str(rank) + ".json")) for rank in (0, 1) if (directory / "output/probe" / ("rank" + str(rank) + ".json")).exists()]}
            candidate["attempts"].append(attempt)
            candidate.update({name: attempt[name] for name in ("job_id", "compute_status", "long_tail_full_zero_status", "model_or_adapter_save_reload", "generation17_beam20_status", "training_update_peak_allocated_gib", "training_update_peak_reserved_gib", "ordinary_mixed_timing")})
            candidate["route_status"] = "PARTIAL" if compute != "BLOCKED" else "BLOCKED"
        results["routes"][key + "-" + route] = candidate
    cost_path = args.evidence / "budget.json"
    if cost_path.exists():
        results["budget_identity"] = source(cost_path)
    results["new_packages"] = {"system": [], "language": [], "toolchains": [], "global_config_changes": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"overall_status": results["overall_status"], "routes": {key: {name: value[name] for name in ["route_status", "compute_status", "long_tail_full_zero_status"] if name in value} for key, value in results["routes"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
