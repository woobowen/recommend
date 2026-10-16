"""Independent CPU/stdlib review of existing formal MLP saved results only.

Training/validation selection is separately audited by the source reviewer.
This verifies complete prediction evidence, every per-sample original metric,
vocabulary behavior, accepted identity, strict-load receipt and terminal budget.
No model, torch, checkpoint deserialization, GPU or Git operations.
"""
import argparse
from collections import Counter
import datetime as dt
import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path):
    p = Path(path)
    return {"path": str(p), "bytes": p.stat().st_size, "sha256": sha(p)}


def load(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    assert not path.exists(), f"Preserve existing review: {path}"
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--support", type=Path, required=True)
    p.add_argument("--metadata-jobs", type=Path, required=True)
    p.add_argument("--repo", type=Path, required=True)
    a = p.parse_args()
    receipt = load(a.support / "collection-receipt.json")
    collected = Path(receipt["directory"])
    manifest_path = collected / "COLLECTION_MANIFEST.json"
    manifest = load(manifest_path)
    expected = set()
    for entry in manifest["files"]:
        name = entry["path"]
        assert name not in expected
        expected.add(name)
        q = PurePosixPath(name)
        assert not q.is_absolute() and ".." not in q.parts
        local = collected / name
        assert not local.is_symlink() and local.stat().st_size == entry["bytes"] and sha(local) == entry["sha256"]
    assert expected == {str(v.relative_to(collected)) for v in collected.rglob("*") if v.is_file() and v != manifest_path}
    assert manifest["checkpoints_excluded"] and manifest["source_data_excluded"]
    source_map_reference = load(a.support / "results-review-r4/beauty-six-gate-review-index.json")["source_python_sha256"]
    destination = a.support / "mlp-results-review"
    completed = []
    for domain, count, legal_sha in (
        ("toys", 19412, "2f7cc7d373e342c6627ba14d0bede2afe9c1349534e8b2224125261a4b587885"),
        ("sports", 35598, "29cc6e5b496d8fba37d0c3b6a128b46092598d330b569b4bb48e7c5bbfd2a682"),
    ):
        job_id = f"gf-20261015-{domain}-mlp-formal-v1"
        j = collected / job_id
        request, accepted, status, cleanup = [load(j / name) for name in ("request.json", "accepted.json", "status.json", "cleanup.json")]
        invocation = load(j / "output/invocation.json")
        result = load(j / "output/RESULT.json")
        training = load(j / "output/training_result.json")
        strict = load(j / "output/best_checkpoint_load.json")
        structure = load(j / "output/structure.json")
        used = load(j / "output/used_config.json")
        original = load(j / "output/test/recomputed_metrics.json")
        trainer = load(j / "output/test/metrics.json")
        independent_path = destination / f"{domain}-independent-stdlib.json"
        independent = load(independent_path)
        spec, deployment = invocation["spec"], invocation["code_identity"]
        fingerprint = hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        assert fingerprint == accepted["request_sha256"]
        assert status["status"] == "SUCCEEDED" and status["exit_code"] == 0 and status["cleanup"] == cleanup
        assert cleanup["verified"] and cleanup["last_observation"]["verified_empty"] and status["worker_may_be_live"] is False
        assert request["argv"][4] == invocation["phase"] == "train" and invocation["physical_gpu"] == request["env"]["CUDA_VISIBLE_DEVICES"]
        assert request["timeout_seconds"] == 43200 and spec["hard_cap_hours"] == 12
        assert spec["domain"] == domain and spec["route"] == "MLP" and spec["shuffle_n"] == 0 and spec["seed"] == 42 and spec["generation_k"] == 10
        assert spec["formal_approved"] and not spec.get("checkpoint") and not spec.get("resume_checkpoint")
        source_map = {v["path"]: v["sha256"] for v in deployment["files"] if v["path"].endswith(".py")}
        assert len(source_map) == 171 and source_map == source_map_reference
        assert deployment["source_commit"] == "4495c38459971aee36b00d881eb50f50fafa3159"
        assert deployment["snapshot_sha256"] == "77c012c07de3db19bc86108212ab292aabe704922f1f65d1d9c51da62b05a108"
        assert request["capture_version"] == deployment["source_commit"] + "/" + deployment["snapshot_sha256"]
        assert deployment["snapshot_sha256"] == hashlib.sha256(json.dumps({"source_commit": deployment["source_commit"], "files": deployment["files"]}, sort_keys=True).encode()).hexdigest()
        input_records = {v["path"]: v["sha256"] for v in request["inputs"]}
        assert input_records[request["argv"][-1]] == invocation["spec_sha256"]
        raw_manifest_sha = hashlib.sha256((json.dumps(deployment, indent=2) + "\n").encode()).hexdigest()
        assert input_records[request["cwd"] + "/DEPLOYMENT.json"] == raw_manifest_sha
        for name in ("config", "sid", "evaluation_index", "test_index", "attention_cohort", "nonauthor_gate"):
            assert input_records[spec[name]["path"]] == spec[name]["sha256"]
        admission_path = a.support / f"mlp-admission/{domain}-gate-review.json"
        admission = load(admission_path)
        authorization_path = a.support / f"{domain}-formal-authorization.json"
        authorization = load(authorization_path)
        plan = {k: v for k, v in spec.items() if k not in ("formal_approved", "nonauthor_gate")}
        plan_sha = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
        assert authorization["kind"] == "FORMAL_RUN_AUTHORIZED" and authorization["plan_sha256"] == admission["plan_sha256"] == plan_sha
        assert sha(authorization_path) == spec["nonauthor_gate"]["sha256"] and sha(admission_path) == authorization["independent_review"]["sha256"]
        assert source_map == authorization["source_python_sha256"] == admission["source_python_sha256"]
        for name in ("config", "sid"):
            assert spec[name] == authorization["assets"][name] == admission["assets"][name]
        assert result["status"] == training["status"] == "COMPLETE" and result["formal"] and training["formal"]
        assert training["training_complete"] and training["test_complete"] and training["fresh_initialization"] and training["full_parameter_training"]
        assert all(result[k] == v for k, v in training.items())
        assert result["checkpoint"] == training["best_checkpoint"] == strict["checkpoint"]
        assert result["sid"] == training["sid_asset"] == spec["sid"] and result["config"] == training["approved_config_asset"] == spec["config"]
        assert strict["missing_keys"] == strict["unexpected_keys"] == []
        assert strict["all_keys_shapes_values_exact"] and strict["embedded_sid_exact"] and strict["optimizer_and_callbacks_present"]
        assert strict["state_tensor_count"] == len(structure["state_dict_shapes"]) == 180
        assert structure["shared_sid_table"] and not structure["has_sep"] and structure["parameters_total"] == structure["parameters_trainable"]
        assert structure["num_hierarchies"] == 4 and structure["embedding_dim"] == 128 and structure["top_k_for_generation"] == 10
        for key, shape in {"item_aggregation_mlp.0.weight": [256, 512], "item_aggregation_mlp.3.weight": [128, 256]}.items():
            assert structure["state_dict_shapes"][key] == shape
        assert used["model"]["_target_"] == "experiments.beauty_extensions.author.models.AuthorMLPModel"
        assert not any(name.startswith("limit_") for name in used["trainer"]) and not used["trainer"].get("fast_dev_run") and not used["trainer"].get("overfit_batches")
        assert trainer["formal"] and trainer["samples"] == training["test_target_count"] == count
        assert trainer["complete_test_coverage"] and trainer["bounded_test_batches"] is None
        assert training["test_metrics"] == original and original["coverage_exact"] and original["samples"] == original["unique_users"] == count
        assert original["prediction_sha256"] == independent["prediction_sha256"] == sha(j / "output/test/predictions.jsonl")
        assert original["per_sample_metrics_sha256"] == sha(j / "output/test/per_sample_metrics.jsonl")
        assert independent["rows"] == independent["unique_users"] == count and independent["exact_key_target_multiset"]
        assert max(independent["absolute_differences"].values()) <= 1e-7
        index_path = a.metadata_jobs / f"gf-20261015-{domain}-metadata-v1/retrieved/output/test_target_index.jsonl"
        assert sha(index_path) == independent["targets_sha256"] == spec["test_index"]["sha256"]
        legal_path = a.support / f"{domain}-legal-sids.json"
        assert sha(legal_path) == legal_sha
        vocabulary = load(legal_path)
        assert vocabulary["source"] == spec["sid"] and vocabulary["shape"][0] == 4
        legal = set(map(tuple, vocabulary["legal_raw_sids"]))
        assert len(legal) == vocabulary["unique_four_code_count"] == vocabulary["row_count"]
        rows = invalid = duplicates = outside = hit5 = hit10 = tie_rows = 0
        maximum = {k: 0.0 for k in independent["metrics_independent_stdlib"]}
        differences_nonzero_users = Counter()
        diagnostics_path = destination / f"{domain}-independent-per-sample.jsonl.gz"
        assert not diagnostics_path.exists()
        with (j / "output/test/predictions.jsonl").open() as predictions, (j / "output/test/per_sample_metrics.jsonl").open() as per_sample, diagnostics_path.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as diagnostics:
            for line in predictions:
                row = json.loads(line)
                saved = json.loads(per_sample.readline())
                assert saved["sample_key"] == row["sample_key"]
                candidates = list(map(tuple, row["sids"]))
                target, scores = tuple(row["target"]), row["scores"]
                invalid += sum(v not in legal for v in candidates)
                duplicates += len(candidates) - len(set(candidates))
                outside += target not in legal
                relevant = [v == target for v in candidates]
                relevant_count = sum(relevant)
                order = sorted(range(10), key=lambda i: scores[i], reverse=True)
                tie_rows += len(set(scores)) != 10
                each = {}
                differences = {}
                for k in (5, 10):
                    chosen = [int(relevant[i]) for i in order[:k]]
                    if k == 5:
                        hit5 += sum(chosen)
                    else:
                        hit10 += sum(chosen)
                    each[f"recall@{k}"] = sum(chosen) / max(min(relevant_count, k), 1)
                    ideal = sum(1 / math.log2(i + 2) for i in range(min(relevant_count, k)))
                    each[f"ndcg@{k}"] = sum(v / math.log2(i + 2) for i, v in enumerate(chosen)) / (ideal or 1)
                for key, value in each.items():
                    difference = abs(value - saved["metrics"][key])
                    assert difference <= 1e-7
                    maximum[key] = max(maximum[key], difference)
                    differences[key] = difference
                    differences_nonzero_users[key] += difference != 0
                diagnostics.write((json.dumps({"sample_key": row["sample_key"], "independent_metrics": each, "absolute_differences": differences}, separators=(",", ":")) + "\n").encode())
                rows += 1
            assert not per_sample.readline()
        assert rows == count and invalid == original["invalid_candidate_count"] and duplicates == original["duplicate_candidate_count"] == 0 and outside == 0
        artifact_by_path = {v["path"]: v for v in status["artifacts"]}
        checkpoint_relative = "output/checkpoints/" + Path(training["best_checkpoint"]["path"]).name
        checkpoint_artifact = artifact_by_path[checkpoint_relative]
        assert checkpoint_artifact["sha256"] == training["best_checkpoint"]["sha256"] and checkpoint_artifact["bytes"] == training["best_checkpoint"]["bytes"]
        for path in ("output/RESULT.json", "output/training_result.json", "output/best_checkpoint_load.json", "output/test/predictions.jsonl", "output/test/per_sample_metrics.jsonl", "output/test/metrics.json", "output/test/recomputed_metrics.json"):
            assert artifact_by_path[path]["sha256"] == sha(j / path) and artifact_by_path[path]["bytes"] == (j / path).stat().st_size
        relevant_sources = ["references/wmh-original/GRID/src/components/eval_metrics.py", "experiments/beauty_extensions/results.py", "experiments/formal_20261015/train.py"]
        for name in relevant_sources:
            assert sha(a.repo / name) == source_map[name]
        wall = (dt.datetime.fromisoformat(status["ended_at"]) - dt.datetime.fromisoformat(status["started_at"])).total_seconds()
        assert 0 < wall < 43200 and 0 < training["fit_wall_seconds"] < wall
        report = {
            "schema": "grid-formal-MLP-full-prediction-independent-review-v1",
            "status": "FORMAL_EVALUATION_ACCEPTANCE_PASS", "reviewer": "/root/results_publication_review",
            "reviewed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "domain": domain,
            "scope": "Actual complete formal test prediction/evaluator/identity/strict-load receipt acceptance; training/validation selection is covered by the companion nonauthor source review.",
            "job": {"job_id": job_id, "status": status["status"], "exit_code": 0, "cleanup_verified": True,
                    "worker_may_be_live": False, "started_at": status["started_at"], "ended_at": status["ended_at"],
                    "worker_wall_seconds_including_training_validation_best_restore_test_recompute_cleanup": wall,
                    "GPU_request_worker_wall_hours": wall / 3600, "allowed_seconds": 43200,
                    "remaining_from_per_domain_12h_allowance_seconds": 43200 - wall,
                    "physical_GPU_uuid": invocation["physical_gpu"],
                    "fit_wall_seconds_recorded": training["fit_wall_seconds"],
                    "gpu_peak_allocated_bytes_recorded": training["gpu_peak_allocated_bytes"],
                    "gpu_peak_reserved_bytes_recorded": training["gpu_peak_reserved_bytes"]},
            "identity": {"source_commit": deployment["source_commit"], "snapshot_sha256": deployment["snapshot_sha256"],
                         "DEPLOYMENT_raw_sha256": raw_manifest_sha, "source_python_count": 171,
                         "source_map_exact_to_previously_reviewed_current_source": True,
                         "plan_sha256": plan_sha, "spec_sha256": invocation["spec_sha256"],
                         "accepted_request_fingerprint_sha256": fingerprint,
                         "request_inputs_bound": len(input_records), "approved_protocol": spec["approved_protocol"],
                         "config": spec["config"], "sid": spec["sid"], "test_index": spec["test_index"],
                         "original_non_author_admission": record(admission_path), "controller_authorization": record(authorization_path)},
            "collection": {"receipt": record(a.support / "collection-receipt.json"),
                           "manifest": record(manifest_path), "all_collected_members_size_and_sha256_independently_verified": True,
                           "collected_member_count": len(manifest["files"]), "big_checkpoints_intentionally_excluded": True},
            "complete_predictions": {"samples": count, "unique_users": count, "candidates": count * 10,
                                     "candidate_shape_per_user": [10, 4], "exact_original_key_target_multiset": True,
                                     "prediction_sha256": original["prediction_sha256"], "targets_sha256": independent["targets_sha256"],
                                     "finite_scores": True, "score_tie_rows": tie_rows,
                                     "positive_negative_ambiguous_score_ties": 0,
                                     "invalid_candidates_independently_counted": invalid,
                                     "duplicate_candidates_independently_counted": duplicates,
                                     "targets_outside_legal_vocabulary": outside,
                                     "source_legal_unique_SIDs": len(legal), "legal_vocabulary_projection": record(legal_path),
                                     "original_invalid_candidates_retained_without_filter_completion_rescore_or_reorder": True},
            "metrics": {"original_evaluator_metrics": original["metrics"],
                        "independent_stdlib_metrics": independent["metrics_independent_stdlib"],
                        "independent_to_Trainer_absolute_differences": independent["absolute_differences"],
                        "all_per_user_saved_original_metrics_compared": count,
                        "per_sample_max_absolute_differences": maximum,
                        "per_sample_nonzero_float_rounding_differences_user_counts": dict(differences_nonzero_users),
                        "all_independent_per_user_metrics_and_differences": record(diagnostics_path),
                        "matching_candidates_at5": hit5, "matching_candidates_at10": hit10, "absolute_tolerance": 1e-7,
                        "formula": "Original exact four-code equality, scores-ranked topk, Recall relevant-count denominator clamped to1 and original NDCG DCG/IDCG. No vocabulary filter or evaluator input modification."},
            "checkpoint_evidence": {"best_checkpoint": training["best_checkpoint"],
                                    "strict_load_receipt": strict,
                                    "strict_load_and_initial_structure_have_same180_state_entries": True,
                                    "shared_sid_table": True, "SEP_removed": True,
                                    "original_per_item_MLP_linear_shapes": [[256, 512], [128, 256]],
                                    "matching_runner_checkpoint_artifact": checkpoint_artifact,
                                    "big_checkpoint_payload_downloaded_or_deserialized_by_reviewer": False},
            "training_facts_as_recorded_pending_companion_trace_audit": {
                "termination": training["termination"], "global_step": training["global_step"],
                "best_global_step": strict["best_global_step"], "completed_full_validations": training["completed_full_validations"],
                "actual_microbatches": training["actual_microbatches"], "expanded_windows": training["expanded_windows"],
                "early_stopping": training["early_stopping"], "fresh_initialization": True,
                "full_parameter_training": True, "MLP_parameter_updates": training["mlp_parameter_max_abs_updates"],
                "not_substituted_for_source_reviewer_independent_training_selection_trace_audit": True},
            "boundaries": [
                "Complete predictions are original saved test outputs from the runtime strict-loaded validation-best checkpoint; no model was rerun.",
                "The big selected checkpoint remains remote. Its hash/size match original runner artifact and runtime strict-load receipt; this reviewer did not deserialize its payload.",
                "Toys historical bundle retains original provenance limitations; Sports is an approved new Qwen/RK plus original per-item MLP experiment, not historical exact Sports reproduction.",
                "Training termination/validation-best/callback/optimizer trajectory are separately covered by source_conditions_review; this result report does not self-sign that companion audit.",
                "No claim of exact uninterrupted resume: original recovery lacks infinite worker cursor/prefetch and unfinished gradients.",
                "Any future attention plan needs its own exact best checkpoint/config/SID/cohort/evaluation identity admission. This report does not authorize or accept attention output.",
                "Original3/3 correction rounds and one separately approved configuration serialization exception remain consumed; no repair or reset occurred here."],
            "reviewer_actions": {"repository_edits": 0, "Git_operations": 0, "remote_actions": 0,
                                 "models_loaded": 0, "GPU_calls": 0, "CPU_saved_predictions_only": True},
            "inputs": [record(j / path) for path in (
                "request.json", "accepted.json", "status.json", "cleanup.json", "work.log", "output/invocation.json",
                "output/RESULT.json", "output/training_result.json", "output/best_checkpoint_load.json", "output/structure.json",
                "output/used_config.json", "output/domain_protocol.json", "output/test/predictions.jsonl",
                "output/test/per_sample_metrics.jsonl", "output/test/metrics.json", "output/test/recomputed_metrics.json")]
                + [record(index_path), record(independent_path), record(Path(__file__).resolve())],
        }
        output = destination / f"{domain}-formal-review.json"
        write_json(output, report)
        completed.append({"review": record(output), "domain": domain, "status": report["status"],
                          "invalid_candidates": invalid, "duplicates": duplicates,
                          "max_metric_difference": max(independent["absolute_differences"].values()),
                          "worker_wall_seconds": wall, "remaining_allowance_seconds": 43200 - wall})
    invocation = {"argv": sys.argv, "interpreter": sys.executable, "script": record(Path(__file__).resolve()),
                  "collection_manifest": record(manifest_path), "outputs": completed,
                  "exit_status": "COMPLETED", "model_or_GPU_execution": False}
    write_json(destination / "formal-prediction-review-invocation.json", invocation)
    print(json.dumps(completed, indent=2))


if __name__ == "__main__":
    main()
