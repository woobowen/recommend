"""Independent saved-artifact review for one complete Toys/Sports Shuffle test.

This reads original JSON/JSONL and runtime load receipts. It never imports torch,
loads a checkpoint, executes a model, filters candidates, or changes source data.
Metric formula follows the accepted Beauty stdlib verifier; positive/negative
score ties are rejected because Python tie ordering cannot certify torch.topk.
"""
import argparse
import ast
from collections import Counter
import datetime as dt
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys


REVIEWER = "/root/results_audit"
METRICS = ("recall@5", "ndcg@5", "recall@10", "ndcg@10")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    assert not path.exists(), f"Preserve existing review: {path}"
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def metric_values(document):
    values = document["metrics"]
    if isinstance(values, list):
        assert len(values) == 1
        return {name.removeprefix("test/"): value for name, value in values[0].items()}
    return values


def verify_trace(path, lifecycle_path, route, n, batch_count):
    trace, cleanup = read(path), read(lifecycle_path)
    roles = trace["encoder_role_counts"]
    assert roles["loss"] == roles["generate"] == batch_count
    assert trace["encoder_calls"] == 2 * batch_count
    assert trace["shuffle_n"] == n and trace["loss_and_generate_sample_separately"]
    assert trace["exception"] is None
    if route == "A":
        calls, events = trace["shuffle_call_count"], trace["permutation_events"]
    else:
        calls, events = trace["shuffle_summary"]["calls"], trace["shuffle_events"]
        assert all(trace["shuffle_role_counts"].get(role, 0) == (batch_count if n else 0) for role in ("loss", "generate"))
    assert calls == (2 * batch_count if n else 0)
    assert bool(events) == bool(n)
    variants, observed_roles = set(), set()
    for event in events or []:
        observed_roles.add(event["role"])
        assert event["role"] in ("loss", "generate")
        permutations = event.get("permutation", event.get("permutations"))
        assert permutations
        for user in permutations:
            for item in user:
                assert sorted(item) == list(range(n))
                variants.add(tuple(item))
        if route == "B":
            assert event["multiset_exact"] and event["padding_unchanged"] and not event["replay"]
    if n:
        assert observed_roles == {"loss", "generate"} and len(variants) > 1
    assert not cleanup["is_testing_after"] and cleanup["evaluator_restored"]
    assert all(cleanup[k] for k in ("encoder_mask_cleared", "replay_cleared", "author_replay_queue_cleared"))
    assert len(cleanup["real_batch_devices"]) == batch_count
    for batch in cleanup["real_batch_devices"]:
        assert batch["input_device"].startswith("cuda") and batch["parameter_devices"] == [batch["input_device"]]
    return {"encoder_calls": trace["encoder_calls"], "loss_calls": roles["loss"], "generate_calls": roles["generate"],
            "shuffle_calls": calls, "bounded_native_events": len(events or []), "observed_permutation_variants": len(variants),
            "loss_and_generate_sample_separately": True, "lifecycle_cleanup_and_actual_cuda_batch_devices_checked": True,
            "full_test_permutation_history_claimed": False}


def audit(args):
    root = args.job_directory
    request, accepted, status = [read(root / n) for n in ("request.json", "accepted.json", "status.json")]
    invocation = read(root / "output/invocation.json")
    spec, deployment = invocation["spec"], invocation["code_identity"]
    job = request["job_id"]
    assert status["job_id"] == job
    assert status["status"] == "SUCCEEDED" and status["exit_code"] == 0
    assert status["cleanup"]["verified"] and not status.get("worker_may_be_live", False)
    fingerprint = hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    assert accepted["request_sha256"] == fingerprint
    assert invocation["phase"] == "evaluate" and invocation["interpreter"] == "/data/bwwu/envs/grid-20261009-e61baf3/bin/python"
    assert request["argv"][3:5] == ["experiments.formal_20261015.run", "evaluate"]
    assert invocation["physical_gpu"] == request["env"]["CUDA_VISIBLE_DEVICES"]
    assert spec["domain"] in ("toys", "sports") and spec["route"] in ("A", "B") and spec["shuffle_n"] in (0, 3, 4)
    assert spec["seed"] == 42 and spec["generation_k"] == 10 and spec["formal_approved"]
    assert read(args.spec) == spec and sha(args.spec) == invocation["spec_sha256"]
    assert deployment["snapshot_sha256"] == hashlib.sha256(json.dumps({"source_commit": deployment["source_commit"], "files": deployment["files"]}, sort_keys=True).encode()).hexdigest()
    assert request["capture_version"] == deployment["source_commit"] + "/" + deployment["snapshot_sha256"]
    for row in deployment["files"]:
        if row["path"].endswith(".py"):
            source = args.repo / row["path"]
            assert source.stat().st_size == row["bytes"] and sha(source) == row["sha256"], source
    source_inputs = {row["path"]: row["sha256"] for row in request["inputs"]}
    manifest_sha = hashlib.sha256((json.dumps(deployment, indent=2) + "\n").encode()).hexdigest()
    assert source_inputs[request["cwd"] + "/DEPLOYMENT.json"] == manifest_sha
    assert source_inputs[request["argv"][request["argv"].index("--spec") + 1]] == invocation["spec_sha256"]
    for name in ("config", "sid", "checkpoint", "test_index", "nonauthor_gate"):
        assert source_inputs[spec[name]["path"]] == spec[name]["sha256"]
    assert sha(args.targets) == spec["test_index"]["sha256"]
    index = Counter()
    for line in args.targets.read_bytes().splitlines():
        row = json.loads(line)
        assert len(row["target"]) == 4
        index[(row["sample_key"], tuple(row["target"]))] += 1
    count = sum(index.values())
    assert count > 0 and len(index) == len({key[0] for key in index}) == count
    legal_document = read(args.legal_sids)
    assert legal_document["source"] == spec["sid"] and legal_document["shape"][0] == 4
    legal = set(map(tuple, legal_document["legal_raw_sids"]))
    assert len(legal) == legal_document["unique_four_code_count"]
    used, loading = read(root / "output/used_config.json"), read(root / "output/loading.json")
    assert used["model"]["top_k_for_generation"] == 10 and used["model"]["num_hierarchies"] == 4
    if spec["route"] == "B":
        assert used["model"]["shuffle_n"] == spec["shuffle_n"]
    else:
        assert used["model"]["shuffle_embedding_on_test"] is bool(spec["shuffle_n"])
        assert used["model"]["hierarchies_to_shuffle"] == (spec["shuffle_n"] or 3)
    prefix_source = args.repo / ("GRID baseline/src/models/modules/semantic_id/tiger_generation_model.py" if spec["route"] == "B" else "references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py")
    if "should_check_prefix" in used["model"]:
        assert used["model"]["should_check_prefix"] is False
        prefix_check = {"effective_value": False, "basis": "explicit used_config.model.should_check_prefix"}
    else:
        tree = ast.parse(prefix_source.read_bytes())
        initializer = next(method for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SemanticIDEncoderDecoder" for method in node.body if isinstance(method, ast.FunctionDef) and method.name == "__init__")
        defaults = dict(zip((arg.arg for arg in initializer.args.args[-len(initializer.args.defaults):]), initializer.args.defaults))
        assert isinstance(defaults["should_check_prefix"], ast.Constant) and defaults["should_check_prefix"].value is False
        prefix_check = {"effective_value": False, "basis": "used config omits field; byte-bound actual SemanticIDEncoderDecoder constructor default is False", "source": identity(prefix_source)}
    assert not any(k.startswith("limit_") for k in used["trainer"])
    assert not used["trainer"].get("fast_dev_run") and not used["trainer"].get("overfit_batches")
    assert used["eval"]["evaluator"]["top_k_list"] == [5, 10]
    strict = loading["strict_load"]
    assert strict["missing_keys"] == strict["unexpected_keys"] == []
    assert strict["all_parameter_keys_shapes_values_equal"] and strict["embedded_sid_exact"]
    assert loading["shared_sid_table"] is (spec["route"] == "B")
    loaded = {r["path"]: r for r in loading["assets"]}
    for name in ("sid", "checkpoint"):
        assert loaded[spec[name]["path"]]["sha256"] == spec[name]["sha256"] and loaded[spec[name]["path"]]["bytes"] > 1024
    original = read(root / "output/test/recomputed_metrics.json")
    trainer = read(root / "output/test/metrics.json")
    result = read(root / "output/RESULT.json")
    assert result["status"] == "COMPLETE" and result["formal"] and result["test_complete"]
    assert result["route"] == spec["route"] and result["shuffle_n"] == spec["shuffle_n"]
    assert all(result[name] == spec[name] for name in ("config", "sid", "checkpoint"))
    assert result["metrics"] == original and original["coverage_exact"]
    assert original["samples"] == original["unique_users"] == count
    assert trainer["formal"] and trainer["samples"] == count
    assert "Trainer.test ckpt_path=None" in trainer["checkpoint_load"]
    actual, invalid_hist = Counter(), Counter()
    totals = {name: [] for name in METRICS}
    maximum = {name: 0.0 for name in METRICS}
    invalid = duplicates = targets_outside = ties = rows = 0
    diagnostics = args.output.with_name(args.output.stem + "-per-user.jsonl.gz")
    assert not diagnostics.exists()
    prediction_path, sample_path = root / "output/test/predictions.jsonl", root / "output/test/per_sample_metrics.jsonl"
    with prediction_path.open() as predictions, sample_path.open() as samples, diagnostics.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as out:
        for line in predictions:
            row = json.loads(line)
            per = json.loads(samples.readline())
            key, target = row["sample_key"], tuple(row["target"])
            ids, scores = list(map(tuple, row["sids"])), row["scores"]
            assert per["sample_key"] == key and isinstance(key, str) and len(key) == 24
            assert len(ids) == len(scores) == 10 and len(target) == 4
            assert all(len(sid) == 4 and all(type(v) is int and v >= 0 for v in sid) for sid in ids)
            assert all(math.isfinite(v) for v in scores) and all(a >= b for a, b in zip(scores, scores[1:]))
            actual[(key, target)] += 1
            row_invalid = sum(sid not in legal for sid in ids)
            invalid += row_invalid; invalid_hist[row_invalid] += 1
            duplicates += len(ids) - len(set(ids)); targets_outside += target not in legal
            relevant = [sid == target for sid in ids]
            groups = {}
            for score, hit in zip(scores, relevant):
                groups.setdefault(score, set()).add(hit)
            assert all(len(group) == 1 for group in groups.values()), "Ambiguous positive/negative score tie requires exact native topk review"
            ties += len(set(scores)) != 10
            order, relevant_count = sorted(range(10), key=lambda i: scores[i], reverse=True), sum(relevant)
            each = {}
            for k in (5, 10):
                chosen = [int(relevant[i]) for i in order[:k]]
                each[f"recall@{k}"] = sum(chosen) / max(min(relevant_count, k), 1)
                ideal = math.fsum(1 / math.log2(i + 2) for i in range(min(relevant_count, k)))
                each[f"ndcg@{k}"] = math.fsum(v / math.log2(i + 2) for i, v in enumerate(chosen)) / (ideal or 1)
            differences = {}
            for name, value in each.items():
                diff = abs(value - per["metrics"][name]); assert diff <= 1e-7, (key, name, diff)
                maximum[name] = max(maximum[name], diff); differences[name] = diff; totals[name].append(value)
            out.write((json.dumps({"sample_key": key, "independent_metrics": each, "absolute_differences": differences, "invalid_candidates": row_invalid}, separators=(",", ":")) + "\n").encode())
            rows += 1
        assert not samples.readline()
    assert actual == index and rows == count and targets_outside == 0
    assert invalid == original["invalid_candidate_count"] and duplicates == original["duplicate_candidate_count"] == 0
    assert sha(prediction_path) == original["prediction_sha256"] and sha(sample_path) == original["per_sample_metrics_sha256"]
    values = {name: math.fsum(v) / count for name, v in totals.items()}
    reference, recomputed = metric_values(trainer), original["metrics"]
    differences = {name: abs(value - reference[name]) for name, value in values.items()}
    assert all(v <= 1e-7 for v in differences.values())
    assert all(abs(values[name] - recomputed[name]) <= 1e-7 for name in METRICS)
    lifecycle = read(root / "output/test/lifecycle_cleanup.json")
    batch_count = len(lifecycle["real_batch_devices"])
    assert batch_count > 0 and [b["batch_index"] for b in lifecycle["real_batch_devices"]] == list(range(batch_count))
    if used["datamodule"]["test_dataloader_config"]["num_workers"] == 0:
        assert batch_count == math.ceil(count / used["datamodule"]["test_dataloader_config"]["batch_size_per_device"])
    trace = verify_trace(root / "output/intervention_trace.json", root / "output/test/lifecycle_cleanup.json", spec["route"], spec["shuffle_n"], batch_count)
    assert result["intervention_calls"] == trace["shuffle_calls"]
    checked = []
    artifacts = {r["path"]: r for r in status["artifacts"]}
    for relative in ("output/RESULT.json", "output/invocation.json", "output/used_config.json", "output/loading.json", "output/intervention_trace.json", "output/test/predictions.jsonl", "output/test/per_sample_metrics.jsonl", "output/test/recomputed_metrics.json", "output/test/metrics.json", "output/test/lifecycle_cleanup.json"):
        local, saved = root / relative, artifacts[relative]
        assert sha(local) == saved["sha256"] and local.stat().st_size == saved["bytes"]
        checked.append(identity(local))
    wall = (dt.datetime.fromisoformat(status["ended_at"]) - dt.datetime.fromisoformat(status["started_at"])).total_seconds()
    assert 0 < wall <= request["timeout_seconds"] <= 1800
    receipt = {"status": "FORMAL_EVALUATION_ACCEPTANCE_PASS", "reviewer": REVIEWER,
        "reviewed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "domain": spec["domain"], "route": spec["route"], "shuffle_n": spec["shuffle_n"], "job_id": job,
        "scope": "Complete original saved test predictions, every per-user metric, candidate legality, exact coverage, runtime load/identity/trace receipts, and terminal worker budget. No checkpoint payload deserialization or new model computation.",
        "source_commit": deployment["source_commit"], "snapshot_sha256": deployment["snapshot_sha256"], "DEPLOYMENT_raw_sha256": manifest_sha,
        "spec_sha256": invocation["spec_sha256"], "source_python_bytes_match_current_reviewed_checkout": True, "request_sha256": fingerprint,
        "config": spec["config"], "sid": spec["sid"], "checkpoint": spec["checkpoint"], "test_index": spec["test_index"],
        "prefix_constraint_configuration": prefix_check,
        "samples": count, "unique_users": len({k[0] for k in actual}), "exact_key_target_multiset": True,
        "metrics": recomputed, "independent_stdlib_metrics": values, "independent_to_trainer_absolute_differences": differences,
        "all_per_user_metrics_checked": count, "maximum_per_user_metric_absolute_difference": maximum, "per_user_diagnostics": identity(diagnostics),
        "prediction_sha256": sha(prediction_path), "targets_sha256": sha(args.targets), "legal_vocabulary": identity(args.legal_sids),
        "invalid_candidate_count": invalid, "invalid_candidates_per_user_histogram": dict(sorted(invalid_hist.items())), "duplicate_candidate_count": duplicates,
        "targets_outside_legal_vocabulary": targets_outside, "score_tie_rows": ties, "positive_negative_score_tie_rows": 0,
        "original_candidate_filtering_deduplication_rescoring_or_reordering_applied": False,
        "strict_load_runtime_receipt": strict, "checkpoint_payload_deserialized_by_reviewer": False,
        "intervention_trace": trace, "terminal_state": status["status"], "exit_code": status["exit_code"], "worker_may_be_live": False,
        "worker_wall_seconds_including_CPU_preparation": wall, "requested_budget_seconds": request["timeout_seconds"], "GPU_request_worker_wall_hours": wall / 3600,
        "physical_GPU": invocation["physical_gpu"], "inputs": checked + [identity(args.targets), identity(args.spec), identity(root / "request.json"), identity(root / "accepted.json"), identity(root / "status.json")],
        "program": identity(Path(__file__)), "argv": sys.argv, "new_model_or_GPU_calls": 0, "Git_or_original_evidence_writes": 0,
        "boundaries": ["Semantic historical pairing and weight/SID selection depend on the companion asset/source audit; shapes and hashes alone do not establish that pairing.", "Normal OFF equivalence and occurrence-level random sampling implementation depend on the companion source/preflight review; bounded runtime traces certify actual test hook activation.", "This result receipt does not accept attention or claim cross-seed stability, invariance, or a required metric value."]}
    write(args.output, receipt)
    print(json.dumps({"review": identity(args.output), "status": receipt["status"], "job_id": job, "domain": spec["domain"], "route": spec["route"], "shuffle_n": spec["shuffle_n"], "samples": count, "invalid_candidates": invalid, "wall_seconds": wall}, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job-directory", type=Path, required=True)
    p.add_argument("--targets", type=Path, required=True)
    p.add_argument("--legal-sids", type=Path, required=True)
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--repo", type=Path, default=Path("/home/addaswsw/recommend-worktrees/sid-extensions-p0"))
    p.add_argument("--output", type=Path, required=True)
    audit(p.parse_args())
