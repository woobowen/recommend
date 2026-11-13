#!/usr/bin/env python3
"""Full A1 Qwen3 Instruments prompt0/beam20 test, with auditable rank predictions.

Run under two-rank torchrun after best-checkpoint adapter export. The author's
LoRA inference branch uses a full BF16 base, not the int8 training base. This
entry keeps the original positional token callback and unfiltered evaluator,
including its treatment of duplicate/colliding SIDs. It never reduces beams.
The aggregation helpers import only the Python standard library and can be
used to independently recheck the saved predictions without loading a model.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import resource
import shutil
import socket
import sys
import time
import traceback
from types import SimpleNamespace


METRICS = ("hit@5", "hit@10", "ndcg@5", "ndcg@10")
BEAMS = 20
INPUT_SHA256 = {
    "index": "33c87e28c52ae367985bdf494cb49d676d181e13a39497b4ad035f46111546dd",
    "inter": "0a11d47791d90483005da93c36d826679e20e4592d1d2b86518d5f56f218e743",
    "item": "1d9938e576a8d964e6b977a2b63aa56402840a788eaa6685f26f0d910a7133f2",
}
MODEL_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def author_source(root):
    root = Path(root).resolve()
    nested = root / "lc_rec/letter/vendor/LETTER/LETTER-LC-Rec"
    return nested if nested.is_dir() else root


def load_evaluator(source):
    """Load the actual fixed vendor evaluator, without importing Torch/data."""
    spec = importlib.util.spec_from_file_location("a0_original_evaluate", Path(source) / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalized_prediction(decoded):
    # Exactly the author's normalization; do not replace with a regex parser.
    return decoded.split("Response:")[-1].strip().replace(" ", "")


def catalog_and_users(data_root, verify_hashes=True):
    root = Path(data_root) / "Instruments"
    files = {kind: root / f"Instruments.{kind}.json" for kind in INPUT_SHA256}
    identities = {}
    for kind, path in files.items():
        digest = sha(path)
        if verify_hashes and digest != INPUT_SHA256[kind]:
            raise AssertionError(f"Fixed author {kind} identity mismatch: {path}")
        identities[kind] = {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest}
    indices = json.loads(files["index"].read_text())
    inters = json.loads(files["inter"].read_text())
    items = json.loads(files["item"].read_text())
    if len(indices) != 9922 or len(inters) != 24772 or set(indices) != set(items):
        raise AssertionError("Author item/user axis differs from frozen Instruments assets")
    catalog = {}
    for item, tokens in indices.items():
        if len(tokens) != 4:
            raise AssertionError("Author SID does not have four positions")
        catalog.setdefault("".join(tokens), []).append(item)
    users = []
    for row_idx, (uid, history) in enumerate(inters.items()):
        if not history or any(str(item) not in indices for item in history):
            raise AssertionError("Interaction/SID item axis mismatch")
        item = str(history[-1])
        users.append({"row_idx": row_idx, "user_id": uid, "target_item_id": item,
                      "target": "".join(indices[item])})
    if len(catalog) != 9897 or sum(len(ids) > 1 for ids in catalog.values()) != 23:
        raise AssertionError("Author catalog SID collision identity changed")
    return indices, catalog, users, identities


def audit_prediction(row, expected_user, catalog, evaluator):
    """Return original relevance/metrics and independent anomaly counts."""
    for key in ("row_idx", "user_id", "target_item_id", "target"):
        if row.get(key) != expected_user[key]:
            raise AssertionError(f"Prediction identity/target mismatch: {key}")
    candidates = row["candidates"]
    if len(candidates) != BEAMS or [item["generation_order"] for item in candidates] != list(range(BEAMS)):
        raise AssertionError("Prediction must preserve exactly twenty original beams")
    decoded = [item["raw_decoded"] for item in candidates]
    scores = [item["sequence_score"] for item in candidates]
    if any(not isinstance(score, (int, float)) or not math.isfinite(score) for score in scores):
        raise AssertionError("Nonfinite or nonnumeric candidate score")
    predictions = [normalized_prediction(text) for text in decoded]
    if any(item["sid"] != sid for item, sid in zip(candidates, predictions)):
        raise AssertionError("Saved SID does not match original decoded-string parsing")
    order = sorted(range(BEAMS), key=lambda index: scores[index], reverse=True)
    relevance = [int(predictions[index] == row["target"]) for index in order]
    original = evaluator.get_topk_results(decoded, list(scores), [row["target"]], BEAMS, all_items=None)
    if original != [relevance]:
        raise AssertionError("Independent stable score ordering disagrees with original evaluator")
    metrics = evaluator.get_metrics_results(original, list(METRICS))
    independent = {}
    for k in (5, 10):
        independent[f"hit@{k}"] = float(any(relevance[:k]))
        independent[f"ndcg@{k}"] = sum(value / math.log2(i + 2) for i, value in enumerate(relevance[:k]))
    if any(not math.isclose(metrics[key], independent[key], rel_tol=1e-13, abs_tol=1e-13) for key in METRICS):
        raise AssertionError("Independent metric formula disagrees with original evaluator")
    if row.get("sorted_generation_order") != order or row.get("relevance") != relevance:
        raise AssertionError("Saved original candidate order/relevance differs from recomputation")
    invalid = sum(sid not in catalog for sid in predictions)
    colliding = sum(len(catalog.get(sid, ())) > 1 for sid in predictions)
    duplicate_count = len(predictions) - len(set(predictions))
    return metrics, {
        "candidate_count": BEAMS,
        "legal_complete_sid_candidates": BEAMS - invalid,
        "invalid_complete_sid_candidates": invalid,
        "duplicate_candidate_occurrences": duplicate_count,
        "colliding_sid_candidate_occurrences": colliding,
        "users_with_invalid_sid": int(invalid > 0),
        "users_with_duplicate_sid": int(duplicate_count > 0),
        "users_with_colliding_sid_candidate": int(colliding > 0),
        "targets_with_colliding_sid": int(len(catalog[row["target"]]) > 1),
    }


def aggregate_predictions(paths, users, catalog, evaluator, require_rank_done=True, run_commit=None):
    seen, metric_sums, anomaly = set(), dict.fromkeys(METRICS, 0.0), {}
    file_receipts, rank_identity, generation_seconds = [], None, []
    for rank, path in enumerate(map(Path, paths)):
        count = 0
        if require_rank_done:
            status = json.loads((path.parent / f"rank{rank}.json").read_text())
            if status["status"] not in {"PREDICTIONS_COMPLETE", "PASS"}:
                raise AssertionError(f"Rank {rank} did not finish its complete prediction shard")
            if status["rank"] != rank or (run_commit is not None and status["RUN_COMMIT"] != run_commit):
                raise AssertionError("Rank completion identity differs from requested evaluation")
            identity = {key: status[key] for key in ("RUN_COMMIT", "adapter_files", "data_identity",
                                                    "author_sources", "generation", "allowed_callback_sha256", "model")}
            if rank_identity is not None and identity != rank_identity:
                raise AssertionError("Rank shards used different adapter/data/source/generation identities")
            rank_identity = identity
            generation_seconds.append(status["elapsed_generation_seconds"])
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                index = row["row_idx"]
                if type(index) is not int or not 0 <= index < len(users) or index in seen:
                    raise AssertionError("Duplicate/out-of-range test row; no deduplication or skipping permitted")
                if row["rank"] != rank or row["prompt_id"] != 0:
                    raise AssertionError("Prediction shard rank or prompt mismatch")
                metrics, counts = audit_prediction(row, users[index], catalog, evaluator)
                for key, value in metrics.items():
                    metric_sums[key] += value
                for key, value in counts.items():
                    anomaly[key] = anomaly.get(key, 0) + value
                seen.add(index)
                count += 1
        if require_rank_done and count != status["completed_users"]:
            raise AssertionError("Prediction shard length disagrees with rank completion evidence")
        file_receipts.append({"rank": rank, "path": str(path.resolve()), "users": count,
                              "bytes": path.stat().st_size, "sha256": sha(path)})
    if seen != set(range(len(users))):
        raise AssertionError(f"Test coverage incomplete: {len(seen)} of {len(users)} users")
    if not seen:
        raise AssertionError("Empty test cannot be successful")
    denominator = len(seen)
    anomaly["complete_sid_legal_rate"] = anomaly["legal_complete_sid_candidates"] / anomaly["candidate_count"]
    user_identity = hashlib.sha256(json.dumps(users, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    return {
        "status": "PASS", "test_complete": True, "actual_test_users": denominator,
        "valid_result_users": denominator, "failed_users": 0, "skipped_users": 0,
        "all_row_indices_exactly_once": True, "expected_user_axis_sha256": user_identity,
        "metric_sums": metric_sums, "metrics": {key: value / denominator for key, value in metric_sums.items()},
        "prediction_wall_seconds_max_rank": max(generation_seconds) if generation_seconds else None,
        "anomalies": anomaly, "prediction_files": file_receipts,
        "independent_standard_library_recomputation_matches_original_evaluator": True,
        "evaluator_policy": {
            "parsing": "split('Response:')[-1].strip().replace(' ', '')",
            "sorting": "stable descending original sequences_scores",
            "filter_items": False, "deduplicate_candidates": False,
            "collisions": "exact complete SID string match; original item identities are not disambiguated",
            "ndcg": "sum all matching candidate discounts, including repeated matches, as original evaluator",
            "hit": "at least one target SID in first k; one held-out item per user (HR/Recall terminology)",
        },
    }


def strict_reload(args, report, torch, local_rank, source, indices):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "qwen_preflight_20261110/scripts"))
    from common import load_model, tokenizer_for, sharing
    from transformers import AutoTokenizer
    from peft import PeftModel, get_peft_model_state_dict
    assert importlib.metadata.version("transformers") == "4.57.6"
    assert importlib.metadata.version("peft") == "0.21.2"
    adapter = Path(args.adapter).resolve()
    required = ["adapter_config.json", "adapter_model.bin", "tokenizer_config.json", "tokenizer.json"]
    assert all((adapter / name).is_file() for name in required)
    assert not (adapter / "adapter_model.safetensors").exists()
    config = json.loads((adapter / "adapter_config.json").read_text())
    assert (config["r"], config["lora_alpha"], config["lora_dropout"]) == (8, 16, .05)
    assert set(config["target_modules"]) == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"}
    assert set(config["modules_to_save"]) == {"embed_tokens", "lm_head"} and config.get("bias", "none") == "none"
    assert config.get("ensure_weight_tying") is True
    tokens = sorted({token for parts in indices.values() for token in parts})
    reference, metadata = tokenizer_for(args.model, tokens)
    tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True, padding_side="left")
    assert len(tokenizer) == 152587 and tokenizer.get_vocab() == reference.get_vocab()
    assert tokenizer.get_added_vocab() == reference.get_added_vocab()
    assert tokenizer.eos_token_id == reference.eos_token_id and tokenizer.pad_token_id == reference.pad_token_id
    report["tokenizer"] = dict(metadata, padding_side="left", matches_reconstructed_author_tokenizer=True)
    report["adapter_files"] = {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in adapter.iterdir() if p.is_file()}
    model_root = Path(args.model).resolve()
    assert model_root.name == MODEL_REVISION
    report["model"] = {"name": "Qwen/Qwen3-0.6B-Base", "revision": MODEL_REVISION, "path": str(model_root),
                       "config_sha256": sha(model_root / "config.json"), "attention": "native SDPA",
                       "weight_sha256": "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba",
                       "manifest_sha256": sha(model_root / "ASSET_MANIFEST.json"),
                       "weight_bytes": (model_root / "model.safetensors").stat().st_size,
                       "weight_mtime_ns": (model_root / "model.safetensors").stat().st_mtime_ns}
    manifest = json.loads((model_root / "ASSET_MANIFEST.json").read_text())
    assert manifest["revision"] == MODEL_REVISION and manifest["model_id"] == "Qwen/Qwen3-0.6B-Base"
    assert report["model"]["weight_bytes"] == 1192135096
    base, loading = load_model(args.model, "qwen3", quantized=False, device=local_rank)
    base.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    model = PeftModel.from_pretrained(base, str(adapter), torch_dtype=torch.bfloat16,
                                     device_map={"": local_rank}, is_trainable=False, autocast_adapter_dtype=False)
    saved = torch.load(adapter / "adapter_model.bin", map_location="cpu", weights_only=True)
    actual = get_peft_model_state_dict(model, save_embedding_layers=False)
    assert set(saved) == set(actual) and len(saved) == 394
    checks = {}
    for key, tensor in saved.items():
        loaded = actual[key].detach()
        assert loaded.dtype == torch.bfloat16, (key, str(loaded.dtype))
        assert tensor.shape == loaded.shape, key
        for begin in range(0, tensor.shape[0], 1024):
            part = loaded[begin:begin+1024].cpu()
            assert torch.isfinite(part).all() and torch.equal(tensor[begin:begin+1024].to(part.dtype), part), key
        checks[key] = {"shape": list(tensor.shape), "saved_dtype": str(tensor.dtype), "loaded_dtype": str(loaded.dtype),
                       "finite_and_exact_after_explicit_dtype_conversion": True}
    tied = sharing(model)
    assert tied["same_parameter_object"] and tied["same_storage"]
    report["strict_adapter_reload"] = {"keys": 394, "lora_keys": 392, "unique_training_parameters": 393,
                                       "all_active_embedding_head_and_sid_rows_checked": True,
                                       "no_missing_keys_tolerated": True, "tensor_checks": checks, "sharing": tied}
    del saved, actual
    model.eval()
    model.config.use_cache = True
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "data-root", "model", "adapter", "output", "run-commit"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--aggregate-only", action="store_true", help="CPU recheck of already complete two-rank JSONL files")
    parser.add_argument("--preflight-users-per-rank", type=int, default=0,
                        help="Explicit interface preflight only; does not emit formal metrics or full-test PASS")
    parser.add_argument("--minimum-host-free-gib", type=float, default=40.)
    parser.add_argument("--minimum-disk-free-gib", type=float, default=20.)
    parser.add_argument("--max-seconds", type=int, default=96 * 3600,
                        help="Evaluation stage cap; the enclosing job enforces the cumulative 96-hour cap")
    args = parser.parse_args()
    if args.preflight_users_per_rank not in (0, 8) or (args.preflight_users_per_rank and args.aggregate_only):
        raise ValueError("Only a separate fixed eight-user/rank preflight is supported")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = author_source(args.source_root)
    if len(args.run_commit) != 40 or any(c not in "0123456789abcdef" for c in args.run_commit):
        raise ValueError("--run-commit must be a complete lowercase commit SHA")
    if args.aggregate_only:
        _, catalog, users, _ = catalog_and_users(args.data_root)
        result = aggregate_predictions([output / f"predictions.rank{rank}.jsonl" for rank in range(2)],
                                       users, catalog, load_evaluator(source), run_commit=args.run_commit)
        result["RUN_COMMIT"] = args.run_commit
        write(output / "metrics.recomputed.json", result)
        return 0
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    report_path, prediction_path = output / f"rank{rank}.json", output / f"predictions.rank{rank}.jsonl"
    if report_path.exists() or prediction_path.exists():
        raise FileExistsError("Evaluation attempt output already exists; preserve it and use a new attempt directory")
    started = time.monotonic()
    report = {"schema": "letter_a1_qwen3_formal_test_v1", "status": "RUNNING", "rank": rank,
              "world_size": world, "local_rank": local_rank, "RUN_COMMIT": args.run_commit,
              "host": socket.gethostname(), "pid": os.getpid(), "argv": sys.argv,
              "cwd": str(Path.cwd()), "interpreter": sys.executable,
              "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "completed_users": 0,
              "generation": {"prompt_id": 0, "num_beams": BEAMS, "num_return_sequences": BEAMS,
                             "max_new_tokens": 10, "early_stopping": True, "filter_items": False,
                             "prefix": "fixed original per-position callback; no catalog Trie",
                             "per_rank_batch_size": 1, "oom_beam_reduction": False,
                             "base_dtype": "BF16, unquantized original author LoRA inference branch",
                             "sampler": "original DistributedSampler shuffle=True, seed=0, epoch=0, drop_last=True"}}
    torch, dist = None, None

    def guard():
        if time.monotonic() - started > args.max_seconds:
            raise TimeoutError("Evaluation stage time budget exhausted")
        available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
                         if line.startswith("MemAvailable:"))
        free = shutil.disk_usage(output).free
        report["minimum_observed_host_memavailable_bytes"] = min(available, report.get("minimum_observed_host_memavailable_bytes", available))
        report["minimum_observed_output_free_bytes"] = min(free, report.get("minimum_observed_output_free_bytes", free))
        if available < args.minimum_host_free_gib * 1024**3:
            raise MemoryError("Host free memory reserve exhausted")
        if free < args.minimum_disk_free_gib * 1024**3:
            raise OSError("Evaluation output disk reserve exhausted")

    try:
        if world != 2 or not 0 <= rank < 2 or args.minimum_host_free_gib < 40 or args.minimum_disk_free_gib < 20:
            raise ValueError("Requires two ranks, at least 40GiB host and 20GiB output-disk reserve")
        if not 0 < args.max_seconds <= 96 * 3600:
            raise ValueError("Evaluation stage budget must be positive and no greater than 96 hours")
        write(report_path, report)
        guard()
        indices, catalog, users, identities = catalog_and_users(args.data_root)
        report["data_identity"] = identities
        report["author_sources"] = {name: {"path": str(source / name), "sha256": sha(source / name)}
                                    for name in ("data.py", "collator.py", "evaluate.py", "test_ddp.py", "prompt.py")}
        report["entry_sha256"] = sha(__file__)
        evaluator = load_evaluator(source)
        import torch
        import torch.distributed as dist
        from torch.utils.data import DataLoader, Dataset
        from torch.utils.data.distributed import DistributedSampler
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", timeout=dt.timedelta(minutes=30))
        sys.path.insert(0, str(source))
        from data import SeqRecDataset
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "qwen_preflight_20261110/scripts"))
        from common import Allowed
        from utils import set_seed
        set_seed(42)
        torch.cuda.reset_peak_memory_stats()
        report["versions"] = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "accelerate")}
        report["gpu"] = {"name": torch.cuda.get_device_name(local_rank),
                         "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                         "total_memory_bytes": torch.cuda.get_device_properties(local_rank).total_memory}
        load_started = time.monotonic()
        model, tokenizer = strict_reload(args, report, torch, local_rank, source, indices)
        torch.cuda.synchronize()
        report["model_load_and_strict_verification_seconds"] = time.monotonic() - load_started
        guard()
        data_args = SimpleNamespace(dataset="Instruments", data_path=str(Path(args.data_root).resolve()),
                                    max_his_len=20, his_sep=", ", index_file=".index.json", add_prefix=False)
        data = SeqRecDataset(data_args, mode="test", sample_num=-1)
        data.set_prompt(0)
        if len(data) != 24772 or list(data.remapped_inters) != [row["user_id"] for row in users]:
            raise AssertionError("Actual author test dataset user axis/size mismatch")
        if any(data[index]["labels"] != user["target"] for index, user in enumerate(users)):
            raise AssertionError("Actual author response differs from frozen held-out target")
        allowed = Allowed(tokenizer, indices)
        report["allowed_callback_sha256"] = hashlib.sha256(inspect.getsource(Allowed).encode()).hexdigest()
        tokenizer.padding_side = "left"

        class IndexedDataset(Dataset):
            def __len__(self):
                return len(data)

            def __getitem__(self, index):
                return index, data[index]

        def collate(batch):
            texts = [item[1]["input_ids"] for item in batch]
            inputs = tokenizer(texts, padding=True, truncation=False, return_tensors="pt", return_attention_mask=True)
            assert inputs["input_ids"].shape[1] <= 2048
            targets = [item[1]["labels"] for item in batch]
            return [item[0] for item in batch], inputs, targets

        indexed = IndexedDataset()
        sampler = DistributedSampler(indexed, num_replicas=world, rank=rank, shuffle=True, seed=0, drop_last=True)
        assignment = list(sampler)
        if len(data) % world or len(assignment) != 12386 or len(set(assignment)) != len(assignment):
            raise AssertionError("Original sampler would lose or duplicate test users")
        report["assigned_users"] = len(assignment)
        report["assignment_sha256"] = hashlib.sha256(json.dumps(assignment, separators=(",", ":")).encode()).hexdigest()
        if args.preflight_users_per_rank:
            # Fixed raw-text envelope strata, plus the maximum Qwen-token length.
            # This does not alter the formal sampler/user axis.
            lengths = sorted(range(len(data)), key=lambda i: (len(tokenizer(data[i]["input_ids"])["input_ids"]), i))
            positions = [0, len(data)//4, len(data)//2, len(data)-2]
            assignment = [lengths[min(len(data)-1, j + rank)] for j in positions]
            assignment += [lengths[-1-rank], lengths[len(data)*3//4+rank], lengths[len(data)//8+rank], lengths[len(data)*7//8+rank]]
            assert len(set(assignment)) == 8
            report["preflight_assignment"] = assignment
            report["full_test_max_input_tokens"] = len(tokenizer(data[lengths[-1]]["input_ids"])["input_ids"])
        loader = DataLoader(indexed, batch_size=1, sampler=assignment, collate_fn=collate,
                            num_workers=0, pin_memory=True)
        write(report_path, report)
        dist.barrier()
        generation_started = time.monotonic()
        with prediction_path.open("x") as stream, torch.no_grad():
            for iteration, (row_indices, inputs, targets) in enumerate(loader):
                if args.preflight_users_per_rank and iteration >= args.preflight_users_per_rank:
                    break
                guard()
                if row_indices != [assignment[iteration]]:
                    raise AssertionError("DataLoader order differs from original sampler assignment")
                index = row_indices[0]
                input_ids = inputs["input_ids"][0].tolist()
                allowed.bind(input_ids, data[index]["input_ids"])
                inputs = inputs.to(torch.device("cuda", local_rank))
                tick = time.monotonic()
                generated = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                                           max_new_tokens=10, prefix_allowed_tokens_fn=allowed, do_sample=False, use_cache=True,
                                           num_beams=BEAMS, num_return_sequences=BEAMS, output_scores=True,
                                           return_dict_in_generate=True, early_stopping=True,
                                           eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
                torch.cuda.synchronize()
                elapsed = time.monotonic() - tick
                sequences = generated.sequences.cpu().tolist()
                scores = generated.sequences_scores.float().cpu().tolist()
                if len(sequences) != BEAMS or any(not math.isfinite(score) for score in scores):
                    raise AssertionError("Generation did not return twenty finite scored beams")
                decoded = tokenizer.batch_decode(generated.sequences, skip_special_tokens=True)
                candidates = []
                for order, (sequence, score, text) in enumerate(zip(sequences, scores, decoded)):
                    if sequence[:len(input_ids)] != input_ids:
                        raise AssertionError("Generated sequence does not preserve original prompt")
                    suffix = sequence[len(input_ids):]
                    if tokenizer.eos_token_id not in suffix:
                        raise AssertionError("Original ten-token generation failed to reach EOS")
                    eos = suffix.index(tokenizer.eos_token_id)
                    assert eos == 4, "Exactly four SID tokens then real Base EOS required"
                    for position, token in enumerate(suffix[:eos + 1]):
                        permitted = allowed(0, torch.tensor(input_ids + suffix[:position], dtype=torch.long))
                        if permitted is None or token not in permitted:
                            raise AssertionError("Generated token violates original positional callback")
                    if any(token not in {tokenizer.eos_token_id, tokenizer.pad_token_id} for token in suffix[eos + 1:]):
                        raise AssertionError("Unexpected nonpadding tokens after generated EOS")
                    candidates.append({"generation_order": order, "generated_token_ids": suffix,
                                       "eos_position": eos, "raw_decoded": text,
                                       "sequence_score": score, "sid": normalized_prediction(text)})
                order = sorted(range(BEAMS), key=lambda i: scores[i], reverse=True)
                row = {**users[index], "rank": rank, "prompt_id": 0, "input_token_ids": input_ids,
                       "input_nonpadding_tokens": int(inputs["attention_mask"].sum()),
                       "candidates": candidates, "sorted_generation_order": order,
                       "relevance": [int(candidates[i]["sid"] == targets[0]) for i in order],
                       "generation_seconds": elapsed, "original_callback_verified_all_tokens": True}
                audit_prediction(row, users[index], catalog, evaluator)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                report["completed_users"] += 1
                del generated, inputs
                if (iteration + 1) % 25 == 0:
                    os.fsync(stream.fileno())
                    report["elapsed_generation_seconds"] = time.monotonic() - generation_started
                    report["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                    report["gpu_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
                    write(report_path, report)
                    print(json.dumps({"stage": "test", "rank": rank, "completed_users": report["completed_users"],
                                      "assigned_users": len(assignment), "elapsed_seconds": report["elapsed_generation_seconds"]}), flush=True)
            os.fsync(stream.fileno())
        if report["completed_users"] != (args.preflight_users_per_rank or len(assignment)):
            raise AssertionError("Author test shard did not complete")
        report["elapsed_generation_seconds"] = time.monotonic() - generation_started
        report["status"] = "PREDICTIONS_COMPLETE"
        write(report_path, report)
        dist.barrier()
        if rank == 0 and not args.preflight_users_per_rank:
            result = aggregate_predictions([output / f"predictions.rank{i}.jsonl" for i in range(world)],
                                           users, catalog, evaluator, run_commit=args.run_commit)
            result.update(RUN_COMMIT=args.run_commit, adapter_path=str(Path(args.adapter).resolve()),
                          adapter_model_sha256=report["adapter_files"]["adapter_model.bin"]["sha256"],
                          data_identity=identities, generation=report["generation"],
                          original_evaluator_sha256=report["author_sources"]["evaluate.py"]["sha256"])
            write(output / "metrics.json", result)
            print(json.dumps({"stage": "test_complete", "metrics": result["metrics"],
                              "users": result["actual_test_users"], "anomalies": result["anomalies"]}), flush=True)
        dist.barrier()
        report["status"] = "PREFLIGHT_INTERFACE_PASS" if args.preflight_users_per_rank else "PASS"
        report["preflight_only"] = bool(args.preflight_users_per_rank)
    except BaseException as error:
        report["status"] = "FAIL"
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        print(report["error"]["traceback"], file=sys.stderr, flush=True)
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        report["cpu_process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        if torch is not None and torch.cuda.is_initialized():
            report["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["gpu_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        write(report_path, report)
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()
    return 0 if report["status"] in ("PASS", "PREFLIGHT_INTERFACE_PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
