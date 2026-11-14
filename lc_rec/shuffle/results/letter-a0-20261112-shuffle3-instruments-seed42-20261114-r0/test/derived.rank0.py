#!/usr/bin/env python3
"""Full A0 Instruments prompt0/beam20 test, with auditable rank predictions.

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
MODEL_REVISION = "4782ad278652c7c71b72204d462d6d01eaaf7549"


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


def expected_adapter_shapes(vocab_size=32918):
    shapes = {}
    for layer in range(32):
        for family, modules in [("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
                                ("mlp", ("gate_proj", "down_proj", "up_proj"))]:
            for module in modules:
                input_size = 11008 if module == "down_proj" else 4096
                output_size = 11008 if module in {"gate_proj", "up_proj"} else 4096
                prefix = f"base_model.model.model.layers.{layer}.{family}.{module}."
                shapes[prefix + "lora_A.weight"] = (8, input_size)
                shapes[prefix + "lora_B.weight"] = (output_size, 8)
    shapes["base_model.model.model.embed_tokens.weight"] = (vocab_size, 4096)
    shapes["base_model.model.lm_head.weight"] = (vocab_size, 4096)
    return shapes


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
                                                    "author_sources", "generation", "allowed_callback_sha256")}
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
    from transformers import LlamaForCausalLM, LlamaTokenizer
    from peft import PeftModel, get_peft_model_state_dict
    if importlib.metadata.version("peft") != "0.5.0":
        raise RuntimeError("Strict loading is pinned to the already audited PEFT 0.5.0")
    adapter = Path(args.adapter).resolve()
    required = ("adapter_config.json", "tokenizer_config.json", "tokenizer.model", "added_tokens.json")
    if any(not (adapter / name).is_file() for name in required):
        raise AssertionError("Best adapter lacks its saved tokenizer/configuration")
    tensor_path = adapter / "adapter_model.bin"
    if not tensor_path.is_file() or (adapter / "adapter_model.safetensors").exists():
        raise AssertionError("Pinned adapter must contain one unambiguous adapter_model.bin")
    config = json.loads((adapter / "adapter_config.json").read_text())
    if (config["r"], config["lora_alpha"], config["lora_dropout"]) != (8, 16, .05):
        raise AssertionError("Best adapter LoRA hyperparameters changed")
    if set(config["target_modules"]) != {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"}:
        raise AssertionError("Best adapter LoRA module targets changed")
    if set(config["modules_to_save"]) != {"embed_tokens", "lm_head"} or config.get("bias", "none") != "none":
        raise AssertionError("Best adapter active embedding/head or bias policy changed")
    tokenizer = LlamaTokenizer.from_pretrained(adapter, local_files_only=True)
    tokenizer.pad_token_id, tokenizer.padding_side, tokenizer.model_max_length = 0, "left", 2048
    reference = LlamaTokenizer.from_pretrained(args.model, local_files_only=True)
    reference.pad_token_id = 0
    reference.add_tokens(sorted({token for tokens in indices.values() for token in tokens}))
    if len(tokenizer) != 32918 or len(tokenizer.get_added_vocab()) != 918 or tokenizer.get_vocab() != reference.get_vocab():
        raise AssertionError("Best adapter tokenizer differs from base plus sorted original 918 SID tokens")
    if tokenizer.get_added_vocab() != reference.get_added_vocab():
        raise AssertionError("Best adapter SID added-token ID ordering changed")
    report["tokenizer"] = {"vocab_size": len(tokenizer), "added_tokens": tokenizer.get_added_vocab(),
                           "matches_reconstructed_author_tokenizer": True, "padding_side": "left", "max_length": 2048}
    report["adapter_files"] = {path.name: {"bytes": path.stat().st_size, "sha256": sha(path)}
                               for path in adapter.iterdir() if path.is_file()}
    model_root = Path(args.model).resolve()
    if model_root.name != MODEL_REVISION:
        raise AssertionError("Model directory must identify the frozen officially verified revision")
    report["model"] = {"name": "huggyllama/llama-7b", "revision": MODEL_REVISION,
                       "path": str(model_root), "config_sha256": sha(model_root / "config.json"),
                       "weight_identity": "Reuse previously verified fixed official-LFS asset; full weight hashes not recomputed"}
    model_config = json.loads((model_root / "config.json").read_text())
    if (model_config["hidden_size"], model_config["intermediate_size"], model_config["num_hidden_layers"],
        model_config["num_attention_heads"], model_config["vocab_size"]) != (4096, 11008, 32, 32, 32000):
        raise AssertionError("Fixed LLaMA-7B model architecture differs")
    base = LlamaForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                          device_map={"": local_rank}, local_files_only=True)
    base.resize_token_embeddings(len(tokenizer))
    reads = []

    def audit_open(event, arguments):
        if event == "open" and isinstance(arguments[0], (str, bytes, os.PathLike)):
            if os.fsdecode(arguments[0]) == str(tensor_path):
                reads.append({"path": str(tensor_path), "mode": str(arguments[1])})

    sys.addaudithook(audit_open)
    model = PeftModel.from_pretrained(base, str(adapter), torch_dtype=torch.bfloat16,
                                     device_map={"": local_rank}, is_trainable=False)
    if not reads:
        raise AssertionError("PEFT did not read the expected local adapter payload")
    report["peft_actual_weight_reads"] = list(reads)
    saved = torch.load(tensor_path, map_location="cpu", weights_only=True)
    actual = get_peft_model_state_dict(model)
    expected = expected_adapter_shapes(len(tokenizer))
    if set(saved) != set(actual) or set(saved) != set(expected):
        raise AssertionError("Saved/loaded/expected full 450-key adapter sets differ")
    checks = {}
    for key, tensor in saved.items():
        loaded = actual[key].detach()
        if tuple(tensor.shape) != expected[key] or tensor.shape != loaded.shape:
            raise AssertionError("Strict adapter shape mismatch: " + key)
        for begin in range(0, tensor.shape[0], 1024):
            part = loaded[begin:begin + 1024].cpu()
            if not torch.isfinite(part).all() or not torch.equal(tensor[begin:begin + 1024].to(part.dtype), part):
                raise AssertionError("Strict adapter saved/reloaded value mismatch: " + key)
        checks[key] = {"shape": list(tensor.shape), "saved_dtype": str(tensor.dtype),
                       "loaded_dtype": str(loaded.dtype), "finite_and_exact_after_explicit_dtype_conversion": True}
    elements = sum(value.numel() for value in saved.values())
    if len(saved) != 450 or elements != 289652736:
        raise AssertionError("Unexpected adapter tensor/element count")
    report["strict_adapter_reload"] = {"keys": 450, "lora_keys": 448, "elements": elements,
                                       "all_active_embedding_head_and_sid_rows_checked": True,
                                       "no_missing_keys_tolerated": True, "tensor_checks": checks}
    del saved, actual
    model.eval()
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
    if args.preflight_users_per_rank not in (0, 16) or (args.preflight_users_per_rank and args.aggregate_only):
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
    report = {"schema": "letter_a0_formal_test_v1", "status": "RUNNING", "rank": rank,
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
        from collator import TestCollator
        from utils import set_seed
        set_seed(42)
        torch.cuda.reset_peak_memory_stats()
        report["versions"] = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "accelerate")}
        report["gpu"] = {"name": torch.cuda.get_device_name(local_rank),
                         "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                         "total_memory_bytes": torch.cuda.get_device_properties(local_rank).total_memory}
        load_started = time.monotonic()
        model, tokenizer = strict_reload(args, report, torch, local_rank, source, indices)
        runtime.loaded(model, tokenizer, args, report)
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
        allowed = data.get_prefix_allowed_tokens_fn(tokenizer)
        report["allowed_callback_sha256"] = hashlib.sha256(inspect.getsource(data.get_prefix_allowed_tokens_fn).encode()).hexdigest()
        original_collator = TestCollator(data_args, tokenizer)

        class IndexedDataset(Dataset):
            def __len__(self):
                return len(data)

            def __getitem__(self, index):
                return index, data[index]

        def collate(batch):
            inputs, targets = original_collator([item[1] for item in batch])
            return [item[0] for item in batch], inputs, targets

        indexed = IndexedDataset()
        sampler = DistributedSampler(indexed, num_replicas=world, rank=rank, shuffle=True, seed=0, drop_last=True)
        assignment = list(sampler)
        if len(data) % world or len(assignment) != 12386 or len(set(assignment)) != len(assignment):
            raise AssertionError("Original sampler would lose or duplicate test users")
        report["assigned_users"] = len(assignment)
        report["assignment_sha256"] = hashlib.sha256(json.dumps(assignment, separators=(",", ":")).encode()).hexdigest()
        if args.preflight_users_per_rank:
            assignment = runtime.preflight_assignment(rank)
            report["preflight_assignment"] = assignment
        loader = DataLoader(indexed, batch_size=1, sampler=assignment, collate_fn=collate,
                            num_workers=2, pin_memory=True)
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
                inputs, shuffle_record = runtime.intervene(inputs, index, data, tokenizer)
                input_ids = inputs["input_ids"][0].tolist()
                inputs = inputs.to(torch.device("cuda", local_rank))
                tick = time.monotonic()
                generated = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                                           max_new_tokens=10, prefix_allowed_tokens_fn=allowed,
                                           num_beams=BEAMS, num_return_sequences=BEAMS, output_scores=True,
                                           return_dict_in_generate=True, early_stopping=True)
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
                row = {**users[index], "rank": rank, "prompt_id": 0, "input_token_ids": input_ids, "shuffle": shuffle_record,
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
