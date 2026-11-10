#!/usr/bin/env python3
"""Bounded LC-Rec SIX-TASK LoRA engineering probe, never a formal seed.

The original Dataset, Collator, HF Llama loss, SID order and PEFT recipe are
used. Pre-engine diagnostics have no optimizer and no gradient communication;
only the separately reported ZeRO-2 global128 updates measure training. Every
optimizer boundary, including zero-LR warmup, reserves the shared six-update
ledger *before* engine.step. A failed reservation is not refunded.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import gc
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import random
import re
import resource
import shutil
import socket
import sys
import time
import traceback


TASKS = ["seqrec", "item2index", "index2item", "fusionseqrec", "itemsearch", "preferenceobtain"]
COUNTS = [131837, 9922, 9922, 20000, 24772, 24772]
TARGETS = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
GIB = 1024 ** 3


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b""):
            digest.update(block)
    return digest.hexdigest()


def record_file(path):
    path = Path(path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha_file(path)}


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": repr(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def host_resources():
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, rest = line.split(":", 1)
        if key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
            mem[key + "_bytes"] = int(rest.strip().split()[0]) * 1024
    status = Path("/proc/self/status").read_text().splitlines()
    for line in status:
        key, _, rest = line.partition(":")
        if key in ("VmRSS", "VmHWM", "VmSwap"):
            mem[key + "_bytes"] = int(rest.strip().split()[0]) * 1024
    mem["process_peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
    return mem


def tuple_tree(value):
    return tuple(tuple_tree(v) for v in value) if isinstance(value, list) else value


def reserve_update(path, job_id, run_commit, update, maximum=6):
    """Rank zero only. Count attempted real boundaries conservatively."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name + ".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(path.read_text()) if path.exists() else {"used": 0, "limit": maximum, "reservations": []}
        used = int(state["used"])
        if used >= maximum or int(state.get("limit", maximum)) != maximum:
            raise RuntimeError("Task-wide six-global-update budget exhausted or inconsistent")
        reservation = {"number": used + 1, "job_id": job_id, "run_commit": run_commit,
                       "local_update": update, "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                       "meaning": "Reserved before optimizer boundary; failures and zero LR count"}
        state.update(used=used + 1, limit=maximum)
        state.setdefault("reservations", []).append(reservation)
        write_json(path, state)
        return reservation


def parse_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-commit", default=os.environ.get("RUN_COMMIT", ""))
    parser.add_argument("--job-id", default=os.environ.get("GRID_JOB_ID", os.environ.get("JOB_ID", "unlabelled")))
    parser.add_argument("--candidate", choices=("bf16", "int8"), default="bf16")
    parser.add_argument("--max-updates", type=int, default=2)
    parser.add_argument("--micro-batch", type=int, choices=(8, 4, 2, 1), default=8)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-f0-9]{40}", args.run_commit):
        parser.error("--run-commit must be the complete immutable 40-character RUN_COMMIT")
    if not 0 <= args.max_updates <= 6:
        parser.error("--max-updates must be within the shared task-wide six-update budget")
    for name in ("config", "samples", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    return args


def run(args, report, flush, event):
    import numpy as np
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    import transformers
    import deepspeed
    import peft
    from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict, PeftModel
    from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments

    cfg = json.loads(args.config.read_text())
    sample_doc = json.loads(args.samples.read_text())
    rank = int(os.environ.get("RANK", 0))
    local = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world != 2:
        raise RuntimeError("Exactly two GPU ranks are required")
    torch.cuda.set_device(local)
    if not dist.is_initialized():
        dist.init_process_group("nccl", timeout=dt.timedelta(seconds=180))
    device = torch.device("cuda", local)
    job_start = time.monotonic()
    output = args.output_dir
    seed = int(cfg.get("seed", 42))
    if seed != 42:
        raise RuntimeError("This task authorizes seed42 only")
    source = Path(cfg["vendor_dir"]).resolve()
    base_model = str(Path(cfg["base_model"]).resolve())
    limit_updates = min(args.max_updates, int(cfg.get("max_updates", 2)))
    fixed_settings = {"seed": 42, "model_max_length": 2048, "max_his_len": 20,
        "learning_rate": 5e-5, "weight_decay": .01, "lora_r": 8, "lora_alpha": 32,
        "lora_dropout": .05, "global_batch_size": 128, "test_num_beams": 20, "test_prompt_ids": "all"}
    for key, value in fixed_settings.items():
        if key in cfg and cfg[key] != value:
            raise AssertionError("Effective technical recipe conflicts with immutable candidate config: " + key)
    report.update(candidate=args.candidate, run_commit=args.run_commit, job_id=args.job_id,
                  rank=rank, world_size=world, local_rank=local,
                  config=cfg, config_identity=record_file(args.config), samples_identity=record_file(args.samples))
    report["environment"] = {"interpreter": sys.executable, "python": sys.version,
        "hostname": socket.gethostname(), "versions": {name: importlib.metadata.version(name) for name in
            ("torch", "transformers", "peft", "deepspeed", "accelerate")},
        "import_paths": {"torch": torch.__file__, "transformers": transformers.__file__,
                         "peft": peft.__file__, "deepspeed": deepspeed.__file__},
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(local),
        "gpu_total_bytes": torch.cuda.get_device_properties(local).total_memory,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    report["engineering_differences"] = [
        "Official lora_finetune.py recipe, Dataset and Collator; finite engineering harness replaces Trainer.train",
        "Explicit complete base BF16 loading (author entry omits torch_dtype); BF16 initialization/rounding differs from FP32-load plus autocast",
        "Two GPUs and ZeRO-2 without any offload; author example run.sh uses full-parameter finetune.py and ZeRO-3 on eight GPUs",
        "No torch.compile and no FastChat/LETTER model, prompt, temperature loss or runtime patch",
        "Pre-engine per-task F/B diagnostics have no optimizer and no gradient communication",
        "Fixed replayed training-period technical samples for updates; source auxiliary assets remain blocked for trustworthy formal reproduction",
        "Microbatch mean response-token CE followed by equal microbatch accumulation; global128 alone does not preserve token-weighting across layouts",
        "No complete epoch, formal seed, full validation/test scan, or training-state checkpoint"]
    report["precision"] = {"base": args.candidate, "bf16_training": True,
                           "global_autocast": False, "loss": "Unmodified HF LlamaForCausalLM shifted CrossEntropyLoss; no temperature"}
    sys.path.insert(0, str(source))
    import utils as original_utils
    import data as original_data
    import collator as original_collator
    for module, file in ((original_utils, "utils.py"), (original_data, "data.py"), (original_collator, "collator.py")):
        if Path(module.__file__).resolve() != source / file:
            raise RuntimeError("Official LC-Rec import shadowed: " + file)
    report["official_sources"] = [record_file(source / file) for file in
        ("lora_finetune.py", "data.py", "collator.py", "utils.py", "prompt.py", "test.py", "test_ddp.py", "evaluate.py", "run.sh")]
    original_utils.set_seed(seed)
    dataset_args = dict(sample_doc["args_namespace"])
    dataset_args["data_path"] = cfg["data_path"]
    dataset_args["base_model"] = base_model
    dataset_args["model_max_length"] = 2048
    dataset_args["only_train_response"] = True
    data_args = argparse.Namespace(**dataset_args)
    report["effective_dataset_args"] = dataset_args
    if data_args.tasks.split(",") != TASKS or data_args.max_his_len != 20:
        raise RuntimeError("Six-task/max-history identity differs from authorization")
    if data_args.train_prompt_sample_num != "1,1,1,1,1,1" or data_args.train_data_sample_num != "0,0,0,20000,0,0":
        raise RuntimeError("Author Instruments sampling must remain intact")
    t0 = time.monotonic()
    train_data, valid_data = original_utils.load_datasets(data_args)
    tokenizer = LlamaTokenizer.from_pretrained(base_model, model_max_length=2048, padding_side="right", local_files_only=True)
    tokenizer.pad_token_id = 0
    original_vocab = len(tokenizer)
    sid_tokens = train_data.datasets[0].get_new_tokens()
    added = tokenizer.add_tokens(sid_tokens)
    if added != 930 or len(train_data) != 221225 or [len(d) for d in train_data.datasets] != COUNTS:
        raise AssertionError("Original SID vocabulary or complete six-task Dataset identity mismatch")
    sid_ids = tokenizer.convert_tokens_to_ids(sid_tokens)
    if sid_ids != list(range(original_vocab, original_vocab + 930)):
        raise AssertionError("SID sorted-token order is not an exact appended mapping")
    collator = original_collator.Collator(data_args, tokenizer)
    report["data"] = {"tasks": TASKS, "counts": COUNTS, "total": len(train_data),
        "classes": [type(d).__module__ + "." + type(d).__name__ for d in train_data.datasets],
        "validation_rows": len(valid_data), "validation_prompt_sample_num": data_args.valid_prompt_sample_num,
        "validation_sample_valid": data_args.sample_valid, "dataset_tokenizer_seconds": time.monotonic() - t0,
        "original_vocab": original_vocab, "new_tokens": added, "tokenizer_length": len(tokenizer),
        "sid_tokens": sid_tokens, "sid_ids": sid_ids, "bos": tokenizer.bos_token_id,
        "eos": tokenizer.eos_token_id, "pad": tokenizer.pad_token_id}
    records = sample_doc["records"]
    plans = sample_doc["plans"]
    replayed = {}

    def get_record(sample_id, require_eligible=False):
        record = records[sample_id]
        if require_eligible and not record.get("technical_update_eligible", False):
            raise AssertionError("Diagnostic-only / unverified auxiliary sample cannot enter optimizer update: " + sample_id)
        if sample_id not in replayed:
            state = record.get("python_random_state_before_getitem")
            if state is None:
                raise AssertionError("Exact original Dataset replay RNG evidence is mandatory: " + sample_id)
            old = random.getstate()
            try:
                random.setstate(tuple_tree(state))
                task_index = TASKS.index(record["task"])
                result = train_data.datasets[task_index][int(record["dataset_local_index"])]
            finally:
                random.setstate(old)
            if result != {"input_ids": record["input_ids"], "labels": record["labels"]}:
                raise AssertionError("Original Dataset byte replay failed: " + sample_id)
            encoded = collator([result])
            expected = record["collated"]
            for key in ("input_ids", "labels", "attention_mask"):
                actual = encoded[key].tolist()
                want = expected[key]
                if want and not isinstance(want[0], list):
                    want = [want]
                if actual != want:
                    raise AssertionError("Official Collator token/mask differs from CPU sample audit: " + sample_id + " " + key)
            replayed[sample_id] = result
        return replayed[sample_id]

    def cpu_batch(ids, require_eligible=False):
        batch = collator([get_record(sample_id, require_eligible) for sample_id in ids])
        counts = {"samples": len(ids), "tokens": int(batch["attention_mask"].sum()),
                  "padded_tokens": batch["input_ids"].numel(), "label_tokens": int((batch["labels"] != -100).sum()),
                  "shift_label_tokens": int((batch["labels"][:, 1:] != -100).sum()),
                  "shape": list(batch["input_ids"].shape), "sample_ids": ids,
                  "tasks": [records[sample_id]["task"] for sample_id in ids]}
        if counts["shift_label_tokens"] == 0:
            raise ValueError("Selected batch has no actual shifted supervision")
        return batch, counts

    def guard():
        resources = host_resources()
        if resources["MemAvailable_bytes"] < 40 * GIB:
            raise RuntimeError("Host memory floor 40GiB violated")
        disk = shutil.disk_usage(output)
        if disk.free < max(30 * GIB, .10 * disk.total):
            raise RuntimeError("Filesystem free-space floor violated")
        if time.monotonic() - job_start > min(float(cfg.get("job_max_seconds", 2550)), 2700):
            raise TimeoutError("Per-job 45-minute hard limit approached")
        return resources

    def max_rank_seconds(seconds):
        value = torch.tensor(seconds, device=device, dtype=torch.float64)
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
        return value.item()

    def global_counts(counts):
        values = torch.tensor([counts[k] for k in ("samples", "tokens", "padded_tokens", "label_tokens", "shift_label_tokens")],
                              device=device, dtype=torch.long)
        dist.all_reduce(values)
        return {k: int(v) for k, v in zip(("samples", "tokens", "padded_tokens", "label_tokens", "shift_label_tokens"), values.tolist())}

    memory_peaks = {"allocated_gib": 0., "reserved_gib": 0.}

    def memory():
        result = {"allocated_gib": torch.cuda.memory_allocated(local) / GIB,
                "reserved_gib": torch.cuda.memory_reserved(local) / GIB,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(local) / GIB,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(local) / GIB,
                "free_gib": torch.cuda.mem_get_info(local)[0] / GIB}
        memory_peaks["allocated_gib"] = max(memory_peaks["allocated_gib"], result["peak_allocated_gib"])
        memory_peaks["reserved_gib"] = max(memory_peaks["reserved_gib"], result["peak_reserved_gib"])
        report["gpu_all_stage_peaks"] = dict(memory_peaks)
        return result

    def load_base():
        options = {"torch_dtype": torch.bfloat16, "low_cpu_mem_usage": True,
                   "device_map": {"": local}, "local_files_only": True}
        if args.candidate == "int8":
            options["load_in_8bit"] = True
        value = LlamaForCausalLM.from_pretrained(base_model, **options)
        value.resize_token_embeddings(len(tokenizer))
        return value

    guard()
    event("before_base_load")
    start = time.monotonic()
    model = load_base()
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=32, lora_dropout=.05, target_modules=TARGETS,
        modules_to_save=["embed_tokens", "lm_head"], bias="none", task_type=TaskType.CAUSAL_LM, inference_mode=False))
    for name, parameter in model.named_parameters():
        if "original_module" in name and any(module in name for module in ("embed_tokens", "lm_head")):
            parameter.requires_grad = False
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()
    torch.cuda.synchronize()
    report["model_initialization_seconds"] = max_rank_seconds(time.monotonic() - start)
    report["model_initialization_memory"] = memory()
    report["model_class"] = type(model.base_model.model).__module__ + "." + type(model.base_model.model).__name__
    report["standard_model_source"] = record_file(inspect.getfile(LlamaForCausalLM))
    named = dict(model.named_parameters())
    train_names = sorted(name for name, parameter in named.items() if parameter.requires_grad)
    lora_names = [name for name in train_names if "lora_" in name]
    active_names = [name for name in train_names if "modules_to_save.default" in name]
    frozen_names = [name for name in named if "original_module" in name]
    for name, parameter in named.items():
        if parameter.requires_grad != ("lora_" in name or "modules_to_save.default" in name):
            raise AssertionError("Unexpected requires_grad: " + name)
    numel = sum(named[name].numel() for name in train_names)
    if (len(train_names), len(lora_names), len(active_names), numel) != (450, 448, 2, 289751040):
        raise AssertionError("Official r8 + complete active embedding/head trainable inventory differs")
    embed_name = next(name for name in active_names if "embed_tokens" in name)
    head_name = next(name for name in active_names if "lm_head" in name)
    if named[embed_name].data_ptr() == named[head_name].data_ptr():
        raise AssertionError("LLaMA input embedding/head unexpectedly share storage")
    report["trainable_identity"] = {"tensors": len(train_names), "lora_tensors": len(lora_names),
        "active_tensors": len(active_names), "elements": numel, "lora_elements": sum(named[n].numel() for n in lora_names),
        "active_elements": sum(named[n].numel() for n in active_names), "nonshared_embed_head": True,
        "parameters": {name: {"shape": list(named[name].shape), "dtype": str(named[name].dtype)} for name in train_names},
        "frozen_originals": frozen_names}

    def tensor_digest(value):
        digest = hashlib.sha256()
        flat = value.detach().reshape(-1)
        for offset in range(0, flat.numel(), 2 ** 20):
            part = flat[offset:offset + 2 ** 20].contiguous().cpu()
            digest.update(part.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    sid_tensor = torch.tensor(sid_ids, device=device, dtype=torch.long)

    def small_snapshot(name):
        p = named[name].detach()
        return (p.index_select(0, sid_tensor) if name in active_names else p).cpu().clone()

    before = {name: small_snapshot(name) for name in train_names}
    frozen_before = {name: tensor_digest(named[name]) for name in frozen_names}
    report["snapshot_scope"] = {"lora": "Every value of all 448 tensors", "active": "Every column of every new SID row",
                                "frozen_originals": "SHA-256 of all bytes of both complete original embedding/head matrices"}
    gradient_scalars = {}
    gradient_sid_scalars = {}
    gradient_handles = []

    def gradient_hook(name):
        def hook(gradient):
            # Reduction accumulates in FP32 without copying whole gradients to CPU.
            maximum = torch.linalg.vector_norm(gradient.detach(), ord=float("inf"), dtype=torch.float32)
            gradient_scalars.setdefault(name, []).append(maximum)
            if name in active_names:
                sid_maximum = torch.linalg.vector_norm(gradient.detach().index_select(0, sid_tensor),
                                                      ord=float("inf"), dtype=torch.float32)
                gradient_sid_scalars.setdefault(name, []).append(sid_maximum)
            return gradient
        return hook

    for name in train_names:
        gradient_handles.append(named[name].register_hook(gradient_hook(name)))

    def summarize_gradients(clear=True):
        result = {}
        for name, values in gradient_scalars.items():
            combined = torch.stack(values)
            result[name] = {"calls": len(values), "finite": bool(torch.isfinite(combined).all().item()),
                            "max_abs": float(combined.max().item())}
            if name in gradient_sid_scalars:
                sid_combined = torch.stack(gradient_sid_scalars[name])
                result[name].update(sid_rows_max_abs=float(sid_combined.max().item()),
                                    sid_rows_finite=bool(torch.isfinite(sid_combined).all().item()))
        if clear:
            gradient_scalars.clear()
            gradient_sid_scalars.clear()
        return result

    report["diagnostics"] = []
    report["layout_probes"] = []
    report["timing_method"] = {"clock": "CUDA-synchronized phase wall time; MAX rank duration, SUM actual rank samples/tokens",
        "diagnostics": "Original model F/B without optimizer, accumulation or ZeRO communication; two ranks each run their assigned real samples",
        "mixed": "Complete ZeRO-2 accumulation/update including communication; two ranks are simultaneous, durations are never added",
        "instrumentation": "Synchronized phases and full-trainable gradient finite/max hooks add overhead; JSON and audit hashes are outside timed phases"}
    ce_checked = False

    def plain_trial(ids, label):
        nonlocal ce_checked
        guard()
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.reset_peak_memory_stats(local)
        row = {"label": label, "status": "RUNNING", "optimizer_steps": 0, "communication": "No gradient communication; synchronized timing only"}
        batch = result = None
        started = time.monotonic()
        try:
            t = time.monotonic()
            batch, counts = cpu_batch(ids)
            row["cpu_collator_seconds"] = time.monotonic() - t
            row["rank_counts"] = counts
            torch.cuda.synchronize()
            t = time.monotonic()
            batch = {k: value.to(device) for k, value in batch.items()}
            torch.cuda.synchronize()
            row["transfer_seconds"] = time.monotonic() - t
            t = time.monotonic()
            result = model(**batch)
            torch.cuda.synchronize()
            row["forward_seconds"] = time.monotonic() - t
            if not torch.isfinite(result.loss).all().item():
                raise FloatingPointError("Nonfinite original HF CE")
            row["loss"] = float(result.loss.detach().item())
            if not ce_checked:
                with torch.no_grad():
                    actual_ce = F.cross_entropy(result.logits[:, :-1, :].reshape(-1, len(tokenizer)), batch["labels"][:, 1:].reshape(-1))
                    report["loss_verification"] = {"model": row["loss"], "independent_temperature1_ce": float(actual_ce),
                                                   "logits_dtype": str(result.logits.dtype),
                                                   "equal": bool(torch.allclose(result.loss, actual_ce, atol=1e-6, rtol=1e-6))}
                    if not report["loss_verification"]["equal"]:
                        raise AssertionError("Unmodified HF standard CE agreement failed")
                ce_checked = True
            t = time.monotonic()
            result.loss.backward()
            torch.cuda.synchronize()
            row["backward_seconds"] = time.monotonic() - t
            grads = summarize_gradients()
            row["gradient_summary"] = {"tensors": len(grads), "finite": all(v["finite"] for v in grads.values()),
                "nonzero_lora_B": sum(v["max_abs"] > 0 for n, v in grads.items() if "lora_B" in n),
                "active": {n: grads.get(n) for n in active_names}}
            if len(grads) != 450 or not row["gradient_summary"]["finite"]:
                raise FloatingPointError("Missing or nonfinite trainable gradients")
            row["status"] = "PASS"
        except Exception as error:
            row.update(status="OOM" if isinstance(error, torch.cuda.OutOfMemoryError) else "FAIL",
                       error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        finally:
            row["rank_seconds"] = time.monotonic() - started
            row["memory"] = memory()
            del result, batch
            model.zero_grad(set_to_none=True)
            gradient_scalars.clear()
            gradient_sid_scalars.clear()
            gc.collect()
            torch.cuda.empty_cache()
        statuses = [None] * world
        dist.all_gather_object(statuses, {"status": row["status"], "error": row.get("error"), "memory": row["memory"]})
        row["all_rank_statuses"] = statuses
        row["max_rank_seconds"] = max_rank_seconds(row["rank_seconds"])
        if all(value["status"] == "PASS" for value in statuses):
            row["global_counts"] = global_counts(row["rank_counts"])
            ids_all = [None] * world
            dist.all_gather_object(ids_all, ids)
            row["distinct_records_across_ranks"] = len(set(sample_id for group in ids_all for sample_id in group))
            row["sample_count_interpretation"] = "Actual GPU sample presentations; task representatives repeat the same single original record on both ranks"
            row["samples_per_second_compute_only"] = row["global_counts"]["samples"] / row["max_rank_seconds"]
            row["effective_tokens_per_second_compute_only"] = row["global_counts"]["tokens"] / row["max_rank_seconds"]
        else:
            row["status"] = "OOM" if any(value["status"] == "OOM" for value in statuses) else "FAIL"
        return row

    event("before_task_diagnostics")
    diag_start = time.monotonic()
    diag_limit = float(cfg.get("diagnostic_time_limit_seconds", 600))
    for task in TASKS:
        representatives = plans["task_representatives"][task]
        for position, sample_id in enumerate(representatives):
            row = plain_trial([sample_id], task + ":" + ("short", "middle", "long", "extreme")[min(position, 3)])
            row.update(task=task, sample_ids=[sample_id], diagnostic_only=records[sample_id].get("diagnostic_only", True))
            report["diagnostics"].append(row)
            event("task_diagnostic", task=task, position=position, status=row["status"])
            if row["status"] != "PASS":
                # Preserve unrelated task diagnostics and inference work; no unreviewed numerical patch.
                report.setdefault("diagnostic_failures", []).append(task + ":" + sample_id)
            if time.monotonic() - diag_start > diag_limit:
                report["diagnostic_deadline_reached"] = True
                break
        if report.get("diagnostic_deadline_reached"):
            break
    # Extra full-load layout probes are deliberately separate from task timings.
    # Include a padding stress batch but do not alter formal sampler semantics.
    mixed = plans["mixed_global128_0"]
    if len(mixed) != 128:
        raise AssertionError("Mixed update plan must contain exactly128 records")
    selected_micro = None
    for micro in [m for m in (8, 4, 2, 1) if m <= args.micro_batch]:
        # Probe the largest actual padded batch of the two fixed mixed plans,
        # rather than whichever unusually short rows happen to occur first.
        possible = []
        for plan_name in ("mixed_global128_0", "mixed_global128_1"):
            plan = plans[plan_name]
            for begin in range(rank * micro, len(plan), world * micro):
                batch_ids = plan[begin:begin + micro]
                width = max(records[i]["metrics"]["full_tokens"] for i in batch_ids)
                possible.append((width, begin, plan_name, batch_ids))
        _, _, capacity_plan, ids = max(possible, key=lambda value: (value[0], -value[1], value[2]))
        row = plain_trial(ids, "capacity_micro" + str(micro))
        row["selection"] = "Largest real padded width among this rank's batches in both fixed mixed128 plans; deterministic tie break"
        row["capacity_plan"] = capacity_plan
        row["micro_batch"] = micro
        row["accumulation"] = 128 // (world * micro)
        row["is_actual_zero2_layout_test"] = False
        # Reserve FP32 masters/Adam states, partitioned grads, and communications.
        reserve_gib = float(cfg.get("pre_engine_free_reserve_gib", 3.0))
        peaks = [value["memory"]["peak_allocated_gib"] for value in row["all_rank_statuses"]]
        total = torch.cuda.get_device_properties(local).total_memory / GIB
        row["optimizer_and_communication_reserve_gib"] = reserve_gib
        row["passes_reserve_gate"] = row["status"] == "PASS" and max(peaks) + reserve_gib < total
        report["layout_probes"].append(row)
        event("layout_probe", micro=micro, status=row["status"], reserve_gate=row["passes_reserve_gate"])
        if row["passes_reserve_gate"]:
            selected_micro = micro
            break
    if selected_micro is None:
        report["training_status"] = "BLOCKED_NO_BF16_LAYOUT_WITH_RESERVE"
        raise RuntimeError("No permitted microbatch fits the measured pre-engine reserve gate")
    accumulation = 128 // (world * selected_micro)
    if plans.get("padding_stress"):
        stress_ids = plans["padding_stress"][:selected_micro]
        report["padding_stress"] = plain_trial(stress_ids, "length_padding_stress")
        event("padding_stress", status=report["padding_stress"]["status"])

    # Inspect the actual pinned Trainer/Accelerate loader and its scheduler.
    # train() is never called; only the finite explicit engine loop below updates.
    train_args = TrainingArguments(output_dir=str(output / "unused_trainer_output"), seed=42,
        per_device_train_batch_size=selected_micro, per_device_eval_batch_size=selected_micro,
        gradient_accumulation_steps=accumulation, num_train_epochs=4,
        learning_rate=5e-5, weight_decay=.01, warmup_ratio=.01, lr_scheduler_type="cosine",
        bf16=True, optim="adamw_torch", max_grad_norm=1.0, gradient_checkpointing=True,
        report_to=[], local_rank=local, ddp_find_unused_parameters=False)
    trainer = Trainer(model=model, args=train_args, train_dataset=train_data, eval_dataset=valid_data, data_collator=collator)
    rng = (torch.get_rng_state(), torch.cuda.get_rng_state(local), random.getstate(), np.random.get_state())
    loader = trainer.get_train_dataloader()
    loader_length = len(loader)
    report["formal_dataloader"] = {"class": type(loader).__module__ + "." + type(loader).__name__,
        "length_per_rank": loader_length, "sampler_class": type(loader.sampler).__module__ + "." + type(loader.sampler).__name__,
        "batch_sampler_class": type(loader.batch_sampler).__module__ + "." + type(loader.batch_sampler).__name__,
        "batch_sampler_attributes": {key: getattr(loader.batch_sampler, key, None) for key in
            ("batch_size", "drop_last", "even_batches", "split_batches", "num_processes", "process_index")},
        "full_iteration_executed": False, "training_loop": "Pinned HF4.31 Trainer floor(len(loader)/accum), 4 epochs; no formal run"}
    del loader
    torch.set_rng_state(rng[0]); torch.cuda.set_rng_state(rng[1], local)
    random.setstate(rng[2]); np.random.set_state(rng[3])
    formal_steps = max(loader_length // accumulation, 1) * 4
    warmup_steps = train_args.get_warmup_steps(formal_steps)
    trainer.create_optimizer()
    optimizer = trainer.optimizer
    trainer.create_scheduler(num_training_steps=formal_steps, optimizer=optimizer)
    scheduler = trainer.lr_scheduler
    id_to_name = {id(p): name for name, p in named.items()}
    opt_ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    if len(opt_ids) != len(set(opt_ids)) or set(opt_ids) != {id(named[name]) for name in train_names}:
        raise AssertionError("AdamW has missing, extra, or duplicate parameters")
    report["optimizer"] = {"class": type(optimizer).__module__ + "." + type(optimizer).__name__,
        "groups": [{"weight_decay": group["weight_decay"], "names": [id_to_name[id(p)] for p in group["params"]]} for group in optimizer.param_groups],
        "exact_trainable_membership": True, "learning_rate": 5e-5, "warmup_steps": warmup_steps, "formal_schedule_steps": formal_steps}
    ds_config = {"bf16": {"enabled": True}, "zero_optimization": {"stage": 2,
        "overlap_comm": False, "contiguous_gradients": True, "reduce_scatter": True,
        "allgather_partitions": True, "reduce_bucket_size": 5000000, "allgather_bucket_size": 5000000},
        "gradient_accumulation_steps": accumulation, "gradient_clipping": 1.0,
        "train_batch_size": 128, "train_micro_batch_size_per_gpu": selected_micro,
        "steps_per_print": 100000, "wall_clock_breakdown": False, "zero_allow_untested_optimizer": True,
        "flops_profiler": {"enabled": False}}
    write_json(output / ("deepspeed_effective_rank" + str(rank) + ".json"), ds_config)
    report["deepspeed_config"] = ds_config
    event("before_zero2_init", micro=selected_micro, accumulation=accumulation)
    guard()
    t0 = time.monotonic()
    engine, _, _, scheduler = deepspeed.initialize(model=model, optimizer=optimizer, lr_scheduler=scheduler,
        config=ds_config, model_parameters=[p for p in model.parameters() if p.requires_grad], dist_init_required=False)
    torch.cuda.synchronize()
    report["zero2_initialization_seconds"] = max_rank_seconds(time.monotonic() - t0)
    groups = getattr(engine.optimizer, "bit16_groups", None)
    if groups is None or {id(p) for group in groups for p in group} != set(opt_ids):
        raise AssertionError("Actual ZeRO-2 original parameter membership is not exact")
    report["optimizer"]["zero2_exact_trainable_membership"] = True
    report["optimizer"]["zero2_class"] = type(engine.optimizer).__module__ + "." + type(engine.optimizer).__name__
    report["optimizer"]["master_partitions"] = [{"shape": list(p.shape), "dtype": str(p.dtype), "device": str(p.device)}
        for p in getattr(engine.optimizer, "single_partition_of_fp32_groups", [])]
    if not report["optimizer"]["master_partitions"] or any(value["dtype"] != "torch.float32" or not value["device"].startswith("cuda")
            for value in report["optimizer"]["master_partitions"]):
        raise AssertionError("ZeRO-2 must retain FP32 master partitions on GPU without optimizer offload")
    report["training_definition"] = {"micro_batch": selected_micro, "accumulation": accumulation, "global_batch": 128,
        "max_updates": limit_updates, "formal_epochs_estimated_only": 4, "formal_loader_length": loader_length,
        "formal_steps": formal_steps, "formal_warmup_steps": warmup_steps, "loss_normalization":
        "HF CE mean over all valid shifted tokens in each rank microbatch; DeepSpeed divides by accumulation then averages ranks"}
    report["trainable_dtypes_after_zero2"] = {name: str(named[name].dtype) for name in train_names}
    report["pre_engine_parameter_differences"] = {name: {"max_abs_in_fp32": float((small_snapshot(name).float() - before[name].float()).abs().max()),
        "before_dtype": str(before[name].dtype), "after_dtype": str(named[name].dtype)} for name in train_names}
    # Engine dtype/flattening is setup, not an optimizer update. Baseline all
    # actual update evidence after engine initialization and before the first LR.
    before = {name: small_snapshot(name) for name in train_names}
    frozen_before = {name: tensor_digest(named[name]) for name in frozen_names}
    report["update_baseline"] = "Post-ZeRO-initialization, pre-first-optimizer-step; setup casting is excluded from parameter deltas"
    # Drop Trainer references before fresh-base reload later.
    del trainer, groups
    engine.train()
    gradient_scalars.clear()
    report["updates"] = []
    report["global_update_reservations"] = []
    ledger = cfg.get("update_ledger", cfg.get("update_budget_path", "/home/bwwu/lc-rec/six-task-20261110/update-budget.json"))
    report["global_update_ledger_path"] = ledger

    def optimizer_progress():
        inner = engine.optimizer.optimizer
        steps = []
        for state in inner.state.values():
            if "step" in state:
                value = state["step"]
                steps.append(int(value.item()) if hasattr(value, "item") else int(value))
        return {"engine_global_steps": int(engine.global_steps), "engine_micro_steps": int(engine.micro_steps),
                "adam_steps": steps, "state_entries": len(inner.state), "overflow": bool(getattr(engine.optimizer, "overflow", False))}

    for update in range(limit_updates):
        guard()
        ids = plans["mixed_global128_" + str(update % 2)]
        if len(ids) != 128 or set(records[i]["task"] for i in ids) != set(TASKS):
            raise AssertionError("Every global128 technical update must jointly contain all six tasks")
        prepared = []
        prep_start = time.monotonic()
        for micro_index in range(accumulation):
            begin = (micro_index * world + rank) * selected_micro
            prepared.append(cpu_batch(ids[begin:begin + selected_micro], True))
        prep_seconds = time.monotonic() - prep_start
        dist.barrier(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(local)
        started = time.monotonic()
        row = {"update": update, "status": "RUNNING", "microsteps": [], "lr_before": [float(g["lr"]) for g in optimizer.param_groups],
               "progress_before": optimizer_progress(), "cpu_collator_seconds": prep_seconds,
               "rank_counts": {key: 0 for key in ("samples", "tokens", "padded_tokens", "label_tokens", "shift_label_tokens")}}
        report["current_update"] = row
        previous_global = engine.global_steps
        for micro_index, (cpu, counts) in enumerate(prepared):
            guard()
            for key in row["rank_counts"]:
                row["rank_counts"][key] += counts[key]
            step = {"micro": micro_index, "counts": counts, "global_steps_before": engine.global_steps,
                    "micro_steps_before": engine.micro_steps}
            row["microsteps"].append(step)
            torch.cuda.synchronize(); t0 = time.monotonic()
            batch = {key: value.to(device) for key, value in cpu.items()}
            torch.cuda.synchronize(); step["transfer_seconds"] = time.monotonic() - t0
            boundary = engine.is_gradient_accumulation_boundary()
            step["accumulation_boundary"] = boundary
            if boundary != (micro_index == accumulation - 1):
                raise AssertionError("Actual DeepSpeed accumulation boundary differs from global128 plan")
            t0 = time.monotonic()
            result = engine(**batch)
            torch.cuda.synchronize(); step["forward_seconds"] = time.monotonic() - t0
            if not torch.isfinite(result.loss).all().item():
                raise FloatingPointError("Nonfinite mixed-training loss")
            step["loss"] = float(result.loss.detach())
            t0 = time.monotonic()
            engine.backward(result.loss)
            torch.cuda.synchronize(); step["backward_seconds"] = time.monotonic() - t0
            # Reserve only the actual global optimizer boundary; micro-step calls
            # still advance the engine's accumulation counter exactly once.
            if boundary:
                decision = [None]
                if rank == 0:
                    try:
                        decision[0] = {"reservation": reserve_update(ledger, args.job_id, args.run_commit, update)}
                    except Exception as error:
                        decision[0] = {"error": str(error)}
                dist.broadcast_object_list(decision, src=0)
                if "error" in decision[0]:
                    raise RuntimeError(decision[0]["error"])
                report["global_update_reservations"].append(decision[0]["reservation"])
                flush()
            t0 = time.monotonic()
            engine.step()
            torch.cuda.synchronize(); step["engine_step_seconds"] = time.monotonic() - t0
            step.update(global_steps_after=engine.global_steps, micro_steps_after=engine.micro_steps)
            if engine.global_steps != previous_global + int(boundary) or engine.micro_steps != step["micro_steps_before"] + 1:
                raise AssertionError("Optimizer/microstep progress escaped unique global128 boundary")
            del batch, result
        row["rank_seconds"] = time.monotonic() - started
        row["max_rank_seconds"] = max_rank_seconds(row["rank_seconds"])
        row["global_counts"] = global_counts(row["rank_counts"])
        row["samples_per_second"] = row["global_counts"]["samples"] / row["max_rank_seconds"]
        row["effective_tokens_per_second"] = row["global_counts"]["tokens"] / row["max_rank_seconds"]
        row["supervised_tokens_per_second"] = row["global_counts"]["shift_label_tokens"] / row["max_rank_seconds"]
        row["padding_ratio"] = 1 - row["global_counts"]["tokens"] / row["global_counts"]["padded_tokens"]
        row["memory"] = memory()
        row["host_resources"] = host_resources()
        row["nonzero_learning_rate"] = any(value > 0 for value in row["lr_before"])
        row["progress_after"] = optimizer_progress()
        gradients = summarize_gradients()
        row["gradient_evidence"] = gradients
        if len(gradients) != 450 or not all(value["finite"] for value in gradients.values()):
            raise FloatingPointError("Every mixed-update trainable gradient must be present and finite")
        if not all(gradients[name].get("sid_rows_finite") and gradients[name].get("sid_rows_max_abs", 0) > 0 for name in active_names):
            raise AssertionError("Both actual active SID row gradients must be finite and nonzero")
        progress = [None] * world
        dist.all_gather_object(progress, row["progress_after"])
        row["all_rank_progress"] = progress
        if any(p["engine_global_steps"] != update + 1 or set(p["adam_steps"]) != {update + 1} or p["overflow"] for p in progress):
            raise AssertionError("Real AdamW optimizer update/state is not synchronized on both ranks")
        row["phase_seconds"] = {key: sum(m[key] for m in row["microsteps"]) for key in
            ("transfer_seconds", "forward_seconds", "backward_seconds", "engine_step_seconds")}
        # Quantify the exact microbatch token-weighting consequence. A token
        # in microbatch b gets weight1/(world*accum*valid_tokens_b), whereas a
        # hypothetical global-token mean would weight1/global_valid_tokens.
        micro_labels_all = [None] * world
        dist.all_gather_object(micro_labels_all, [m["counts"]["shift_label_tokens"] for m in row["microsteps"]])
        relative_weights = [row["global_counts"]["shift_label_tokens"] / (world * accumulation * value)
                            for values in micro_labels_all for value in values]
        row["loss_token_weighting"] = {"all_rank_micro_shift_label_counts": micro_labels_all,
            "relative_to_global_token_mean_min": min(relative_weights), "relative_to_global_token_mean_max": max(relative_weights),
            "formula": "weight_relative_to_global_token_mean = global_valid_tokens/(world*accum*micro_valid_tokens)",
            "loss_changed": False, "formal_micro_layout_method_decision_needed": True}
        row["status"] = "PASS"
        report["updates"].append(row)
        report.pop("current_update", None)
        event("global_update_complete", update=update, nonzero_lr=row["nonzero_learning_rate"], samples_per_second=row["samples_per_second"])
    report["training_status"] = "PASS" if report["updates"] else "NO_UPDATES_REQUESTED"
    changes = {}
    for name in train_names:
        current = small_snapshot(name)
        difference = (current.float() - before[name].float()).abs()
        changes[name] = {"changed_values": int((difference != 0).sum()), "max_abs": float(difference.max()),
                         "finite": bool(torch.isfinite(current).all())}
    report["parameter_changes"] = changes
    report["full_trainable_parameter_finite"] = {name: bool(torch.isfinite(torch.linalg.vector_norm(
        named[name].detach(), ord=float("inf"), dtype=torch.float32)).item()) for name in train_names}
    if not all(report["full_trainable_parameter_finite"].values()):
        raise FloatingPointError("At least one complete post-update trainable tensor is nonfinite")
    report["optimizer"]["actual_state_tensors"] = [{"state": key, "shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
        for state in engine.optimizer.optimizer.state.values() for key, value in state.items() if torch.is_tensor(value)]
    if any(value["state"] in ("exp_avg", "exp_avg_sq") and not value["device"].startswith("cuda")
           for value in report["optimizer"]["actual_state_tensors"]):
        raise AssertionError("AdamW moment tensors were unexpectedly offloaded")
    report["frozen_originals"] = {name: {"before_sha256": frozen_before[name], "after_sha256": tensor_digest(named[name])} for name in frozen_names}
    if any(value["before_sha256"] != value["after_sha256"] for value in report["frozen_originals"].values()):
        raise AssertionError("Frozen original embedding/head changed")
    if report["updates"]:
        if not any(row["nonzero_learning_rate"] for row in report["updates"]):
            raise AssertionError("No real nonzero-LR optimizer update was performed")
        if not all(changes[name]["changed_values"] > 0 for name in active_names):
            raise AssertionError("Active SID embedding/head rows did not both actually change")
        if not any(value["changed_values"] > 0 for name, value in changes.items() if "lora_B" in name):
            raise AssertionError("No LoRA_B actual value changed")
        if not all(value["finite"] for value in changes.values()):
            raise FloatingPointError("Nonfinite post-update parameters")
    fingerprints = {name: tensor_digest(named[name]) for name in train_names}
    all_fingerprints = [None] * world
    dist.all_gather_object(all_fingerprints, fingerprints)
    report["rank_parameter_sync"] = {"scope": "All bytes of all450 complete active/LoRA tensors", "equal": all_fingerprints[0] == all_fingerprints[1],
                                      "sha256": hashlib.sha256(json.dumps(fingerprints, sort_keys=True).encode()).hexdigest()}
    if not report["rank_parameter_sync"]["equal"]:
        raise AssertionError("Replicated trained parameters differ between GPU ranks")
    for handle in gradient_handles:
        handle.remove()
    gradient_scalars.clear()
    event("training_checks_complete")

    # Validation is isolated from loss-based model selection: three real rows.
    model.eval()
    report["validation"] = {"status": "RUNNING", "full_rows_per_epoch": len(valid_data),
        "sample_valid": data_args.sample_valid, "sampled_prompts_per_user": data_args.valid_prompt_sample_num,
        "checkpoint_selection": "Author Trainer eval_loss with load_best_model_at_end=True; no checkpoint selected in this probe", "timings": []}
    valid_manifest_path = args.samples.with_name("validation_samples.json")
    if valid_manifest_path.exists():
        valid_manifest = json.loads(valid_manifest_path.read_text())
        valid_ids = [record["validation_index"] for record in valid_manifest["records"]]
        for record in valid_manifest["records"]:
            actual = valid_data[record["validation_index"]]
            if actual != {"input_ids": record["input_ids"], "labels": record["labels"]}:
                raise AssertionError("Actual author validation dataset differs from CPU fixed sample evidence")
            collated = collator([actual])
            if any(collated[key][0].tolist() != record["collated"][key] for key in ("input_ids", "attention_mask", "labels")):
                raise AssertionError("Actual author validation collator differs from CPU fixed evidence")
        report["validation"]["sample_manifest"] = record_file(valid_manifest_path)
        report["validation"]["selection"] = valid_manifest["selection"]
    else:
        ordered = sorted(range(len(valid_data.inter_data)), key=lambda i: (len(valid_data.inter_data[i]["inters"]), i))
        valid_users = [ordered[int((len(ordered) - 1) * q)] for q in (.1, .5, .9)]
        valid_ids = [index * data_args.valid_prompt_sample_num + slot for index in valid_users for slot in range(data_args.valid_prompt_sample_num)]
        report["validation"]["selection"] = "History-string lengthp10/p50/p90 user quantiles, index tie break; both actual original sampled templates per user"
    with torch.no_grad():
        for index in valid_ids:
            guard()
            index = int(index.get("dataset_index", index.get("index"))) if isinstance(index, dict) else int(index)
            t0 = time.monotonic()
            cpu = collator([valid_data[index]])
            collator_seconds = time.monotonic() - t0
            torch.cuda.synchronize(); t0 = time.monotonic()
            batch = {key: value.to(device) for key, value in cpu.items()}
            torch.cuda.synchronize(); transfer_seconds = time.monotonic() - t0
            dist.barrier(); torch.cuda.synchronize(); t0 = time.monotonic()
            result = model(**batch)
            torch.cuda.synchronize()
            report["validation"]["timings"].append({"dataset_index": index, "seconds_max_rank": max_rank_seconds(time.monotonic() - t0),
                "cpu_collator_seconds": collator_seconds, "transfer_seconds": transfer_seconds,
                "input_tokens": int(cpu["attention_mask"].sum()), "label_tokens": int((cpu["labels"] != -100).sum()),
                "loss": float(result.loss), "finite": bool(torch.isfinite(result.loss))})
            del batch, result
    report["validation"]["status"] = "PASS" if all(row["finite"] for row in report["validation"]["timings"]) else "FAIL"
    event("validation_complete", status=report["validation"]["status"])

    # All necessary trainable matrices are saved. No base/optimizer checkpoint.
    adapter_dir = output / "adapter"
    dist.barrier(); torch.cuda.synchronize(); t0 = time.monotonic()
    if rank == 0:
        guard()
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(adapter_dir), safe_serialization=False)
        tokenizer.save_pretrained(str(adapter_dir))
        model.config.save_pretrained(str(adapter_dir))
        write_json(adapter_dir / "probe_effective_config.json", cfg)
        write_json(adapter_dir / "sid_token_order.json", {"tokens": sid_tokens, "ids": sid_ids})
    dist.barrier(); torch.cuda.synchronize()
    report["save"] = {"status": "PASS", "seconds_max_rank": max_rank_seconds(time.monotonic() - t0),
                      "path": str(adapter_dir), "training_state_checkpoint": "NOT_TESTED; adapter saving does not prove resumable optimizer/scheduler/RNG state"}
    saved = torch.load(adapter_dir / "adapter_model.bin", map_location="cpu", weights_only=True)
    expected_state = get_peft_model_state_dict(model)
    if set(saved) != set(expected_state):
        raise AssertionError("Saved adapter keyset does not contain exactly every expected active/LoRA tensor")
    adapter_metadata = {}
    for key in expected_state:
        if saved[key].shape != expected_state[key].shape or not torch.equal(saved[key], expected_state[key].detach().cpu()):
            raise AssertionError("Saved adapter tensor shape/value mismatch: " + key)
        adapter_metadata[key] = {"shape": list(saved[key].shape), "dtype": str(saved[key].dtype), "sha256": tensor_digest(saved[key])}
    if len(saved) != 450:
        raise AssertionError("Saved official LC-Rec LoRA adapter must have448 LoRA plus2 full active tensors")
    report["save"].update(strict_key_shape_value_check=True, saved_tensors=len(saved),
        saved_elements=sum(value.numel() for value in saved.values()),
        files=[record_file(path) for path in sorted(adapter_dir.iterdir()) if path.is_file()],
        tensor_manifest=adapter_metadata)
    del expected_state, saved
    event("adapter_save_verified")
    # Release the old GPU model and every optimizer reference before loading a
    # genuinely fresh base. No Int8/LoRA path is silently substituted.
    report["reload"] = {"status": "RUNNING", "kind": "Fresh fixed base + real PEFT PeftModel.from_pretrained", "memory_before_release": memory()}
    del named, optimizer, scheduler, engine, model, sid_tensor, parameter
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    report["reload"]["memory_after_release"] = memory()
    dist.barrier(); t0 = time.monotonic()
    guard()
    model = load_base()
    model = PeftModel.from_pretrained(model, str(adapter_dir), torch_dtype=torch.bfloat16,
                                     device_map={"": local}, is_trainable=False)
    torch.cuda.synchronize()
    report["reload"]["seconds_max_rank"] = max_rank_seconds(time.monotonic() - t0)
    loaded_state = get_peft_model_state_dict(model)
    if set(loaded_state) != set(adapter_metadata):
        raise AssertionError("Fresh PEFT reload is missing/adding expected adapter tensors")
    saved = torch.load(adapter_dir / "adapter_model.bin", map_location="cpu", weights_only=True)
    reload_dtype_checks = {}
    for key, tensor in loaded_state.items():
        actual = tensor.detach().cpu()
        expected = saved[key]
        if list(tensor.shape) != adapter_metadata[key]["shape"] or not torch.equal(actual, expected.to(actual.dtype)):
            raise AssertionError("Fresh PEFT reload tensor shape/value differs: " + key)
        # BF16 to FP32 is lossless; verify the original stored bytes as well as
        # exact equality in the public PEFT API's actual inference dtype.
        if tensor_digest(actual.to(expected.dtype)) != adapter_metadata[key]["sha256"]:
            raise AssertionError("Fresh PEFT reload cannot reproduce the exact stored tensor bytes: " + key)
        reload_dtype_checks[key] = {"saved_dtype": str(expected.dtype), "reload_dtype": str(actual.dtype),
            "exact_after_explicit_dtype_conversion": True, "max_abs_value_difference_fp32": float((actual.float() - expected.float()).abs().max())}
    loaded_tokenizer = LlamaTokenizer.from_pretrained(str(adapter_dir), local_files_only=True)
    if loaded_tokenizer.convert_tokens_to_ids(sid_tokens) != sid_ids or len(loaded_tokenizer) != len(tokenizer):
        raise AssertionError("Saved tokenizer SID order changed during real reload")
    report["reload"].update(status="PASS", strict_all_key_shape_values=True, stored_bytes_after_inverse_dtype_conversion=True,
        native_peft_dtype_checks=reload_dtype_checks, tokenizer_sid_order=True,
        missing_trainable_keys=[], unexpected_adapter_keys=[], training_state_restore="NOT_TESTED")
    del loaded_state, saved, actual, expected, tensor
    event("fresh_base_adapter_reload_pass")
    from inference_probe import run_inference_probe
    report["generation"] = run_inference_probe(model=model, tokenizer=loaded_tokenizer, data_args=data_args,
        sample_doc=sample_doc, config=cfg, output_dir=output, rank=rank, local_rank=local,
        event=event, guard=guard, max_rank_seconds=max_rank_seconds)
    report["status"] = "PASS" if (report["training_status"] == "PASS" and report["reload"]["status"] == "PASS"
        and report["validation"]["status"] == "PASS" and report["generation"]["status"] == "PASS"
        and not report.get("diagnostic_failures") and not report.get("diagnostic_deadline_reached")) else "PARTIAL"
    report["final_memory"] = memory()
    report["sample_replay_verified"] = {"records": len(replayed), "original_dataset_getitem": True, "original_collator_exact": True}
    event("all_gpu_stages_complete", status=report["status"])


def main():
    args = parse_cli()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("RANK", 0))
    report_path = args.output_dir / ("probe_rank" + str(rank) + ".json")
    event_path = args.output_dir / ("events_rank" + str(rank) + ".jsonl")
    report = {"status": "RUNNING", "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "argv": sys.argv, "cwd": os.getcwd(), "script": record_file(__file__), "diagnostics": [], "updates": []}
    start = time.monotonic()

    def flush():
        report["elapsed_seconds"] = time.monotonic() - start
        write_json(report_path, report)

    def event(stage, **values):
        report["stage"] = stage
        row = {"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - start,
               "rank": rank, "stage": stage, **values}
        with event_path.open("a") as stream:
            stream.write(json.dumps(clean(row), allow_nan=False) + "\n")
            stream.flush()
        flush()
        print(json.dumps(clean(row), allow_nan=False), flush=True)

    try:
        event("started")
        run(args, report, flush, event)
    except BaseException as error:
        report["status"] = "PARTIAL" if report.get("updates") or report.get("diagnostics") else "FAIL"
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        print(report["error"]["traceback"], file=sys.stderr, flush=True)
    finally:
        report["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        report["host_final"] = host_resources()
        report["elapsed_seconds"] = time.monotonic() - start
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_initialized():
            try:
                report["gpu_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / GIB
                report["gpu_peak_reserved_gib"] = torch.cuda.max_memory_reserved() / GIB
            except Exception as error:
                report["gpu_final_query_error"] = str(error)
        flush()
        dist = sys.modules.get("torch.distributed")
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
