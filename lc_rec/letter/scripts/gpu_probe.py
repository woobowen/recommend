#!/usr/bin/env python3
"""At most nine LETTER A technical updates; never a full-epoch training entry.

Run with torchrun --nproc_per_node=2. Uses the original LETTER model, FastChat
patch, SeqRec data and collator. Only the measurement loop and explicit DS auto
values differ from the author's Trainer entry. Mixed batches are fixed real
rows/prompts from audit_data.py; representative batches repeat a real row to
measure the length envelope, which is explicitly distinct from mixed throughput.
No recommendation metric or complete epoch is computed. Inference is separate.
"""
from __future__ import annotations

import argparse
import gc
import datetime as dt
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import resource
import random
import socket
import sys
import time
import traceback
from types import SimpleNamespace


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def write(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def tensor_json(value):
    if hasattr(value, "numel"):
        return value.item() if value.numel() == 1 else value.tolist()
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--max-updates", type=int, default=9,
                        help="Task-wide remaining global64 update allowance; never more than nine. Caller must subtract previous jobs' real updates.")
    parser.add_argument("--max-seconds", type=int, default=3300)
    parser.add_argument("--minimum-host-free-gib", type=float, default=40)
    parser.add_argument("--skip-adapter-save", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    report = {"status": "RUNNING", "rank": rank, "local_rank": local_rank, "world_size": world,
              "host": socket.gethostname(), "pid": os.getpid(), "argv": sys.argv,
              "cwd": str(Path.cwd()), "interpreter": sys.executable,
              "source_commit": args.source_commit,
              "precision_context": {"explicit_global_bf16_autocast": False,
                                    "history_binding": "Author did not bind framework versions; no claim of historical precision equivalence",
                                    "mode": "Native DS/PEFT context; after a proven dtype error, user-approved probe-only BF16 autocast inside necessary LoRA A linear calls"},
              "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "stages": [], "updates": [], "technical_probe_only": True}
    report_path = output / f"rank{rank}.json"
    torch = None

    def stage(name):
        report["stages"].append({"stage": name, "elapsed_seconds": time.monotonic() - started})
        write(report_path, report)
        print(json.dumps({"rank": rank, "stage": name}), flush=True)

    def guard():
        if time.monotonic() - started > args.max_seconds:
            raise TimeoutError("Bounded probe time limit reached")
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                available = int(line.split()[1]) / 1024**2
                report["minimum_observed_host_memavailable_gib"] = min(available, report.get("minimum_observed_host_memavailable_gib", available))
                if available < args.minimum_host_free_gib:
                    raise MemoryError(f"MemAvailable {available:.2f}GiB < {args.minimum_host_free_gib}GiB")
                break

    def host_resources():
        memory = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in ("MemAvailable", "SwapTotal", "SwapFree"):
                memory[key + "_gib"] = int(value.split()[0]) / 1024**2
        status = {}
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(("VmRSS:", "VmSwap:")):
                key, value = line.split(":", 1)
                status[key + "_gib"] = int(value.split()[0]) / 1024**2
        return {**memory, **status}

    def track_gpu_peak():
        if torch is not None and torch.cuda.is_initialized():
            for label, amount in (("allocated", torch.cuda.max_memory_allocated()),
                                  ("reserved", torch.cuda.max_memory_reserved())):
                key = "gpu_entire_job_peak_" + label + "_gib"
                report[key] = max(report.get(key, 0), amount / 1024**3)

    try:
        if (world, args.micro_batch, args.accumulation) != (2, 8, 4):
            raise ValueError("This approved probe requires exactly two ranks, micro_batch=8, accumulation=4, global64")
        if not 0 < args.max_seconds <= 3300:
            raise ValueError("--max-seconds must be at most 3300 (55 minutes)")
        if not 1 <= args.max_updates <= 9:
            raise ValueError("--max-updates must be within the remaining task-wide allowance, between one and nine")
        guard()
        import torch
        import torch.distributed as dist
        import torch.nn.functional as F
        import deepspeed
        import bitsandbytes as bnb
        import numpy as np
        import transformers
        from transformers import LlamaTokenizer, Trainer, TrainingArguments
        import peft
        from peft import LoraConfig, TaskType, get_peft_model
        from peft import get_peft_model_state_dict
        from compat import (restore_legacy_bnb_state_field, peft05_lora_a_targets,
                            install_peft05_lora_a_bf16_autocast, install_peft05_out_of_place)

        if importlib.metadata.version("peft") != "0.5.0":
            raise RuntimeError("This probe is fixed to the user-approved installed PEFT 0.5.0; no version upgrade")

        prepare_function = getattr(peft, "prepare_model_for_int8_training", None)
        if prepare_function is None:
            prepare_function = peft.prepare_model_for_kbit_training
        report["prepare_api"] = {"requested_author_symbol": "prepare_model_for_int8_training",
                                 "actual_callable": prepare_function.__module__ + "." + prepare_function.__qualname__,
                                 "signature": str(inspect.signature(prepare_function)),
                                 "source_sha256": hashlib.sha256(inspect.getsource(prepare_function).encode()).hexdigest(),
                                 "fallback_to_kbit_alias": not hasattr(peft, "prepare_model_for_int8_training")}

        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", timeout=dt.timedelta(minutes=8))
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 unsupported")
        report["versions"] = {name: importlib.metadata.version(name) for name in
                              ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "deepspeed", "fschat", "flash-attn")}
        report["gpu"] = {"name": torch.cuda.get_device_name(local_rank), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                         "total_memory_gib": torch.cuda.get_device_properties(local_rank).total_memory / 1024**3}
        source = Path(args.source_root).resolve()
        sys.path.insert(0, str(source))
        # Match author import order: LETTER imports transformers, then patch the
        # shared LlamaAttention class before constructing any model.
        from modeling_letter import LETTER
        from fastchat.train.llama2_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
        replace_llama_attn_with_flash_attn()
        from compat import adapt_fastchat_unpad_arity
        report["fastchat_unpad_arity_compatibility"] = adapt_fastchat_unpad_arity()
        from transformers.models.llama.modeling_llama import LlamaAttention
        from data import SeqRecDataset
        from collator import Collator
        from utils import set_seed
        set_seed(42)
        report["attention_forward"] = {"module": LlamaAttention.forward.__module__,
                                       "source_file": inspect.getsourcefile(LlamaAttention.forward),
                                       "source_sha256": hashlib.sha256(inspect.getsource(LlamaAttention.forward).encode()).hexdigest()}
        if "fastchat" not in LlamaAttention.forward.__module__:
            raise AssertionError("Author FastChat attention patch is not active")
        report["source_identity"] = {name: sha(source / name) for name in
                                     ("modeling_letter.py", "data.py", "collator.py", "utils.py", "lora_finetune.py", "config/ds_z2_bf16.json")}
        report["probe_identity"] = {Path(__file__).name: sha(__file__), "compat.py": sha(Path(__file__).with_name("compat.py"))}
        audited = Path(args.audit_dir).resolve()
        selections = json.loads((audited / "probe_samples.json").read_text())
        if selections["schema"] != "letter_seqrec_probe_samples_v1" or selections["seed"] != 42:
            raise AssertionError("Unexpected CPU audit selection schema/seed")
        report["probe_samples_sha256"] = sha(audited / "probe_samples.json")
        report["model_config_identity"] = {"path": str(Path(args.model).resolve()),
                                           "config_sha256": sha(Path(args.model) / "config.json"),
                                           "model_weight_hashing": "use parent fixed-asset manifest; do not repeatedly hash 13GiB"}
        data_args = SimpleNamespace(dataset="Instruments", data_path=str(Path(args.data_root).resolve()), index_file=".index.json",
                                    max_his_len=20, his_sep=", ", add_prefix=False, only_train_response=True,
                                    sample_valid=True, valid_prompt_id=0)
        for kind, identity in selections["a_data_identity"].items():
            path = Path(data_args.data_path) / "Instruments" / f"Instruments.{kind}.json"
            if sha(path) != identity["sha256"]:
                raise AssertionError(f"Audited A {kind} identity mismatch")
        train_data = SeqRecDataset(data_args, mode="train", prompt_sample_num=1, sample_num=0)
        valid_data = SeqRecDataset(data_args, mode="valid", prompt_sample_num=2)
        tokenizer = LlamaTokenizer.from_pretrained(args.model, model_max_length=2048, padding_side="right", local_files_only=True)
        tokenizer.pad_token_id = 0
        original_vocab = len(tokenizer)
        tokenizer.add_tokens(train_data.get_new_tokens())
        audited_tokenizer = LlamaTokenizer.from_pretrained(audited / "tokenizer_a", local_files_only=True)
        if tokenizer.get_vocab() != audited_tokenizer.get_vocab():
            raise AssertionError("Training expanded token IDs differ from CPU audit")
        sid_ids = sorted(tokenizer.convert_tokens_to_ids(t) for t in train_data.get_new_tokens())
        collator = Collator(data_args, tokenizer)
        report["data"] = {"train_rows": len(train_data), "valid_two_prompt_rows": len(valid_data),
                          "base_vocab": original_vocab, "expanded_vocab": len(tokenizer), "sid_tokens": len(sid_ids),
                          "tokenizer_exact_match_to_cpu_audit": True}
        report["measurement_sampling_limit"] = "Six mixed updates reuse the fixed CPU-audited 128 real row/prompt pool; steady updates 3/4/5 provide three timings of only two distinct global64 groups. Length-envelope updates repeat one real row. No full-epoch performance measurement."

        def text(selected):
            row = train_data.inter_data[selected["row_idx"]]
            prompt = train_data.prompts[selected["prompt_id"]]
            inp, full = train_data._get_text_data(row, prompt)
            return {"input_ids": inp, "labels": full}

        def batch_for(update, micro):
            if update < 6:
                samples = selections["train_mixed"]
                begin = (update * 64 + micro * world * args.micro_batch + rank * args.micro_batch) % len(samples)
                selected = [samples[(begin + j) % len(samples)] for j in range(args.micro_batch)]
            else:
                selected = [selections["train_representatives"][("q10", "q50", "max")[update - 6]]] * args.micro_batch
            pairs = [text(item) for item in selected]
            batch = collator(pairs)
            for i, pair in enumerate(pairs):
                input_ids = tokenizer(pair["input_ids"])["input_ids"]
                full_ids = tokenizer(pair["labels"] + tokenizer.eos_token)["input_ids"]
                expected = [-100] * len(input_ids) + full_ids[len(input_ids):]
                expected += [-100] * (batch["labels"].shape[1] - len(expected))
                if full_ids[:len(input_ids)] != input_ids or batch["labels"][i].tolist() != expected:
                    raise AssertionError("Original response-only label/mask mismatch")
                if len(full_ids) != selected[i]["full_length_including_eos"]:
                    raise AssertionError("Real sample length differs from CPU audit")
            return batch, selected

        first_batch, _ = batch_for(0, 0)
        report["mask_first_batch"] = {"input_ids": first_batch["input_ids"].tolist(), "attention_mask": first_batch["attention_mask"].tolist(),
                                      "labels": first_batch["labels"].tolist(), "prefix_mask_exact": True}
        stage("data_and_mask_ready")
        guard()
        load_started = time.monotonic()
        model = LETTER.from_pretrained(args.model, load_in_8bit=True, device_map={"": local_rank}, local_files_only=True)
        torch.cuda.synchronize()
        report["base_model_load_seconds"] = time.monotonic() - load_started
        model.set_hyper(.8)
        model.resize_token_embeddings(len(tokenizer))
        model = prepare_function(model)
        report["compatibility_changes"] = restore_legacy_bnb_state_field(model)
        config = LoraConfig(r=8, lora_alpha=16,
                            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
                            modules_to_save=["embed_tokens", "lm_head"], lora_dropout=.05, bias="none",
                            inference_mode=False, task_type=TaskType.CAUSAL_LM)
        model = get_peft_model(model, config)
        report["peft05_out_of_place_compatibility"] = install_peft05_out_of_place(model)
        if report["peft05_out_of_place_compatibility"]["target_count"] != 224:
            raise AssertionError("Out-of-place compatibility did not find exactly 32 layers x seven int8 LoRA modules")
        for name, param in model.named_parameters():
            if "original_module" in name and any(module_name in name for module_name in config.modules_to_save):
                param.requires_grad = False
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        train_args = TrainingArguments(output_dir=str(output / "unused_trainer_output"), seed=42,
                                       per_device_train_batch_size=args.micro_batch, gradient_accumulation_steps=args.accumulation,
                                       num_train_epochs=4, learning_rate=2e-4, weight_decay=.01, warmup_ratio=.01,
                                       lr_scheduler_type="cosine", bf16=True, optim="adamw_torch", max_grad_norm=1.0,
                                       gradient_checkpointing=True, report_to=[], local_rank=local_rank, ddp_find_unused_parameters=False)
        trainer = Trainer(model=model, args=train_args, train_dataset=train_data, data_collator=collator)
        # Inspect the actual Trainer/Accelerate loader without consuming rows or
        # changing dropout/initialization RNG streams. Tail filling is not
        # inferred from a DistributedSampler formula alone.
        loader_rng = (torch.get_rng_state().clone(), torch.cuda.get_rng_state(local_rank).clone(),
                      random.getstate(), np.random.get_state())
        try:
            actual_loader = trainer.get_train_dataloader()
            actual_loader_batches = len(actual_loader)
            def loader_identity(obj):
                info = {"class": type(obj).__module__ + "." + type(obj).__qualname__}
                for key in ("batch_size", "drop_last", "even_batches", "split_batches", "num_processes", "process_index"):
                    val = getattr(obj, key, None)
                    if isinstance(val, (bool, int, float, str, type(None))):
                        info[key] = val
                return info
            loader_info = {"loader": loader_identity(actual_loader), "length_per_rank": actual_loader_batches,
                           "sampler": loader_identity(actual_loader.sampler),
                           "batch_sampler": loader_identity(actual_loader.batch_sampler),
                           "iteration_executed": False, "tail_behavior": "Structural inspection only; no complete epoch traversed"}
            inner_sampler = getattr(actual_loader.batch_sampler, "batch_sampler", None)
            if inner_sampler is not None:
                loader_info["inner_batch_sampler"] = loader_identity(inner_sampler)
                if hasattr(inner_sampler, "sampler"):
                    loader_info["inner_sampler"] = loader_identity(inner_sampler.sampler)
            report["formal_train_dataloader_inspection"] = loader_info
            del actual_loader
        finally:
            torch.set_rng_state(loader_rng[0])
            torch.cuda.set_rng_state(loader_rng[1], local_rank)
            random.setstate(loader_rng[2])
            np.random.set_state(loader_rng[3])
        trainer.create_optimizer()
        optimizer = trainer.optimizer
        # Transformers 4.31 Trainer floors dataloader/accum, then multiplies by
        # epochs. The remaining partial accumulation is runtime-sensitive and
        # the conservative wall-time estimate separately uses ceil(rows/64).
        dataloader_batches = actual_loader_batches
        updates_per_epoch = max(dataloader_batches // args.accumulation, 1)
        formal_updates = updates_per_epoch * 4
        trainer.create_scheduler(num_training_steps=formal_updates, optimizer=optimizer)
        scheduler = trainer.lr_scheduler
        named = dict(model.named_parameters())
        name_by_id = {id(p): n for n, p in named.items()}
        groups = [{"weight_decay": group["weight_decay"], "names": [name_by_id[id(p)] for p in group["params"]]}
                  for group in optimizer.param_groups]
        selected_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
        if selected_ids != {id(p) for p in named.values() if p.requires_grad}:
            raise AssertionError("Trainer optimizer parameter set differs from requires_grad set")
        if sum(len(group["params"]) for group in optimizer.param_groups) != len(selected_ids):
            raise AssertionError("Duplicate parameter in optimizer groups")
        for n, p in named.items():
            expected = "lora_" in n or "modules_to_save.default" in n
            if p.requires_grad != expected:
                raise AssertionError(f"Unexpected trainability: {n} requires_grad={p.requires_grad}")
        trainable_names = sorted(n for n, p in named.items() if p.requires_grad)
        lora_names = [n for n in trainable_names if "lora_" in n]
        wrapper_names = [n for n in trainable_names if "modules_to_save.default" in n]
        trainable_numel = sum(named[n].numel() for n in trainable_names)
        if (len(trainable_names), len(lora_names), len(wrapper_names), trainable_numel) != (450, 448, 2, 289652736):
            raise AssertionError("Current real trainable inventory differs from fixed LETTER-LC-Rec r8 + active embed/head")
        report["trainable_identity"] = {"parameter_tensors": len(trainable_names), "lora_tensors": len(lora_names),
                                        "active_wrapper_tensors": len(wrapper_names), "parameter_elements": trainable_numel,
                                        "active_wrapper_names": wrapper_names,
                                        "names_shapes_sha256": hashlib.sha256(json.dumps(
                                            [(n, list(named[n].shape)) for n in trainable_names],
                                            separators=(",", ":")).encode()).hexdigest()}
        report["optimizer"] = {"class": type(optimizer).__module__ + "." + type(optimizer).__name__,
                               "defaults": {k: v for k, v in optimizer.defaults.items() if isinstance(v, (str, int, float, bool, tuple, type(None)))},
                               "groups": groups, "exact_trainable_membership": True}
        report["training_definition"] = {"seed": 42, "temperature": .8, "micro_batch": args.micro_batch,
                                          "accumulation": args.accumulation, "global_batch": 64, "epochs_budgeted": 4,
                                          "formal_dataloader_batches_per_rank": dataloader_batches,
                                          "formal_updates_per_epoch_floor": updates_per_epoch, "formal_scheduler_updates": formal_updates,
                                          "warmup_steps": train_args.get_warmup_steps(formal_updates),
                                          "probe_updates": args.max_updates, "probe_global_samples": 64 * args.max_updates,
                                          "task_wide_update_ceiling": 9, "caller_must_account_prior_completed_updates": True,
                                          "scheduler_starts_at_author_warmup": True,
                                          "warmup_timing_note": "3 measurement warmup updates excluded; LR remains in formal warmup. First LR is zero, so LoRA A first becomes nonzero after B is updated on second update."}
        report["parameter_inventory"] = {n: {"shape": list(p.shape), "dtype_before_ds": str(p.dtype), "requires_grad": p.requires_grad,
                                              "in_optimizer": id(p) in selected_ids} for n, p in named.items()}
        report["to_method"] = {"module": model.to.__func__.__module__, "qualname": model.to.__func__.__qualname__}
        ds_config = json.loads((source / "config/ds_z2_bf16.json").read_text())
        ds_config.update(bf16={"enabled": True}, gradient_accumulation_steps=args.accumulation,
                         gradient_clipping=1.0, train_batch_size=64, train_micro_batch_size_per_gpu=args.micro_batch,
                         zero_allow_untested_optimizer=True)
        ds_config["flops_profiler"]["enabled"] = False
        report["deepspeed_config"] = ds_config
        report["engineering_differences"] = ["author Trainer entry replaced with finite 9-update timing loop",
                                             "DS auto fields resolved explicitly; author flops profiler disabled to avoid timing pollution",
                                             "no torch.compile call: author invokes compile only after Trainer construction and Trainer retains original reference",
                                             f"two ranks micro{args.micro_batch}/accum{args.accumulation} instead of four ranks micro8/accum2; global64 preserved",
                                             "Pinned isolated PEFT dependency; historical author PEFT version is unknown"]
        report["engineering_differences"].append("User approved this bounded probe only: after actual original dtype error, BF16 autocast only inside necessary PEFT 0.5 LoRA A linear calls; historical equivalence unknown")
        stage("before_deepspeed")
        engine, _, _, scheduler = deepspeed.initialize(model=model, optimizer=optimizer, lr_scheduler=scheduler, config=ds_config,
                                                       model_parameters=[p for p in model.parameters() if p.requires_grad], dist_init_required=False)
        ds_original_groups = getattr(engine.optimizer, "bit16_groups", None)
        if ds_original_groups is None:
            raise AssertionError("ZeRO-2 original BF16 optimizer membership is not inspectable")
        ds_selected_ids = {id(p) for group in ds_original_groups for p in group}
        if ds_selected_ids != selected_ids or sum(map(len, ds_original_groups)) != len(selected_ids):
            raise AssertionError("ZeRO-2 original parameter groups differ from exact Trainer trainable membership")
        report["optimizer"]["zero2_original_groups_exact_trainable_membership"] = True
        engine.train()
        for n, p in named.items():
            report["parameter_inventory"][n]["dtype_after_ds"] = str(p.dtype)
            report["parameter_inventory"][n]["device_after_ds"] = str(p.device)
        bnb_example = next((m for m in model.modules() if isinstance(m, bnb.nn.Linear8bitLt) and hasattr(m, "state")), None)
        if bnb_example is None:
            raise AssertionError("No actual bnb Linear8bitLt module")

        def int8_evidence():
            result = {"weight_class": type(bnb_example.weight).__name__, "weight_dtype": str(bnb_example.weight.dtype),
                      "weight_device": str(bnb_example.weight.device), "has_fp16_weights": bnb_example.state.has_fp16_weights}
            for owner, obj in (("weight", bnb_example.weight), ("state", bnb_example.state)):
                for key in ("CB", "SCB"):
                    val = getattr(obj, key, None)
                    result[f"{owner}.{key}"] = None if val is None else {"dtype": str(val.dtype), "shape": list(val.shape), "device": str(val.device)}
            return result

        report["int8_before_forward"] = int8_evidence()
        # Hooks retain scalar tensors only. Full-gradient maxima are observed
        # for every real backward, so a late NaN/Inf is not hidden by warmup.
        # Their bounded measurement overhead is included in reported timings.
        gradient_evidence, handles = {}, []
        sid_tensor = torch.tensor(sid_ids, device=local_rank, dtype=torch.long)
        sid_position = {token: i for i, token in enumerate(sid_ids)}
        current_sid_tensors, current_sid_positions = {}, {}
        input_sid_seen, supervised_sid_seen = set(), set()

        def hook_for(name):
            def hook(grad):
                value = grad.detach()
                maximum = value.abs().max()
                current = gradient_evidence.get(name)
                entry = {"max_abs": maximum if current is None else torch.maximum(current["max_abs"], maximum),
                         "calls": 1 if current is None else current["calls"] + 1,
                         "gradient_shape": list(value.shape), "gradient_dtype": str(value.dtype)}
                if "modules_to_save.default" in name:
                    kind = "input" if "embed_tokens" in name else "supervised"
                    selected_sid = current_sid_tensors[kind]
                    if selected_sid.numel():
                        per_row = value.index_select(0, selected_sid).abs().amax(dim=1)
                        sidmax = per_row.max()
                        old = None if current is None else current.get("observed_sid_rows_max_abs")
                        entry["observed_sid_rows_max_abs"] = sidmax if old is None else torch.maximum(old, sidmax)
                        old_rows = torch.zeros(len(sid_ids), device=value.device, dtype=value.dtype) if current is None else current["sid_row_max_abs"]
                        new_rows = old_rows.clone()
                        positions = current_sid_positions[kind]
                        new_rows.index_copy_(0, positions, torch.maximum(old_rows.index_select(0, positions), per_row))
                        entry["sid_row_max_abs"] = new_rows
                    elif current is not None and "observed_sid_rows_max_abs" in current:
                        entry["observed_sid_rows_max_abs"] = current["observed_sid_rows_max_abs"]
                        entry["sid_row_max_abs"] = current["sid_row_max_abs"]
                    entry["sid_gradient_scope"] = kind + " SID rows actually present in each current microbatch"
                gradient_evidence[name] = entry
            return hook

        for n, p in named.items():
            if p.requires_grad:
                handles.append(p.register_hook(hook_for(n)))
        activation = {}
        input_gradient = {"calls": 0}

        def activation_hook(module, inp, out):
            activation["lm_head_input_dtype"] = str(inp[0].dtype)

        activation_handle = model.base_model.model.lm_head.register_forward_hook(activation_hook)

        def embedding_output_hook(module, inp, out):
            activation["embedding_output_requires_grad"] = out.requires_grad
            if out.requires_grad and len(report["updates"]) < 3:
                def observe_input_gradient(grad):
                    maximum = grad.detach().abs().max()
                    old = input_gradient.get("max_abs")
                    input_gradient["max_abs"] = maximum if old is None else torch.maximum(old, maximum)
                    input_gradient["calls"] += 1
                out.register_hook(observe_input_gradient)

        input_activation_handle = model.base_model.model.model.embed_tokens.register_forward_hook(embedding_output_hook)
        snapshot_names = [n for n in named if "modules_to_save.default" in n or "original_module" in n]
        snapshot_names += [n for n in named if "lora_" in n]

        def small_snapshot(name):
            p = named[name].detach()
            if "embed_tokens" in name or "lm_head" in name:
                # All columns of every added SID row; ordinary vocabulary rows
                # are checked for finiteness later, not claimed as delta samples.
                return p.index_select(0, sid_tensor).cpu().clone()
            return p.cpu().clone()

        def tensor_digest(parameter):
            digest = hashlib.sha256()
            flat = parameter.detach().reshape(-1)
            # Preserve actual dtype bytes (including BF16) in bounded chunks.
            for offset in range(0, flat.numel(), 2**20):
                part = flat[offset:offset + 2**20].contiguous().cpu()
                digest.update(part.view(torch.uint8).numpy().tobytes())
            return digest.hexdigest()

        before = {n: small_snapshot(n) for n in snapshot_names}
        frozen_hash_before = {n: tensor_digest(p) for n, p in named.items() if "original_module" in n}
        report["parameter_snapshot_scope"] = {"lora": "All values of all 448 LoRA tensors",
                                               "active_wrappers": "All columns of all added SID rows in active embedding/head",
                                               "frozen_originals": "SHA-256 of every byte of each complete original embedding/head tensor",
                                               "sid_row_ids": sid_ids}
        # Record the actual original failure before installing any precision
        # adaptation. No backward/optimizer update occurs; restore every RNG
        # stream consumed by this diagnostic forward before technical updates.
        guard()
        targets, target_identity = peft05_lora_a_targets(model)
        observation = {"targets": target_identity, "last_lora_a_entered": None}
        observation_handles = []

        def observe_a(name):
            def prehook(module, inputs):
                observation["last_lora_a_entered"] = {"module": name, "input_dtype": str(inputs[0].dtype),
                                                       "weight_dtype": str(module.weight.dtype),
                                                       "weight_device": str(module.weight.device),
                                                       "autocast_enabled": torch.is_autocast_enabled()}
            return prehook

        for name, linear in targets:
            observation_handles.append(linear.register_forward_pre_hook(observe_a(name)))
        rng_cpu = torch.get_rng_state().clone()
        rng_cuda = torch.cuda.get_rng_state(local_rank).clone()
        rng_python, rng_numpy = random.getstate(), np.random.get_state()
        original_global_steps, original_micro_steps = engine.global_steps, engine.micro_steps
        original_optimizer_state_entries = len(optimizer.state)
        original_failure = {"status": "UNTESTED", "global_autocast_enabled": torch.is_autocast_enabled(),
                            "engine_global_steps_before": original_global_steps,
                            "engine_micro_steps_before": original_micro_steps}
        original_batch = {k: v.to(local_rank) for k, v in first_batch.items()}
        try:
            if torch.is_autocast_enabled():
                raise AssertionError("Original precision diagnostic must have no global autocast")
            original_result = engine(**original_batch)
            original_failure.update(status="ORIGINAL_FORWARD_SUCCEEDED", matched_dtype_conflict=False,
                                    original_loss=float(original_result.loss.detach()))
            del original_result
        except RuntimeError as error:
            last = observation["last_lora_a_entered"]
            text_error = str(error).lower()
            matching = (last is not None and last["input_dtype"] == "torch.float32"
                        and last["weight_dtype"] == "torch.bfloat16" and not last["autocast_enabled"]
                        and "dtype" in text_error and "float" in text_error and "bfloat16" in text_error)
            original_failure.update(status="ORIGINAL_DTYPE_ERROR" if matching else "UNEXPECTED_ERROR",
                                    matched_dtype_conflict=matching,
                                    error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()},
                                    actual_failed_module=last)
            report["original_precision_diagnostic"] = original_failure
            write(report_path, report)
            if not matching:
                raise
        finally:
            for handle in observation_handles:
                handle.remove()
            del original_batch
            torch.set_rng_state(rng_cpu)
            torch.cuda.set_rng_state(rng_cuda, local_rank)
            random.setstate(rng_python)
            np.random.set_state(rng_numpy)
            gc.collect()
            torch.cuda.empty_cache()
            original_failure["rng_cpu_restored"] = torch.equal(torch.get_rng_state(), rng_cpu)
            original_failure["rng_cuda_restored"] = torch.equal(torch.cuda.get_rng_state(local_rank), rng_cuda)
            original_failure["engine_global_steps_after"] = engine.global_steps
            original_failure["engine_micro_steps_after"] = engine.micro_steps
            original_failure["optimizer_state_entries_before_after"] = [original_optimizer_state_entries, len(optimizer.state)]
            original_failure["sampled_parameters_unchanged"] = all(torch.equal(small_snapshot(n), value) for n, value in before.items())
            original_failure["parameter_sample_scope"] = "All LoRA and saved/frozen embed/head sampled values; no full-model byte comparison"
            original_failure["bnb_readiness_note"] = "Original forward may populate int8 caches; caches are retained, no backward or optimizer step is invoked"
            report["original_precision_diagnostic"] = original_failure
            write(report_path, report)
        if (engine.global_steps != original_global_steps or engine.micro_steps != original_micro_steps
                or len(optimizer.state) != original_optimizer_state_entries or not original_failure["sampled_parameters_unchanged"]):
            raise AssertionError("Original diagnostic changed optimizer progress or sampled parameters")
        if not original_failure["rng_cpu_restored"] or not original_failure["rng_cuda_restored"]:
            raise AssertionError("Original diagnostic RNG restoration failed")
        if original_failure.get("matched_dtype_conflict"):
            report["lora_a_precision_adaptation"] = install_peft05_lora_a_bf16_autocast(model, original_failure)
        else:
            report["lora_a_precision_adaptation"] = {"adaptation": "Not installed: original forward unexpectedly succeeded"}
        stage("original_precision_diagnostic_complete")
        stage("before_updates")
        report["finite_checks"] = {"loss_scope": "Every real training microbatch loss; checked outside autocast", "loss_microbatches_checked": 0,
                                   "gradient_scope": "Full tensor abs max of every trainable gradient during all requested updates; warmup and final summaries; no per-element host synchronization",
                                   "parameter_scope": "After requested updates, all elements of every requires_grad parameter scanned outside timed throughput"}

        def optimizer_progress():
            inner = getattr(engine.optimizer, "optimizer", None)
            if inner is None:
                raise AssertionError("Underlying AdamW optimizer is not inspectable")
            steps = []
            for state in inner.state.values():
                if "step" in state:
                    value = state["step"]
                    steps.append(int(value.item()) if hasattr(value, "item") else int(value))
            return {"state_entries": len(inner.state), "adam_step_values": steps,
                    "engine_global_steps": engine.global_steps, "engine_micro_steps": engine.micro_steps,
                    "overflow": bool(getattr(engine.optimizer, "overflow", False))}

        report["timing_method"] = {"clock": "monotonic wall time with CUDA synchronize at each transfer/forward/backward/step boundary",
                                   "denominator": "MAX of rank wall times; SUM actual rank samples and token counts",
                                   "overhead_included": "Resource guard, finite loss checks, full-gradient scalar hooks, CUDA phase synchronization, microstep boundary queries",
                                   "overhead_excluded": "Original CPU collator/mask construction, interrank barrier/reductions, JSON write, final parameter/adapter checks",
                                   "interpretation": "Observable synchronized technical-probe phases, not unsynchronized production kernel overlap"}
        report["optimizer_progress_before_updates"] = optimizer_progress()
        if any(step != 0 for step in report["optimizer_progress_before_updates"]["adam_step_values"]):
            raise AssertionError("Pinned AdamW optimizer starts at an unexpected nonzero state step")
        for update in range(args.max_updates):
            guard()
            # CPU original collator and mask verification happen outside timed
            # GPU loop; report this limit when extrapolating end-to-end time.
            preparation_started = time.monotonic()
            prepared = [batch_for(update, micro) for micro in range(args.accumulation)]
            preparation_seconds = time.monotonic() - preparation_started
            torch.cuda.synchronize()
            dist.barrier()
            track_gpu_peak()
            torch.cuda.reset_peak_memory_stats()
            timer = time.monotonic()
            count_tokens, padded_tokens, supervised_tokens = 0, 0, 0
            losses, lengths, microsteps = [], [], []
            previous_global_step = engine.global_steps
            progress_before = optimizer_progress()
            lr_before = [float(group["lr"]) for group in optimizer.param_groups]
            for micro, (batch, chosen) in enumerate(prepared):
                guard()
                count_tokens += int(batch["attention_mask"].sum())
                padded_tokens += batch["input_ids"].numel()
                supervised_tokens += int((batch["labels"] != -100).sum())
                lengths += [x["full_length_including_eos"] for x in chosen]
                input_sid = set(int(v) for v in batch["input_ids"].reshape(-1).tolist() if v >= original_vocab)
                supervised_sid = set(int(v) for v in batch["labels"].reshape(-1).tolist() if v >= original_vocab)
                input_sid_seen.update(input_sid)
                supervised_sid_seen.update(supervised_sid)
                current_sid_tensors.update(input=torch.tensor(sorted(input_sid), device=local_rank, dtype=torch.long),
                                           supervised=torch.tensor(sorted(supervised_sid), device=local_rank, dtype=torch.long))
                current_sid_positions.update(input=torch.tensor([sid_position[v] for v in sorted(input_sid)], device=local_rank, dtype=torch.long),
                                             supervised=torch.tensor([sid_position[v] for v in sorted(supervised_sid)], device=local_rank, dtype=torch.long))
                torch.cuda.synchronize()
                phase_started = time.monotonic()
                batch = {k: v.to(local_rank) for k, v in batch.items()}
                torch.cuda.synchronize()
                transfer_seconds = time.monotonic() - phase_started
                boundary = engine.is_gradient_accumulation_boundary()
                if boundary != (micro == args.accumulation - 1):
                    raise AssertionError(f"Wrong accumulation boundary at update={update} micro={micro}: {boundary}")
                micro_row = {"micro": micro, "accumulation_boundary": boundary, "forward": "RUNNING",
                             "backward": "NOT_STARTED", "step": "NOT_STARTED", "transfer_seconds": transfer_seconds,
                             "engine_micro_steps_before": engine.micro_steps,
                             "engine_global_steps_before": engine.global_steps}
                microsteps.append(micro_row)
                report["current_update"] = {"update": update, "microsteps": microsteps}
                if update == 0 and micro == 0:
                    report["precision_context"]["observed_global_autocast_during_forward"] = torch.is_autocast_enabled()
                if torch.is_autocast_enabled():
                    raise AssertionError("Global autocast must remain disabled")
                torch.cuda.synchronize()
                phase_started = time.monotonic()
                result = engine(**batch)
                torch.cuda.synchronize()
                micro_row.update(forward="PASS", forward_seconds=time.monotonic() - phase_started)
                if torch.is_autocast_enabled():
                    raise AssertionError("LoRA A adaptation leaked autocast outside its linear call")
                if not torch.isfinite(result.loss).all().item():
                    raise FloatingPointError(f"Nonfinite training loss at update={update} micro={micro}")
                report["finite_checks"]["loss_microbatches_checked"] += 1
                if update == 0 and micro == 0:
                    with torch.no_grad():
                        logits = result.logits[:, :-1, :].contiguous().view(-1, len(tokenizer))
                        labels = batch["labels"][:, 1:].contiguous().view(-1)
                        report["precision_context"]["observed_ce_global_autocast"] = torch.is_autocast_enabled()
                        report["precision_context"]["observed_ce_logits_dtype"] = str(logits.dtype)
                        loss08 = F.cross_entropy(logits / .8, labels)
                        loss1 = F.cross_entropy(logits, labels)
                if update == 0 and micro == 0:
                    report["temperature_loss"] = {"model": result.loss.item(), "independent_ce_temp08": loss08.item(),
                                                  "independent_ce_temp1": loss1.item(),
                                                  "exact_temp08_agreement": torch.allclose(result.loss, loss08, atol=1e-6, rtol=1e-6)}
                    if not report["temperature_loss"]["exact_temp08_agreement"] or torch.allclose(loss08, loss1, atol=1e-6, rtol=1e-6):
                        raise AssertionError("Actual LETTER temperature loss evidence failed")
                    del logits, labels, loss08, loss1
                losses.append(result.loss.detach())
                phase_started = time.monotonic()
                engine.backward(result.loss)
                torch.cuda.synchronize()
                micro_row.update(backward="PASS", backward_seconds=time.monotonic() - phase_started)
                phase_started = time.monotonic()
                micro_row["step"] = "RUNNING"
                engine.step()
                torch.cuda.synchronize()
                micro_row.update(step="PASS", optimizer_step_seconds=time.monotonic() - phase_started,
                                 engine_global_steps_after=engine.global_steps, engine_micro_steps_after=engine.micro_steps)
                expected_step = previous_global_step + int(boundary)
                if engine.global_steps != expected_step:
                    raise AssertionError("Engine global step changed outside its unique accumulation boundary")
                if engine.micro_steps != micro_row["engine_micro_steps_before"] + 1:
                    raise AssertionError("Engine microstep did not advance exactly once")
                del result, batch
            torch.cuda.synchronize()
            seconds = time.monotonic() - timer
            global_seconds = torch.tensor(seconds, device=local_rank)
            dist.all_reduce(global_seconds, op=dist.ReduceOp.MAX)
            token_counts = torch.tensor([count_tokens, padded_tokens, supervised_tokens], device=local_rank, dtype=torch.long)
            dist.all_reduce(token_counts)
            if engine.global_steps != previous_global_step + 1:
                raise AssertionError("Exactly one real optimizer update per global64 was not observed")
            progress_after = optimizer_progress()
            if progress_after["overflow"] or not progress_after["adam_step_values"]:
                raise AssertionError("Real AdamW state did not advance, or ZeRO reported gradient overflow")
            if any(step != update + 1 for step in progress_after["adam_step_values"]):
                raise AssertionError("AdamW state step differs from actual completed optimizer boundary count")
            rank_progress = [None] * world
            dist.all_gather_object(rank_progress, progress_after)
            if any(p["engine_global_steps"] != previous_global_step + 1 or
                   p["engine_micro_steps"] != engine.micro_steps or
                   set(p["adam_step_values"]) != {update + 1} or p["overflow"] for p in rank_progress):
                raise AssertionError("Optimizer boundary/state progress is not synchronized across both ranks")
            preparation_max = torch.tensor(preparation_seconds, device=local_rank)
            dist.all_reduce(preparation_max, op=dist.ReduceOp.MAX)
            row = {"update": update, "phase": "warmup" if update < 3 else "mixed_steady" if update < 6 else ("short", "middle", "long")[update - 6],
                   "rank_seconds": seconds, "max_rank_seconds": global_seconds.item(), "global_samples": 64,
                   "cpu_batch_preparation_and_mask_verification_seconds": preparation_seconds,
                   "max_rank_cpu_batch_preparation_and_mask_verification_seconds": preparation_max.item(),
                   "effective_input_tokens": int(token_counts[0]), "padded_tokens": int(token_counts[1]),
                   "padding_tokens": int(token_counts[1] - token_counts[0]),
                   "supervised_tokens": int(token_counts[2]), "rank_true_lengths": lengths,
                   "samples_per_second": 64 / global_seconds.item(), "effective_tokens_per_second": int(token_counts[0]) / global_seconds.item(),
                   "mean_loss_rank": torch.stack(losses).mean().item(), "lr_after_update": scheduler.get_last_lr(),
                   "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                   "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                   "engine_global_steps": engine.global_steps, "lr_before_update": lr_before,
                   "nonzero_learning_rate": any(lr > 0 for lr in lr_before),
                   "optimizer_progress_before": progress_before, "optimizer_progress_after": progress_after,
                   "all_rank_optimizer_progress": rank_progress, "microsteps": microsteps,
                   "rank_phase_seconds": {key: sum(m[key] for m in microsteps) for key in
                                          ("transfer_seconds", "forward_seconds", "backward_seconds", "optimizer_step_seconds")},
                   "host_resources": host_resources()}
            report["updates"].append(row)
            track_gpu_peak()
            report.pop("current_update", None)
            if update == 2:
                report["gradient_evidence"] = {n: {k: tensor_json(v) for k, v in entry.items()}
                                                for n, entry in gradient_evidence.items()}
                report["embedding_output_gradient"] = {k: v.item() if hasattr(v, "item") else v for k, v in input_gradient.items()}
                if set(gradient_evidence) != {n for n, p in named.items() if p.requires_grad}:
                    raise AssertionError("Some trainable parameters never received a gradient")
                if any(not math.isfinite(entry["max_abs"].item()) for entry in gradient_evidence.values()):
                    raise AssertionError("Some trainable parameters have nonfinite gradients")
                zero_gradients = [n for n, entry in gradient_evidence.items() if entry["max_abs"].item() == 0]
                report["gradient_zero_analysis"] = {"zero_tensor_names_after_three_updates": zero_gradients,
                    "expected_initial_behavior": "LoRA B starts at zero; LoRA A can have zero gradient before B changes. First optimizer LR is zero.",
                    "interpretation": "Every trainable tensor must receive a finite gradient. Zero aggregates are listed and checked by branch; final real LoRA A/B and SID deltas are independently required."}
                if any(all(gradient_evidence[n]["max_abs"].item() == 0 for n in lora_names if marker in n)
                       for marker in ("lora_A", "lora_B")):
                    raise AssertionError("An entire necessary LoRA A/B branch has zero gradients after nonzero-LR updates")
                if any(entry.get("observed_sid_rows_max_abs", torch.tensor(1)).item() <= 0 for entry in gradient_evidence.values()):
                    raise AssertionError("Necessary SID wrapper rows had zero gradient")
                if any(p.grad is not None for n, p in named.items() if "original_module" in n):
                    raise AssertionError("Frozen original module unexpectedly has gradient")
                if (not input_gradient["calls"] or not math.isfinite(input_gradient["max_abs"].item())
                        or input_gradient["max_abs"].item() <= 0):
                    raise AssertionError("Required embedding output/input gradient was lost or nonfinite")
            stage(f"update_{update}_complete")
        for handle in handles:
            handle.remove()
        patch_calls = report["peft05_out_of_place_compatibility"]["module_calls"]
        expected_micro_forwards = args.max_updates * args.accumulation
        if (len(patch_calls) != 224 or any(entry["lora_add_calls"] < expected_micro_forwards or
                entry["grad_enabled_calls"] < expected_micro_forwards or entry["no_grad_calls"] < expected_micro_forwards or
                entry["base_only_calls"] or entry["rank_zero_calls"] for entry in patch_calls.values())):
            raise AssertionError("Guarded out-of-place forward did not reach all 224 LoRA modules and checkpoint replays")
        report["checkpoint_replay_evidence"] = {"all_224_modules_hit": True,
            "minimum_grad_enabled_calls_per_module": min(entry["grad_enabled_calls"] for entry in patch_calls.values()),
            "minimum_no_grad_checkpoint_forward_calls_per_module": min(entry["no_grad_calls"] for entry in patch_calls.values()),
            "minimum_lora_add_calls_per_module": min(entry["lora_add_calls"] for entry in patch_calls.values()),
            "real_training_microbatches_per_rank": expected_micro_forwards,
            "interpretation": "Every int8 LoRA projection ran in no-grad checkpoint forward and grad-enabled backward recomputation; no adapter-disabled/rank-zero shortcut"}
        report["gradient_evidence"] = {n: {k: tensor_json(v) for k, v in entry.items()}
                                       for n, entry in gradient_evidence.items()}
        if set(gradient_evidence) != set(trainable_names):
            raise AssertionError("Some required trainable parameters never received a gradient")
        if any(not math.isfinite(entry["max_abs"].item()) for entry in gradient_evidence.values()):
            raise FloatingPointError("A trainable gradient was nonfinite during the nine-update probe")
        final_zero_gradients = [n for n, entry in gradient_evidence.items() if entry["max_abs"].item() == 0]
        report.setdefault("gradient_zero_analysis", {"expected_initial_behavior": "LoRA B starts at zero; first LR is zero. LoRA A can remain zero until the third update's backward."})
        report["gradient_zero_analysis"].update(zero_tensor_names_after_requested_updates=final_zero_gradients,
            completed_updates=len(report["updates"]), actual_nonzero_lr_updates=sum(row["nonzero_learning_rate"] for row in report["updates"]))
        if final_zero_gradients and args.max_updates >= 3:
            raise AssertionError(f"Unresolved all-zero gradients after {args.max_updates} updates of nonoptional dense LoRA/SID modules; inspect saved per-parameter evidence rather than assume a disconnect")
        report["embedding_output_gradient"] = {k: tensor_json(v) for k, v in input_gradient.items()}
        if (not input_gradient["calls"] or not math.isfinite(input_gradient["max_abs"].item())
                or input_gradient["max_abs"].item() <= 0):
            raise AssertionError("Required embedding output/input gradient was lost or nonfinite")
        activation_handle.remove()
        input_activation_handle.remove()
        finite_tensors, finite_elements = 0, 0
        for name, parameter in named.items():
            if parameter.requires_grad:
                if not torch.isfinite(parameter.detach()).all().item():
                    raise FloatingPointError(f"Nonfinite updated trainable parameter: {name}")
                finite_tensors += 1
                finite_elements += parameter.numel()
        report["finite_checks"].update(all_trainable_parameters_finite=True, checked_parameter_tensors=finite_tensors,
                                        checked_parameter_elements=finite_elements)
        report["activation"] = activation
        report["int8_after_updates"] = int8_evidence()
        if bnb_example.weight.dtype != torch.int8 or activation.get("lm_head_input_dtype") != "torch.bfloat16":
            raise AssertionError("Required actual int8 base weight / BF16 activation not observed")
        changes = {n: float((small_snapshot(n).float() - original.float()).abs().max()) for n, original in before.items()}
        report["small_parameter_change_max_abs"] = changes
        if not any(row["nonzero_learning_rate"] for row in report["updates"]):
            report["status"] = "PARTIAL"
            report["training_acceptance"] = "Only zero-LR formal warmup boundary observed; real nonzero parameter update is NOT CONFIRMED. Validation/save were not run."
            stage("bounded_update_allowance_exhausted_before_nonzero_lr")
            return 2
        report["parameter_changes"] = {"scope": report["parameter_snapshot_scope"], "max_abs_by_parameter": changes,
                                        "changed_lora_A_tensors": sum(v > 0 for n, v in changes.items() if "lora_A" in n),
                                        "changed_lora_B_tensors": sum(v > 0 for n, v in changes.items() if "lora_B" in n),
                                        "interpretation": "BF16 model values may round away small LoRA A changes during the first nine formal-warmup updates; at least one real LoRA change and both active SID modules must change."}
        frozen_hash_after = {n: tensor_digest(p) for n, p in named.items() if "original_module" in n}
        report["frozen_original_checks"] = {n: {"before_sha256": value, "after_sha256": frozen_hash_after[n],
                                               "all_bytes_identical": value == frozen_hash_after[n],
                                               "requires_grad": named[n].requires_grad,
                                               "in_original_optimizer_set": id(named[n]) in selected_ids}
                                             for n, value in frozen_hash_before.items()}
        if frozen_hash_before != frozen_hash_after:
            raise AssertionError("A complete frozen original embedding/head tensor changed")
        if any(value != 0 for n, value in changes.items() if "original_module" in n):
            raise AssertionError("Frozen original sampled SID rows changed")
        if not any(value > 0 for n, value in changes.items() if "lora_" in n):
            raise AssertionError("No sampled LoRA parameter changed")
        for wrapper in ("embed_tokens", "lm_head"):
            if not any(v > 0 for n, v in changes.items() if wrapper in n and "modules_to_save.default" in n):
                raise AssertionError(f"No sampled {wrapper} SID wrapper row changed")
        report["active_sid_row_changes"] = {}
        report["active_sid_row_gradients"] = {}
        for name in wrapper_names:
            used = input_sid_seen if "embed_tokens" in name else supervised_sid_seen
            row_gradient = gradient_evidence[name]["sid_row_max_abs"].detach().cpu()
            used_gradients = {str(token): row_gradient[sid_position[token]].item() for token in sorted(used)}
            report["active_sid_row_gradients"][name] = {"used_row_max_abs_gradients": used_gradients,
                "used_row_kind": "input" if "embed_tokens" in name else "supervised",
                "zero_gradient_used_rows": [token for token, value in used_gradients.items() if value == 0],
                "nonzero_gradient_used_rows": sum(value > 0 for value in used_gradients.values()),
                "unused_rows": "Not required to receive nonzero gradients; input and supervised row sets are distinct"}
            row_deltas = (small_snapshot(name).float() - before[name].float()).abs().amax(dim=1)
            used_deltas = {str(token): row_deltas[sid_position[token]].item() for token in sorted(used)}
            report["active_sid_row_changes"][name] = {"used_row_max_abs_deltas": used_deltas,
                "used_row_kind": "input" if "embed_tokens" in name else "supervised",
                "nonzero_delta_used_rows": sum(value > 0 for value in used_deltas.values()),
                "zero_delta_used_rows": [token for token, value in used_deltas.items() if value == 0],
                "note": "Zero per-row BF16 deltas are reported; a branch-wide zero delta does not pass."}
            if not any(value > 0 for value in used_deltas.values()):
                raise AssertionError(f"No actually used SID row changed in active module: {name}")
        report["observed_sid_rows"] = {"input_ids": sorted(input_sid_seen), "supervised_labels": sorted(supervised_sid_seen),
                                       "gradient_observation": "All nine updates; dynamic current-microbatch input/label sets",
                                       "parameter_delta_observation": "All added SID rows, all columns; input and supervised row IDs also listed"}
        synchronized_parts = {n: tensor_digest(named[n]) if "lora_" in n else
                              hashlib.sha256(small_snapshot(n).contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                              for n in trainable_names}
        rank_parameter_hashes = [None] * world
        dist.all_gather_object(rank_parameter_hashes, synchronized_parts)
        if any(value != rank_parameter_hashes[0] for value in rank_parameter_hashes):
            raise AssertionError("Final LoRA and active SID values differ across ZeRO-2 ranks")
        report["cross_rank_parameter_synchronization"] = {"all_equal": True,
                                                         "scope": "Every LoRA value and all added SID rows of active embedding/head",
                                                         "rank_tensor_sha256": synchronized_parts}
        # Real two-prompt validation forwards only, without selecting a checkpoint.
        # The original validation dataset and original Collator/loss are retained.
        engine.eval()
        matrix = np.load(audited / "length_rows.npz")
        valid_prompt_ids = matrix["valid_prompt_ids_seed42"]
        valid_lengths = np.take_along_axis(matrix["valid_full_lengths"], valid_prompt_ids, axis=1).reshape(-1)
        max_valid = int(np.argmax(valid_lengths))
        short_valid = int(np.argmin(np.abs(valid_lengths - np.quantile(valid_lengths, .1))))
        middle_valid = int(np.argmin(np.abs(valid_lengths - np.quantile(valid_lengths, .5))))
        if len(valid_lengths) != len(valid_data):
            raise AssertionError("Validation length table does not match original two-prompt validation dataset")
        vrng = np.random.RandomState(42)
        mixed_valid = vrng.choice(len(valid_data), 64, replace=False).tolist()
        report["validation_timings"] = []
        for vphase in ("warmup", "mixed", "short", "middle", "long"):
            guard()
            preparation_started = time.monotonic()
            representative = {"short": short_valid, "middle": middle_valid, "long": max_valid}
            indices = ([representative[vphase]] * args.micro_batch if vphase in representative else
                       mixed_valid[rank * args.micro_batch:(rank + 1) * args.micro_batch])
            pairs = [valid_data[i] for i in indices]
            batch = collator(pairs)
            for position, (index, pair) in enumerate(zip(indices, pairs)):
                input_ids = tokenizer(pair["input_ids"])["input_ids"]
                full_ids = tokenizer(pair["labels"] + tokenizer.eos_token)["input_ids"]
                expected = [-100] * len(input_ids) + full_ids[len(input_ids):]
                expected += [-100] * (batch["labels"].shape[1] - len(expected))
                if (full_ids[:len(input_ids)] != input_ids or batch["labels"][position].tolist() != expected
                        or len(full_ids) != int(valid_lengths[index])):
                    raise AssertionError("Original validation response mask or CPU-audited length differs")
            preparation_seconds = time.monotonic() - preparation_started
            tokens = int(batch["attention_mask"].sum())
            counts = torch.tensor([tokens, batch["input_ids"].numel(), int((batch["labels"] != -100).sum())],
                                  device=local_rank, dtype=torch.long)
            track_gpu_peak()
            torch.cuda.reset_peak_memory_stats()
            batch = {k: v.to(local_rank) for k, v in batch.items()}
            dist.barrier(); torch.cuda.synchronize(); vt = time.monotonic()
            with torch.no_grad():
                vout = engine(**batch)
            torch.cuda.synchronize(); duration = torch.tensor(time.monotonic()-vt, device=local_rank)
            dist.all_reduce(duration, op=dist.ReduceOp.MAX)
            dist.all_reduce(counts)
            if not torch.isfinite(vout.loss).all().item():
                raise FloatingPointError("Nonfinite real validation loss")
            report["validation_timings"].append({"phase":vphase,"max_rank_seconds":duration.item(),
                "global_samples":args.micro_batch*world,"rank_effective_tokens":tokens,
                "global_effective_tokens": int(counts[0]), "padded_tokens": int(counts[1]),
                "padding_tokens": int(counts[1] - counts[0]), "supervised_tokens": int(counts[2]),
                "samples_per_second": args.micro_batch * world / duration.item(),
                "effective_tokens_per_second": int(counts[0]) / duration.item(),
                "cpu_batch_preparation_and_mask_verification_seconds": preparation_seconds,
                "rank_rows":indices,"rank_lengths":[int(valid_lengths[i]) for i in indices],
                "loss":vout.loss.item(),"checkpoint_selection_executed":False,
                "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3})
            track_gpu_peak()
            del vout,batch
            stage(f"validation_{vphase}_complete")
        engine.train()
        if not args.skip_adapter_save:
            guard()
            dist.barrier()
            save_started = time.monotonic()
            def perform_save():
                save_dir = output / "adapter"
                model.save_pretrained(save_dir, safe_serialization=False)
                tokenizer.save_pretrained(save_dir)
                model.config.save_pretrained(save_dir)
                write_seconds = time.monotonic() - save_started
                loaded = torch.load(save_dir / "adapter_model.bin", map_location="cpu", weights_only=True)
                expected = get_peft_model_state_dict(model)
                direct_trainable = {n.replace(".modules_to_save.default", "").replace(".lora_A.default.", ".lora_A.")
                                    .replace(".lora_B.default.", ".lora_B."): named[n] for n in trainable_names}
                if set(expected) != set(direct_trainable):
                    raise AssertionError("PEFT adapter state does not contain exactly all active trainable tensors")
                if set(loaded) != set(expected):
                    raise AssertionError("Saved adapter key set differs from PEFT expected state")
                metadata = {}
                for key, value in loaded.items():
                    guard()
                    if (value.shape != expected[key].shape or value.dtype != expected[key].dtype
                            or not torch.equal(value, expected[key].detach().cpu())
                            or not torch.equal(value, direct_trainable[key].detach().cpu())):
                        raise AssertionError(f"Saved adapter differs: {key}")
                    metadata[key] = {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": tensor_digest(value)}
                saved_tokenizer = LlamaTokenizer.from_pretrained(save_dir, local_files_only=True)
                if (saved_tokenizer.get_vocab() != tokenizer.get_vocab() or
                        saved_tokenizer.pad_token_id != tokenizer.pad_token_id or
                        saved_tokenizer.convert_tokens_to_ids(train_data.get_new_tokens()) !=
                        tokenizer.convert_tokens_to_ids(train_data.get_new_tokens())):
                    raise AssertionError("Saved/reloaded tokenizer vocab, pad or ordered SID token IDs differ")
                report["adapter_save"] = {"path": str(save_dir), "seconds_rank0": time.monotonic() - save_started,
                                          "write_seconds_rank0": write_seconds,
                                          "keys": sorted(loaded), "strict_tensor_equality_to_expected_peft_state": True,
                                          "exact_direct_trainable_key_and_value_match": True,
                                          "tensor_metadata": metadata, "tokenizer_exact_vocab_sid_order_and_pad_reload": True,
                                          "checkpoint_format": "PyTorch adapter_model.bin; complete active embed/head plus LoRA only; no optimizer/full-model checkpoint",
                                          "full_model_reload_generation": "separate inference job required",
                                          "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in save_dir.iterdir() if p.is_file()}}
                del loaded, expected, direct_trainable
            save_result = [None]
            if rank == 0:
                try:
                    perform_save()
                    save_result[0] = {"status": "PASS", "path": report["adapter_save"]["path"]}
                except Exception as error:
                    save_result[0] = {"status": "FAIL", "error": {"type": type(error).__name__,
                                        "message": str(error), "traceback": traceback.format_exc()}}
            dist.broadcast_object_list(save_result, src=0)
            report["adapter_save_rank0_receipt"] = save_result[0]
            if save_result[0]["status"] != "PASS":
                raise RuntimeError("Rank0 adapter save/check failed: " + json.dumps(save_result[0]["error"]))
            dist.barrier()
        report["status"] = "PASS"
        phases = {row["phase"] for row in report["updates"]}
        report["measurement_completeness"] = {"status": "COMPLETE" if args.max_updates == 9 else "PARTIAL",
            "requested_updates": args.max_updates, "completed_updates": len(report["updates"]),
            "missing_training_phases": sorted({"warmup", "mixed_steady", "short", "middle", "long"} - phases),
            "core_training_acceptance": "Real backward, finite gradients, nonzero-LR AdamW update, exact parameter membership and active SID/LoRA deltas passed"}
        stage("probe_complete")
    except Exception as error:
        report["status"] = "PARTIAL" if isinstance(error, (TimeoutError, MemoryError)) else "FAIL"
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        if "gradient_evidence" in locals():
            try:
                report["partial_gradient_evidence"] = {n: {k: tensor_json(v) for k, v in entry.items()}
                                                        for n, entry in gradient_evidence.items()}
            except Exception as summary_error:
                report["partial_gradient_evidence_unavailable"] = str(summary_error)
        print(report["error"]["traceback"], file=sys.stderr, flush=True)
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["cpu_process_peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
        report["host_resources_final"] = host_resources()
        report["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        if torch is not None and torch.cuda.is_initialized():
            track_gpu_peak()
            report["gpu_final_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
            report["gpu_final_peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 1024**3
        write(report_path, report)
        if "dist" in locals() and dist.is_initialized():
            dist.destroy_process_group()
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
