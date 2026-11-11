"""Finite dual-rank LC-Rec six-task Qwen3 probes; never a formal epoch."""

from __future__ import annotations

import argparse
import datetime as datetime_module
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

from common import Allowed, Collator, EXPECTED_ROWS, MODELS, TARGETS, TASKS
from common import load_model, loss, sha, sharing, tokenizer_for, write


GIB = 1024 ** 3


def host_resources():
    result = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, value = line.split(":", 1)
        if name in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
            result[name + "_bytes"] = int(value.strip().split()[0]) * 1024
    for line in Path("/proc/self/status").read_text().splitlines():
        name, _, value = line.partition(":")
        if name in ("VmRSS", "VmHWM", "VmSwap"):
            result[name + "_bytes"] = int(value.strip().split()[0]) * 1024
    result["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
    return result


def record_file(path):
    path = Path(path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha(path)}


def reserve_update(path, key, route, job_id, run_commit, update_index):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    route_key = key + "-" + route
    limits = {"q06-L": 4, "q06-F": 3, "q17-L": 4, "q17-F": 3}
    with path.with_name(path.name + ".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(path.read_text()) if path.exists() else {"reservations": [], "limit": 14, "route_limits": limits}
        reservations = state["reservations"]
        if state.get("limit", 14) != 14 or state.get("route_limits", limits) != limits:
            raise RuntimeError("Shared optimizer budget limits differ from this task authorization")
        route_used = sum(entry["route_key"] == route_key for entry in reservations)
        if len(reservations) >= 14 or route_used >= limits[route_key]:
            raise RuntimeError("Shared optimizer update budget exhausted: " + route_key)
        reservation = {
            "number": len(reservations) + 1, "route_number": route_used + 1,
            "route_key": route_key, "job_id": job_id, "run_commit": run_commit,
            "index_in_job": update_index,
            "utc": datetime_module.datetime.now(datetime_module.timezone.utc).isoformat(),
            "charged_before_boundary_step": True, "includes_lr_zero_and_failed_attempts": True,
        }
        reservations.append(reservation)
        write(path, state)
        return reservation


def parse_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--samples", type=Path)
    parser.add_argument("--output", "--output-dir", required=True, type=Path, dest="output")
    parser.add_argument("--run-commit", default=os.environ.get("RUN_COMMIT", ""))
    parser.add_argument("--job-id", default=os.environ.get("GRID_JOB_ID", os.environ.get("JOB_ID", "")))
    arguments = parser.parse_args()
    if not re.fullmatch("[0-9a-f]{40}", arguments.run_commit):
        parser.error("--run-commit must be the complete fixed 40-character source commit")
    if not arguments.job_id:
        parser.error("A stable --job-id or GRID_JOB_ID is required")
    return arguments


def run(arguments, report, emit, flush):
    import numpy as numpy_module
    import torch
    import torch.distributed as distributed
    import deepspeed
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model, get_peft_model_state_dict
    from transformers import AutoTokenizer, Trainer, TrainingArguments

    config = json.loads(arguments.config.read_text())
    samples_path = arguments.samples or Path(config["audit_dir"]) / "samples.json"
    sample_document = json.loads(samples_path.read_text())
    key = config["key"]
    route = config["route"]
    model_path = Path(config["model_path"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    micro_batch = int(config["micro"])
    requested_updates = int(config.get("updates", 3))
    if key not in MODELS or route not in ("L", "F") or world_size != 2:
        raise AssertionError("Only the four authorized two-rank Qwen3 routes are accepted")
    if micro_batch not in (1, 2, 4, 8) or int(config.get("seed", 42)) != 42:
        raise AssertionError("Unapproved batch layout or random seed")
    if not 1 <= requested_updates <= (4 if route == "L" else 3):
        raise AssertionError("Per-route optimizer update allowance exceeded")
    if config.get("precision", "bf16") != "bf16":
        raise AssertionError("This entry implements BF16 routes only; Int8 requires a separately identified probe")
    if model_path.name != MODELS[key][1] or config["revision"] != MODELS[key][1]:
        raise AssertionError("Fixed model revision does not match the formal asset path")
    for name, expected in {"global_batch": 128, "epochs": 4, "learning_rate": 5e-5, "weight_decay": .01, "warmup_ratio": .01}.items():
        if name in config and config[name] != expected:
            raise AssertionError("The effective fixed technical recipe differs from its configuration: " + name)
    if "lora" in config:
        for name, expected in {"r": 8, "lora_alpha": 32, "lora_dropout": .05, "ensure_weight_tying": True}.items():
            if config["lora"].get(name) != expected:
                raise AssertionError("The fixed LoRA candidate differs from its configuration: " + name)
        if set(config["lora"]["target_modules"]) != set(TARGETS) or set(config["lora"]["modules_to_save"]) != {"embed_tokens", "lm_head"}:
            raise AssertionError("The fixed LoRA module scope differs from its configuration")
    if "generation" in config:
        generation_config = config["generation"]
        if any(generation_config.get(name) != value for name, value in {"num_beams": 20, "num_return_sequences": 20, "max_new_tokens": 10, "filter_items": False}.items()) or generation_config.get("prompt_ids") != list(range(17)):
            raise AssertionError("The generation configuration differs from the original LC-Rec17-prompt beam20 protocol")
    accumulation = 128 // (world_size * micro_batch)
    start = time.monotonic()
    torch.cuda.set_device(local_rank)
    if not distributed.is_initialized():
        distributed.init_process_group("nccl", timeout=datetime_module.timedelta(seconds=180))
    device = torch.device("cuda", local_rank)
    random.seed(42)
    numpy_module.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    output_directory = arguments.output

    def synchronize():
        torch.cuda.synchronize(device)

    def gather(value):
        values = [None] * world_size
        distributed.all_gather_object(values, value)
        return values

    def max_rank_seconds(value):
        tensor = torch.tensor(value, dtype=torch.float64, device=device)
        distributed.all_reduce(tensor, op=distributed.ReduceOp.MAX)
        return float(tensor)

    def sum_counts(counts):
        names = list(counts)
        tensor = torch.tensor([counts[name] for name in names], dtype=torch.int64, device=device)
        distributed.all_reduce(tensor)
        return dict(zip(names, tensor.tolist()))

    def memory():
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        return {
            "allocated_gib": torch.cuda.memory_allocated(device) / GIB,
            "reserved_gib": torch.cuda.memory_reserved(device) / GIB,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / GIB,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / GIB,
            "cuda_free_gib": free_bytes / GIB, "cuda_total_gib": total_bytes / GIB,
            **host_resources(),
        }

    def guard():
        resources = host_resources()
        if resources["MemAvailable_bytes"] < 40 * GIB:
            raise RuntimeError("Safety stop: host MemAvailable fell below40 GiB")
        for directory in {str(output_directory.resolve()), str(model_path.resolve())}:
            disk = shutil.disk_usage(directory)
            if disk.free < max(30 * GIB, disk.total * .10):
                raise RuntimeError("Safety stop: filesystem free capacity below max(30 GiB,10%): " + directory)
        if time.monotonic() - start > min(2700, float(config.get("job_max_seconds", 2700))):
            raise TimeoutError("The single-job GPU wallclock allowance has expired")

    def tensor_digest(tensor):
        host_tensor = tensor.detach().contiguous().cpu()
        return hashlib.sha256(host_tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()

    report.update(
        key=key, route=route, route_key=key + "-" + route, precision="bf16",
        rank=rank, local_rank=local_rank, world_size=world_size,
        job_id=arguments.job_id, run_commit=arguments.run_commit,
        config=config, config_identity=record_file(arguments.config), samples_identity=record_file(samples_path),
        technical_only=True, full_epoch_executed=False,
        environment={
            "interpreter": sys.executable, "python": sys.version, "hostname": socket.gethostname(),
            "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "accelerate", "deepspeed")},
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(local_rank),
            "gpu_total_bytes": torch.cuda.get_device_properties(local_rank).total_memory,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "allocator_environment": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        },
        training_definition={
            "micro": micro_batch, "accumulation": accumulation, "global_batch": 128,
            "seed": 42, "learning_rate": 5e-5, "weight_decay": .01,
            "scheduler": "cosine", "warmup_ratio": .01, "gradient_checkpointing": True,
            "epochs_for_estimate_only": 4, "temperature": 1.,
            "loss_aggregation": "Original LC-Rec shifted-token mean per rank microbatch, divided by accumulation and averaged across ranks by ZeRO-2",
            "new_trainer_global_token_normalization_used": False,
        },
        engineering_differences=[
            "Qwen3 Base replaces the original LLaMA downstream causal LM; natural-language six-task prompts and supervision are retained",
            "Complete base weights load as BF16; initialization/casting differs from the original unspecified-FP32-load/autocast path",
            "Qwen PAD may equal EOS; attention_mask identifies padding and preserves the real supervised EOS",
            "Native Qwen tied embedding/head remains one optimized active Parameter; PEFT ensure_weight_tying preserves this sharing",
            "Saved finite engine loop replaces Trainer.train, retaining original per-micro CE aggregation and using the actual installed Trainer sampler/scheduler",
            "Two RTX4090 GPUs with ZeRO-2 GPU FP32 masters and AdamW moments; no CPU offload or quantization",
        ],
    )
    emit("environment_and_budget_checked")
    guard()
    asset_manifest_path = model_path / "ASSET_MANIFEST.json"
    manifest = json.loads(asset_manifest_path.read_text())
    if manifest["revision"] != MODELS[key][1] or not manifest["model_id"].endswith(MODELS[key][0]):
        raise AssertionError("The pinned formal model manifest has an unexpected identity")
    for entry in manifest["files"]:
        if (model_path / entry["path"]).stat().st_size != entry["size"]:
            raise AssertionError("Model asset size differs from its audited identity: " + entry["path"])
    report["formal_asset"] = {
        "model_id": manifest["model_id"], "revision": manifest["revision"],
        "manifest": record_file(asset_manifest_path), "all_manifest_file_sizes_match": True,
        "large_weight_hash": "Reuse prior complete audited SHA-256 receipt; this probe does not repeatedly hash unchanged base weights",
    }
    tokenizer, token_identity = tokenizer_for(model_path, sample_document["sid_tokens"])
    sid_ids = token_identity["sid_ids"]
    sid_id_set = set(sid_ids)
    if len(sid_ids) != 930 or len(sample_document["indices"]) != 9922:
        raise AssertionError("The fixed LC-Rec Instruments SID map is incomplete")
    collator = Collator(tokenizer)
    records = sample_document["records"]
    plans = sample_document["plans"]
    report["tokenizer"] = token_identity
    report["data_identity"] = sample_document.get("identity", sample_document.get("assets", {}))
    report["auxiliary_asset_boundary"] = {
        "train_mismatch_records": 12233, "test_mismatch_records": 7473,
        "itemsearch_directly_affected": True, "preference_label_context_fully_proven": False,
        "technical_updates": "Only CPU-audited eligible training-period paired records",
        "uncertain_rows": "Diagnostic computation only; not evidence for recommendation quality",
        "formal_six_task_reproduction": "BLOCKED_BY_PREEXISTING_AUXILIARY_ASSET_MISMATCH",
    }
    replayed = set()

    def cpu_batch(sample_ids, require_eligible=False):
        rows = []
        for sample_id in sample_ids:
            record = records[sample_id]
            if require_eligible and not record.get("technical_update_eligible", False):
                raise AssertionError("An unaudited or mismatched auxiliary record entered a technical optimizer update: " + sample_id)
            row = {"input_ids": record["input_ids"], "labels": record["labels"]}
            if sample_id not in replayed:
                actual = collator([row])
                if any(actual[name][0].tolist() != record["collated"][name] for name in ("input_ids", "labels", "attention_mask")):
                    raise AssertionError("GPU-process LC-Rec tokenizer/collator replay differs from the fixed CPU record: " + sample_id)
                replayed.add(sample_id)
            rows.append(row)
        batch = collator(rows)
        counts = {
            "samples": len(rows), "input_tokens": int(batch["attention_mask"].sum()),
            "padded_tokens": int(batch["input_ids"].numel()),
            "label_tokens": int((batch["labels"] != -100).sum()),
            "shift_label_tokens": int((batch["labels"][:, 1:] != -100).sum()),
        }
        if counts["shift_label_tokens"] == 0 or int(batch["input_ids"].max()) >= len(tokenizer):
            raise AssertionError("Invalid empty supervision or out-of-range vocabulary index")
        return batch, counts

    emit("load_model")
    load_started = time.monotonic()
    model, load_information = load_model(model_path)
    report["load_information"] = load_information
    report["model_identity"] = {
        "class": type(model).__module__ + "." + type(model).__name__,
        "source": record_file(inspect.getfile(type(model))), "config": model.config.to_dict(),
        "original_unique_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    report["sharing_before_resize"] = sharing(model)
    if not sharing(model)["same_parameter_object"]:
        raise AssertionError("Official Qwen3 input/head weights are unexpectedly untied")
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    report["sharing_after_resize"] = sharing(model)
    if not sharing(model)["same_parameter_object"] or model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        raise AssertionError("Embedding resize changed Qwen weight tying or physical vocabulary row count")
    expanded_base_parameters = sum(parameter.numel() for parameter in model.parameters())
    if route == "L":
        target_names = [name for name, module in model.named_modules() if name.rsplit(".", 1)[-1] in TARGETS]
        if len(target_names) != int(model.config.num_hidden_layers) * 7:
            raise AssertionError("Expected all seven native projection modules in every Transformer layer")
        lora_config = LoraConfig(
            r=8, lora_alpha=32, lora_dropout=.05, target_modules=TARGETS,
            modules_to_save=["embed_tokens", "lm_head"], ensure_weight_tying=True,
            bias="none", task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        for name, parameter in model.named_parameters():
            if ".original_module." in name:
                parameter.requires_grad = False
        report["lora"] = {
            "config": json.loads(json.dumps(model.peft_config["default"].to_dict(), default=lambda value: sorted(value) if isinstance(value, set) else str(value))),
            "target_names": target_names, "full_active_embedding_head_saved": True,
            "official_lora_entry_default_lr": "Read separately from fixed official lora_finetune.py;5e-5 is this approved technical candidate, not a proven LoRA optimum",
        }
    else:
        for parameter in model.parameters():
            parameter.requires_grad = True
        if not all(parameter.requires_grad for parameter in model.parameters()):
            raise AssertionError("The full-parameter route contains unexpectedly frozen parameters")
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(device)
    synchronize()
    report["model_load_resize_wrap_seconds"] = max_rank_seconds(time.monotonic() - load_started)
    report["sharing_after_wrap"] = sharing(model)
    if not sharing(model)["same_parameter_object"]:
        raise AssertionError("PEFT wrapping lost the active tied embedding/head parameter")
    model_signatures = gather({"config": model.config.to_dict(), "tokenizer": token_identity})
    report["both_rank_model_tokenizer_identity_equal"] = model_signatures[0] == model_signatures[1]
    if not report["both_rank_model_tokenizer_identity_equal"]:
        raise AssertionError("Model or tokenizer identity differs between ranks")
    named_parameters = dict(model.named_parameters())
    trainable = {name: parameter for name, parameter in named_parameters.items() if parameter.requires_grad}
    frozen = {name: parameter for name, parameter in named_parameters.items() if not parameter.requires_grad}
    embedding = model.get_input_embeddings().weight
    embedding_name = next(name for name, parameter in trainable.items() if parameter is embedding)
    report["parameter_inventory"] = {
        name: {"shape": list(parameter.shape), "numel": parameter.numel(), "dtype_before_ds": str(parameter.dtype), "requires_grad": parameter.requires_grad}
        for name, parameter in named_parameters.items()
    }
    report["parameter_scope"] = {
        "expanded_base_parameters": expanded_base_parameters,
        "allocated_unique_parameters_including_frozen_peft_originals": sum(parameter.numel() for parameter in named_parameters.values()),
        "trainable_unique_parameters": sum(parameter.numel() for parameter in trainable.values()),
        "trainable_tensors": len(trainable), "frozen_tensors": len(frozen),
        "trainable_fraction_of_expanded_base": sum(parameter.numel() for parameter in trainable.values()) / expanded_base_parameters,
        "active_tied_embedding_name": embedding_name,
        "active_embedding_modules": {name: type(module).__module__ + "." + type(module).__name__ for name, module in model.named_modules() if name.endswith(("embed_tokens", "lm_head", "original_module", "modules_to_save.default"))},
    }
    report["initial_model_memory"] = memory()
    report["initial_forward_long_tail"] = []
    model.eval()
    for task in TASKS:
        guard()
        sample_id = plans["task_representatives"][task][-1]
        trial = {"task": task, "sample_id": sample_id, "micro": 1,
                 "stage": "Before optimizer/ZeRO state initialization", "gradients": False,
                 "distinct_records_across_ranks": 1, "compute_presentations_across_ranks": 2,
                 "diagnostic_only": True}
        result = batch = None
        try:
            cpu, counts = cpu_batch([sample_id])
            batch = {name: tensor.to(device) for name, tensor in cpu.items()}
            torch.cuda.reset_peak_memory_stats(device)
            synchronize()
            phase_started = time.monotonic()
            with torch.no_grad():
                result = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
            synchronize()
            trial.update(status="PASS", forward_seconds=time.monotonic() - phase_started,
                         counts_per_rank=counts, input_shape=list(batch["input_ids"].shape))
        except Exception as error:
            trial.update(status="OOM" if isinstance(error, torch.cuda.OutOfMemoryError) else "FAIL",
                         error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        finally:
            trial["memory"] = memory()
            del result, batch
            torch.cuda.empty_cache()
        trial["all_rank_status"] = gather({"status": trial["status"], "memory": trial["memory"]})
        report["initial_forward_long_tail"].append(trial)
        emit("initial_model_long_tail_forward_complete", task=task, status=trial["status"])

    class LengthOnlyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return sum(EXPECTED_ROWS)

        def __getitem__(self, index):
            raise RuntimeError("A full epoch is not authorized; only inspect this dataset's actual sampler indices")

    training_arguments = TrainingArguments(
        output_dir=str(output_directory / "unused_trainer"), seed=42,
        per_device_train_batch_size=micro_batch, per_device_eval_batch_size=micro_batch,
        gradient_accumulation_steps=accumulation, num_train_epochs=4,
        learning_rate=5e-5, weight_decay=.01, warmup_ratio=.01, lr_scheduler_type="cosine",
        bf16=True, optim="adamw_torch", max_grad_norm=1., gradient_checkpointing=True,
        report_to=[], local_rank=local_rank, ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=training_arguments, train_dataset=LengthOnlyDataset(), data_collator=collator)
    sampler_rng = (torch.get_rng_state(), torch.cuda.get_rng_state(device), random.getstate(), numpy_module.random.get_state())
    loader = trainer.get_train_dataloader()
    if not hasattr(trainer, "set_initial_training_values"):
        raise RuntimeError("Installed Trainer lacks the source-verified schedule inspection API")
    initial_training_values = trainer.set_initial_training_values(training_arguments, loader, 128)
    formal_steps = int(initial_training_values[-1])
    if config.get("scheduled_steps") is not None and int(config["scheduled_steps"]) != formal_steps:
        raise AssertionError("Configured formal update count differs from the actual pinned Trainer calculation")
    sampler_indices = [int(index) for batch_indices in loader.batch_sampler for index in batch_indices]
    all_sampler_indices = gather(sampler_indices)
    flattened_indices = [index for rank_indices in all_sampler_indices for index in rank_indices]
    sampler_attributes = {name: getattr(loader.batch_sampler, name, None) for name in ("batch_size", "drop_last", "even_batches", "split_batches", "num_processes", "process_index")}
    report["formal_loader"] = {
        "dataset_length": sum(EXPECTED_ROWS), "dataset_wrapper": "Length-only wrapper; no raw record is loaded or trained during sampler inspection",
        "class": type(loader).__module__ + "." + type(loader).__name__, "length_per_rank": len(loader),
        "batch_sampler_class": type(loader.batch_sampler).__module__ + "." + type(loader.batch_sampler).__name__,
        "batch_sampler_attributes": sampler_attributes,
        "trainer_initial_training_values": list(initial_training_values),
        "set_initial_training_values_source_sha256": hashlib.sha256(inspect.getsource(trainer.set_initial_training_values).encode()).hexdigest(),
        "trainer_source": record_file(inspect.getfile(Trainer)),
        "formal_steps": formal_steps, "warmup_steps": training_arguments.get_warmup_steps(formal_steps),
        "epoch_sampler_presentations_across_ranks": len(flattened_indices),
        "epoch_distinct_records": len(set(flattened_indices)),
        "epoch_duplicate_presentations": len(flattened_indices) - len(set(flattened_indices)),
        "per_rank_presentations": [len(rank_indices) for rank_indices in all_sampler_indices],
        "rank_sampler_order_sha256": hashlib.sha256(json.dumps(sampler_indices, separators=(",", ":")).encode()).hexdigest(),
        "epoch_tail_microsteps_per_rank": len(loader) % accumulation,
        "last_epoch_update_effective_presentations": (len(loader) % accumulation or accumulation) * micro_batch * world_size,
        "full_dataloader_iteration_executed": False,
    }
    if len(set(flattened_indices)) != sum(EXPECTED_ROWS):
        raise AssertionError("Actual formal sampler omits original six-task records")
    del loader, sampler_indices, all_sampler_indices, flattened_indices
    torch.set_rng_state(sampler_rng[0])
    torch.cuda.set_rng_state(sampler_rng[1], device)
    random.setstate(sampler_rng[2])
    numpy_module.random.set_state(sampler_rng[3])
    trainer.create_optimizer()
    optimizer = trainer.optimizer
    trainer.create_scheduler(num_training_steps=formal_steps, optimizer=optimizer)
    scheduler = trainer.lr_scheduler
    optimizer_parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    trainable_ids = {id(parameter) for parameter in trainable.values()}
    if len(optimizer_parameters) != len({id(parameter) for parameter in optimizer_parameters}) or {id(parameter) for parameter in optimizer_parameters} != trainable_ids:
        raise AssertionError("AdamW membership contains missing, extra, or duplicate shared parameters")
    names_by_id = {id(parameter): name for name, parameter in named_parameters.items()}
    report["optimizer"] = {
        "class": type(optimizer).__module__ + "." + type(optimizer).__name__,
        "exact_unique_trainable_membership": True, "active_shared_parameter_occurrences": sum(parameter is embedding for parameter in optimizer_parameters),
        "groups": [{"weight_decay": group["weight_decay"], "names": [names_by_id[id(parameter)] for parameter in group["params"]]} for group in optimizer.param_groups],
        "cpu_offload": False,
    }
    deepspeed_config = {
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 2, "overlap_comm": False, "contiguous_gradients": True,
            "reduce_scatter": True, "allgather_partitions": True,
            "reduce_bucket_size": 5000000, "allgather_bucket_size": 5000000,
        },
        "gradient_accumulation_steps": accumulation, "gradient_clipping": 1.,
        "train_batch_size": 128, "train_micro_batch_size_per_gpu": micro_batch,
        "steps_per_print": 100000, "wall_clock_breakdown": False,
        "zero_allow_untested_optimizer": True, "flops_profiler": {"enabled": False},
    }
    if "zero" in config:
        requested_zero = config["zero"]
        if requested_zero.get("stage", 2) != 2 or any(requested_zero.get(name, {}).get("device", "none") != "none" for name in ("offload_optimizer", "offload_param")):
            raise AssertionError("Only GPU ZeRO-2 without offload is authorized")
        deepspeed_config["zero_optimization"].update(requested_zero)
    report["deepspeed_config"] = deepspeed_config
    emit("deepspeed_initialize")
    guard()
    init_started = time.monotonic()
    engine, _, _, scheduler = deepspeed.initialize(
        model=model, optimizer=optimizer, lr_scheduler=scheduler,
        model_parameters=list(trainable.values()), config=deepspeed_config, dist_init_required=False,
    )
    synchronize()
    report["deepspeed_initialization_seconds"] = max_rank_seconds(time.monotonic() - init_started)
    original_zero_parameters = [parameter for group in engine.optimizer.bit16_groups for parameter in group]
    if len(original_zero_parameters) != len(trainable_ids) or {id(parameter) for parameter in original_zero_parameters} != trainable_ids:
        raise AssertionError("ZeRO-2 original parameter membership is not exact and unique")
    masters = engine.optimizer.single_partition_of_fp32_groups
    if not masters or any(parameter.device.type != "cuda" or parameter.dtype != torch.float32 for parameter in masters):
        raise AssertionError("FP32 master partitions must reside on GPU")
    report["optimizer"]["zero_original_membership_exact"] = True
    report["optimizer"]["fp32_master_partitions"] = [{"shape": list(parameter.shape), "dtype": str(parameter.dtype), "device": str(parameter.device)} for parameter in masters]
    report["optimizer"]["actual_gradient_accumulation_dtype"] = str(getattr(engine.optimizer, "gradient_accumulation_dtype", None))
    trainable_count = sum(parameter.numel() for parameter in trainable.values())
    report["checkpoint_static"] = {
        "trainable_unique_parameters": trainable_count,
        "model_only_unique_bf16_bytes": expanded_base_parameters * 2,
        "fp32_master_plus_two_adam_moments_total_bytes": trainable_count * 12,
        "model_state_dict_logical_tensor_bytes_including_tied_keys": sum(tensor.numel() * tensor.element_size() for tensor in model.state_dict().values()),
        "excludes": ["Serialization metadata", "Scheduler/RNG state", "Per-rank model-state duplication if emitted by the current DeepSpeed checkpoint implementation"],
    }
    report["sharing_after_zero"] = sharing(model)
    if not sharing(model)["same_parameter_object"]:
        raise AssertionError("ZeRO setup lost the active shared embedding/head parameter")
    for name, parameter in named_parameters.items():
        report["parameter_inventory"][name]["dtype_after_ds"] = str(parameter.dtype)
    del trainer, optimizer_parameters, original_zero_parameters, masters

    lora_parameter_ids = {id(parameter) for name, parameter in trainable.items() if "lora_" in name}

    def snapshot(parameter):
        if parameter is embedding:
            return parameter.detach()[sid_ids].float().cpu().clone()
        if parameter.numel() <= 32768 or id(parameter) in lora_parameter_ids:
            return parameter.detach().float().cpu().clone()
        flat_parameter = parameter.detach().reshape(-1)
        selected = torch.linspace(0, flat_parameter.numel() - 1, 8192, device=flat_parameter.device).long()
        return flat_parameter[selected].float().cpu().clone()

    monitored = {
        name: parameter for name, parameter in trainable.items()
        if "lora_" in name or parameter is embedding or name.endswith(("q_proj.weight", "up_proj.weight", "input_layernorm.weight", "norm.weight"))
    }
    baseline = {name: snapshot(parameter) for name, parameter in monitored.items()}
    frozen_before = {name: tensor_digest(parameter) for name, parameter in frozen.items()}
    initial_parameter_evidence = {"frozen_tensor_hashes": frozen_before, "monitored_trainable_snapshot_hashes": {name: tensor_digest(value) for name, value in baseline.items()}}
    initial_parameter_evidence_ranks = gather(initial_parameter_evidence)
    report["initial_rank_parameter_identity"] = {"equal": initial_parameter_evidence_ranks[0] == initial_parameter_evidence_ranks[1], "scope": "All complete frozen tensors, all930 active SID rows, all complete LoRA tensors, representative full-parameter tensor samples"}
    if not report["initial_rank_parameter_identity"]["equal"]:
        raise AssertionError("Pre-update parameter evidence differs between GPU ranks")
    report["frozen_baseline_hashes"] = frozen_before
    report["update_baseline"] = "After ZeRO initialization and dtype flattening; pre-first optimizer update"
    gradient_observations = {name: {"calls": 0, "nonzero_calls": 0, "finite": True, "max_abs": 0.} for name in trainable}
    activation_observations = {"input_embedding_backward_calls": 0, "output_logits_backward_calls": 0}
    hooks = []
    sid_tensor = torch.tensor(sid_ids, device=device)
    current_gradient_scope = {"update": -1}

    for name, parameter in trainable.items():
        def observe_gradient(gradient, parameter_name=name, current_parameter=parameter):
            observation = gradient_observations[parameter_name]
            maximum = torch.linalg.vector_norm(gradient.detach(), ord=float("inf"), dtype=torch.float32)
            finite = bool(torch.isfinite(maximum))
            maximum_value = float(maximum)
            observation["calls"] += 1
            observation["nonzero_calls"] += int(maximum_value > 0)
            observation["finite"] = observation["finite"] and finite
            if math.isfinite(maximum_value):
                observation["max_abs"] = max(observation["max_abs"], maximum_value)
            if maximum_value > 0:
                observation.setdefault("first_nonzero_update", current_gradient_scope["update"])
            if current_parameter is embedding:
                observation["sid_rows_max_abs"] = max(observation.get("sid_rows_max_abs", 0.), float(gradient[sid_ids].abs().max()))
                observation["sid_rows_finite"] = bool(torch.isfinite(gradient[sid_ids]).all())
            if not finite:
                raise FloatingPointError("Nonfinite trainable gradient: " + parameter_name)
            return gradient
        hooks.append(parameter.register_hook(observe_gradient))

    def observe_embedding(module, input_values, output_values):
        if output_values.requires_grad:
            positions = torch.isin(input_values[0], sid_tensor)
            def observe_activation(gradient):
                activation_observations["input_embedding_backward_calls"] += 1
                if bool(positions.any()):
                    activation_observations["actual_input_sid_position_max_abs"] = max(activation_observations.get("actual_input_sid_position_max_abs", 0.), float(gradient[positions].abs().max()))
                return gradient
            output_values.register_hook(observe_activation)
    hooks.append(model.get_input_embeddings().register_forward_hook(observe_embedding))

    def optimizer_progress():
        inner_optimizer = engine.optimizer.optimizer
        steps = [int(state["step"]) for state in inner_optimizer.state.values() if "step" in state]
        return {
            "engine_global_steps": int(engine.global_steps), "engine_micro_steps": int(engine.micro_steps),
            "adam_steps": sorted(set(steps)), "state_entries": len(inner_optimizer.state),
            "overflow": bool(getattr(engine.optimizer, "overflow", False)),
        }

    def optimizer_state_inventory():
        inventory = []
        for state in engine.optimizer.optimizer.state.values():
            for name, value in state.items():
                if torch.is_tensor(value):
                    inventory.append({"state": name, "shape": list(value.shape), "numel": value.numel(), "dtype": str(value.dtype), "device": str(value.device)})
                    if name in ("exp_avg", "exp_avg_sq") and (value.device.type != "cuda" or value.dtype != torch.float32):
                        raise AssertionError("AdamW FP32 moment tensors are not resident on GPU")
        return inventory

    def calculate_loss(logits, labels, verify=False):
        value = loss(logits, labels)
        if not bool(torch.isfinite(value)):
            raise FloatingPointError("Nonfinite original LC-Rec CE")
        if verify:
            with torch.no_grad():
                targets = labels[:, 1:]
                mask = targets != -100
                valid_logits = logits.detach()[:, :-1, :][mask].float()
                independent = (torch.logsumexp(valid_logits, dim=-1) - valid_logits.gather(1, targets[mask].unsqueeze(1)).squeeze(1)).mean()
                difference = float((value.detach() - independent).abs())
                if difference > 2e-5:
                    raise AssertionError("LC-Rec causal CE differs from independent selected-token logsumexp reference")
                report["independent_ce"] = {"temperature": 1., "abs_diff": difference, "shifted_valid_labels": int(mask.sum()), "status": "PASS"}
        return value

    plan_names = ["mixed_global128_0", "mixed_global128_1", "mixed_global128_0", "mixed_global128_tail"] if route == "L" else ["mixed_global128_0", "mixed_global128_1", "mixed_global128_tail"]
    ledger = config.get("update_ledger", "/home/bwwu/lc-rec/qwen-six-task-20261111/update-budget.json")
    report["update_ledger_path"] = ledger
    report["updates"] = []
    engine.train()
    emit("bounded_training_begin")
    for update_index in range(requested_updates):
        guard()
        plan_name = plan_names[update_index]
        sample_ids = plans[plan_name]
        if len(sample_ids) != 128 or {records[sample_id]["task"] for sample_id in sample_ids} != set(TASKS):
            raise AssertionError("Each real global128 update must contain all six original tasks")
        current_gradient_scope["update"] = update_index
        synchronize()
        distributed.barrier()
        torch.cuda.reset_peak_memory_stats(device)
        update_started = time.monotonic()
        update_row = {
            "index": update_index, "kind": "mixed_tail" if "tail" in plan_name else "mixed",
            "plan": plan_name, "status": "RUNNING", "microsteps": [],
            "lr_before": [float(group["lr"]) for group in optimizer.param_groups],
            "progress_before": optimizer_progress(), "sample_ids": sample_ids,
            "distinct_records": len({records[sample_id]["concat_index"] for sample_id in sample_ids}),
            "compute_presentations": 128,
            "task_counts": {task: sum(records[sample_id]["task"] == task for sample_id in sample_ids) for task in TASKS},
            "rank_counts": {name: 0 for name in ("samples", "input_tokens", "padded_tokens", "label_tokens", "shift_label_tokens")},
            "optimizer_states_resident_at_start": bool(engine.optimizer.optimizer.state),
            "timing_scope": "Instrumented real ZeRO update wallclock including gradient guards, memory/event sampling and CPU collation; no instrumentation overhead is subtracted",
            "zero_communication": "Included in synchronized backward and engine.step measurements; standalone NCCL duration is NOT ISOLATED",
        }
        report["current_update"] = update_row
        previous_global_steps = int(engine.global_steps)
        for micro_index in range(accumulation):
            guard()
            begin = (micro_index * world_size + rank) * micro_batch
            local_ids = sample_ids[begin:begin + micro_batch]
            phase_started = time.monotonic()
            cpu, counts = cpu_batch(local_ids, require_eligible=True)
            collator_seconds = time.monotonic() - phase_started
            for name in counts:
                update_row["rank_counts"][name] += counts[name]
            synchronize()
            phase_started = time.monotonic()
            batch = {name: tensor.to(device) for name, tensor in cpu.items()}
            synchronize()
            transfer_seconds = time.monotonic() - phase_started
            phase_started = time.monotonic()
            result = engine(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
            logits = result.logits
            def observe_output_gradient(gradient):
                activation_observations["output_logits_backward_calls"] += 1
                targets = batch["labels"][:, 1:]
                positions = torch.isin(targets, sid_tensor)
                selected = gradient[:, :-1, :].gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)[positions]
                if selected.numel():
                    activation_observations["actual_supervised_sid_logit_max_abs"] = max(activation_observations.get("actual_supervised_sid_logit_max_abs", 0.), float(selected.abs().max()))
                return gradient
            logits.register_hook(observe_output_gradient)
            value = calculate_loss(logits, batch["labels"], verify=update_index == 0 and micro_index == 0)
            synchronize()
            forward_seconds = time.monotonic() - phase_started
            phase_started = time.monotonic()
            engine.backward(value)
            synchronize()
            backward_seconds = time.monotonic() - phase_started
            boundary = bool(engine.is_gradient_accumulation_boundary())
            if boundary != (micro_index == accumulation - 1):
                raise AssertionError("The actual ZeRO accumulation boundary differs from the global128 plan")
            if boundary:
                decision = [None]
                if rank == 0:
                    try:
                        decision[0] = {"reservation": reserve_update(ledger, key, route, arguments.job_id, arguments.run_commit, update_index)}
                    except Exception as error:
                        decision[0] = {"error": str(error)}
                distributed.broadcast_object_list(decision, src=0)
                if "error" in decision[0]:
                    raise RuntimeError(decision[0]["error"])
                update_row["budget_reservation"] = decision[0]["reservation"]
                flush()
            phase_started = time.monotonic()
            engine.step()
            synchronize()
            step_seconds = time.monotonic() - phase_started
            if int(engine.global_steps) != previous_global_steps + int(boundary):
                raise AssertionError("The engine performed an unexpected number of global optimizer updates")
            micro_row = {
                "micro": micro_index, "sample_ids": local_ids, "counts": counts,
                "input_shape": list(batch["input_ids"].shape), "loss": float(value.detach()),
                "cpu_collator_seconds": collator_seconds, "transfer_seconds": transfer_seconds,
                "forward_seconds": forward_seconds, "backward_seconds": backward_seconds,
                "engine_step_seconds": step_seconds, "boundary": boundary,
                "global_steps_after": int(engine.global_steps), "memory": memory(),
            }
            update_row["microsteps"].append(micro_row)
            emit("training_microstep", flush_report=False, update=update_index, **micro_row)
            del batch, cpu, logits, result, value
        synchronize()
        seconds = max_rank_seconds(time.monotonic() - update_started)
        update_row["global_counts"] = sum_counts(update_row["rank_counts"])
        update_row.update(
            seconds=seconds, samples_per_second=128 / seconds,
            effective_tokens_per_second=update_row["global_counts"]["input_tokens"] / seconds,
            supervised_tokens_per_second=update_row["global_counts"]["shift_label_tokens"] / seconds,
            padding_ratio=1 - update_row["global_counts"]["input_tokens"] / update_row["global_counts"]["padded_tokens"],
            nonzero_lr=any(value > 0 for value in update_row["lr_before"]),
            lr_after=[float(group["lr"]) for group in optimizer.param_groups],
            progress_after=optimizer_progress(), memory=memory(), status="PASS",
        )
        all_progress = gather(update_row["progress_after"])
        update_row["all_rank_progress"] = all_progress
        if any(progress["engine_global_steps"] != update_index + 1 or progress["adam_steps"] != [update_index + 1] or progress["overflow"] for progress in all_progress):
            raise AssertionError("The real GPU AdamW steps are not complete and synchronized between ranks")
        update_row["phase_seconds"] = {
            name: sum(micro[name] for micro in update_row["microsteps"])
            for name in ("cpu_collator_seconds", "transfer_seconds", "forward_seconds", "backward_seconds", "engine_step_seconds")
        }
        valid_counts = gather([micro["counts"]["shift_label_tokens"] for micro in update_row["microsteps"]])
        relative_weights = [update_row["global_counts"]["shift_label_tokens"] / (world_size * accumulation * count) for rank_counts in valid_counts for count in rank_counts]
        update_row["original_micro_loss_weighting"] = {
            "per_rank_micro_shift_label_counts": valid_counts,
            "relative_to_global_token_mean_min": min(relative_weights),
            "relative_to_global_token_mean_max": max(relative_weights),
            "formula": "global_valid_tokens/(world*accumulation*micro_valid_tokens)",
            "original_micro_mean_semantics_preserved": True,
        }
        update_row["actual_optimizer_state_tensors"] = optimizer_state_inventory()
        report["updates"].append(update_row)
        report.pop("current_update", None)
        emit("global_optimizer_update_complete", update=update_index, nonzero_lr=update_row["nonzero_lr"], seconds=seconds, samples_per_second=128 / seconds)

    report["gradient_observations"] = gradient_observations
    report["activation_gradients"] = activation_observations
    report["parameter_changes"] = {}
    for name, parameter in monitored.items():
        difference = (snapshot(parameter) - baseline[name]).abs()
        report["parameter_changes"][name] = {"sampled_elements": baseline[name].numel(), "changed_elements": int((difference != 0).sum()), "max_abs_change": float(difference.max())}
    report["frozen_parameters"] = {
        name: {"before_sha256": frozen_before[name], "after_sha256": tensor_digest(parameter)}
        for name, parameter in frozen.items()
    }
    if any(entry["before_sha256"] != entry["after_sha256"] for entry in report["frozen_parameters"].values()):
        raise AssertionError("A frozen original parameter changed during the optimizer updates")
    if not all(observation["calls"] > 0 and observation["finite"] for observation in gradient_observations.values()):
        raise AssertionError("Some expected trainable parameter has missing or nonfinite gradients")
    if not any(update["nonzero_lr"] for update in report["updates"]):
        raise AssertionError("No genuine nonzero-learning-rate optimizer update was completed")
    if report["parameter_changes"][embedding_name]["changed_elements"] == 0:
        raise AssertionError("No required SID embedding/head row actually changed")
    if activation_observations.get("actual_input_sid_position_max_abs", 0) <= 0 or activation_observations.get("actual_supervised_sid_logit_max_abs", 0) <= 0:
        raise AssertionError("The actual SID input and output paths lack nonzero gradients")
    report["all_trainable_finite"] = all(bool(torch.isfinite(torch.linalg.vector_norm(parameter.detach(), ord=float("inf"), dtype=torch.float32))) for parameter in trainable.values())
    if not report["all_trainable_finite"]:
        raise FloatingPointError("A complete updated trainable tensor contains NaN/Inf")
    parameter_hashes = {name: tensor_digest(parameter) for name, parameter in trainable.items()}
    all_parameter_hashes = gather(parameter_hashes)
    report["rank_parameter_sync"] = {"all_complete_trainable_tensor_bytes_equal": all_parameter_hashes[0] == all_parameter_hashes[1], "sha256": hashlib.sha256(json.dumps(parameter_hashes, sort_keys=True).encode()).hexdigest(), "tensor_hashes": parameter_hashes}
    if not report["rank_parameter_sync"]["all_complete_trainable_tensor_bytes_equal"]:
        raise AssertionError("Replicated trainable parameters differ between ranks")
    if route == "L":
        report["lora_gradient_update_scope"] = {
            "all_lora_A_nonzero_after_B_update": all(observation["nonzero_calls"] > 0 for name, observation in gradient_observations.items() if "lora_A" in name),
            "all_lora_B_nonzero": all(observation["nonzero_calls"] > 0 for name, observation in gradient_observations.items() if "lora_B" in name),
            "lora_A_tensors_with_sampled_change": sum(entry["changed_elements"] > 0 for name, entry in report["parameter_changes"].items() if "lora_A" in name),
            "lora_A_tensor_count": sum("lora_A" in name for name in trainable),
            "lora_B_tensors_with_sampled_change": sum(entry["changed_elements"] > 0 for name, entry in report["parameter_changes"].items() if "lora_B" in name),
            "first_step_zero_A_gradient_expected_from_zero_B_initialization": True,
        }
        if not report["lora_gradient_update_scope"]["all_lora_B_nonzero"] or report["lora_gradient_update_scope"]["lora_B_tensors_with_sampled_change"] == 0:
            raise AssertionError("Expected nonzero LoRA_B gradients or real parameter changes were not observed")
    elif not any(entry["changed_elements"] > 0 for name, entry in report["parameter_changes"].items() if name != embedding_name):
        raise AssertionError("No representative Transformer parameter changed on the full-parameter route")
    report["core_training_status"] = "PASS"
    if route == "L" and (not report["lora_gradient_update_scope"]["all_lora_A_nonzero_after_B_update"] or report["lora_gradient_update_scope"]["lora_A_tensors_with_sampled_change"] != report["lora_gradient_update_scope"]["lora_A_tensor_count"]):
        report["core_training_status"] = "PARTIAL_LORA_A_UPDATE_NOT_FULLY_PROVEN"
    report["training_peak"] = {name: max(update["memory"][name] for update in report["updates"]) for name in ("peak_allocated_gib", "peak_reserved_gib", "peak_rss_gib")}
    report["long_tail_zero_training"] = {
        "status": "PASS" if any(update["kind"] == "mixed_tail" and update["optimizer_states_resident_at_start"] for update in report["updates"]) else "NOT_TESTED",
        "micro": micro_batch, "accumulation": accumulation, "global_batch": 128,
        "communication": "Real ZeRO-2 hooks, gradient reduction/accumulation and complete AdamW boundary update",
        "sample_ids": plans["mixed_global128_tail"],
        "tail_record_lengths": {sample_id: records[sample_id]["metrics"] for sample_id in plans["mixed_global128_tail"] if sample_id in {sample for group in plans["task_technical_representatives"].values() for sample in group}},
    }
    report["long_tail_zero_training"]["actual_dataset_maximum_coverage"] = {
        task: {
            "actual_maximum_sample": plans["task_representatives"][task][-1],
            "eligible_maximum_sample": plans["task_technical_representatives"][task][-1],
            "actual_maximum_in_complete_zero_update": plans["task_representatives"][task][-1] in plans["mixed_global128_tail"],
            "actual_maximum_metrics": records[plans["task_representatives"][task][-1]]["metrics"],
            "eligible_maximum_metrics": records[plans["task_technical_representatives"][task][-1]]["metrics"],
        }
        for task in TASKS
    }
    report["training_state_restore"] = {"status": "NOT_TESTED", "adapter_or_model_reload_is_not_training_restore": True}
    emit("training_correctness_complete", core_status=report["core_training_status"])

    training_state_fallback = {}

    def check_training_state():
        checkpoint_started = time.monotonic()
        guard()
        checkpoint_directory = output_directory / "training_state"
        projected_checkpoint_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.state_dict().values()) + sum(value.numel() * value.element_size() for state in engine.optimizer.optimizer.state.values() for value in state.values() if torch.is_tensor(value)) * world_size + sum(parameter.numel() * parameter.element_size() for parameter in engine.optimizer.single_partition_of_fp32_groups) * world_size
        report["training_state_restore"]["projected_bytes"] = projected_checkpoint_bytes
        if projected_checkpoint_bytes > int(config.get("max_training_state_bytes", 11 * GIB)):
            raise RuntimeError("Projected bounded training-state checkpoint exceeds its configured byte allowance")
        expected_state_hashes = {
            str(position) + ":" + name: tensor_digest(value)
            for position, state in enumerate(engine.optimizer.optimizer.state.values())
            for name, value in state.items() if torch.is_tensor(value)
        }
        checkpoint_rng = {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device), "python": random.getstate(), "numpy": numpy_module.random.get_state()}
        training_state_fallback["rng"] = checkpoint_rng
        training_state_fallback["state_hashes"] = expected_state_hashes
        expected_progress = optimizer_progress()
        expected_scheduler = scheduler.state_dict()
        client_state = {"rng": checkpoint_rng, "probe_global_steps": expected_progress["engine_global_steps"], "run_commit": arguments.run_commit}
        engine.save_checkpoint(str(checkpoint_directory), tag="bounded", client_state=client_state)
        rng_file = checkpoint_directory / ("rank" + str(rank) + "_rng.pt")
        torch.save(checkpoint_rng, rng_file)
        distributed.barrier()
        synchronize()
        save_seconds = max_rank_seconds(time.monotonic() - checkpoint_started)
        first_moment = next(state["exp_avg"] for state in engine.optimizer.optimizer.state.values() if "exp_avg" in state)
        training_state_fallback["first_moment_scalar"] = first_moment.reshape(-1)[0].detach().cpu().clone()
        first_moment.reshape(-1)[0].add_(1.)
        torch.rand(8, device=device)
        random.random()
        numpy_module.random.random()
        restore_started = time.monotonic()
        load_path, restored_client = engine.load_checkpoint(str(checkpoint_directory), tag="bounded", load_optimizer_states=True, load_lr_scheduler_states=True, load_module_strict=True)
        if load_path is None or restored_client["run_commit"] != arguments.run_commit:
            raise AssertionError("DeepSpeed training checkpoint failed strict recovery")
        restored_rng = torch.load(rng_file, map_location="cpu", weights_only=False)
        torch.set_rng_state(restored_rng["torch"].cpu())
        torch.cuda.set_rng_state(restored_rng["cuda"].cpu(), device)
        random.setstate(restored_rng["python"])
        numpy_module.random.set_state(restored_rng["numpy"])
        restored_hashes = {
            str(position) + ":" + name: tensor_digest(value)
            for position, state in enumerate(engine.optimizer.optimizer.state.values())
            for name, value in state.items() if torch.is_tensor(value)
        }
        if restored_hashes != expected_state_hashes or optimizer_progress() != expected_progress or scheduler.state_dict() != expected_scheduler:
            raise AssertionError("Restored GPU optimizer/scheduler/global-step state differs")
        restored_model_hashes = {name: tensor_digest(parameter) for name, parameter in trainable.items()}
        if restored_model_hashes != parameter_hashes or not torch.equal(torch.get_rng_state(), checkpoint_rng["torch"]) or not torch.equal(torch.cuda.get_rng_state(device), checkpoint_rng["cuda"]):
            raise AssertionError("Restored model or PyTorch RNG differs from the checkpoint")
        restored_numpy_state = numpy_module.random.get_state()
        expected_numpy_state = checkpoint_rng["numpy"]
        if random.getstate() != checkpoint_rng["python"] or restored_numpy_state[0] != expected_numpy_state[0] or not numpy_module.array_equal(restored_numpy_state[1], expected_numpy_state[1]) or restored_numpy_state[2:] != expected_numpy_state[2:]:
            raise AssertionError("Restored Python or NumPy RNG differs from the checkpoint")
        report["training_state_restore"] = {
            "status": "PASS", "path": str(checkpoint_directory), "save_seconds": save_seconds,
            "reload_seconds": time.monotonic() - restore_started,
            "strict_model_optimizer_scheduler_step_and_rng": True,
            "moment_mutation_was_restored": True, "rng_ownership": "Saved and explicitly restored per-rank client state; DeepSpeed does not automatically restore caller RNG",
            "scope": "Disk save/load into the existing ZeRO-2 engine; a new-process resume is NOT TESTED",
            "files": [record_file(path) for path in sorted(checkpoint_directory.rglob("*")) if path.is_file()],
        }
        del first_moment, checkpoint_rng, client_state, restored_client
        emit("training_state_disk_restore_complete")

    if config.get("save_training_state", config.get("full_training_state_check", False)):
        try:
            check_training_state()
        except Exception as error:
            report["training_state_restore"].update(status="PARTIAL", error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
            if "first_moment_scalar" in training_state_fallback:
                current_first_moment = next(state["exp_avg"] for state in engine.optimizer.optimizer.state.values() if "exp_avg" in state)
                current_first_moment.reshape(-1)[0].copy_(training_state_fallback["first_moment_scalar"].to(device))
            if "rng" in training_state_fallback:
                fallback_rng = training_state_fallback["rng"]
                torch.set_rng_state(fallback_rng["torch"].cpu())
                torch.cuda.set_rng_state(fallback_rng["cuda"].cpu(), device)
                random.setstate(fallback_rng["python"])
                numpy_module.random.set_state(fallback_rng["numpy"])
            report["training_state_restore"]["trained_parameters_unchanged_after_failed_restore"] = parameter_hashes == {name: tensor_digest(parameter) for name, parameter in trainable.items()}
            emit("optional_training_state_check_failed", status="PARTIAL")
        training_state_fallback.clear()
        current_first_moment = None
        report["training_state_restore"]["all_rank_status"] = gather({"status": report["training_state_restore"]["status"], "error": report["training_state_restore"].get("error")})
        if any(state["status"] != "PASS" for state in report["training_state_restore"]["all_rank_status"]):
            report["training_state_restore"]["status"] = "PARTIAL"

    for handle in hooks:
        handle.remove()
    hooks.clear()
    destroy_source = inspect.getsource(type(engine.optimizer).destroy)
    report["zero_hook_cleanup"] = {
        "engine_destroy_source_sha256": hashlib.sha256(inspect.getsource(type(engine).destroy).encode()).hexdigest(),
        "optimizer_destroy_source_sha256": hashlib.sha256(destroy_source.encode()).hexdigest(),
        "optimizer_source": record_file(inspect.getfile(type(engine.optimizer))),
        "purpose": "Retain actual FP32 master/moment tensors while removing autograd communication hooks for explicitly no-communication diagnostics",
    }
    engine.destroy()
    model.zero_grad(set_to_none=True)
    report["zero_hook_cleanup"]["called"] = True
    report["resident_optimizer_state_after_hook_cleanup"] = optimizer_state_inventory()
    diagnostic_started = time.monotonic()
    report["task_diagnostics"] = []
    emit("resident_state_no_communication_diagnostics_begin")
    for task in TASKS:
        representative_ids = plans["task_representatives"][task]
        for quantile, sample_id in zip(("short", "p50", "p90", "max"), representative_ids):
            guard()
            if time.monotonic() - diagnostic_started > float(config.get("diagnostics_max_seconds", 600)):
                report["diagnostics_deadline_reached"] = True
                break
            row = {
                "task": task, "quantile": quantile, "sample_id": sample_id,
                "optimizer_updates": 0, "micro": 1,
                "communication": "NONE: ZeRO autograd hooks removed; actual GPU master/Adam moments remain resident",
                "distinct_records_across_ranks": 1, "compute_presentations_across_ranks": 2,
                "diagnostic_only": True,
                "technical_update_eligible": records[sample_id].get("technical_update_eligible", False),
            }
            result = batch = value = None
            try:
                model.zero_grad(set_to_none=True)
                torch.cuda.reset_peak_memory_stats(device)
                phase_started = time.monotonic()
                cpu, counts = cpu_batch([sample_id])
                row["cpu_collator_seconds"] = time.monotonic() - phase_started
                row["counts_per_rank"] = counts
                phase_started = time.monotonic()
                batch = {name: tensor.to(device) for name, tensor in cpu.items()}
                synchronize()
                row["transfer_seconds"] = time.monotonic() - phase_started
                if quantile == "max":
                    model.eval()
                    phase_started = time.monotonic()
                    with torch.no_grad():
                        forward_only = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
                    synchronize()
                    row["resident_state_forward_only_seconds"] = time.monotonic() - phase_started
                    del forward_only
                model.train()
                phase_started = time.monotonic()
                result = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
                value = calculate_loss(result.logits, batch["labels"])
                synchronize()
                row["forward_seconds"] = time.monotonic() - phase_started
                phase_started = time.monotonic()
                value.backward()
                synchronize()
                row["backward_seconds"] = time.monotonic() - phase_started
                row["loss"] = float(value.detach())
                row["status"] = "PASS"
            except Exception as error:
                row.update(status="OOM" if isinstance(error, torch.cuda.OutOfMemoryError) else "FAIL", error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
            finally:
                row["memory"] = memory()
                del result, batch, value
                model.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
            row["all_rank_status"] = gather({"status": row["status"], "memory": row["memory"], "error": row.get("error")})
            if any(status["status"] != "PASS" for status in row["all_rank_status"]):
                row["status"] = "OOM" if any(status["status"] == "OOM" for status in row["all_rank_status"]) else "FAIL"
            report["task_diagnostics"].append(row)
            emit("task_no_communication_diagnostic_complete", task=task, quantile=quantile, status=row["status"])
        if report.get("diagnostics_deadline_reached"):
            break
    report["no_communication_diagnostics_parameters_unchanged"] = parameter_hashes == {name: tensor_digest(parameter) for name, parameter in trainable.items()}
    if not report["no_communication_diagnostics_parameters_unchanged"]:
        raise AssertionError("No-step diagnostic unexpectedly modified trainable weights")
    model.eval()
    report["validation"] = {"status": "RUNNING", "cases": [], "formal_rows_per_evaluation": 49544, "sample_valid": True, "prompts_per_user": 2, "full_validation_executed": False}
    with torch.no_grad():
        for position, record in enumerate(sample_document["validation"]):
            guard()
            cpu_started = time.monotonic()
            cpu = collator([record])
            if any(cpu[name][0].tolist() != record["collated"][name] for name in ("input_ids", "labels", "attention_mask")):
                raise AssertionError("Validation fixed tokenizer/collator replay differs")
            collator_seconds = time.monotonic() - cpu_started
            synchronize()
            phase_started = time.monotonic()
            batch = {name: tensor.to(device) for name, tensor in cpu.items()}
            synchronize()
            transfer_seconds = time.monotonic() - phase_started
            phase_started = time.monotonic()
            result = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
            value = calculate_loss(result.logits, batch["labels"])
            synchronize()
            seconds = max_rank_seconds(time.monotonic() - phase_started)
            report["validation"]["cases"].append({"sample_position": position, "validation_index": record["validation_index"], "user_ref": record.get("user_ref"), "input_tokens": int(cpu["attention_mask"].sum()), "shift_label_tokens": int((cpu["labels"][:, 1:] != -100).sum()), "cpu_collator_seconds": collator_seconds, "transfer_seconds": transfer_seconds, "forward_seconds_max_rank": seconds, "loss": float(value), "distinct_records": 1, "compute_presentations": 2, "memory": memory()})
            del batch, result, value
    report["validation"]["status"] = "PASS"
    emit("bounded_validation_complete")

    saved_model_directory = output_directory / ("adapter" if route == "L" else "full_model")
    report["save_reload"] = {"status": "RUNNING", "path": str(saved_model_directory), "training_restore_distinct": True}
    del engine, optimizer, scheduler, named_parameters, trainable, frozen, monitored, embedding, parameter
    transfer_started = time.monotonic()
    model.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    synchronize()
    report["memory_after_training_model_release"] = memory()
    report["save_reload"]["training_model_to_cpu_seconds"] = max_rank_seconds(time.monotonic() - transfer_started)
    loaded_model = loaded_state = None

    def reload_saved_model():
        if route == "L":
            fresh_model, information = load_model(model_path)
            fresh_model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
            fresh_model.to(device)
            fresh_model = PeftModel.from_pretrained(fresh_model, saved_model_directory, is_trainable=False, autocast_adapter_dtype=False)
            fresh_state = get_peft_model_state_dict(fresh_model, save_embedding_layers=False)
        else:
            fresh_model, information = load_model(saved_model_directory, device=device)
            fresh_state = fresh_model.state_dict()
        if set(fresh_state) != set(state_metadata):
            raise AssertionError("Reload added or omitted expected adapter/model parameter keys")
        for name, tensor in fresh_state.items():
            metadata = state_metadata[name]
            if list(tensor.shape) != metadata["shape"] or str(tensor.dtype) != metadata["dtype"] or tensor_digest(tensor) != metadata["sha256"]:
                raise AssertionError("Reload changed saved parameter shape, dtype, or exact bytes: " + name)
        reloaded_tokenizer = AutoTokenizer.from_pretrained(saved_model_directory, local_files_only=True)
        if reloaded_tokenizer.get_vocab() != tokenizer.get_vocab() or reloaded_tokenizer.convert_tokens_to_ids(sample_document["sid_tokens"]) != sid_ids:
            raise AssertionError("Saved tokenizer changed the original930 SID token order or vocabulary")
        if not sharing(fresh_model)["same_parameter_object"]:
            raise AssertionError("Public reload did not restore active tied embedding/head semantics")
        return fresh_model, information

    try:
        expected_state = get_peft_model_state_dict(model, save_embedding_layers=False) if route == "L" else model.state_dict()
        state_metadata = {name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "sha256": tensor_digest(tensor)} for name, tensor in expected_state.items()}
        del expected_state
        distributed.barrier()
        save_started = time.monotonic()
        guard()
        save_decision = [None]
        if rank == 0:
            try:
                if route == "L":
                    model.save_pretrained(saved_model_directory, safe_serialization=False, save_embedding_layers=False)
                else:
                    model.save_pretrained(saved_model_directory, safe_serialization=True, max_shard_size="4GB")
                tokenizer.save_pretrained(saved_model_directory)
                write(saved_model_directory / "sid_token_order.json", {"tokens": token_identity["sid_tokens"], "ids": sid_ids})
                save_decision[0] = {"status": "PASS"}
            except Exception as error:
                save_decision[0] = {"error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}}
        distributed.broadcast_object_list(save_decision, src=0)
        if "error" in save_decision[0]:
            raise RuntimeError("Model/adapter save failed: " + json.dumps(save_decision[0]["error"]))
        synchronize()
        report["save_reload"].update(save_seconds=max_rank_seconds(time.monotonic() - save_started), tensor_manifest=state_metadata)
        report["save_reload"]["saved_files"] = [record_file(path) for path in sorted(saved_model_directory.iterdir()) if path.is_file()]
        emit("model_or_adapter_saved")
        reload_started = time.monotonic()
        local_reload_error = None
        try:
            loaded_model, reload_information = reload_saved_model()
            synchronize()
        except Exception as error:
            local_reload_error = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        all_reload_errors = gather(local_reload_error)
        report["save_reload"]["all_rank_reload_errors"] = all_reload_errors
        report["save_reload"]["reload_seconds"] = max_rank_seconds(time.monotonic() - reload_started)
        if any(error is not None for error in all_reload_errors):
            loaded_model = None
            raise RuntimeError("At least one rank failed strict saved-model reload: " + json.dumps(all_reload_errors))
        report["save_reload"].update(status="PASS", all_key_shape_dtype_value_bytes_exact=True, tokenizer_vocab_and_sid_order_exact=True, sharing=sharing(loaded_model), loading_information=reload_information)
        model = loaded_model
        emit("fresh_saved_model_reload_complete")
    except Exception as error:
        report["save_reload"].update(status="PARTIAL", error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}, generation_fallback="Retained exact trained CPU model returned to GPU; save/reload remains failed")
        loaded_model = loaded_state = None
        gc.collect()
        torch.cuda.empty_cache()
        model.to(device)
        emit("save_reload_failed_continue_generation", status="PARTIAL")
    loaded_model = loaded_state = None

    vendor_directory = config.get("vendor_dir", sample_document.get("vendor_dir", str(Path(__file__).parents[2] / "vendor" / "original")))
    sys.path.insert(0, str(vendor_directory))
    from evaluate import get_topk_results

    model.eval()
    model.config.use_cache = True
    model.gradient_checkpointing_disable()
    canonical_items = {"".join(parts) for parts in sample_document["indices"].values()}
    generation_records = sample_document["generation"]
    if len(generation_records) != 51 or {record["prompt_id"] for record in generation_records} != set(range(17)):
        raise AssertionError("Expected three fixed real test users for every original17 SeqRec prompts")
    report["generation"] = {
        "status": "RUNNING", "beam": 20, "return_sequences": 20, "max_new_tokens": 10,
        "templates": 17, "cases": [], "prompt_timings": [], "filter_items": False,
        "constraint": "Original LC-Rec per-position legal SID tokens, then EOS; delimiter locating adapted to actual Qwen prompt token boundaries; no strict trie",
        "postprocess": "Original split('Response:')[-1].strip().replace(' ','') and score sort; no added deduplication",
        "formal_metrics_computed": False, "untrained_canonical_rate_is_not_recommendation_quality": True,
        "canonical_distinct_sids": len(canonical_items), "original_items": len(sample_document["indices"]),
    }
    generation_started = time.monotonic()
    for prompt_id in range(17):
        guard()
        template_cases = sorted([record for record in generation_records if record["prompt_id"] == prompt_id], key=lambda record: (record["row_idx"], record["history_bucket"]))
        distributed.barrier()
        synchronize()
        template_started = time.monotonic()
        local_cases = []
        for position, record in enumerate(template_cases):
            if position % world_size != rank:
                continue
            guard()
            case = {"prompt_id": prompt_id, "row_idx": record["row_idx"], "user_ref": record["user_ref"], "history_bucket": record["history_bucket"], "target_sid": record["target_sid"], "status": "RUNNING"}
            generated = inputs = None
            try:
                token_ids = tokenizer(record["input_text"])["input_ids"]
                if token_ids != record["input_ids"]:
                    raise AssertionError("Original SeqRec generation prompt encoding differs from CPU replay")
                callback = Allowed(tokenizer, sample_document["indices"])
                callback.bind(token_ids, record["input_text"])
                inputs = {"input_ids": torch.tensor([token_ids], device=device), "attention_mask": torch.ones((1, len(token_ids)), dtype=torch.long, device=device)}
                case["input_tokens"] = len(token_ids)
                case["input_text_sha256"] = hashlib.sha256(record["input_text"].encode()).hexdigest()
                torch.cuda.reset_peak_memory_stats(device)
                synchronize()
                phase_started = time.monotonic()
                with torch.no_grad():
                    generated = model.generate(
                        **inputs, max_new_tokens=10, prefix_allowed_tokens_fn=callback,
                        num_beams=20, num_return_sequences=20, output_scores=True,
                        return_dict_in_generate=True, early_stopping=True,
                        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                    )
                synchronize()
                case["generation_seconds"] = time.monotonic() - phase_started
                suffixes = generated.sequences[:, len(token_ids):].tolist()
                decoded = tokenizer.batch_decode(generated.sequences, skip_special_tokens=True)
                texts = [text.split("Response:")[-1].strip().replace(" ", "") for text in decoded]
                legal = []
                eos_positions = []
                for suffix in suffixes:
                    allowed = True
                    eos_position = None
                    for token_position, token_id in enumerate(suffix):
                        if token_id not in callback.allowed.get(token_position, set()):
                            allowed = False
                        if token_id == tokenizer.eos_token_id:
                            eos_position = token_position
                            break
                    legal.append(allowed)
                    eos_positions.append(eos_position)
                official_topk = get_topk_results(decoded, generated.sequences_scores.clone(), [record["target_sid"]], 20, all_items=None)
                case.update(
                    status="PASS", returned_sequences=len(suffixes), generated_token_ids=suffixes,
                    generated_text=texts, generated_scores=generated.sequences_scores.float().cpu().tolist(),
                    eos_positions=eos_positions, all_position_legal=all(legal),
                    canonical_count=sum(text in canonical_items for text in texts),
                    canonical_fraction=sum(text in canonical_items for text in texts) / 20,
                    duplicate_sid_count=len(texts) - len(set(texts)),
                    callback_calls=callback.calls, callback_events=callback.events,
                    callback_position_sizes={str(position): len(tokens) for position, tokens in callback.allowed.items()},
                    official_evaluator_topk=official_topk,
                )
                if len(suffixes) != 20 or not all(legal) or any(position != 4 for position in eos_positions):
                    raise AssertionError("Original per-position beam20 interface returned wrong count, illegal tokens, or unexpected EOS")
            except Exception as error:
                case.update(status="OOM" if isinstance(error, torch.cuda.OutOfMemoryError) else "FAIL", error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}, lowered_beam=False)
            finally:
                case["memory"] = memory()
                del generated, inputs
                torch.cuda.empty_cache()
            local_cases.append(case)
            emit("generation_case_complete", flush_report=False, **case)
        seconds = max_rank_seconds(time.monotonic() - template_started)
        all_cases = [case for rank_cases in gather(local_cases) for case in rank_cases]
        report["generation"]["cases"].extend(all_cases)
        successful_times = [case["generation_seconds"] for case in all_cases if case["status"] == "PASS"]
        report["generation"]["prompt_timings"].append({"prompt_id": prompt_id, "wall_seconds_max_rank": seconds, "users": len(all_cases), "successes": len(successful_times), "individual_seconds": successful_times})
        emit("generation_template_complete", prompt_id=prompt_id, successes=len(successful_times))
        if time.monotonic() - generation_started > float(config.get("generation_max_seconds", 900)):
            report["generation"]["deadline_reached"] = True
            break
    generation = report["generation"]
    generation["status"] = "PASS" if len(generation["cases"]) == 51 and all(case["status"] == "PASS" for case in generation["cases"]) else "PARTIAL"
    generation["elapsed_seconds_max_rank"] = max_rank_seconds(time.monotonic() - generation_started)
    report["replay_verified_records"] = len(replayed)
    report["unused_layouts"] = {"micro" + str(layout): "NOT_TESTED" for layout in (8, 4, 2, 1) if layout != micro_batch}
    report["final_memory"] = memory()
    report["status"] = "PASS" if report["core_training_status"] == "PASS" and report["save_reload"]["status"] == "PASS" and generation["status"] == "PASS" and report["long_tail_zero_training"]["status"] == "PASS" and len(report["task_diagnostics"]) == 24 and all(row["status"] == "PASS" for row in report["task_diagnostics"]) and report["training_state_restore"]["status"] != "PARTIAL" else "PARTIAL"
    report["completed_global_updates"] = len(report["updates"])
    report["nonzero_lr_updates"] = sum(update["nonzero_lr"] for update in report["updates"])
    emit("all_bounded_gpu_stages_complete", status=report["status"])


def main():
    arguments = parse_cli()
    arguments.output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("RANK", 0))
    report_path = arguments.output / ("rank" + str(rank) + ".json")
    events_path = arguments.output / ("events_rank" + str(rank) + ".jsonl")
    started = time.monotonic()
    report = {
        "status": "RUNNING", "argv": sys.argv, "cwd": str(Path.cwd()),
        "script": record_file(__file__), "common_script": record_file(Path(__file__).with_name("common.py")),
        "started_utc": datetime_module.datetime.now(datetime_module.timezone.utc).isoformat(),
        "updates": [], "rank": rank,
    }

    def flush():
        report["elapsed_seconds"] = time.monotonic() - started
        write(report_path, report)

    def emit(stage, flush_report=True, **values):
        report["stage"] = stage
        event = {"rank": rank, "stage": stage, "elapsed_seconds": time.monotonic() - started, "utc": datetime_module.datetime.now(datetime_module.timezone.utc).isoformat(), **values}
        with events_path.open("a") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
        if flush_report:
            flush()
        print(json.dumps({"rank": rank, "stage": stage, **{name: value for name, value in values.items() if name in ("update", "micro", "task", "quantile", "prompt_id", "status", "seconds")}}, allow_nan=False), flush=True)

    try:
        emit("started")
        run(arguments, report, emit, flush)
    except BaseException as error:
        report["status"] = "PARTIAL" if report.get("updates") else "BLOCKED"
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        print(report["error"]["traceback"], file=sys.stderr, flush=True)
    finally:
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_initialized():
            try:
                report["final_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / GIB
                report["final_peak_reserved_gib"] = torch.cuda.max_memory_reserved() / GIB
            except Exception as error:
                report["final_gpu_query_error"] = str(error)
        report["finished_utc"] = datetime_module.datetime.now(datetime_module.timezone.utc).isoformat()
        report["host_final"] = host_resources()
        flush()
        distributed = sys.modules.get("torch.distributed")
        if distributed is not None and distributed.is_initialized():
            distributed.destroy_process_group()
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
