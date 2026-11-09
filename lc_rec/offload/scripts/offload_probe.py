#!/usr/bin/env python3
"""Two-update, real SeqRec, ZeRO-3 CPU-offload engineering probe.

Draft for controller review. This is not the six-task reproduction or a formal
training entry point. No evaluation, checkpoint, optimizer state, or model is
saved. All parameter evidence uses tiny samples from existing local CPU views.

Important precision boundary: omitting torch_dtype preserves the HF call default,
but HF 4.31's implicit zero.Init reads BF16 from the DeepSpeed configuration.
DS 0.16.9 constructs FP32 masters from its low-precision partitions. This script
records that fact; it does not claim an unchanged FP32-checkpoint master source.
"""
import argparse
import collections
import contextlib
import datetime
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import socket
import sys
import time
import traceback


FORMAL_STEPS = 6912
FORMAL_WARMUP = 70
EXPECTED_SID_TOKENS = 930
EXPECTED_VERSIONS = {"torch": "2.6.0", "transformers": "4.31.0",
                     "accelerate": "0.21.0", "deepspeed": "0.16.9"}


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha_file(path)}


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": repr(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(clean_json(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n")
    os.replace(temporary, path)


def tensor_hash(tensor):
    # Used on one small token/mask record, never on model tensors.
    values = tensor.detach().cpu().contiguous().tolist()
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-identity", type=Path, required=True,
                        help="Controller's preverified, lightweight weight identity manifest")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--deepspeed-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-commit", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--seed", type=int, default=42, choices=[42])
    parser.add_argument("--optimizer-updates", type=int, default=2, choices=[2])
    cli = parser.parse_args()
    for field in ["source", "data_root", "model", "model_identity", "deepspeed_config", "output"]:
        setattr(cli, field, getattr(cli, field).resolve())
    for field in ["run_commit", "model_revision"]:
        if not re.fullmatch(r"[0-9a-f]{40}", getattr(cli, field)):
            parser.error(field + " must be a complete lowercase 40-character revision")
    return cli


def run(cli, report, flush):
    import torch
    import torch.distributed as dist
    from torch.utils.data import Dataset, SequentialSampler
    import transformers
    import deepspeed
    from transformers import LlamaForCausalLM, LlamaTokenizer, TrainingArguments, Trainer, TrainerCallback

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world != 2:
        raise RuntimeError("This probe requires exactly two torchrun ranks, micro1/accum64/effective128")
    torch.cuda.set_device(local_rank)
    torch.cuda.reset_peak_memory_stats(local_rank)
    report["environment"] = {
        "interpreter": sys.executable, "python": sys.version,
        "versions": {name: importlib.metadata.version(name) for name in EXPECTED_VERSIONS},
        "imports": {"torch": torch.__file__, "transformers": transformers.__file__,
                    "deepspeed": deepspeed.__file__},
        "torch_cuda": torch.version.cuda, "rank": rank, "local_rank": local_rank,
        "world_size": world, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(local_rank),
        "gpu_total_bytes": torch.cuda.get_device_properties(local_rank).total_memory,
    }
    for name, expected in EXPECTED_VERSIONS.items():
        actual = report["environment"]["versions"][name].split("+")[0]
        if actual != expected:
            raise RuntimeError("Version mismatch: " + name + " " + actual + " != " + expected)

    ds_config = json.loads(cli.deepspeed_config.read_text())
    zero = ds_config.get("zero_optimization", {})
    if zero.get("stage") != 3 or zero.get("offload_optimizer", {}).get("device") != "cpu":
        raise RuntimeError("Explicit ZeRO-3 CPU optimizer offload is required")
    if zero.get("offload_param", {}).get("device", "none") not in ("none", "cpu"):
        raise RuntimeError("This probe does not authorize NVMe offload")
    if ds_config.get("bf16", {}).get("enabled") is not True:
        raise RuntimeError("BF16 must be explicitly enabled")
    if "optimizer" in ds_config or "scheduler" in ds_config:
        raise RuntimeError("Keep Trainer AdamW decay groups and the explicit formal HF scheduler")
    for key in ("reduce_bucket_size", "stage3_prefetch_bucket_size", "stage3_param_persistence_threshold"):
        if type(zero.get(key)) is not int or zero[key] < 1:
            raise RuntimeError("Bucket/persistence field must be a positive explicit integer: " + key)
    for key, expected in (("train_micro_batch_size_per_gpu", 1),
                          ("gradient_accumulation_steps", 64), ("train_batch_size", 128)):
        if ds_config.get(key) != expected:
            raise RuntimeError(key + " must explicitly equal " + str(expected))
    if ds_config.get("zero_force_ds_cpu_optimizer", True) is not True:
        raise RuntimeError("Keep Accelerate's CPUAdam mapping; do not silently run CPU torch.AdamW")
    if zero.get("stage3_gather_16bit_weights_on_model_save", False):
        raise RuntimeError("No gathered save is allowed in this probe")
    if ds_config.get("flops_profiler", {}).get("enabled", False):
        raise RuntimeError("This technical probe must disable the profiler explicitly in its own DS config")
    report["deep_speed_input_config"] = ds_config
    report["configuration_identity"] = file_record(cli.deepspeed_config)

    sys.path.insert(0, str(cli.source))
    import utils as original_utils
    import data as original_data
    import collator as original_collator
    for module, filename in [(original_utils, "utils.py"), (original_data, "data.py"),
                             (original_collator, "collator.py")]:
        if Path(module.__file__).resolve() != cli.source / filename:
            raise RuntimeError("Official module shadowed: " + filename)
    report["official_source"] = [file_record(cli.source / filename) for filename in
        ["utils.py", "data.py", "collator.py", "finetune.py", "prompt.py", "run.sh"]]
    report["asset_identity"] = [file_record(cli.data_root / "Instruments" / filename) for filename in
        ["Instruments.inter.json", "Instruments.index.json"]]
    report["model_identity"] = {
        "model_path": str(cli.model), "requested_revision": cli.model_revision,
        "external_weight_manifest": file_record(cli.model_identity),
        "manifest_scope": "Preverified by controller before run; this probe does not rehash 13GB on every rank",
        "small_files": [file_record(cli.model / filename) for filename in
            ["config.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json"]
            if (cli.model / filename).is_file()],
    }
    report["phase"] = "construct_training_arguments_before_model"
    flush()
    # Establish HF's ZeRO-3 integration BEFORE from_pretrained. Do not use
    # device_map, low_cpu_mem_usage, or quantization in this training path.
    training_args = TrainingArguments(
        output_dir=str(cli.output / "trainer-no-checkpoints"),
        seed=42, data_seed=42, per_device_train_batch_size=1,
        gradient_accumulation_steps=64, num_train_epochs=4,
        max_steps=2, learning_rate=5e-5, weight_decay=0.01,
        optim="adamw_torch", lr_scheduler_type="cosine", warmup_ratio=0.01,
        bf16=True, fp16=False, gradient_checkpointing=True,
        deepspeed=ds_config, max_grad_norm=1.0,
        evaluation_strategy="no", save_strategy="no", load_best_model_at_end=False,
        logging_strategy="no", report_to=[], disable_tqdm=True,
        remove_unused_columns=False, dataloader_num_workers=0,
        dataloader_pin_memory=False, ddp_find_unused_parameters=False,
        logging_nan_inf_filter=False, auto_find_batch_size=False,
    )
    report["training_arguments"] = training_args.to_dict()
    report["scope"] = {
        "technical_probe_only": True, "formal_training_runs": 0,
        "optimizer_update_budget": 2, "micro_batch_per_rank": 1,
        "accumulation": 64, "world_size": 2, "effective_batch": 128,
        "formal_six_task_candidate_changed": False,
        "optimizer_dataset": "SeqRecDataset train only; auxiliary labels are not consumed",
        "formal_schedule_steps": FORMAL_STEPS, "formal_warmup_steps": FORMAL_WARMUP,
        "local_max_steps": 2, "local_max_steps_not_a_convergence_schedule": True,
        "not_performed": ["six-task joint optimization", "validation", "model save/reload",
                          "generation", "recommendation metrics", "optimizer checkpoint/resume"],
        "resource_claim_limit": "20-item real SeqRec path; not all six tasks or 2048-token longest path",
    }
    report["numeric_boundary"] = {
        "from_pretrained_torch_dtype_kwarg": "omitted, matching official HF call default",
        "torch_default_dtype_before_model": str(torch.get_default_dtype()),
        "hf_zero_init_enabled": True,
        "zero_init_dtype_expected_from_ds_config": "torch.bfloat16",
        "fp32_master_source": "DS0.16.9 clones and casts its low-precision partition to float32",
        "preserves_full_fp32_checkpoint_master_source": False,
        "limitation": "BF16 zero.Init is a numerical difference even when torch_dtype is omitted; technical results do not prove an identical official initialization trajectory",
    }
    original_utils.set_seed(42)
    parser = argparse.ArgumentParser()
    parser = original_utils.parse_global_args(parser)
    parser = original_utils.parse_train_args(parser)
    parser = original_utils.parse_dataset_args(parser)
    original_args = parser.parse_args([])
    original_args.dataset = "Instruments"
    original_args.data_path = str(cli.data_root)
    original_args.only_train_response = True
    original_args.model_max_length = 2048
    original_args.max_his_len = 20
    original_args.seed = 42
    dataset = original_data.SeqRecDataset(original_args, mode="train", prompt_sample_num=1, sample_num=0)
    if len(dataset) < 256:
        raise RuntimeError("Need 256 distinct real SeqRec prefixes for two effective-batch128 updates")
    history_counts = [len(re.findall(r"<a_\d+>", row["inters"])) for row in dataset.inter_data]
    # No metric or outcome selection. Resolve ties using the original prefix index.
    selection = sorted(range(len(dataset)), key=lambda i: (-history_counts[i], i))[:256]
    if max(history_counts[i] for i in selection) != 20:
        raise RuntimeError("The selected real SeqRec set does not include a 20-item history")
    selected_set = set(selection)
    prefix_identity = {}
    prefix_cursor = 0
    for uid, items in dataset.inters.items():
        for target_position in range(1, len(items) - 2):
            if prefix_cursor in selected_set:
                target_id = str(items[target_position])
                canonical = "".join(dataset.indices[target_id])
                if canonical != dataset.inter_data[prefix_cursor]["item"]:
                    raise RuntimeError("Selected prefix identity does not match official dataset construction")
                prefix_identity[prefix_cursor] = {"user_id": str(uid),
                    "target_position_zero_based": target_position, "target_item_id": target_id,
                    "canonical_target_sid": canonical,
                    "history_item_ids": list(map(str, items[:target_position][-20:]))}
            prefix_cursor += 1
    if prefix_cursor != len(dataset) or len(prefix_identity) != len(selection):
        raise RuntimeError("Cannot trace selected prefixes back to real user/item identities")
    tokenizer = LlamaTokenizer.from_pretrained(str(cli.model), local_files_only=True,
        model_max_length=2048, padding_side="right")
    tokenizer.pad_token_id = 0
    base_vocab = len(tokenizer)
    sid_tokens = dataset.get_new_tokens()
    added = tokenizer.add_tokens(sid_tokens)
    if added != EXPECTED_SID_TOKENS or len(sid_tokens) != EXPECTED_SID_TOKENS:
        raise RuntimeError("Expected exactly 930 actual author SID tokens, not a synthetic full1024 vocabulary")
    token_map = {token: int(tokenizer.convert_tokens_to_ids(token)) for token in sid_tokens}
    if any(tokenizer.encode(token, add_special_tokens=False) != [token_id]
           for token, token_id in token_map.items()):
        raise RuntimeError("An actual SID token is not one tokenizer token")
    appearances = []
    for group in (selection[:128], selection[128:]):
        appearances.append(set(re.findall(r"<[abcd]_\d+>", " ".join(
            dataset.inter_data[index]["inters"] + dataset.inter_data[index]["item"] for index in group))))
    common = sorted(appearances[0] & appearances[1])
    if not common:
        raise RuntimeError("No shared, actually appearing SID row for a two-update evidence sample")
    sampled_sid_ids = [token_map[token] for token in common[:4]]
    report["data_selection"] = {
        "official_seqrec_prefix_rows": len(dataset), "selected_rows": len(selection),
        "rule": "descending count of real history items, ties by official prefix index; first256, without replacement; no metric selection",
        "selected_source_indices": selection,
        "selected_prefix_identities": [{"source_index": i, **prefix_identity[i]} for i in selection],
        "selected_history_items": [history_counts[i] for i in selection],
        "fixed_indices_sha256": hashlib.sha256(json.dumps(selection).encode()).hexdigest(),
        "global_batch_source_indices": [selection[:128], selection[128:]],
        "sampler": "SequentialSampler on the fixed subset; Accelerate shards it across the two ranks",
        "prompt_sampling": "official dynamic __getitem__ sampling remains; actual strings/tokens are recorded by collator",
        "selection_max_total_tokens_not_preclaimed": True,
    }
    report["tokenizer"] = {"base_vocab": base_vocab, "new_tokens": added,
        "total_vocab": len(tokenizer), "sid_token_map": token_map,
        "sampled_sid_token_ids": sampled_sid_ids,
        "sampled_sid_presence": "selected tokens occur in both fixed effective128 input groups",
        "bos_token_id": tokenizer.bos_token_id, "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id, "padding_side": tokenizer.padding_side}
    report["phase"] = "load_base_model"
    flush()
    model, loading = LlamaForCausalLM.from_pretrained(str(cli.model),
        local_files_only=True, output_loading_info=True)
    report["base_loading_info"] = loading
    if any(loading.get(key) for key in ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"]):
        raise RuntimeError("Base loading contains missing/unexpected/mismatched/error entries; SID is resized only AFTER this strict base check")
    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False
    named = dict(model.named_parameters())
    if any(not parameter.requires_grad for parameter in named.values()):
        raise RuntimeError("A backbone parameter is frozen")
    report["model"] = {
        "config": model.config.to_dict(), "all_parameters_trainable": True,
        "trainable_numel": sum(int(getattr(p, "ds_numel", p.numel())) for p in named.values()),
        "parameter_objects": len(named), "actual_parameter_dtypes": dict(collections.Counter(str(p.dtype) for p in named.values())),
        "input_head_tied_by_object": model.get_input_embeddings().weight is model.get_output_embeddings().weight,
        "tie_word_embeddings_config": model.config.tie_word_embeddings,
        "intentional_initialization": {"input_embedding_new_rows": [base_vocab, len(tokenizer)],
            "output_head_new_rows": [base_vocab, len(tokenizer)],
            "method": "TF4.31 resize normal initialization in actual parameter dtype; no mean_resizing, no strict=False"},
    }
    report["phase"] = "prepare_trainer"
    report["actual_micro_batches"] = []
    report["losses"] = []
    report["optimizer_updates"] = []
    flush()

    class ProbeDataset(Dataset):
        def __len__(self):
            return len(selection)

        def __getitem__(self, index):
            source_index = selection[index]
            record = dict(dataset[source_index])
            record["_probe_subset_index"] = index
            record["_probe_source_index"] = source_index
            return record

    base_collator = original_collator.Collator(original_args, tokenizer)

    class AuditCollator:
        def __call__(self, records):
            tensors = base_collator(records)
            for pos, record in enumerate(records):
                ids = tensors["input_ids"][pos]
                mask = tensors["attention_mask"][pos]
                labels = tensors["labels"][pos]
                used_sid = sorted(set(ids[(ids >= base_vocab) & (mask != 0)].tolist()))
                count = int((labels != -100).sum())
                if count <= 0:
                    raise RuntimeError("Real SeqRec sample has no supervised response token")
                report["actual_micro_batches"].append({
                    "local_collator_record": len(report["actual_micro_batches"]),
                    "subset_index": record["_probe_subset_index"], "source_index": record["_probe_source_index"],
                    "real_prefix_identity": prefix_identity[record["_probe_source_index"]],
                    "history_items": history_counts[record["_probe_source_index"]],
                    "input_text": record["input_ids"], "full_text_without_appended_eos": record["labels"],
                    "input_ids": ids.tolist(), "labels": labels.tolist(), "attention_mask": mask.tolist(),
                    "input_ids_sha256": tensor_hash(ids), "labels_sha256": tensor_hash(labels),
                    "attention_mask_sha256": tensor_hash(mask), "nonpadding_tokens": int(mask.sum()),
                    "supervised_tokens": count, "actual_sid_token_ids": used_sid,
                })
            return tensors

    def cpu_tensor_sample(tensor, indices=None, limit=8):
        if tensor is None:
            return None
        flat = tensor.detach().view(-1)
        if indices is None:
            indices = sorted({0, max(0, flat.numel() - 1), *[
                (flat.numel() - 1) * i // max(limit - 1, 1) for i in range(limit)]}) if flat.numel() else []
        index = torch.tensor(indices, dtype=torch.long, device=flat.device)
        # At most a few dozen scalar values cross the CPU/GPU boundary here.
        values = flat.index_select(0, index).float().cpu().tolist() if indices else []
        return {"device": str(tensor.device), "dtype": str(tensor.dtype), "numel": tensor.numel(),
                "indices": indices, "values": values}

    def state_snapshot(engine):
        zopt = engine.optimizer
        inner = zopt.optimizer
        flats = zopt.fp32_partitioned_groups_flat
        states = []
        for group_index, flat in enumerate(flats):
            state = inner.state.get(flat, {})
            states.append({"subgroup_index": group_index,
                "logical_group_index": int(zopt.sub_group_to_group_id[group_index]),
                "master": {"device": str(flat.device), "dtype": str(flat.dtype), "numel": flat.numel()},
                "states": {key: cpu_tensor_sample(value) if torch.is_tensor(value) else value
                           for key, value in state.items()}})
        return {"engine_class": type(engine).__module__ + "." + type(engine).__name__,
            "zero_optimizer_class": type(zopt).__module__ + "." + type(zopt).__name__,
            "inner_optimizer_class": type(inner).__module__ + "." + type(inner).__name__,
            "offload_optimizer": zopt.offload_optimizer, "swap_optimizer": zopt.swap_optimizer,
            "actual_groups": [{key: value for key, value in group.items() if key != "params"}
                               for group in inner.param_groups],
            "states_by_subgroup": states}

    sample_names = ["model.embed_tokens.weight", "lm_head.weight",
        "model.layers.0.self_attn.q_proj.weight", "model.layers.0.mlp.down_proj.weight",
        "model.layers.31.self_attn.q_proj.weight", "model.norm.weight"]
    for name in sample_names:
        if name not in named:
            raise RuntimeError("Expected representative parameter missing: " + name)

    def local_views(engine, parameter):
        # DS0.16.9 safe_get_local_* moves WHOLE local partitions to CUDA.
        # Read the same validated CPU-offload storage through tiny narrow views.
        zopt = engine.optimizer
        if not zopt.offload_optimizer or zopt.swap_optimizer:
            raise RuntimeError("Sampling is audited only for CPU optimizer offload, no NVMe")
        zopt.reduce_and_partition_stream.synchronize()
        group, offset, size = zopt.grad_position[zopt.get_param_id(parameter)]
        master_flat = zopt.fp32_partitioned_groups_flat[group]
        if master_flat.device.type != "cpu" or master_flat.dtype != torch.float32:
            raise RuntimeError("Expected local CPU FP32 master storage")
        master = master_flat.narrow(0, offset, size)
        gradient = master_flat.grad.narrow(0, offset, size) if master_flat.grad is not None else None
        part_rank = dist.get_rank(group=zopt.dp_process_group)
        part_world = dist.get_world_size(group=zopt.dp_process_group)
        expected_size = (parameter.ds_numel + part_world - 1) // part_world
        if size != expected_size:
            raise RuntimeError("Unexpected DS partition mapping; do not invent global SID row identity")
        return master, gradient, int(part_rank * size), int(group), int(offset)

    def parameter_samples(engine, include_grad):
        samples = {}
        hidden = int(model.config.hidden_size)
        columns = [0, 1, 7, 63, hidden // 2, hidden - 1]
        for name in sample_names:
            parameter = named[name]
            master, gradient, start, subgroup, offset = local_views(engine, parameter)
            end = min(start + master.numel(), parameter.ds_numel)
            if name in ("model.embed_tokens.weight", "lm_head.weight"):
                global_indices = [token_id * hidden + column for token_id in sampled_sid_ids for column in columns]
                kind = "sid_embedding" if name.startswith("model.embed") else "sid_head"
            else:
                global_indices = [start + (end - start - 1) * i // 15 for i in range(16)] if end > start else []
                kind = "backbone"
            global_indices = sorted({i for i in global_indices if start <= i < end})
            local_indices = [i - start for i in global_indices]
            entry = {"category": kind, "parameter_name": name, "ds_id": int(parameter.ds_id),
                "ds_shape": list(parameter.ds_shape), "ds_numel": int(parameter.ds_numel),
                "local_partition_global_start": start, "local_partition_numel": master.numel(),
                "zero_subgroup": subgroup, "subgroup_offset": offset, "global_flat_indices": global_indices,
                "sid_row_column": [[i // hidden, i % hidden] for i in global_indices] if kind != "backbone" else None,
                "master": cpu_tensor_sample(master, local_indices),
                "absence_is_expected_on_nonowning_rank": kind != "backbone" and not local_indices}
            if include_grad:
                entry["gradient"] = cpu_tensor_sample(gradient, local_indices)
            samples[name] = entry
        return samples

    def all_cpu_gradients_finite(engine):
        checked = 0
        missing = []
        for group, master in enumerate(engine.optimizer.fp32_partitioned_groups_flat):
            gradient = master.grad
            if gradient is None:
                missing.append(group)
                continue
            if gradient.device.type != "cpu":
                raise RuntimeError("Gradient finite scan expected CPU offloaded flat storage")
            flat = gradient.detach().view(-1)
            # Bounded bool scratch (~1 MiB), never a full-gradient clone.
            for offset in range(0, flat.numel(), 1024 * 1024):
                chunk = flat.narrow(0, offset, min(1024 * 1024, flat.numel() - offset))
                if not bool(torch.isfinite(chunk).all()):
                    return {"finite": False, "checked_numel": checked,
                            "first_nonfinite_subgroup": group, "chunk_start": offset}
                checked += chunk.numel()
        return {"finite": not missing, "checked_numel": checked, "missing_grad_subgroups": missing}

    def global_flags(flags, op):
        tensor = torch.tensor(flags, dtype=torch.int32, device=training_args.device)
        dist.all_reduce(tensor, op=op)
        return [bool(value) for value in tensor.cpu().tolist()]

    class ProbeTrainer(Trainer):
        def _get_train_sampler(self):
            return SequentialSampler(self.train_dataset)

        def create_scheduler(self, num_training_steps, optimizer=None):
            report["scheduler"] = {"trainer_requested_local_total": num_training_steps,
                "actual_total": FORMAL_STEPS, "warmup": FORMAL_WARMUP,
                "method": "official HF cosine with warmup_ratio=.01 and total6912, despite local max_steps2"}
            if self.args.get_warmup_steps(FORMAL_STEPS) != FORMAL_WARMUP:
                raise RuntimeError("Formal warmup must remain ceil(6912*.01)=70")
            return super().create_scheduler(FORMAL_STEPS, optimizer=optimizer)

        def create_optimizer(self):
            result = super().create_optimizer()
            by_id = {id(value): name for name, value in self.model.named_parameters()}
            report["trainer_optimizer_groups_before_cpuadam_mapping"] = [
                {"group": i, "names": [by_id[id(p)] for p in group["params"]],
                 "weight_decay": group["weight_decay"], "lr": group["lr"],
                 "betas": group["betas"], "eps": group["eps"]}
                for i, group in enumerate(self.optimizer.param_groups)]
            report["trainer_optimizer_class_before_mapping"] = type(self.optimizer).__module__ + "." + type(self.optimizer).__name__
            return result

        def compute_loss(self, model, inputs, return_outputs=False):
            result = super().compute_loss(model, inputs, return_outputs=return_outputs)
            loss = result[0] if return_outputs else result
            value = float(loss.detach().float().cpu())
            report["losses"].append({"local_micro_step": len(report["losses"]) + 1,
                                      "raw_model_loss": value, "finite": math.isfinite(value)})
            if not math.isfinite(value):
                raise FloatingPointError("Nonfinite raw model loss")
            return result

        def save_model(self, *args, **kwargs):
            raise RuntimeError("Probe explicitly forbids model saving")

        def _save_checkpoint(self, *args, **kwargs):
            raise RuntimeError("Probe explicitly forbids checkpoint saving")

        def evaluate(self, *args, **kwargs):
            raise RuntimeError("Probe explicitly forbids evaluation")

    trainer = ProbeTrainer(model=model, args=training_args, train_dataset=ProbeDataset(),
        tokenizer=tokenizer, data_collator=AuditCollator())
    original_no_sync = trainer.accelerator.no_sync
    report["compatibility_shims"] = {
        "accelerate_021_deepspeed_0169_no_sync": {
            "reason": "Old Accelerate calls model.no_sync during accumulation; DS0.16.9 now defines it but rejects ZeRO3. Older DS had no such method, yielding nullcontext.",
            "scope": "Only this Trainer's Accelerator and only a real DeepSpeedEngine; engine still performs its native64-micro accumulation",
            "uses": 0}}

    @contextlib.contextmanager
    def compatible_no_sync(wrapped_model):
        if isinstance(wrapped_model, deepspeed.DeepSpeedEngine):
            report["compatibility_shims"]["accelerate_021_deepspeed_0169_no_sync"]["uses"] += 1
            yield
        else:
            with original_no_sync(wrapped_model):
                yield

    trainer.accelerator.no_sync = compatible_no_sync

    class StepEvidence(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            engine = trainer.model_wrapped
            if not isinstance(engine, deepspeed.DeepSpeedEngine):
                raise RuntimeError("Trainer is not using a real DeepSpeedEngine")
            if not hasattr(engine.optimizer, "grad_position"):
                raise RuntimeError("Expected actual ZeRO-3 parameter partitions")
            if type(engine.optimizer.optimizer).__name__ != "DeepSpeedCPUAdam":
                raise RuntimeError("Actual optimizer is not DeepSpeedCPUAdam")
            report["actual_deepspeed_config"] = engine.config
            report["initial_optimizer_state"] = state_snapshot(engine)
            report["initial_parameter_samples"] = parameter_samples(engine, False)
            original_step = engine.step

            def audited_step(*step_args, **step_kwargs):
                if not engine.is_gradient_accumulation_boundary():
                    return original_step(*step_args, **step_kwargs)
                index = len(report["optimizer_updates"]) + 1
                if index > 2:
                    raise RuntimeError("Optimizer update budget exceeded")
                started = time.monotonic()
                before = parameter_samples(engine, True)
                gradients = all_cpu_gradients_finite(engine)
                finite_all = global_flags([gradients["finite"]], dist.ReduceOp.MIN)[0]
                event = {"update_index": index, "engine_global_steps_before": engine.global_steps,
                    "engine_micro_steps_before": engine.micro_steps,
                    "lr_before": [float(group["lr"]) for group in engine.optimizer.optimizer.param_groups],
                    "before": before, "gradient_finite_scan": gradients,
                    "all_rank_gradients_finite": finite_all}
                report["optimizer_updates"].append(event)
                flush()
                if not finite_all:
                    raise FloatingPointError("A rank has nonfinite or missing accumulated gradients")
                expected_lr = 0.0 if index == 1 else 5e-5 / FORMAL_WARMUP
                if any(abs(lr - expected_lr) > 1e-12 for lr in event["lr_before"]):
                    raise RuntimeError("Actual update LR disagrees with preserved formal warmup")
                result = original_step(*step_args, **step_kwargs)
                after = parameter_samples(engine, False)
                event["after"] = after
                event["engine_global_steps_after"] = engine.global_steps
                event["optimizer_state_after"] = state_snapshot(engine)
                norm = engine.get_global_grad_norm()
                norm = float(norm.detach().cpu()) if torch.is_tensor(norm) else float(norm)
                event["actual_global_grad_norm"] = norm
                event["overflow_flag"] = bool(getattr(engine.optimizer, "overflow", False))
                witnesses = {kind: {"gradient_nonzero": False, "master_changed": False} for kind in
                             ["sid_embedding", "sid_head", "backbone"]}
                for name, entry in before.items():
                    old = entry["master"]["values"]
                    new = after[name]["master"]["values"]
                    delta = [right - left for left, right in zip(old, new)]
                    after[name]["sample_delta"] = delta
                    after[name]["max_abs_sample_delta"] = max(map(abs, delta), default=0.0)
                    gradient = entry.get("gradient")
                    values = gradient["values"] if gradient else []
                    kind = entry["category"]
                    witnesses[kind]["gradient_nonzero"] |= any(value != 0.0 for value in values)
                    witnesses[kind]["master_changed"] |= any(value != 0.0 for value in delta)
                ordered = [witnesses[kind][field] for kind in ["sid_embedding", "sid_head", "backbone"]
                           for field in ["gradient_nonzero", "master_changed"]]
                reduced = global_flags(ordered, dist.ReduceOp.MAX)
                event["global_sample_witnesses"] = {
                    kind: {"gradient_nonzero": reduced[2 * i], "master_changed": reduced[2 * i + 1]}
                    for i, kind in enumerate(["sid_embedding", "sid_head", "backbone"])}
                event["sample_scope"] = "tiny actual SID row and backbone CPU FP32 master/gradient samples; no full gather or full-model clone"
                event["elapsed_seconds_in_audited_boundary"] = time.monotonic() - started
                flush()
                if not math.isfinite(norm) or event["overflow_flag"]:
                    raise FloatingPointError("Nonfinite global gradient norm or optimizer overflow")
                if engine.global_steps != event["engine_global_steps_before"] + 1:
                    raise RuntimeError("DeepSpeed did not complete the expected boundary")
                if index == 2 and not all(reduced):
                    raise RuntimeError("Second update lacks a sampled nonzero gradient/FP32-master delta witness")
                # First LR is exactly zero. No master delta is required there.
                return result

            engine.step = audited_step
            report["phase"] = "training_two_updates"
            flush()

        def on_step_end(self, args, state, control, **kwargs):
            control.should_save = False
            control.should_evaluate = False
            if state.global_step >= 2:
                control.should_training_stop = True
            flush()
            return control

        def on_epoch_end(self, args, state, control, **kwargs):
            control.should_save = False
            control.should_evaluate = False
            return control

    trainer.add_callback(StepEvidence())
    outcome = trainer.train(resume_from_checkpoint=None)
    report["trainer_result"] = {"global_step": outcome.global_step, "training_loss": outcome.training_loss,
                                "metrics": outcome.metrics}
    if outcome.global_step != 2 or len(report["optimizer_updates"]) != 2:
        raise RuntimeError("Did not actually finish exactly two optimizer boundaries")
    if len(report["losses"]) != 128:
        raise RuntimeError("Expected exactly128 local forwards/backwards at micro1/accum64/two updates")
    final_opt = state_snapshot(trainer.model_wrapped)
    report["final_optimizer_state"] = final_opt
    for group in final_opt["states_by_subgroup"]:
        states = group["states"]
        step = states.get("step")
        step = step["values"][0] if isinstance(step, dict) and step.get("values") else step
        if step != 2:
            raise RuntimeError("CPUAdam subgroup did not actually advance its state step to2")
        for key in ["exp_avg", "exp_avg_sq"]:
            if states.get(key, {}).get("device") != "cpu" or states[key].get("dtype") != "torch.float32":
                raise RuntimeError("CPUAdam moments not confirmed as CPU FP32")
    report["input_coverage"] = {"actual_local_micro_records": len(report["actual_micro_batches"]),
        "max_actual_nonpadding_tokens": max(row["nonpadding_tokens"] for row in report["actual_micro_batches"]),
        "max_actual_history_items": max(row["history_items"] for row in report["actual_micro_batches"]),
        "all_six_task_longest_path_verified": False}
    report["status"] = "TECHNICAL_TWO_UPDATE_PASS"
    report["phase"] = "finished"
    flush()


def main():
    cli = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    started = time.monotonic()
    report_path = cli.output / ("probe_rank" + str(rank) + ".json")
    report = {"schema": "lc-rec-offload-technical-probe-v1", "status": "RUNNING",
        "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "job_id": cli.job_id, "run_commit": cli.run_commit, "hostname": socket.gethostname(),
        "rank": rank, "actual_argv": sys.argv, "cwd": str(Path.cwd()),
        "script_identity": file_record(__file__), "phase": "import_and_preflight",
        "formal_training_runs": 0, "model_saves": 0, "generated_users": 0,
        "external_watchdog_required": True, "errors": []}

    def flush():
        report["elapsed_wall_seconds"] = time.monotonic() - started
        write_json(report_path, report)

    exit_code = 1
    flush()
    def on_signal(number, frame):
        raise RuntimeError("Probe received termination signal " + str(number))
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    try:
        run(cli, report, flush)
        exit_code = 0
    except BaseException as exc:
        report["status"] = "FAILED"
        report["errors"].append({"type": type(exc).__name__, "message": str(exc),
                                  "traceback": traceback.format_exc()})
        traceback.print_exc()
    finally:
        report["exit_code"] = exit_code
        usage = resource.getrusage(resource.RUSAGE_SELF)
        report["process_resources"] = {"peak_rss_bytes_linux": int(usage.ru_maxrss * 1024),
            "cpu_user_seconds": usage.ru_utime, "cpu_system_seconds": usage.ru_stime}
        torch = sys.modules.get("torch")
        try:
            if torch is not None and torch.cuda.is_initialized():
                device = int(os.environ.get("LOCAL_RANK", "0"))
                report["gpu_allocator_peak"] = {"local_rank": device,
                    "max_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "max_reserved_bytes": torch.cuda.max_memory_reserved(device),
                    "scope": "PyTorch allocator for this rank, includes model load; use external watchdog for process/device total"}
        except BaseException as resource_error:
            report["resource_read_error"] = repr(resource_error)
        report["ended_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        flush()
        print(json.dumps({"rank": rank, "status": report["status"], "exit_code": exit_code,
                          "report": str(report_path)}, ensure_ascii=False), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
