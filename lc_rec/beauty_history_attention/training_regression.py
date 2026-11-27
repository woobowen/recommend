"""Compare real 7B LETTER gradients on the historical and matched causal paths."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import sys
from types import SimpleNamespace

from .identity import (
    HISTORICAL_COMMIT, HISTORICAL_CONFIG_SHA, NEW_TOKEN_COUNT, TRAINABLE_PARAMETERS,
    VOCAB_SIZE, atomic_json, require, sha, tokenizer_identity, verify_assets, verify_base_model,
)
from .train import HISTORICAL_VERSIONS, VENDOR, VENDOR_HASHES, build_serial_sampler


@contextmanager
def original_attention(historical_config, event):
    """Install the historical FastChat/FA2 hooks and restore every changed attribute."""
    from transformers.models.llama.modeling_llama import LlamaAttention, LlamaModel
    from fastchat.train import llama2_flash_attn_monkey_patch as fastchat
    from flash_attn import flash_attn_interface as flash
    from compat import adapt_fastchat_unpad_arity
    from formal_attention import install

    owners = (
        (LlamaAttention, "forward"), (LlamaModel, "_prepare_decoder_attention_mask"),
        (fastchat, "unpad_input"), (fastchat, "flash_attn_func"),
        (fastchat, "flash_attn_varlen_kvpacked_func"),
        (flash, "_wrapped_flash_attn_backward"), (flash, "_wrapped_flash_attn_varlen_backward"),
        (flash.flash_attn_gpu, "bwd"), (flash.flash_attn_gpu, "varlen_bwd"),
    )
    previous = [(owner, name, getattr(owner, name)) for owner, name in owners]
    try:
        fastchat.replace_llama_attn_with_flash_attn()
        report = install(historical_config, event)
        report["unpad_compat"] = adapt_fastchat_unpad_arity()
        dtype_counts = defaultdict(int)
        report["qkv_dtypes"] = dtype_counts

        def observe(function, packed):
            @wraps(function)
            def observed(*args, **kwargs):
                q, k, v = (args[0], args[1], args[1]) if packed else args[:3]
                key = "/".join(str(value.dtype) for value in (q, k, v))
                dtype_counts[key] += 1
                return function(*args, **kwargs)
            return observed

        fastchat.flash_attn_func = observe(fastchat.flash_attn_func, False)
        fastchat.flash_attn_varlen_kvpacked_func = observe(fastchat.flash_attn_varlen_kvpacked_func, True)
        yield report
    finally:
        for owner, name, value in previous:
            setattr(owner, name, value)


def new_error():
    return {"elements": 0, "max_abs": 0.0, "abs_sum": 0.0, "error_sq": 0.0,
            "old_sq": 0.0, "new_sq": 0.0, "dot": 0.0}


def add_error(acc, old, new):
    """Aggregate in float64 without allocating an FP64 copy of a full embedding table."""
    import torch

    require(old.shape == new.shape, "regression operand shape mismatch")
    for left, right in zip(old.reshape(-1).split(1 << 18), new.reshape(-1).split(1 << 18)):
        left, right = left.double(), right.double()
        error = left - right
        require(torch.isfinite(left).all() and torch.isfinite(right).all(), "nonfinite regression operand")
        acc["elements"] += error.numel()
        acc["max_abs"] = max(acc["max_abs"], float(error.abs().max()))
        acc["abs_sum"] += float(error.abs().sum())
        acc["error_sq"] += float(error.dot(error))
        acc["old_sq"] += float(left.dot(left))
        acc["new_sq"] += float(right.dot(right))
        acc["dot"] += float(left.dot(right))


def summary(acc):
    squared = acc["error_sq"]
    old_norm = math.sqrt(acc["old_sq"])
    new_norm = math.sqrt(acc["new_sq"])
    return {
        "elements": acc["elements"], "max_abs": acc["max_abs"],
        "mean_abs": acc["abs_sum"] / acc["elements"],
        "rms": math.sqrt(squared / acc["elements"]),
        "relative_l2_vs_old": math.sqrt(squared) / old_norm if old_norm else None,
        "old_l2": old_norm, "new_l2": new_norm,
        "cosine": acc["dot"] / (old_norm * new_norm) if old_norm and new_norm else None,
        "both_exact_zero": old_norm == new_norm == 0,
    }


def error_summary(old, new):
    acc = new_error()
    add_error(acc, old, new)
    return summary(acc)


def weight_sha(value):
    import torch

    tensor = value.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(tensor)).hexdigest()


def run_pass(model, batch, rng, *, profile):
    import torch
    from torch.profiler import ProfilerActivity
    from contextlib import nullcontext

    torch.set_rng_state(rng["cpu"])
    torch.cuda.set_rng_state(rng["cuda"])
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    observer = torch.profiler.profile(activities=[ProfilerActivity.CPU]) if profile else nullcontext()
    with observer:
        outputs = model(**batch, use_cache=False)
        loss = outputs.loss
        logits = outputs.logits.detach().cpu()
        scalar = float(loss.detach().float().cpu())
        loss.backward()
    torch.cuda.synchronize()
    require(math.isfinite(scalar), "nonfinite training loss")
    backend_ops = {}
    if profile:
        backend_ops = {row.key: row.count for row in observer.key_averages()
                       if any(text in row.key.lower() for text in ("scaled_dot", "efficient_attention", "flash_attn"))}
    alloc = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    return logits, {
        "loss": scalar, "loss_dtype": str(loss.dtype), "logit_dtype": str(logits.dtype),
        "backend_ops": backend_ops, "peak_allocated_bytes": alloc, "peak_reserved_bytes": reserved,
    }


def run(args, report):
    import torch
    import peft
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import LlamaTokenizer

    sys.path.insert(0, str(VENDOR))
    from utils import load_datasets, set_seed
    from collator import Collator
    from modeling_letter import LETTER
    from compat import (restore_legacy_bnb_state_field, install_peft05_out_of_place,
                        install_peft05_lora_a_bf16_autocast)
    from formal_attention import assert_active
    from .attention import install_attention
    from .inputs import HistoryCollator

    torch.set_num_threads(2)
    require(torch.cuda.device_count() == 1, "one visible GPU required")
    require(re.fullmatch(r"[a-f0-9]{40}", args.run_commit), "full source commit required")
    versions = {key: importlib.metadata.version(key) for key in HISTORICAL_VERSIONS}
    require(versions == HISTORICAL_VERSIONS, "historical training dependency versions differ")
    require({name: sha(VENDOR / name) for name in VENDOR_HASHES} == VENDOR_HASHES,
            "historical training vendor source drift")
    config_path = Path(__file__).with_name("historical_train_config.json")
    require(sha(config_path) == HISTORICAL_CONFIG_SHA, "historical training config drift")
    historical_config = json.loads(config_path.read_text())
    cfg = SimpleNamespace(**historical_config, data_path=str(args.data_root))
    report.update(
        versions=versions, historical_commit=HISTORICAL_COMMIT, historical_config_sha256=HISTORICAL_CONFIG_SHA,
        data_identity=verify_assets(args.data_root), base_model_identity=verify_base_model(args.model),
        device_name=torch.cuda.get_device_name(), gpu_visible_devices=__import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
    )
    set_seed(cfg.seed)
    train, valid = load_datasets(cfg)
    require((len(train), len(valid)) == (131413, 44726), "historical dataset mismatch")
    tok = LlamaTokenizer.from_pretrained(
        args.model, model_max_length=2048, padding_side="right", local_files_only=True,
    )
    tok.pad_token_id = 0
    sid_tokens = train.datasets[0].get_new_tokens()
    require(tok.add_tokens(sid_tokens) == NEW_TOKEN_COUNT and len(tok) == VOCAB_SIZE, "historical vocab size")
    report["tokenizer_identity"] = tokenizer_identity(tok, sid_tokens)
    native = Collator(cfg, tok)
    from itertools import islice
    rows = list(islice(iter(build_serial_sampler(train)), 8))
    examples = [train[row] for row in rows]
    history_batch = HistoryCollator(native, tok, sid_tokens)(examples)
    old_batch = native(examples)
    require(all(torch.equal(old_batch[name], history_batch[name]) for name in old_batch),
            "history collator changed the original batch")
    report["real_batch"] = {
        "row_indices": rows, "scope": "initial historical rank-0 eight-example batch",
        "shape": list(old_batch["input_ids"].shape), "real_tokens": int(old_batch["attention_mask"].sum()),
        "labelled_prediction_positions": int((old_batch["labels"][:, 1:] != -100).sum()),
        "sha256": {name: weight_sha(value) for name, value in history_batch.items()},
    }
    atomic_json(args.output, report)
    new_batch = {name: value.cuda(0) for name, value in history_batch.items()}
    old_batch_gpu = {name: value.cuda(0) for name, value in old_batch.items()}

    events = []
    def record(kind, **info):
        events.append({"event": kind, **info})

    # Accelerate captures the attention forward when loading device_map.
    # Install the old class patch before loading, as the old trainer did.
    with original_attention(historical_config, record) as old_report:
        raw = LETTER.from_pretrained(args.model, load_in_8bit=True, device_map={"": 0}, local_files_only=True)
        raw.set_hyper(cfg.temperature)
        raw.resize_token_embeddings(len(tok))
        raw = peft.prepare_model_for_int8_training(raw)
        report["bnb_compat"] = restore_legacy_bnb_state_field(raw)
        model = get_peft_model(raw, LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules.split(","), modules_to_save=["embed_tokens", "lm_head"],
            bias="none", inference_mode=False, task_type=TaskType.CAUSAL_LM,
        ))
        report["out_of_place"] = install_peft05_out_of_place(model)
        for name, value in model.named_parameters():
            if "original_module" in name and any(s in name for s in ("embed_tokens", "lm_head")):
                value.requires_grad = False
        model.config.use_cache = False
        # DeepSpeed 0.16.9 DeepSpeedEngine._configure_distributed_model also casts
        # this already placed 8-bit module by calling self.module.bfloat16().
        model.bfloat16()
        report["precision_adaptation"] = install_peft05_lora_a_bf16_autocast(model, {
            "matched_dtype_conflict": True, "source": "fixed historical 6e1b211 LETTER Beauty precision adaptation",
        })
        active = {name: value for name, value in model.named_parameters() if value.requires_grad}
        require(len(active) == 450 and sum(value.numel() for value in active.values()) == TRAINABLE_PARAMETERS,
                "historical trainable set mismatch")
        require(all(value.dtype == torch.bfloat16 for value in active.values()), "historical DeepSpeed BF16 trainable cast")
        require(raw.model.gradient_checkpointing, "historical gradient checkpointing not enabled")
        require(all(torch.count_nonzero(value) == 0 for name, value in active.items() if ".lora_B." in name),
                "fresh initialization does not have zero LoRA B")
        before = {name: weight_sha(value) for name, value in active.items()}
        report["initialized_trainable_sha256"] = before
        report["execution_limits"] = [
            "Single training microbatch under each backend, with native LETTER loss and gradient checkpointing.",
            "DeepSpeed BF16 cast replicated from installed engine; raw gradients observed before ZeRO-2 reduction.",
            "No optimizer step or checkpoint load. The two paths share the very same parameter objects.",
            "Initial LoRA B is zero, so both paths should initially have zero LoRA A gradients.",
            "Kernel-level fusions and BF16 rounding can differ; the output reports measured errors rather than hiding them behind one tolerance.",
        ]
        model.train()
        rng = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}
        old_logits, old_summary = run_pass(model, old_batch_gpu, rng, profile=args.profile)
        assert_active(old_report)
        old_gradients = {}
        for name, parameter in active.items():
            require(parameter.grad is not None, f"historical active parameter missing gradient: {name}")
            old_gradients[name] = parameter.grad.detach().cpu()
        require(all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad),
                "frozen historical parameter acquired a gradient")
        report["old_causal"] = old_summary
        report["old_flash"] = old_report
        report["old_flash_events"] = events
        atomic_json(args.output, report)
    model.zero_grad(set_to_none=True)

    report["new_attention"] = install_attention(raw, "causal")
    new_logits, new_summary = run_pass(model, new_batch, rng, profile=args.profile)
    require(report["new_attention"]["layer_calls"] >= 2 * len(raw.model.layers),
            "new attention did not execute every decoder layer in both forward and checkpoint recomputation")
    report["new_causal"] = new_summary
    valid_positions = old_batch["attention_mask"].bool()
    supervised = old_batch["labels"][:, 1:] != -100
    old_predict, new_predict = old_logits[:, :-1][supervised], new_logits[:, :-1][supervised]
    report["logit_errors"] = {
        "all_valid": error_summary(old_logits[valid_positions], new_logits[valid_positions]),
        "supervised_predictions": error_summary(old_predict, new_predict),
        "supervised_top1_exact_count": int((old_predict.argmax(-1) == new_predict.argmax(-1)).sum()),
        "supervised_prediction_count": int(supervised.sum()),
    }
    report["loss_abs_error"] = abs(old_summary["loss"] - new_summary["loss"])

    by_family = defaultdict(new_error)
    by_parameter = {}
    for name, parameter in active.items():
        require(parameter.grad is not None, f"matched causal active parameter missing gradient: {name}")
        grad = parameter.grad.detach().cpu()
        old_grad = old_gradients.pop(name)
        require(grad.dtype == old_grad.dtype, f"active gradient dtype changed: {name}")
        if ".lora_A." in name:
            family = "lora_A"
        elif ".lora_B." in name:
            family = "lora_B"
        elif ".embed_tokens." in name:
            family = "embed_tokens"
        else:
            require(".lm_head." in name, f"unknown trainable tensor: {name}")
            family = "lm_head"
        acc = new_error()
        add_error(acc, old_grad, grad)
        for key, value in acc.items():
            if key == "max_abs":
                by_family[family][key] = max(by_family[family][key], value)
            else:
                by_family[family][key] += value
        by_parameter[name] = {"dtype": str(grad.dtype), **summary(acc)}
    require(all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad),
            "frozen matched causal parameter acquired a gradient")
    require(all(weight_sha(value) == before[name] for name, value in active.items()), "diagnostic changed trainable parameters")
    report["gradient_errors"] = {
        "families": {name: summary(error) for name, error in by_family.items()},
        "parameters": by_parameter, "unchanged_weights_and_trainable_membership": True,
    }
    report["status"] = "REGRESSION_MEASURED"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "model", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--run-commit", required=True)
    parser.add_argument("--profile", action="store_true", help="Capture causal kernel dispatch counters from the CPU profiler")
    args = parser.parse_args()
    require(not args.output.exists(), "diagnostic output already exists")
    report = {"status": "RUNNING", "argv": sys.argv, "run_commit": args.run_commit,
              "python": sys.executable, "entry_sha256": sha(__file__)}
    try:
        run(args, report)
    except BaseException as error:
        import traceback
        report.update(status="FAILED", error=repr(error), traceback=traceback.format_exc())
        atomic_json(args.output, report)
        raise
    atomic_json(args.output, report)
    print(json.dumps({"status": report["status"], "path": str(args.output),
                      "old_loss": report["old_causal"]["loss"], "new_loss": report["new_causal"]["loss"]}))


if __name__ == "__main__":
    main()
