"""Approved Toys/Sports scratch MLP training through the preserved lifecycle.

Derived from beauty_extensions.mlp_formal: keep model/optimizer/callbacks,
full validation, validation-best strict loading, complete test and recovery.
Only the explicitly approved domain settings and bookkeeping guards differ.
The caller in formal_20261015.run owns admission and detached runner identity.
The twelve-hour allowance includes full testing; exhaustion stays INCOMPLETE.
"""
from __future__ import annotations

from collections import Counter
import datetime as dt
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

from experiments.beauty_extensions.common import (
    bind_source, load_config, sha256, verify_asset, write_json,
)


class SafetyInterruption(RuntimeError):
    """Time/space exhaustion is an incomplete run, not early stopping."""


def configure(spec, output):
    """Apply only the user-approved domain overlay before model construction."""
    from omegaconf import OmegaConf

    domain = spec.get("domain")
    if spec.get("route") != "MLP" or domain not in ("toys", "sports"):
        raise ValueError("This entry is restricted to Toys/Sports original per-item MLP")
    if spec.get("formal_approved") is not True:
        raise ValueError("The caller must admit an explicitly approved formal plan")
    if spec.get("checkpoint") or spec.get("resume_checkpoint") or spec.get("shuffle_n", 0):
        raise ValueError("Formal cross-domain MLP requires a fresh unshuffled start")
    if spec.get("engineering_overlay") or spec.get("preflight_rows"):
        raise ValueError("Formal training cannot carry a bounded engineering overlay")
    hours = float(spec.get("hard_cap_hours", 12))
    if not math.isfinite(hours) or hours != 12:
        raise ValueError("The approved total hard allowance is twelve hours including testing")
    bind_source("MLP")
    cfg = load_config(spec, output)
    # load_config preserves original workers for formal_approved=True. Compare
    # with the hash-verified original source, rather than copying Beauty's eight.
    original = OmegaConf.load(spec["config"]["path"])
    cfg.train, cfg.test, cfg.ckpt_path = True, True, None
    raw_batch, accumulation, cadence, window_cap = {
        "toys": (32, 48, 4800, 512),
        "sports": (16, 16, 1600, 128),
    }[domain]
    before = {
        "trainer.accumulate_grad_batches": int(cfg.trainer.accumulate_grad_batches),
        "trainer.val_check_interval": int(cfg.trainer.val_check_interval),
    }
    cfg.trainer.accumulate_grad_batches = accumulation
    cfg.trainer.val_check_interval = cadence
    train = cfg.data_loading.train_dataloader_config.dataloader
    if (int(train.batch_size_per_device), int(train.collate_fn.max_batch_size)) != (raw_batch, window_cap):
        raise ValueError("Source raw microbatch/collate cap differs from the approved domain condition")
    if not train.assign_all_files_per_worker or not train.drop_last:
        raise ValueError("Original infinite training file/window protocol differs")
    worker_settings = {}
    for split in ("train", "val", "test"):
        lc = cfg.data_loading[f"{split}_dataloader_config"].dataloader
        original_lc = original.data_loading[f"{split}_dataloader_config"].dataloader
        for name in ("num_workers", "timeout", "persistent_workers"):
            if lc.get(name) != original_lc.get(name):
                raise ValueError("Source worker protocol changed: " + split + "." + name)
        worker_settings[split] = {
            "num_workers": int(lc.num_workers), "timeout": int(lc.timeout),
            "persistent_workers": bool(lc.persistent_workers),
        }
        if domain == "toys" and int(lc.num_workers) != 8:
            raise ValueError("Approved Toys source requires eight workers")
        if split != "train" and (lc.drop_last or lc.should_shuffle_rows or lc.get("assign_all_files_per_worker", False)):
            raise ValueError("Full validation/test must retain every nonduplicated source row")
    if cfg.seed != 42 or cfg.model.top_k_for_generation != 10:
        raise ValueError("Approved seed42/four-code K10 generation differs")
    if cfg.trainer.precision != "32-true" or cfg.trainer.num_sanity_val_steps != 0:
        raise ValueError("Source precision/sanity-validation protocol differs")
    if cfg.trainer.devices != 1 or cfg.trainer.strategy != "auto":
        raise ValueError("This plan requires the existing single-GPU strategy")
    if any(name.startswith("limit_") for name in cfg.trainer) or cfg.trainer.get("fast_dev_run", False) or cfg.trainer.get("overfit_batches", 0):
        raise ValueError("Formal trainer contains a development batch override")
    if cfg.optim.optimizer._target_ != "torch.optim.Adam" or cfg.optim.optimizer.lr != 0.001 or cfg.optim.optimizer.weight_decay != 0.0001 or cfg.optim.scheduler is not None:
        raise ValueError("Approved original Adam/no-scheduler contract differs")
    write_json(output / "domain_protocol.json", {
        "domain": domain, "source_config": spec["config"],
        "initialization": "scratch; no preflight checkpoint or optimizer state reused",
        "raw_microbatch": raw_batch, "accumulate_grad_batches": accumulation,
        "validation_interval_microbatches": cadence, "collate_window_cap": window_cap,
        "source_worker_protocol_preserved": worker_settings,
        "explicit_approved_overlay": {key: {"source": before[key], "approved": value}
            for key, value in (("trainer.accumulate_grad_batches", accumulation),
                               ("trainer.val_check_interval", cadence))},
        "max_steps": int(cfg.trainer.max_steps), "max_epochs": int(cfg.trainer.max_epochs),
        "hard_cap_hours_including_full_test": hours,
        "safety_allowance_is_not_training_endpoint": True,
    })
    return cfg


def read_index(record):
    verified = verify_asset(record)
    result = Counter()
    with Path(record["path"]).open() as stream:
        for line in stream:
            row = json.loads(line)
            target = row["target"]
            if len(target) != 4 or any(type(v) is not int or v < 0 for v in target):
                raise ValueError("Invalid four-code target in verified metadata index")
            if not isinstance(row["sample_key"], str) or len(row["sample_key"]) != 24:
                raise ValueError("Expected an opaque sample key in metadata index")
            result[(row["sample_key"], tuple(target))] += 1
    if not result:
        raise ValueError("Empty metadata index")
    return result, verified


def deadline_record(spec, request, status):
    hours = float(spec.get("hard_cap_hours", 12))
    if not math.isfinite(hours) or not 0 < hours <= 12:
        raise ValueError("The authorized wall-clock allowance is at most twelve hours")
    allowance = hours * 3600
    started = dt.datetime.now(dt.timezone.utc)
    source = "entry wall clock; engineering invocation without runner"
    if request is not None:
        started = dt.datetime.fromisoformat(status["started_at"].replace("Z", "+00:00"))
        if started.tzinfo is None:
            raise ValueError("Runner started_at must carry a timezone")
        allowance = min(allowance, float(request["timeout_seconds"]))
        source = "actual runner status.started_at and min(request timeout, approved allowance)"
    if allowance <= 180:
        raise ValueError("Runner allowance cannot reserve checkpoint safety time")
    return {"runner_started_at": started.isoformat(), "allowance_seconds": allowance,
            "safety_reserve_seconds": 180, "source": source,
            "internal_stop_timestamp": started.timestamp() + allowance - 180,
            "internal_stop_at": dt.datetime.fromtimestamp(started.timestamp() + allowance - 180,
                                                          dt.timezone.utc).isoformat(),
            "training_termination_condition": False}


def append_json(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def progress(output, value):
    temporary = output / "progress.json.tmp"
    write_json(temporary, {"at": dt.datetime.now(dt.timezone.utc).isoformat(), **value})
    os.replace(temporary, output / "progress.json")


def execute(phase, spec, cfg, output, deadline):
    import hydra
    import numpy as np
    import torch
    from lightning import Callback, seed_everything
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichModelSummary
    from omegaconf import OmegaConf
    from experiments.beauty_extensions.author.checks import describe_author_model
    from experiments.beauty_extensions.cli import PredictionRecorder
    from experiments.beauty_extensions.common import batch_keys, load_checkpoint
    from experiments.beauty_extensions.results import recompute

    if phase != "formal":
        raise ValueError("This cross-domain entry exposes only admitted formal training")
    formal = True
    accumulation = int(cfg.trainer.accumulate_grad_batches)
    train_config = cfg.data_loading.train_dataloader_config.dataloader
    val_config = cfg.data_loading.val_dataloader_config.dataloader
    test_config = cfg.data_loading.test_dataloader_config.dataloader
    raw_batch = int(train_config.batch_size_per_device)
    window_cap = int(train_config.collate_fn.max_batch_size)
    sequence_tokens = int(train_config.sequence_length)
    expected_val, val_index = read_index(spec["evaluation_index"])
    expected_test, test_index = read_index(spec["test_index"])
    minimum_free = int(spec.get("min_free_bytes", 20 * 1024**3))
    if minimum_free < 20 * 1024**3:
        raise ValueError("Disk safety margin must be at least twenty GiB")
    if shutil.disk_usage(output).free < minimum_free:
        raise SafetyInterruption("DISK_SAFETY_MARGIN_BEFORE_INITIALIZATION")
    if time.time() >= deadline["internal_stop_timestamp"]:
        raise SafetyInterruption("TIME_ALLOWANCE_BEFORE_INITIALIZATION")
    seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("medium")
    model = hydra.utils.instantiate(cfg.model)
    dm = hydra.utils.instantiate(cfg.data_loading.datamodule)
    if not all(p.requires_grad for p in model.parameters()):
        raise ValueError("Approved MLP training requires every parameter trainable")
    if model.get_embedding_table("encoder") is not model.get_embedding_table("decoder"):
        raise ValueError("Original MLP shared SID table identity changed")
    write_json(output / "structure.json", describe_author_model(model))
    trainer_cfg = cfg.trainer.copy()
    if formal:
        if any(name.startswith("limit_") for name in trainer_cfg):
            raise ValueError("Formal trainer must not contain preflight batch limits")
        if trainer_cfg.get("fast_dev_run", False) or trainer_cfg.get("overfit_batches", 0):
            raise ValueError("Formal trainer contains a development override")
    else:
        trainer_cfg.max_steps = 1
        trainer_cfg.val_check_interval = accumulation
        trainer_cfg.limit_test_batches = 2
    if trainer_cfg.get("limit_val_batches", 1.0) != 1.0:
        raise ValueError("Both phases require full validation")
    OmegaConf.save(cfg, output / "approved_formal_effective_config.yaml", resolve=False)
    OmegaConf.save(trainer_cfg, output / "actual_trainer.yaml", resolve=True)
    used = {key: OmegaConf.to_container(cfg[key], resolve=True)
            for key in ("model", "optim", "eval", "callbacks", "logger")}
    used["trainer"] = OmegaConf.to_container(trainer_cfg, resolve=True)
    used["datamodule"] = OmegaConf.to_container(cfg.data_loading.datamodule, resolve=True)
    write_json(output / "used_config.json", used)
    write_json(output / "preflight_formal_diff.json", {
        "engineering_only": not formal, "formal_state_reuse_allowed": False,
        "overrides": {} if formal else {
            "trainer.max_steps": {"preflight": 1, "formal": int(cfg.trainer.max_steps)},
            "trainer.val_check_interval": {"preflight": accumulation, "formal": int(cfg.trainer.val_check_interval)},
            "trainer.limit_test_batches": {"preflight": 2, "formal": "absent; complete testing"}},
        "same_callbacks_and_logger": True, "validation_is_complete_in_both_phases": True})

    preserved = [hydra.utils.instantiate(value) for value in cfg.callbacks.values()
                 if value is not None and "_target_" in value]
    checkpoint_callbacks = [value for value in preserved if isinstance(value, ModelCheckpoint)]
    early_callbacks = [value for value in preserved if isinstance(value, EarlyStopping)]
    summaries = [value for value in preserved if isinstance(value, RichModelSummary)]
    if len(checkpoint_callbacks) != 1 or len(early_callbacks) != 1 or len(summaries) != 1 or len(preserved) != 3:
        raise ValueError("Expected exactly the original checkpoint, early stopping and rich summary")
    best_callback, early_callback = checkpoint_callbacks[0], early_callbacks[0]
    if (best_callback.monitor, best_callback.mode, best_callback.save_top_k) != ("val/recall@5", "max", 1):
        raise ValueError("Original validation checkpoint selection changed")
    if early_callback.patience != 10 or early_callback.min_delta != 0 or early_callback.monitor != "val/recall@5":
        raise ValueError("Original early-stopping contract changed")
    loggers = [hydra.utils.instantiate(value) for value in cfg.logger.values()
               if value is not None and "_target_" in value]
    if len(loggers) != 1 or type(loggers[0]).__name__ != "CSVLogger":
        raise ValueError("This entry uses only the original local CSV logger")
    watched = dict(model.item_aggregation_mlp.named_parameters())
    initial_mlp = {name: p.detach().cpu().clone() for name, p in watched.items()}

    class Trace(Callback):
        def __init__(self):
            self.stage = "initialization"
            self.microbatches = 0
            self.total_windows = 0
            self.pending_windows = []
            self.pending_losses = []
            self.optimizer_event = None
            self.last_step = 0
            self.validation_count = 0
            self.validation_rows = 0
            self.seen = Counter()
            self.test_seen = Counter()
            self.recorder = None
            self.saving = False
            self.exception_snapshot_attempted = False
            self.recovery_count = 0
            self.weights_source = "current training state"

        def check_budget(self):
            free = shutil.disk_usage(output).free
            if free < minimum_free:
                raise SafetyInterruption("DISK_SAFETY_MARGIN: " + str(free))
            if time.time() >= deadline["internal_stop_timestamp"]:
                raise SafetyInterruption("TIME_ALLOWANCE_EXHAUSTED_WITH_CHECKPOINT_RESERVE")
            return free

        def save_recovery(self, trainer, tag, reason):
            if self.saving:
                return None
            self.saving = True
            self.recovery_count += 1
            folder = output / "recovery" / (tag + "-" + str(self.recovery_count))
            try:
                folder.mkdir(parents=True, exist_ok=False)
                numpy_state = np.random.get_state()
                write_json(folder / "python_numpy_rng.json", {
                    "python_random_state": random.getstate(),
                    "numpy_random_state": {"bit_generator": numpy_state[0],
                        "keys": numpy_state[1].tolist(), "position": int(numpy_state[2]),
                        "has_gauss": int(numpy_state[3]), "cached_gaussian": float(numpy_state[4])}})
                torch.save({"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()},
                           folder / "torch_rng.pt")
                path = folder / "full-state.ckpt"
                checkpoint_error = None
                try:
                    trainer.save_checkpoint(path, weights_only=False)
                except BaseException as error:
                    checkpoint_error = repr(error)
                state = {
                    "reason": reason, "stage": self.stage, "formal": formal,
                    "global_step": int(trainer.global_step), "epoch": int(trainer.current_epoch),
                    "microbatches_completed": self.microbatches,
                    "microbatches_into_unfinished_accumulation": self.microbatches % accumulation,
                    "partial_parameter_gradients_in_checkpoint": False,
                    "explicit_rng_capture": "immediately before checkpoint serialization",
                    "datamodule_state_dict": dm.state_dict(),
                    "datamodule_state_is_empty": dm.state_dict() == {},
                    "infinite_worker_cursor_saved": False,
                    "worker_process_RNG_and_prefetch_queue_saved": False,
                    "exact_uninterrupted_resume_claim": False,
                    "automatic_resume_allowed": False,
                    "weights_source": self.weights_source,
                    "optimizer_source": "final/current training optimizer; not rewound to validation best",
                    "resume_requires_explicit_decision": True,
                    "checkpoint_error": checkpoint_error,
                    "checkpoint": {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
                                  if checkpoint_error is None and path.is_file() else None,
                    "rng_files": [{"path": str(folder / name), "sha256": sha256(folder / name)}
                                  for name in ("python_numpy_rng.json", "torch_rng.pt")]}
                write_json(folder / "recovery.json", state)
                write_json(output / "latest_recovery.json", state)
                return state
            finally:
                self.saving = False

        def on_train_start(self, trainer, pl_module):
            self.stage = "train"
            self.check_budget()
            torch.cuda.synchronize()
            self.last_update_time = time.monotonic()
            loader = trainer.train_dataloader[0]
            if (loader.num_workers, loader.batch_size, loader.drop_last) != (int(train_config.num_workers), raw_batch, bool(train_config.drop_last)):
                raise ValueError("Actual approved training loader settings differ")
            if not loader.dataset.assign_all_files_per_worker or not loader.dataset.is_for_training:
                raise ValueError("Original infinite training stream contract differs")
            write_json(output / "actual_train_loader.json", {
                "num_workers": loader.num_workers, "raw_microbatch": loader.batch_size,
                "drop_last": loader.drop_last, "persistent_workers": loader.persistent_workers,
                "assigned_files": loader.dataset.list_of_file_paths,
                "assign_all_files_per_worker": loader.dataset.assign_all_files_per_worker,
                "collate_window_cap": window_cap})

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            self.stage = "train"
            self.check_budget()

        def on_before_optimizer_step(self, trainer, pl_module, optimizer):
            self.check_budget()
            actual = {id(p) for group in optimizer.param_groups for p in group["params"]}
            trainable = {id(p) for p in pl_module.parameters() if p.requires_grad}
            if actual != trainable:
                raise ValueError("Optimizer does not cover every trainable parameter")
            gradients = [p.grad for p in pl_module.parameters() if p.grad is not None]
            if not gradients or not bool(torch.stack([torch.isfinite(g).all() for g in gradients]).all()):
                raise FloatingPointError("Nonfinite or absent optimizer gradients")
            self.optimizer_event = {"step_before": int(trainer.global_step),
                "optimizer_covers_every_trainable_parameter": True,
                "parameter_count": sum(p.numel() for group in optimizer.param_groups for p in group["params"]),
                "available_gradients_finite": True,
                "mlp_gradient_norms": {name: float(p.grad.detach().norm()) if p.grad is not None else None
                                       for name, p in watched.items()}}

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            data = batch[0]
            x = data[0].transformed_sequences["sequence_data"]
            if not 0 < x.shape[0] <= window_cap or x.shape[1] != sequence_tokens:
                raise ValueError("Original training window shape differs")
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
            loss_value = float(loss.detach().cpu())
            if not math.isfinite(loss_value):
                raise FloatingPointError("Nonfinite training loss")
            self.microbatches += 1
            self.total_windows += int(x.shape[0])
            self.pending_windows.append(int(x.shape[0]))
            self.pending_losses.append(loss_value)
            if trainer.global_step != self.last_step:
                if self.microbatches != int(trainer.global_step) * accumulation or len(self.pending_windows) != accumulation:
                    raise ValueError("Optimizer updates do not match the approved accumulation microbatches")
                torch.cuda.synchronize()
                now = time.monotonic()
                event = {"optimizer_update": int(trainer.global_step), "microbatches": accumulation,
                    "cumulative_microbatches": self.microbatches,
                    "cumulative_expanded_windows": self.total_windows,
                    "expanded_windows": self.pending_windows,
                    "returned_training_losses": self.pending_losses,
                    "loss_observation": "Lightning batch-end returned losses; original model train/loss remains in CSV",
                    "wall_seconds_since_previous_update_including_intervening_validation": now - self.last_update_time,
                    "optimizer": self.optimizer_event}
                append_json(output / "optimizer_updates.jsonl", event)
                progress(output, {"stage": "train", "global_step": int(trainer.global_step),
                    "microbatches": self.microbatches, "expanded_windows": self.total_windows,
                    "last_returned_loss": loss_value, "free_bytes": shutil.disk_usage(output).free})
                self.pending_windows, self.pending_losses = [], []
                self.last_step, self.last_update_time = int(trainer.global_step), now
            self.check_budget()

        def on_validation_start(self, trainer, pl_module):
            self.stage = "validation"
            self.check_budget()
            self.seen = Counter()
            self.validation_rows = 0
            self.val_batches = 0
            self.val_shapes = Counter()
            torch.cuda.synchronize()
            self.validation_start = time.monotonic()
            loaders = trainer.val_dataloaders
            loader = loaders[0] if isinstance(loaders, (list, tuple)) else loaders
            if (loader.num_workers, loader.batch_size, loader.drop_last) != (int(val_config.num_workers), int(val_config.batch_size_per_device), bool(val_config.drop_last)):
                raise ValueError("Actual full validation loader settings differ")
            if loader.dataset.assign_all_files_per_worker or trainer.limit_val_batches != 1.0:
                raise ValueError("Validation is duplicated or limited")

        def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
            self.check_budget()
            keys = batch_keys(batch)
            targets = batch[1].labels["sequence_data"].detach().cpu().reshape(len(keys), 4).tolist()
            self.seen.update((key, tuple(target)) for key, target in zip(keys, targets))
            self.validation_rows += len(keys)
            self.val_batches += 1
            self.val_shapes[len(keys)] += 1
            if batch_idx % 256 == 0:
                progress(output, {"stage": "validation", "global_step": int(trainer.global_step),
                    "batch_idx": batch_idx, "observed_targets": self.validation_rows})

        def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
            if not bool(torch.isfinite(outputs[1]).all()) or not math.isfinite(float(pl_module.val_loss.compute())):
                raise FloatingPointError("Nonfinite validation score/loss")
            self.check_budget()

        def on_validation_end(self, trainer, pl_module):
            self.check_budget()
            if self.seen != expected_val:
                raise ValueError("Validation key-target multiset differs from complete metadata")
            metrics = {key: float(value) for key, value in trainer.callback_metrics.items()
                       if key.startswith("val/")}
            if not metrics or not all(math.isfinite(v) for v in metrics.values()):
                raise FloatingPointError("Validation metrics are absent or nonfinite")
            if "val/recall@5" not in metrics:
                raise ValueError("Checkpoint monitor was not produced")
            torch.cuda.synchronize()
            self.validation_count += 1
            record = {"validation_number": self.validation_count, "global_step": int(trainer.global_step),
                "exact_key_target_multiset": True, "targets": self.validation_rows,
                "unique_users": len({key[0] for key in self.seen}), "batches": self.val_batches,
                "batch_size_histogram": dict(self.val_shapes), "metrics": metrics,
                "hook_seconds": time.monotonic() - self.validation_start, "expected_index": val_index}
            append_json(output / "validations.jsonl", record)
            write_json(output / "latest_validation.json", record)

        def on_test_start(self, trainer, pl_module):
            self.stage = "test"
            self.check_budget()
            self.test_seen = Counter()
            loaders = trainer.test_dataloaders
            loader = loaders[0] if isinstance(loaders, (list, tuple)) else loaders
            if (loader.num_workers, loader.batch_size, loader.drop_last) != (
                    int(test_config.num_workers), int(test_config.batch_size_per_device), bool(test_config.drop_last)):
                raise ValueError("Actual full test loader settings differ")
            if loader.dataset.assign_all_files_per_worker or trainer.limit_test_batches != 1.0:
                raise ValueError("Test is duplicated or limited")

        def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
            self.check_budget()
            keys = batch_keys(batch)
            self.recorder.current_keys = keys
            targets = batch[1].labels["sequence_data"].detach().cpu().reshape(len(keys), 4).tolist()
            self.test_seen.update((key, tuple(target)) for key, target in zip(keys, targets))

        def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
            if not bool(torch.isfinite(outputs[1]).all()) or not math.isfinite(float(pl_module.test_loss.compute())):
                raise FloatingPointError("Nonfinite test score/loss")
            if batch_idx % 256 == 0:
                progress(output, {"stage": "test", "batch_idx": batch_idx,
                    "recorded_predictions": self.recorder.count})
            self.check_budget()

        def on_exception(self, trainer, pl_module, exception):
            if not self.exception_snapshot_attempted:
                self.exception_snapshot_attempted = True
                try:
                    self.save_recovery(trainer, "interrupted", repr(exception))
                except BaseException as save_error:
                    write_json(output / "recovery_error.json", {"error": repr(save_error),
                        "original_exception": repr(exception), "automatic_resume_allowed": False})
            pl_module.clear_runtime_state()

    trace = Trace()
    trainer = hydra.utils.instantiate(trainer_cfg, callbacks=[trace, *preserved], logger=loggers)
    torch.cuda.reset_peak_memory_stats()
    fit_start = time.monotonic()
    try:
        trainer.fit(model, datamodule=dm, ckpt_path=None)
        trace.check_budget()
        if trace.validation_count == 0:
            raise ValueError("Fit ended without a full validation/checkpoint selection")
        if int(trainer.global_step) >= int(trainer.max_steps):
            termination = "configured_max_steps"
        elif trainer.max_epochs >= 0 and int(trainer.current_epoch) >= int(trainer.max_epochs):
            termination = "configured_max_epochs"
        elif trainer.should_stop and early_callback.wait_count >= early_callback.patience:
            termination = "original_patience10_validation_early_stop"
        else:
            raise RuntimeError("Training stopped without an authorized research termination condition")
        if not formal and (trainer.global_step != 1 or trace.microbatches != accumulation):
            raise ValueError("Lifecycle check did not execute exactly one approved accumulation update")
        updates = {name: float((p.detach().cpu() - initial_mlp[name]).abs().max())
                   for name, p in watched.items()}
        if not any(value > 0 for value in updates.values()):
            raise ValueError("Aggregation MLP parameters did not update")
        final_recovery = trace.save_recovery(trainer, "training-final", "Fit reached " + termination)
        if final_recovery is None or final_recovery["checkpoint"] is None:
            raise RuntimeError("Required final optimizer/callback recovery checkpoint was not saved")
        best_path = Path(best_callback.best_model_path)
        if not best_callback.best_model_path or not best_path.is_file():
            raise ValueError("Original ModelCheckpoint did not save a best validation checkpoint")
        best = {"path": str(best_path), "sha256": sha256(best_path), "bytes": best_path.stat().st_size,
                "monitor": best_callback.monitor, "mode": best_callback.mode,
                "score": float(best_callback.best_model_score), "engineering_only": not formal}
        if not math.isfinite(best["score"]):
            raise FloatingPointError("Nonfinite best validation score")
        state = load_checkpoint(best_path)
        if len(state.get("optimizer_states", [])) != 1 or not state.get("callbacks"):
            raise ValueError("Original best checkpoint lacks optimizer/callback state")
        embedded_sid = state.get("hyper_parameters", {}).get("codebooks")
        selected_sid = torch.load(spec["sid"]["path"], map_location="cpu", weights_only=True)
        if not isinstance(embedded_sid, torch.Tensor) or not torch.equal(embedded_sid.cpu(), selected_sid.cpu()):
            raise ValueError("Best checkpoint embedded SID does not match the approved asset")
        loaded = model.load_state_dict(state["state_dict"], strict=True)
        actual = model.state_dict()
        if set(actual) != set(state["state_dict"]) or any(
            actual[key].shape != value.shape or not torch.equal(actual[key].detach().cpu(), value.detach().cpu())
            for key, value in state["state_dict"].items()):
            raise ValueError("Strict best checkpoint load changed parameter keys/shapes/values")
        if model.get_embedding_table("encoder") is not model.get_embedding_table("decoder"):
            raise ValueError("Strict best load changed shared-table identity")
        write_json(output / "best_checkpoint_load.json", {
            "protocol": "weights_only=True with checkpoint-local explicitly reviewed safe_globals",
            "checkpoint": best, "missing_keys": list(loaded.missing_keys),
            "unexpected_keys": list(loaded.unexpected_keys), "all_keys_shapes_values_exact": True,
            "embedded_sid_exact": True, "optimizer_and_callbacks_present": True,
            "state_tensor_count": len(actual), "best_global_step": int(state["global_step"]),
            "selected_by": "validation val/recall@5 only; original ModelCheckpoint"})
        trace.weights_source = "strictly loaded validation-best checkpoint " + best["sha256"]
        training_result = {"status": "TRAINING_COMPLETE_TEST_PENDING" if formal else "ENGINEERING_TEST_PENDING",
            "formal": formal, "engineering_only": not formal, "training_complete": formal,
            "test_complete": False, "fresh_initialization": True, "full_parameter_training": True,
            "termination": termination, "safety_allowance_is_not_training_endpoint": True,
            "global_step": int(trainer.global_step), "current_epoch": int(trainer.current_epoch),
            "actual_microbatches": trace.microbatches, "expanded_windows": trace.total_windows,
            "completed_full_validations": trace.validation_count,
            "fit_wall_seconds": time.monotonic() - fit_start,
            "early_stopping": {"wait_count": early_callback.wait_count,
                "patience": early_callback.patience, "stopped_epoch": early_callback.stopped_epoch,
                "trainer_should_stop": bool(trainer.should_stop)},
            "best_checkpoint": best, "final_recovery": final_recovery,
            "mlp_parameter_max_abs_updates": updates, "formal_state_reuse_allowed": False,
            "approved_config_asset": spec["config"], "sid_asset": spec["sid"],
            "evaluation_index": val_index, "test_index": test_index}
        write_json(output / "training_result.json", training_result)
        trace.check_budget()
        test_output = output / "test"
        test_output.mkdir()
        original_evaluator = model.evaluator
        trace.recorder = PredictionRecorder(original_evaluator, test_output / "predictions.jsonl")
        model.evaluator = trace.recorder
        try:
            metrics = trainer.test(model, datamodule=dm, ckpt_path=None)
        finally:
            model.evaluator = original_evaluator
            model.clear_runtime_state()
        if not all(math.isfinite(float(value)) for record in metrics for value in record.values()):
            raise FloatingPointError("Nonfinite final test metrics")
        if formal:
            if trace.test_seen != expected_test:
                raise ValueError("Formal test did not cover exact complete key-target multiset")
        elif trace.test_seen - expected_test:
            raise ValueError("Engineering test rows are not a subset of the original test metadata")
        write_json(test_output / "metrics.json", {"metrics": metrics, "samples": trace.recorder.count,
            "formal": formal, "checkpoint_load": "strict exact validation-best weights before Trainer.test",
            "complete_test_coverage": formal, "bounded_test_batches": None if formal else 2})
        trace.check_budget()
        verified_metrics = recompute(spec, cfg, test_output / "predictions.jsonl", test_output,
            coverage_index=test_index["path"] if formal else None,
            trainer_metrics=test_output / "metrics.json")
        trace.check_budget()
        training_result.update({"status": "COMPLETE" if formal else "LIFECYCLE_CHECK_PASS",
            "test_complete": formal, "engineering_test_complete": not formal,
            "test_target_count": trace.recorder.count, "test_exact_key_target_multiset": formal,
            "test_metrics": verified_metrics,
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "attention_collected": False})
        write_json(output / "training_result.json", training_result)
        progress(output, {"stage": "complete", "status": training_result["status"],
            "global_step": int(trainer.global_step), "formal": formal})
        return training_result
    except BaseException as error:
        if not trace.exception_snapshot_attempted:
            trace.on_exception(trainer, model, error)
        path = output / "training_result.json"
        interrupted = json.loads(path.read_text()) if path.is_file() else {
            "formal": formal, "training_complete": False, "test_complete": False}
        interrupted.update({"status": "INCOMPLETE" if isinstance(error, SafetyInterruption) else "FAILED",
            "error": repr(error), "failure_stage": trace.stage, "automatic_resume_allowed": False,
            "safety_allowance_is_not_training_endpoint": True})
        write_json(path, interrupted)
        raise
