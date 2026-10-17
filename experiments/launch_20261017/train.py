"""Frozen, scratch dual-table Sports training using the audited Beauty A bridge.

The launch runner owns deployment identity, user/nonauthor gates and GPU locks.
This entry never tests, initializes from a recommender, or resumes a formal run.
Training success means authorized termination + complete validation + strict best
checkpoint gate. Resource exhaustion remains INCOMPLETE. Admission evidence is
separate and can never supply weights or optimizer state to formal training.
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
    REPO, batch_keys, load_checkpoint, load_config, sha256, verify_asset, write_json,
)
from experiments.formal_20261015.train import (
    SafetyInterruption, append_json, deadline_record, progress, read_index,
)

SOURCE_CONFIG = "6f336505677ded65b16023e0cb798df2497f9c6c8fb34ff400dcb5ccc8e0dc9d"
WORKFLOWS = {
    "Sports-A1", "Sports-A2", "Sports-A1-QwenSID-DualTable-NewTraining",
    "Sports-A2-T5RKMeans-DualTable-NewTraining",
}


def configure(spec, output):
    """Relocate Beauty A inputs and assert the one shared, source-bound protocol."""
    from omegaconf import OmegaConf

    output = Path(output)
    if spec.get("route") != "A" or spec.get("domain") != "sports" or spec.get("workflow") not in WORKFLOWS:
        raise ValueError("Only the two approved new Sports dual-table workflows are allowed")
    if not spec.get("formal_approved"):
        raise ValueError("Explicit authorization is required, including for isolated admission")
    if spec.get("checkpoint") or spec.get("resume_checkpoint") or spec.get("shuffle_n", 0):
        raise ValueError("Sports training/admission must start fresh, without Shuffle")
    if spec.get("engineering_overlay") or spec.get("preflight_rows"):
        raise ValueError("A formal base config cannot contain a hidden development overlay")
    if spec["config"]["sha256"] != SOURCE_CONFIG:
        raise ValueError("Model/data protocol must derive from the actual audited Beauty A config")
    protocol_asset = verify_asset(spec["training_protocol"])
    bundled = Path(__file__).with_name("training_protocol.json")
    if protocol_asset["sha256"] != sha256(bundled):
        raise ValueError("A1/A2 training protocol differs from the frozen common protocol")
    protocol = json.loads(bundled.read_text())
    verify_asset(spec["sid"])
    cfg = load_config(spec, output)
    cfg.train, cfg.test, cfg.ckpt_path = True, False, None
    cfg.trainer.accumulate_grad_batches = protocol["accumulate_grad_batches"]
    cfg.trainer.val_check_interval = protocol["val_check_interval_train_loader_batches"]
    cfg.model.should_check_prefix = False
    cfg.model.top_k_for_generation = 10
    cfg.model.shuffle_embedding_on_test = False
    if cfg.model._target_ != protocol["model_target"]:
        raise ValueError("Audited author dual-table adapter was not selected")
    if cfg.seed != 42 or cfg.trainer.precision != "32-true":
        raise ValueError("Fixed seed or FP32 precision differs")
    if cfg.trainer.devices != 1 or cfg.trainer.strategy != "auto" or cfg.trainer.num_nodes != 1:
        raise ValueError("Only single-GPU, single-node new training is authorized")
    if (cfg.trainer.max_steps, cfg.trainer.max_epochs, cfg.trainer.num_sanity_val_steps) != (320000, 10, 0):
        raise ValueError("Source training endpoints or sanity-validation settings differ")
    if any(name.startswith("limit_") for name in cfg.trainer) or cfg.trainer.get("fast_dev_run", False) or cfg.trainer.get("overfit_batches", 0):
        raise ValueError("Formal configuration contains a development limit")
    if (cfg.optim.optimizer._target_, cfg.optim.optimizer.lr, cfg.optim.optimizer.weight_decay, cfg.optim.scheduler) != (
            "torch.optim.Adam", 0.001, 0.0001, None):
        raise ValueError("Fixed Adam/no-scheduler protocol differs")
    if cfg.loss.loss_function._target_ != "torch.nn.CrossEntropyLoss":
        raise ValueError("The original cross-entropy objective must be retained")
    if int(cfg.num_hierarchies) != 4 or int(cfg.sequence_length) != 120:
        raise ValueError("Source H4 or sequence120 differs")
    train = cfg.data_loading.train_dataloader_config.dataloader
    if (int(train.batch_size_per_device), int(train.collate_fn.max_batch_size)) != (16, 512):
        raise ValueError("Original raw16/causal-window-cap512 changed; memory fixes must not shrink it")
    if not train.assign_all_files_per_worker or not train.drop_last or not train.should_shuffle_rows:
        raise ValueError("Original infinite training stream/file assignment differs")
    source = OmegaConf.load(spec["config"]["path"])
    for split in ("train", "val", "test"):
        loader = cfg.data_loading[f"{split}_dataloader_config"].dataloader
        old = source.data_loading[f"{split}_dataloader_config"].dataloader
        for field in ("num_workers", "timeout", "persistent_workers"):
            if loader.get(field) != old.get(field):
                raise ValueError("Source worker setting changed: " + split + "." + field)
        if split != "train" and (loader.drop_last or loader.should_shuffle_rows or loader.get("assign_all_files_per_worker", False)):
            raise ValueError("Validation must enumerate the complete nonduplicated evaluation split")
    checkpoint = cfg.callbacks.model_checkpoint
    early = cfg.callbacks.early_stopping
    if (checkpoint.monitor, checkpoint.mode, checkpoint.save_top_k, checkpoint.save_weights_only) != (
            "val/recall@5", "max", 1, False):
        raise ValueError("Original validation-best checkpoint policy differs")
    if (early.monitor, early.mode, early.patience, early.min_delta, early.check_on_train_epoch_end) != (
            "val/recall@5", "max", 10, 0, False):
        raise ValueError("Patience must count ten complete validation events")
    if early.stopping_threshold is not None or early.divergence_threshold is not None:
        raise ValueError("Unapproved metric-threshold training stop")
    write_json(output / "training_protocol.json", {"protocol": protocol, "asset": spec["training_protocol"],
        "source_config": spec["config"], "selected_sid": spec["sid"],
        "all_parameters_fresh_random": True, "no_test_execution_in_this_stage": True})
    OmegaConf.save(cfg, output / "effective_config.yaml", resolve=False)
    used = {key: OmegaConf.to_container(cfg[key], resolve=True)
            for key in ("model", "trainer", "optim", "eval", "callbacks", "logger")}
    used["datamodule"] = OmegaConf.to_container(cfg.data_loading.datamodule, resolve=True)
    write_json(output / "used_config.json", used)
    return cfg


def _fresh_model(cfg, output):
    import hydra
    import torch
    from lightning import seed_everything
    from experiments.beauty_extensions.author.checks import describe_author_model

    seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("medium")  # preserved author train.py setting
    model = hydra.utils.instantiate(cfg.model)
    if not all(p.requires_grad for p in model.parameters()):
        raise ValueError("Every generator parameter must be trained from scratch")
    enc, dec = (model.get_embedding_table(name) for name in ("encoder", "decoder"))
    if enc is dec or enc.weight.data_ptr() == dec.weight.data_ptr() or torch.equal(enc.weight, dec.weight):
        raise ValueError("Fresh encoder and decoder SID tables must be independent random draws")
    if model.num_hierarchies != 4 or model.should_check_prefix or model.shuffle_embedding_on_test:
        raise ValueError("New training must retain unshuffled four-step native generation")
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("Generator parameters must use FP32")
    write_json(Path(output) / "structure.json", describe_author_model(model))
    return model


def _watched(model):
    return {"encoder_sid": model.get_embedding_table("encoder").weight,
            "decoder_sid": model.get_embedding_table("decoder").weight,
            "decoder_bos": model.decoder.bos_token,
            "encoder_transformer": next(model.encoder.parameters()),
            "decoder_transformer": next(model.decoder.decoder.parameters())}


def _strict_best(model, spec, path, score):
    import torch

    state = load_checkpoint(path)
    if len(state.get("optimizer_states", [])) != 1 or not state.get("callbacks"):
        raise ValueError("Best checkpoint lacks the original optimizer/callback state")
    sid = torch.load(spec["sid"]["path"], map_location="cpu", weights_only=True)
    embedded = state.get("hyper_parameters", {}).get("codebooks")
    if not isinstance(embedded, torch.Tensor) or not torch.equal(embedded.cpu(), sid.cpu()):
        raise ValueError("Best checkpoint embedded SID differs from the frozen training SID")
    loaded = model.load_state_dict(state["state_dict"], strict=True)
    actual = model.state_dict()
    if set(actual) != set(state["state_dict"]) or any(actual[k].shape != v.shape or
            not torch.equal(actual[k].detach().cpu(), v.detach().cpu()) for k, v in state["state_dict"].items()):
        raise ValueError("Strict best checkpoint keys/shapes/tensor values differ")
    if model.get_embedding_table("encoder") is model.get_embedding_table("decoder"):
        raise ValueError("Best checkpoint collapsed the independent SID tables")
    if not all(torch.isfinite(v).all() for v in state["state_dict"].values() if v.is_floating_point()):
        raise FloatingPointError("Best checkpoint has nonfinite tensors")
    if not math.isfinite(float(score)):
        raise FloatingPointError("Best validation score is nonfinite")
    return {"checkpoint": {"path": str(path), "sha256": sha256(path), "bytes": Path(path).stat().st_size,
                           "score": float(score), "monitor": "val/recall@5", "mode": "max"},
            "missing_keys": list(loaded.missing_keys), "unexpected_keys": list(loaded.unexpected_keys),
            "all_keys_shapes_values_exact": True, "embedded_sid_exact": True,
            "independent_sid_tables": True, "optimizer_and_callbacks_present": True,
            "best_global_step": int(state["global_step"]), "selected_by": "validation only"}


def run_training(spec, output, request=None, status=None):
    """Train on the original infinite stream, with full validation and no test."""
    import hydra
    import numpy as np
    import torch
    from lightning import Callback
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichModelSummary

    output = Path(output)
    cfg = configure(spec, output)
    if float(spec.get("hard_cap_hours", 12)) != 12:
        raise ValueError("The fixed Sports training wall allowance is twelve hours")
    deadline = deadline_record(spec, request, status)
    write_json(output / "deadline.json", deadline)
    expected_val, val_index = read_index(spec["evaluation_index"])
    # The test index is neither opened nor consumed during training selection.
    minimum_free = int(spec.get("min_free_bytes", 20 * 1024**3))
    if minimum_free < 20 * 1024**3:
        raise ValueError("Disk safety margin must be at least twenty GiB")
    model = _fresh_model(cfg, output)
    dm = hydra.utils.instantiate(cfg.data_loading.datamodule)
    watch = _watched(model)
    initial = {name: p.detach().cpu().clone() for name, p in watch.items()}
    accumulation = int(cfg.trainer.accumulate_grad_batches)
    train_config = cfg.data_loading.train_dataloader_config.dataloader
    val_config = cfg.data_loading.val_dataloader_config.dataloader
    preserved = [hydra.utils.instantiate(v) for v in cfg.callbacks.values() if v is not None and "_target_" in v]
    best_callbacks = [v for v in preserved if isinstance(v, ModelCheckpoint)]
    early_callbacks = [v for v in preserved if isinstance(v, EarlyStopping)]
    if len(best_callbacks) != 1 or len(early_callbacks) != 1 or len(preserved) != 3 or not any(isinstance(v, RichModelSummary) for v in preserved):
        raise ValueError("Only the original best checkpoint, early stop and summary callbacks are expected")
    best_callback, early_callback = best_callbacks[0], early_callbacks[0]
    loggers = [hydra.utils.instantiate(v) for v in cfg.logger.values() if v is not None and "_target_" in v]
    if len(loggers) != 1 or type(loggers[0]).__name__ != "CSVLogger":
        raise ValueError("Only the preserved local CSV logger is allowed")

    class Trace(Callback):
        def __init__(self):
            self.stage = "initialization"
            self.microbatches, self.total_windows, self.last_step = 0, 0, 0
            self.validation_count, self.validation_rows = 0, 0
            self.pending_windows, self.pending_losses = [], []
            self.seen = Counter()
            self.optimizer_event = None
            self.saving = False
            self.exception_snapshot_attempted = False
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
            try:
                folder = output / "recovery" / tag
                folder.mkdir(parents=True, exist_ok=False)
                ns = np.random.get_state()
                write_json(folder / "python_numpy_rng.json", {"python_random_state": random.getstate(),
                    "numpy_random_state": {"bit_generator": ns[0], "keys": ns[1].tolist(),
                    "position": int(ns[2]), "has_gauss": int(ns[3]), "cached_gaussian": float(ns[4])}})
                torch.save({"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}, folder / "torch_rng.pt")
                path = folder / "full-state.ckpt"
                trainer.save_checkpoint(path, weights_only=False)
                record = {"reason": reason, "stage": self.stage, "global_step": int(trainer.global_step),
                    "microbatches_completed": self.microbatches,
                    "microbatches_into_unfinished_accumulation": self.microbatches % accumulation,
                    "partial_parameter_gradients_in_checkpoint": False,
                    "datamodule_state_dict": dm.state_dict(), "infinite_worker_cursor_saved": False,
                    "worker_RNG_and_prefetch_queue_saved": False, "exact_uninterrupted_resume_claim": False,
                    "automatic_resume_allowed": False, "resume_requires_explicit_decision": True,
                    "weights_source": self.weights_source,
                    "checkpoint": {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}}
                write_json(folder / "recovery.json", record)
                write_json(output / "latest_recovery.json", record)
                return record
            finally:
                self.saving = False

        def on_train_start(self, trainer, pl_module):
            self.stage = "train"
            self.check_budget()
            if trainer.world_size != 1 or trainer.global_step != 0 or len(trainer.optimizers) != 1 or trainer.lr_scheduler_configs:
                raise ValueError("Formal training must be fresh single-GPU Adam, without scheduler")
            optimizer = trainer.optimizers[0]
            if type(optimizer) is not torch.optim.Adam or any(g["lr"] != 0.001 or g["weight_decay"] != 0.0001 for g in optimizer.param_groups):
                raise ValueError("Actual optimizer differs from the fixed protocol")
            if optimizer.state:
                raise ValueError("A fresh formal optimizer must not contain prior state")
            loader = trainer.train_dataloader[0]
            if (loader.num_workers, loader.batch_size, loader.drop_last) != (int(train_config.num_workers), 16, True):
                raise ValueError("Actual raw training loader differs")
            if not loader.dataset.assign_all_files_per_worker or not loader.dataset.is_for_training:
                raise ValueError("Original infinite training stream differs")
            write_json(output / "actual_train_loader.json", {"num_workers": loader.num_workers,
                "raw_microbatch": loader.batch_size, "drop_last": loader.drop_last,
                "persistent_workers": loader.persistent_workers, "assigned_files": loader.dataset.list_of_file_paths,
                "assign_all_files_per_worker": True, "collate_window_cap": 512,
                "single_gpu_world_size": trainer.world_size, "fresh_optimizer_state_empty": True})

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            self.stage = "train"
            self.check_budget()

        def on_before_optimizer_step(self, trainer, pl_module, optimizer):
            self.check_budget()
            if {id(p) for g in optimizer.param_groups for p in g["params"]} != {id(p) for p in pl_module.parameters() if p.requires_grad}:
                raise ValueError("Optimizer must cover every trainable parameter")
            gradients = [p.grad for p in pl_module.parameters() if p.grad is not None]
            if not gradients or not bool(torch.stack([torch.isfinite(g).all() for g in gradients]).all()):
                raise FloatingPointError("Nonfinite or absent gradients")
            norms = {name: float(p.grad.detach().norm()) if p.grad is not None else None for name, p in watch.items()}
            if any(norms[name] is None or norms[name] <= 0 for name in ("encoder_sid", "decoder_sid")):
                raise ValueError("Both independent SID tables must receive nonzero gradients")
            self.optimizer_event = {"step_before": int(trainer.global_step), "all_available_gradients_finite": True,
                "optimizer_covers_every_trainable_parameter": True, "watched_gradient_norms": norms}

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            x = batch[0][0].transformed_sequences["sequence_data"]
            if not 0 < x.shape[0] <= 512 or x.shape[1] != 120:
                raise ValueError("Original causal training window shape differs")
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
            value = float(loss.detach().cpu())
            if not math.isfinite(value):
                raise FloatingPointError("Nonfinite training loss")
            self.microbatches += 1
            self.total_windows += int(x.shape[0])
            self.pending_windows.append(int(x.shape[0]))
            self.pending_losses.append(value)
            if int(trainer.global_step) != self.last_step:
                if self.microbatches != int(trainer.global_step) * accumulation or len(self.pending_windows) != accumulation:
                    raise ValueError("global_step must count optimizer updates, not loader batches")
                append_json(output / "optimizer_updates.jsonl", {"at": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "optimizer_update": int(trainer.global_step), "cumulative_microbatches": self.microbatches,
                    "cumulative_expanded_windows": self.total_windows, "expanded_windows": self.pending_windows,
                    "returned_training_losses": self.pending_losses, "optimizer": self.optimizer_event})
                self.pending_windows, self.pending_losses = [], []
                self.last_step = int(trainer.global_step)
            if self.microbatches <= 2 or self.microbatches % accumulation == 0:
                progress(output, {"stage": "train", "global_step": int(trainer.global_step),
                    "microbatches": self.microbatches, "expanded_windows": self.total_windows,
                    "last_returned_loss": value, "free_bytes": self.check_budget()})
            self.check_budget()

        def on_validation_start(self, trainer, pl_module):
            self.stage = "validation"
            self.check_budget()
            self.seen, self.validation_rows, self.val_batches = Counter(), 0, 0
            self.validation_start = time.monotonic()
            loaders = trainer.val_dataloaders
            loader = loaders[0] if isinstance(loaders, (list, tuple)) else loaders
            if (loader.num_workers, loader.batch_size, loader.drop_last) != (int(val_config.num_workers), 8, False):
                raise ValueError("Full validation loader differs")
            if loader.dataset.assign_all_files_per_worker or trainer.limit_val_batches != 1.0:
                raise ValueError("Validation is duplicated or limited")
            if self.microbatches % 1600 != 0 or int(trainer.global_step) * accumulation != self.microbatches:
                raise ValueError("Validation did not occur at the approved loader-batch cadence")

        def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
            self.check_budget()
            keys = batch_keys(batch)
            targets = batch[1].labels["sequence_data"].detach().cpu().reshape(len(keys), 4).tolist()
            self.seen.update((key, tuple(target)) for key, target in zip(keys, targets))
            self.validation_rows += len(keys)
            self.val_batches += 1
            if batch_idx % 128 == 0:
                progress(output, {"stage": "validation", "global_step": int(trainer.global_step),
                    "batch_idx": batch_idx, "observed_targets": self.validation_rows})

        def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
            if not bool(torch.isfinite(outputs[1]).all()) or not math.isfinite(float(pl_module.val_loss.compute())):
                raise FloatingPointError("Nonfinite validation scores or loss")
            self.check_budget()

        def on_validation_end(self, trainer, pl_module):
            self.check_budget()
            if self.seen != expected_val:
                raise ValueError("Validation key-target multiset differs from complete metadata")
            metrics = {k: float(v) for k, v in trainer.callback_metrics.items() if k.startswith("val/")}
            if "val/recall@5" not in metrics or not all(math.isfinite(v) for v in metrics.values()):
                raise FloatingPointError("Full validation metrics absent or nonfinite")
            self.validation_count += 1
            record = {"at": dt.datetime.now(dt.timezone.utc).isoformat(), "validation_number": self.validation_count,
                "global_step": int(trainer.global_step), "exact_key_target_multiset": True,
                "targets": self.validation_rows, "batches": self.val_batches, "metrics": metrics,
                "hook_seconds": time.monotonic() - self.validation_start, "expected_index": val_index}
            append_json(output / "validations.jsonl", record)
            write_json(output / "latest_validation.json", record)

        def on_exception(self, trainer, pl_module, exception):
            if not self.exception_snapshot_attempted:
                self.exception_snapshot_attempted = True
                try:
                    self.save_recovery(trainer, "interrupted", repr(exception))
                except BaseException as error:
                    write_json(output / "recovery_error.json", {"error": repr(error), "original_exception": repr(exception)})
            pl_module.clear_runtime_state()

    trace = Trace()
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=[trace, *preserved], logger=loggers)
    trace.check_budget()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    try:
        trainer.fit(model, datamodule=dm, ckpt_path=None)
        trace.check_budget()
        if trace.validation_count == 0:
            raise ValueError("Training ended without a complete validation")
        if int(trainer.global_step) >= 320000:
            termination = "configured_max_steps"
        elif int(trainer.current_epoch) >= 10:
            termination = "configured_max_epochs"
        elif trainer.should_stop and early_callback.wait_count >= 10:
            termination = "patience10_complete_validation_early_stop"
        else:
            raise RuntimeError("Training stopped without an authorized termination condition")
        updates = {name: float((p.detach().cpu() - initial[name]).abs().max()) for name, p in watch.items()}
        if any(updates[name] <= 0 for name in ("encoder_sid", "decoder_sid")):
            raise ValueError("Both independent SID tables must have updated")
        recovery = trace.save_recovery(trainer, "training-final", termination)
        if recovery is None:
            raise RuntimeError("Final optimizer/callback recovery checkpoint was not saved")
        if not best_callback.best_model_path or not Path(best_callback.best_model_path).is_file():
            raise ValueError("No original validation-best checkpoint was saved")
        gate = _strict_best(model, spec, best_callback.best_model_path, best_callback.best_model_score)
        trace.weights_source = "strictly loaded validation-best checkpoint " + gate["checkpoint"]["sha256"]
        write_json(output / "best_checkpoint_load.json", gate)
        trace.check_budget()
        result = {"status": "TRAINING_COMPLETE_EVALUATION_PENDING", "formal": True,
            "training_complete": True, "test_complete": False, "attention_collected": False,
            "workflow": spec["workflow"], "fresh_initialization": True, "full_parameter_training": True,
            "independent_sid_tables": True, "termination": termination,
            "global_step": int(trainer.global_step), "current_epoch": int(trainer.current_epoch),
            "actual_microbatches": trace.microbatches, "expanded_windows": trace.total_windows,
            "completed_full_validations": trace.validation_count, "full_validation_identity_exact": True,
            "evaluation_index": val_index, "best_checkpoint": gate["checkpoint"], "best_checkpoint_gate": gate,
            "selected_by": "complete validation Recall@5 only; test never executed",
            "watched_parameter_max_abs_updates": updates, "final_recovery": recovery,
            "early_stopping": {"wait_count": early_callback.wait_count, "patience": 10},
            "fit_and_gate_wall_seconds": time.monotonic() - started,
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "sid": spec["sid"], "config": spec["config"], "training_protocol": spec["training_protocol"],
            "safety_allowance_is_not_training_endpoint": True, "automatic_resume_allowed": False}
        write_json(output / "training_result.json", result)
        progress(output, {"stage": "complete", "status": result["status"], "global_step": int(trainer.global_step)})
        return result
    except BaseException as error:
        trace.on_exception(trainer, model, error)
        write_json(output / "training_result.json", {"status": "INCOMPLETE" if isinstance(error, SafetyInterruption) else "FAILED",
            "formal": True, "training_complete": False, "test_complete": False,
            "error": repr(error), "failure_stage": trace.stage, "global_step": int(trainer.global_step),
            "automatic_resume_allowed": False, "safety_allowance_is_not_training_endpoint": True})
        raise
    finally:
        model.clear_runtime_state()


def admission(spec, output):
    """Small real lifecycle gate; CPU evidence explicitly leaves GPU capacity pending.

CPU uses two post-collation windows from a real raw16 author batch. GPU uses the
actual unmodified raw16/cap512 loader and original datamodule. Both train one
complete accumulation16 update, run two engineering validation batches, save
optimizer/callback state, and resume one additional update in a fresh model.
No test examples, scores or formal checkpoint initialization are involved.
"""
    import hydra
    import torch
    from lightning import Callback, LightningDataModule
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from experiments.beauty_extensions.common import make_loader

    output = Path(output)
    cfg = configure(spec, output)
    device = spec.get("admission_device", "cpu")
    if device not in ("cpu", "gpu"):
        raise ValueError("Admission device must be cpu or gpu")
    cpu = device == "cpu"
    if cpu:
        import dataclasses
        from src.data.loading.components.interfaces import SequentialModuleLabelData
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        def slice_rows(value, n=2):
            if isinstance(value, SequentialModuleLabelData):
                # NextKTokenMasking stores four target codes/locations per row
                # on a flattened first axis. Slice complete row groups here;
                # ordinary input/mask tensors below still have one entry/row.
                hierarchies = int(cfg.num_hierarchies)
                sliced_labels, sliced_locations = {}, {}
                for key, tensor in value.labels.items():
                    if tensor.ndim != 1 or tensor.numel() % hierarchies:
                        raise ValueError("CPU admission requires original flattened four-code targets")
                    sliced_labels[key] = tensor.reshape(-1, hierarchies)[:n].reshape(-1)
                for key, tensor in value.label_location.items():
                    if tensor.ndim != 2 or tensor.shape[1] != 2 or tensor.shape[0] % hierarchies:
                        raise ValueError("CPU admission requires original four target locations per row")
                    sliced_locations[key] = tensor.reshape(-1, hierarchies, 2)[:n].reshape(-1, 2)
                return type(value)(labels=sliced_labels, label_location=sliced_locations,
                                   attention_mask=slice_rows(value.attention_mask, n))
            if isinstance(value, torch.Tensor):
                return value[:n]
            if isinstance(value, dict):
                return {k: slice_rows(v, n) for k, v in value.items()}
            if dataclasses.is_dataclass(value):
                return type(value)(**{f.name: slice_rows(getattr(value, f.name), n) for f in dataclasses.fields(value)})
            if isinstance(value, list):
                return value[:n]
            if isinstance(value, tuple):
                return tuple(slice_rows(v, n) for v in value)
            return value
        original_train = next(iter(make_loader(cfg, "train")))
        original_shape = list(original_train[0].transformed_sequences["sequence_data"].shape)
        train_batch = slice_rows(original_train)
        val_batch = slice_rows(next(iter(make_loader(cfg, "val"))))
        batch_shapes = {}
        for split, batch in (("train", train_batch), ("val", val_batch)):
            inputs, labels = batch
            rows = inputs.transformed_sequences["sequence_data"].shape[0]
            target = labels.labels["sequence_data"]
            locations = labels.label_location["sequence_data"]
            if rows != 2 or target.shape != (rows * 4,) or locations.shape != (rows * 4, 2):
                raise ValueError("CPU admission input/target/location row groups differ")
            if not torch.equal(locations[:, 0], torch.arange(rows).repeat_interleave(4)):
                raise ValueError("CPU admission target locations do not cover each retained row four times")
            batch_shapes[split] = {"input": list(inputs.transformed_sequences["sequence_data"].shape),
                "mask": list(inputs.mask.shape), "flat_targets": list(target.shape),
                "four_code_targets": list(target.reshape(rows, 4).shape),
                "label_locations": list(locations.shape), "complete_four_code_groups": True}
        write_json(output / "cpu_admission_batch_shapes.json", batch_shapes)
        class BoundedDataModule(LightningDataModule):
            def train_dataloader(self):
                return (DataLoader([train_batch] * 64, batch_size=None, collate_fn=lambda value: value),)
            def val_dataloader(self):
                return (DataLoader([val_batch] * 2, batch_size=None, collate_fn=lambda value: value),)
        dm = BoundedDataModule()
    else:
        if not torch.cuda.is_available():
            raise ValueError("GPU capacity admission requires the real selected CUDA device")
        dm = hydra.utils.instantiate(cfg.data_loading.datamodule)
        original_shape = None
    traces = []
    class Check(Callback):
        def __init__(self):
            self.optimizer_events = []
            self.microbatches = 0
        def on_before_optimizer_step(self, trainer, pl_module, optimizer):
            norms = {n: float(p.grad.norm()) if p.grad is not None else None for n, p in _watched(pl_module).items()}
            if {id(p) for g in optimizer.param_groups for p in g["params"]} != {id(p) for p in pl_module.parameters()}:
                raise ValueError("Admission optimizer does not cover the full model")
            if any(norms[n] is None or not math.isfinite(norms[n]) or norms[n] <= 0 for n in ("encoder_sid", "decoder_sid")):
                raise ValueError("Admission SID table gradients are absent/nonfinite/zero")
            if any(not torch.isfinite(p.grad).all() for p in pl_module.parameters() if p.grad is not None):
                raise FloatingPointError("Nonfinite admission gradient")
            self.optimizer_events.append({"step_before": int(trainer.global_step), "gradient_norms": norms})
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            value = float((outputs["loss"] if isinstance(outputs, dict) else outputs).detach().cpu())
            if not math.isfinite(value):
                raise FloatingPointError("Nonfinite admission loss")
            self.microbatches += 1
            progress(output, {"stage": "engineering_admission", "device": device,
                "global_step": int(trainer.global_step), "microbatches": self.microbatches, "loss": value})
    resume = None
    for attempt in (0, 1):
        folder = output / ("first" if attempt == 0 else "resumed")
        folder.mkdir()
        model = _fresh_model(cfg, folder)
        initial = {n: p.detach().cpu().clone() for n, p in _watched(model).items()}
        if resume:
            state = load_checkpoint(resume)
            loaded = model.load_state_dict(state["state_dict"], strict=True)
            if loaded.missing_keys or loaded.unexpected_keys or any(not torch.equal(model.state_dict()[k].cpu(), v.cpu()) for k, v in state["state_dict"].items()):
                raise ValueError("Admission checkpoint strict restore differs")
            if len(state["optimizer_states"]) != 1:
                raise ValueError("Admission full optimizer state is missing")
            initial = {n: p.detach().cpu().clone() for n, p in _watched(model).items()}
        trainer_cfg = cfg.trainer.copy()
        trainer_cfg.accelerator, trainer_cfg.devices, trainer_cfg.strategy = device, 1, "auto"
        trainer_cfg.max_steps = attempt + 1
        trainer_cfg.max_epochs = 10
        trainer_cfg.val_check_interval = 16
        trainer_cfg.limit_val_batches = 2
        trainer_cfg.enable_model_summary = False
        trainer_cfg.sync_batchnorm = False if cpu else trainer_cfg.sync_batchnorm
        callback = ModelCheckpoint(dirpath=str(folder / "checkpoints"), save_last=True, save_top_k=1,
            monitor="val/recall@5", mode="max", save_on_train_epoch_end=False,
            filename="engineering-{step:06d}")
        trace = Check()
        OmegaConf.save(trainer_cfg, folder / "actual_engineering_trainer.yaml", resolve=True)
        trainer = hydra.utils.instantiate(trainer_cfg, callbacks=[trace, callback], logger=CSVLogger(str(folder), name="csv"))
        if not cpu:
            torch.cuda.reset_peak_memory_stats()
        trainer.fit(model, datamodule=dm, ckpt_path=resume)
        if int(trainer.global_step) != attempt + 1 or trace.microbatches != 16 or len(trace.optimizer_events) != 1:
            raise ValueError("Admission did not execute exactly one complete accumulation16 update")
        changes = {n: float((p.detach().cpu() - initial[n]).abs().max()) for n, p in _watched(model).items()}
        if any(changes[n] <= 0 for n in ("encoder_sid", "decoder_sid")):
            raise ValueError("Both admission SID tables must update")
        if not callback.best_model_path:
            raise ValueError("Admission validation-best selection did not save a checkpoint")
        selected = _strict_best(model, spec, callback.best_model_path, callback.best_model_score)
        resume = str(folder / "full-state.ckpt")
        trainer.save_checkpoint(resume, weights_only=False)
        traces.append({"attempt": attempt, "global_step": int(trainer.global_step), "microbatches": trace.microbatches,
            "optimizer_events": trace.optimizer_events, "parameter_changes": changes, "selected": selected,
            "full_state": {"path": resume, "sha256": sha256(resume)},
            "peak_reserved_bytes": torch.cuda.max_memory_reserved() if not cpu else None})
        model.clear_runtime_state()
        del trainer, model
        if not cpu:
            torch.cuda.empty_cache()
    value = {"status": "CPU_ADMISSION_PASS_GPU_PENDING" if cpu else "GPU_ADMISSION_PASS",
        "engineering_only": True, "formal_state_reuse_allowed": False, "device": device,
        "gpu_capacity_verified": not cpu, "cpu_post_collation_windows": 2 if cpu else None,
        "original_raw16_batch_collated_shape": original_shape, "unchanged_formal_collate_cap": 512,
        "full_validation_executed": False, "validation_batches_engineering_only": 2,
        "test_executed": False, "strict_checkpoint_and_optimizer_resume": True,
        "independent_new_model_for_resume": True, "traces": traces,
        "training_protocol": spec["training_protocol"], "sid": spec["sid"]}
    write_json(output / "admission.json", value)
    return value
