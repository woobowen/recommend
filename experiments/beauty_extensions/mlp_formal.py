"""Original Beauty MLP lifecycle check and independently admitted fresh training.

The engineering phase changes only max_steps, validation cadence and test limit.
The formal phase requires an exact plan/source gate. Neither phase resumes any
state. Safety interruptions are incomplete executions, never training endpoints.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

from .common import REPO, bind_source, sha256, verify_asset, write_json


class SafetyInterruption(RuntimeError):
    """Time/space exhaustion is an incomplete run, not early stopping."""


def plan_sha256(spec):
    value = {key: item for key, item in spec.items() if key != "nonauthor_gate"}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


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


def deployment_and_job(spec_path, formal):
    deployment_path = REPO / "DEPLOYMENT.json"
    deployment = json.loads(deployment_path.read_text())
    for record in deployment["files"]:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid deployment member")
        path = REPO / relative
        if path.is_symlink() or not path.is_file() or sha256(path) != record["sha256"]:
            raise ValueError("Deployment bytes differ: " + str(relative))
    job_dir = Path(os.environ["GRID_JOB_DIR"]) if os.environ.get("GRID_JOB_DIR") else None
    request = status = None
    if job_dir is not None:
        request = json.loads((job_dir / "request.json").read_text())
        status = json.loads((job_dir / "status.json").read_text())
        identity = deployment["source_commit"] + "/" + deployment["snapshot_sha256"]
        if request["capture_version"] != identity or Path(request["cwd"]).resolve() != REPO.resolve():
            raise ValueError("Runner request is not bound to this deployment")
        required = {str(deployment_path): sha256(deployment_path),
                    str(spec_path.resolve()): sha256(spec_path)}
        inputs = {r["path"]: r["sha256"] for r in request["inputs"]}
        if any(inputs.get(path) != digest for path, digest in required.items()):
            raise ValueError("Runner lacks exact deployment/spec input identity")
        if "experiments.beauty_extensions.mlp_formal" not in request["argv"]:
            raise ValueError("Runner argv is not this independent entrypoint")
    elif formal:
        raise ValueError("Formal execution requires the accepted detached runner identity")
    return deployment, job_dir, request, status


def verify_gate(spec, deployment):
    verified = verify_asset(spec["nonauthor_gate"])
    gate = json.loads(Path(verified["path"]).read_text())
    if gate.get("status") != "NONAUTHOR_GATE_PASS" or gate.get("plan_sha256") != plan_sha256(spec):
        raise ValueError("Nonauthor gate does not admit this exact formal plan")
    records = gate.get("source_files", [])
    named = {record["path"]: record["sha256"] for record in records}
    if len(named) != len(records):
        raise ValueError("Gate source identity has duplicate members")
    required = {record["path"]: record["sha256"] for record in deployment["files"]
                if record["path"].endswith(".py") and record["path"].startswith((
                    "references/wmh-original/GRID/src/", "experiments/beauty_extensions/"))}
    if not required or any(named.get(path) != digest for path, digest in required.items()):
        raise ValueError("Gate does not bind every deployed research Python source")
    for name, digest in named.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or sha256(REPO / relative) != digest:
            raise ValueError("Gated source bytes differ: " + name)
    return {"gate_asset": verified, "plan_sha256": plan_sha256(spec),
            "bound_python_source_count": len(required), "review": gate}


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
    from .author.checks import describe_author_model
    from .cli import PredictionRecorder
    from .common import batch_keys, load_checkpoint
    from .results import recompute

    formal = phase == "formal"
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
        trainer_cfg.val_check_interval = 64
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
            "trainer.max_steps": {"preflight": 1, "formal": 320000},
            "trainer.val_check_interval": {"preflight": 64, "formal": 6400},
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
                    "microbatches_into_unfinished_accumulation": self.microbatches % 64,
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
            if (loader.num_workers, loader.batch_size, loader.drop_last) != (8, 32, True):
                raise ValueError("Actual approved training loader settings differ")
            if not loader.dataset.assign_all_files_per_worker or not loader.dataset.is_for_training:
                raise ValueError("Original infinite training stream contract differs")
            write_json(output / "actual_train_loader.json", {
                "num_workers": loader.num_workers, "raw_microbatch": loader.batch_size,
                "drop_last": loader.drop_last, "persistent_workers": loader.persistent_workers,
                "assigned_files": loader.dataset.list_of_file_paths,
                "assign_all_files_per_worker": loader.dataset.assign_all_files_per_worker,
                "collate_window_cap": 512})

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
            if not 0 < x.shape[0] <= 512 or x.shape[1] != 120:
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
                if self.microbatches != int(trainer.global_step) * 64 or len(self.pending_windows) != 64:
                    raise ValueError("Optimizer updates do not match accumulation64 microbatches")
                torch.cuda.synchronize()
                now = time.monotonic()
                event = {"optimizer_update": int(trainer.global_step), "microbatches": 64,
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
            if (loader.num_workers, loader.batch_size, loader.drop_last) != (8, 8, False):
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
        if not formal and (trainer.global_step != 1 or trace.microbatches != 64):
            raise ValueError("Lifecycle check did not execute exactly one accumulation64 update")
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("lifecycle-check", "formal"))
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    spec_path = args.spec.resolve()
    spec = json.loads(spec_path.read_text())
    output = (args.output or Path(os.environ["GRID_JOB_DIR"]) / "output").resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError("Refuse to reuse a nonempty run output identity")
    formal = args.phase == "formal"
    start = time.monotonic()
    try:
        deployment, job_dir, request, status = deployment_and_job(spec_path, formal)
        if not spec.get("formal_protocol_user_approved") or not spec.get("formal_approved"):
            raise ValueError("User approval for this original Beauty MLP protocol is required")
        if formal and not spec.get("formal_training_allowed"):
            raise ValueError("This spec does not authorize formal training")
        if spec.get("checkpoint") or spec.get("resume_checkpoint") or spec.get("shuffle_n", 0):
            raise ValueError("Both phases require a fresh unshuffled original MLP run")
        gate = verify_gate(spec, deployment) if formal else None
        deadline = deadline_record(spec, request, status)
        write_json(output / "invocation.json", {"argv": sys.argv, "interpreter": sys.executable,
            "cwd_at_entry": os.getcwd(), "phase": args.phase, "spec": spec,
            "spec_sha256": sha256(spec_path), "plan_sha256": plan_sha256(spec),
            "deployment": deployment, "runner_job_dir": str(job_dir) if job_dir else None,
            "runner_request_sha256": sha256(job_dir / "request.json") if job_dir else None,
            "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"), "deadline": deadline,
            "nonauthor_gate": gate})
        assets = {key: verify_asset(spec[key]) for key in ("config", "sid", "evaluation_index", "test_index")}
        if spec.get("resource_check"):
            assets["resource_check"] = verify_asset(spec["resource_check"])
        write_json(output / "assets.json", assets)
        bind_source("MLP")
        from .mlp_execution import configure
        cfg = configure(spec, output)
        cfg.test = True
        result = execute(args.phase, spec, cfg, output, deadline)
        write_json(output / "environment.json", {
            "versions": {name: importlib.metadata.version(name) for name in (
                "torch", "transformers", "lightning", "hydra-core", "tensorflow-cpu", "numpy")},
            "imports": {key: str(value.__file__) for key, value in sys.modules.copy().items()
                        if key in ("src", "src.models.modules.base_module",
                            "src.data.loading.components.collate_functions",
                            "src.models.modules.semantic_id.tiger_generation_model_mlp")}})
        write_json(output / "RESULT.json", {"status": "PASS", "phase": args.phase,
            "formal": formal, "training_complete": result["training_complete"],
            "test_complete": result["test_complete"], "elapsed_seconds": time.monotonic() - start})
    except BaseException as error:
        write_json(output / "RESULT.json", {
            "status": "INCOMPLETE" if isinstance(error, SafetyInterruption) else "FAILED",
            "phase": args.phase, "formal": formal, "error": repr(error),
            "elapsed_seconds": time.monotonic() - start, "automatic_resume_allowed": False})
        traceback.print_exc()
        raise SystemExit(2 if isinstance(error, SafetyInterruption) else 1)


if __name__ == "__main__":
    main()
