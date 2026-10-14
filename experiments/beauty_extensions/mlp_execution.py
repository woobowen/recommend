"""Approved single-GPU MLP resource checks; separate from the blocked B harness.

Metadata runs in its own CPU process. The resource check starts from scratch,
performs two actual accumulation64 updates, then validates every evaluation
record using the preserved author model, datamodule and evaluator. Its state
must never initialize a formal run.
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
import sys
import time

from .common import REPO, bind_source, load_config, sha256, verify_asset, write_json


def configure(spec, output):
    """Apply the approved conditions before constructing any model/dataloader."""
    if spec["route"] != "MLP" or not spec.get("formal_protocol_user_approved"):
        raise ValueError("This entry requires the approved original Beauty MLP protocol")
    if spec.get("checkpoint") or spec.get("resume_checkpoint"):
        raise ValueError("Resource checks and formal fresh starts cannot reuse engineering checkpoints")
    cfg = load_config(spec, output)
    cfg.train, cfg.test, cfg.ckpt_path = True, False, None
    cfg.trainer.accumulate_grad_batches = 64
    cfg.trainer.val_check_interval = 6400
    assert cfg.trainer.max_steps == 320000 and cfg.trainer.max_epochs == 10
    assert cfg.trainer.precision == "32-true" and cfg.trainer.num_sanity_val_steps == 0
    assert cfg.trainer.devices == 1 and cfg.trainer.strategy == "auto"
    for split in ("train", "val", "test"):
        lc = cfg.data_loading[f"{split}_dataloader_config"].dataloader
        assert lc.num_workers == 8
    train = cfg.data_loading.train_dataloader_config.dataloader
    assert train.batch_size_per_device == 32 and train.collate_fn.max_batch_size == 512
    assert train.assign_all_files_per_worker and train.drop_last
    for split in ("val", "test"):
        lc = cfg.data_loading[f"{split}_dataloader_config"].dataloader
        assert not lc.get("assign_all_files_per_worker", False) and not lc.drop_last
    assert cfg.seed == 42 and cfg.model.top_k_for_generation == 10
    assert cfg.optim.optimizer._target_ == "torch.optim.Adam"
    assert cfg.optim.optimizer.lr == 0.001 and cfg.optim.optimizer.weight_decay == 0.0001
    assert cfg.optim.scheduler is None
    return cfg


def evaluation_index(spec, cfg, output):
    from .data_checks import (_iter_metadata, _key, _loader_and_sid,
                              _source_targets, _sources_unchanged, _validate_items)
    lc, sid, metadata = _loader_and_sid(cfg, spec, "val")
    counts = Counter()
    pending = []
    path = output / "evaluation_target_index.jsonl"
    with path.open("x") as stream:
        def emit(rows):
            targets, _ = _source_targets(rows, lc, sid)
            for row, target in zip(rows, targets.tolist()):
                key = _key(row.user)
                counts[(key, tuple(target))] += 1
                stream.write(json.dumps({"sample_key": key, "target": target,
                    "source_file_index": row.file_index, "row_index": row.row_index},
                    separators=(",", ":")) + "\n")
        for row in _iter_metadata(metadata["file_order"]):
            _validate_items(row, sid.shape[1])
            pending.append(row)
            if len(pending) == 512:
                emit(pending); pending = []
        if pending:
            emit(pending)
    _sources_unchanged(metadata)
    assert counts
    result = {"split": "evaluation", "model_execution": False,
        "rows": sum(counts.values()), "unique_users": len({key[0] for key in counts}),
        "sid": spec["sid"], "loader": metadata,
        "target_index": {"path": str(path), "sha256": sha256(path)},
        "comparison_rule": "Exact (opaque user key, four-code target) multiset; worker arrival order is irrelevant",
        "original_NextKTokenMasking_and_hidden_target_verified": True,
        "raw_user_ids_exported": False}
    write_json(output / "evaluation_coverage.json", result)
    return result


def resource_check(spec, cfg, output):
    import hydra
    import torch
    from lightning import Callback, seed_everything
    from lightning.pytorch.loggers import CSVLogger
    from omegaconf import OmegaConf
    from .author.checks import describe_author_model
    from .common import load_checkpoint

    index = verify_asset(spec["evaluation_index"])
    expected = Counter()
    for line in Path(index["path"]).read_text().splitlines():
        row = json.loads(line)
        expected[(row["sample_key"], tuple(row["target"]))] += 1
    assert expected
    seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision("medium")  # preserved src/train.py:34
    model = hydra.utils.instantiate(cfg.model)
    dm = hydra.utils.instantiate(cfg.data_loading.datamodule)
    assert all(p.requires_grad for p in model.parameters())
    watch = dict(model.item_aggregation_mlp.named_parameters())
    initial = {name: p.detach().cpu().clone() for name, p in watch.items()}
    write_json(output / "structure.json", describe_author_model(model))
    trainer_cfg = cfg.trainer.copy()
    trainer_cfg.max_steps = 2
    # Keep formal cadence; explicit Trainer.validate below executes exactly one
    # complete validation after this isolated two-update engineering fit.
    trainer_cfg.enable_checkpointing = False
    trainer_cfg.enable_model_summary = False
    OmegaConf.save(cfg, output / "approved_formal_effective_config.yaml", resolve=False)
    OmegaConf.save(trainer_cfg, output / "actual_resource_trainer.yaml", resolve=True)
    write_json(output / "preflight_formal_diff.json", {
        "trainer.max_steps": {"preflight": 2, "formal": 320000},
        "trainer.enable_checkpointing": {"preflight": False, "formal": "original callback"},
        "trainer.enable_model_summary": {"preflight": False, "formal": "original callback"},
        "validation_invocation": {"preflight": "one complete Trainer.validate after two updates", "formal": "each 6400 train microbatches"},
        "callbacks": {"preflight": "resource observer only", "formal": "original best val/recall@5 max and patience10"},
        "formal_state_reuse_allowed": False})
    used = {key: OmegaConf.to_container(cfg[key], resolve=True) for key in ("model", "optim", "eval")}
    used["datamodule"] = OmegaConf.to_container(cfg.data_loading.datamodule, resolve=True)
    write_json(output / "used_config.json", used)

    class Trace(Callback):
        def __init__(self):
            self.microbatches = 0
            self.windows = []
            self.losses_finite = True
            self.optimizer_events = []
            self.update_timings = []
            self.seen = Counter()
            self.val_batches = 0
            self.val_shapes = Counter()
            self.validation_seconds = None

        def on_train_start(self, trainer, pl_module):
            torch.cuda.synchronize()
            self.start = self.last_update = time.monotonic()
            self.last_step = trainer.global_step
            loader = trainer.train_dataloader[0]
            assert loader.num_workers == 8 and loader.batch_size == 32
            assert loader.drop_last and loader.dataset.assign_all_files_per_worker
            write_json(output / "actual_train_loader.json", {
                "workers": loader.num_workers, "raw_microbatch_size": loader.batch_size,
                "drop_last": loader.drop_last, "persistent_workers": loader.persistent_workers,
                "assigned_files": len(loader.dataset.list_of_file_paths),
                "all_files_per_worker": loader.dataset.assign_all_files_per_worker,
                "collate_window_cap": 512})

        def on_before_optimizer_step(self, trainer, pl_module, optimizer):
            actual = {id(p) for group in optimizer.param_groups for p in group["params"]}
            trainable = {id(p) for p in pl_module.parameters() if p.requires_grad}
            assert actual == trainable
            grads = [p.grad for p in pl_module.parameters() if p.grad is not None]
            assert grads and bool(torch.stack([torch.isfinite(g).all() for g in grads]).all())
            self.optimizer_events.append({"global_step_before": trainer.global_step,
                "microbatches_completed_before_current": self.microbatches,
                "all_available_gradients_finite": True,
                "optimizer_covers_every_trainable_parameter": True,
                "optimizer_parameter_count": sum(p.numel() for group in optimizer.param_groups for p in group["params"]),
                "mlp_gradient_norms": {name: float(p.grad.detach().norm()) if p.grad is not None else None
                    for name, p in watch.items()}})

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            data = batch[0]
            x = data[0].transformed_sequences["sequence_data"]
            assert 0 < x.shape[0] <= 512 and x.shape[1] == 120
            self.windows.append(int(x.shape[0]))
            self.microbatches += 1
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
            if not bool(torch.isfinite(loss).all()):
                raise FloatingPointError("Nonfinite training loss")
            if trainer.global_step != self.last_step:
                torch.cuda.synchronize()
                now = time.monotonic()
                assert self.microbatches == trainer.global_step * 64
                self.update_timings.append({"completed_optimizer_update": trainer.global_step,
                    "microbatches": 64, "seconds": now - self.last_update,
                    "expanded_windows": sum(self.windows[-64:])})
                self.last_update, self.last_step = now, trainer.global_step
                write_json(output / "progress.json", {"stage": "train", "global_step": trainer.global_step,
                    "microbatches": self.microbatches, "last_update": self.update_timings[-1]})

        def on_validation_start(self, trainer, pl_module):
            torch.cuda.synchronize()
            self.val_start = time.monotonic()
            loaders = trainer.val_dataloaders
            loader = loaders[0] if isinstance(loaders, (tuple, list)) else loaders
            assert loader.num_workers == 8 and loader.batch_size == 8 and not loader.drop_last
            assert not loader.dataset.assign_all_files_per_worker
            write_json(output / "actual_validation_loader.json", {
                "workers": loader.num_workers, "raw_microbatch_size": loader.batch_size,
                "drop_last": loader.drop_last, "all_files_per_worker": loader.dataset.assign_all_files_per_worker,
                "assigned_files": len(loader.dataset.list_of_file_paths),
                "limit_val_batches": trainer.limit_val_batches})

        def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
            values = batch[0].transformed_sequences["user_id"][:, 0].detach().cpu().tolist()
            keys = [hashlib.sha256(("beauty-extensions-v1:" + str(int(v))).encode()).hexdigest()[:24]
                    for v in values]
            targets = batch[1].labels["sequence_data"].detach().cpu().reshape(len(keys), 4).tolist()
            for key, target in zip(keys, targets):
                self.seen[(key, tuple(target))] += 1
            self.val_batches += 1
            self.val_shapes[len(keys)] += 1
            if batch_idx % 256 == 0:
                write_json(output / "progress.json", {"stage": "validation", "batch_index": batch_idx,
                    "targets_observed": sum(self.seen.values())})

        def on_validation_end(self, trainer, pl_module):
            torch.cuda.synchronize()
            self.validation_seconds = time.monotonic() - self.val_start
            assert self.seen == expected, "Full validation worker coverage differs from evaluation metadata"
            write_json(output / "validation_coverage_check.json", {
                "status": "PASS", "split": "evaluation", "exact_key_target_multiset": True,
                "rows": sum(self.seen.values()), "unique_users": len({k[0] for k in self.seen}),
                "batch_count": self.val_batches, "batch_size_histogram": dict(self.val_shapes),
                "seconds": self.validation_seconds, "expected_index": index})

    trace = Trace()
    logger = CSVLogger(str(output), name="resource_csv")
    trainer = hydra.utils.instantiate(trainer_cfg, callbacks=[trace], logger=logger)
    torch.cuda.reset_peak_memory_stats()
    fit_start = time.monotonic()
    trainer.fit(model, datamodule=dm, ckpt_path=None)
    fit_seconds = time.monotonic() - fit_start
    assert trainer.global_step == 2 and trace.microbatches == 128 and len(trace.optimizer_events) == 2
    differences = {name: float((p.detach().cpu() - initial[name]).abs().max()) for name, p in watch.items()}
    assert all(value > 0 for value in differences.values())
    checkpoint = output / "checkpoints" / "resource-only-step2.ckpt"
    checkpoint.parent.mkdir()
    save_start = time.monotonic()
    trainer.save_checkpoint(checkpoint)
    save_seconds = time.monotonic() - save_start
    state = load_checkpoint(checkpoint)
    assert state["global_step"] == 2 and len(state["optimizer_states"]) == 1
    assert all(torch.isfinite(value).all() for value in state["state_dict"].values() if isinstance(value, torch.Tensor))
    validation_wall_start = time.monotonic()
    metrics = trainer.validate(model, datamodule=dm, ckpt_path=None)
    validation_wall_seconds = time.monotonic() - validation_wall_start
    assert trace.validation_seconds is not None and trace.seen == expected
    assert all(math.isfinite(float(value)) for entry in metrics for value in entry.values())
    warm_update = trace.update_timings[-1]["seconds"]
    val_seconds = validation_wall_seconds
    historical_hours = (3500 * warm_update + 35 * val_seconds + save_seconds) / 3600
    minimum_no_improvement_hours = (1100 * warm_update + 11 * val_seconds + save_seconds) / 3600
    maximum_hours = (320000 * warm_update + 3200 * val_seconds + save_seconds) / 3600
    # These are extrapolation scenarios, not substitute training stop criteria.
    result = {"status": "PASS", "engineering_only": True,
        "formal_training_started": False, "fresh_initialization": True, "full_parameter_training": True,
        "optimizer_updates": 2, "accumulation": 64, "actual_microbatches": trace.microbatches,
        "raw_training_rows_nominal": 32 * trace.microbatches,
        "expanded_training_windows": {"total": sum(trace.windows), "min": min(trace.windows),
            "max": max(trace.windows), "mean": sum(trace.windows) / len(trace.windows), "per_microbatch": trace.windows},
        "optimizer_events": trace.optimizer_events, "per_update_timings": trace.update_timings,
        "saved_optimizer_step_values": sorted({int(value["step"]) for value in state["optimizer_states"][0]["state"].values()}),
        "fit_wall_seconds_including_startup": fit_seconds, "save_wall_seconds": save_seconds,
        "full_validation_wall_seconds": val_seconds,
        "full_validation_hook_seconds_excluding_prior_setup": trace.validation_seconds,
        "full_validation_metrics_engineering_only": metrics,
        "full_validation_target_count": sum(trace.seen.values()), "mlp_parameter_max_abs_updates": differences,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint), "bytes": checkpoint.stat().st_size},
        "formal_state_reuse_allowed": False,
        "budget_scenarios": {"warm_update_seconds": warm_update,
            "historical_3500_updates_35_validations_hours": historical_hours,
            "earliest_patience10_no_improvement_1100_updates_11_validations_hours": minimum_no_improvement_hours,
            "configured_320000_updates_hours": maximum_hours, "hard_cap_hours": 12,
            "excludes_full_test_import_and_extra_checkpoint_cost": True,
            "historical_termination_is_not_a_new_run_guarantee": True,
            "decision": "PAUSE_FORMAL_BUDGET_REVIEW" if historical_hours > 12 else "AWAIT_NONAUTHOR_ADMISSION"}}
    write_json(output / "resource_check.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("evaluation-index", "resource-check"))
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    output = (args.output or Path(os.environ["GRID_JOB_DIR"]) / "output").resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "invocation.json").exists():
        raise RuntimeError("Refuse to reuse a job output identity")
    start = time.monotonic()
    write_json(output / "invocation.json", {"argv": sys.argv, "interpreter": sys.executable,
        "cwd_at_entry": os.getcwd(), "spec": spec, "spec_sha256": sha256(args.spec),
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "code_identity": json.loads((REPO / "DEPLOYMENT.json").read_text()),
        "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES")})
    bind_source("MLP")
    cfg = configure(spec, output)
    verified = {key: verify_asset(spec[key]) for key in ("config", "sid")}
    write_json(output / "assets.json", verified)
    if args.phase == "evaluation-index":
        evaluation_index(spec, cfg, output)
    else:
        resource_check(spec, cfg, output)
    write_json(output / "environment.json", {
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "lightning", "hydra-core", "tensorflow-cpu", "numpy")},
        "imports": {k: str(v.__file__) for k, v in sys.modules.copy().items() if k in (
            "src", "src.models.modules.base_module", "src.data.loading.components.collate_functions",
            "src.models.modules.semantic_id.tiger_generation_model_mlp")}})
    write_json(output / "RESULT.json", {"status": "PASS", "phase": args.phase,
        "formal": False, "elapsed_seconds": time.monotonic() - start})


if __name__ == "__main__":
    main()
