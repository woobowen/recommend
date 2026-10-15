"""Bounded real entrypoints; this module has no formal execution command.

Reuse the accepted model/data/generation/evaluator paths without changing their
research computation. All limits and scratch checkpoints belong to this stage.
"""
import argparse
import collections
import dataclasses
import datetime as dt
import faulthandler
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from experiments.beauty_extensions.common import (
    REPO, bind_source, batch_keys, batch_to, instantiate, load_checkpoint,
    make_loader, sha256, verify_asset, write_json,
)


def require_preflight(spec):
    if spec.get("formal_approved") is not False:
        raise ValueError("Every stage spec must explicitly set formal_approved=false")
    if spec.get("nonauthor_gate") or spec.get("approval_token"):
        raise ValueError("This stage cannot carry a formal approval token")
    if spec.get("route") not in ("A", "B", "MLP"):
        raise ValueError("Unknown route")
    if not 1 <= int(spec.get("preflight_rows", 19)) <= 64:
        raise ValueError("Bounded row cap is 64")
    if spec["route"] == "MLP" and spec.get("domain") not in ("toys", "sports"):
        raise ValueError("This stage cannot train Beauty or another domain")
    if spec.get("optimizer_updates", 2) != 2 or spec.get("resume_updates", 1) != 1:
        raise ValueError("Only the fixed 2+1 engineering optimizer updates are supported")


def state_digest(model):
    import torch
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if isinstance(value, torch.Tensor):
            h.update(name.encode()); h.update(str((value.dtype, tuple(value.shape))).encode())
            h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def sampler_count(model):
    return getattr(model, "shuffle_call_count", getattr(model, "shuffle_summary", {}).get("calls", 0))


def tensor_tree_digest(value):
    import torch
    h = hashlib.sha256()
    def visit(x):
        if isinstance(x, torch.Tensor):
            h.update(str((x.dtype, tuple(x.shape))).encode())
            h.update(x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif dataclasses.is_dataclass(x):
            for field in dataclasses.fields(x): h.update(field.name.encode()); visit(getattr(x, field.name))
        elif isinstance(x, dict):
            for k in sorted(x, key=str): h.update(str(k).encode()); visit(x[k])
        elif isinstance(x, (list, tuple)):
            for v in x: visit(v)
        else: h.update(repr(x).encode())
    visit(value)
    return h.hexdigest()


def bounded_batches(cfg, split, rows):
    return list(make_loader(cfg, split, limit_rows=rows))


def check_batches(batches, split, output):
    import torch
    seen = []
    report = []
    for batch in batches:
        x, y = batch
        values = x.transformed_sequences["sequence_data"]
        mask = x.mask.reshape(values.shape[0], -1, 4)
        assert torch.all(mask == mask[..., :1]), "Partial valid item group"
        assert values.dtype in (torch.int32, torch.int64)
        keys = batch_keys(batch); seen.extend(keys)
        targets = next(iter(y.labels.values())).reshape(values.shape[0], 4)
        report.append({"shape": list(values.shape), "keys": keys,
            "target_sha256": tensor_tree_digest(targets),
            "valid_items": mask[..., -1].sum(1).tolist(),
            "has_padding": bool((mask == 0).any()), "input_sha256": tensor_tree_digest(batch)})
    assert len(seen) == len(set(seen)), "Repeated users in fixed engineering subset"
    write_json(output / (split + "_subset.json"), {"split": split, "rows": len(seen),
        "unique_users": len(set(seen)), "batches": report,
        "selection": "first rows in the actual single-worker assigned file order; before predictions",
        "source_modified": False, "drop_last": False, "engineering_only": True})
    return tensor_tree_digest(batches)


def ordinary_validation(cfg, model, batches, output):
    import hydra
    import torch
    from lightning import Callback
    from torch.utils.data import DataLoader
    from experiments.beauty_extensions.cli import clear_runtime, model_arguments
    original_evaluator = model.evaluator
    count_before = sampler_count(model)
    rows = []
    class ValidateState(Callback):
        def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
            assert not pl_module.is_testing
            args = model_arguments(batch)
            rows.append({"batch_index": batch_idx, "input_device": str(args["input_ids"].device),
                "parameter_devices": sorted({str(p.device) for p in pl_module.parameters()}),
                "is_testing": pl_module.is_testing, "training": pl_module.training})
    tc = cfg.trainer.copy(); tc.pop("profiler", None)
    from .resources import callback
    trainer = hydra.utils.instantiate(tc, logger=False, callbacks=[ValidateState(), callback()], enable_checkpointing=False)
    try:
        metrics = trainer.validate(model, dataloaders=DataLoader(batches, batch_size=None), ckpt_path=None)
    finally:
        clear_runtime(model)
        assert model.evaluator is original_evaluator
    write_json(output / "ordinary_validation.json", {"metrics": metrics, "batch_devices": rows,
        "parameter_devices_after": sorted({str(p.device) for p in model.parameters()}),
        "test_flag_after": model.is_testing, "engineering_only": True,
        "shuffle_call_delta": sampler_count(model) - count_before})
    assert sampler_count(model) == count_before


def fault_and_recovery(cfg, model, dm, test_batches, val_batches, output):
    import torch
    from lightning import Callback
    from experiments.beauty_extensions.cli import test_lifecycle, model_arguments, numeric_difference
    class InjectedPreflightError(RuntimeError):
        pass
    events = []
    class Fault(Callback):
        def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
            assert pl_module.is_testing
            args = model_arguments(batch)
            events.append({"hook": "on_test_batch_start", "batch": batch_idx,
                "input_device": str(args["input_ids"].device),
                "parameter_devices": sorted({str(p.device) for p in pl_module.parameters()}),
                "flag": pl_module.is_testing})
        def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
            raise InjectedPreflightError("expected bounded failure after one real test batch")
        def on_exception(self, trainer, pl_module, exception):
            events.append({"hook": "on_exception", "exception_type": type(exception).__name__,
                "flag_after_cleanup_callback": pl_module.is_testing})
    model.to("cuda:0").eval(); model.on_validation_start()
    args = model_arguments(batch_to(test_batches[0], "cuda:0"))
    with torch.inference_mode(): reference = model.generate(**args)[:2]
    original_evaluator = model.evaluator
    before = state_digest(model)
    try:
        test_lifecycle(cfg, model, dm, output / "injected-test-failure", test_batches[:1], [Fault()])
    except InjectedPreflightError:
        (output / "injected-test-failure" / "expected_traceback.txt").write_text(traceback.format_exc())
    else:
        raise AssertionError("Fault did not reach the real Trainer exception path")
    assert any(r["hook"] == "on_exception" for r in events)
    assert not model.is_testing and model.evaluator is original_evaluator
    for field in ("_encoder_mask", "_replay", "_replay_queue"):
        assert getattr(model, field, None) is None, field
    model.to("cuda:0").eval(); model.on_validation_start()
    args = model_arguments(batch_to(test_batches[0], "cuda:0"))
    with torch.inference_mode(): recovered = model.generate(**args)[:2]
    assert torch.equal(reference[0], recovered[0])
    diff = numeric_difference(reference[1], recovered[1]); assert diff["allclose"]
    assert before == state_digest(model), "Evaluation changed model parameters/buffers"
    ordinary_validation(cfg, model, val_batches, output)
    write_json(output / "exception_recovery.json", {"status": "PASS", "actual_trainer_exception": True,
        "events": events, "same_model_instance": True, "evaluator_restored": True,
        "state_digest_before_after": before, "off_sid_exact": True, "off_scores": diff,
        "validation_after_exception": "ordinary_validation.json", "exception_was_expected_and_caught_by_type_only": True})


def shuffle_check(spec, cfg, model, dm, output, test_batches, val_batches):
    import hydra
    import torch
    from experiments.beauty_extensions.cli import smoke
    from experiments.beauty_extensions.results import recompute
    initial = state_digest(model)
    source_inputs = tensor_tree_digest(test_batches)
    if hasattr(model, "permutation_trace_limit"):
        model.permutation_trace_limit = 2 * len(test_batches)
    if hasattr(model, "trace_limit"):
        model.trace_limit = 2 * len(test_batches)
    smoke(spec, cfg, model, dm, output, test_batches)
    assert initial == state_digest(model)
    assert source_inputs == tensor_tree_digest(test_batches), "Input or labels modified in place"
    for n in (0, 3, 4):
        folder = output / f"trainer-test-{n}"
        recompute(spec, cfg, folder / "predictions.jsonl", folder, trainer_metrics=folder / "metrics.json")
        trace = json.loads((folder / "intervention_trace.json").read_text())
        assert bool(trace["events"]) == (n != 0)
        delta = trace["role_count_delta"]["encoder_role_counts"]
        assert delta.get("loss", 0) == len(test_batches) and delta.get("generate", 0) == len(test_batches)
        if spec["route"] == "A":
            assert trace["shuffle_call_count_delta"] == (2 * len(test_batches) if n else 0)
        else:
            changed = trace["role_count_delta"]["shuffle_role_counts"]
            assert changed.get("loss", 0) == (len(test_batches) if n else 0)
            assert changed.get("generate", 0) == (len(test_batches) if n else 0)
    ordinary_validation(cfg, model, val_batches, output / "after-three-tests")
    fault_and_recovery(cfg, model, dm, test_batches, val_batches, output)
    # Cover the author's actual multi-worker data-module path under a hard
    # Trainer limit. It is separate from the fixed small per-condition subset.
    from omegaconf import OmegaConf
    source_cfg = OmegaConf.load(spec["config"]["path"])
    source_workers = int(source_cfg.data_loading.test_dataloader_config.dataloader.num_workers)
    if source_workers:
        from experiments.beauty_extensions.cli import test_lifecycle
        worker_cfg = cfg.copy()
        for split in ("train", "val", "test"):
            lc = worker_cfg.data_loading[f"{split}_dataloader_config"].dataloader
            lc.num_workers = int(source_cfg.data_loading[f"{split}_dataloader_config"].dataloader.num_workers)
            lc.timeout = int(source_cfg.data_loading[f"{split}_dataloader_config"].dataloader.timeout)
            lc.persistent_workers = False
        worker_cfg.trainer.limit_test_batches = 4
        worker_dm = hydra.utils.instantiate(worker_cfg.data_loading.datamodule)
        if spec["route"] == "A": model.shuffle_embedding_on_test = False
        else: model.shuffle_n = 0
        folder = output / "source-workers-bounded-test"
        test_lifecycle(worker_cfg, model, worker_dm, folder, bounded_datamodule=True)
        result = recompute(spec, worker_cfg, folder / "predictions.jsonl", folder, trainer_metrics=folder / "metrics.json")
        assert result["samples"] == 32 and result["unique_users"] == 32
        write_json(folder / "worker_contract.json", {"original_workers": source_workers,
            "trainer_limit_test_batches": 4, "metrics_are_engineering_only": True,
            "full_datamodule_entry_used": True, "source_data_changed": False,
            "prefetch_is_not_model_evaluation": True})
    model.to("cuda:0").train(); model.on_train_start()
    calls = sampler_count(model)
    with torch.no_grad():
        batch = batch_to(test_batches[0], "cuda:0")
        _, train_loss = model.model_step(batch[0], batch[1])
    assert bool(torch.isfinite(train_loss)) and not model.is_testing
    assert sampler_count(model) == calls
    write_json(output / "train_off_forward.json", {"training_hook_and_real_forward": True,
        "is_testing": model.is_testing, "loss": float(train_loss), "shuffle_call_delta": 0,
        "optimizer_updates": 0, "rng_note": "training Dropout may consume RNG; Shuffle sampler remains OFF"})
    model.eval(); model.on_validation_start()
    assert initial == state_digest(model) and source_inputs == tensor_tree_digest(test_batches)
    write_json(output / "shuffle_result.json", {"status": "PASS", "route": spec["route"],
        "engineering_only": True, "formal_approved": False, "real_test_split": True,
        "state_digest_unchanged": initial, "input_and_target_digest_unchanged": source_inputs,
        "three_real_trainer_tests": [0, 3, 4], "post_test_validation": True,
        "actual_exception_recovery": True, "original_failed_job_retained": "beauty-b-smoke-r3"})


def mlp_boundary_probes(cfg, model, first, output):
    """Report original/adapter behavior for invalid histories, never change masks."""
    import hydra
    import torch
    from experiments.beauty_extensions.cli import model_arguments
    cpu_rng = torch.get_rng_state(); gpu_rng = torch.cuda.get_rng_state()
    original_cfg = cfg.model.copy()
    original_cfg._target_ = "src.models.modules.semantic_id.tiger_generation_model_mlp.SemanticIDEncoderDecoder"
    original_cfg.pop("save_attentions_on_test", None)
    original = hydra.utils.instantiate(original_cfg)
    original.load_state_dict(model.state_dict(), strict=True); original.to("cuda:0").eval()
    report = []
    def tensors(value):
        if isinstance(value, torch.Tensor): return [value]
        if dataclasses.is_dataclass(value):
            return sum((tensors(getattr(value, f.name)) for f in dataclasses.fields(value)), [])
        if isinstance(value, dict): return sum((tensors(v) for v in value.values()), [])
        if isinstance(value, (tuple, list)): return sum((tensors(v) for v in value), [])
        return []
    try:
        args = model_arguments(first)
        for name in ("empty", "all_padding"):
            probe = {k: (v[:1].clone() if v is not None else None) for k,v in args.items()}
            if name == "empty":
                probe["input_ids"] = probe["input_ids"][:, :0]
                probe["attention_mask"] = probe["attention_mask"][:, :0]
            else:
                probe["input_ids"].fill_(-1); probe["attention_mask"].zero_()
            outcomes = []; raw = []
            for target in (original, model):
                try:
                    with torch.inference_mode(): value = target.encoder_forward_pass(**probe)
                    ts = tensors(value); raw.append(ts)
                    outcomes.append({"returned": True, "shapes": [list(v.shape) for v in ts],
                        "finite": all(bool(torch.isfinite(v).all()) for v in ts if v.is_floating_point())})
                except (RuntimeError, ValueError, IndexError) as exc:
                    raw.append(None); outcomes.append({"returned": False, "error_type": type(exc).__name__, "error": str(exc)})
            assert outcomes[0]["returned"] == outcomes[1]["returned"]
            if raw[0] is not None:
                assert len(raw[0]) == len(raw[1])
                assert all(torch.allclose(a, b, rtol=1e-5, atol=1e-6, equal_nan=True) for a,b in zip(*raw))
            else:
                assert outcomes[0]["error_type"] == outcomes[1]["error_type"]
            report.append({"input": name, "original": outcomes[0], "adapter": outcomes[1],
                "same_behavior": True, "invalid_history_probe_only": True})
    finally:
        del original
        torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state(gpu_rng)
    write_json(output / "empty_padding_probes.json", {"probes": report,
        "semantic_change": False, "scientific_input_support_claimed": False,
        "rng_restored_after_engineering_probe": True})


def mlp_check(spec, cfg, model, dm, output, test_batches, val_batches, resume=False):
    import hydra
    import torch
    from lightning import Callback
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from experiments.beauty_extensions.author.checks import describe_author_model, check_item_aggregation
    from experiments.beauty_extensions.author.models import validate_item_groups
    from experiments.beauty_extensions.cli import test_lifecycle
    from experiments.beauty_extensions.results import recompute
    initial_step, restore = 0, None
    resume_path = None
    if resume:
        verify_asset(spec["resume_checkpoint"])
        resume_path = spec["resume_checkpoint"]["path"]
        restore = load_checkpoint(resume_path)
        loaded = model.load_state_dict(restore["state_dict"], strict=True)
        assert not loaded.missing_keys and not loaded.unexpected_keys
        assert tensor_tree_digest(model.state_dict()) == tensor_tree_digest(restore["state_dict"])
        initial_step = int(restore["global_step"]); assert initial_step == 2
        assert len(restore["optimizer_states"]) == 1 and restore["callbacks"]
        write_json(output / "strict_restore.json", {"status": "PASS", "missing_keys": [], "unexpected_keys": [],
            "all_keys_shapes_values_equal": True, "initial_step": initial_step,
            "optimizer_state_sha256": tensor_tree_digest(restore["optimizer_states"][0]),
            "callback_keys": list(restore["callbacks"]), "independent_process": True,
            "protocol": "hash-verified own engineering checkpoint; local safe_globals/weights_only for inspection; Lightning restores this trusted own file"})
    model.to("cuda:0").eval()
    first = batch_to(test_batches[0], "cuda:0")
    write_json(output / "mlp_structure.json", describe_author_model(model))
    write_json(output / "mlp_aggregation.json", check_item_aggregation(model,
        first[0].transformed_sequences["sequence_data"], first[0].mask))
    mlp_boundary_probes(cfg, model, first, output)
    invalid = first[0].mask.clone(); invalid[0, :4] = torch.tensor([1, 1, 0, 0], device=invalid.device)
    try: validate_item_groups(invalid, 4)
    except ValueError as exc: partial_error = str(exc)
    else: raise AssertionError("Partial item group was accepted")
    assert all(bool(b[0].mask.any(1).all()) for b in test_batches + val_batches)
    write_json(output / "mask_contract.json", {"partial_group_rejected": partial_error,
        "real_history_has_visible_items": True,
        "mask_reduction": "last code, original implementation",
        "empty_history": "this fixed subset has visible history; min_sequence_length field is not an enabled filter and does not guarantee nonempty history after hiding a single-item target; empty_padding_probes.json records actual boundaries"})
    # Check strict restored predictions before permitting the extra update.
    if resume:
        test_lifecycle(cfg, model, dm, output / "before-resume-update-test", test_batches)
        expected = Path(spec["prior_predictions"]["path"]); verify_asset(spec["prior_predictions"])
        left = [json.loads(v) for v in expected.read_text().splitlines()]
        right = [json.loads(v) for v in (output / "before-resume-update-test/predictions.jsonl").read_text().splitlines()]
        assert len(left) == len(right)
        maximum = 0.0
        for a, b in zip(left, right):
            assert a["sample_key"] == b["sample_key"] and a["target"] == b["target"] and a["sids"] == b["sids"]
            maximum = max(maximum, max(abs(x-y) for x,y in zip(a["scores"], b["scores"])))
        assert maximum <= 1e-6
        write_json(output / "restored_prediction_parity.json", {"samples": len(left), "sid_exact": True,
            "target_and_order_exact": True, "maximum_score_absolute_error": maximum})
    watched = dict(model.item_aggregation_mlp.named_parameters())
    initial = {k: p.detach().cpu().clone() for k,p in watched.items()}
    updates = 1 if resume else 2
    accumulation = int(cfg.trainer.accumulate_grad_batches)
    assert 1 <= accumulation <= 64
    raw_batch = int(cfg.data_loading.train_dataloader_config.dataloader.batch_size_per_device)
    train_loader = make_loader(cfg, "train", limit_rows=updates * accumulation * raw_batch)
    tc = cfg.trainer.copy(); tc.pop("profiler", None)
    tc.max_steps = initial_step + updates; tc.max_epochs = -1; tc.min_steps = 0
    tc.val_check_interval = accumulation; tc.limit_val_batches = len(val_batches)
    tc.limit_test_batches = len(test_batches); tc.num_sanity_val_steps = 0
    traces, microbatches, validations, restored = [], [], [], []
    class TrainingTrace(Callback):
        def on_train_start(self, trainer, pl_module):
            if restore is not None:
                actual = trainer.optimizers[0].state_dict()
                assert tensor_tree_digest(actual) == tensor_tree_digest(restore["optimizer_states"][0])
                assert trainer.global_step == initial_step
                restored.append({"global_step": trainer.global_step, "optimizer_exact": True,
                    "rng_restoration": "new-process seed reset; no saved RNG restoration or uninterrupted stochastic trajectory claimed",
                    "data_cursor": "fresh bounded dataset; not exact replay of uninterrupted data stream"})
                callbacks = []
                for callback in trainer.callbacks:
                    if callback.state_key not in restore["callbacks"]: continue
                    prior = restore["callbacks"][callback.state_key]; current = callback.state_dict()
                    exact = tensor_tree_digest(prior) == tensor_tree_digest(current)
                    if callback.__class__.__name__ == "EarlyStopping": assert exact
                    if callback is checkpoint and not exact:
                        assert prior["dirpath"] != current["dirpath"]
                        expected = checkpoint_fresh_state.copy()
                        expected["best_model_path"] = prior["best_model_path"]
                        assert tensor_tree_digest(expected) == tensor_tree_digest(current), "Unexpected callback restore difference"
                    callbacks.append({"key": callback.state_key, "exact": exact,
                        "prior": prior, "current": current,
                        "difference_boundary": None if exact else "ModelCheckpoint output dir changed; Lightning restores best_model_path but resets directory-bound ranking state; not exact uninterrupted selection continuation"})
                def plain(value):
                    if isinstance(value, torch.Tensor): return value.detach().cpu().tolist()
                    if isinstance(value, dict): return {k: plain(v) for k,v in value.items()}
                    if isinstance(value, (list, tuple)): return [plain(v) for v in value]
                    if isinstance(value, float) and not __import__('math').isfinite(value): return str(value)
                    return value
                assert all(k in [v["key"] for v in callbacks] for k in restore["callbacks"])
                write_json(output / "callback_restore.json", plain({"callbacks": callbacks,
                    "all_callback_keys_accounted": True,
                    "exact_full_callback_restore_claimed": all(v["exact"] for v in callbacks)}))
        def on_before_optimizer_step(self, trainer, pl_module, optimizer):
            ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
            required = {id(p) for p in pl_module.parameters() if p.requires_grad}
            assert ids == required and all(p.requires_grad for p in pl_module.parameters())
            assert all(torch.isfinite(p.grad).all() for p in pl_module.parameters() if p.grad is not None)
            traces.append({"step_before": trainer.global_step,
                "optimizer_parameter_count": sum(p.numel() for g in optimizer.param_groups for p in g["params"]),
                "all_trainable_parameters_covered": True,
                "mlp_gradient_norms": {k: float(p.grad.norm()) if p.grad is not None else None for k,p in watched.items()}})
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            pieces = batch if isinstance(batch[0], (tuple, list)) else [batch]
            windows = [int(v[0].transformed_sequences["sequence_data"].shape[0]) for v in pieces]
            loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
            assert loss is not None and bool(torch.isfinite(loss).all())
            microbatches.append({"batch_index": batch_idx, "global_step": trainer.global_step,
                "raw_rows_per_config": raw_batch, "expanded_windows_by_piece": windows,
                "actual_expanded_windows": sum(windows), "loss": float(loss.detach())})
            write_json(output / "progress.json", microbatches[-1])
        def on_validation_end(self, trainer, pl_module):
            validations.append({"step": trainer.global_step,
                "metrics": {k: float(v) for k,v in trainer.callback_metrics.items() if k.startswith("val/")}})
    checkpoint = ModelCheckpoint(dirpath=str(output / "checkpoints"), save_last=True, save_top_k=1,
        monitor="val/recall@5", mode="max", save_on_train_epoch_end=False, filename="engineering-{step:06d}")
    checkpoint_fresh_state = checkpoint.state_dict().copy()
    early = hydra.utils.instantiate(cfg.callbacks.early_stopping) if cfg.callbacks.get("early_stopping") else None
    from .resources import callback as resource_callback
    callbacks = [checkpoint, TrainingTrace(), resource_callback()] + ([early] if early else [])
    trainer = hydra.utils.instantiate(tc, callbacks=callbacks, logger=CSVLogger(str(output), name="training_csv"))
    OmegaConf.save(tc, output / "actual_smoke_trainer.yaml", resolve=True)
    started = time.monotonic()
    # The original SequenceDataModule returns (loader,); the original
    # TransformerBaseModule.training_step unwraps batch[0]. Preserve that input
    # protocol while bounding the dataset, rather than rewriting training_step.
    trainer.fit(model, train_dataloaders=(train_loader,),
        val_dataloaders=DataLoader(val_batches, batch_size=None), ckpt_path=resume_path)
    assert trainer.global_step == initial_step + updates and len(traces) == updates
    assert len(microbatches) == updates * accumulation and len(validations) == updates
    if resume: assert restored
    saved = output / "checkpoints/engineering-final.ckpt"; trainer.save_checkpoint(saved)
    state = load_checkpoint(saved)
    assert int(state["global_step"]) == trainer.global_step
    assert all(torch.isfinite(v).all() for v in state["state_dict"].values() if isinstance(v, torch.Tensor))
    changes = {k: float((p.detach().cpu()-initial[k]).abs().max()) for k,p in watched.items()}
    assert any(v > 0 for v in changes.values())
    test_lifecycle(cfg, model, dm, output / "after-training-test", test_batches)
    metric = recompute(spec, cfg, output / "after-training-test/predictions.jsonl", output / "after-training-test",
        trainer_metrics=output / "after-training-test/metrics.json")
    write_json(output / "short_training.json", {"status": "PASS", "engineering_only": True,
        "domain": spec["domain"], "initialization": "strict own engineering checkpoint" if resume else "scratch all parameters",
        "new_optimizer_updates": updates, "initial_step": initial_step, "final_step": trainer.global_step,
        "accumulate_grad_batches": accumulation, "actual_training_microbatches": len(microbatches),
        "raw_batch_size": raw_batch, "actual_total_collate_windows": sum(r["actual_expanded_windows"] for r in microbatches),
        "microbatch_trace": microbatches, "optimizer_steps": traces, "validations": validations,
        "restore_events": restored, "aggregation_parameter_changes": changes,
        "checkpoint": {"path": str(saved), "bytes": saved.stat().st_size, "sha256": sha256(saved)},
        "optimizer_state_sha256": tensor_tree_digest(state["optimizer_states"][0]),
        "checkpoint_callback_keys": list(state["callbacks"]), "elapsed_seconds": time.monotonic()-started,
        "saved_prediction_verification": metric, "formal_state_reuse_allowed": False})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("prepare", "shuffle", "mlp-short", "mlp-resume"))
    p.add_argument("--spec", required=True, type=Path)
    args = p.parse_args(); spec = json.loads(args.spec.read_text()); require_preflight(spec)
    if args.phase.startswith("mlp") and spec["route"] != "MLP": raise ValueError("MLP route required")
    if args.phase == "shuffle" and spec["route"] not in ("A", "B"): raise ValueError("Shuffle route required")
    output = Path(os.environ["GRID_JOB_DIR"]) / "output"
    if (output / "invocation.json").exists(): raise RuntimeError("Run identity already used")
    output.mkdir(exist_ok=True)  # the accepted runner creates this directory
    start = time.monotonic()
    faulthandler.enable()
    faulthandler.dump_traceback_later(120, repeat=True)
    write_json(output / "invocation.json", {"argv": sys.argv, "interpreter": sys.executable,
        "cwd_at_entry": os.getcwd(), "spec": spec, "spec_sha256": sha256(args.spec),
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(), "formal_approved": False,
        "code_identity": json.loads((REPO / "DEPLOYMENT.json").read_text()),
        "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"), "phase": args.phase})
    bind_source(spec["route"])
    cfg, model, dm = instantiate(spec, output, checkpoint=spec["route"] in ("A", "B"))
    from experiments.beauty_extensions.cli import check_data
    import torch
    check_data(spec, cfg, output)  # bounded metadata/8 raw rows per split; no full coverage scan
    rows = int(spec.get("preflight_rows", 19))
    test_batches = bounded_batches(cfg, "test", rows)
    val_batches = bounded_batches(cfg, "val", 11)
    assert 2 <= len(test_batches) <= 4 and len(val_batches) == 2
    check_batches(test_batches, "test", output); check_batches(val_batches, "val", output)
    if "expected_long_history_case" in spec:
        case = spec["expected_long_history_case"]
        found = [(batch, i) for batch in test_batches for i, key in enumerate(batch_keys(batch)) if key == case["sample_key"]]
        assert len(found) == 1
        batch, i = found[0]
        assert int(batch[0].mask[i].sum()) == 4 * case["visible_items"]
        write_json(output / "long_history_case.json", {**case, "verified_in_actual_subset": True})
    import inspect
    from lightning.pytorch.strategies.strategy import Strategy
    framework_path = Path(inspect.getfile(Strategy))
    write_json(output / "environment.json", {"versions": {v: importlib.metadata.version(v) for v in
        ("torch", "transformers", "lightning", "hydra-core", "tensorflow-cpu", "numpy")},
        "imports": {k: str(v.__file__) for k,v in sys.modules.copy().items() if k in
            ("src", "src.models.modules.base_module", "src.models.modules.semantic_id.tiger_generation_model",
             "src.models.modules.semantic_id.tiger_generation_model_shuffle", "src.models.modules.semantic_id.tiger_generation_model_mlp",
             "src.data.loading.components.collate_functions")}, "model_size": sum(p.numel() for p in model.parameters()),
        "gpu": [{"logical": i, "name": torch.cuda.get_device_name(i)} for i in range(torch.cuda.device_count())],
        "framework_strategy": {"file": str(framework_path), "resolved": str(framework_path.resolve()), "sha256": sha256(framework_path)}})
    if args.phase != "prepare":
        physical = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        assert physical.startswith("GPU-") and "," not in physical
        assert torch.cuda.device_count() == 1, "Exactly one isolated visible GPU required"
        from .resources import ResourceMonitor
        with ResourceMonitor(output, physical, os.environ.get("GRID_SHARED_PREFLIGHT") == "1"):
            torch.cuda.reset_peak_memory_stats()
            if args.phase == "shuffle": shuffle_check(spec, cfg, model, dm, output, test_batches, val_batches)
            else: mlp_check(spec, cfg, model, dm, output, test_batches, val_batches, resume=args.phase == "mlp-resume")
    write_json(output / "RESULT.json", {"status": "PASS", "phase": args.phase,
        "engineering_only": True, "formal_approved": False, "elapsed_seconds": time.monotonic()-start,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if args.phase != "prepare" else 0,
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved() if args.phase != "prepare" else 0})
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__": main()
