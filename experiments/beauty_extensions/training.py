"""Bounded real MLP Trainer lifecycle checks, isolated from formal new runs."""
from pathlib import Path
import time

from .common import write_json, sha256


def train_check(spec, cfg, model, dm, output, resume=False):
    import hydra
    import torch
    from lightning import Callback
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from omegaconf import OmegaConf
    from .author.checks import describe_author_model, check_item_aggregation
    from .common import make_loader, batch_to, verify_asset, load_checkpoint
    if spec["route"] != "MLP":
        raise ValueError("Short training is authorized for the original item MLP only")
    # These are explicitly engineering-only limits, never formal training values.
    updates = 1 if resume else 2
    resume_path = None
    initial_step = 0
    if resume:
        verify_asset(spec["resume_checkpoint"])
        resume_path = spec["resume_checkpoint"]["path"]
        state = load_checkpoint(resume_path)
        initial_step = int(state["global_step"])
        assert len(state["optimizer_states"]) == 1
        model.load_state_dict(state["state_dict"], strict=True)
    model.to("cuda:0")
    first = batch_to(next(iter(make_loader(cfg, "train"))), "cuda:0")
    structure = describe_author_model(model)
    aggregation = check_item_aggregation(model, first[0].transformed_sequences["sequence_data"], first[0].mask)
    write_json(output / "mlp_structure.json", structure)
    write_json(output / "mlp_aggregation.json", aggregation)
    watched = dict(model.item_aggregation_mlp.named_parameters())
    initial = {k: v.detach().cpu().clone() for k,v in watched.items()}
    trace = []
    class TrainingTrace(Callback):
        def on_before_optimizer_step(self, trainer, pl_module, optimizer):
            optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
            assert all(id(p) in optimizer_ids for p in watched.values() if p.requires_grad)
            assert all(torch.isfinite(p.grad).all() for p in watched.values() if p.grad is not None)
            trace.append({"optimizer_step_before": trainer.global_step,
                "mlp_grad_norms": {k: float(p.grad.detach().norm()) if p.grad is not None else None for k,p in watched.items()},
                "optimizer_parameter_count": sum(p.numel() for group in optimizer.param_groups for p in group["params"])})
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if batch_idx < 4 or trainer.global_step > initial_step:
                data_batch = batch[0] if isinstance(batch[0], (tuple, list)) else batch
                x = data_batch[0].transformed_sequences["sequence_data"]
                loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
                write_json(output / "progress.json", {"step": trainer.global_step, "batch_index": batch_idx,
                    "expanded_training_batch": list(x.shape), "loss": float(loss.detach()) if loss is not None else None})
    trainer_cfg = cfg.trainer.copy()
    trainer_cfg.pop("profiler", None)
    accumulation = int(cfg.trainer.accumulate_grad_batches)
    trainer_cfg.max_steps = initial_step + updates
    trainer_cfg.max_epochs = -1
    trainer_cfg.min_steps = 0
    trainer_cfg.val_check_interval = accumulation
    trainer_cfg.limit_val_batches = 2
    trainer_cfg.limit_test_batches = 2
    trainer_cfg.num_sanity_val_steps = 0
    callback = ModelCheckpoint(dirpath=str(output / "checkpoints"), save_last=True, save_top_k=1,
        monitor="val/recall@5", mode="max", save_on_train_epoch_end=False,
        filename="engineering-{step:06d}")
    logger = CSVLogger(str(output), name="training_csv")
    OmegaConf.save(cfg, output / "base_effective_config.yaml", resolve=False)
    OmegaConf.save(trainer_cfg, output / "actual_smoke_trainer.yaml", resolve=True)
    trainer = hydra.utils.instantiate(trainer_cfg, callbacks=[callback, TrainingTrace()], logger=logger)
    start = time.monotonic()
    trainer.fit(model, datamodule=dm, ckpt_path=resume_path)
    assert trainer.global_step == initial_step + updates
    checkpoint = output / "checkpoints" / "engineering-final.ckpt"
    trainer.save_checkpoint(checkpoint)
    state = load_checkpoint(checkpoint)
    assert int(state["global_step"]) == trainer.global_step and len(state["optimizer_states"]) == 1
    differences = {k: float((v.detach().cpu() - initial[k]).abs().max()) for k,v in watched.items()}
    assert any(v > 0 for v in differences.values())
    assert any(value is not None and value > 0 for step in trace for value in step["mlp_grad_norms"].values())
    assert all(torch.isfinite(v).all() for v in state["state_dict"].values() if isinstance(v,torch.Tensor))
    # This run's checkpoint is engineering evidence, never a formal best model.
    metrics = trainer.test(model, datamodule=dm, ckpt_path=None)
    write_json(output / "short_training.json", {"engineering_only": True, "new_optimizer_updates": updates,
        "initial_step": initial_step, "final_step": trainer.global_step, "accumulate_grad_batches": accumulation,
        "expected_training_microbatches": updates * accumulation, "elapsed_training_seconds": time.monotonic() - start,
        "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint), "bytes": checkpoint.stat().st_size},
        "best_smoke_checkpoint": callback.best_model_path, "mlp_updates": differences, "optimizer_steps": trace,
        "test_metrics_engineering_only": metrics, "resume_is_independent_process": resume,
        "formal_state_reuse_allowed": False})
