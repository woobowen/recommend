"""Observe the frozen GRID entrypoint without changing its computation."""
import importlib.util
import json
import os
from contextlib import contextmanager
from pathlib import Path
import time

import hydra
import torch
from lightning.pytorch.callbacks import Callback
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent
SOURCES = ROOT / "native"
N_ITEMS = 9922


class FitObservation(Callback):
    def __init__(self, output):
        self.output = output
        self.events = output / "fit_batches.jsonl"
        self.start = time.monotonic()
        self.previous = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        active = pl_module.current_layer
        layers = pl_module.quantization_layer_list
        self.previous = (
            time.monotonic(),
            active,
            trainer.global_step,
            layers[active].is_initialized,
            int(layers[active].init_buffer.shape[0]),
            [layer.get_centroids().detach().clone() for layer in layers],
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        started, active, before_step, before_initialized, buffered, previous = self.previous
        layers = pl_module.quantization_layer_list
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        if not isinstance(loss, torch.Tensor):
            raise ValueError("The native quantizer did not return a loss")
        event = {
            "batch_idx": batch_idx,
            "batch_size": len(batch[0].item_ids),
            "item_ids": [int(item) for item in batch[0].item_ids],
            "layer": active,
            "step_before": before_step,
            "step_after": trainer.global_step,
            "initialization_buffer_before": buffered,
            "initialization_buffer_after": int(layers[active].init_buffer.shape[0]),
            "initializing": bool(layers[active].is_initial_step),
            "initialized_before": bool(before_initialized),
            "initialized_after": bool(layers[active].is_initialized),
            "centroids_changed": [
                not torch.equal(layer.get_centroids().detach(), old)
                for layer, old in zip(layers, previous)
            ],
            "centroids_finite": [
                bool(torch.isfinite(layer.get_centroids()).all()) for layer in layers
            ],
            "loss": float(loss.detach().cpu()),
            "batch_elapsed_s": time.monotonic() - started,
            "fit_elapsed_s": time.monotonic() - self.start,
        }
        with self.events.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")
        if not torch.isfinite(loss).all() or not all(event["centroids_finite"]):
            raise ValueError(f"Non-finite loss or centroid in native batch {batch_idx}")


class PredictionObservation(Callback):
    def __init__(self, output):
        self.output = output
        self.seen = torch.zeros(N_ITEMS, dtype=torch.bool)
        self.codes = torch.empty((N_ITEMS, 3), dtype=torch.long)
        self.batch_sizes = []
        self.started = time.monotonic()

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if outputs is None:
            raise ValueError("Unexpected empty native quantizer prediction")
        ids = torch.as_tensor(outputs.keys, dtype=torch.long)
        values = outputs.predictions.detach().cpu()
        if ids.ndim != 1 or not len(ids) or values.shape != (len(ids), 3) or values.dtype != torch.long:
            raise ValueError(f"Invalid prediction shape/type at batch {batch_idx}")
        if ids.min() < 0 or ids.max() >= N_ITEMS or ids.unique().numel() != len(ids):
            raise ValueError(f"Invalid or duplicate item axis within batch {batch_idx}")
        if self.seen[ids].any():
            raise ValueError(f"Repeated native item IDs across prediction batches at batch {batch_idx}")
        if [int(item) for item in batch.item_ids] != ids.tolist():
            raise ValueError(f"Native prediction changed item keys at batch {batch_idx}")
        self.seen[ids] = True
        self.codes[ids] = values
        self.batch_sizes.append(len(ids))

    def on_predict_end(self, trainer, pl_module):
        if not self.seen.all():
            raise ValueError(f"Quantizer did not predict all {N_ITEMS} items, saw {int(self.seen.sum())}")
        torch.save(self.codes, self.output / "predicted_codes_by_item.pt")
        value = {
            "native_batch_sizes": self.batch_sizes,
            "items_observed": int(self.seen.sum()),
            "elapsed_s": time.monotonic() - self.started,
        }
        (self.output / "prediction_observation.json").write_text(json.dumps(value, indent=2) + "\n")


def load_entry():
    entry = SOURCES / "_运行" / "grid_stage.py"
    spec = importlib.util.spec_from_file_location("frozen_grid_stage", entry)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {entry}")
    stage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage)
    return stage


@hydra.main(version_base="1.3", config_path="native/configs", config_name="train")
def main(cfg):
    output = Path(cfg.paths.output_dir)
    (output / "resolved.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    if os.environ.get("GRID_UPSTREAM_CONFIG_ONLY") == "1":
        return
    native_launcher = NATIVE.pipeline_launcher

    @contextmanager
    def observe(cfg):
        with native_launcher(cfg) as pipeline:
            if cfg.task_name == "train":
                pipeline.trainer.callbacks.append(FitObservation(output))
            else:
                pipeline.trainer.callbacks.append(PredictionObservation(output))
            yield pipeline

    NATIVE.pipeline_launcher = observe
    NATIVE.main.__wrapped__(cfg)
    if cfg.task_name == "train":
        inspect_fit_checkpoint(output)


def inspect_fit_checkpoint(output):
    # Lightning saves this run's native ItemDataModule config along with model state.
    # Load only the checkpoint emitted in the current process and retain the numeric evidence.
    result = json.loads((output / "stage_result.json").read_text())
    checkpoint = Path(result["checkpoint"]).resolve(strict=True)
    if checkpoint.parent != (output / "checkpoints").resolve():
        raise ValueError("Observed checkpoint escapes the native fit output")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    centroid_keys = [f"quantization_layer_list.{layer}.centroids" for layer in range(3)]
    centroids = [saved["state_dict"][key] for key in centroid_keys]
    checks = {
        "global_step": saved["global_step"],
        "current_layer": saved["current_layer"],
        "layers_initialized": saved["layers_initialized"],
        "centroid_shapes": [list(values.shape) for values in centroids],
        "centroids_finite": [bool(torch.isfinite(values).all()) for values in centroids],
    }
    (output / "checkpoint_evidence.json").write_text(json.dumps(checks, indent=2) + "\n")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    NATIVE = load_entry()
    main()
