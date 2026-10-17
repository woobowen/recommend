"""Frozen T5-small and author residual MiniBatchKMeans Sports A2 stages.

The original model, datamodule, tokenization, pooling, quantizer, initializer,
SGD initialization loop and collision functions are imported unchanged.  This
entrypoint supplies explicit relocated paths, output/identity checks and a
prediction writer that also works without a single-device process group.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import sys
import time

from experiments.beauty_extensions.common import REPO, sha256, verify_asset, write_json


EMBED_CONFIG = "A800_GRID/logs/inference/runs/2025-09-15/19-47-36/.hydra/config.yaml"
RQ_CONFIG = "A800_GRID/logs/train/runs/2025-09-15/20-27-55/.hydra/config.yaml"
PREDICT_CONFIG = "A800_GRID/logs/inference/runs/2025-09-15/20-54-16/.hydra/config.yaml"
SPORTS_ITEMS = 18357


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def bind_sources(assets, output):
    from experiments.beauty_extensions.author.bootstrap import bind_author_source
    bind_author_source()
    sources = json.loads(Path(__file__).with_name("a2_sources.json").read_text())
    verified = []
    for record in sources["source_files"]:
        verified.append(verify_asset({**record, "path": str(REPO / record["path"])}))
    for record in sources["original_assets"]:
        if record["archive_path"].startswith("A800_T5_SMALL/") or record["archive_path"] in (EMBED_CONFIG, RQ_CONFIG, PREDICT_CONFIG):
            verified.append(verify_asset({**record, "path": str(assets / record["archive_path"])}))
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        write_json(output / "source_identity.json", {"verified": verified,
                   "source_manifest_sha256": sha256(Path(__file__).with_name("a2_sources.json")),
                   "identity": sources["identity"], "historical_uncertainties": sources["historical_uncertainties"]})
    return sources


def relocate_config(member, assets, data, output):
    from omegaconf import OmegaConf
    import src.utils.custom_hydra_resolvers  # noqa: F401
    cfg = OmegaConf.load(assets / member)
    cfg.data_dir = str(data)
    cfg.ckpt_path = None
    for key, value in {"root_dir": REPO / "references/wmh-original/GRID", "work_dir": REPO,
                       "output_dir": output, "log_dir": output, "profile_dir": output / "profile",
                       "metadata_dir": output / "metadata"}.items():
        cfg.paths[key] = str(value)
    # The original Hydra runtime path is unavailable in this explicit runner.
    # Other model/loader fields are inherited rather than reconstructed.
    if "callbacks" in cfg and "model_checkpoint" in cfg.callbacks:
        cfg.callbacks.model_checkpoint.dirpath = str(output / "checkpoints")
    cfg.trainer.default_root_dir = str(output)
    cfg.trainer.enable_progress_bar = False
    cfg.trainer.enable_model_summary = False
    cfg.trainer.num_sanity_val_steps = 0
    return cfg


def item_axis_gate(sources, data, output):
    """Verify current immutable item files, including the existing static text."""
    source = next(q for q in sources["current_item_axis_audit"]["items"] if q["domain"] == "sports")
    found = sorted((data / "items").glob("*.tfrecord.gz"))
    expected = source["files"]
    if [q.name for q in found] != [Path(q["path"]).name for q in expected]:
        raise RuntimeError("Sports item file coverage changed")
    for path, record in zip(found, expected):
        verify_asset({**record, "path": str(path)})
    assert source["n"] == SPORTS_ITEMS and source["axis_dense_zero_based"]
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        write_json(output / "item_axis.json", {"items": SPORTS_ITEMS, "id_min": 0, "id_max": SPORTS_ITEMS - 1,
                   "all_item_files_sha256_exact": True, "text_field": "preexisting TFRecord text; no rewriting",
                   "id_text_stream_sha256": source["id_text_stream_sha256"],
                   "frozen_splits": "same formal Sports items; no user holdout labels used"})


def components():
    import torch
    from lightning.pytorch.callbacks import Callback

    class Progress(Callback):
        def __init__(self, output, stage, hard_seconds, gpu_duty_cycle=1.0):
            self.output, self.stage, self.hard_seconds = output, stage, hard_seconds
            self.started = time.monotonic()
            self.processed = 0
            self.gpu_duty_cycle = gpu_duty_cycle
            self.compute_seconds = self.throttle_sleep_seconds = 0.0
            self.batch_started = None

        def receipt(self, trainer, module, **extra):
            if time.monotonic() - self.started >= self.hard_seconds:
                raise TimeoutError(self.stage + " worker hard time limit")
            if shutil.disk_usage(self.output).free < 10 * 1024**3:
                raise RuntimeError("Output filesystem has less than 10GiB reserve")
            if not trainer.is_global_zero:
                return
            value = {"stage": self.stage, "status": "RUNNING", "time": now(), "pid": os.getpid(),
                     "global_step": trainer.global_step, "processed_items_rank0": self.processed,
                     "world_size": trainer.world_size, "elapsed_seconds": time.monotonic() - self.started,
                     "gpu_duty_cycle_target": self.gpu_duty_cycle, "measured_compute_seconds": self.compute_seconds,
                     "throttle_sleep_seconds": self.throttle_sleep_seconds,
                     "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), **extra}
            write_json(self.output / "progress.json", value)
            print(json.dumps(value, allow_nan=False), flush=True)

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            value = outputs.get("loss") if isinstance(outputs, dict) else outputs
            if value is None or not torch.isfinite(value).all():
                raise RuntimeError("Nonfinite or missing residual quantization loss")
            self.receipt(trainer, module, loader_batch=batch_idx, loss=float(value.detach().cpu()),
                         current_layer=module.current_layer,
                         initialized=[bool(q.is_initialized) for q in module.quantization_layer_list])

        def on_predict_batch_start(self, trainer, module, batch, batch_idx, dataloader_idx=0):
            if self.gpu_duty_cycle < 1.0:
                torch.cuda.synchronize()
                self.batch_started = time.monotonic()

        def on_predict_batch_end(self, trainer, module, outputs, batch, batch_idx, dataloader_idx=0):
            self.processed += len(outputs.keys)
            if self.gpu_duty_cycle < 1.0:
                torch.cuda.synchronize()
                compute = time.monotonic() - self.batch_started
                sleep = compute * (1 / self.gpu_duty_cycle - 1)
                self.compute_seconds += compute
                time.sleep(sleep)
                self.throttle_sleep_seconds += sleep
            if batch_idx % 64 == 0:
                self.receipt(trainer, module, loader_batch=batch_idx)

    class PredictionWriter(Callback):
        def __init__(self, output, name):
            self.output, self.name = output, name
            self.rows = []

        def on_predict_batch_end(self, trainer, module, outputs, batch, batch_idx, dataloader_idx=0):
            for key, tensor in zip(outputs.keys, outputs.predictions):
                self.rows.append({"item_id": int(key), self.name: tensor.detach().cpu()})

        def on_predict_end(self, trainer, module):
            from src.utils.tensor_utils import merge_list_of_keyed_tensors_to_single_tensor
            rank_path = self.output / f"prediction-rank-{trainer.global_rank}.pt"
            torch.save(self.rows, rank_path)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            if trainer.is_global_zero:
                rows = []
                for rank in range(trainer.world_size):
                    rows.extend(torch.load(self.output / f"prediction-rank-{rank}.pt", weights_only=True))
                ids = [q["item_id"] for q in rows]
                if len(ids) != SPORTS_ITEMS or len(set(ids)) != SPORTS_ITEMS or set(ids) != set(range(SPORTS_ITEMS)):
                    raise RuntimeError("Prediction item coverage/identity mismatch")
                tensor = merge_list_of_keyed_tensors_to_single_tensor(rows, "item_id", self.name).cpu()
                if not torch.isfinite(tensor).all():
                    raise RuntimeError("Nonfinite item output")
                torch.save(tensor, self.output / "predictions.pt")
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

    return Progress, PredictionWriter


def encode(args, sources=None):
    if sources is None:
        sources = bind_sources(args.assets, args.output)
        item_axis_gate(sources, args.data, args.output)
    import hydra
    import torch
    from lightning import seed_everything
    from omegaconf import OmegaConf
    Progress, PredictionWriter = components()
    seed_everything(42, workers=True)
    cfg = relocate_config(EMBED_CONFIG, args.assets, args.data, args.output)
    cfg.embedding_model = str(args.assets / "A800_T5_SMALL")
    cfg.model.semantic_embedding_model.huggingface_model.local_files_only = True
    cfg.data_loading.tokenizer_config.tokenizer.local_files_only = True
    cfg.trainer.devices = 1
    cfg.trainer.strategy = "auto"
    cfg.trainer.precision = "32-true"
    model = hydra.utils.instantiate(cfg.model)
    assert all(not p.requires_grad for p in model.parameters())
    dm = hydra.utils.instantiate(cfg.data_loading.datamodule)
    OmegaConf.save(cfg, args.output / "effective_config.yaml", resolve=False)
    write_json(args.output / "used_config.json", {key: OmegaConf.to_container(cfg[key], resolve=True)
               for key in ("model", "trainer", "data_loading")})
    duty = getattr(args, "gpu_duty_cycle", 1.0)
    if not 0.0 < duty <= 1.0:
        raise ValueError("Invalid authorized GPU duty cycle")
    write_json(args.output / "resource_schedule.json", {"gpu_duty_cycle_target": duty,
               "throttle": "after exact native batch8 forward: synchronize and sleep compute*(1/duty-1)",
               "changes_to_model_inputs_method_rng_precision_batch": False,
               "idle_wall_charged_to_3hour_gpu_request_cap": True})
    trainer = hydra.utils.instantiate(cfg.trainer, logger=False, enable_checkpointing=False,
              callbacks=[Progress(args.output, "t5_encode", 10800, duty), PredictionWriter(args.output, "embedding")])
    trainer.predict(model, datamodule=dm, return_predictions=False)
    tensor = torch.load(args.output / "predictions.pt", weights_only=True)
    assert tuple(tensor.shape) == (SPORTS_ITEMS, 512) and tensor.dtype == torch.float32
    target = args.output / "item_embeddings.pt"
    torch.save(tensor, target)
    os.chmod(target, 0o444)
    write_json(args.output / "stage_result.json", {"status": "SUCCEEDED", "stage": "t5_encode", "time": now(),
               "path": str(target), "sha256": sha256(target), "shape": list(tensor.shape), "dtype": str(tensor.dtype),
               "frozen_t5": True, "normalization": "Identity after MeanAggregation; no vector normalization",
               "items": SPORTS_ITEMS, "item_axis_gate": "exact", "source_manifest_sha256": sha256(Path(__file__).with_name("a2_sources.json"))})


def quantize(args, sources=None):
    fd_engineering = getattr(args, "fd_engineering", None)
    if fd_engineering:
        from experiments.a2_fd_recovery_20261018.adapter import configure_process, adapt_config, resource_callback
        configure_process(args.output)
    if sources is None:
        sources = bind_sources(args.assets, args.output)
        item_axis_gate(sources, args.data, args.output)
    import hydra
    import torch
    from lightning import seed_everything
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from omegaconf import OmegaConf
    from src.utils.tensor_utils import deduplicate_rows_in_tensor, transpose_tensor_from_file
    Progress, PredictionWriter = components()
    assert args.embedding is not None
    embedding_result = json.loads((args.embedding.parent / "stage_result.json").read_text())
    assert embedding_result["status"] == "SUCCEEDED" and embedding_result["stage"] == "t5_encode"
    assert embedding_result["source_manifest_sha256"] == sha256(Path(__file__).with_name("a2_sources.json"))
    verify_asset({"path": str(args.embedding), "sha256": embedding_result["sha256"]})
    tensor = torch.load(args.embedding, map_location="cpu", weights_only=True)
    assert tuple(tensor.shape) == (SPORTS_ITEMS, 512) and tensor.dtype == torch.float32 and torch.isfinite(tensor).all()
    if torch.cuda.device_count() != 2:
        raise RuntimeError("Author Beauty A residual quantizer requires both locked GPUs")
    seed_everything(42, workers=True)
    cfg = relocate_config(RQ_CONFIG, args.assets, args.data, args.output)
    cfg.embedding_path = str(args.embedding)
    cfg.trainer.devices = 2
    cfg.trainer.strategy = "ddp_find_unused_parameters_true"
    cfg.trainer.precision = "bf16-mixed"
    # Add exactly the preserved inference loader to the preserved training DM.
    pred = relocate_config(PREDICT_CONFIG, args.assets, args.data, args.output)
    pred.embedding_path = str(args.embedding)
    cfg.data_loading.datamodule.predict_dataloader_config = OmegaConf.create(
        OmegaConf.to_container(pred.data_loading.datamodule.predict_dataloader_config, resolve=True))
    fd_callbacks = []
    if fd_engineering:
        changes = adapt_config(cfg)
        write_json(args.output / ("fd-config-rank-" + os.environ.get("LOCAL_RANK", "0") + ".json"), changes)
        fd_callbacks = [resource_callback(args.output, tensor)]
    model = hydra.utils.instantiate(cfg.model)
    dm = hydra.utils.instantiate(cfg.data_loading.datamodule)
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        OmegaConf.save(cfg, args.output / "effective_config.yaml", resolve=False)
        write_json(args.output / "used_config.json", {key: OmegaConf.to_container(cfg[key], resolve=True)
                   for key in ("model", "trainer", "optim", "data_loading")})
    checkpoint = ModelCheckpoint(dirpath=str(args.output / "checkpoints"), filename="quantizer_{step:06d}",
                    monitor="train/loss", mode="min", save_top_k=1, every_n_train_steps=30,
                    save_on_train_epoch_end=False, auto_insert_metric_name=False)
    trainer = hydra.utils.instantiate(cfg.trainer, logger=CSVLogger(str(args.output), name="csv"),
              callbacks=[checkpoint, Progress(args.output, "rkmeans_sid", fd_engineering["worker_hard_seconds"] if fd_engineering else 3600),
                         PredictionWriter(args.output, "cluster_ids"), *fd_callbacks])
    trainer.fit(model, datamodule=dm)
    if trainer.global_step != 30 or not all(q.is_initialized for q in model.quantization_layer_list):
        raise RuntimeError("Author residual quantization training did not complete all 30 steps/layers")
    trainer.save_checkpoint(str(args.output / "quantizer.ckpt"))
    if fd_engineering and trainer.is_global_zero:
        payload = torch.load(args.output / "quantizer.ckpt", map_location="cpu", weights_only=False)
        assert payload["global_step"] == 30
        current = model.state_dict()
        assert current.keys() == payload["state_dict"].keys()
        assert all(torch.equal(value.detach().cpu(), payload["state_dict"][key]) for key, value in current.items())
        strict = model.load_state_dict(payload["state_dict"], strict=True)
        assert not strict.missing_keys and not strict.unexpected_keys
        write_json(args.output / "quantizer-strict-load.json", {"strict_load": True, "global_step": 30,
                   "state_exact_to_trained_model": True, "keys": len(current),
                   "initialized": [bool(q.is_initialized) for q in model.quantization_layer_list]})
    trainer.predict(model, datamodule=dm, return_predictions=False)
    if trainer.is_global_zero:
        codes = torch.load(args.output / "predictions.pt", weights_only=True)
        assert tuple(codes.shape) == (SPORTS_ITEMS, 3) and 0 <= codes.min() and codes.max() < 256
        sid_path = args.output / "semantic_ids.pt"
        torch.save(codes.long(), sid_path)
        # Exact preserved export functions: fourth column is collision rank.
        deduplicate_rows_in_tensor(str(sid_path))
        transpose_tensor_from_file(str(sid_path))
        sid = torch.load(sid_path, weights_only=True)
        assert tuple(sid.shape) == (4, SPORTS_ITEMS) and sid.dtype == torch.long
        assert torch.equal(sid[:3], codes.T.long()) and 0 <= sid.min() and sid.max() < 256
        assert torch.unique(sid.T, dim=0).shape[0] == SPORTS_ITEMS
        os.chmod(sid_path, 0o444)
        write_json(args.output / "stage_result.json", {"status": "SUCCEEDED", "stage": "rkmeans_sid", "time": now(),
                   "path": str(sid_path), "sha256": sha256(sid_path), "shape": list(sid.shape),
                   "dtype": str(sid.dtype), "items": SPORTS_ITEMS, "fourth_code": "deterministic collision disambiguation",
                   "max_fourth_code": int(sid[3].max()), "quantization_layers": 3,
                   "world_size": trainer.world_size, "global_step": trainer.global_step,
                   "quantizer_checkpoint": str(args.output / "quantizer.ckpt"),
                   "quantizer_checkpoint_sha256": sha256(args.output / "quantizer.ckpt"),
                   "embedding_sha256": embedding_result["sha256"], "item_axis_gate": "exact",
                   "source_manifest_sha256": sha256(Path(__file__).with_name("a2_sources.json")),
                   "future_results_review": "NOT_PERFORMED; mechanical SID identity gate only"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["encode", "quantize"])
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("/data/bwwu/sid-extensions-assets/563dc50b1c7e/original_assets/data/amazon_data/sports"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedding", type=Path)
    parser.add_argument("--gpu-duty-cycle", type=float, default=1.0)
    args = parser.parse_args()
    args.assets, args.data, args.output = args.assets.resolve(), args.data.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "stage_result.json").exists():
        raise RuntimeError("Stage output already exists; do not duplicate/restart original job")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    sources = bind_sources(args.assets, args.output)
    item_axis_gate(sources, args.data, args.output)
    if args.stage == "encode":
        encode(args, sources)
    else:
        quantize(args, sources)


if __name__ == "__main__":
    main()
