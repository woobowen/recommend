"""Normal-only native GRID training and bounded preflight."""
import argparse
from collections import Counter
from contextlib import ExitStack
import importlib.metadata
import fcntl
import hashlib
import json
import math
import os
import random
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from unittest.mock import patch

from candidates import CandidateSink, score_words
from data_assets import HISTORICAL, PREPARED, load, official, sha, verify as verify_assets, write
from evaluate import fixtures, verify as evaluate

HERE = Path(__file__).resolve().parent
NATIVE_ROOT = HERE / "native"
OLD_ROOT = "/home/jcy/GRID_wmh_Qwen_TIGER_20261009"
OLD_IN = OLD_ROOT + "/codex/letter_official_downstream_20261116/inputs/instruments"
OLD_OUT = OLD_ROOT + "/experiments/letter_official_instruments_yelp_20261116_142754/instruments/formal"
SEED, EVAL_SEED = 42, 2026
PREFLIGHT_TRAIN = 1600
PREFLIGHT_VAL = 24
TOTAL_USERS = 24772
RUNNER_PREFIX = "grid-letter-instruments-normal-20261124"


def configure(assets, output, preflight):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict
    import src.utils.custom_hydra_resolvers  # registers the native YAML resolvers

    prepared = assets / PREPARED
    paths = {OLD_IN: str(prepared), OLD_OUT: str(output), OLD_ROOT: str(NATIVE_ROOT)}

    def relocate(value):
        if isinstance(value, str):
            for old, new in paths.items():
                value = value.replace(old, new)
            return value
        if isinstance(value, dict):
            return {k: relocate(v) for k, v in value.items()}
        if isinstance(value, list):
            return [relocate(v) for v in value]
        return value

    reference = OmegaConf.to_container(OmegaConf.load(HISTORICAL / "config.yaml"), resolve=True)
    reference = relocate(reference)
    overrides = relocate(load(HISTORICAL / "overrides.json"))
    with initialize_config_dir(config_dir=str(NATIVE_ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=overrides)
    with open_dict(cfg):
        cfg.paths.output_dir = str(output)
        cfg.paths.profile_dir = str(output / "profile_output")
        cfg.paths.metadata_dir = str(output / "metadata")
        cfg.trainer.default_root_dir = str(output)
        cfg.callbacks.model_checkpoint.dirpath = str(output / "checkpoints")
        cfg.logger.csv.save_dir = str(output)
        # This legacy feature list is never used by an instantiated object. Every
        # live dataset stores its own resolved semantic_id_map pointing to the tensor.
        placeholder = cfg.data_loading.features_config.features[0]
        assert OmegaConf.is_missing(placeholder, "semantic_ids")
        del placeholder["semantic_ids"]
        del reference["data_loading"]["features_config"]["features"][0]["semantic_ids"]
    actual = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    # Hydra's timestamped run identifier is metadata only.
    differences = {k: (actual[k], reference.get(k)) for k in actual if k != "id" and actual[k] != reference.get(k)}
    assert not differences and set(actual) == set(reference), differences
    assert cfg.ckpt_path is None and cfg.seed == SEED and cfg.model.eval_seed == EVAL_SEED
    assert cfg.trainer.accumulate_grad_batches == 16 and cfg.trainer.val_check_interval == 1600
    assert cfg.model.top_k_for_generation == 10 and not cfg.model.shuffle_embedding_on_test
    cfg.id = output.parents[1].name
    if preflight:
        additions = ["trainer.max_steps=100", "trainer.max_epochs=1", "++trainer.limit_train_batches=1600",
                     "++trainer.limit_val_batches=24",
                     f"data_loading.test_dataloader_config.dataloader.data_folder={prepared / 'preflight_testing'}"]
        cfg.trainer.max_steps = 100
        cfg.trainer.max_epochs = 1
        with open_dict(cfg.trainer):
            cfg.trainer.limit_train_batches = PREFLIGHT_TRAIN
            cfg.trainer.limit_val_batches = PREFLIGHT_VAL
        cfg.data_loading.test_dataloader_config.dataloader.data_folder = str(prepared / "preflight_testing")
        overrides.extend(additions)
    return cfg, overrides


def verify_sources():
    originals = load(HISTORICAL / "source_manifest.json")["files"]
    patched = {"codex/plan_v1_1/protocol.py", "src/components/eval_metrics.py"}
    mismatches = {name for name, digest in originals.items() if sha(NATIVE_ROOT / name) != digest}
    assert mismatches == patched, mismatches
    return {"frozen_manifest_entries": len(originals), "adapted_files": sorted(patched)}


def gpu_admission():
    for key in ("LOCAL_RANK", "RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "METADATA_DIR"):
        os.environ.pop(key, None)
    uuid = os.environ["CUDA_VISIBLE_DEVICES"]
    assert uuid.startswith("GPU-") and "," not in uuid
    cmd = ["nvidia-smi", "-i", uuid, "--query-gpu=index,name,uuid,memory.total,memory.used",
           "--format=csv,noheader,nounits"]
    fields = subprocess.check_output(cmd, text=True).strip().split(", ")
    assert len(fields) == 5 and fields[2] == uuid and "4090" in fields[1]
    assert int(fields[4]) <= 256, f"GPU already occupied: {fields}"
    assert shutil.disk_usage(os.environ["GRID_JOB_DIR"]).free > 2 * 1024**3
    return dict(zip(("index", "name", "uuid", "memory_total_mib", "memory_used_at_admission_mib"), fields))


def model_identity(model, codes):
    import torch

    table = model.get_embedding_table("encoder")
    sid_match = torch.equal(model.codebooks.cpu(), torch.tensor(codes))
    heads = [list(head.weight.shape) for head in model.decoder.decoder_mlp]
    initial = hashlib.sha256()
    for name, param in model.named_parameters():
        initial.update(name.encode() + b"\x00")
        initial.update(param.detach().cpu().contiguous().numpy().tobytes())
    info = {"initial_parameters_sha256": initial.hexdigest(),
            "parameters": sum(t.numel() for t in model.parameters()),
            "trainable_parameters": sum(t.numel() for t in model.parameters() if t.requires_grad),
            "dtype_counts": dict(Counter(str(t.dtype) for t in model.parameters())),
            "encoder_layers": len(model.encoder.encoder.encoder.block),
            "decoder_layers": len(model.decoder.decoder.block),
            "heads": heads, "bos": list(model.decoder.bos_token.shape),
            "sep": list(model.sep_token.shape), "uses_user_embedding": model.user_embedding is not None,
            "sid_slots": table.num_embeddings, "embedding_width": table.embedding_dim,
            "shared": table is model.get_embedding_table("decoder"), "sid_matches_official": sid_match,
            "beam": model.top_k_for_generation, "prefix_filter": model.should_check_prefix,
            "matmul_precision": torch.get_float32_matmul_precision(),
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"]}
    assert info["parameters"] == info["trainable_parameters"] == 13128064
    assert info["shared"] and sid_match and heads == [[256, 128]] * 4 and table.num_embeddings == 1024
    assert not info["prefix_filter"] and not info["uses_user_embedding"] and info["beam"] == 10
    return info


def train_and_test(cfg, assets, output, preflight):
    import lightning as L
    import numpy as np
    from lightning.pytorch.callbacks import Callback
    import torch
    from src.utils.launcher_utils import pipeline_launcher

    class Recorder(Callback):
        def __init__(self):
            self.expanded = []
            self.val_batches = 0
            self.training_start = None
            self.validation_start = None
            self.training_window = None
            self.validation_seconds = []
            self.evaluations = 0
            self.update_probes = []
            self.pending = None

        def event(self, trainer, phase, **extra):
            with (output / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"event": phase, "time": time.time(), "step": trainer.global_step,
                                         "epoch": trainer.current_epoch, **extra}, allow_nan=False) + "\n")

        def on_train_start(self, t, m):
            assert t.global_step == 0 and not t.optimizers[0].state and m.shuffle_calls == 0
            params = {k: v for k, v in t.optimizers[0].param_groups[0].items() if k != "params"}
            self.event(t, "train_start", optimizer_state_entries=0, adam=params)
            self.training_start = self.training_window = time.monotonic()

        def on_train_batch_end(self, t, m, outputs, batch, batch_idx):
            if preflight:
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs
                loss_value = float(loss.detach())
                assert math.isfinite(loss_value)
                self.expanded.append(int(batch[0].mask.shape[0]))
                assert 0 < self.expanded[-1] <= 128
                if self.pending is not None:
                    old, gradient_norms = self.pending
                    changes = {k: not torch.equal(v, dict(m.named_parameters())[k].detach()) for k, v in old.items()}
                    assert all(changes.values())
                    probe = {"step": t.global_step, "loss_accumulation_scaled": loss_value,
                             "gradient_l2": gradient_norms, "changed": changes}
                    self.update_probes.append(probe)
                    self.event(t, "optimizer_probe", **{k: v for k, v in probe.items() if k != "step"})
                    self.pending = None
            if batch_idx == 0 or (batch_idx + 1) % 160 == 0:
                self.event(t, "train_batch", microbatch=batch_idx + 1, shuffle_calls=m.shuffle_calls)

        def on_before_optimizer_step(self, t, m, optimizer):
            if preflight and t.global_step in (0, 99):
                selected = {}
                gradients = {}
                for name, param in m.named_parameters():
                    if param.grad is None:
                        assert name.endswith(("shared.weight", "embed_tokens.weight")), name
                        gradients[name] = None
                    else:
                        assert torch.isfinite(param.grad).all(), name
                        gradients[name] = float(param.grad.norm())
                    if name in ("sep_token", "decoder.bos_token", "item_sid_embedding_table_encoder.weight") or (
                            "decoder_mlp" in name or name.endswith("block.0.layer.0.SelfAttention.q.weight")):
                        assert gradients[name] and gradients[name] > 0, name
                        selected[name] = param.detach().clone()
                assert len(selected) == 9, selected.keys()
                self.pending = (selected, gradients)

        def on_validation_start(self, t, m):
            torch.cuda.synchronize()
            self.validation_start = time.monotonic()
            self.val_batches = 0
            assert not m.shuffle_embedding_on_test and m.shuffle_calls == 0
            self.event(t, "validation_start", microbatches=len(self.expanded) if preflight else None,
                       train_seconds_since_validation=self.validation_start - self.training_window)

        def on_validation_batch_end(self, t, m, outputs, batch, batch_idx, dataloader_idx=0):
            self.val_batches += 1

        def on_validation_end(self, t, m):
            torch.cuda.synchronize()
            self.training_window = time.monotonic()
            duration = self.training_window - self.validation_start
            self.validation_seconds.append(duration)
            self.event(t, "validation_end", seconds=duration, batches=self.val_batches,
                       monitor=float(t.callback_metrics["val/recall@5"]))
            if preflight:
                assert t.global_step == 100 and len(self.expanded) == 1600 and self.val_batches == 24

        def on_test_start(self, t, m):
            assert m.shuffle_calls == 0 and not m.shuffle_embedding_on_test
            self.evaluations += 1
            self.test_batch_sizes = Counter()
            self.test_metric_counts = {}
            self.event(t, "test_start", evaluation=self.evaluations)

        def on_test_batch_end(self, t, m, outputs, batch, batch_idx, dataloader_idx=0):
            self.test_batch_sizes[int(batch[0].mask.shape[0])] += 1
            self.test_metric_counts = {k: v.total_values for k, v in m.evaluator.metrics.items()}

        def on_test_end(self, t, m):
            expected = 20 if preflight else TOTAL_USERS
            assert self.test_batch_sizes == {8: expected // 8, 4: 1}
            assert set(self.test_metric_counts.values()) == {expected}
            self.event(t, "test_end", evaluation=self.evaluations, shuffle_calls=m.shuffle_calls,
                       batch_sizes=dict(self.test_batch_sizes), metric_denominators=self.test_metric_counts)

        def on_exception(self, t, m, exc):
            self.event(t, "error", error=repr(exc))

    before = time.monotonic()
    codes, sequences, _ = official(assets)
    axes = [seq[-21:] for seq in sequences]
    recorder = Recorder()
    result = {"mode": "preflight" if preflight else "formal", "test": "Normal",
              "train_seed": SEED, "eval_seed": EVAL_SEED}
    with pipeline_launcher(cfg) as p:
        model, trainer = p.model, p.trainer
        trainer.callbacks.append(recorder)
        info = model_identity(model, codes)
        write(output / "model.json", info)
        # pipeline_launcher seeds before instantiation. Preserve the source's reseeding before fit.
        L.seed_everything(SEED, workers=True)
        result["startup_and_instance_seconds"] = time.monotonic() - before
        trainer.fit(model, datamodule=p.datamodule, ckpt_path=None)
        assert model.shuffle_calls == 0 and not model.shuffle_embedding_on_test
        result["global_step"] = trainer.global_step
        result["fit_seconds"] = time.monotonic() - recorder.training_start
        result["validation_seconds"] = recorder.validation_seconds
        if preflight:
            assert trainer.global_step == 100 and recorder.val_batches == 24 and len(recorder.update_probes) == 2
            result["expanded_microbatch"] = {"min": min(recorder.expanded), "max": max(recorder.expanded),
                                              "mean": statistics.mean(recorder.expanded),
                                              "distribution": dict(Counter(recorder.expanded))}
        best = Path(trainer.checkpoint_callback.best_model_path).resolve()
        assert best.is_file() and best.is_relative_to(output)
        best_state = torch.load(best, map_location="cpu", weights_only=False)
        assert set(best_state["state_dict"]) == set(model.state_dict()) and best_state["optimizer_states"]
        assert best_state["optimizer_states"][0]["state"]
        result["checkpoint"] = {"path": str(best.relative_to(output)), "sha256": sha(best),
                                "bytes": best.stat().st_size, "step": best_state["global_step"],
                                "score": float(trainer.checkpoint_callback.best_model_score),
                                "optimizer_state_entries": len(best_state["optimizer_states"][0]["state"]),
                                "callbacks": list(best_state["callbacks"])}
        model.strict_loading = True
        offline_stop = next(c for c in trainer.callbacks if type(c).__name__ == "EarlyStopping")
        if preflight:
            from copy import deepcopy
            check = deepcopy(offline_stop)
            check.wait_count, check.best_score = 0, torch.tensor(0.25)
            for i in range(10):
                stop, _ = check._evaluate_stopping_criteria(torch.tensor(0.25))
                assert check.wait_count == i + 1 and bool(stop) == (i == 9)
            check._evaluate_stopping_criteria(torch.tensor(0.5))
            assert check.wait_count == 0
            result["early_stopping_fixture"] = {"equal_checks_to_stop": 10, "better_score_resets_count": True}
        del best_state

        # The shadow pass is bounded preflight only. The formal test has one call
        # per batch. Both preflight passes use the same best weights and input order.
        signatures = []

        def testing(active):
            L.seed_everything(EVAL_SEED, workers=True)
            output_file = output / "candidates.jsonl"
            fingerprints = []
            generate = model.generate
            decoder = model.decoder_forward_pass
            cache_checks = [0]

            def observe_generate(*args, **kwargs):
                assert not args and set(kwargs) == {"attention_mask", "output_attentions", "input_ids", "user_id"}
                assert kwargs["output_attentions"] is False
                initial = cache_checks[0]
                ids, scores = generate(*args, **kwargs)
                assert cache_checks[0] == initial + 1
                if preflight:
                    fingerprints.append((ids.cpu().tolist(), [score_words(v) for v in scores.cpu().tolist()]))
                return ids, scores

            def observe_decoder(*args, **kwargs):
                if kwargs.get("use_cache") and kwargs.get("future_ids") is None:
                    cache = kwargs["past_key_values"]
                    assert len(cache.self_attention_cache) == len(cache.cross_attention_cache) == 0
                    cache_checks[0] += 1
                return decoder(*args, **kwargs)

            start = time.monotonic()
            with ExitStack() as stack:
                sink = stack.enter_context(CandidateSink(output_file, model, axes, active=active))
                if preflight:
                    stack.enter_context(patch.object(model, "generate", observe_generate))
                    stack.enter_context(patch.object(model, "decoder_forward_pass", observe_decoder))
                metrics = trainer.test(model, datamodule=p.datamodule, ckpt_path=str(best))
            elapsed = time.monotonic() - start
            assert model.shuffle_calls == 0 and model.strict_loading
            assert all(math.isfinite(v) for v in metrics[0].values())
            if active:
                assert len(sink.users) == (20 if preflight else TOTAL_USERS)
                assert sink.generation_calls == (3 if preflight else 3097)
            if preflight:
                assert cache_checks[0] == len(fingerprints) == 3
                numpy_rng = np.random.get_state()
                signatures.append((fingerprints, metrics, torch.random.get_rng_state().tolist(),
                                   torch.cuda.get_rng_state().tolist(), random.getstate(),
                                   (numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:])))
            return metrics, elapsed

        metrics, result["test_seconds"] = testing(True)
        if preflight:
            _, result["shadow_test_seconds"] = testing(False)
            assert signatures[0] == signatures[1], "Observer affected native output or RNG"
            result["observer_equivalence"] = {"batches": len(signatures[0][0]), "directions": 2,
                                              "candidate_values_float32_bits_metrics_rng_equal": True,
                                              "fresh_cache_starts_from_bos": True}
        result["shuffle_calls"] = model.shuffle_calls
        result["peak_gpu_bytes"] = torch.cuda.max_memory_allocated()
    write(output / "native_metrics.json", metrics)
    cpu_t0 = time.monotonic()
    with (output / "per_user_metrics.jsonl").open("x", encoding="utf-8") as stream:
        evaluation = evaluate(assets, output / "candidates.jsonl", result["mode"], metrics[0], stream)
    write(output / "offline_evaluation.json", {"replay": evaluation, "fixtures": fixtures()})
    result["offline_seconds"] = time.monotonic() - cpu_t0
    write(output / "stage_result.json", result)


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("plan", "preflight", "formal"))
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plan-dir", type=Path)
    args = parser.parse_args()
    assets, output = args.assets.resolve(), args.output.resolve()
    is_plan = args.mode == "plan"
    assert not output.exists(), output
    if is_plan:
        assert args.plan_dir and "GRID_JOB_ID" not in os.environ
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        job_id = os.environ["GRID_JOB_ID"]
        assert job_id.startswith(f"{RUNNER_PREFIX}-{args.mode}-")
        assert output == Path(os.environ["GRID_JOB_DIR"]) / "output/run"
    os.environ.update(OMP_NUM_THREADS="4", TF_CPP_MIN_LOG_LEVEL="3", TF_NUM_INTRAOP_THREADS="2",
                      TF_NUM_INTEROP_THREADS="2", CUBLAS_WORKSPACE_CONFIG=":4096:8", HYDRA_FULL_ERROR="1",
                      PROJECT_ROOT=str(NATIVE_ROOT))
    sys.path.insert(0, str(NATIVE_ROOT))
    from omegaconf import OmegaConf
    asset_report = verify_assets(assets)
    sources = verify_sources()
    cfg, overrides = configure(assets, output, args.mode == "preflight")
    receipt_dir = args.plan_dir if is_plan else output
    receipt_dir.mkdir(parents=True)
    OmegaConf.save(cfg, receipt_dir / "effective_config.yaml", resolve=True)
    write(receipt_dir / "overrides.json", overrides)
    if is_plan:
        write(receipt_dir / "dry_run.json", {"config_only": True, "formal_output_exists": output.exists(),
                                           "formal_output_path": str(output), "coverage": asset_report,
                                           "source": sources, "argv": sys.argv})
        return
    # The file is shared by this experiment's deployments and attempts.
    with (assets.parent / "gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        compute(cfg, assets, output, args.mode == "preflight", asset_report, sources, started)


def compute(cfg, assets, output, preflight, asset_report, sources, started):
    gpu = gpu_admission()
    import torch
    assert torch.cuda.device_count() == 1
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_float32_matmul_precision("medium")
    keys = ("numpy", "tensorflow-cpu", "lightning", "torchmetrics", "transformers", "pyarrow", "hydra-core")
    write(output / "startup.json", {"argv": sys.argv, "cwd": os.getcwd(), "python": sys.executable,
                                    "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                                    "dependencies": {k: importlib.metadata.version(k) for k in keys},
                                    "gpu": gpu, "asset_coverage": asset_report, "source": sources,
                                    "initial_checks_seconds": time.monotonic() - started})
    train_and_test(cfg, assets, output, preflight)


if __name__ == "__main__":
    main()
