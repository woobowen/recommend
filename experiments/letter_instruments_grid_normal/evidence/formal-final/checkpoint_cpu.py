"""Read the trusted runner-bound checkpoint and strict-load a fresh model on CPU."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("assets", type=Path)
    parser.add_argument("runner_checkpoint_sha256")
    args = parser.parse_args()
    t0 = time.monotonic()
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4", TF_CPP_MIN_LOG_LEVEL="3",
                      TF_NUM_INTRAOP_THREADS="2", TF_NUM_INTEROP_THREADS="2")
    sys.path[:0] = [str(args.runtime / "native"), str(args.runtime)]
    import lightning as L
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    import torch
    from run_normal import PREPARED, sha, state_digest

    root = args.output.resolve()
    stage = json.loads((root / "stage_result.json").read_text())
    expected = stage["checkpoint"]
    ckpt = (root / expected["path"]).resolve(strict=True)
    assert ckpt.is_relative_to(root) and sha(ckpt) == args.runner_checkpoint_sha256 == expected["sha256"]
    assert ckpt.stat().st_size == expected["bytes"]
    assert stage["mode"] == "formal" and stage["shuffle_calls"] == 0
    assert torch.cuda.device_count() == 0
    torch.set_num_threads(4)
    cfg = OmegaConf.load(root / "effective_config.yaml")
    sid = args.assets / PREPARED / "semantic_ids.pt"
    cfg.model.codebooks._args_[0].file_path = str(sid.resolve())
    assert cfg.seed == 42 and cfg.ckpt_path is None and cfg.model.eval_seed == 2026
    L.seed_everything(cfg.seed, workers=True)
    model = instantiate(cfg.model)
    named = list(model.named_parameters())
    assert sum(p.numel() for _, p in named if p.requires_grad) == 13_128_064
    assert all(p.dtype == torch.float32 and p.device.type == "cpu" for _, p in named)
    assert model.get_embedding_table("encoder") is model.get_embedding_table("decoder")
    assert torch.equal(model.codebooks.cpu(), torch.load(sid, map_location="cpu", weights_only=True).T)
    initial = hashlib.sha256()
    for name, param in named:
        initial.update(name.encode() + b"\0")
        initial.update(param.detach().cpu().contiguous().numpy().tobytes())
    assert initial.hexdigest() == json.loads((root / "model.json").read_text())["initial_parameters_sha256"]
    # The original runner and stage both bind this locally produced checkpoint's
    # SHA-256. Reuse the frozen run's full Lightning deserializer because
    # hyperparameters contain functools.partial and native classes.
    data = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = data["state_dict"]
    assert data["global_step"] == expected["step"] and data["loops"]
    assert all(torch.isfinite(t).all() for t in state.values() if t.is_floating_point())
    assert set(state) == set(model.state_dict())
    receipt = model.load_state_dict(state, strict=True)
    assert receipt.missing_keys == receipt.unexpected_keys == []
    assert all(torch.equal(t, state[key]) for key, t in model.state_dict().items())
    digest = state_digest(model.state_dict())
    assert digest == expected["model_state_sha256"] == state_digest(state)
    original = json.loads((root / "reload_enabled.json").read_text())
    assert original["strict"] and original["missing"] == original["unexpected"] == []
    assert original["keys"] == len(state)
    assert original["checkpoint_sha256"] == expected["sha256"] and original["model_state_sha256"] == digest

    (adam,) = data["optimizer_states"]
    (group,) = adam["param_groups"]
    assert len(group["params"]) == len(named)
    assert (group["lr"], group["weight_decay"], group["betas"], group["eps"]) == (.001, .0001, (.9, .999), 1e-8)
    assert data["lr_schedulers"] == []
    names = dict(zip(group["params"], (name for name, _ in named)))
    assert set(adam["state"]) <= set(names)
    missing_adam = sorted(names[index] for index in names.keys() - adam["state"].keys())
    assert all(name.endswith(("shared.weight", "embed_tokens.weight")) for name in missing_adam)
    adam_steps = set()
    for index, value in adam["state"].items():
        saved = state[names[index]]
        assert saved.shape == value["exp_avg"].shape == value["exp_avg_sq"].shape
        assert torch.isfinite(value["exp_avg"]).all() and torch.isfinite(value["exp_avg_sq"]).all()
        assert (value["exp_avg_sq"] >= 0).all()
        adam_steps.add(int(value["step"].item()))
    assert adam_steps == {expected["step"]}
    assert len(adam["state"]) == expected["optimizer_state_entries"]

    callbacks = data["callbacks"]
    (best_key,) = [name for name in callbacks if name.startswith("ModelCheckpoint")]
    (early_key,) = [name for name in callbacks if name.startswith("EarlyStopping")]
    best, early = callbacks[best_key], callbacks[early_key]
    assert "val/recall@5" in best_key and "val/recall@5" in early_key
    assert math.isfinite(float(best["best_model_score"]))
    assert float(best["best_model_score"]) == float(early["best_score"]) == expected["score"]
    assert Path(best["best_model_path"]).name == ckpt.name
    assert early["patience"] == 10 and early["wait_count"] == 0
    print(json.dumps(dict(
        argv=sys.argv, python=sys.executable, torch=torch.__version__, lightning=L.__version__, cpu_only=True,
        checkpoint_sha256=expected["sha256"], bytes=expected["bytes"], global_step=data["global_step"],
        epoch=data["epoch"], strict_fresh_cpu_load=True, missing=receipt.missing_keys,
        unexpected=receipt.unexpected_keys, state_keys=len(state), model_state_sha256=digest,
        initial_parameter_sha256=initial.hexdigest(), verified_original_test_receipt=True,
        optimizer_states=len(adam["state"]), optimizer_unique_steps=sorted(adam_steps), missing_adam=missing_adam,
        callbacks=sorted(callbacks), best_score=expected["score"], checkpoint_early_stop_wait=early["wait_count"],
        checkpoint_early_stop_patience=early["patience"], elapsed_seconds=time.monotonic()-t0,
    ), indent=2))

if __name__ == "__main__":
    main()
