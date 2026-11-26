"""Extract metadata and the frozen state digest from the terminal job's own best, on CPU."""
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import sys

os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  TF_CPP_MIN_LOG_LEVEL="3", TF_NUM_INTRAOP_THREADS="1", TF_NUM_INTEROP_THREADS="1")
job = Path(sys.argv[1]).resolve(strict=True)
status = json.loads((job / "status.json").read_text())
assert status["status"] == "SUCCEEDED" and status["cleanup"]["verified"]
request = json.loads((job / "request.json").read_text())
baseline = Path(request["argv"][2]).parents[1] / "letter_instruments_grid_normal"
sys.path[:0] = [str(baseline), str(baseline / "native")]
import torch
from src.utils.launcher_utils import pipeline_launcher  # Establish the training import order.
from run_normal import state_digest

training = job / "output/run/training"
best = json.loads((training / "best.json").read_text())
path = (training / best["file"]).resolve(strict=True)
assert path.parent == (training / "checkpoints").resolve()
assert list(path.parent.glob("*.ckpt")) == [path]
before = path.stat()
blob = path.read_bytes()
after = path.stat()
assert (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
checkpoint_sha = hashlib.sha256(blob).hexdigest()
assert checkpoint_sha == best["sha256"]
data = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)
model_sha = state_digest(data["state_dict"])
assert model_sha == best["state_sha256"]
assert data["global_step"] == best["best_step"]
key, selected = next((key, value) for key, value in data["callbacks"].items() if key.startswith("ModelCheckpoint"))
early_key, early = next((key, value) for key, value in data["callbacks"].items() if key.startswith("EarlyStopping"))
assert selected["monitor"] == "val/recall@5" and Path(selected["best_model_path"]) == path
assert float(selected["best_model_score"]) == best["score"]
optimizer, = data["optimizer_states"]
print(json.dumps({
    "captured_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "source": str(path), "bytes": before.st_size, "inode": before.st_ino, "mtime_ns": before.st_mtime_ns,
    "checkpoint_sha256": checkpoint_sha, "state_sha256": model_sha, "state_keys": len(data["state_dict"]),
    "global_step": data["global_step"], "epoch": data["epoch"], "lightning_version": data["pytorch-lightning_version"],
    "selection": {"key": key, "monitor": selected["monitor"], "best_model_path": selected["best_model_path"],
                  "best_model_score": float(selected["best_model_score"]),
                  "best_k_models": {name: float(score) for name, score in selected["best_k_models"].items()}},
    "early_stopping_at_best": {"key": early_key, "patience": early["patience"],
                               "wait_count": early["wait_count"], "best_score": float(early["best_score"]),
                               "stopped_epoch": early["stopped_epoch"]},
    "optimizer": {"state_entries": len(optimizer["state"]),
                  "step_values": sorted({int(item["step"]) for item in optimizer["state"].values()}),
                  "param_groups": [{key: g[key] for key in ("lr", "weight_decay", "betas")}
                                   for g in optimizer["param_groups"]]}
}, allow_nan=False))
