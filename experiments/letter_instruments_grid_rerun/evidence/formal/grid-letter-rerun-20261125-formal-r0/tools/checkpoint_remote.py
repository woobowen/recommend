"""Read-only CPU extraction from this job's trusted, stable checkpoints."""
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import torch

JOB = Path("/home/bwwu/grid-audits/sid-extensions-p0/jobs/grid-letter-rerun-20261125-formal-r0")
RUN = JOB / "output/run"


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def identity(info):
    return {"inode": info.st_ino, "device": info.st_dev, "bytes": info.st_size, "mtime_ns": info.st_mtime_ns}


def stable(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular checkpoint under {JOB}: {path}")
    with path.open("rb") as stream:
        before = identity(os.fstat(stream.fileno()))
        blob = stream.read()
        after = identity(os.fstat(stream.fileno()))
    assert before == after and before == identity(path.stat()) and len(blob) == before["bytes"], path
    return blob, {"source": str(path), "checkpoint_sha256": sha(blob), "before": before, "after": after,
                  "captured_at": utc()}


def extract_fit(request):
    upstream = RUN / "upstream"
    gate = json.loads((upstream / "fit_gate.json").read_text())
    checkpoint = (upstream / gate["checkpoint"]).resolve(strict=True)
    assert checkpoint.parent == upstream / "quantizer_train/checkpoints"
    blob, evidence = stable(checkpoint)
    assert evidence["checkpoint_sha256"] == gate["checkpoint_sha256"]
    package = Path(request["argv"][2]).parent
    sys.path[:0] = [str(package), str(package / "upstream")]
    from runtime import load_entry
    load_entry()
    full = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)
    names = ("global_step", "epoch", "current_layer", "layers_initialized", "pytorch-lightning_version",
             "optimizer_states")
    retained = {name: full[name] for name in names}
    retained["state_dict"] = {name: array.clone() for name, array in full["state_dict"].items()}
    assert retained["global_step"] == 30 and retained["layers_initialized"] == [True, True, True]
    payload = io.BytesIO()
    torch.save(retained, payload)
    data = payload.getvalue()
    loaded = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    assert all(retained[n] == loaded[n] for n in names)
    assert loaded["state_dict"].keys() == full["state_dict"].keys()
    codebooks = {}
    for key, value in loaded["state_dict"].items():
        original = full["state_dict"][key].contiguous()
        assert value.shape == original.shape and value.dtype == original.dtype
        bits = value.contiguous().numpy().tobytes()
        assert bits == original.numpy().tobytes()
        codebooks[key] = {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": sha(bits)}
    evidence.update(kind="lossless state_dict and optimizer state extraction", mode=gate["mode"],
                    omitted_top_level_keys=sorted(full.keys() - retained.keys()), codebooks=codebooks,
                    metadata={key: retained[key] for key in names if key != "optimizer_states"})
    return evidence, {"quantizer_native_state.pt": data}


def extract_best(request, rel_path, expected):
    folder = RUN / "training/checkpoints"
    path = folder / rel_path
    assert path.parent == folder and path.suffix == ".ckpt"
    blob, evidence = stable(path)
    assert all(evidence["before"][k] == expected[k] for k in ("bytes", "mtime_ns", "inode", "device"))
    # The run is our own pinned source. Its Lightning checkpoint serializes native config classes.
    baseline = Path(request["argv"][2]).parents[1] / "letter_instruments_grid_normal"
    sys.path[:0] = [str(baseline), str(baseline / "native")]
    # Initialize native modules in the same order as downstream.fit.
    from src.utils.launcher_utils import pipeline_launcher  # registers native imports

    full = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)
    choices = []
    for key, cb in full["callbacks"].items():
        if not key.startswith("ModelCheckpoint"):
            continue
        choices.append({"key": key, "monitor": cb["monitor"], "best_model_path": cb["best_model_path"],
                        "best_model_score": float(cb["best_model_score"]),
                        "current_score": float(cb["current_score"]),
                        "best_k_models": {key: float(val) for key, val in cb["best_k_models"].items()}})
    assert len(choices) == 1 and choices[0]["monitor"] == "val/recall@5"
    assert Path(choices[0]["best_model_path"]) == path
    opt = full["optimizer_states"]
    evidence.update(kind="read-only bounded best checkpoint metadata", global_step=full["global_step"],
                    epoch=full["epoch"], callbacks=choices,
                    optimizer=[{"state_entries": len(o["state"]),
                                "step_values": sorted({int(v["step"]) for v in o["state"].values()}),
                                "param_groups": [{k: g[k] for k in ("lr", "weight_decay", "betas")} for g in o["param_groups"]]}
                               for o in opt],
                    pytorch_lightning_version=full["pytorch-lightning_version"])
    return evidence, {}


def main():
    started = utc()
    request = json.loads((JOB / "request.json").read_text())
    mode = sys.argv[1]
    if mode == "quantizer":
        records, files = extract_fit(request)
    else:
        assert mode == "best"
        records, files = extract_best(request, sys.argv[2], json.loads(sys.argv[3]))
    files["checkpoint_extraction.json"] = (json.dumps(records, indent=2) + "\n").encode()
    with tarfile.open(mode="w|", fileobj=sys.stdout.buffer) as dest:
        entries = {name: {"bytes": len(data), "sha256": sha(data)} for name, data in files.items()}
        manifest = {"selection": mode, "started_utc": started, "ended_utc": utc(), "files": entries}
        files["capture.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
        for name, data in files.items():
            member = tarfile.TarInfo(name)
            member.mode = 0o600
            member.size = len(data)
            dest.addfile(member, io.BytesIO(data))


if __name__ == "__main__":
    main()
