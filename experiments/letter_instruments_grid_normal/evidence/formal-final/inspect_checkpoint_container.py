"""Verify bound Lightning checkpoint tensor bytes without importing framework code."""
import argparse
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import pickletools
import struct
import time
from zipfile import ZipFile

STORAGE_DTYPES = {"FloatStorage": ("torch.float32", 4), "LongStorage": ("torch.int64", 8)}


@dataclass
class Tensor:
    storage: tuple
    offset: int
    shape: tuple
    strides: tuple
    requires_grad: bool
    hooks: OrderedDict


class CheckpointMetadata(pickle.Unpickler):
    """Decode only the tensor and callback prefix with inert storage descriptors."""

    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return OrderedDict
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return Tensor
        if module == "torch" and name in STORAGE_DTYPES:
            return name
        raise pickle.UnpicklingError(f"Unexpected checkpoint reference {module}.{name}")

    def persistent_load(self, identifier):
        kind, storage_type, key, device, count = identifier
        assert kind == "storage" and storage_type in STORAGE_DTYPES
        assert isinstance(key, str) and key.isascii() and key.isdecimal()
        assert device in ("cpu", "cuda:0") and type(count) is int and count > 0
        return storage_type, key, device, count


def tensor_bytes(bundle, tensor):
    typ, key, device, size = tensor.storage
    dtype, width = STORAGE_DTYPES[typ]
    expected_strides = tuple(math.prod(tensor.shape[i + 1:]) for i in range(len(tensor.shape)))
    assert all(s == e or n == 1 for s, e, n in zip(tensor.strides, expected_strides, tensor.shape, strict=True))
    raw = bundle.read(f"archive/data/{key}")
    assert len(raw) == size * width and tensor.offset >= 0
    final = tensor.offset + math.prod(tensor.shape)
    assert final <= size
    return dtype, raw[tensor.offset * width:final * width]


def verify(run):
    start = time.monotonic()
    model = json.loads((run / "stage_result.json").read_bytes())["checkpoint"]
    receipt = json.loads((run / "reload_enabled.json").read_bytes())
    assert receipt["strict"] and receipt["missing"] == receipt["unexpected"] == []
    path = run / model["path"]
    assert path.resolve().is_relative_to(run.resolve()) and path.stat().st_size == model["bytes"]
    with path.open("rb") as handle:
        file_sha = hashlib.file_digest(handle, "sha256").hexdigest()
    assert file_sha == model["sha256"] == receipt["checkpoint_sha256"]
    with ZipFile(path) as bundle:
        assert bundle.testzip() is None
        assert bundle.read("archive/byteorder") == b"little"
        member_count = len(bundle.namelist())
        payload = bundle.read("archive/data.pkl")
        [hparams] = [i for op, value, i in pickletools.genops(payload)
                     if op.name == "BINUNICODE" and value == "hparams_name"]
        # Only the prefix before custom hyperparameter objects participates.
        # SETITEMS closes the already-open top-level dictionary and STOP returns it.
        checkpoint = CheckpointMetadata(io.BytesIO(payload[:hparams] + b"u.")).load()
        assert set(checkpoint) == {"epoch", "global_step", "pytorch-lightning_version",
                                   "state_dict", "loops", "callbacks", "optimizer_states", "lr_schedulers"}
        assert checkpoint["global_step"] == model["step"]
        assert checkpoint["loops"] and checkpoint["lr_schedulers"] == []
        assert len(checkpoint["state_dict"]) == receipt["keys"]
        digest = hashlib.sha256()
        for name, tensor in sorted(checkpoint["state_dict"].items()):
            dtype, raw = tensor_bytes(bundle, tensor)
            digest.update(json.dumps([name, dtype, list(tensor.shape)]).encode() + b"\0")
            digest.update(raw)
        hashed_state = digest.hexdigest()
        assert hashed_state == model["model_state_sha256"] == receipt["model_state_sha256"]
        [adam] = checkpoint["optimizer_states"]
        [group] = adam["param_groups"]
        assert group["lr"] == .001 and group["weight_decay"] == .0001
        assert group["betas"] == (.9, .999) and group["eps"] == 1e-8
        assert len(adam["state"]) == len(group["params"]) == model["optimizer_state_entries"]
        adam_steps = {int(struct.unpack("<f", tensor_bytes(bundle, v["step"])[1])[0])
                      for v in adam["state"].values()}
        assert adam_steps == {model["step"]}
        callbacks = checkpoint["callbacks"]
        assert set(callbacks) == set(model["callbacks"])
        [early] = [v for k, v in callbacks.items() if k.startswith("EarlyStopping")]
        [best] = [v for k, v in callbacks.items() if k.startswith("ModelCheckpoint")]
        assert Path(best["best_model_path"]).name == path.name
        assert early["wait_count"] == 0 and early["patience"] == 10
        callback_scores = [struct.unpack("<f", tensor_bytes(bundle, v)[1])[0]
                           for v in (early["best_score"], best["best_model_score"])]
        assert callback_scores == [model["score"]] * 2
    return {"checkpoint_sha256": file_sha, "container_crc_members": member_count,
            "global_step": checkpoint["global_step"], "epoch": checkpoint["epoch"],
            "checkpoint_model_tensor_keys": receipt["keys"], "checkpoint_model_state_sha256": hashed_state,
            "optimizer_state_entries": len(adam["state"]), "optimizer_unique_steps": sorted(adam_steps),
            "val_recall_at_5_callback_score": callback_scores[0], "early_stopping_wait_count_at_best": early["wait_count"],
            "early_stopping_patience": early["patience"], "framework_execution": False,
            "inspect_wall_seconds": time.monotonic() - start}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.run), indent=2))
