"""Extract the exact fitted model tensors from the run's own native checkpoint."""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from upstream import FIT_GATE, digest
from runtime import load_entry


def extract(upstream, output):
    gate = json.loads((upstream / FIT_GATE).read_text())
    checkpoint = (upstream / gate["checkpoint"]).resolve(strict=True)
    if checkpoint.parent != (upstream / "quantizer_train/checkpoints").resolve():
        raise ValueError("Checkpoint escaped its run directory")
    checkpoint_hash = digest(checkpoint)
    if checkpoint_hash != gate["checkpoint_sha256"]:
        raise ValueError("Checkpoint failed the recorded fit identity check")
    load_entry()
    original = torch.load(checkpoint, map_location="cpu", weights_only=False)
    names = ("global_step", "epoch", "current_layer", "layers_initialized", "pytorch-lightning_version")
    evidence = {name: original[name] for name in names}
    evidence["state_dict"] = {name: tensor.clone() for name, tensor in original["state_dict"].items()}
    evidence["optimizer_states"] = original["optimizer_states"]
    output.mkdir(parents=True, exist_ok=False)
    artifact = output / "quantizer_native_state.pt"
    torch.save(evidence, artifact)
    recovered = torch.load(artifact, map_location="cpu", weights_only=True)
    for key in names + ("optimizer_states",):
        if recovered[key] != evidence[key]:
            raise ValueError(f"Extraction roundtrip changed {key}")
    if set(recovered["state_dict"]) != set(original["state_dict"]):
        raise ValueError("Extraction roundtrip lost native model state entries")
    for key, actual in recovered["state_dict"].items():
        expected = original["state_dict"][key]
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise ValueError(f"Extraction roundtrip changed shape or dtype for {key}")
        if actual.contiguous().numpy().tobytes() != expected.contiguous().numpy().tobytes():
            raise ValueError(f"Extraction roundtrip changed tensor bits for {key}")
    manifest = {
        "kind": "lossless model state extraction from current-run native checkpoint",
        "mode": gate["mode"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "artifact": artifact.name,
        "artifact_sha256": digest(artifact),
        "artifact_bytes": artifact.stat().st_size,
        "metadata": {name: evidence[name] for name in names},
        "native_model_state": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype), "elements": value.numel()}
            for key, value in evidence["state_dict"].items()
        },
        "saved_native_top_level_keys": sorted(evidence),
        "omitted_native_top_level_keys": sorted(set(original) - set(evidence)),
    }
    (output / "checkpoint_extraction.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(extract(args.upstream, args.output), indent=2, sort_keys=True))
