"""Non-pickle persistence and exact beam ancestry for small attention bundles.

An attention row describes a decoder *parent before selection*.  The parent of a
selected child must be followed before indexing that step's attention tensor.
The existing GRID beam routine does not return its ``replace_indices``.  Its
unique prefixes identify those indices exactly, without repeating topk or
consuming random numbers.  Ambiguous or missing prefixes are errors.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def reconstruct_parents(previous_ids: np.ndarray | None, current_ids: np.ndarray) -> np.ndarray:
    """Return local preselection parent indices, shaped [samples, child_beams]."""
    current = np.asarray(current_ids)
    if current.ndim != 3 or current.shape[-1] < 1:
        raise ValueError("current_ids must have shape [samples, beams, prefix_length]")
    if current.dtype.kind not in "iu":
        raise ValueError("beam SID values must have an integer dtype")
    if previous_ids is None:
        if current.shape[-1] != 1:
            raise ValueError("the first beam step must generate exactly one SID code")
        parents = np.zeros(current.shape[:2], dtype=np.int64)
    else:
        previous = np.asarray(previous_ids)
        if previous.ndim != 3 or previous.shape[0] != current.shape[0]:
            raise ValueError("previous and current beam sample axes differ")
        if previous.shape[-1] + 1 != current.shape[-1]:
            raise ValueError("beam prefix length must increase by exactly one")
        matches = (current[:, :, None, :-1] == previous[:, None, :, :]).all(axis=-1)
        if not (matches.sum(axis=-1) == 1).all():
            raise ValueError("beam ancestry is missing or ambiguous; cannot fabricate a parent chain")
        parents = matches.argmax(axis=-1).astype(np.int64)
    # The uniqueness is a property of the original beam algorithm, and is also
    # required to reconstruct the next step's actual parent indices.
    for sample in current:
        if np.unique(sample, axis=0).shape[0] != sample.shape[0]:
            raise ValueError("duplicate full beam prefixes make subsequent parent identity ambiguous")
    return parents


def finalize_paths(steps: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Build local-first and final-Top1 matrices from actual preselection rows."""
    if not steps:
        raise ValueError("no generation steps were observed")
    count = steps[0]["cross_heads"].shape[0]
    count_steps = len(steps)
    child = np.zeros(count, dtype=np.int64)
    sample = np.arange(count)
    final_heads = [None] * count_steps
    final_parent = np.zeros((count, count_steps), dtype=np.int64)
    final_child = np.zeros_like(final_parent)
    chosen_tokens = np.zeros_like(final_parent)
    for hierarchy in reversed(range(count_steps)):
        step = steps[hierarchy]
        parents = step["parent_indices"][sample, child]
        if np.any(parents >= step["cross_heads"].shape[1]):
            raise ValueError("a chosen parent has no recorded attention")
        final_heads[hierarchy] = step["cross_heads"][sample, parents]
        chosen_tokens[:, hierarchy] = step["generated_ids"][sample, child, -1]
        final_parent[:, hierarchy] = parents
        final_child[:, hierarchy] = child
        child = parents
    if not np.array_equal(chosen_tokens, steps[-1]["generated_ids"][:, 0]):
        raise ValueError("backtracked tokens do not equal the final Top1 SID")
    local = np.stack([step["cross_heads"][:, 0] for step in steps], axis=1)
    final = np.stack(final_heads, axis=1)
    return {
        "local_first_heads": local,
        "final_top1_heads": final,
        "local_first_cross": local.mean(axis=2),
        "final_top1_cross": final.mean(axis=2),
        "final_parent_indices": final_parent,
        "final_child_indices": final_child,
        "backtracked_top1_sid": chosen_tokens,
    }


def quality_diagnostics(paths: dict[str, np.ndarray], encoder_mask: np.ndarray) -> dict[str, Any]:
    """Report raw normalization/masked mass; impose no similarity threshold."""
    result = {}
    for name in ("local_first_heads", "final_top1_heads"):
        values = paths[name]
        if not np.isfinite(values).all():
            raise ValueError("non-finite attention values were observed")
        sums = values.sum(axis=-1)
        hidden_mass = (values * (1 - encoder_mask[:, None, None, :])).sum(axis=-1)
        result[name] = {
            "shape": list(values.shape),
            "min_value": float(values.min()),
            "max_value": float(values.max()),
            "row_head_sum_min": float(sums.min()),
            "row_head_sum_max": float(sums.max()),
            "masked_mass_min": float(hidden_mass.min()),
            "masked_mass_max": float(hidden_mass.max()),
        }
    return result


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_bundle(bundle: dict[str, Any], directory: str | Path) -> dict[str, Any]:
    """Write a new JSON + numeric NPZ bundle; existing output is never replaced."""
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=False)
    arrays: dict[str, np.ndarray] = {}

    def encode(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            if value.dtype.kind not in "biuf":
                raise ValueError("only numeric/bool arrays are allowed in attention NPZ files")
            name = f"array_{len(arrays):04d}"
            arrays[name] = value
            return {"__array__": name, "shape": list(value.shape), "dtype": str(value.dtype)}
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(key): encode(entry) for key, entry in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(entry) for entry in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError(f"unsupported bundle value: {type(value).__name__}")

    encoded = encode(bundle)
    array_path = destination / "arrays.npz"
    np.savez_compressed(array_path, **arrays)
    document = {
        "schema": "grid-beauty-cross-attention-v1",
        "arrays_file": array_path.name,
        "arrays_sha256": sha256_file(array_path),
        "bundle": encoded,
    }
    json_path = destination / "record.json"
    json_path.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    # This reads the actual files, with allow_pickle=False and the content hash.
    restored = load_bundle(json_path)
    if restored.get("capture", {}) != bundle.get("capture", {}):
        raise ValueError("attention record metadata changed during persistence")
    return {
        "record": str(json_path),
        "record_sha256": sha256_file(json_path),
        "arrays": str(array_path),
        "arrays_sha256": document["arrays_sha256"],
        "bytes": json_path.stat().st_size + array_path.stat().st_size,
    }


def load_bundle(record_path: str | Path) -> dict[str, Any]:
    """Load only a verified numeric NPZ referenced by its neighboring JSON."""
    record_path = Path(record_path)
    document = json.loads(record_path.read_text())
    if document.get("schema") != "grid-beauty-cross-attention-v1":
        raise ValueError("unknown attention record schema")
    if document.get("arrays_file") != "arrays.npz":
        raise ValueError("unexpected attention arrays path")
    array_path = record_path.parent / document["arrays_file"]
    if sha256_file(array_path) != document["arrays_sha256"]:
        raise ValueError("attention array SHA-256 mismatch")
    references: set[str] = set()
    with np.load(array_path, allow_pickle=False) as archive:
        def decode(value: Any) -> Any:
            if isinstance(value, dict) and "__array__" in value:
                name = value["__array__"]
                array = archive[name].copy()
                if list(array.shape) != value["shape"] or str(array.dtype) != value["dtype"]:
                    raise ValueError("array dimensions/dtype do not match the JSON record")
                references.add(name)
                return array
            if isinstance(value, dict):
                return {key: decode(entry) for key, entry in value.items()}
            if isinstance(value, list):
                return [decode(entry) for entry in value]
            return value
        result = decode(document["bundle"])
        if set(archive.files) != references:
            raise ValueError("unreferenced arrays are present in the attention archive")
    return result
