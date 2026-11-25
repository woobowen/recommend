"""Check native fourth-digit behavior, full assets, and fail-closed SID admission on CPU."""
import argparse
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from upstream import (
    FIT_GATE, INPUT_MANIFEST, SID_GATE, SID_JSONL, SID_TENSOR,
    digest, identity_rows, validate_codes, validate_embeddings, verify_sid_bundle,
)


def reject(callback, expected):
    try:
        callback()
    except ValueError as exc:
        if expected not in str(exc):
            raise AssertionError(f"Expected to reject {expected!r}, got {exc!r}") from exc
        return str(exc)
    raise AssertionError(f"Unexpected acceptance of input expected to reject {expected!r}")


def check_copied_preview(preview, embedding, identity):
    gate = verify_sid_bundle(preview, "preview", embedding, identity)
    fit = json.loads((preview / FIT_GATE).read_text())
    source_files = [
        FIT_GATE, INPUT_MANIFEST, SID_GATE, SID_JSONL, SID_TENSOR, fit["checkpoint"],
        "catalogue/items/part-00000.tfrecord.gz", "quantizer_train/stage_result.json",
        "sid_generation/predicted_codes_by_item.pt",
    ]
    with TemporaryDirectory(prefix="grid-sid-copy-") as temp:
        copied = Path(temp)
        for filename in source_files:
            destination = copied / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(preview / filename, destination)
        if verify_sid_bundle(copied, "preview", embedding, identity) != gate:
            raise AssertionError("An exact copy of the preview artifacts changed SID admission")
        path = copied / SID_TENSOR
        sids = torch.load(path, map_location="cpu", weights_only=True)
        sids[:, [0, 1]] = sids[:, [1, 0]]
        torch.save(sids, path)
        return {
            "original_preview_sid_sha256": gate["sid_sha256"],
            "mutated_copy_sha256": digest(path),
            "original": str(preview),
            "mutation": "swap actual SID columns 0 and 1 in an isolated verified copy",
            "rejection": reject(
                lambda: verify_sid_bundle(copied, "preview", embedding, identity), "SID bundle identity mismatch",
            ),
            "formal_rejection": reject(
                lambda: verify_sid_bundle(preview, "formal", embedding, identity), "does not authorize a formal",
            ),
        }


def check(embedding, identity, preview):
    from runtime import load_entry

    load_entry()
    from src.utils.tensor_utils import deduplicate_rows_in_tensor, transpose_tensor_from_file

    rows = identity_rows(identity)
    embedded = validate_embeddings(embedding, rows)
    codes = torch.tensor([
        [3, 9, 4],
        [3, 7, 7],
        [3, 9, 4],
        [0, 0, 0],
        [3, 9, 4],
    ], dtype=torch.long)
    with TemporaryDirectory(prefix="grid-admission-") as folder:
        tensor_path = Path(folder) / "oracle.pt"
        torch.save(codes.float(), tensor_path)
        deduplicate_rows_in_tensor(str(tensor_path))
        transpose_tensor_from_file(str(tensor_path))
        canonical = torch.load(tensor_path, weights_only=True)
        if canonical[-1].tolist() != [1, 0, 2, 0, 3]:
            raise AssertionError("Native fourth-digit output disagrees with the handwritten oracle")
        counts = validate_codes(canonical, codes.T, expected_items=5)
        mismatched_rows = rows[:]
        mismatched_rows[0], mismatched_rows[1] = mismatched_rows[1], mismatched_rows[0]
        reversed_axis = Path(folder) / "wrong_axis.jsonl"
        reversed_axis.write_text("".join(json.dumps(row) + "\n" for row in mismatched_rows))
        wrong_axis = reject(lambda: identity_rows(reversed_axis), "identity row 0")
    rejected = {
        "identity_with_swapped_products": wrong_axis,
        "sid_with_swapped_prefixes": reject(
            lambda: validate_codes(canonical[:, [1, 0, 2, 3, 4]], codes.T, expected_items=5),
            "changed item axis",
        ),
    }

    colliding = torch.zeros((3, 256), dtype=torch.long)
    impossible = torch.cat((colliding, torch.arange(1, 257)[None, :]))
    rejected["fourth_level_overflow"] = reject(
        lambda: validate_codes(impossible, colliding, expected_items=256),
        "fourth-level capacity",
    )
    if preview is not None:
        rejected["actual_preview_binding"] = check_copied_preview(preview, embedding, identity)
    return {
        "full_embedding_identity_check": embedded,
        "synthetic_native_deduplication_oracle": {
            "prefixes": codes.tolist(), "fourth": canonical[-1].tolist(), "counts": counts,
        },
        "rejected": rejected,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.embedding, args.identity, args.preview), indent=2, sort_keys=True))
