"""Synthetic beam-index/persistence/PNG checks, explicitly not a real preflight."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from .capture import build_encoder_position_map, recover_history_item_ids
from .plot import plot_bundle
from .records import finalize_paths, load_bundle, quality_diagnostics, reconstruct_parents, save_bundle


def selfcheck(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    # Construct a chain whose final winner is not the earlier local first beam.
    candidates = [
        np.array([[[10], [20]]], dtype=np.int64),
        np.array([[[20, 21], [10, 11]]], dtype=np.int64),
        np.array([[[10, 11, 12], [20, 21, 22]]], dtype=np.int64),
        np.array([[[20, 21, 22, 23], [10, 11, 12, 13]]], dtype=np.int64),
    ]
    steps = []
    previous = None
    for hierarchy, generated in enumerate(candidates):
        parents = reconstruct_parents(previous, generated)
        beams = 1 if hierarchy == 0 else 2
        # Parent 0 attends slot 0 and parent 1 attends slot 1.  Both heads
        # preserve that identity, making postselection-index errors observable.
        values = np.zeros((1, beams, 2, 6), dtype=np.float32)
        for beam in range(beams):
            values[:, beam, :, beam] = 1
        steps.append({"cross_heads": values, "parent_indices": parents, "generated_ids": generated, "scores": np.array([[0.6, 0.4]], dtype=np.float32)})
        previous = generated
    paths = finalize_paths(steps)
    np.testing.assert_array_equal(paths["final_parent_indices"], [[0, 1, 0, 1]])
    np.testing.assert_array_equal(paths["final_child_indices"], [[1, 0, 1, 0]])
    np.testing.assert_array_equal(paths["backtracked_top1_sid"], [[20, 21, 22, 23]])
    np.testing.assert_array_equal(paths["final_top1_cross"][0].argmax(axis=-1), [0, 1, 0, 1])
    np.testing.assert_array_equal(paths["local_first_cross"][0].argmax(axis=-1), [0, 0, 0, 0])
    # Separate samples cannot be mistaken for beams, including a final small batch.
    two = np.concatenate([candidates[0], candidates[0] + 100], axis=0)
    next_two = np.concatenate([candidates[1], candidates[1] + 100], axis=0)
    np.testing.assert_array_equal(reconstruct_parents(two, next_two), [[1, 0], [1, 0]])
    try:
        reconstruct_parents(np.array([[[1], [1]]]), np.array([[[1, 2], [1, 3]]]))
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous parents must not be accepted")
    sids = np.array([[1, 2, 3, 4, -1, -1, -1, -1]])
    mask = np.array([[1, 1, 1, 1, 0, 0, 0, 0]])
    items = recover_history_item_ids(sids, mask, np.array([[1, 9], [2, 9], [3, 9], [4, 0]]))
    np.testing.assert_array_equal(items, [[0, -1]])
    permutation = np.array([[[2, 0, 1, 3], [2, 0, 1, 3]]])
    positions = build_encoder_position_map(sids, mask, num_hierarchies=4, sep_enabled=True, user_enabled=True, aggregated=False, source_permutations=permutation, num_embeddings_per_hierarchy=256)
    assert len(positions[0]) == 11 and positions[0][1]["lookup_code_position"] == 2
    assert positions[0][4]["lookup_code_position"] == 3 and positions[0][5]["role"] == "sep"
    assert positions[0][1]["lookup_table_index"] == 515
    assert positions[0][6]["lookup_code_position"] == 0 and positions[0][6]["sampled_source_code_position"] == 2
    assert positions[0][6]["lookup_table_index"] == 0
    mlp_positions = build_encoder_position_map(sids, mask, num_hierarchies=4, sep_enabled=False, user_enabled=False, aggregated=True)
    assert len(mlp_positions[0]) == 2 and [entry["visible"] for entry in mlp_positions[0]] == [1, 0]
    encoder_mask = np.array([[1, 1, 0, 0, 0, 0]])
    bundle = {
        "capture": {"selected_indices": [0], "sample_keys": ["synthetic-only"]},
        "metadata": {"evidence_type": "SYNTHETIC_UNIT_CHECK_NOT_REAL_PREFLIGHT"},
        "paths": paths,
        "steps": steps,
        "encoder_mask": encoder_mask,
        "final_generated_ids": candidates[-1],
        "quality": quality_diagnostics(paths, encoder_mask),
    }
    persisted = save_bundle(bundle, output / "bundle")
    restored = load_bundle(persisted["record"])
    np.testing.assert_array_equal(restored["paths"]["final_top1_cross"], paths["final_top1_cross"])
    plotted = plot_bundle(persisted["record"], output / "plots")
    return {"evidence_type": "SYNTHETIC_UNIT_CHECK_NOT_REAL_PREFLIGHT", "status": "PASS", "checks": ["preselection_parent_chain", "two_sample_layout", "ambiguous_parent_rejection", "shuffle_source_position_map", "MLP_last_code_mask_map", "numeric_NPZ_SHA256_roundtrip", "headless_PNG_reopen"], "bundle": persisted, "plots": plotted}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    output = arguments.output or Path(tempfile.mkdtemp(prefix="beauty-attention-unit-")) / "check"
    print(json.dumps(selfcheck(output), indent=2))


if __name__ == "__main__":
    main()
