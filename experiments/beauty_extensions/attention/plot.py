"""Headless figures of raw four-step cross-attention with both beam identities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .records import load_bundle, sha256_file


def plot_bundle(record_path: str | Path, output_directory: str | Path) -> dict[str, Any]:
    """Write and reread one unrescaled, untrimmed PNG per recorded fixed sample."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from PIL import Image

    bundle = load_bundle(record_path)
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=False)
    paths = bundle["paths"]
    local = paths["local_first_cross"]
    final = paths["final_top1_cross"]
    if local.shape != final.shape or local.ndim != 3 or local.shape[1] != 4:
        raise ValueError("expected [fixed_samples, four_generated_codes, encoder_positions]")
    products = []
    for sample in range(local.shape[0]):
        # One raw scale per paired figure, never independent row normalization.
        maximum = float(max(local[sample].max(), final[sample].max()))
        if not np.isfinite(maximum) or maximum <= 0:
            raise ValueError("cannot plot empty or non-finite attention values")
        figure, axes = plt.subplots(3, 1, figsize=(18, 8), gridspec_kw={"height_ratios": [4, 4, 0.6]}, constrained_layout=True)
        for axis, matrix, title in zip(
            axes[:2],
            (local[sample], final[sample]),
            ("Local first parent at each step (independent prefixes)", "Final Top-1 SID ancestor parents"),
        ):
            image = axis.imshow(matrix, aspect="auto", interpolation="nearest", cmap="YlOrRd", vmin=0, vmax=maximum)
            axis.set_title(title)
            axis.set_ylabel("Target code generated")
            axis.set_yticks(range(4), labels=[f"Code {index + 1}" for index in range(4)])
            axis.set_xticks(np.arange(0, matrix.shape[1], max(1, matrix.shape[1] // 15)))
        figure.colorbar(image, ax=list(axes[:2]), label="Raw cross-attention (last layer, mean heads)")
        axes[2].imshow(bundle["encoder_mask"][sample][None], aspect="auto", interpolation="nearest", cmap="Greys", vmin=0, vmax=1)
        axes[2].set_yticks([0], labels=["Visible mask"])
        axes[2].set_xlabel("Encoder input position (all slots retained; mapping in record.json)")
        axes[2].set_xticks(np.arange(0, local.shape[-1], max(1, local.shape[-1] // 15)))
        final_sid = bundle["final_generated_ids"][sample, 0].tolist()
        figure.suptitle(f"Fixed sample index {bundle['capture']['selected_indices'][sample]} | final Top-1 SID {final_sid}\nFour rows are generation steps; self-attention is not included")
        path = destination / f"sample_{sample:03d}_cross_attention.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        with Image.open(path) as saved:
            saved.load()
            width, height = saved.size
            pixels = np.asarray(saved.convert("RGB"))
            if width < 100 or height < 100 or pixels.max() == pixels.min():
                raise ValueError("written PNG is empty or degenerate")
        products.append({
            "file": path.name,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "width": width,
            "height": height,
            "reopened": True,
            "sample_record_index": sample,
            "scale_min": 0,
            "scale_max": maximum,
        })
    receipt = {
        "source_record": str(Path(record_path)),
        "source_record_sha256": sha256_file(record_path),
        "display": "raw head-mean; last layer and last query; all encoder slots; nearest interpolation; shared figure scale",
        "scientific_similarity_claim": None,
        "products": products,
    }
    (destination / "plot_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(plot_bundle(arguments.record, arguments.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
