"""Plot every finite logged train/validation point from a verified local CSV."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import time

from .common import REPO, sha256, verify_asset, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plot",))
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    output = (args.output or Path(os.environ["GRID_JOB_DIR"]) / "output").resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError("Refuse to reuse an output identity")
    started = time.monotonic()
    source = verify_asset(spec["csv"])
    metrics = ("train/loss_step", "train/loss_epoch", "val/loss", "val/recall@5",
               "val/recall@10", "val/ndcg@5", "val/ndcg@10")
    points = {name: [] for name in metrics}
    count = 0
    with Path(source["path"]).open(newline="") as stream:
        for row in csv.DictReader(stream):
            count += 1
            for name in metrics:
                value = row.get(name)
                if value not in (None, ""):
                    x, y = float(row["step"]), float(value)
                    if not math.isfinite(x) or not math.isfinite(y):
                        raise ValueError("Nonfinite recorded training point")
                    points[name].append([x, y])
    if not points["val/loss"]:
        raise ValueError("The verified CSV contains no validation curve")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    for name in metrics[:3]:
        if points[name]:
            x, y = zip(*points[name]); axes[0].plot(x, y, marker=".", label=name)
    for name in metrics[3:]:
        if points[name]:
            x, y = zip(*points[name]); axes[1].plot(x, y, marker=".", label=name)
    axes[0].set_ylabel("Original logged loss")
    axes[1].set_ylabel("Validation retrieval metric")
    axes[1].set_xlabel("Lightning CSV step (unchanged source values)")
    for axis in axes:
        axis.legend(); axis.grid(alpha=0.25)
    fig.suptitle(spec["title"])
    png = output / "training_validation_curves.png"
    fig.savefig(png, dpi=160); plt.close(fig)
    with Image.open(png) as image:
        image.load(); dimensions = list(image.size)
    write_json(output / "curves.json", {"csv": source, "csv_rows": count, "points": points,
        "x_axis": "raw Lightning CSV step; fit validation records use the original logging index",
        "smoothing_or_selection": None, "formal": spec["formal"]})
    write_json(output / "invocation.json", {"argv": sys.argv, "interpreter": sys.executable,
        "spec": spec, "spec_sha256": sha256(args.spec),
        "code_identity": json.loads((REPO / "DEPLOYMENT.json").read_text())})
    write_json(output / "RESULT.json", {"status": "PASS", "formal": spec["formal"],
        "csv": source, "png": {"path": str(png), "sha256": sha256(png), "dimensions": dimensions,
        "actually_reopened": True}, "elapsed_seconds": time.monotonic() - started})


if __name__ == "__main__":
    main()
