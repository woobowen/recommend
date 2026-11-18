"""Export and summarize the completed GRID five-seed test records."""

import csv
import itertools
import json
import math
from pathlib import Path
from statistics import mean, stdev
import sys


DATASETS = ("beauty", "toys")
SEEDS = range(42, 47)
MODES = {"shuffle": ("unshuffled", "shuffle_3", "shuffle_4"), "mlp": ("mlp",)}
METRICS = ("recall@5", "recall@10", "ndcg@5", "ndcg@10")


def load(path):
    return json.loads(path.read_bytes())


def main(root):
    rows = []
    for dataset, seed, (variant, modes) in itertools.product(
        DATASETS, SEEDS, MODES.items()
    ):
        run = root / "runs" / dataset / variant / f"seed_{seed}"
        result = load(run / "stage_result.json")
        assert (result["dataset"], result["variant"], result["train_seed"]) == (
            dataset, variant, seed
        )
        assert result["completed"] and result["train_completed"]
        assert result["trained_from_scratch"] and not result["preflight_only"]
        assert set(result["test_metrics"]) == set(modes)
        assert set(result["attention_groups"]) == set(modes)
        for mode in modes:
            test = load(run / f"test_{mode}.json")
            assert test == result["test_metrics"][mode]
            assert test["train_seed"] == seed and test["eval_seed"] == 2026
            assert test["full_test_set"] and not test["attention_export"]
            assert test["checkpoint_sha256"] == result["checkpoint_sha256"]
            assert bool(test["shuffle_calls"]) == mode.startswith("shuffle_")
            assert result["attention_groups"][mode] == {"complete": True, "users": 100}
            values = {key: test["metrics"][0][f"test/{key}"] for key in METRICS}
            assert all(math.isfinite(v) and 0 <= v <= 1 for v in values.values())
            rows.append(
                {"dataset": dataset, "seed": seed, "mode": mode, **values,
                 "checkpoint_sha256": test["checkpoint_sha256"]}
            )
    assert len(rows) == 40

    def write_csv(name, data):
        with (root / name).open("w", encoding="utf-8", newline="") as out:
            writer = csv.DictWriter(out, list(data[0]))
            writer.writeheader()
            writer.writerows(data)

    write_csv("per_seed.csv", rows)
    summary = []
    for dataset, mode in itertools.product(DATASETS, (*MODES["shuffle"], "mlp")):
        group = [r for r in rows if (r["dataset"], r["mode"]) == (dataset, mode)]
        assert [r["seed"] for r in group] == list(SEEDS)
        row = {"dataset": dataset, "mode": mode, "n": len(group)}
        for key in METRICS:
            values = [r[key] for r in group]
            row[f"{key}_mean"] = mean(values)
            row[f"{key}_sample_sd"] = stdev(values)
        summary.append(row)
    write_csv("summary.csv", summary)
    print("Verified 20 training runs, 40 full tests and 40 attention groups.")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent)
