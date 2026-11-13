#!/usr/bin/env python3
"""Nonauthor numerical comparison of primary formulas to separate implementation."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--independent", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing output")
    primary = load_module("primary_for_nonauthor_formula_check", args.production)
    independent = load_module("separate_nonauthor_formulas", args.independent)
    official = load_module("official_for_nonauthor_formula_check", args.official)
    source = random.Random(20261113)
    cases, comparisons = 0, 0
    for _ in range(1024):
        names = ["a", "b", "c", "d", "target", "invalid"]
        sizes = {name: source.randint(1, 4) for name in names[:-1]}
        sids = [source.choice(names) for _ in range(20)]
        mine, first = independent.ranking(sids, "target")
        theirs, primary_first = primary.ranking(sids, "target")
        if first != (primary_first or None):
            raise AssertionError("First target rank mismatch")
        for key in mine:
            if not math.isclose(mine[key], theirs[key], rel_tol=0, abs_tol=1e-14):
                raise AssertionError("Ranking mismatch: " + key)
            comparisons += 1
        mine = independent.expanded_cce(sids, "target", sizes)
        theirs = primary.cce(sids, "target", sizes)
        original = official.compute_cce_metrics(sids, "target", sizes, [5, 10, 20])
        for key in mine:
            if not (math.isclose(mine[key], theirs[key], rel_tol=0, abs_tol=1e-14)
                    and math.isclose(mine[key], original[key], rel_tol=0, abs_tol=1e-14)):
                raise AssertionError("CCE mismatch: " + key)
            comparisons += 2
        cases += 1
    result = {"status": "PASS", "reviewer": "final_metrics_review", "cases": cases,
              "numerical_comparisons": comparisons, "absolute_tolerance": 1e-14,
              "seed": 20261113, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "scope": "Synthetic formula comparison only; full prediction audit is a separate required gate",
              "sources": {str(path): {"bytes": path.stat().st_size,
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                          for path in (args.production, args.independent, args.official, Path(__file__))}}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
