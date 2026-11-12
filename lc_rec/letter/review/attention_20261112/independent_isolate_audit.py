#!/usr/bin/env python3
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse, hashlib, importlib.util, json, math, statistics, sys
from pathlib import Path
import torch
sys.dont_write_bytecode = True
EXPECTED = "baf6d0fe48db5a3d5edf7c9eab6871ba35bd9f5dd1ebae316224e4702641202b"
KEYS = ("out", "dQ", "dK", "dV")

def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def tensor_dict(path):
    d = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
    assert isinstance(d, dict) and set(d) == set(KEYS)
    assert all(isinstance(t, torch.Tensor) and t.device.type == "cpu" for t in d.values())
    return d

def summary(values):
    assert values and all(math.isfinite(float(x)) and float(x) >= 0 for x in values)
    values = [float(x) for x in values]
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--isolate", required=True)
    p.add_argument("--audit-module", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    assert file_sha(a.audit_module) == EXPECTED
    spec = importlib.util.spec_from_file_location("independent_boundary_audit", a.audit_module)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    root = Path(a.isolate)
    records = {(rank, mode): [] for rank in (0, 1) for mode in ("default", "det")}
    report = {"script_sha256": file_sha(__file__), "comparison_module_sha256": EXPECTED,
              "argv": sys.argv, "CUDA_VISIBLE_DEVICES": "", "sources": [], "ranks": {}}
    for process in range(3):
        for rank in (0, 1):
            directory = root / f"process{process}.rank{rank}"
            result_path = directory / "result.json"
            result = json.loads(result_path.read_text())
            rows = result["rows"]
            assert len(rows) == 6
            indexed = {(row["mode"], row["repeat"]): row for row in rows}
            assert set(indexed) == {(mode, repeat) for mode in (False, True) for repeat in range(3)}
            report["sources"].append({"path": str(result_path), "sha256": file_sha(result_path),
                "metadata": {k: v for k, v in result.items() if k != "rows"}})
            for mode, flag in (("default", False), ("det", True)):
                for repeat in range(3):
                    row = indexed[(flag, repeat)]
                    path = directory / f"{mode}.repeat{repeat}.pt"
                    tensors = tensor_dict(path)
                    hashes = {key: audit.digest(tensors[key]) for key in KEYS}
                    assert hashes == row["hashes"], (path, "recorded tensor hashes differ")
                    records[(rank, mode)].append({"process": process, "repeat": repeat,
                        "path": str(path), "hashes": hashes, "telemetry": row})
    for rank in (0, 1):
        output = {}
        for mode in ("default", "det"):
            entries = records[(rank, mode)]
            assert len(entries) == 9
            reference = tensor_dict(entries[0]["path"])
            comparisons = []
            for entry in entries:
                tensors = tensor_dict(entry["path"])
                comparisons.append({"process": entry["process"], "repeat": entry["repeat"],
                    "tensors": {key: audit.diff(reference[key], tensors[key]) for key in KEYS}})
            performance = {key: summary([entry["telemetry"][key] for entry in entries])
                for key in ("backward_ms", "peak_allocated", "peak_reserved")}
            output[mode] = {"reference": entries[0]["path"], "comparisons": comparisons,
                "unique_hash_counts": {key: len({entry["hashes"][key] for entry in entries}) for key in KEYS},
                "reported_performance": performance,
                "first_repeat_kernels": [entry["telemetry"].get("cuda_kernels", [])
                    for entry in entries if entry["repeat"] == 0]}
        cross = []
        for left, right in zip(records[(rank, "default")], records[(rank, "det")]):
            x, y = tensor_dict(left["path"]), tensor_dict(right["path"])
            cross.append({"process": left["process"], "repeat": left["repeat"],
                "tensors": {key: audit.diff(x[key], y[key]) for key in KEYS}})
        output["default_vs_det_matched_repeats"] = cross
        report["ranks"][str(rank)] = output
    report["limits"] = ["CPU recomputation; no independent GPU replay",
        "Performance summarizes recorded CUDA telemetry",
        "Full-model trajectory and large acceptance state are outside this script"]
    path = Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(path), "sha256": file_sha(path)}))

if __name__ == "__main__":
    main()
