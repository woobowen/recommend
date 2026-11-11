"""Targeted CPU-only source and delivery-arithmetic validation."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    repository = root.parents[1]
    checks = {"status": "PASS_TARGETED_LOCAL_VALIDATION", "argv": sys.argv,
              "python": sys.version, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    scripts = list((root / "scripts").glob("*.py"))
    for path in scripts:
        ast.parse(path.read_text(), filename=str(path))
    checks["python_ast_files"] = len(scripts)
    json_paths = list(root.rglob("*.json"))
    for path in json_paths:
        json.loads(path.read_text())
    checks["json_files_parsed"] = len(json_paths)
    cost = json.loads((root / "evidence/budget.json").read_text())
    expected = "820e2f30a51c507f65d5f3068eab018868410c8d"
    assert cost["calculation_run_commit"] == expected
    relative = str((root / "scripts/estimate_budget.py").relative_to(repository))
    fixed = subprocess.check_output(["git", "show", expected + ":" + relative], cwd=repository)
    assert hashlib.sha256(fixed).hexdigest() == cost["calculation_script"]["sha256"]
    assert fixed == (root / "scripts/estimate_budget.py").read_bytes()
    checks["cost_source_bytes_match_fixed_commit"] = expected
    for row in cost["routes"].values():
        assert not row.get("calculation_error") and row["semantic_status"] == "PARTIAL_BOUNDARY"
        assert row["training"]["formal_steps"] == 6916
        for scenario in ("center", "conservative"):
            report = row[scenario]
            assert not report["missing_components"]
            assert abs(sum(report["component_seconds"].values()) / 3600 - report["conditional_total_hours"]) < 1e-10
    checks["all_eight_cost_component_sums"] = True
    summary = json.loads((root / "evidence/execution_results.json").read_text())
    assert len(summary["routes"]) == 4
    assert all(row["route_status"] == "PARTIAL" and row["compute_status"] == "PASS" for row in summary["routes"].values())
    assert sum(row["attempts"][0]["successful_global_updates_per_rank"][0] for row in summary["routes"].values()) == 14
    assert sum(row["attempts"][0]["nonzero_lr_updates_per_rank"][0] for row in summary["routes"].values()) == 10
    checks["four_actual_terminals_and_update_caps"] = True
    checks["scope"] = "Source/config/evidence arithmetic and bounded terminal checks only; no CUDA, formal training or installation"
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(checks, ensure_ascii=False))


if __name__ == "__main__":
    main()
