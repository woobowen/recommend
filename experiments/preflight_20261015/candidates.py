"""Read/validate unsigned next-stage plans, without running a formal handler."""
import copy
import json
from pathlib import Path

from experiments.beauty_extensions.authorization import require_formal_gate
from experiments.beauty_extensions.common import write_json


def main():
    root = Path(__file__).resolve().parent
    rows = []
    for name in ("beauty-a", "beauty-b", "toys-mlp", "sports-mlp"):
        source = root / "specs" / (name + ".json")
        smoke = json.loads(source.read_text())
        candidate = copy.deepcopy(smoke)
        for key in ("preflight_rows", "engineering_overlay", "optimizer_updates", "resume_updates", "source_failed_job"):
            candidate.pop(key, None)
        candidate.update({"candidate_only": True, "formal_approved": False,
            "formal_gate_pending_user": True, "purpose": "unsigned next-stage plan; cannot authorize or submit any formal run"})
        if smoke["route"] == "MLP":
            candidate["research_config_confirmed"] = False
            candidate["training_budget_approved"] = False
        path = root / "specs" / (name + "-candidate.json")
        write_json(path, candidate)
        try: require_formal_gate(candidate)
        except ValueError as exc:
            assert str(exc).startswith("FORMAL_NOT_AUTHORIZED")
            rejection = str(exc)
        else: raise AssertionError("Unsigned plan passed the formal gate")
        fields = sorted(k for k in set(smoke) | set(candidate) if smoke.get(k) != candidate.get(k))
        assert candidate["config"] == smoke["config"] and candidate["sid"] == smoke["sid"]
        assert candidate.get("checkpoint") == smoke.get("checkpoint")
        rows.append({"line": name, "candidate": str(path.relative_to(root.parent.parent)),
            "formal_approved": False, "dispatch": rejection, "model_loaded": False,
            "candidate_vs_smoke_fields": fields, "differences": {k: {"smoke": smoke.get(k), "candidate": candidate.get(k)} for k in fields},
            "shared_actual_core": "common.instantiate; cli.test_lifecycle (explicit bounded dataloaders in this stage)",
            "future_execution": "new authorization and hash-bound independent gate required; no active approval file emitted"})
    write_json(root.parent.parent / "manifests/grid-shuffle-mlp-preflight-20261015/candidate-checks.json",
        {"status": "PASS", "engineering_only": True, "no_formal_handler_invoked": True, "lines": rows})
    print(json.dumps({"unsigned_candidates": len(rows), "formal_dispatch_rejected": len(rows)}))


if __name__ == "__main__": main()
