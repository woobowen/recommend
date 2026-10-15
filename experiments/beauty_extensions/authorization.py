"""Fail-closed identity gate for a future, separately authorized A/B run.

The October 15 preflight creates no FORMAL_RUN_AUTHORIZED gate. This validator
does no model loading, generation, or evaluation.
"""
import hashlib
import json
from pathlib import Path

from .common import REPO, sha256, verify_asset


def canonical_plan(spec):
    return {k: v for k,v in spec.items() if k not in ("nonauthor_gate", "formal_approved")}


def require_formal_gate(spec):
    if spec.get("formal_approved") is not True:
        raise ValueError("FORMAL_NOT_AUTHORIZED: explicit new user authorization required")
    record = spec.get("nonauthor_gate")
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError("Formal gate must be a hash-bound report file, not a boolean")
    verify_asset(record)
    gate = json.loads(Path(record["path"]).read_text())
    if gate.get("kind") != "FORMAL_RUN_AUTHORIZED" or not gate.get("explicit_user_approval_reference"):
        raise ValueError("Preflight review is not formal user authorization")
    digest = hashlib.sha256(json.dumps(canonical_plan(spec), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    if gate.get("plan_sha256") != digest:
        raise ValueError("Formal plan differs from approved gate")
    deployment = json.loads((REPO / "DEPLOYMENT.json").read_text())
    records = [r for r in deployment["files"] if r["path"].endswith(".py")]
    code = {r["path"]: r["sha256"] for r in records}
    if gate.get("source_python_sha256") != code:
        raise ValueError("Formal source differs from reviewed source")
    for r in records:
        if sha256(REPO / r["path"]) != r["sha256"]:
            raise ValueError("Deployment source bytes changed")
    for name in ("config", "sid", "checkpoint"):
        verify_asset(spec[name])
        if gate.get("assets", {}).get(name) != spec[name]:
            raise ValueError("Formal asset differs from approved asset: " + name)
    review = gate.get("independent_review")
    verify_asset(review)
    report = json.loads(Path(review["path"]).read_text())
    if report.get("status") != "PREFLIGHT_GATE_PASS" or report.get("plan_sha256") != digest:
        raise ValueError("Independent review does not bind this plan")
    return gate
