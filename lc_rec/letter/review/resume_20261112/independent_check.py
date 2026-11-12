#!/usr/bin/env python3
"""Read-only independent audit; raw-file execution is bridged by controller."""
import collections
import hashlib
import json
import math
from pathlib import Path

ROOT = Path("/home/addaswsw/recommend-worktrees/lc-rec/lc_rec/letter/evidence/resume_20261112")
JOBS = ("letter-resume-20261112-load-r1", "letter-resume-20261112-repeat-r2")
CPU = ROOT / "jobs/letter-resume-20261112-cpu-r0/run/inspection"
R = {"scope": "independent stdlib audit through execution bridge",
     "threshold": {"atol": 1e-6, "rtol": 1e-3}, "files": {}, "checks": {}}

def read(path, jsonl=False):
    raw = path.read_bytes()
    R["files"][str(path.relative_to(ROOT))] = {
        "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    return ([json.loads(line) for line in raw.splitlines() if line.strip()]
            if jsonl else json.loads(raw))

def training(job, filename):
    return ROOT / "jobs" / job / "run/training" / filename

def diff(a, b):
    found = []
    def walk(x, y, path):
        if type(x) is not type(y):
            found.append((path, "type"))
        elif isinstance(x, dict):
            for key in sorted(set(x) | set(y)):
                if key not in x or key not in y:
                    found.append((f"{path}.{key}", "missing"))
                else:
                    walk(x[key], y[key], f"{path}.{key}")
        elif isinstance(x, list):
            if len(x) != len(y):
                found.append((path, "length"))
            for i, (xx, yy) in enumerate(zip(x, y)):
                walk(xx, yy, f"{path}[{i}]")
        elif isinstance(x, float) and (
                not math.isfinite(x) or not math.isfinite(y)):
            found.append((path, "nonfinite"))
        elif x != y:
            found.append((path, "value"))
    walk(a, b, "$")
    return {"status": "EXACT" if not found else "DIFFERENT",
            "difference_count": len(found), "first_paths": found[:12]}

def coverage(value):
    if isinstance(value, dict):
        if {"shape", "dtype", "sha256"} <= set(value):
            sha = value["sha256"]
            assert isinstance(sha, str) and len(sha) == 64
            assert all(c in "0123456789abcdef" for c in sha.lower())
            return 1
        return sum(coverage(v) for v in value.values())
    return sum(map(coverage, value)) if isinstance(value, list) else 0

def canonical_bnb(mapping):
    result = {}
    for name, value in mapping.items():
        state, attrs = value["state"], value["weight_attrs"]
        item = {k: v for k, v in value.items()
                if k not in ("state", "weight_attrs")}
        item["state_other"] = {k: v for k, v in state.items()
                               if k not in ("CB", "SCB")}
        item["weight_other"] = {k: v for k, v in attrs.items()
                                if k not in ("CB", "SCB")}
        for key in ("CB", "SCB"):
            item["effective_" + key] = (
                state[key] if state[key] is not None else attrs[key])
        result[name] = item
    return result

def numeric(a, b):
    a, b = float(a), float(b)
    assert math.isfinite(a) and math.isfinite(b)
    delta, limit = b - a, 1e-6 + 1e-3 * abs(a)
    return {"reference": a, "candidate": b, "delta": delta,
            "threshold": limit, "pass": abs(delta) <= limit,
            "relative_percent": 100 * abs(delta) / abs(a) if a else None}

def probes(events, kind):
    return [(i + 1, e) for i, e in enumerate(events)
            if e["event"] == kind and e["step"] == 2
            and e["micro"] in (9, 10, 11, 12)]

def probe_check(a, b, kind):
    aa, bb = probes(a, kind), probes(b, kind)
    assert aa and len(aa) == len(bb), (kind, len(aa), len(bb))
    field = "output" if kind == "diagnosis_forward" else "gradient"
    name = "module" if kind == "diagnosis_forward" else "parameter"
    mismatches, first = 0, None
    for (la, ea), (lb, eb) in zip(aa, bb):
        header_a = [ea["step"], ea["micro"], ea[name], ea.get("grad_enabled")]
        header_b = [eb["step"], eb["micro"], eb[name], eb.get("grad_enabled")]
        assert header_a == header_b, (header_a, header_b)
        comparison = diff(ea[field], eb[field])
        if comparison["difference_count"]:
            mismatches += 1
            if first is None:
                first = {"reference_line": la, "candidate_line": lb,
                         "comparison": comparison, "reference": ea, "candidate": eb}
    return {"pairs": len(aa), "exact": len(aa) - mismatches,
            "different": mismatches, "first_difference": first}

def loss_check(a, b):
    aa = [(i + 1, e) for i, e in enumerate(a) if e["event"] == "micro_loss"]
    bb = [(i + 1, e) for i, e in enumerate(b) if e["event"] == "micro_loss"]
    assert len(aa) == len(bb) == 8, (len(aa), len(bb))
    rows, first = [], None
    for ordinal, ((la, ea), (lb, eb)) in enumerate(zip(aa, bb), 1):
        row = {"ordinal": ordinal, **numeric(ea["loss"], eb["loss"])}
        rows.append(row)
        if not row["pass"] and first is None:
            first = {"reference_line": la, "candidate_line": lb,
                     "reference": ea, "candidate": eb}
    return {"pairs": len(rows), "exact": sum(v["delta"] == 0 for v in rows),
            "threshold_failures": sum(not v["pass"] for v in rows),
            "rows": rows, "first_failed_raw_event": first}

def norm_values(events):
    grouped = collections.defaultdict(list)
    for e in events:
        if "grad_norm" in e and e["step"] in (3, 4):
            grouped[e["step"]].append(e["grad_norm"])
    assert set(grouped) == {3, 4}, dict(grouped)
    assert all(len(set(values)) == 1 for values in grouped.values())
    return {step: values[0] for step, values in grouped.items()}

try:
    s1_model = read(CPU / "mp_rank_00_model_states.pt.json")["module"]
    for rank in (0, 1):
        prefix = f"rank{rank}."
        s1_opt = read(CPU / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt.json")
        s2 = read(training(JOBS[0], f"S2-train-begin.rank{rank}.json"))
        for label, source, loaded in (
                ("model", s1_model, s2["model"]),
                ("optimizer", s1_opt["optimizer_state_dict"], s2["optimizer"])):
            R["checks"][prefix + "S1_to_S2." + label] = {
                **diff(source, loaded), "source_tensors": coverage(source),
                "loaded_tensors": coverage(loaded)}
        pre = [read(training(j, f"before_first_forward.rank{rank}.json")) for j in JOBS]
        for field in ("model", "optimizer", "scheduler", "rng", "parameters", "gradients"):
            R["checks"][prefix + "pre." + field] = diff(pre[0][field], pre[1][field])
        R["checks"][prefix + "pre.canonical_bnb"] = diff(
            canonical_bnb(pre[0]["bnb"]), canonical_bnb(pre[1]["bnb"]))
        events = [read(training(j, f"events.rank{rank}.jsonl"), True) for j in JOBS]
        for kind in ("diagnosis_forward", "diagnosis_parameter_gradient"):
            R["checks"][prefix + kind] = probe_check(*events, kind)
        R["checks"][prefix + "loss"] = loss_check(*events)
        if rank == 0:
            an, bn = map(norm_values, events)
            R["checks"]["grad_norm"] = {s: numeric(an[s], bn[s]) for s in (3, 4)}
    R["execution_status"] = "FINISHED"
except Exception as exc:
    R["execution_status"] = "ERROR"
    R["error"] = {"type": type(exc).__name__, "message": str(exc)}
print(json.dumps(R, ensure_ascii=False, indent=2, sort_keys=True))
