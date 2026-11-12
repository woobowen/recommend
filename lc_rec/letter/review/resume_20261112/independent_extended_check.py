#!/usr/bin/env python3
"""Independent read-only evidence audit, executed through controller bridge."""
import collections
import hashlib
import json
import math
from pathlib import Path

BASE = Path("/home/addaswsw/recommend-worktrees/lc-rec/lc_rec/letter")
ROOT = BASE / "evidence/resume_20261112"
PREFIX = "letter-resume-20261112-"
R = {"scope": "independent stdlib audit via execution bridge",
     "files": {}, "checks": {}, "unverified": [
         "Nonpersistent rotary caches were not captured.",
         "Parameter summaries do not prove alias or optimizer membership.",
         "Isolated attention case does not identify its originating layer.",
         "Final GPU budget and native-deterministic pair await separate audit."]}

def read(path, jsonl=False):
    raw = path.read_bytes()
    R["files"][str(path.relative_to(BASE))] = {
        "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    return ([json.loads(x) for x in raw.splitlines() if x.strip()]
            if jsonl else json.loads(raw))

def path(job, filename):
    return ROOT / "jobs" / (PREFIX + job) / "run/training" / filename

def diff(a, b):
    found = []
    def walk(x, y, p):
        if type(x) is not type(y):
            found.append((p, "type"))
        elif isinstance(x, dict):
            for k in sorted(set(x) | set(y)):
                if k not in x or k not in y:
                    found.append((p + "." + k, "missing"))
                else:
                    walk(x[k], y[k], p + "." + k)
        elif isinstance(x, list):
            if len(x) != len(y):
                found.append((p, "length"))
            for i, (xx, yy) in enumerate(zip(x, y)):
                walk(xx, yy, f"{p}[{i}]")
        elif isinstance(x, float) and (
                not math.isfinite(x) or not math.isfinite(y)):
            found.append((p, "nonfinite"))
        elif x != y:
            found.append((p, "value"))
    walk(a, b, "$")
    return {"difference_count": len(found), "first_paths": found[:12]}

def unknowns(x, p="$"):
    found = []
    if isinstance(x, dict):
        if "UNVERIFIED_STATE" in x:
            found.append((p, x["UNVERIFIED_STATE"]))
        for k, v in x.items():
            found.extend(unknowns(v, p + "." + k))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            found.extend(unknowns(v, f"{p}[{i}]"))
    return found

def numerical(a, b):
    assert type(a) in (int, float) and type(b) in (int, float)
    assert math.isfinite(a) and math.isfinite(b)
    delta, limit = b - a, 1e-6 + 1e-3 * abs(a)
    return {"reference": a, "candidate": b, "delta": delta,
            "threshold": limit, "pass": abs(delta) <= limit,
            "relative_percent": 100 * abs(delta) / abs(a) if a else None}

def numbered(events, select, payload):
    counts, result = collections.Counter(), {}
    for line, e in enumerate(events, 1):
        if not select(e):
            continue
        base = (e["event"], e.get("step"), e.get("micro"),
                e.get("module"), e.get("parameter"), e.get("grad_enabled"))
        key = base + (counts[base],)
        counts[base] += 1
        result[key] = (line, e, payload(e))
    return result

def aligned(aa, bb):
    only_a, only_b = set(aa) - set(bb), set(bb) - set(aa)
    different, first = 0, None
    for key, (la, ea, va) in aa.items():
        if key not in bb:
            continue
        lb, eb, vb = bb[key]
        comparison = diff(va, vb)
        if comparison["difference_count"]:
            different += 1
            if first is None:
                first = {"key": key, "reference_line": la, "candidate_line": lb,
                         "comparison": comparison, "reference": ea, "candidate": eb}
    return {"reference_count": len(aa), "candidate_count": len(bb),
            "matched": len(set(aa) & set(bb)), "different": different,
            "exact": len(set(aa) & set(bb)) - different,
            "missing_candidate": len(only_a), "extra_candidate": len(only_b),
            "first_difference_in_reference_order": first}

def scalar_checks(a, b):
    losses = [
        [e for e in es
         if e["event"] == "micro_loss" and e["step"] in (2, 3)]
        for es in (a, b)
    ]
    assert len(losses[0]) == len(losses[1]) == 8
    rows = [{"ordinal": i, **numerical(x["loss"], y["loss"])}
            for i, (x, y) in enumerate(zip(*losses), 1)]
    norms = []
    for es in (a, b):
        grouped = collections.defaultdict(list)
        for e in es:
            if "grad_norm" in e and e["step"] in (3, 4):
                grouped[e["step"]].append(e["grad_norm"])
        assert set(grouped) == {3, 4}
        assert all(len(set(v)) == 1 for v in grouped.values())
        norms.append({k: v[0] for k, v in grouped.items()})
    return {"loss": rows, "loss_exact": sum(x["delta"] == 0 for x in rows),
            "loss_failures": sum(not x["pass"] for x in rows),
            "norm": {s: numerical(norms[0][s], norms[1][s]) for s in (3, 4)}}

def identity_fields(e):
    result = {}
    def walk(x, p=""):
        if not isinstance(x, dict):
            return
        for k, v in x.items():
            key, q = k.lower(), p + "." + k
            identity = (("input" in key or "label" in key) and
                        ("sha" in key or key in ("input_ids", "labels")))
            if identity or key in ("lr", "lrs", "learning_rate"):
                result[q] = v
            elif isinstance(v, dict):
                walk(v, q)
    walk(e)
    return result

def identity_window(e):
    if e["event"] == "optimizer_step":
        return e["step"] in (3, 4)
    return e.get("step") in (2, 3)

def effective_bnb(mapping):
    result = {}
    for name, v in mapping.items():
        state, attrs = v["state"], v["weight_attrs"]
        result[name] = {
            k: state[k] if state[k] is not None else attrs[k]
            for k in ("CB", "SCB")}
    return result

try:
    for label, ja, jb in (
            ("default", "default-cont-r2", "default-resume-r2"),
            ("trace", "trace1-r3", "trace2-r3")):
        for rank in (0, 1):
            tag = f"{label}.rank{rank}."
            events = [read(path(j, f"events.rank{rank}.jsonl"), True) for j in (ja, jb)]
            R["checks"][tag + "scalars"] = scalar_checks(*events)
            pre = [read(path(j, f"before_first_forward.rank{rank}.json")) for j in (ja, jb)]
            R["checks"][tag + "pre"] = {
                k: diff(pre[0][k], pre[1][k]) for k in
                ("model", "optimizer", "scheduler", "rng", "parameters", "gradients")}
            R["checks"][tag + "effective_CB_SCB"] = diff(
                effective_bnb(pre[0]["bnb"]), effective_bnb(pre[1]["bnb"]))
            R["checks"][tag + "runtime_raw"] = diff(pre[0]["runtime"], pre[1]["runtime"])
            R["checks"][tag + "unknown_markers"] = [unknowns(x) for x in pre]
            identity = [
                numbered(
                    es,
                    lambda e: identity_window(e) and bool(identity_fields(e)),
                    identity_fields,
                )
                for es in events
            ]
            result = aligned(*identity)
            result["covered_fields"] = sorted({
                k for records in identity for _, _, values in records.values()
                for k in values})
            result["lr_values"] = [
                {"key": key, "values": {k: v for k, v in values.items()
                                      if k.rsplit(".", 1)[-1] in ("lr", "lrs", "learning_rate")}}
                for key, (_, _, values) in identity[0].items()
                if any(k.rsplit(".", 1)[-1] in ("lr", "lrs", "learning_rate")
                       for k in values)]
            result["coverage_status"] = (
                "FIELDS_PRESENT" if all(any(token in k.lower()
                    for k in result["covered_fields"]) for token in ("input", "label"))
                and result["lr_values"] else "UNVERIFIED")
            R["checks"][tag + "input_label_lr"] = result
            if label == "trace":
                for kind, fields, expected in (
                        ("trace_forward", ("input", "output"), 449),
                        ("trace_gradient_output", ("gradient",), 225),
                        ("trace_parameter_gradient", ("gradient",), 450)):
                    records = [numbered(es, lambda e, kind=kind:
                        e["event"] == kind and e["step"] == 2 and e["micro"] == 9,
                        lambda e, fields=fields: {k: e[k] for k in fields}) for es in events]
                    assert all(len(x) == expected for x in records), (tag, kind)
                    R["checks"][tag + kind] = aligned(*records)

    for rank in (0, 1):
        p = ROOT / "jobs" / (PREFIX + "attention-r2") / "run/isolate" / f"isolate.rank{rank}.json"
        data = read(p)
        groups = {}
        hashes = {}
        for mode in ("False", "True"):
            records = data["modes"][mode]
            assert len(records) == 5 and [x["repeat"] for x in records] == list(range(5))
            groups[mode] = [[x["output"], *x["gradients"]] for x in records]
            assert all(len(x) == 3 for x in groups[mode])
            hashes[mode] = [[t["sha256"] for t in row] for row in groups[mode]]
            for row in groups[mode]:
                for t in row:
                    assert len(t["sha256"]) == 64
            for record, tensors in zip(records, groups[mode]):
                for i, stats in enumerate(record["difference_vs_first"], 1):
                    if tensors[i]["sha256"] == groups[mode][0][i]["sha256"]:
                        assert stats["different"] == 0
                        assert stats["outside_original_tolerance"] == 0
                        assert stats["max_abs"] == 0
        R["checks"][f"isolate.rank{rank}"] = {
            "case_sha256": data["case_sha256"], "function": data["function"],
            "source": data["source"], "hashes_output_dq_dkv": hashes,
            "within_modes": {m: [diff(groups[m][0], row) for row in groups[m]]
                             for m in groups},
            "between_modes": [diff(a, b) for a, b in zip(groups["False"], groups["True"])]}
    R["execution_status"] = "FINISHED"
except Exception as exc:
    R["execution_status"] = "ERROR"
    R["error"] = {"type": type(exc).__name__, "message": str(exc)}
print(json.dumps(R, ensure_ascii=False, indent=2, sort_keys=True))
