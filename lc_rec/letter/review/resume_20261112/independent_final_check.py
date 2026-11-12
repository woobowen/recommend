#!/usr/bin/env python3
"""Independent final audit, read-only, through controller execution bridge."""
import ast
import datetime
import subprocess
from pathlib import Path

REPO = Path("/home/addaswsw/recommend-worktrees/lc-rec")
BASE = REPO / "lc_rec/letter"
ROOT = BASE / "evidence/resume_20261112"
PREFIX = "letter-resume-20261112-"
helper = BASE / "review/resume_20261112/independent_extended_check.py"
helper_raw = helper.read_bytes()
parsed = ast.parse(helper_raw)
definitions = [n for n in parsed.body
               if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
exec(compile(ast.Module(body=definitions, type_ignores=[]),
             str(helper), "exec"))
assert all(name in globals() for name in (
    "read", "diff", "scalar_checks", "numbered", "aligned",
    "identity_fields", "identity_window", "effective_bnb"))
R = {"scope": "independent final audit via execution bridge",
     "helper_sha256": hashlib.sha256(helper_raw).hexdigest(),
     "files": {}, "checks": {}}

def file_hashes(value):
    result = collections.defaultdict(set)
    def filename(x):
        if not isinstance(x, str):
            return None
        name = Path(x).name
        return name if name == "latest" or name.endswith(
            (".pt", ".pth", ".bin", ".json")) else None
    def digest(x):
        return (isinstance(x, str) and len(x) == 64
                and all(c in "0123456789abcdef" for c in x.lower()))
    def walk(x, context=None):
        if isinstance(x, dict):
            labels = [filename(x.get(k)) for k in
                      ("path", "file", "filename", "name", "relative_path")]
            label = next((v for v in labels if v), filename(context))
            if label:
                for v in x.values():
                    if digest(v):
                        result[label].add(v.lower())
            for k, v in x.items():
                walk(v, k)
        elif isinstance(x, list):
            for v in x:
                walk(v, context)
        elif filename(context) and digest(x):
            result[filename(context)].add(x.lower())
    walk(value)
    return result

def gitfile(commit, name):
    return subprocess.check_output(
        ["git", "-C", str(REPO), "show", f"{commit}:{name}"])

try:
    for rank in (0, 1):
        jobs = ("torchdet-cont-r3", "torchdet-resume-r3")
        events = [read(path(j, f"events.rank{rank}.jsonl"), True) for j in jobs]
        pre = [read(path(j, f"before_first_forward.rank{rank}.json")) for j in jobs]
        identity = [numbered(es, lambda e: identity_window(e)
                              and bool(identity_fields(e)), identity_fields)
                    for es in events]
        R["checks"][f"native.rank{rank}"] = {
            "scalars": scalar_checks(*events),
            "input_label_lr": aligned(*identity),
            "input_label_lr_reference": [
                {"key": key, "line": line, "values": payload}
                for key, (line, event, payload) in identity[0].items()],
            "pre": {k: diff(pre[0][k], pre[1][k]) for k in
                    ("model", "optimizer", "scheduler", "rng",
                     "parameters", "gradients")},
            "effective_CB_SCB": diff(effective_bnb(pre[0]["bnb"]),
                                      effective_bnb(pre[1]["bnb"]))}

    rows, cpu_statuses = [], []
    expected = {PREFIX + x for x in (
        "attention-r2", "default-cont-r2", "default-resume-r2", "load-r0",
        "load-r1", "repeat-r2", "torchdet-cont-r3", "torchdet-resume-r3",
        "trace1-r3", "trace2-r3")}
    for directory in sorted((ROOT / "jobs").glob(PREFIX + "*")):
        request_path = directory / "job/request.json"
        if not request_path.exists():
            continue
        request = read(request_path)
        status = read(directory / "job/status.json")
        devices = request["env"].get("CUDA_VISIBLE_DEVICES")
        if devices in ("", None):
            cpu_statuses.append({"job": directory.name, "status": status})
            continue
        assert len(set(devices.split(","))) == 2
        start = datetime.datetime.fromisoformat(
            status["started_at"].replace("Z", "+00:00"))
        end = datetime.datetime.fromisoformat(
            status["ended_at"].replace("Z", "+00:00"))
        wall = (end - start).total_seconds()
        assert 0 <= wall <= 1800, (directory.name, wall)
        pipeline_path = directory / "run/pipeline.json"
        if pipeline_path.exists():
            pipeline = read(pipeline_path)
            pipeline_charge = pipeline["gpu_device_seconds"]
            assert math.isfinite(pipeline_charge) and pipeline_charge >= 0
        else:
            assert directory.name == PREFIX + "load-r0"
            pipeline_charge = 0
        rows.append({
            "job": directory.name, "devices": devices,
            "outer_wall_seconds": wall, "outer_device_seconds": 2 * wall,
            "pipeline_device_seconds": pipeline_charge,
            "charged_device_seconds": max(2 * wall, pipeline_charge),
            "status_raw": status})
    assert len(rows) == 10
    assert {x["job"] for x in rows} == expected
    total = sum(x["charged_device_seconds"] for x in rows)
    assert total <= 7200
    R["checks"]["budget"] = {
        "rows": rows, "requests": len(rows), "device_seconds": total,
        "device_minutes": total / 60, "remaining_device_seconds": 7200 - total,
        "cpu_statuses_raw": cpu_statuses}

    receipt_path = (ROOT / "jobs" / (PREFIX + "cpu-final-receipt") /
                    "run/final_resources.json")
    receipt = read(receipt_path)
    R["checks"]["final_resources_raw"] = receipt
    inspection = (
        ROOT / "jobs" / (PREFIX + "cpu-r0") / "run/inspection"
    )
    initial_members = read(inspection / "cpu_inspection.json")["files"]
    final_members = receipt["historical_checkpoint_rehashed_unchanged"]
    assert len(initial_members) == len(final_members) == 20
    for members in (initial_members, final_members):
        assert len(members) == 20
        assert all(
            isinstance(row["sha256"], str)
            and len(row["sha256"]) == 64
            and all(c in "0123456789abcdef"
                    for c in row["sha256"].lower())
            for row in members
        )
    initial = {
        Path(row["path"]).name: {row["sha256"].lower()}
        for row in initial_members
    }
    final = {
        Path(row["path"]).name: {row["sha256"].lower()}
        for row in final_members
    }
    initial_sizes = {
        Path(row["path"]).name: row["bytes"] for row in initial_members
    }
    final_sizes = {
        Path(row["path"]).name: row["bytes"] for row in final_members
    }
    assert len(initial_sizes) == len(final_sizes) == 20
    assert initial_sizes == final_sizes
    common = sorted(set(initial) & set(final))
    assert len(common) == 20, {
        "common": common, "initial_names": sorted(initial),
        "final_names": sorted(final)}
    protection = []
    for name in common:
        assert len(initial[name]) == len(final[name]) == 1, name
        assert initial[name] == final[name], name
        protection.append({"file": name, "sha256": next(iter(initial[name]))})
    R["checks"]["original_checkpoint_protection"] = {
        "files": len(protection), "all_exact": True, "members": protection}

    old = "87ba6556ce7237f8ef32c690165449d0d6a28294"
    run = "be15ad0a7c8b3cb8f1adac2c4668e7e6973b7164"
    unchanged = ["formal_pipeline.py", "formal_runner.py", "formal_control.py",
                 "formal_ledger.py", "compat.py"]
    identities = {}
    for name in ["formal_train.py", "formal_compare.py"] + unchanged:
        filename = "lc_rec/letter/scripts/" + name
        a, b = gitfile(old, filename), gitfile(run, filename)
        identities[name] = {
            "old_sha256": hashlib.sha256(a).hexdigest(),
            "run_sha256": hashlib.sha256(b).hexdigest(), "unchanged": a == b}
        if name in unchanged:
            assert a == b, name
    config = "lc_rec/letter/config/formal_instruments_seed42.json"
    assert gitfile(old, config) == gitfile(run, config)
    identities["formal_config_sha256"] = hashlib.sha256(
        gitfile(run, config)).hexdigest()
    R["checks"]["fixed_source_identity"] = identities
    R["checks"]["fixed_source_diff_raw"] = subprocess.check_output([
        "git", "-C", str(REPO), "diff", old, run, "--",
        "lc_rec/letter/scripts/formal_train.py",
        "lc_rec/letter/scripts/formal_compare.py"]).decode()
    R["execution_status"] = "FINISHED"
except Exception as exc:
    R["execution_status"] = "ERROR"
    R["error"] = {"type": type(exc).__name__, "message": str(exc)}
print(json.dumps(R, ensure_ascii=False, indent=2, sort_keys=True))
