#!/usr/bin/env python3
"""Budgeted CPU worker for fixed G3 Beauty completion and direct review."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import resource
import runpy
import shutil
import sys
import time

from common import (CONDITIONS, digest, facts, need, original_snapshot, pin,
                    query_original_pair, read, read_bound_release, sha, utc, write)

HERE = Path(__file__).resolve().parent


def signature(path):
    lst = path.lstat()
    target = path.stat()
    return {
        "resolved": str(path.resolve(strict=True)),
        "link_stat": [lst.st_dev, lst.st_ino, lst.st_mode, lst.st_ctime_ns],
        "file_stat": [target.st_dev, target.st_ino, target.st_size, target.st_mtime_ns, target.st_ctime_ns],
    }


def writable_handles(paths):
    """Check writable descriptors for the original inodes under this account."""
    keys = {(p.stat().st_dev, p.stat().st_ino) for p in paths}
    open_writers = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            for fd in (proc / "fd").iterdir():
                try:
                    status = fd.stat()
                    if (status.st_dev, status.st_ino) in keys:
                        flag_line = next(x for x in (proc / "fdinfo" / fd.name).read_text().splitlines()
                                         if x.startswith("flags:"))
                        flags = int(flag_line.split()[1], 8)
                        if flags & os.O_ACCMODE != os.O_RDONLY:
                            open_writers.append({"pid": int(proc.name), "fd": int(fd.name)})
                except (FileNotFoundError, ProcessLookupError):
                    continue
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return open_writers


class Timings:
    def __init__(self):
        self.rows = []

    @contextmanager
    def record(self, operation):
        started = time.monotonic()
        utc_start = utc()
        print(json.dumps({"event": "start", "step": operation, "utc": utc_start}), flush=True)
        try:
            yield
        except BaseException:
            print(json.dumps({"event": "failed", "step": operation, "seconds": time.monotonic()-started}), flush=True)
            raise
        row = {"step": operation, "started_utc": utc_start, "ended_utc": utc(), "wall_seconds": time.monotonic()-started}
        self.rows.append(row)
        print(json.dumps({"event": "end", **row}), flush=True)


def validate_frozen(cfg, plan):
    release = Path(cfg["release"])
    export = pin(plan["export_manifest"]["path"], cfg["export_manifest_sha256"])
    version = read(export)
    need(version["run_commit"] == cfg["run_commit"] and version["release"] == str(release), "frozen export identity")
    files = {}
    for row in version["files"]:
        relative = Path(row["relative"])
        # Verify the production modules, exact independent statistics source,
        # installed-vendor evaluation files and included small configs.
        if (relative.parts[:2] == ("lc_rec", "a0_beauty") and relative.suffix == ".py"
                and len(relative.parts) == 3) or str(relative) in {
            "lc_rec/letter/review/final_audit_20261113/statistics_verify.py",
            "lc_rec/letter/review/final_audit_20261113/external/official_cce.py",
            "lc_rec/a0_beauty/config/formal_g3_beauty_seed42.json",
        }:
            need(not relative.is_absolute() and ".." not in relative.parts, "unsafe export file")
            path = release / relative
            need(str(path) == row["path"], f"export path mismatch: {relative}")
            pin(path, row["sha256"], row["bytes"])
            files[str(relative)] = row
    critical = ["production_identity.py", "formal_train.py", "formal_control.py", "formal_evaluate.py",
                "g3_identity.py", "single_gpu_training.py", "paired_beauty.py", "offline_metrics.py",
                "shuffle_entry.py", "shuffle_runtime.py", "config/formal_g3_beauty_seed42.json"]
    need({f"lc_rec/a0_beauty/{name}" for name in critical} <= set(files), "missing original source pins")
    need(facts(plan["config"])["sha256"] == cfg["original_config_sha256"], "effective original configuration identity")
    need("lc_rec/letter/review/final_audit_20261113/statistics_verify.py" in files
         and "lc_rec/letter/review/final_audit_20261113/external/official_cce.py" in files, "missing statistics/evaluator pin")
    return files


def setup_inputs(cfg, plan, state, sources):
    root = Path(cfg["original_root"])
    training = root / "training"
    adapter = training / "best_adapter"
    outputs = {name: Path(entry["output"]) for name, entry in state["stages"].items()}
    paths = {root / name for name in ("state.json", "plan.json", "plan.identity.json", "controller.log")}
    paths |= {Path(plan["export_manifest"]["path"]), Path(plan["config"]), Path(cfg["original_runner"])}
    paths |= {Path(row["path"]) for row in sources.values()}
    for key in cfg["asset_sha256"]:
        path = Path(plan["data_root"]) / "Beauty" / f"Beauty.{key}.json"
        paths.add(path)
    copy_rows = {}
    receipts = {}
    for name, expected in cfg["stage_receipts_sha256"].items():
        path = outputs[name] / "stage_result.json"
        paths.add(path)
        receipts[name] = pin(path, expected)
        copy_rows[f"original/receipts/{name}.json"] = (path, expected, path.stat().st_size)
    for name in CONDITIONS:
        gen = outputs[name]
        evaluation = outputs[f"{name}_eval"]
        primary = evaluation / "primary"
        test = gen / "test"
        receipts_raw = read(receipts[f"{name}_eval"])
        files = {Path(row["path"]): row for row in receipts_raw["files"]}
        essential = {primary / n for n in ("metrics.json", "per_user.csv", "per_user_metrics.npz",
                    "bootstrap.json", "bootstrap_draw_means.npy", "groups.json", "examples.json", "unit_tests.json")}
        essential |= {evaluation / n for n in ("evaluation_config.json", "stage_a.json")}
        need(essential <= set(files), f"missing production evaluation file pins: {name}")
        for path in sorted(essential):
            row = files[path]
            relative = path.relative_to(evaluation)
            copy_rows[f"original/{name}/{relative}"] = (path, row["sha256"], row["bytes"])
            paths.add(path)
        prior = {Path(row["path"]): row for row in cfg["prior_metric_files"][name]}
        for path in (primary / "metrics.json", primary / "per_user.csv", primary / "per_user_metrics.npz"):
            need(prior[path]["sha256"] == files[path]["sha256"] and prior[path]["bytes"] == files[path]["bytes"],
                 f"current receipt differs from pinned evaluation: {path}")
        rank = prior[test / "rank0.json"]
        copy_rows[f"original/{name}/test/rank0.json"] = (test / "rank0.json", rank["sha256"], rank["bytes"])
        paths |= {test / "rank0.json", test / "predictions.rank0.jsonl"}
        pred = cfg["prediction_files"][name]
        need(Path(pred["path"]) == test / "predictions.rank0.jsonl" and (test / "predictions.rank0.jsonl").stat().st_size == pred["bytes"],
             f"prediction path/size mismatch: {name}")
    original_pair = {}
    for name, row in cfg["old_pairing"].items():
        path = outputs["paired"] / "s3" / name
        need(str(path) == row["path"], "old pairing path mismatch")
        original_pair[name] = path
        copy_rows[f"original/paired_s3/{name}"] = (path, row["sha256"], row["bytes"])
        paths.add(path)
    for step, expected in cfg["checkpoint_manifests"].items():
        cp = training / f"checkpoint-{step}"
        path = pin(cp / "complete.json", expected)
        paths.add(path)
        paths |= {cp / row["path"] for row in read(path)["files"]}
    paths |= set(adapter.iterdir())
    paths |= set((training / "checkpoint_receipts").glob("*.json"))
    paths |= {training / name for name in ("best_selection.json", "rank0.json", "events.rank0.jsonl")}
    for attempt in state["attempts"]:
        job = Path(plan["job_root"]) / attempt["job_id"]
        paths |= {Path(attempt["request_path"]), job / "request.json", job / "accepted.json", job / "status.json"}
    return outputs, paths, copy_rows, original_pair


def copy_verified(output, rows):
    inventory = {}
    for relative, (origin, expected, size) in rows.items():
        dest = output / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        with origin.open("rb") as source, dest.open("xb") as handle:
            shutil.copyfileobj(source, handle)
        pin(dest, expected, size)
        inventory[relative] = {"source": str(origin), "bytes": size, "sha256": expected}
    return inventory


def snapshot_paths(paths):
    return {str(path): signature(path) for path in sorted(paths)}


def verify_master_outputs(output, cfg, best, prediction_audits):
    for name in CONDITIONS:
        identity = prediction_audits[name]["audit"]
        need(identity["run_commit"] == cfg["run_commit"] and identity["adapter_files"] == best["adapter_files"],
             f"audit consumer binding changed: {name}")
        need((output / "independent" / f"{name}.json").is_file() and
             (output / "independent" / f"{name}.npz").is_file(), f"missing recomputation product: {name}")


def compute(cfg, stamp, output, timings):
    from predictions import audit_all
    from paired_stats import run_pairing

    with timings.record("read_original_and_pin_sources"):
        plan, state, original_charge = original_snapshot(cfg)
        original_runner = query_original_pair(cfg, plan)
        original_root = Path(cfg["original_root"])
        frozen_sources = validate_frozen(cfg, plan)
        outputs, paths, copies, original_pair = setup_inputs(cfg, plan, state, frozen_sources)
        before = snapshot_paths(paths)
        need(not writable_handles(paths), "original inputs have live writers")
        for row in frozen_sources.values():
            pin(row["path"], row["sha256"], row["bytes"])
        for key, expected in cfg["asset_sha256"].items():
            pin(Path(plan["data_root"]) / "Beauty" / f"Beauty.{key}.json", expected)
        for path in (Path(cfg["original_root"]) / "state.json", Path(cfg["original_root"]) / "controller.log"):
            copies[f"original/{path.name}"] = (path, sha(path), path.stat().st_size)
        write(output / "original-paired-runner-query.json", original_runner, exclusive=True)
        for path in original_pair.values():
            row = cfg["old_pairing"][path.name]
            pin(path, row["sha256"], row["bytes"])
    with timings.record("verify_frozen_training_once"):
        sys.path.insert(0, str(Path(cfg["release"]) / "lc_rec/a0_beauty"))
        from production_identity import verify_training
        best = verify_training(original_root / "training", cfg["run_commit"], cfg["original_config_sha256"], "FORMAL")
        need(best == read(outputs["verify_best_adapter"] / "stage_result.json")["proof"], "fresh training proof differs from pinned receipt")
        need(best["best_step"] == cfg["best_step"] and best["adapter_files"]["g3_identity.json"]["sha256"] == cfg["adapter_identity_sha256"]
             and best["adapter_files"]["adapter_model.bin"]["sha256"] == cfg["adapter_sha256"], "wrong selected adapter")
    with timings.record("independent_full_prediction_recomputation"):
        verified = audit_all(plan, outputs, output, run_commit=cfg["run_commit"], adapter_files=best["adapter_files"],
                             prediction_pins=cfg["prediction_files"])
        verify_master_outputs(output, cfg, best, verified)
    with timings.record("paired_s3_s4_and_original_s3_regression"):
        pairs = run_pairing(output, verified, original_pair["paired_comparison.json"],
                            original_pair["paired_bootstrap_difference_means.npy"], Path(cfg["release"]))
    with timings.record("collect_and_recheck_original_inputs"):
        originals = copy_verified(output, copies)
        after = snapshot_paths(paths)
        need(before == after and not writable_handles(paths), "original input path/identity changed during CPU closure")
        write(output / "input-verification.json", {
            "status": "PASS", "original_files_read_only": True, "tracked_before_and_after": before,
            "post_consume_signature_match": True, "writable_handles_before_and_after": 0,
            "full_training_verifications_this_worker": 1, "frozen_training_verification": best,
            "frozen_sources": frozen_sources, "original_copies": originals,
            "original_state_sha256": sha(original_root / "state.json"),
        }, exclusive=True)
    return {
        "status": "PASS", "scope": "SUPPLEMENTAL_CPU_CLOSURE", "execution_topology": "CPU_ONLY",
        "closure_id": cfg["closure_id"], "original_run_commit": cfg["run_commit"],
        "implementation_commit": stamp["implementation_commit"], "original_pipeline_status": "PARTIAL_PAIRED_TIMEOUT_PRESERVED",
        "original_plan_raw_sha256": cfg["plan_raw_sha256"], "original_plan_canonical_sha256": cfg["plan_canonical_sha256"],
        "original_config_sha256": cfg["original_config_sha256"], "original_stages": cfg["original_stages"],
        "original_collect": "NOT_SUBMITTED", "original_worker_wall_seconds": original_charge,
        "plan_topology_original": "SINGLE_GPU", "prediction_refs": cfg["prediction_files"],
        "input_verification": "input-verification.json",
        "conditions": {c: {"metrics": verified[c]["metrics"], "recomputation": f"independent/{c}.json",
                           "original_production": f"original/{c}/primary/metrics.json"} for c in CONDITIONS},
        "paired": pairs, "timings": timings.rows,
        "approval": "Original launch and a0 failure remain unchanged; supplemental user CPU approval governs this result.",
    }


def cpu_resources(cfg, output):
    budget = cfg["budget"]
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and os.environ.get("NVIDIA_VISIBLE_DEVICES") == "none" and
         os.environ.get("DS_ACCELERATOR") == "cpu", "GPU is not disabled")
    need(all(os.environ.get(v) == "1" for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")),
         "CPU thread environment mismatch")
    affinity = sorted(os.sched_getaffinity(0))[:budget["maximum_cpu_affinity"]]
    os.sched_setaffinity(0, affinity)
    for limit, size in ((resource.RLIMIT_AS, budget["memory_limit_bytes"]),
                        (resource.RLIMIT_FSIZE, budget["space_limit_bytes"])):
        _, hard = resource.getrlimit(limit)
        if hard != resource.RLIM_INFINITY:
            size = min(size, hard)
        resource.setrlimit(limit, (size, size))
    st = os.statvfs(output)
    need(st.f_frsize * st.f_bavail >= budget["space_limit_bytes"], "CPU free disk below planned temporary space")
    return affinity


def resources_after(output, cfg, affinity, start):
    tmp_bytes = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    need(tmp_bytes < cfg["budget"]["space_limit_bytes"], "CPU output exceeds space limit")
    cuda = [str(p) for p in Path("/proc/self/fd").iterdir() if os.path.realpath(p).startswith("/dev/nvidia")]
    torch = sys.modules.get("torch")
    initialized = bool(torch is not None and torch.cuda.is_initialized())
    need(not cuda and not initialized, "worker acquired GPU devices")
    return {
        "affinity": affinity, "address_space_bytes_limit": resource.getrlimit(resource.RLIMIT_AS)[0],
        "temporary_and_output_bytes": tmp_bytes, "rss_high_water_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "open_gpu_device_handles": cuda,
        "torch_cuda_initialized": initialized, "new_gpu_jobs": 0, "gpu_device_minutes": 0,
        "worker_internal_elapsed_seconds": time.monotonic()-start, "python": sys.executable,
    }


def main():
    started = time.monotonic()
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--kind", choices=("compute", "review"), required=True)
    args.add_argument("--output", type=Path, required=True)
    args.add_argument("--completion", type=Path)
    opts = args.parse_args()
    cfg, stamp = read_bound_release(HERE)
    job = Path(os.environ["GRID_JOB_DIR"])
    need(job == Path(cfg["cpu_job_root"]) / os.environ["GRID_JOB_ID"]
         and opts.output == job / "output" and not opts.output.is_symlink(), "worker output binding")
    request = read(job / "request.json")
    accepted = read(job / "accepted.json")
    need(digest(request) == accepted["request_sha256"] and request["capture_version"] == stamp["implementation_commit"]
         and request["closure_kind"] == opts.kind and request["closure_id"] == cfg["closure_id"],
         "worker immutable request mismatch")
    need(opts.output.is_dir() and not any(opts.output.iterdir()), "worker output must be new and empty")
    affinity = cpu_resources(cfg, opts.output)
    timings = Timings()
    if opts.kind == "compute":
        need(opts.completion is None, "compute does not take a prior result")
        result = compute(cfg, stamp, opts.output, timings)
        result["resources"] = resources_after(opts.output, cfg, affinity, started)
        write(opts.output / "completion.json", result, exclusive=True)
    else:
        need(opts.completion is not None, "direct review requires a completion directory")
        release_file = HERE / "review_raw.py"
        sys.argv = [str(release_file), "--config", str(HERE/"closure_config.json"),
                    "--completion", str(opts.completion), "--output", str(opts.output)]
        with timings.record("non_author_direct_prediction_and_statistics_review"):
            runpy.run_path(str(release_file), run_name="__main__")
        need(read(opts.output / "review_checks.json")["status"] == "PASS", "direct reviewer did not record PASS")
        write(opts.output / "execution.json", {"timings": timings.rows, "resources": resources_after(opts.output, cfg, affinity, started)},
              exclusive=True)
    manifest = {str(p.relative_to(opts.output)): {k: v for k, v in facts(p).items() if k != "path"}
                for p in sorted(opts.output.rglob("*")) if p.is_file()}
    write(opts.output / "manifest.json", {"kind": opts.kind, "implementation_commit": stamp["implementation_commit"],
          "original_run_commit": cfg["run_commit"], "files": manifest}, exclusive=True)


if __name__ == "__main__":
    main()
