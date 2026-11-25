"""Run the scoped Instruments stages under the existing P0 runner."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
SNAPSHOT = HERE.parents[1]
PACKAGE = HERE.relative_to(SNAPSHOT)
BASE = Path("/data/bwwu/grid-letter-rerun-20261125")
INPUTS = BASE / "inputs"
PYTHON = Path("/data/bwwu/envs/grid-20261009-e61baf3/bin/python")
JOBS = Path("/home/bwwu/grid-audits/sid-extensions-p0/jobs")
PREFIX = "grid-letter-rerun-20261125"
DEPLOYMENT = "e610be95a47894472c3f6b8396af8eb48fb1a998696a3a4914af2d886cc00877"
STAGES = ("quantizer", "sid", "bound", "training", "Normal", "Shuffle3a", "Shuffle4", "summary")
REUSE_FOLDERS = {"quantizer": "upstream", "sid": "upstream", "bound": "bound", "training": "training",
                 "Normal": "evaluation/Normal", "Shuffle3a": "evaluation/Shuffle3a",
                 "Shuffle4": "evaluation/Shuffle4", "summary": "evaluation/summary.json"}


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def read(path):
    return json.loads(path.read_bytes())


def contents(path):
    if path.is_file():
        return {path.name: checksum(path)}
    return {str(file.relative_to(path)): checksum(file)
            for file in sorted(path.rglob("*")) if file.is_file() and "__pycache__" not in file.parts}


def check_identity(root, hashes, *, exact=False):
    if exact:
        assert contents(root) == hashes, root
    else:
        for file, digest in hashes.items():
            relative = Path(file)
            assert not relative.is_absolute() and ".." not in relative.parts
            path = root / relative
            assert path.is_relative_to(root) and path.is_file() and not path.is_symlink()
            assert checksum(path) == digest, path


def options(mode):
    return read(HERE / f"{mode}.json")


def build_request(args):
    source = SNAPSHOT.resolve()
    commit = (source / "VERSION").read_text().strip()
    assert len(commit) == 40 and all(c in "0123456789abcdef" for c in commit)
    scope = options(args.mode)
    assert args.job.startswith(f"{PREFIX}-{args.mode}-")
    runtime = args.runtime.resolve()
    assert runtime == BASE / f"source-{commit}"
    assert args.gpu.startswith("GPU-") and "," not in args.gpu
    input_hashes = read(args.asset_hashes)
    required = {"item_identity.jsonl", "items_qwen.jsonl", "item_embeddings.pt", "user_axis.json",
                "assets_manifest.json", "Instruments.inter.json", "Instruments.item.json", "Instruments.index.json"}
    assert required <= set(input_hashes)
    source_hashes = contents(source)
    request = {"schema": 1, "host": "zwlab-01", "uid": 1005, "user": "bwwu",
               "namespace": "sid-extensions-p0/new-server", "deployment_sha256": DEPLOYMENT,
               "capture_version": commit, "job_id": args.job, "timeout_seconds": scope["runner_timeout_seconds"],
               "argv": [str(PYTHON), "-B", str(runtime / PACKAGE / "pipeline.py"), "execute", "--mode", args.mode],
               "cwd": str(runtime), "env": {"CUDA_VISIBLE_DEVICES": args.gpu, "PYTHONUNBUFFERED": "1",
                                            "TF_CPP_MIN_LOG_LEVEL": "3"},
               "inputs": [{"path": str(runtime / file), "sha256": digest} for file, digest in source_hashes.items()]
                         + [{"path": str(INPUTS / file), "sha256": digest} for file, digest in sorted(input_hashes.items())],
               "scope": scope | {"mode": args.mode, "input_hashes": input_hashes, "source_hashes": source_hashes,
                                 "reuse_job": args.reuse_job}}
    if args.reuse_job is not None:
        assert args.reuse_job.startswith(f"{PREFIX}-{args.mode}-") and args.reuse_job != args.job
        assert Path(args.reuse_job).name == args.reuse_job
    write(args.request, request)


def command(stage, root, mode):
    upstream = root / "upstream"
    bound = root / "bound"
    training = root / "training"
    evaluation = root / "evaluation"
    if stage in ("quantizer", "sid"):
        return [sys.executable, "-B", str(HERE / "upstream.py"), "fit" if stage == "quantizer" else "sid",
                "--mode", mode, "--embedding", str(INPUTS / "item_embeddings.pt"),
                "--identity", str(INPUTS / "item_identity.jsonl"), "--output", str(upstream)]
    if stage == "sid_verify":
        return [sys.executable, "-B", str(HERE / "upstream.py"), "verify", "--mode", mode,
                "--embedding", str(INPUTS / "item_embeddings.pt"),
                "--identity", str(INPUTS / "item_identity.jsonl"), "--output", str(upstream)]
    action = {"bound": "prepare", "training": "fit", "summary": "summarize"}.get(stage, "evaluate")
    output = training if action == "fit" else evaluation if action == "summarize" else evaluation / stage
    cmd = [sys.executable, "-B", str(HERE / "downstream.py"), action, "--mode", mode,
           "--inputs", str(INPUTS), "--sid", str(upstream), "--bound", str(bound),
           "--training", str(training), "--output", str(output)]
    return cmd + (["--condition", stage] if action == "evaluate" else [])


def run(args):
    mode = args.mode
    job = Path(os.environ["GRID_JOB_DIR"]).resolve()
    assert job.parent == JOBS and job.name == os.environ["GRID_JOB_ID"]
    assert job.name.startswith(f"{PREFIX}-{mode}-") and Path.cwd() == SNAPSHOT
    assert Path(sys.executable) == PYTHON
    request = read(job / "request.json")
    scope = request["scope"]
    assert scope["mode"] == mode and all(scope[key] == val for key, val in options(mode).items())
    commit = (SNAPSHOT / "VERSION").read_text().strip()
    assert commit == request["capture_version"] and SNAPSHOT == BASE / f"source-{commit}"
    check_identity(SNAPSHOT, scope["source_hashes"], exact=True)
    check_identity(INPUTS, scope["input_hashes"])
    root = job / "output/run"
    root.mkdir(parents=True, exist_ok=False)
    receipts = root / "stages"
    receipts.mkdir()
    (root / "evaluation").mkdir()
    from run_resources import resources
    from assets import verify as verify_assets
    verify_assets(INPUTS, scope["input_hashes"]["assets_manifest.json"])
    with (BASE / "gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(root / "startup.json", {"time": datetime.now(timezone.utc).isoformat(), "mode": mode,
                                     "job": job.name, "version": commit, "cwd": str(SNAPSHOT), "argv": sys.argv,
                                     "python": sys.executable, "resources": resources(), "scope": scope})
        reuse_name = scope["reuse_job"]
        reusable = None
        if reuse_name is not None:
            assert Path(reuse_name).name == reuse_name and reuse_name.startswith(f"{PREFIX}-{mode}-")
            old_job = job.parent / reuse_name
            old_request = read(old_job / "request.json")
            old_status = read(old_job / "status.json")
            assert old_status["status"] in ("FAILED", "TIMED_OUT", "SUCCEEDED", "INTERRUPTED"), old_status["status"]
            assert old_request["scope"]["mode"] == mode and old_request["capture_version"] == commit
            for key in ("source_hashes", "input_hashes"):
                assert old_request["scope"][key] == scope[key]
            reusable = old_job / "output/run"
        for stage in STAGES:
            rec = receipts / (stage + ".json")
            artifact = root / REUSE_FOLDERS[stage]
            args_ = command(stage, root, mode)
            started = time.monotonic()
            print(f"Starting {stage} ({mode})", flush=True)
            previous = reusable / "stages" / rec.name if reusable else None
            if stage == "bound":
                subprocess.run(command("sid_verify", root, mode), check=True)
            if previous is not None and previous.is_file():
                origin = read(previous)
                assert origin["exit_code"] == 0 and origin["mode"] == mode
                source = reusable / REUSE_FOLDERS[stage]
                check_identity(source.parent if source.is_file() else source, origin["artifacts"])
                if source.is_file():
                    shutil.copyfile(source, artifact)
                else:
                    for relative in origin["artifacts"]:
                        target = artifact / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if target.exists():
                            assert checksum(target) == origin["artifacts"][relative]
                        else:
                            shutil.copyfile(source / relative, target)
                argv = origin["argv"]
            else:
                if stage == "training" and reusable and (reusable / "training").exists():
                    raise RuntimeError("Previous training lacked a complete best receipt. Preserve its attempt and inspect recovery state before starting another fit.")
                proc = subprocess.run(args_, check=False)
                if proc.returncode != 0:
                    write(receipts / (stage + ".failed.json"), {"stage": stage, "argv": args_,
                          "mode": mode, "exit_code": proc.returncode, "seconds": time.monotonic() - started})
                    raise SystemExit(proc.returncode if proc.returncode > 0 else 128 - proc.returncode)
                argv = args_
            write(rec, {"stage": stage, "mode": mode, "argv": argv, "exit_code": 0,
                        "artifacts": contents(artifact), "seconds": time.monotonic() - started,
                        "reused_from": reuse_name if previous is not None and previous.is_file() else None})
        assert read(root / "evaluation/summary.json")["mode"] == mode


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="action", required=True)
    for action in ("request", "execute"):
        command_parser = sub.add_parser(action)
        command_parser.add_argument("--mode", choices=("preview", "formal"), required=True)
    request = sub.choices["request"]
    for flag in ("asset-hashes", "request", "runtime"):
        request.add_argument("--" + flag, type=Path, required=True)
    for flag in ("job", "gpu"):
        request.add_argument("--" + flag, required=True)
    request.add_argument("--reuse-job")
    args = p.parse_args()
    (build_request if args.action == "request" else run)(args)


if __name__ == "__main__":
    main()
