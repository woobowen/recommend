"""Bounded final-preparation checks, using only the existing preflight adapter."""

import argparse
from contextlib import contextmanager, nullcontext
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .identity import atomic_json


ROOT = Path("/data/bwwu/lc-rec/beauty-history-attention-final-20261127")
ADAPTER = Path("/data/bwwu/lc-rec/beauty-history-attention-20261127/train-r1/training/preflight_adapter")
OLD_GPU_LOCK = ADAPTER.parents[2] / "gpu-1.lock"
PACKAGE = "lc_rec.beauty_history_attention"
STAGES = ("contract", "old-causal", "b-mask", "generate-stop", "generate-resume", "metrics")


def resource_record():
    queries = {
        "gpu": ["--query-gpu=index,uuid,name,memory.total,memory.used", "--format=csv,noheader,nounits"],
        "compute": ["--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"],
    }
    values = {name: subprocess.check_output(["nvidia-smi", *query], text=True) for name, query in queries.items()}
    disk = os.statvfs(ROOT)
    return {**values, "visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "disk_free_bytes": disk.f_bavail * disk.f_frsize}


@contextmanager
def physical_gpu():
    uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    before = resource_record()
    if not uuid.startswith("GPU-") or "," in uuid:
        raise RuntimeError("select one physical GPU by the observed UUID")
    cards = [list(map(str.strip, line.split(","))) for line in before["gpu"].splitlines()]
    card, = [g for g in cards if g[1] == uuid and "4090" in g[2]]
    lock = OLD_GPU_LOCK if card[0] == "1" else ROOT / f"{uuid}.lock"
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = resource_record()
        if any(line.split(",")[0].strip() == uuid for line in before["compute"].splitlines()):
            raise RuntimeError("the selected GPU already has a compute process")
        yield before


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if sys.flags.optimize or output.parent != ROOT:
        raise ValueError("run in the isolated final-preflight output root with assertions enabled")
    output.mkdir(exist_ok=False)
    gpu = args.stage not in ("contract", "metrics")
    if not gpu and os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU phase requires CUDA explicitly hidden")
    pilot = ROOT / "pilot"
    if args.stage in ("contract", "old-causal", "b-mask"):
        module = "inference_probe"
        parameters = ["--kind", args.stage, "--output", str(output / "result.json")]
        if args.stage == "b-mask":
            parameters += ["--adapter", str(ADAPTER)]
    elif args.stage in ("generate-stop", "generate-resume"):
        module = "evaluate"
        parameters = ["generate", "--adapter", str(ADAPTER), "--run-commit", args.commit,
                      "--output", str(pilot), "--purpose", "preflight", "--max-seconds", "550",
                      "--backend", "eager", "--sample-budget", "20", "--chunk-users", "8"]
        if args.stage == "generate-stop":
            parameters += ["--stop-after-chunks", "1"]
    else:
        module = "evaluate"
        parameters = ["metrics", "--predictions", str(pilot), "--output", str(output / "metrics"),
                      "--purpose", "preflight", "--evaluation-commit", args.commit]
    command = [sys.executable, "-B", "-m", f"{PACKAGE}.{module}", *parameters]
    start = time.monotonic()
    record = {"stage": args.stage, "run_commit": args.commit, "argv": command,
              "cwd": str(Path.cwd()), "runner_job_id": os.environ.get("GRID_JOB_ID")}
    try:
        with physical_gpu() if gpu else nullcontext() as before:
            record["resources_before"] = before
            atomic_json(output / "execution.json", record)
            result = subprocess.run(command, check=False)
            record["returncode"] = result.returncode
            if gpu:
                record["resources_after"] = resource_record()
            result.check_returncode()
    finally:
        record["wall_seconds"] = time.monotonic() - start
        atomic_json(output / "execution.json", record)


if __name__ == "__main__":
    main()
