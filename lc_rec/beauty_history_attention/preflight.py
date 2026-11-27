"""Run one bounded stage under the existing new-server Beauty runner."""

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys


GPU = "GPU-59cac3bd-3645-668b-79c8-839e0d878893"
ROOT = Path("/data/bwwu/lc-rec/beauty-history-attention-20261127")
DATA = Path("/home/bwwu/grid-audits/sid-extensions-p0/transfers/letter-a0-beauty-author-8d0154e28de3")
MODEL = Path("/data/bwwu/lc-rec/assets/llama-7b/4782ad278652c7c71b72204d462d6d01eaaf7549")
VENDOR = Path("lc_rec/letter/vendor/LETTER/LETTER-LC-Rec").resolve()
PACKAGE = "lc_rec.beauty_history_attention"
CONFIG = Path("lc_rec/beauty_history_attention/config_bidirectional.json").resolve()


def record_nvidia(output, name):
    queries = [
        ["--query-gpu=index,uuid,name,memory.total,memory.used", "--format=csv,noheader,nounits"],
        ["--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"],
    ]
    devices, work = [subprocess.check_output(["nvidia-smi", *query], text=True) for query in queries]
    schedulers = [tool for tool in ("squeue", "qstat", "bjobs") if __import__("shutil").which(tool)]
    output.joinpath(name + ".json").write_text(json.dumps({
        "devices_csv": devices, "compute_csv": work, "visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "matching_scheduler_cli": schedulers,
    }, indent=2) + "\n")
    return work


@contextmanager
def gpu_slot(output):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU:
        raise RuntimeError("expected physical GPU 1 UUID as the single device")
    with ROOT.joinpath("gpu-1.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        work = record_nvidia(output, "before-gpu")
        if any(line.split(",")[0].strip() == GPU for line in work.splitlines()):
            raise RuntimeError("physical GPU 1 already has another compute process")
        try:
            yield
        finally:
            record_nvidia(output, "after-gpu")


def run(module, *args, launch=(), env=None, output):
    command = [sys.executable, "-B", *launch, "-m", f"{PACKAGE}.{module}", *map(str, args)]
    output.joinpath(module + "-argv.json").write_text(json.dumps(command, indent=2) + "\n")
    subprocess.run(command, check=True, env=env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("cpu", "cuda-check", "train", "generate", "metrics", "regression"),
                        required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--stop-after-epoch", type=int, choices=(1, 3))
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    input_args = ("--source-root", VENDOR, "--data-root", DATA, "--model", MODEL)
    if args.stage == "cpu":
        run("check", *input_args, "--output", output / "cpu.json", output=output)
        return
    if args.stage == "metrics":
        assert args.predictions
        run("evaluate", "metrics", "--predictions", args.predictions, "--output", output / "metrics",
            "--evaluation-commit", args.commit, "--purpose", "preflight", output=output)
        return
    with gpu_slot(output):
        if args.stage == "cuda-check":
            run("check", *input_args, "--output", output / "cuda.json", "--device", "cuda", output=output)
        elif args.stage == "train":
            training = args.training if args.training is not None else output / "training"
            train_args = ["--output", training, "--config", CONFIG, "--run-commit", args.commit,
                          "--boundary-preflight"]
            if args.stop_after_epoch:
                train_args += ["--stop-after-epoch", args.stop_after_epoch]
            if args.resume:
                train_args += ["--resume", args.resume]
            # Avoid torchrun starting its worker in a different process group.
            # The verified runner supervises this single foreground process.
            env = os.environ | {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1",
                                "LOCAL_WORLD_SIZE": "1", "MASTER_ADDR": "127.0.0.1"}
            import socket
            with socket.socket() as port:
                port.bind(("127.0.0.1", 0))
                env["MASTER_PORT"] = str(port.getsockname()[1])
            run("train", *input_args, *train_args, env=env, output=output)
        elif args.stage == "generate":
            assert args.training
            run("evaluate", "generate", "--adapter", args.training / "preflight_adapter", "--run-commit",
                args.commit, "--output", output / "predictions", "--purpose", "preflight",
                "--max-seconds", 720, output=output)
        else:
            run("regression", "--output", output / "regression", "--profile", output=output)


if __name__ == "__main__":
    main()
