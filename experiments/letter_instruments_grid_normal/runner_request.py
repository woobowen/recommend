"""Create requests for the existing P0 runner. This file only writes local plans."""
import argparse
from pathlib import Path

from data_assets import PREPARED, RAW, load, sha, write

BASE = Path("/data/bwwu/grid-letter-instruments-normal-20261124")
JOBS = Path("/home/bwwu/grid-audits/sid-extensions-p0/jobs")
PYTHON = "/data/bwwu/envs/grid-20261009-e61baf3/bin/python"
DEPLOYMENT = "e610be95a47894472c3f6b8396af8eb48fb1a998696a3a4914af2d886cc00877"
PREFIX = "grid-letter-instruments-normal-20261124"


def request(mode, job_id, timeout, capture, source, runtime, asset_hashes, gpu_uuid=None):
    assert job_id.startswith(f"{PREFIX}-{mode}-")
    assert 0 < timeout <= {"cpu": 28_800, "boundary": 300, "preflight": 10_800, "formal": 86_400}[mode]
    assets = BASE / "assets"
    if mode == "cpu":
        argv = [PYTHON, "-B", str(runtime / "data_assets.py"), "prepare", str(assets)]
    else:
        if mode == "boundary":
            assert gpu_uuid is None
        else:
            assert gpu_uuid and gpu_uuid.startswith("GPU-")
        argv = [PYTHON, "-B", str(runtime / "run_normal.py"), mode, "--assets", str(assets),
                "--output", str(JOBS / job_id / "output" / "run")]
    env = {"CUDA_VISIBLE_DEVICES": gpu_uuid or "", "PYTHONUNBUFFERED": "1",
           "TF_CPP_MIN_LOG_LEVEL": "3"}
    included = [p for p in source.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                and (mode != "cpu" or p.relative_to(source).parts[0] in
                     {"native", "historical", "data_assets.py"})]
    expected = [{"path": str(runtime / p.relative_to(source)), "sha256": sha(p)} for p in sorted(included)]
    expected.extend({"path": str(assets / p), "sha256": h} for p, h in sorted(asset_hashes.items()))
    return {"schema": 1, "host": "zwlab-01", "uid": 1005, "user": "bwwu",
            "namespace": "sid-extensions-p0/new-server", "deployment_sha256": DEPLOYMENT,
            "capture_version": capture, "job_id": job_id, "timeout_seconds": timeout,
            "argv": argv, "cwd": str(runtime), "env": env, "inputs": expected,
            "scope": {"mode": mode, "max_gpu_devices": int(mode in ("preflight", "formal")),
                      "max_train_updates": {"cpu": 0, "boundary": 2, "preflight": 100, "formal": 320000}[mode],
                      "max_train_microbatches": {"cpu": 0, "boundary": 4, "preflight": 1600, "formal": 5120000}[mode],
                      "max_validation_batches": {"cpu": 0, "boundary": 2, "preflight": 24, "formal": 3097}[mode],
                      "test_users": {"cpu": 0, "boundary": 20, "preflight": 20, "formal": 24772}[mode]}}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("cpu", "boundary", "preflight", "formal"))
    p.add_argument("--job-id", required=True)
    p.add_argument("--timeout", type=int, required=True)
    p.add_argument("--capture", required=True)
    p.add_argument("--source", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--assets", type=Path, help="Local copy with matching input bytes for GPU requests")
    p.add_argument("--gpu")
    a = p.parse_args()
    original = load(a.source / "historical/mapping_audit.json")
    hashes = {str(RAW / Path(p).name): h for p, h in original["official_sources"].items()}
    if a.mode != "cpu":
        audit = load(a.assets / PREPARED / "mapping_audit.json")
        hashes.update(audit["data_files"])
        for filename in ("mapping_audit.json", "preparation.json", "semantic_ids.pt",
                         "preflight_testing/part-00000.tfrecord.gz"):
            if a.mode == "formal" and filename.startswith("preflight_testing"):
                continue
            path = PREPARED / filename
            hashes[str(path)] = sha(a.assets / path)
    write(a.output, request(a.mode, a.job_id, a.timeout, a.capture, a.source, a.runtime, hashes, a.gpu))
