"""Collect the approved fixed MLP cohort using actual-generation dual paths."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import sys
import time

from .common import REPO, batch_to, bind_source, instantiate, sha256, verify_asset, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("engineering-check", "capture"))
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    if spec["route"] != "MLP" or spec.get("shuffle_n", 0):
        raise ValueError("This entry is restricted to approved ordinary MLP attention")
    output = (args.output or Path(os.environ["GRID_JOB_DIR"]) / "output").resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "invocation.json").exists():
        raise RuntimeError("Refuse to reuse an output identity")
    started = time.monotonic()
    cohort_identity = verify_asset(spec["attention_cohort"])
    cohort = json.loads(Path(cohort_identity["path"]).read_text())
    assert cohort["selected_users"] == 32 and len(set(cohort["sample_keys"])) == 32
    assert cohort["sid"]["sha256"] == spec["sid"]["sha256"]
    assert not cohort["model_outputs_inspected"] and not cohort["raw_user_ids_exported"]
    training_identity = training_status_identity = None
    if args.phase == "capture":
        training_identity = verify_asset(spec["training_result"])
        training = json.loads(Path(training_identity["path"]).read_text())
        training_status_identity = verify_asset(spec["training_job_status"])
        training_status = json.loads(Path(training_status_identity["path"]).read_text())
        assert training_status["status"] == "SUCCEEDED" and training_status["exit_code"] == 0
        assert training_status["cleanup"]["verified"]
        assert training["status"] == "COMPLETE" and training["formal"] and training["training_complete"] and training["test_complete"]
        assert training["best_checkpoint"]["sha256"] == spec["checkpoint"]["sha256"]
        assert training["sid_asset"]["sha256"] == spec["sid"]["sha256"]
        assert training["approved_config_asset"]["sha256"] == spec["config"]["sha256"]
    write_json(output / "invocation.json", {"argv": sys.argv, "interpreter": sys.executable,
        "cwd_at_entry": os.getcwd(), "spec": spec, "spec_sha256": sha256(args.spec),
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "code_identity": json.loads((REPO / "DEPLOYMENT.json").read_text()),
        "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"), "cohort": cohort_identity,
        "training_result": training_identity, "training_job_status": training_status_identity})
    bind_source("MLP")
    cfg, model, _ = instantiate(spec, output, checkpoint=True)
    import torch
    from src.data.loading.components.interfaces import SequentialModelInputData, SequentialModuleLabelData
    from .observation import capture_batch
    model.to("cuda:0").eval()
    count = 8 if args.phase == "engineering-check" else 32
    records = []
    for start in range(0, count, 8):
        end = min(count, start + 8)
        keys = cohort["sample_keys"][start:end]
        inputs = SequentialModelInputData(user_id_list=keys,
            transformed_sequences={"sequence_data": torch.tensor(cohort["input_ids"][start:end], dtype=torch.long)},
            mask=torch.tensor(cohort["attention_mask"][start:end], dtype=torch.long))
        labels = SequentialModuleLabelData(labels={"sequence_data": torch.tensor(cohort["targets"][start:end], dtype=torch.long)})
        batch = batch_to((inputs, labels), "cuda:0")
        destination = output / f"batch-{start // 8:02d}"
        checks = capture_batch(spec, model, batch, destination, engineering=args.phase == "engineering-check")
        records.append({"sample_keys": keys, "bundle": str(destination / "bundle/record.json"),
            "bundle_sha256": sha256(destination / "bundle/record.json"), "off_on": checks})
    write_json(output / "RESULT.json", {"status": "PASS", "formal": args.phase == "capture",
        "phase": args.phase, "samples": count, "cohort": cohort_identity,
        "checkpoint": spec["checkpoint"], "batches": records,
        "main_path": "actual generation final Top-1 complete ancestry",
        "historical_comparison_path": "per-step local first parent beam",
        "display": "last decoder layer, head mean; raw per-head matrices retained",
        "scientific_similarity_threshold": None, "scientific_similarity_claim": None,
        "elapsed_seconds": time.monotonic() - started,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated()})


if __name__ == "__main__":
    main()
