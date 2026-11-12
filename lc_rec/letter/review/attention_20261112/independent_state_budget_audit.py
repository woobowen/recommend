#!/usr/bin/env python3
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse, hashlib, json, math, sys
from datetime import datetime, timedelta
from pathlib import Path
FIELDS = ("model", "optimizer", "scheduler", "rng", "parameters", "gradients")

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def quantized(bnb):
    assert isinstance(bnb, dict) and len(bnb) == 224
    result = {}
    for module, record in sorted(bnb.items()):
        result[module] = {}
        for name, dtype in (("CB", "torch.int8"), ("SCB", "torch.float32")):
            value = record["state"][name]
            if value is None:
                value = record["weight_attrs"][name]
            assert isinstance(value, dict) and value["dtype"] == dtype
            assert value["shape"] and all(isinstance(x, int) and x > 0 for x in value["shape"])
            assert len(value["sha256"]) == 64
            int(value["sha256"], 16)
            result[module][name] = {k: value[k] for k in ("shape", "dtype", "sha256", "elements")}
        assert result[module]["SCB"]["shape"] == [result[module]["CB"]["shape"][0]]
    return result

def resources(root):
    rows, cpu_jobs = [], []
    for request_path in sorted(root.glob("*/job/request.json")):
        request = json.loads(request_path.read_text())
        visible = request["env"]["CUDA_VISIBLE_DEVICES"]
        assert isinstance(visible, str)
        job = request_path.parent.parent.name
        if not visible:
            cpu_jobs.append(job)
            continue
        row = {"job_id": job, "CUDA_VISIBLE_DEVICES": visible, "errors": []}
        try:
            status_path = request_path.with_name("status.json")
            pipeline_path = request_path.parent.parent / "run" / "pipeline.json"
            status = json.loads(status_path.read_text())
            start = datetime.fromisoformat(status["started_at"].replace("Z", "+00:00"))
            end = datetime.fromisoformat(status["ended_at"].replace("Z", "+00:00"))
            assert start.utcoffset() == end.utcoffset() == timedelta(0) and end >= start
            wall = (end - start).total_seconds()
            seconds = float(json.loads(pipeline_path.read_text())["gpu_device_seconds"])
            assert math.isfinite(seconds) and seconds >= 0
            if status["status"] not in ("SUCCEEDED", "FAILED", "TIMED_OUT"):
                row["errors"].append("terminal status requires explicit runner-source confirmation")
            if status["cleanup"]["verified"] is not True:
                row["errors"].append("cleanup not verified")
            if status["status"] == "SUCCEEDED" and status["exit_code"] != 0:
                row["errors"].append("SUCCEEDED with nonzero exit_code")
            row.update(status=status["status"], exit_code=status["exit_code"], outer_wall_seconds=wall,
                pipeline_gpu_device_seconds=seconds, charged_device_seconds=max(2 * wall, seconds),
                wall_within_1800_seconds=wall <= 1800,
                receipt_sha256={str(p): file_hash(p) for p in (request_path, status_path, pipeline_path)})
        except Exception as error:
            row["errors"].append(f"{type(error).__name__}: {error}")
        rows.append(row)
    complete = bool(rows) and all(not row["errors"] for row in rows)
    known = sum(row.get("charged_device_seconds", 0) for row in rows)
    return {"gpu_jobs": rows, "cpu_job_ids": cpu_jobs, "complete": complete,
        "known_device_minutes": known / 60, "charged_device_minutes": known / 60 if complete else None,
        "within_8_jobs": len(rows) <= 8, "within_120_device_minutes": complete and known <= 7200,
        "all_walls_within_1800_seconds": complete and all(row["wall_within_1800_seconds"] for row in rows)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pair", action="append", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    report = {"script_sha256": file_hash(__file__), "argv": sys.argv, "CUDA_VISIBLE_DEVICES": "", "pairs": []}
    for pair_text in args.pair:
        left, right = pair_text.split(",")
        assert all("/" not in job and job not in (".", "..") for job in (left, right))
        pair = {"left": left, "right": right, "ranks": {}}
        for rank in (0, 1):
            paths = [root / job / "run" / "training" / f"before_first_forward.rank{rank}.json" for job in (left, right)]
            x, y = [json.loads(path.read_text()) for path in paths]
            fields = {k: {"left": fingerprint(x[k]), "right": fingerprint(y[k]),
                          "exact": fingerprint(x[k]) == fingerprint(y[k])} for k in FIELDS}
            qx, qy = quantized(x["bnb"]), quantized(y["bnb"])
            pair["ranks"][str(rank)] = {"source_sha256": {str(p): file_hash(p) for p in paths},
                "fields": fields, "all_six_exact": all(v["exact"] for v in fields.values()),
                "canonical_quantized_modules": 224, "CB_SCB_exact": qx == qy,
                "canonical_left_sha256": fingerprint(qx), "canonical_right_sha256": fingerprint(qy),
                "raw_bnb_record_exact": fingerprint(x["bnb"]) == fingerprint(y["bnb"])}
        report["pairs"].append(pair)
    report["resources"] = resources(root)
    report["limits"] = "JSON fingerprint and receipt audit; no independent rereading of large acceptance-state tensors."
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "sha256": file_hash(output)}))

if __name__ == "__main__":
    main()
