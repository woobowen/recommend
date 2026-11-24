"""Recompute the phase conclusions from the published observation records."""
import base64
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

from summarize import VAL, records

stage = Path(sys.argv[1]).resolve()
def get(name):
    return json.loads((stage / name).read_text())

def sha(data):
    return hashlib.sha256(data).hexdigest()

request_bytes = (stage / "request.json").read_bytes()
request = json.loads(request_bytes)
canonical = json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
fingerprint = sha(canonical)
admission = get("admission.json")
assert admission["template_sha256"] == sha(request_bytes)
assert admission["canonical_request_sha256"] == fingerprint
assert (stage / "server-request.json").read_bytes() == canonical + b"\n"
assert sha((stage / "runner-deployment.json").read_bytes()) == request["deployment_sha256"]
assert admission["inputs"] and {r["path"]:r["sha256"] for r in admission["inputs"]} == {
    r["path"]:r["sha256"] for r in request["inputs"]
}
assert len(admission["inputs"]) == 113
query = get("gridctl-query.json")
assert query["status"] == "RUNNING" and query["local_request_verified"] and query["request_sha256"] == fingerprint
submission = get("submission.json")
assert submission["submission"] == "ACCEPTED" and submission["request_sha256"] == fingerprint

with tempfile.TemporaryDirectory() as temporary:
    subprocess.run([sys.executable, "-B", str(stage / "summarize.py"), str(stage / "snapshots"), temporary,
                    "--request", str(stage / "request.json")],
                   check=True)
    for name in ("timeline.md", "samples-derived.json"):
        assert (Path(temporary) / name).read_bytes() == (stage / name).read_bytes()

first = get("snapshots/T00.json")
start = get("snapshots/start-reconnect.json")
last = get("snapshots/T30.json")
delivery = get("delivery-status.json")
for point in (first, start, last, delivery):
    status = point["status"]
    assert status["job_id"] == request["job_id"] and status["request_sha256"] == fingerprint
    assert point["boot_id"] == first["boot_id"]
    assert status["worker"] == first["status"]["worker"] and status["supervisor"] == first["status"]["supervisor"]
    assert status["worker_alive"] and status["supervisor_alive"]
    assert len(status["workgroup_observation"]["owned_live"]) == 1
    worker, supervisor = [point["process"][key] for key in ("worker", "supervisor")]
    assert supervisor["ppid"] == 1 and worker["ppid"] == supervisor["pid"]
    for process, logfile in ((worker, "work.log"), (supervisor, "supervisor.log")):
        assert process["pid"] == process["sid"] == process["pgid"] and process["tty_number"] == 0
        fds = process["fd_and_paths"]
        assert fds["fd/0"] == "/dev/null" and fds["fd/1"] == fds["fd/2"]
        assert fds["fd/1"].endswith("/" + request["job_id"] + "/" + logfile)
    assert worker["fd_and_paths"]["cwd"] == request["cwd"]
    assert worker["initial_exec_environment"]["CUDA_VISIBLE_DEVICES"] == request["env"]["CUDA_VISIBLE_DEVICES"]
    assert point["queued_formal_requests"] == ["requests/" + request["job_id"] + ".json"]
    assert point["accepted_formal_requests"] == ["jobs/" + request["job_id"] + "/accepted.json"]

proof = get("captured/capture-manifest.json")
for file in proof["files"]:
    name = Path(file["file"]).relative_to("captured").as_posix()
    data = (stage / file["file"]).read_bytes()
    original = last["files"][name]["prefix"]
    assert file["range"] == [0, len(data)] and len(data) == file["prefix_bytes"]
    assert data == base64.b64decode(original["base64"], validate=True)
    assert sha(data) == file["sha256"]
captured = stage / "captured/output/run"
for name in ("effective_config.yaml", "overrides.json"):
    assert sha((captured / name).read_bytes()) == admission["plan_sha256"][name]
startup = get("captured/output/run/startup.json")
assert startup["argv"] == request["argv"][2:] and startup["cwd"] == request["cwd"]
assert startup["python"] == request["argv"][0]
gpu = startup["gpu"]
assert gpu["index"] == admission["selected_gpu"][0] and gpu["uuid"] == request["env"]["CUDA_VISIBLE_DEVICES"]
assert startup["asset_coverage"]["rows_exact"] == dict(training=24772, evaluation=24772, testing=24772)
assert startup["asset_coverage"]["items"] == 9922
model = get("captured/output/run/model.json")
assert model["parameters"] == model["trainable_parameters"] == 13128064
assert model["shared"] and len(model["heads"]) == 4 and model["sid_matches_official"]
assert model["beam"] == 10 and not model["prefix_filter"]

events = records((captured / "events.jsonl").read_text())
train_start = events[0]
assert train_start["event"] == "train_start" and train_start["step"] == train_start["optimizer_state_entries"] == 0
assert train_start["adam"]["lr"] == .001 and train_start["adam"]["weight_decay"] == .0001
train_batches = [e for e in events if e["event"] == "train_batch"]
assert len(train_batches) > 2 and all(e["step"] == e["microbatch"] // 16 and e["shuffle_calls"] == 0
                                      for e in train_batches)
assert all(a["step"] < b["step"] and a["time"] < b["time"] for a, b in zip(train_batches, train_batches[1:]))
starts = [e for e in events if e["event"] == "validation_start"]
completed = [e for e in events if e["event"] == "validation_end"]
assert completed and all(e["batches"] == 3097 and math.isfinite(e["monitor"]) for e in completed)
assert [e["step"] for e in starts] == list(range(100, 100 * len(starts) + 1, 100))
assert len(starts) in (len(completed), len(completed) + 1)
with (captured / "csv/version_0/metrics.csv").open(newline="") as file:
    rows = list(csv.DictReader(file))
validation_rows = [row for row in rows if row.get("val/recall@5")]
assert len(completed) == len(validation_rows)
for e, row in zip(completed, validation_rows):
    assert float(row["val/recall@5"]) == e["monitor"]
    assert all(math.isfinite(float(row[name])) for name in (*VAL, "val/loss"))
assert any(row.get("train/loss_step") for row in rows)
assert all(math.isfinite(float(row[name])) for row in rows
           for name in ("train/loss_step", "train/loss_epoch") if row.get(name))

points = get("samples-derived.json")
assert len(points) == 7 and not any(point["anomalies"] for point in points)
elapsed = points[-1]["consumed_gpu_device_seconds"]
assert elapsed >= 1800
assert abs(elapsed + points[-1]["remaining_gpu_device_seconds"] - request["timeout_seconds"]) < 1e-6
assert not any(points[-1]["outputs_present"].values())
assert not any(e["event"].startswith("test_") or e["event"] == "error" for e in events)
best = max(completed, key=lambda event: event["monitor"])
assert len(last["checkpoints"]) == 1
assert last["checkpoints"][0]["path"].endswith(f"_step={best['step']:06d}.ckpt")
assert last["checkpoints"][0]["bytes"] > 0
for label in ("T00", "T05", "T10", "T15", "T20", "T25", "T30"):
    t = get(f"snapshots/{label}.json")["observation"]
    for key in ("target_time", "measured_time"):
        assert dt.datetime.fromisoformat(t[key]["utc"]) == dt.datetime.fromisoformat(t[key]["new_york"])
print(json.dumps(dict(
    request_sha256=fingerprint, samples=len(points), elapsed_gpu_device_seconds=elapsed,
    completed_full_validations=len(completed), persisted_optimizer_step=train_batches[-1]["step"],
    best_step=best["step"], best_val_recall_at_5=best["monitor"], delivery_status=delivery["status"]["status"],
), indent=2))
