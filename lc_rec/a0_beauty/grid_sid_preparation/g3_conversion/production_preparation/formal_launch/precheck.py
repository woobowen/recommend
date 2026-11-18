"""Read-only identity and resource checks before first formal submission."""
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys

from formal_control import digest, read
from production_control import EXPERIMENT_REGISTRY, approval_documents
from production_identity import sha


def run(*argv):
    return subprocess.check_output(argv, text=True).splitlines()


plan_path = Path(sys.argv[1])
plan = read(plan_path)
root = plan_path.parent
job = Path(plan["job_root"]) / ("letter-a0-beauty-formal-" + plan["task_id"] + "-train-a0")
key = digest({
    "dataset": "Beauty",
    "assets": plan["asset_sha256"],
    "config": plan["config_sha256"],
    "model_revision": Path(plan["model"]).name,
    "seed": 42,
    "protocol": "GRID-LETTER-SHUFFLE-V1",
    "conditions": ["normal", "s3", "s4"],
    "series": plan["experiment_series"],
})
registry = EXPERIMENT_REGISTRY / (key + ".json")
uuid = run("nvidia-smi", "--id=1", "--query-gpu=uuid",
           "--format=csv,noheader")
compute = run("nvidia-smi", "--id=1", "--query-compute-apps=pid,used_memory",
              "--format=csv,noheader")
gpu = run("nvidia-smi", "--id=1",
          "--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu",
          "--format=csv,noheader")
memory = next(int(line.split()[1]) * 1024
              for line in Path("/proc/meminfo").read_text().splitlines()
              if line.startswith("MemAvailable:"))
disk = os.statvfs(root)
disk_total = disk.f_blocks * disk.f_frsize
disk_free = disk.f_bavail * disk.f_frsize
existing = {str(path): path.exists() for path in (
    registry, root / "plan.identity.json", root / "state.json",
    root / "requests", root / "attempts", root / "training", job,
)}
scoped_processes = [line.strip() for line in run("ps", "-u", str(os.getuid()),
                                               "-o", "pid=,args=")
                    if str(plan_path) in line
                    and any(name in line for name in ("production_control.py run",
                                                      "production_pipeline.py --plan"))]
identity = {
    "bytes": plan_path.stat().st_size,
    "file_sha256": sha(plan_path),
    "canonical_sha256": digest(plan),
    "input_count": len(plan["inputs"]),
    "config_sha256": sha(plan["config"]),
    "runner_sha256": sha(plan["runner"]),
    "export_sha256": sha(plan["export_manifest"]["path"]),
    "run_commit": read(plan["export_manifest"]["path"])["run_commit"],
}
bin_access = os.access(plan["python"], os.X_OK)
root_access = os.access(root, os.R_OK | os.W_OK | os.X_OK)
historical_review = approval_documents(plan, require_pass=True)["prelaunch_review"]
stage_job_matches = sorted(str(path) for path in job.parent.glob(
    "letter-a0-beauty-formal-" + plan["task_id"] + "-*"))
print(json.dumps({
    "observed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "hostname": run("hostname")[0], "uid": os.getuid(),
    "plan": str(plan_path), "plan_identity": identity,
    "gpu1_observation": gpu, "gpu1_compute_processes": compute,
    "disk_total_bytes": disk_total, "disk_available_bytes": disk_free,
    "host_memory_available_bytes": memory, "executable_interpreter": bin_access,
    "formal_root_access": root_access, "existing_formal_markers": existing,
    "existing_formal_jobs": stage_job_matches, "existing_formal_processes": scoped_processes,
    "prior_pinned_web_review_status": historical_review["status"],
}, indent=2), flush=True)
if any(existing.values()) or scoped_processes or stage_job_matches:
    raise SystemExit("Original formal identity already present. Reconcile before submitting.")
if uuid != plan["gpu_uuids"] or compute:
    raise SystemExit("Physical GPU1 resource guard failed.")
if memory < 40 * 1024**3 or disk_free < max(100 * 1024**3, disk_total // 10):
    raise SystemExit("Production host resource guard failed.")
if not bin_access or not root_access:
    raise SystemExit("Interpreter or formal root permission check failed.")
expected = {
    "bytes": 215484,
    "file_sha256": "a56076cf07d67bc088442dc35f863134081b455f551fc45d2c6bfb2bfd246407",
    "canonical_sha256": "526987fa87f0feb565a7a9f42bc58f42e3121b9d9a13cea4613e47f842c394b6",
    "input_count": 541,
    "config_sha256": plan["config_sha256"],
    "runner_sha256": plan["runner_sha256"],
    "export_sha256": plan["export_manifest"]["sha256"],
    "run_commit": "21d495b0be5b606853ec7cec68a3eaa1b58acbd7",
}
if identity != expected:
    raise SystemExit("Fixed plan or deployment identity mismatch.")
