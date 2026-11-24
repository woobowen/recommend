"""Compare the read-only server admission to the frozen Git and request bytes."""
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

stage = Path(__file__).resolve().parent
repo = Path(sys.argv[1]).resolve()
experiment = "experiments/letter_instruments_grid_normal"
review = "098677ab130eae4dfb14bae5f2c56922817d8183"
runtime = "390df9f508b7dc11eaa57b79a5a91fb5a7f99385"
plan_relative = experiment + "/evidence/formal-plan/"
def git(*args):
    return subprocess.check_output(["git", "-C", str(repo), *args])

assert git("rev-parse", "HEAD").decode().strip() == review
assert not git("status", "--porcelain")
branch = git("ls-remote", "origin", "refs/heads/wbw-grid").decode().strip()
assert branch.split() == [review, "refs/heads/wbw-grid"]
template = git("show", review + ":" + plan_relative + "request.NOT_SUBMITTED.json")
request = json.loads(template)
assert request["capture_version"] == runtime and request["argv"][3] == "formal"
assert request["timeout_seconds"] == 86400
assert request["scope"] == dict(mode="formal", max_gpu_devices=1, max_train_updates=320000,
    max_train_microbatches=5120000, max_validation_batches=3097, test_users=24772)
runtime_entries = []
prefix = request["cwd"] + "/"
for entry in request["inputs"]:
    if entry["path"].startswith(prefix):
        name = entry["path"].removeprefix(prefix)
        assert hashlib.sha256(git("show", runtime + ":" + experiment + "/" + name)).hexdigest() == entry["sha256"], name
        runtime_entries.append(name)
assert len(runtime_entries) == 104 and len(request["inputs"]) == 113
downloads = Path("/mnt/c/Users/lenovo/Downloads")
free = {str(p): shutil.disk_usage(p).free for p in (stage, downloads)}
assert min(free.values()) > 2 * 1024**3
command = shlex.join(["/usr/bin/python3", "-I", "-S", "-B", "-c", (stage / "admission.py").read_text()])
ssh = ["/mnt/c/Windows/System32/OpenSSH/ssh.exe", "-F", "C:/Users/lenovo/.ssh/config",
       "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=12",
       "-o", "ClearAllForwardings=yes", "-T", "new-server", command]
begin = time.time()
remote = subprocess.run(ssh, input=template, capture_output=True, timeout=45)
(stage / "admission.stdout").write_bytes(remote.stdout)
(stage / "admission.stderr").write_bytes(remote.stderr)
remote.check_returncode()
record = json.loads(remote.stdout)
assert record["runner_sha256"] == hashlib.sha256(git("show", review + ":tools/sid-extensions-prep/runner.py")).hexdigest()
for name, digest in record["plan_sha256"].items():
    assert hashlib.sha256(git("show", review + ":" + plan_relative + name)).hexdigest() == digest
config = git("show", review + ":" + plan_relative + "effective_config.yaml").decode()
for required in ("ckpt_path: null", "seed: 42", "max_steps: 320000", "max_epochs: 10",
                 "devices: 1", "precision: 32-true", "val_check_interval: 1600",
                 "accumulate_grad_batches: 16", "deterministic: true", "batch_size_per_device: 16",
                 "save_top_k: 1", "patience: 10", "save_weights_only: false",
                 "top_k_for_generation: 10", "eval_seed: 2026"):
    assert required in config, required
assert not any(s in config for s in ("limit_train_batches", "limit_val_batches", "limit_test_batches", "preflight_testing"))
canon = json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
(stage / "request.json").write_bytes(template)
record.update(local_start_epoch=begin, local_end_epoch=time.time(), reviewed_release=review,
              remote_branch=branch, local_free_bytes=free, fixed_runtime_blobs=len(runtime_entries),
              template_sha256=hashlib.sha256(template).hexdigest(), canonical_request_sha256=hashlib.sha256(canon).hexdigest())
(stage / "admission.json").write_text(json.dumps(record, indent=2) + "\n")
print({k: record[k] for k in ("server_utc", "fixed_runtime_blobs", "selected_gpu", "free_bytes", "canonical_request_sha256")})
