"""Plan, submit, recover, and observe the B-only formal Beauty series.

Compute is owned by the existing Beauty runner. The operator supplies the
separate, fixed web review and launch authorization after the preformal stage.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import stat
import subprocess
import sys
import tempfile

from .identity import ADAPTER_MEMBERS, IDENTITY_FILE, atomic_json, require, sha, verify_adapter_identity
from .train import VENDOR, load_config, verify_checkpoint

PACKAGE = "lc_rec.beauty_history_attention"
SNAPSHOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SERVER_ROOT = Path("/data/bwwu/lc-rec/beauty-history-attention-20261127")
RUNNER_ROOT = Path("/home/bwwu/grid-audits/sid-extensions-p0")
RUNNER_JOBS = RUNNER_ROOT / "jobs"
RUNNER = RUNNER_ROOT / "scripts/runner-beauty-cbb963fab68c34cc.py"
RUNNER_SHA = "cbb963fab68c34cc977653afc46f260e7ede962cb4a8f105458331a0e0e19a75"
RUNNER_DEPLOYMENT = Path(str(RUNNER) + ".deployment.json")
RUNNER_DEPLOYMENT_SHA = "e610be95a47894472c3f6b8396af8eb48fb1a998696a3a4914af2d886cc00877"
PYTHON = Path("/data/bwwu/envs/letter-20261110/bin/python")
DATA = RUNNER_ROOT / "transfers/letter-a0-beauty-author-8d0154e28de3"
MODEL = Path("/data/bwwu/lc-rec/assets/llama-7b/4782ad278652c7c71b72204d462d6d01eaaf7549")
REPLAY = Path("/data/bwwu/lc-rec/letter-shuffle-v2-beauty/cpu-replay-r3")
TRAIN_CONFIG = HERE / "config_bidirectional.json"
EVALUATION_CONFIG = SNAPSHOT / "lc_rec/shuffle_v2/a0_beauty_letter.json"
GPU_IDS = {
    "GPU-dfff0e95-1db6-e2f4-3431-55bb852fa0ae": 0,
    "GPU-59cac3bd-3645-668b-79c8-839e0d878893": 1,
}
BUDGETS = {
    "training_seconds": 28 * 3600,
    "generation_seconds": 30 * 3600,
    "metrics_seconds": 2 * 3600,
    "chain_seconds": 64 * 3600,
    "monitor_seconds": 30 * 60,
    "minimum_start_free_bytes": 72 * 1024**3,
}
CONDITIONS = ["normal", "shuffle2", "shuffle3a", "shuffle3b", "shuffle4"]
EPOCH_STEPS = (2053, 4107, 6160, 8214)
TERMINAL = {"SUCCEEDED", "FAILED", "WORKERS_TERMINATED", "TIMED_OUT", "INTERRUPTED", "SUPERVISOR_ERROR"}


def now():
    return datetime.now(timezone.utc).isoformat()


def read_record(path, expected_digest=None):
    """Return data and identity from the same regular, non-symlink file read."""
    path = Path(path)
    require(path.is_absolute() and path.parent.resolve() == path.parent, f"expected canonical record path: {path}")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), f"not a regular record: {path}")
        raw = stream.read()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_digest is not None:
        require(digest == expected_digest, f"record SHA-256 mismatch: {path}")
    value = json.loads(raw)
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value, digest


def source_files(snapshot, expected_digest):
    manifest = snapshot / "SOURCE.sha256"
    require(manifest.is_file() and not manifest.is_symlink(), "source manifest must be a regular file")
    raw = manifest.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_digest, "frozen source manifest mismatch")
    files = {}
    for line in raw.decode().splitlines():
        digest, name = line.split("  ", 1)
        relative = Path(name)
        require(re.fullmatch(r"[a-f0-9]{64}", digest) and not relative.is_absolute()
                and str(relative) == name and relative.parts[0] == "lc_rec" and ".." not in relative.parts
                and name not in files, "invalid or duplicate source manifest path")
        target = snapshot / relative
        require(target.resolve() == target and target.is_file() and sha(target) == digest,
                f"modified or missing frozen source: {target}")
        files[name] = digest
    essentials = [
        "lc_rec/beauty_history_attention/" + name for name in (
            "formal.py", "attention.py", "identity.py", "inputs.py", "train.py", "evaluate.py",
            "inference_attention.py", "prediction_store.py",
            "config_bidirectional.json", "historical_train_config.json")
    ] + [
        "lc_rec/shuffle_v2/" + name for name in (
            "a0_beauty_letter.json", "beauty.py", "beauty_evaluate.py", "core.py",
            "evaluate.py", "reference.py", "replay.py", "runtime.py", "statistics.py")
    ] + [
        "lc_rec/letter/vendor/LETTER/LETTER-LC-Rec/" + name
        for name in ("modeling_letter.py", "data.py", "collator.py", "prompt.py", "utils.py",
                     "config/ds_z2_bf16.json")
    ] + [
        "lc_rec/shuffle/core.py",
        "lc_rec/a0_beauty/checkpoint_receipt.py",
        "lc_rec/a0_beauty/g3_identity.py",
        "lc_rec/a0_beauty/single_gpu_training.py",
        "lc_rec/a0_beauty/trainer_loop_patch.py",
        "lc_rec/a0_beauty/training_rows.py",
        "lc_rec/a0_beauty/zero2_partial_state.py",
        "lc_rec/letter/scripts/compat.py",
        "lc_rec/letter/scripts/formal_bnb_reload.py",
        "lc_rec/letter/scripts/formal_evaluate.py",
        "lc_rec/letter/evidence/formal_20261111/precision_approval.json",
        "lc_rec/letter/review/final_audit_20261113/statistics_verify.py",
    ]
    require(set(essentials) <= files.keys(), "snapshot manifest omits a formal runtime dependency")
    return files


def plan_fields(commit, name, gpu, backend, manifest_sha):
    require(re.fullmatch(r"[a-f0-9]{40}", commit) and SNAPSHOT.name == commit,
            "run from the commit-named frozen deployment")
    require(re.fullmatch(r"[a-z0-9][a-z0-9-]{0,32}", name), "invalid stable B job name")
    require(gpu in GPU_IDS, "use one approved physical RTX 4090 UUID")
    require(backend in ("eager", "sdpa"), "invalid frozen B inference backend")
    return {
        "schema": 1, "experiment": "LETTER Beauty LLaMA history attention B", "mode": "bidirectional",
        "run_commit": commit, "job_name": name, "gpu_uuid": gpu, "backend": backend,
        "snapshot": str(SNAPSHOT), "snapshot_manifest_sha256": manifest_sha,
        "output_root": str(SERVER_ROOT / f"formal-b-{name}"),
        "train_config": str(TRAIN_CONFIG), "config_sha256": sha(TRAIN_CONFIG),
        "train_sha256": sha(HERE / "train.py"),
        "evaluation_config": str(EVALUATION_CONFIG), "evaluation_config_sha256": sha(EVALUATION_CONFIG),
        "replay_dir": str(REPLAY), "base_model": str(MODEL), "data_root": str(DATA),
        "python": str(PYTHON), "runner": str(RUNNER), "runner_sha256": RUNNER_SHA,
        "runner_deployment_sha256": RUNNER_DEPLOYMENT_SHA,
        "conditions": CONDITIONS, "optimizer_updates": 8214, "test_users": 22363, "budgets": BUDGETS,
    }


def validate_plan(plan_path):
    plan, digest = read_record(plan_path)
    expected = plan_fields(plan["run_commit"], plan["job_name"], plan["gpu_uuid"],
                           plan["backend"], plan["snapshot_manifest_sha256"])
    require(plan == expected and load_config(TRAIN_CONFIG)["mode"] == "bidirectional",
            "formal plan differs from fixed B conditions")
    return plan, digest


def gate_records(plan_path, authorization_path, authorization_digest):
    require(plan_path and authorization_path and re.fullmatch(r"[a-f0-9]{64}", authorization_digest or ""),
            "formal B requires the frozen plan and separate user authorization after the matching web review")
    plan, plan_digest = validate_plan(plan_path)
    auth, _ = read_record(authorization_path, authorization_digest)
    binding = {
        "plan_sha256": plan_digest,
        **{key: plan[key] for key in ("run_commit", "config_sha256", "train_sha256", "mode",
                                     "backend", "snapshot_manifest_sha256")},
    }
    require(auth.get("formal_run_authorized") is True and auth.get("external_review_status") == "PASS"
            and all(auth.get(k) == v for k, v in binding.items()),
            "user authorization does not cover the exact B plan, snapshot, backend, and training configuration")
    review_path = Path(auth["web_review_path"])
    review_sha = auth["web_review_sha256"]
    require(re.fullmatch(r"[a-f0-9]{64}", review_sha or "") and review_path != Path(authorization_path),
            "an independently stored web second-review receipt is required")
    review, _ = read_record(review_path, review_sha)
    require(review.get("review_kind") == "WEB_SECOND_REVIEW" and review.get("status") == "PASS"
            and all(review.get(k) == v for k, v in binding.items()),
            "web second review does not cover the exact B plan and source")
    files = source_files(SNAPSHOT, plan["snapshot_manifest_sha256"])
    return plan, plan_digest, files, (review_path, review_sha)


def verify_gate(plan_path, authorization_path, authorization_sha256):
    """Check the independently supplied web review and user authorization."""
    return gate_records(plan_path, authorization_path, authorization_sha256)[0]


def verify_training_launch(args, mode):
    plan, digest, _, _ = gate_records(args.formal_plan, args.formal_authorization,
                                       args.formal_authorization_sha256)
    authority, _ = read_record(args.formal_authorization, args.formal_authorization_sha256)
    require(mode == "bidirectional" and args.run_commit == plan["run_commit"]
            and Path(args.config) == TRAIN_CONFIG and Path(args.source_root) == VENDOR
            and Path(args.data_root) == DATA and Path(args.model) == MODEL,
            "formal training arguments differ from the authorized B plan")
    active = os.environ.get("GRID_JOB_ID", "")
    prefix = f"letter-beauty-v2-history-attn-formal-b-{plan['job_name']}-r"
    require(active.startswith(prefix) and re.fullmatch(r"[1-9][0-9]?", active.removeprefix(prefix)),
            "formal training requires an accepted B runner attempt")
    attempt = int(active.removeprefix(prefix))
    require(active == job_id(plan, attempt) and os.environ.get("GRID_JOB_DIR") == str(RUNNER_JOBS / active)
            and os.environ.get("CUDA_VISIBLE_DEVICES") == plan["gpu_uuid"],
            "formal training runner or physical GPU identity mismatch")
    job = RUNNER_JOBS / active
    request, _ = read_record(job / "request.json")
    acceptance, _ = read_record(job / "accepted.json")
    canonical = json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    require(acceptance.get("runner_sha256") == RUNNER_SHA and
            acceptance.get("request_sha256") == hashlib.sha256(canonical).hexdigest()
            and request.get("job_id") == active and request.get("capture_version") == plan["run_commit"],
            "formal training lacks the exact accepted runner request")
    root = Path(plan["output_root"])
    identity, _ = read_record(root / "identity.json")
    require(identity == {"plan_sha256": digest, "authorization_sha256": args.formal_authorization_sha256},
            "formal training belongs to another plan")
    stage = root / f"attempt-r{attempt}"
    choice, _ = read_record(stage / "training.json")
    launch_record, _ = read_record(stage / "training-execution.json")
    require(choice == {"path": args.output, "resume_from": args.resume}
            and launch_record["argv"] == [str(PYTHON), "-B", "-m", PACKAGE + ".train", *sys.argv[1:]],
            "formal training output, recovery checkpoint, or command differs from the accepted runner launch")
    return authority


def job_id(plan, attempt):
    require(type(attempt) is int and 1 <= attempt <= 99, "invalid supervisor attempt")
    return f"letter-beauty-v2-history-attn-formal-b-{plan['job_name']}-r{attempt}"


def check_runner():
    require(sha(RUNNER) == RUNNER_SHA and sha(RUNNER_DEPLOYMENT) == RUNNER_DEPLOYMENT_SHA,
            "approved Beauty runner deployment identity mismatch")
    info, _ = read_record(RUNNER_DEPLOYMENT, RUNNER_DEPLOYMENT_SHA)
    require(info["job_root"] == str(RUNNER_JOBS) and info["hostname"] == socket.gethostname()
            and info["uid"] == os.getuid(), "formal chain is outside the approved runner host/account")
    return info


def runner(*args):
    raw = subprocess.check_output(
        ["/usr/bin/python3", "-I", "-S", "-B", str(RUNNER), *map(str, args)], text=True
    )
    return json.loads(raw)


def get_status(plan, attempt):
    location = RUNNER_JOBS / job_id(plan, attempt)
    return runner("query", location) if location.exists() else None


def clean_terminal(status):
    return (status is not None and status.get("status") in TERMINAL and status.get("cleanup_verified") is True
            and not any(status.get(key) for key in ("worker_alive", "supervisor_alive", "workgroup_live")))


def remaining_budget(plan, earlier):
    elapsed = 0.0
    for status in earlier:
        start = datetime.fromisoformat(status.get("started_at") or status["received_at"])
        end = datetime.fromisoformat(status["ended_at"])
        require(end >= start, "invalid runner accounting timestamps")
        elapsed += (end - start).total_seconds()
    seconds = BUDGETS["chain_seconds"] - math.ceil(elapsed)
    require(seconds > 0, "B series exhausted its cumulative authorized time cap")
    return seconds


def check_attempt(plan, attempt, authorization_digest, plan_digest, lookup=get_status):
    earlier = []
    for number in range(1, attempt):
        status = lookup(plan, number)
        require(clean_terminal(status), f"attempt {number} has no clean terminal runner status")
        request, _ = read_record(RUNNER_JOBS / job_id(plan, number) / "request.json")
        params = {request["argv"][i]: request["argv"][i+1]
                  for i in range(5, len(request["argv"]) - 1, 2)}
        require(params.get("--plan-sha256") == plan_digest
                and params.get("--formal-authorization-sha256") == authorization_digest,
                f"attempt {number} belongs to another plan or authorization")
        require(status["status"] != "SUCCEEDED", "formal chain already completed")
        earlier.append(status)
    return remaining_budget(plan, earlier)


def runner_request(plan, plan_sha, files, review, args, timeout, deployment):
    cmd = [
        str(PYTHON), "-B", "-m", PACKAGE + ".formal", "worker",
        "--plan", str(args.plan), "--plan-sha256", plan_sha, "--attempt", str(args.attempt),
        "--formal-authorization", str(args.formal_authorization),
        "--formal-authorization-sha256", args.formal_authorization_sha256,
    ]
    paths = [(SNAPSHOT / name, digest) for name, digest in files.items()]
    paths.extend([(SNAPSHOT / "SOURCE.sha256", plan["snapshot_manifest_sha256"]),
                  (args.plan, plan_sha), (args.formal_authorization, args.formal_authorization_sha256), review])
    require(len({str(p) for p, _ in paths}) == len(paths), "formal request has reused input file paths")
    config, _ = read_record(EVALUATION_CONFIG, plan["evaluation_config_sha256"])
    env = dict(config["gpu_environment"], CUDA_VISIBLE_DEVICES=plan["gpu_uuid"], CUDA_DEVICE_ORDER="PCI_BUS_ID",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONOPTIMIZE="0", PYTHONDONTWRITEBYTECODE="1",
               PYTHONUNBUFFERED="1", TORCHINDUCTOR_COMPILE_THREADS="1", TOKENIZERS_PARALLELISM="false",
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", USE_TF="0", USE_FLAX="0")
    return {
        "schema": 1, "host": deployment["hostname"], "uid": deployment["uid"], "user": deployment["user"],
        "namespace": deployment["namespace"], "deployment_sha256": RUNNER_DEPLOYMENT_SHA,
        "capture_version": plan["run_commit"], "job_id": job_id(plan, args.attempt), "cwd": str(SNAPSHOT),
        "env": env, "timeout_seconds": timeout, "argv": cmd,
        "inputs": [{"path": str(p), "sha256": h} for p, h in paths],
    }


@contextmanager
def lock(path):
    flags = os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK
    with os.fdopen(os.open(path, flags, 0o600), "r+") as stream:
        require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), f"lock is not a regular file: {path}")
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def install_immutable(path, payload):
    require(path.is_absolute() and path.parent.resolve() == path.parent, f"expected canonical record path: {path}")
    raw = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(name, path)
        except FileExistsError:
            current, _ = read_record(path)
            require(current == payload, f"immutable record already exists with another identity: {path}")
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(name)


def resources(gpu):
    specs = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True
    ).splitlines()
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
         "--format=csv,noheader,nounits"], text=True
    ).splitlines()
    chosen = [list(map(str.strip, line.split(","))) for line in specs if gpu in line]
    require(len(chosen) == 1 and int(chosen[0][0]) == GPU_IDS[gpu]
            and chosen[0][2] == "NVIDIA GeForce RTX 4090" and int(chosen[0][3]) >= 24000,
            "frozen GPU UUID is no longer the expected RTX 4090")
    return {"gpu": chosen[0], "compute": [line for line in processes if line.split(",")[0].strip() == gpu],
            "disk_free_bytes": shutil.disk_usage(SERVER_ROOT).free,
            "scheduler_commands": [name for name in ("squeue", "qstat", "bjobs") if shutil.which(name)]}


def check_idle(plan, first_attempt):
    snapshot = resources(plan["gpu_uuid"])
    require(not snapshot["compute"] and int(snapshot["gpu"][4]) < 100, "approved physical GPU is occupied")
    free = BUDGETS["minimum_start_free_bytes"] if first_attempt else 24 * 1024**3
    require(snapshot["disk_free_bytes"] >= free, "insufficient free work volume for start or recovery")
    return snapshot


def submit(args):
    plan, digest, files, review = gate_records(args.plan, args.formal_authorization,
                                               args.formal_authorization_sha256)
    deployment = check_runner()
    root = Path(plan["output_root"])
    root.mkdir(exist_ok=True)
    require(root.resolve() == root, "formal output must have a canonical path")
    with lock(root / "submit.lock"):
        timeout = check_attempt(plan, args.attempt, args.formal_authorization_sha256, digest)
        request = runner_request(plan, digest, files, review, args, timeout, deployment)
        identity = {"plan_sha256": digest, "authorization_sha256": args.formal_authorization_sha256}
        install_immutable(root / "identity.json", identity)
        request_path = root / f"runner-request-r{args.attempt}.json"
        prior = get_status(plan, args.attempt)
        if prior is not None:
            actual, _ = read_record(RUNNER_JOBS / job_id(plan, args.attempt) / "request.json")
            require(actual == request, "accepted runner request does not match the verified plan")
            result = completed_output(plan, digest, verify_files=True)
            if result is not None:
                return {**prior, "verified_formal_output": result}
            return prior
        result = completed_output(plan, digest, verify_files=True)
        if result is not None:
            return {"submission": "COMPLETE_OUTPUT_NO_RESTART", "verified_formal_output": result}
        install_immutable(request_path, request)
        check_idle(plan, args.attempt == 1)
        return runner("submit", RUNNER_JOBS, request_path)


def durable_checkpoint(path, plan):
    require(path.resolve() == path and path.is_dir(), f"unexpected checkpoint directory: {path}")
    value = verify_checkpoint(path, plan["run_commit"], plan["config_sha256"], "bidirectional")
    step = int(path.name.removeprefix("checkpoint-"))
    require(value["scope"] == "FORMAL" and value["global_step"] == step and step in EPOCH_STEPS,
            "checkpoint is outside this formal B epoch schedule")
    return value


def validation_choice(final, plan):
    """Independent check of all completed validation boundaries and the earliest best."""
    durable_checkpoint(final, plan)
    state, _ = read_record(final / "trainer_state.json")
    losses = [[row["step"], float(row["eval_loss"])] for row in state["log_history"] if "eval_loss" in row]
    require([row[0] for row in losses] == list(EPOCH_STEPS) and all(math.isfinite(row[1]) for row in losses)
            and state["epoch"] == 4.0 and state["global_step"] == 8214,
            "final state lacks complete validation history")
    step, best_loss = min(losses, key=lambda row: (row[1], row[0]))
    selected = final.parent / f"checkpoint-{step}"
    require(state["best_model_checkpoint"] == str(selected) and state["best_metric"] == best_loss,
            "native checkpoint selection differs from earliest minimum validation loss")
    if selected != final:
        durable_checkpoint(selected, plan)
    return selected, best_loss, losses


def result_for_adapter(plan, training, adapter, selected, best_loss, losses, method):
    value = verify_adapter_identity(adapter, scope="FORMAL", mode="bidirectional", run_commit=plan["run_commit"])
    require(value["config_sha256"] == plan["config_sha256"], "exported adapter belongs to a different config")
    checkpoint, _ = read_record(selected / IDENTITY_FILE)
    if value["adapter_sha256"] != checkpoint["adapter_sha256"]:
        # Native save_pretrained packages the same tensors under its own archive
        # metadata. Check tensor equality if serialization bytes ever differ.
        import torch

        chosen = torch.load(selected / "adapter_model.bin", map_location="cpu", weights_only=True)
        exported = torch.load(adapter / "adapter_model.bin", map_location="cpu", weights_only=True)
        require(set(chosen) == set(exported) and len(chosen) == 450 and
                all(chosen[name].dtype == exported[name].dtype and torch.equal(chosen[name], exported[name])
                    for name in chosen), "exported adapter differs from the verified best checkpoint")
    return {
        "schema": 1, "training": str(training), "adapter": str(adapter), "method": method,
        "mode": "bidirectional", "run_commit": plan["run_commit"], "config_sha256": plan["config_sha256"],
        "best_checkpoint": str(selected), "best_eval_loss": best_loss, "validation_losses": losses,
        "completed_training_step": 8214, "checkpoint_manifest_sha256": sha(selected / "complete.json"),
        "adapter_manifest_sha256": sha(adapter / IDENTITY_FILE), "adapter_sha256": value["adapter_sha256"],
    }


def verify_selected_adapter(plan, adapter):
    root = Path(plan["output_root"])
    adapter = Path(adapter)
    receipt, _ = read_record(root / "training-result.json")
    require(adapter.resolve() == adapter and adapter.is_relative_to(root) and str(adapter) == receipt["adapter"],
            "generation must use the one selected B adapter")
    require(receipt["run_commit"] == plan["run_commit"] and receipt["completed_training_step"] == 8214
            and receipt["config_sha256"] == plan["config_sha256"] and receipt["mode"] == "bidirectional",
            "training handoff belongs to a different formal B run")
    value = verify_adapter_identity(adapter, scope="FORMAL", mode="bidirectional", run_commit=plan["run_commit"])
    require(value["config_sha256"] == plan["config_sha256"] and value["adapter_sha256"] == receipt["adapter_sha256"]
            and sha(adapter / IDENTITY_FILE) == receipt["adapter_manifest_sha256"],
            "selected B adapter differs from completed training handoff")
    return value


def training_resume(root, attempt, plan):
    """Use the latest complete checkpoint only within this run's attempt lineage."""
    for earlier in range(attempt - 1, 0, -1):
        marker = root / f"attempt-r{earlier}/training.json"
        if not marker.is_file():
            continue
        choice, _ = read_record(marker)
        training = Path(choice["path"])
        require(training.parent == root and re.fullmatch(r"training-r[1-9][0-9]?", training.name)
                and training.resolve() == training, "unrecognized prior training output path")
        if not training.is_dir():
            continue
        checkpoints = [p for p in training.iterdir() if re.fullmatch(r"checkpoint-[0-9]+", p.name)]
        for path in checkpoints:
            require(path.is_dir() and not path.is_symlink()
                    and int(path.name.removeprefix("checkpoint-")) in EPOCH_STEPS, "unexpected checkpoint path")
        complete = [p for p in checkpoints if (p / "complete.json").is_file()]
        if not complete:
            continue  # A fresh attempt gets a distinct output directory.
        latest = max(complete, key=lambda p: int(p.name.removeprefix("checkpoint-")))
        if latest.name == "checkpoint-8214":
            return training, latest
        durable_checkpoint(latest, plan)
        return training, latest
    return root / f"training-r{attempt}", None


def archive_partial(training, attempt_dir):
    for path in training.iterdir():
        incomplete = (re.fullmatch(r"checkpoint-[0-9]+", path.name)
                      and path.is_dir() and not (path / "complete.json").exists())
        old_failure = path.name in {"failure.rank0.json", "best_adapter", "best_selection.json"}
        if incomplete or old_failure:
            require(not path.is_symlink(), "refusing to move a linked output")
            clean = attempt_dir / "prior-failed-output"
            clean.mkdir(exist_ok=True)
            target = clean / path.name
            require(not target.exists(), "failure evidence archive already exists")
            path.rename(target)


def launch(module, stage, arguments, attempt_dir, env, timeout):
    argv = [str(PYTHON), "-B", "-m", PACKAGE + "." + module, *map(str, arguments)]
    selected_env = {key: env[key] for key in ("CUDA_VISIBLE_DEVICES", "RANK", "LOCAL_RANK", "WORLD_SIZE",
                                              "MASTER_ADDR", "MASTER_PORT") if key in env}
    install_immutable(attempt_dir / (stage + "-execution.json"), {"argv": argv, "env": selected_env,
                                                                "timeout_seconds": timeout})
    subprocess.run(argv, env=env, stdin=subprocess.DEVNULL, check=True, timeout=timeout)


def child_train_env():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return os.environ | {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1", "LOCAL_WORLD_SIZE": "1",
                         "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port)}


def set_phase(attempt_dir, name, **data):
    atomic_json(attempt_dir / "stage.json", {"phase": name, "updated_at": now(), **data})


def auth_args(args):
    return ["--formal-authorization", args.formal_authorization,
            "--formal-authorization-sha256", args.formal_authorization_sha256]


def finish_training(plan, args, root, attempt_dir):
    if (root / "training-result.json").exists():
        receipt, _ = read_record(root / "training-result.json")
        adapter = Path(receipt["adapter"])
        verify_selected_adapter(plan, adapter)
        return adapter
    training, checkpoint = training_resume(root, args.attempt, plan)
    install_immutable(attempt_dir / "training.json", {"path": str(training),
                                                     "resume_from": str(checkpoint) if checkpoint else None})
    if checkpoint is None or checkpoint.name != "checkpoint-8214":
        if checkpoint:
            archive_partial(training, attempt_dir)
        set_phase(attempt_dir, "training", training=str(training), resume_from=str(checkpoint) if checkpoint else None)
        argv = ["--source-root", VENDOR, "--data-root", DATA, "--model", MODEL, "--output", training,
                "--config", TRAIN_CONFIG, "--run-commit", plan["run_commit"], "--formal-plan", args.plan,
                *auth_args(args)]
        if checkpoint:
            argv += ["--resume", checkpoint]
        launch("train", "training", argv, attempt_dir, child_train_env(), BUDGETS["training_seconds"])
    set_phase(attempt_dir, "best-checkpoint")
    selected, best_loss, losses = validation_choice(training / "checkpoint-8214", plan)
    native = training / "best_adapter"
    rank_path = training / "rank0.json"
    rank = read_record(rank_path)[0] if rank_path.is_file() else {}
    if rank.get("status") == "TRAINING_COMPLETE":
        summary, _ = read_record(training / "best_selection.json")
        require(rank.get("global_step") == 8214 and summary["completed_training_step"] == 8214
                and summary["checkpoint"] == str(selected) and summary["value"] == best_loss
                and summary["export_matches_selected_checkpoint_tensors"] is True,
                "native training completion does not match independently validated best checkpoint")
        adapter, method = native, "native DeepSpeed strict best-checkpoint reload"
    else:
        # A complete step-8214 save followed by a crash during export must not
        # cause the optimizer to run again. The recovery child verifies the
        # exact frozen B adapter via the historical strict PEFT load.
        adapter = attempt_dir / "recovered-best-adapter"
        launch("formal", "recovery-export",
               ["recover-export", "--plan", args.plan, "--attempt", args.attempt, *auth_args(args)],
               attempt_dir, os.environ.copy(), BUDGETS["metrics_seconds"])
        recovery, _ = read_record(attempt_dir / "recovery-reload.json")
        require(recovery["checkpoint"] == str(selected) and recovery["strictly_reloaded"] is True,
                "missing exact best-checkpoint recovery proof")
        method = "validated final training checkpoint and strict PEFT weight reload"
    receipt = result_for_adapter(plan, training, adapter, selected, best_loss, losses, method)
    install_immutable(root / "training-result.json", receipt)
    return adapter


def recover_export(args):
    plan = verify_gate(args.plan, args.formal_authorization, args.formal_authorization_sha256)
    own = job_id(plan, args.attempt)
    require(os.environ.get("GRID_JOB_ID") == own
            and os.environ.get("GRID_JOB_DIR") == str(RUNNER_JOBS / own)
            and os.environ.get("CUDA_VISIBLE_DEVICES") == plan["gpu_uuid"], "recovery export is runner-only")
    root = Path(plan["output_root"])
    attempt_dir = root / f"attempt-r{args.attempt}"
    choice, _ = read_record(attempt_dir / "training.json")
    training = Path(choice["path"])
    require(training.parent == root and training.resolve() == training, "invalid recovery training root")
    selected, _, _ = validation_choice(training / "checkpoint-8214", plan)
    from .evaluate import beauty, load_inference
    config, _ = read_record(EVALUATION_CONFIG)
    baseline = config["baseline"]
    source = beauty.frozen_dependency(config, Path(baseline["entry"]))
    author = source.author_source(baseline["source-root"])
    index, _, _, _ = source.catalog_and_users(baseline["data-root"])
    model_report = {}
    load_inference(selected, model_report, config, source, author, index)
    target = attempt_dir / "recovered-best-adapter"
    target.mkdir()
    for name in sorted(ADAPTER_MEMBERS):
        shutil.copyfile(selected / name, target / name)
    manifest, _ = read_record(selected / IDENTITY_FILE)
    manifest["checkpoint_source"] = {"path": str(selected), "complete_sha256": sha(selected / "complete.json"),
                                      "global_step": int(selected.name.removeprefix("checkpoint-"))}
    install_immutable(target / IDENTITY_FILE, manifest)
    value = verify_adapter_identity(target, scope="FORMAL", mode="bidirectional", run_commit=plan["run_commit"])
    atomic_json(attempt_dir / "recovery-reload.json", {
        "checkpoint": str(selected), "strictly_reloaded": True, "adapter_sha256": value["adapter_sha256"],
        "model": model_report["model"], "strict_adapter_reload": model_report["strict_adapter_reload"],
    })


def finish_metrics(plan, root, attempt_dir, args, generation):
    from .evaluate import verify_completed_metrics
    for number in range(1, args.attempt):
        prior = root / f"attempt-r{number}/metrics"
        if os.path.lexists(prior / "completion.json"):
            verify_completed_metrics(prior, generation, plan["run_commit"])
            return prior
    dest = attempt_dir / "metrics"
    set_phase(attempt_dir, "cpu-metrics", metrics=str(dest))
    launch("evaluate", "metrics", ["metrics", "--config", EVALUATION_CONFIG, "--replay-dir", REPLAY,
                                   "--predictions", generation, "--output", dest,
                                   "--evaluation-commit", plan["run_commit"], "--purpose", "formal"],
           attempt_dir, os.environ | {"CUDA_VISIBLE_DEVICES": ""}, BUDGETS["metrics_seconds"])
    verify_completed_metrics(dest, generation, plan["run_commit"])
    return dest


def completed_output(plan, plan_sha, *, verify_files):
    """Recognize committed verified output even if the runner later loses control."""
    root = Path(plan["output_root"])
    marker = root / "complete.json"
    if not os.path.lexists(marker):
        return None
    result, _ = read_record(marker)
    generation, metrics, adapter = (Path(result[key]) for key in ("generation", "metrics", "adapter"))
    require(result["status"] == "COMPLETE" and result["plan_sha256"] == plan_sha
            and result["run_commit"] == plan["run_commit"] and generation == root / "generation"
            and metrics.name == "metrics" and metrics.parent.parent == root
            and re.fullmatch(r"attempt-r[1-9][0-9]?", metrics.parent.name)
            and adapter.is_relative_to(root) and all(p.resolve() == p for p in (metrics, generation, adapter)),
            "formal result identity or path mismatch")
    handoff, _ = read_record(root / "training-result.json", result["training_sha256"])
    summary, _ = read_record(metrics / "summary.json", result["metrics_summary_sha256"])
    require(handoff["adapter"] == str(adapter) and handoff["run_commit"] == plan["run_commit"]
            and handoff["completed_training_step"] == 8214 and summary["status"] == "PASS_V2_EVALUATION"
            and summary["run_commit"] == plan["run_commit"] and summary["users_per_condition"] == 22363
            and summary["condition_count"] == len(CONDITIONS), "formal result lacks accepted model and metrics")
    if verify_files:
        from .evaluate import verify_completed_generation, verify_completed_metrics

        verify_selected_adapter(plan, adapter)
        verify_completed_generation(generation, plan["run_commit"], adapter, plan["backend"])
        verify_completed_metrics(metrics, generation, plan["run_commit"])
    return result


def worker(args):
    plan, digest, files, review = gate_records(args.plan, args.formal_authorization,
                                               args.formal_authorization_sha256)
    deployment = check_runner()
    own = job_id(plan, args.attempt)
    require(args.plan_sha256 == digest and os.environ.get("GRID_JOB_ID") == own
            and os.environ.get("GRID_JOB_DIR") == str(RUNNER_JOBS / own),
            "formal worker must be owned by the verified runner request")
    timeout = check_attempt(plan, args.attempt, args.formal_authorization_sha256, digest)
    request = runner_request(plan, digest, files, review, args, timeout, deployment)
    actual, _ = read_record(RUNNER_JOBS / own / "request.json")
    require(actual == request and os.environ.get("CUDA_VISIBLE_DEVICES") == plan["gpu_uuid"],
            "worker runtime differs from accepted request")
    root = Path(plan["output_root"])
    with lock(root / "worker.lock"):
        identity, _ = read_record(root / "identity.json")
        require(identity == {"plan_sha256": digest, "authorization_sha256": args.formal_authorization_sha256},
                "output root is bound to another plan")
        previous = completed_output(plan, digest, verify_files=True)
        if previous is not None:
            return {"status": "COMPLETE_OUTPUT_NO_RESTART", "verified_formal_output": previous}
        attempt_dir = root / f"attempt-r{args.attempt}"
        attempt_dir.mkdir()
        set_phase(attempt_dir, "checking-resources")
        try:
            with lock(SERVER_ROOT / f"gpu-{GPU_IDS[plan['gpu_uuid']]}.lock"):
                atomic_json(attempt_dir / "resources-start.json", check_idle(plan, args.attempt == 1))
                adapter = finish_training(plan, args, root, attempt_dir)
                from .evaluate import verify_completed_generation
                generation = root / "generation"
                if not (generation / "final").exists():
                    set_phase(attempt_dir, "generation", generation=str(generation), adapter=str(adapter))
                    launch("evaluate", "generation",
                           ["generate", "--adapter", adapter, "--run-commit", plan["run_commit"], "--output", generation,
                            "--config", EVALUATION_CONFIG, "--replay-dir", REPLAY, "--purpose", "formal",
                            "--backend", plan["backend"], "--max-seconds", BUDGETS["generation_seconds"],
                            "--plan", args.plan, *auth_args(args)], attempt_dir, os.environ.copy(),
                           BUDGETS["generation_seconds"] + 120)
                verify_completed_generation(generation, plan["run_commit"], adapter, plan["backend"])
            metrics = finish_metrics(plan, root, attempt_dir, args, generation)
            atomic_json(attempt_dir / "resources-end.json", resources(plan["gpu_uuid"]))
            result = {
                "status": "COMPLETE", "job_id": own, "plan_sha256": digest, "run_commit": plan["run_commit"],
                "adapter": str(adapter), "training_sha256": sha(root / "training-result.json"),
                "generation": str(generation), "metrics": str(metrics),
                "metrics_summary_sha256": sha(metrics / "summary.json"), "completed_at": now(),
            }
            set_phase(attempt_dir, "complete", metrics=str(metrics))
            install_immutable(root / "complete.json", result)
        except BaseException as exc:
            value, _ = read_record(attempt_dir / "stage.json")
            set_phase(attempt_dir, "failed", prior_phase=value["phase"], error=repr(exc))
            raise


def events_tail(path):
    if not path.is_file():
        return {}
    with path.open("rb") as stream:
        cut = max(stream.seek(0, 2) - (1 << 20), 0)
        stream.seek(cut)
        raw = stream.read()
    lines = raw.split(b"\n")
    complete = lines[int(cut > 0):-1]
    events = {}
    optimizer = []
    for line in complete:
        row = json.loads(line)
        if row.get("event") == "optimizer_step":
            optimizer.append(row)
        if row.get("event") in ("batch", "optimizer_step", "log", "gpu_memory", "checkpoint_complete", "epoch_end"):
            events[row["event"]] = row
    if len(optimizer) >= 2:
        previous, latest = optimizer[-2:]
        elapsed = latest["elapsed_seconds"] - previous["elapsed_seconds"]
        steps = latest["step"] - previous["step"]
        if previous["attempt_job_id"] == latest["attempt_job_id"] and elapsed > 0 and steps > 0:
            events["measured_progress_rate"] = {
                "updates_per_second": steps / elapsed, "micro_batches_per_second": 8 * steps / elapsed,
                "observed_between_optimizer_steps": [previous["step"], latest["step"]],
            }
    return events


def observe(args):
    plan, digest = validate_plan(args.plan)
    check_runner()
    root = Path(plan["output_root"])
    folder = root / f"attempt-r{args.attempt}"
    stage = read_record(folder / "stage.json")[0] if (folder / "stage.json").exists() else None
    training = None
    if (folder / "training.json").is_file():
        candidate = Path(read_record(folder / "training.json")[0]["path"])
        require(candidate.parent == root, "foreign progress path")
        training = events_tail(candidate / "events.rank0.jsonl")
    generation = root / "generation"
    chunks = generation / "chunks"
    runner_state = get_status(plan, args.attempt)
    runner_log = RUNNER_JOBS / job_id(plan, args.attempt) / "work.log"
    log_tail = None
    if runner_log.is_file():
        with runner_log.open("rb") as stream:
            stream.seek(max(stream.seek(0, 2) - 2048, 0))
            log_tail = stream.read().decode(errors="replace")
    result = completed_output(plan, digest, verify_files=args.verify_completion)
    return {
        "scope": "read-only observation, first full timing sample at 30 minutes",
        "observed_at": now(), "plan_sha256": digest, "runner": runner_state, "runner_log_tail": log_tail,
        "hardware": resources(plan["gpu_uuid"]), "stage": stage, "latest_training_events": training,
        "generation_chunk_candidates": sorted(p.name for p in chunks.iterdir() if p.is_dir()) if chunks.is_dir() else [],
        "recommendation_metrics": result["metrics"] if result else "尚未产生",
        "completed_output": result, "completed_output_rehashed": bool(result and args.verify_completion),
    }


def main():
    require(not sys.flags.optimize, "formal chain requires all runtime checks")
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", allow_abbrev=False)
    plan.add_argument("--run-commit", required=True)
    plan.add_argument("--job-name", required=True)
    plan.add_argument("--gpu-uuid", required=True)
    plan.add_argument("--backend", choices=("eager", "sdpa"), required=True)
    plan.add_argument("--output", type=Path, required=True)
    for name in ("submit", "observe", "worker", "recover-export"):
        command = commands.add_parser(name, allow_abbrev=False)
        command.add_argument("--plan", required=True, type=Path)
        command.add_argument("--attempt", type=int, default=1)
        if name != "observe":
            command.add_argument("--formal-authorization", type=Path, required=True)
            command.add_argument("--formal-authorization-sha256", required=True)
        if name == "worker":
            command.add_argument("--plan-sha256", required=True)
        if name == "observe":
            command.add_argument("--output", type=Path)
            command.add_argument("--verify-completion", action="store_true",
                                 help="recheck all committed training, prediction, and metric file identities")
    args = parser.parse_args()
    if args.command == "plan":
        value = plan_fields(args.run_commit, args.job_name, args.gpu_uuid, args.backend, sha(SNAPSHOT / "SOURCE.sha256"))
        source_files(SNAPSHOT, value["snapshot_manifest_sha256"])
        install_immutable(args.output, value)
        print(json.dumps({"plan": str(args.output), "plan_sha256": sha(args.output)}, indent=2))
    else:
        result = {"submit": submit, "observe": observe, "worker": worker, "recover-export": recover_export}[args.command](args)
        if result is not None:
            if getattr(args, "output", None):
                install_immutable(args.output, result)
            print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
