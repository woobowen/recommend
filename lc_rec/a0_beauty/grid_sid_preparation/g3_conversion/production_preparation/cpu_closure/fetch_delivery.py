#!/usr/bin/env python3
"""Read cleared runner outputs and preserve raw evidence for the GitHub delivery."""
import argparse
import base64
from datetime import datetime, timezone
import io
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tarfile

from common import digest, need, pin, read, sha, write

HERE = Path(__file__).resolve().parent
SSH = ("/mnt/c/Windows/System32/OpenSSH/ssh.exe", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12", "new-server")
JOB_RECORDS = ("request.json", "accepted.json", "attempt.json", "status.json", "work.log", "supervisor.log")


def remote(*args):
    return subprocess.check_output([*SSH, shlex.join(map(str, args))])


def receive(job, state, evidence, output):
    raw = subprocess.check_output([*SSH, f"set -o pipefail; tar -C {shlex.quote(str(job))} -cf - -- "
                                  + " ".join(JOB_RECORDS) + " output | base64 -w 0"])
    decoded = base64.b64decode(raw, validate=True)
    need(len(decoded) <= 200 * 1024**2, "unexpected delivery transfer size")
    published = {}
    with tarfile.open(fileobj=io.BytesIO(decoded), mode="r:") as archive:
        seen = set()
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            need(not path.is_absolute() and ".." not in path.parts and path not in seen,
                 f"unsafe or duplicate transfer member: {path}")
            seen.add(path)
            if member.isdir():
                need(path.parts[0] == "output", f"unexpected archive directory: {path}")
                continue
            need(member.isfile(), f"unexpected archive type: {path}")
            blob = archive.extractfile(member).read()
            if len(path.parts) == 1:
                need(path.name in JOB_RECORDS, f"unexpected job record: {path}")
                destination = evidence / path.name
            else:
                need(path.parts[0] == "output", f"unexpected artifact outside the job output: {path}")
                relative = str(PurePosixPath(*path.parts[1:]))
                destination = output / relative
                published[f"output/{relative}"] = destination
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as sink:
                sink.write(blob)
    need(all((evidence / p).is_file() for p in JOB_RECORDS), "missing runner record")
    local_request = read(evidence / "request.json")
    need(digest(local_request) == state["request_sha256"], "transferred request changed")
    need(read(evidence / "accepted.json")["request_sha256"] == state["request_sha256"], "transferred acceptance changed")
    local_status = read(evidence / "status.json")
    need(local_status["status"] == state["status"] and local_status["exit_code"] == state["exit_code"]
         and local_status["ended_at"] == state["ended_at"], "transferred runner status changed")
    rows = {r["path"]: r for r in state["artifacts"]}
    need(set(rows) == set(published), "runner output file set differs in delivery")
    for relative, dest in published.items():
        row = rows[relative]
        pin(dest, row["sha256"], row["bytes"])
    if state["status"] == "SUCCEEDED":
        manifest = read(output / "manifest.json")["files"]
        need(set(f"output/{p}" for p in manifest) | {"output/manifest.json"} == set(rows), "output manifest member set")
        for path, row in manifest.items():
            runner_row = rows[f"output/{path}"]
            need(row["sha256"] == runner_row["sha256"] and row["bytes"] == runner_row["bytes"],
                 f"output manifest mismatch: {path}")
    return {"transferred_bytes_before_base64": len(decoded), "runner_artifacts_verified": len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    dest = args.output
    dest.mkdir()
    cfg = read(HERE / "closure_config.json")
    release = Path(cfg["closure_root"]) / "releases" / args.implementation
    raw = remote("/usr/bin/python3", "-B", release/"launch.py", "query")
    budget = json.loads(raw)
    (dest / "budget.json").write_bytes(raw)
    originals = dest / "original-control"
    originals.mkdir()
    original_root = Path(cfg["original_root"])
    for name in ("plan.json", "plan.identity.json", "state.json", "controller.log"):
        (originals/name).write_bytes(remote("cat", original_root/name))
    pin(originals/"plan.json", cfg["plan_raw_sha256"])
    seal = read(originals/"plan.identity.json")
    need(digest(read(originals/"plan.json")) == cfg["plan_canonical_sha256"] == seal["sha256"],
         "original preserved plan canonical identity")
    records = {}
    deployments = set()
    latest_success = {attempt["kind"]: attempt["job_id"] for attempt in budget["attempts"]
                      if attempt.get("runner", {}).get("status") == "SUCCEEDED"}
    for attempt in budget["attempts"]:
        state = attempt["runner"]
        need(attempt["charged_seconds"] is not None and state["cleanup_verified"] and
             not any(state[k] for k in ("worker_alive", "supervisor_alive", "workgroup_live")),
             f"attempt not cleared: {attempt['job_id']}")
        job = Path(cfg["cpu_job_root"]) / attempt["job_id"]
        evidence = dest / "attempts" / f"{attempt['kind']}-r{attempt['correction_round']}"
        if latest_success.get(attempt["kind"]) == state["job_id"]:
            output = dest / attempt["kind"]
        else:
            output = evidence / ("superseded-output" if state["status"] == "SUCCEEDED" else "partial-output")
        records[attempt["job_id"]] = receive(job, state, evidence, output)
        deployments.add(read(evidence/"request.json")["capture_version"])
    if (dest/"compute/input-verification.json").is_file():
        guard = read(dest/"compute/input-verification.json")
        need(guard["original_state_sha256"] == sha(originals/"state.json"), "original ledger changed after completion")
    (dest / "deployment").mkdir()
    for revision in deployments:
        value = remote("cat", Path(cfg["closure_root"]) / "releases" / revision / "deployment.json")
        (dest / "deployment" / f"{revision}.json").write_bytes(value)
    write(dest / "transfer-verification.json", {"status": "PASS", "utc": datetime.now(timezone.utc).isoformat(),
          "transport": "Windows OpenSSH and lossless base64 tar stream", "budget_sha256": sha(dest/"budget.json"),
          "attempts": records}, exclusive=True)
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
