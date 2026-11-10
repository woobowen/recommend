#!/usr/bin/env python3
"""Download one immutable official Qwen asset set; Python standard library only.

Run this through the already-reviewed project runner. The runner owns process
independence and the global 16-hour/60-GiB budgets. This script owns per-model
locking, the cumulative six-hour active window, resumable partial files, trusted
content checks, atomic publication and an external download receipt. It never
downloads .bin files or chooses a new revision.
"""

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


GIB = 1024 ** 3
MODEL_IDS = {"Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-1.7B-Base", "Qwen/Qwen3.5-0.8B-Base"}


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as out:
        json.dump(obj, out, indent=2, sort_keys=True)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, path)


def identity(path, expected):
    """Check complete payload against an official Git blob or LFS OID and SHA256."""
    if not path.is_file() or path.stat().st_size != expected["size"]:
        return None
    sha = hashlib.sha256()
    blob = hashlib.sha1(f"blob {expected['size']}\0".encode())
    with path.open("rb") as inp:
        while chunk := inp.read(8 * 1024 ** 2):
            sha.update(chunk)
            blob.update(chunk)
    digest = sha.hexdigest()
    if expected.get("sha256") and digest != expected["sha256"]:
        return None
    if expected.get("lfs_oid_sha256"):
        if digest != expected["lfs_oid_sha256"]:
            return None
    elif blob.hexdigest() != expected["git_blob_sha1"]:
        return None
    return {"path": expected["path"], "size": path.stat().st_size,
            "sha256": digest, "git_blob_sha1": blob.hexdigest(),
            "trusted_identity": "lfs_sha256" if expected.get("lfs_oid_sha256") else "git_blob_sha1"}


class AssetRun:
    def __init__(self, args):
        self.args = args
        self.manifest_path = Path(args.manifest).resolve()
        self.manifest = json.loads(self.manifest_path.read_text())
        m = self.manifest
        if m["model_id"] not in MODEL_IDS or len(m["revision"]) != 40:
            raise ValueError("Only the three authorized official Base model IDs and full revisions are accepted")
        if any(c not in "0123456789abcdef" for c in m["revision"]):
            raise ValueError("Invalid immutable revision")
        if m["weight_bytes"] > 24 * GIB:
            raise ValueError("Model weight payload exceeds the entire authorized collection budget")
        for entry in m["files"]:
            path = PurePosixPath(entry["path"])
            if path.is_absolute() or ".." in path.parts or path.suffix == ".bin":
                raise ValueError("Unsafe or disallowed manifest path")
            if not entry.get("sha256") or len(entry["sha256"]) != 64:
                raise ValueError("Each required file needs a SHA256 in the pinned manifest")
        self.root = Path(args.model_root).resolve()
        self.name = m["model_id"].split("/")[-1]
        self.final = self.root / self.name / m["revision"]
        self.stage = self.root / ".incoming" / self.name / m["revision"]
        self.stage.mkdir(parents=True, exist_ok=True)
        self.progress_path = self.stage / "download_progress.json"
        previous = json.loads(self.progress_path.read_text()) if self.progress_path.exists() else {}
        self.prior_active = float(previous.get("active_seconds", 0))
        self.started = time.monotonic()
        self.last_byte = self.started
        self.last_report = 0
        self.receipt_path = Path(args.receipt_root).resolve() / (args.job_id + ".download.json")
        self.receipt = {"schema_version": 1, "job_id": args.job_id,
                        "run_commit": args.run_commit, "model_id": m["model_id"],
                        "revision": m["revision"], "started_utc": utc(),
                        "script": str(Path(__file__).resolve()),
                        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        "manifest_path": str(self.manifest_path),
                        "manifest_sha256": hashlib.sha256(self.manifest_path.read_bytes()).hexdigest(),
                        "argv": sys.argv, "interpreter": sys.executable, "cwd": os.getcwd(),
                        "pid": os.getpid(), "host": socket.gethostname(),
                        "final_path": str(self.final), "staging_path": str(self.stage),
                        "downloaded_bytes_this_attempt": 0, "files": [], "events": [],
                        "status": "RUNNING", "exit_code": None,
                        "budget": {"per_model_active_seconds": args.max_seconds,
                                   "no_valid_byte_seconds": args.stall_seconds,
                                   "collection_weight_limit_bytes": 24 * GIB,
                                   "reserve_free_bytes": 30 * GIB,
                                   "reserve_free_fraction": 0.10,
                                   "collection_global_budgets_enforced_by": "project runner/controller"}}

    def elapsed(self):
        return self.prior_active + time.monotonic() - self.started

    def event(self, event, **kwargs):
        item = {"utc": utc(), "event": event, **kwargs}
        self.receipt["events"].append(item)
        print(json.dumps(item, sort_keys=True), flush=True)
        self.save()

    def save(self):
        self.receipt["active_seconds"] = self.elapsed()
        self.receipt["updated_utc"] = utc()
        dump(self.receipt_path, self.receipt)
        if self.stage.exists():
            dump(self.progress_path, {"active_seconds": self.elapsed(),
                                      "updated_utc": utc(), "job_id": self.args.job_id,
                                      "revision": self.manifest["revision"],
                                      "status": self.receipt["status"]})

    def check_budget(self, needed=0):
        if self.elapsed() >= self.args.max_seconds:
            raise TimeoutError("Cumulative per-model six-hour active window exhausted")
        if time.monotonic() - self.last_byte >= self.args.stall_seconds:
            raise TimeoutError("No new valid payload bytes for the permitted stall window")
        disk = shutil.disk_usage(self.root)
        if disk.free - needed < max(30 * GIB, int(disk.total * 0.10)):
            raise RuntimeError("Storage safety reserve would be violated")

    def fetch(self, entry):
        target = self.stage / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        checked = identity(target, entry)
        if checked:
            self.event("staging_verified_reuse", path=entry["path"])
            return checked
        if target.exists():
            raise RuntimeError(f"Existing unverified complete file retained: {target}")
        partial = target.with_name(target.name + ".part")
        if partial.exists() and partial.stat().st_size > entry["size"]:
            raise RuntimeError(f"Oversized partial retained: {partial}")
        # Reuse only the exact authorized model snapshot or exact official blob.
        cache = Path.home() / ".cache" / "huggingface" / "hub" / ("models--" + self.manifest["model_id"].replace("/", "--"))
        candidates = [cache / "snapshots" / self.manifest["revision"] / entry["path"],
                      cache / "blobs" / (entry.get("lfs_oid_sha256") or entry["git_blob_sha1"])]
        candidates += [Path(p) / entry["path"] for p in self.args.cache_dir]
        for source in candidates:
            if source.exists() and identity(source, entry):
                self.check_budget(entry["size"])
                shutil.copyfile(source, partial)
                self.event("cache_verified_copy", path=entry["path"], source=str(source))
                break
        url = "https://huggingface.co/{}/resolve/{}/{}?download=true".format(
            self.manifest["model_id"], self.manifest["revision"], urllib.parse.quote(entry["path"], safe="/"))
        for attempt in range(self.args.retries + 1):
            self.check_budget()
            offset = partial.stat().st_size if partial.exists() else 0
            if offset == entry["size"]:
                checked = identity(partial, entry)
                if not checked:
                    raise RuntimeError(f"Full partial fails trusted SHA; retained for diagnosis: {partial}")
                os.replace(partial, target)
                self.event("file_verified", **checked)
                return checked
            self.check_budget(entry["size"] - offset)
            headers = {"User-Agent": "recommend-qwen-preflight/1.0", "Accept-Encoding": "identity"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=45) as response:
                    if offset:
                        content_range = response.headers.get("Content-Range", "")
                        if response.status != 206 or not content_range.startswith(f"bytes {offset}-"):
                            raise RuntimeError("Server ignored resume range; valid partial retained")
                    self.event("transfer_started", path=entry["path"], attempt=attempt, resume_offset=offset,
                               http_status=response.status)
                    with partial.open("ab" if offset else "wb") as out:
                        while True:
                            self.check_budget()
                            chunk = response.read(min(1024 ** 2, entry["size"] - offset + 1))
                            if not chunk:
                                break
                            if offset + len(chunk) > entry["size"]:
                                raise RuntimeError("HTTP payload larger than pinned file size")
                            out.write(chunk)
                            offset += len(chunk)
                            self.last_byte = time.monotonic()
                            self.receipt["downloaded_bytes_this_attempt"] += len(chunk)
                            if self.last_byte - self.last_report >= 20:
                                out.flush()
                                self.last_report = self.last_byte
                                self.event("progress", path=entry["path"], valid_candidate_bytes=offset,
                                           expected_bytes=entry["size"])
                        out.flush()
                        os.fsync(out.fileno())
                if offset != entry["size"]:
                    raise IOError(f"Premature EOF at {offset}/{entry['size']}; partial retained")
            except (OSError, urllib.error.URLError, RuntimeError) as exc:
                self.event("transfer_error", path=entry["path"], attempt=attempt,
                           error_type=type(exc).__name__, error=str(exc)[:600],
                           retained_bytes=partial.stat().st_size if partial.exists() else 0)
                if attempt >= self.args.retries:
                    raise RuntimeError("Finite transfer retry budget exhausted") from exc
                delay = min(60, 2 ** attempt)
                self.check_budget()
                time.sleep(delay)
        checked = identity(partial, entry)
        if not checked:
            raise RuntimeError("Completed download fails trusted identity; payload retained")
        os.replace(partial, target)
        self.event("file_verified", **checked)
        return checked

    def run(self):
        lock_path = self.root / ".incoming" / self.name / (self.manifest["revision"] + ".lock")
        with lock_path.open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("An existing job owns this immutable model download") from exc
            self.event("accepted", model_id=self.manifest["model_id"], revision=self.manifest["revision"])
            if self.final.exists():
                self.receipt["files"] = [identity(self.final / f["path"], f) for f in self.manifest["files"]]
                if not all(self.receipt["files"]):
                    raise RuntimeError("Existing official directory fails manifest; preserved without overwriting")
                self.event("formal_asset_verified_reuse")
            else:
                self.receipt["files"] = [self.fetch(f) for f in self.manifest["files"]]
                dump(self.stage / "ASSET_MANIFEST.json", self.manifest)
                self.final.parent.mkdir(parents=True, exist_ok=True)
                os.rename(self.stage, self.final)
                self.event("atomic_publication", path=str(self.final))
            self.receipt["status"] = "PASS"
            self.receipt["exit_code"] = 0
            self.receipt["completed_utc"] = utc()
            self.receipt["cleanup_verified"] = {"worker_children": [], "partial_files_retained": [],
                                                "formal_asset_exists": self.final.is_dir()}
            self.save()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--model-root", required=True)
    p.add_argument("--receipt-root", required=True)
    p.add_argument("--job-id", required=True)
    p.add_argument("--run-commit", required=True)
    p.add_argument("--cache-dir", action="append", default=[])
    p.add_argument("--max-seconds", type=int, default=21600)
    p.add_argument("--stall-seconds", type=int, default=1200)
    p.add_argument("--retries", type=int, default=8)
    args = p.parse_args()
    if not 0 < args.max_seconds <= 21600 or not 0 < args.stall_seconds <= 1200 or not 0 <= args.retries <= 8:
        p.error("Per-model time, stall, or retry allowance exceeds the authorized bound")
    if len(args.run_commit) != 40 or any(c not in "0123456789abcdef" for c in args.run_commit):
        p.error("--run-commit must be a full commit SHA")
    runner = None
    def terminate(signum, frame):
        raise InterruptedError(f"External signal {signum}; valid partials retained")
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        runner = AssetRun(args)
        runner.run()
        return 0
    except Exception as exc:
        if runner:
            runner.receipt["status"] = "BLOCKED"
            runner.receipt["exit_code"] = 1
            runner.receipt["completed_utc"] = utc()
            runner.receipt["error"] = {"type": type(exc).__name__, "message": str(exc)}
            runner.receipt["cleanup_verified"] = {"worker_children": [],
                "partial_files_retained": [str(p) for p in runner.stage.rglob("*.part")],
                "formal_asset_exists": runner.final.is_dir()}
            runner.event("terminal_failure", error_type=type(exc).__name__, error=str(exc))
        else:
            print(json.dumps({"status": "BLOCKED", "error": str(exc)}), flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
