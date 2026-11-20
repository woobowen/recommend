#!/usr/bin/env python3
"""Read fixed GitHub blobs directly, then prepare and verify the review ZIP."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import zipfile

from common import need, sha, write

HERE = Path(__file__).resolve().parent
API = "repos/woobowen/recommend"
POWERSHELL = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
SECRET = re.compile(
    rb"-----BEGIN (?:[A-Z]+ )*PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
    rb"|sk-(?:proj-)?[A-Za-z0-9_-]{25,}|hf_[A-Za-z0-9]{30,}|(?:AKIA|ASIA)[A-Z0-9]{16}"
    rb"|Authorization:\s*Bearer\s+[A-Za-z0-9_.-]{12,}", re.I)


def gh(endpoint):
    return json.loads(subprocess.check_output(["gh", "api", f"{API}/{endpoint}"]))


def git_bytes(*args):
    return subprocess.check_output(["git", *map(str, args)])


def data_identity(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def read_gh_blob(commit, path, expected=None):
    blob = git_bytes("rev-parse", f"{commit}:{path}").decode().strip()
    reply = gh(f"git/blobs/{blob}")
    need(reply["encoding"] == "base64" and reply["sha"] == blob, f"unexpected GitHub blob: {path}")
    data = base64.b64decode(reply["content"])
    need(reply["size"] == len(data) and data == git_bytes("show", f"{commit}:{path}"), f"GitHub byte mismatch: {path}")
    if expected:
        need(data_identity(data) == expected, f"deployed input differs from GitHub: {path}")
    return data


def link():
    return "https://github.com/woobowen/recommend/tree"


def verify_windows(path):
    windows = subprocess.check_output(["wslpath", "-w", str(path)], text=True).strip()
    code = ("$ErrorActionPreference='Stop'; $p='" + windows.replace("'", "''") +
            "'; [PSCustomObject]@{Length=(Get-Item -LiteralPath $p).Length;"
            "SHA256=(Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash}|ConvertTo-Json -Compress")
    output = subprocess.check_output([POWERSHELL, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", code],
                                    text=True, timeout=45)
    actual = json.loads(output.lstrip("\ufeff"))
    need(actual["Length"] == path.stat().st_size and actual["SHA256"].lower() == sha(path), "Windows readback differs")
    return windows


def safe_member(name):
    p = PurePosixPath(name)
    return (bool(p.parts) and p.as_posix() == name and not p.is_absolute() and ".." not in p.parts
            and not any(c in name for c in ("\\", ":", "\x00")))


def pack(output, files):
    for name, value in files.items():
        need(safe_member(name) and len(value) < 16 * 1024**2 and not SECRET.search(value),
             f"unsafe, oversized or sensitive ZIP member: {name}")
    member_manifest = {name: data_identity(value) for name, value in sorted(files.items())}
    files["MEMBER_SHA256.json"] = json.dumps(member_manifest, ensure_ascii=False, indent=2).encode() + b"\n"
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=7) as archive:
        for name, value in sorted(files.items()):
            archive.writestr(name, value)
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        need(len(names) == len(set(names)) and set(names) == set(files) and archive.testzip() is None,
             "ZIP CRC, duplicates or member set failure")
        for member in archive.infolist():
            name = member.filename
            need(safe_member(name) and not stat.S_ISLNK(member.external_attr >> 16)
                 and member.file_size == len(files[name]) and archive.read(name) == files[name],
                 f"ZIP member type or identity mismatch: {name}")
    return member_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    need(not output.exists(), "ZIP destination already exists")
    delivery = args.delivery
    repo = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    prefix = HERE.relative_to(repo).as_posix()
    ref = gh("git/ref/heads/wbw-lc-rec")["object"]["sha"]
    need(ref == delivery, "GitHub wbw-lc-rec does not point to the requested delivery")
    gh_commit = gh(f"git/commits/{delivery}")
    need(gh_commit["tree"]["sha"] == git_bytes("rev-parse", f"{delivery}^{{tree}}").decode().strip(), "GitHub commit tree differs")
    snapshot_root = f"{prefix}/results"
    raw_budget = git_bytes("show", f"{delivery}:{snapshot_root}/budget.json")
    budget = json.loads(raw_budget)
    job_revisions = {}
    for attempt in budget["attempts"]:
        p = f"{snapshot_root}/attempts/{attempt['kind']}-r{attempt['correction_round']}/request.json"
        revision = json.loads(git_bytes("show", f"{delivery}:{p}"))["capture_version"]
        job_revisions[attempt["job_id"]] = revision
    cfg = json.loads(git_bytes("show", f"{delivery}:{prefix}/closure_config.json"))
    run = cfg["run_commit"]
    path_to_arc = {
        (delivery, path): "delivery/" + path.removeprefix(prefix + "/")
        for path in git_bytes("ls-tree", "-r", "--name-only", delivery, "--", prefix).decode().splitlines()
    }
    g3_index = "lc_rec/a0_beauty/grid_sid_preparation/g3_conversion/Beauty.g3.index.json"
    frozen = {
        g3_index,
        "lc_rec/a0_beauty/config/formal_g3_beauty_seed42.json",
        *("lc_rec/a0_beauty/" + name for name in ("formal_train.py", "production_pipeline.py", "production_control.py",
           "production_identity.py", "formal_evaluate.py", "offline_metrics.py", "paired_beauty.py",
           "shuffle_runtime.py", "shuffle_entry.py", "g3_identity.py")),
        "lc_rec/letter/review/final_audit_20261113/statistics_verify.py",
        "lc_rec/letter/review/final_audit_20261113/external/official_cce.py",
    }
    path_to_arc.update({(run, name): "frozen/" + name for name in frozen})
    old_base = prefix.removesuffix("/cpu_closure") + "/formal_launch/observations/20261120T053043Z"
    for name in ("README.md", "runner-query.json", "controller.log", "observation.json"):
        path_to_arc[cfg["status_commit"], f"{old_base}/{name}"] = f"prior_timeout/{name}"
    # Deployed source hashes must bind both the byte-read deployment versions and any unchanged delivery copies.
    deployed_expectations = {}
    for revision in set(job_revisions.values()):
        meta = json.loads(git_bytes("show", f"{delivery}:{snapshot_root}/deployment/{revision}.json"))
        need(meta["implementation_commit"] == revision, "deployment records disagree")
        for name, expected in meta["files"].items():
            p = f"{prefix}/{name}"
            path_to_arc.setdefault((revision, p), f"executed/{revision}/{name}")
            deployed_expectations[revision, p] = expected
    index_bytes = git_bytes("show", f"{run}:{g3_index}")
    need(hashlib.sha256(index_bytes).hexdigest() == cfg["asset_sha256"]["index"], "frozen G3 index differs")
    deployed_expectations[run, g3_index] = data_identity(index_bytes)
    origin_manifest = json.loads(git_bytes("show", f"{delivery}:{snapshot_root}/compute/input-verification.json"))
    for path in frozen:
        if path in origin_manifest["frozen_sources"]:
            row = origin_manifest["frozen_sources"][path]
            deployed_expectations[run, path] = {"bytes": row["bytes"], "sha256": row["sha256"]}

    def get(key):
        raw = read_gh_blob(*key, deployed_expectations.get(key))
        return key, raw

    with ThreadPoolExecutor(max_workers=5) as pool:
        obtained = dict(pool.map(get, path_to_arc))
    files = {name: obtained[key] for key, name in path_to_arc.items()}
    receipt = {
        "status": "PASS", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "branch": "wbw-lc-rec", "github_ref": ref, "delivery_commit": delivery, "original_run_commit": run,
        "server_implementation_commits": job_revisions, "file_content_checks": [
            {"commit": rev, "path": p, "member": path_to_arc[rev,p], **data_identity(value)}
            for (rev,p), value in obtained.items()],
    }
    files["FIXED_GITHUB_READBACK.json"] = json.dumps(receipt, ensure_ascii=False, indent=2).encode() + b"\n"
    files["REVIEW_NAVIGATION.md"] = (
        "# G3 Beauty final web review\n\n"
        "开始阅读 [最终结果](delivery/results/README.md) 和 [非作者报告](delivery/non_author_review.md)。\n\n"
        "ZIP 内的 [原超时证据](prior_timeout/) 与 [冻结源码](frozen/) 对应 GitHub 上的原正式执行。"
        " [GitHub 回读及路径映射](FIXED_GITHUB_READBACK.json)\n\n"
        f"[固定 GitHub 交付]({link()}/{delivery}/{snapshot_root}) · "
        f"[原超时固定 GitHub 提交]({link()}/{cfg['status_commit']}/{old_base})\n").encode()
    member_manifest = pack(output, files)
    zip_hash = sha(output)
    windows = None
    windows_error = None
    try:
        windows = verify_windows(output)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        windows_error = str(error)
    result = {"status": "PASS", "zip_path": str(output), "bytes": output.stat().st_size, "sha256": zip_hash,
              "members": len(files), "crc": "PASS", "member_paths_types_and_hashes": "PASS",
              "credential_pattern_scan": "PASS", "duplicates": 0, "github_exact_readback_count": len(obtained),
              "windows_path_verified": windows, "windows_verification_error": windows_error, "delivery_commit": delivery,
              "members_excluding_manifest": member_manifest}
    write(output.with_suffix(".github-readback.json"), receipt, exclusive=True)
    write(output.with_suffix(".zip.verification.json"), result, exclusive=True)
    with output.with_suffix(".zip.sha256").open("x") as stream:
        stream.write(f"{zip_hash}  {output.name}\n")
    print(json.dumps({k:v for k,v in result.items() if k != "members_excluding_manifest"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
