"""Verify and package the fixed GitHub delivery for initial formal monitoring."""
import argparse
import concurrent.futures
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import zipfile

from verify_observations import captured

HERE = Path(__file__).resolve().parent
PREVIOUS_PACKAGER = HERE.parent / "single_gpu/final_closure/package_closure.py"
spec = importlib.util.spec_from_file_location("package_closure", PREVIOUS_PACKAGER)
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
PREFIX = HERE.relative_to(prior.REPO).as_posix() + "/"
PREP = "4075cf138fcbc2fd2e115001fd9fb5e8b63f6409"
CHECK_SCRIPT = "780040a4a3743d124004811495f9c45e6a3273ef"
EXTERNAL = {
    PREP: [prior.README, prior.PREFIX + "final_plan.json",
           prior.PREFIX + "final_plan_summary.json"],
    prior.PRIOR: [prior.PRIOR_PREFIX + "run_deployment.json"],
    prior.RUN: ["lc_rec/a0_beauty/" + name for name in (
        "production_control.py", "production_pipeline.py", "formal_train.py",
        "config/formal_g3_beauty_seed42.json")],
    CHECK_SCRIPT: [prior.PREFIX + "final_plan_check.py"],
}
EXTRA_SECRET_PATTERNS = [
    rb"xox[baprs]-[A-Za-z0-9-]{20,}", rb"AIza[0-9A-Za-z_-]{35}",
    rb"(?i)authorization:\s*bearer\s+[A-Za-z0-9._~+/=-]{20,}",
]


def scan(path, data):
    prior.check(not any(re.search(pattern, data)
                        for pattern in prior.SECRET_PATTERNS + EXTRA_SECRET_PATTERNS),
                "Secret pattern in " + path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-only", action="store_true")
    args = parser.parse_args()
    commit = prior.git("rev-parse", args.commit + "^{commit}", text=True).strip()
    output = args.output.resolve()
    prior.check(not output.exists(), "Preserve existing package")
    changes = prior.git("diff", "--name-only", PREP, commit, text=True).splitlines()
    prior.check(bool(changes) and all(path == prior.README or path.startswith(PREFIX)
                                    for path in changes), "Delivery changed files outside evidence")
    paths = prior.git("ls-tree", "-r", "--name-only", commit, "--", PREFIX,
                      prior.README, text=True).splitlines()
    files = {name: prior.blob(commit, name) for name in paths}
    prior.check(PREFIX + "README.md" in files and prior.README in files,
                "Missing evidence navigation")
    inputs = []
    for name, data in files.items():
        prior.check(prior.safe_name(name) and len(data) < 3 * 1024**2,
                    "Unsafe member or unexpected size: " + name)
        scan(name, data)
        if name.startswith(PREFIX + "observations/") and name.endswith(".json"):
            for part, copied in captured(json.loads(data)).items():
                scan(name + " / " + part, copied)
        inputs.append((commit, name, data))

    refs = [{"commit": sha, "path": path, "url": prior.BLOB + sha + "/" + path}
            for sha, names in EXTERNAL.items() for path in names]
    for ref in refs:
        inputs.append((ref["commit"], ref["path"], prior.blob(ref["commit"], ref["path"])))
    ref = None
    verified = []
    if not args.local_only:
        ref = prior.git("ls-remote", "origin", "refs/heads/wbw-lc-rec", text=True).strip()
        prior.check(ref.split() == [commit, "refs/heads/wbw-lc-rec"], "GitHub branch mismatch")
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            verified = list(pool.map(prior.verify, inputs))
    receipt = {
        "status": "LOCAL_EVIDENCE_ONLY" if args.local_only else "PASS",
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "delivery_commit": commit, "remote_ref": ref,
        "github_exact_readback": verified,
        "fixed_historical_references": refs,
    }
    files["FIXED_GITHUB_READBACK.json"] = prior.encoded(receipt)
    rows = [{"path": name, "bytes": len(data), "sha256": prior.sha(data)}
            for name, data in sorted(files.items())]
    files["MEMBER_SHA256.json"] = prior.encoded({"members": rows, "self_excluded": True})
    for name, data in files.items():
        scan(name, data)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(files.items()):
            archive.writestr(name, data)
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        prior.check(len(names) == len(set(names)) and all(prior.safe_name(name) for name in names),
                    "Duplicate or unsafe ZIP member")
        prior.check(archive.testzip() is None, "ZIP CRC check failed")
        prior.check(set(names) == set(files) and all(archive.read(name) == files[name] for name in names),
                    "ZIP bytes differ from checked evidence")
    archive_hash = prior.sha(output.read_bytes())
    output.with_suffix(".zip.sha256").write_text(archive_hash + "  " + output.name + "\n")
    result = {
        "zip_path": str(output), "bytes": output.stat().st_size, "sha256": archive_hash,
        "delivery_commit": commit, "members": len(files), "crc": "PASS",
        "duplicates": 0, "unsafe_paths": 0, "secret_pattern_scan": "PASS",
        "member_bytes_and_sha256": "PASS", "github_exact_readback_count": len(verified),
        "all_members": rows + [{"path": "MEMBER_SHA256.json",
                                "bytes": len(files["MEMBER_SHA256.json"]),
                                "sha256": prior.sha(files["MEMBER_SHA256.json"])}],
    }
    windows = subprocess.check_output(["wslpath", "-w", str(output)], text=True).strip()
    power = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
    ps = ("$ErrorActionPreference='Stop'; $p='" + windows.replace("'", "''") +
          "'; [PSCustomObject]@{Length=(Get-Item -LiteralPath $p).Length;" +
          "SHA256=(Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash}|ConvertTo-Json -Compress")
    try:
        actual = json.loads(subprocess.check_output(
            [power, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", ps],
            text=True, timeout=45).lstrip("\ufeff"))
        prior.check(actual["Length"] == result["bytes"] and actual["SHA256"].lower() == archive_hash,
                    "Windows size or SHA-256 differs")
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        result["windows_unc_read"] = {"status": "UNVERIFIED", "error": str(exc)}
    else:
        result.update(windows_path=windows, windows_unc_read={"status": "PASS", **actual})
    output.with_suffix(".zip.verification.json").write_bytes(prior.encoded(result))
    output.with_suffix(".github-readback.json").write_bytes(prior.encoded(receipt))
    print(prior.encoded({key: value for key, value in result.items() if key != "all_members"}).decode())


if __name__ == "__main__":
    main()
