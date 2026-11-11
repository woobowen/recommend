"""Build and fully verify one lightweight review artifact from a fixed Git tree."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
import zipfile

REPO = Path(__file__).resolve().parents[3]
PREFIX = "lc_rec/qwen_six_task_20261111/"
BASELINE = "1cd37ec681bdb3534108aaaee554dbaa9158a12f"


def digest(content):
    return hashlib.sha256(content).hexdigest()


def scan(name, content):
    path = PurePosixPath(name)
    assert not path.is_absolute() and ".." not in path.parts and "\\" not in name
    assert not name.endswith((".whl", ".bin", ".safetensors", ".pt", ".pth", ".pem", ".key", ".p12", ".tar", ".zip"))
    assert len(content) < 12 * 1024 ** 2, "Unexpected large evidence member: " + name
    text = content.decode("utf-8", "replace")
    patterns = [r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", r"\bghp_[A-Za-z0-9]{25,}\b",
                r"\bgithub_pat_[A-Za-z0-9_]{30,}\b", r"\bhf_[A-Za-z0-9]{25,}\b", r"\bAKIA[A-Z0-9]{16}\b"]
    for pattern in patterns:
        assert not re.search(pattern, text), "Secret-like payload in " + name


def git_bytes(commit, path):
    return subprocess.check_output(["git", "show", commit + ":" + path], cwd=REPO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument("--receipts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--windows-copy", required=True, type=Path)
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "--verify", args.commit + "^{commit}"], cwd=REPO, text=True).strip()
    tree = subprocess.check_output(["git", "ls-tree", "-r", "--full-tree", commit, "--", PREFIX], cwd=REPO, text=True).splitlines()
    members = {}
    sources = {}

    def add(name, content, source):
        assert name not in members
        scan(name, content)
        members[name] = content
        sources[name] = source

    for line in tree:
        metadata, name = line.split("\t", 1)
        mode, kind, blob = metadata.split()
        assert mode in ("100644", "100755") and kind == "blob"
        assert name.startswith(PREFIX)
        relative = name[len(PREFIX):]
        add(relative, git_bytes(commit, name), {"git_path": name, "git_blob": blob, "commit": commit})
    assert "REVIEW_ENTRY.md" in members
    references = ["lc_rec/vendor/original/" + name for name in [
        "lora_finetune.py", "finetune.py", "utils.py", "data.py", "collator.py", "prompt.py", "run.sh", "test.py", "test_ddp.py", "evaluate.py"]]
    references += ["lc_rec/six_task_lora/scripts/data_support.py", "lc_rec/evidence/cpu_audit/output/real_data_audit.json"]
    for path in references:
        add("reference_source/" + path, git_bytes(BASELINE, path), {"git_path": path, "commit": BASELINE, "frozen_reference": True})
    for path, reference_commit, name in [
        ("lc_rec/six_task_lora/REVIEW_ENTRY.md", "cceb550a44835b359c21596507f31dc7765da2d4", "PRIOR_LC_REC_REVIEW.md"),
        ("lc_rec/qwen_preflight_20261110/REVIEW_ENTRY.md", BASELINE, "PRIOR_QWEN_ENGINEERING_REVIEW.md"),
    ]:
        add("reference_source/" + name, git_bytes(reference_commit, path), {"git_path": path, "commit": reference_commit})
    source_diff = subprocess.check_output(["git", "diff", "--no-ext-diff", "--binary", BASELINE, commit, "--", PREFIX + "scripts", PREFIX + "config"], cwd=REPO)
    add("reference_source/implementation_from_previous_delivery.diff", source_diff, {"baseline": BASELINE, "delivery": commit, "scope": "New bounded probe source/config only"})
    assert args.receipts.is_dir()
    for path in sorted(args.receipts.rglob("*")):
        if not path.is_file():
            continue
        assert not path.is_symlink()
        add("final_receipts/" + str(path.relative_to(args.receipts)), path.read_bytes(), {"external_receipt": str(path)})
    assert sum(len(content) for content in members.values()) < 96 * 1024 ** 2
    manifest = {"delivery_commit": commit, "manifest_self_hash": "Outside the archive, together with ZIP hash",
                "files": [{"path": name, "bytes": len(content), "sha256": digest(content), "source": sources[name]}
                          for name, content in sorted(members.items())]}
    members["SHA256_MANIFEST.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assert not args.output.exists(), "Do not overwrite a delivered ZIP"
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, content in sorted(members.items()):
            archive.writestr(name, content)
    with zipfile.ZipFile(args.output) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names)) and set(names) == set(members)
        assert archive.testzip() is None and names.count("REVIEW_ENTRY.md") == 1
        with tempfile.TemporaryDirectory(prefix="lc-qwen6-review-verify-") as temporary:
            root = Path(temporary)
            for name in names:
                information = archive.getinfo(name)
                assert ((information.external_attr >> 16) & 0o170000) != 0o120000
                scan(name, archive.read(name))
                assert information.file_size == len(members[name])
            archive.extractall(root)
            for row in manifest["files"]:
                content = (root / row["path"]).read_bytes()
                assert len(content) == row["bytes"] and digest(content) == row["sha256"]
            assert (root / "SHA256_MANIFEST.json").read_bytes() == members["SHA256_MANIFEST.json"]
    zip_hash = digest(args.output.read_bytes())
    args.windows_copy.parent.mkdir(parents=True, exist_ok=True)
    assert not args.windows_copy.exists()
    shutil.copyfile(args.output, args.windows_copy)
    assert digest(args.windows_copy.read_bytes()) == zip_hash
    windows_path = subprocess.check_output(["wslpath", "-w", str(args.windows_copy)], text=True).strip()
    command = "$lcQwenZip='%s';$lcQwenItem=Get-Item -LiteralPath $lcQwenZip;[pscustomobject]@{bytes=$lcQwenItem.Length;sha256=(Get-FileHash -Algorithm SHA256 -LiteralPath $lcQwenZip).Hash}|ConvertTo-Json -Compress" % windows_path.replace("'", "''")
    actual = json.loads(subprocess.check_output(["/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command], text=True))
    assert actual["bytes"] == args.output.stat().st_size and actual["sha256"].lower() == zip_hash
    receipt = {"delivery_commit": commit, "wsl_path": str(args.output.resolve()), "windows_copy_wsl_path": str(args.windows_copy.resolve()),
               "windows_path": windows_path, "bytes": args.output.stat().st_size, "sha256": zip_hash, "members": len(members),
               "crc_pass": True, "full_unpack_all_member_size_sha_pass": True, "duplicate_and_path_safety_pass": True,
               "secret_pattern_scan_pass": True, "windows_actual_file_hash": actual,
               "manifest_sha256": digest(members["SHA256_MANIFEST.json"])}
    args.output.with_suffix(args.output.suffix + ".receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(zip_hash + "  " + args.output.name + "\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
