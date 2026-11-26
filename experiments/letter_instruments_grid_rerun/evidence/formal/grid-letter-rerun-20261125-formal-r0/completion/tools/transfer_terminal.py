"""Reuse same-byte closed-stage files and transfer the new terminal receipts and products."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
LAUNCH = HERE.parent / "formal-launch"
sys.path.insert(0, str(HERE.parents[1] / "sid-extensions-p0/scripts"))
import transport


def read(path):
    return json.loads(path.read_text())


def main():
    q = read(HERE / "original-query.json")
    receipt = HERE / "terminal-export-receipt.json"
    exported = read(receipt) if receipt.is_file() else None
    root = str(Path(q["result_path"]).parent)
    assert q["status"] == "SUCCEEDED" and q["exit_code"] == 0 and q["cleanup_verified"]
    assert not (q["workgroup_live"] or q["worker_alive"] or q["worker_may_be_live"])
    if exported:
        assert exported["path"] == root + "/evidence.tar.gz"
    target = transport.Transport("new-server")
    meta = json.loads(target.ssh(shlex.join(["cat", root + "/export-manifest.json"])))
    entries = {f["path"]: f for f in meta["files"]}
    status_artifacts = {f["path"]: f for f in q["artifacts"]}
    assert all(entries[k] == v for k, v in status_artifacts.items())
    out = HERE / "terminal"
    out.mkdir(exist_ok=False)
    selected = []
    reused = {}
    retained = {}
    for name, expected in entries.items():
        prior = LAUNCH / "closed" / name
        if prior.is_file() and prior.stat().st_size == expected["bytes"]:
            assert transport.sha(prior) == expected["sha256"], name
            dest = out / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(prior, dest)
            reused[name] = {"previous_capture": f"formal-launch/closed/{name}", **expected}
        elif name.startswith("output/run/upstream/") and name.endswith(".ckpt"):
            extraction = read(LAUNCH / "quantizer-checkpoint/checkpoint_extraction.json")
            assert extraction["checkpoint_sha256"] == expected["sha256"]
            retained[name] = {**expected, "previous_lossless_state": "formal-launch/quantizer-checkpoint",
                              "source": root + "/" + name}
        else:
            selected.append(name)

    selected.append("export-manifest.json")
    tmp = HERE / "terminal-selected.tar.partial"
    cmd = shlex.join(["/usr/bin/python3", "-I", "-S", "-B", "-", root, json.dumps(selected)])
    argv = [transport.SSH, *target.options, "-T", target.host, cmd]
    source = (HERE / "stream_terminal.py").read_bytes()
    with tmp.open("xb") as stream:
        run = subprocess.run(argv, input=source, stdout=stream, stderr=subprocess.PIPE, timeout=180)
    (HERE / "terminal-transfer.stderr.log").write_bytes(run.stderr)
    run.check_returncode()
    with tarfile.open(tmp, "r:") as archive:
        members = archive.getmembers()
        assert len(members) == len(selected) and {m.name for m in members} == set(selected)
        for m in members:
            p = PurePosixPath(m.name)
            assert m.isfile() and not p.is_absolute() and ".." not in p.parts
        archive.extractall(out, filter="data")
    assert read(out / "export-manifest.json") == meta
    for name in set(selected + list(reused)) - {"export-manifest.json"}:
        record = entries[name]
        p = out / name
        assert p.stat().st_size == record["bytes"] and transport.sha(p) == record["sha256"], name
    transport.atomic(HERE / "terminal-transfer-verification.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "runner_export": exported, "source": root, "terminal_artifacts": len(status_artifacts),
        "selected_files_including_export_manifest": selected, "reused_byte_identical": reused,
        "server_retained_with_local_lossless_state": retained,
        "selected_archive_bytes": tmp.stat().st_size,
        "selected_archive_sha256": transport.sha(tmp),
        "verified_files": len(selected) + len(reused) - 1,
    })
    tmp.unlink()
    print(f"Terminal transfer verified: {len(selected)-1} fetched, {len(reused)} reused, {len(retained)} source-retained", flush=True)


if __name__ == "__main__":
    main()
