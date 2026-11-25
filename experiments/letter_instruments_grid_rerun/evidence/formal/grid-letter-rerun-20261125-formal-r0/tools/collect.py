"""Use the existing Windows SSH profile for read-only formal evidence collection."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
SUPPORT = HERE.parents[1] / "sid-extensions-p0"
sys.path.insert(0, str(SUPPORT / "scripts"))
import transport

JOB = "grid-letter-rerun-20261125-formal-r0"
TARGET = transport.Transport("new-server")


def query(out):
    completed = subprocess.run([sys.executable, str(SUPPORT / "scripts/gridctl.py"), "--target",
                                "new-server", "query", JOB], capture_output=True, check=False)
    if completed.returncode != 0:
        (out / "query-error.txt").write_bytes(completed.stderr)
        completed.check_returncode()
    (out / "query.json").write_bytes(completed.stdout)


def archive(kind, out, command, code):
    tmp = out / "capture.tar.partial"
    argv = [transport.SSH, *TARGET.options, "-T", TARGET.host, command]
    with tmp.open("xb") as fp:
        done = subprocess.run(argv, input=code, stdout=fp, stderr=subprocess.PIPE, timeout=180, check=False)
    (out / "capture-stderr.log").write_bytes(done.stderr)
    done.check_returncode()
    with tarfile.open(tmp, mode="r:") as tf:
        files = tf.getmembers()
        names = [item.name for item in files]
        assert len(names) == len(set(names))
        for item in files:
            path = PurePosixPath(item.name)
            assert item.isfile() and not path.is_absolute() and ".." not in path.parts
        manifest = json.load(tf.extractfile("capture.json"))
        assert set(names) == {"capture.json"} | manifest["files"].keys()
        tf.extractall(out, filter="data")
    for name, record in manifest["files"].items():
        file = out / name
        with file.open("rb") as fp:
            assert file.stat().st_size == record["bytes"]
            assert hashlib.file_digest(fp, "sha256").hexdigest() == record["sha256"], name
    tmp.unlink()


def main():
    mode, label = sys.argv[1:]
    out = HERE / "observations" / label if mode == "observe" else HERE / label
    out.mkdir(parents=True, exist_ok=False)
    remote = "/usr/bin/python3 -I -S -B - " + shlex.quote(mode)
    code = (HERE / "observe_remote.py").read_bytes()
    if mode in {"observe", "cutoff"}:
        query(out)
    if mode == "observe":
        (out / "sample.json").write_text(TARGET.ssh(remote, code))
    else:
        archive(mode, out, remote, code)
    print(datetime.now(timezone.utc).isoformat(), out, flush=True)


if __name__ == "__main__":
    main()
