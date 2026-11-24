"""Publish scoped observations and preserve exact prefixes from the T+30 read."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("raw", type=Path)
parser.add_argument("destination", type=Path)
args = parser.parse_args()
raw, dest = args.raw, args.destination
snapshots = dest / "snapshots"
snapshots.mkdir(parents=True, exist_ok=True)
captured = dest / "captured"
captured.mkdir(exist_ok=True)
(dest / ".gitattributes").write_text("captured/** -text\n")

def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")

def remove_foreign_pids(stdout, owned):
    lines = []
    for line in stdout.splitlines(keepends=True):
        uuid, pid, memory = line.split(", ", 2)
        lines.append(line if pid in owned else ", ".join((uuid, "[other PID withheld]", memory)))
    return "".join(lines)

def scope(snapshot):
    status = snapshot["status"]
    owned = {str(x["pid"]) for x in status["workgroup_observation"]["owned_live"]}
    owned.update(str(status[k]["pid"]) for k in ("worker", "supervisor") if status.get(k))
    apps = snapshot["gpu_compute_apps"]
    apps["stdout"] = remove_foreign_pids(apps["stdout"], owned)
    return snapshot

for name in ("request.json", "server-request.json", "submission.json", "admission.py",
             "precheck.py", "observe.py", "capture.py", "summarize.py", "materialize.py",
             "verify_phase.py", "readback.py", "runner-deployment.json", "deployed-paths.txt",
             "gridctl-query.json", "budget.py"):
    shutil.copyfile(raw / name, dest / name)

admission = json.loads((raw / "admission.json").read_text())
admission["compute_apps"] = remove_foreign_pids("\n".join(admission["compute_apps"]), set()).splitlines()
write(dest / "admission.json", admission)

start = scope(json.loads((raw / "started-readback.json").read_text()))
for info in start["files"].values():
    if info:
        info.pop("prefix", None)
write(snapshots / "start-reconnect.json", start)
write(dest / "delivery-status.json", scope(json.loads((raw / "delivery-status.json").read_text())))
for error in sorted(raw.glob("T??-query-error-?.json")):
    shutil.copyfile(error, snapshots / error.name)

for minute in range(0, 31, 5):
    name = f"T{minute:02d}.json"
    write(snapshots / name, scope(json.loads((raw / name).read_text())))

last = json.loads((snapshots / "T30.json").read_text())
manifest = []
for name, info in last["files"].items():
    if not info or not info.get("prefix"):
        continue
    prefix = info["prefix"]
    content = base64.b64decode(prefix["base64"], validate=True)
    assert len(content) == prefix["length"]
    assert prefix["complete"] == (info["bytes"] == len(content))
    destination = captured / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    manifest.append({k:info[k] for k in ("path", "bytes", "mtime_utc", "mtime_ns", "inode")} |
                    {"file": str(destination.relative_to(dest)), "source": "T30.json",
                     "prefix_bytes": len(content), "range": [0, len(content)],
                     "complete_at_open": prefix["complete"], "sha256": hashlib.sha256(content).hexdigest()})
write(captured / "capture-manifest.json", dict(
    method="open and fstat each file, read up to its observed byte count, then close; non-atomic across files",
    capture_started_utc=last["server_utc"], capture_ended_utc=last["server_end_utc"], files=manifest,
))
