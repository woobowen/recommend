"""Stream selected regular files from the verified-terminal job."""
import json
from pathlib import Path
import sys
import tarfile

job = Path(sys.argv[1]).resolve(strict=True)
status = json.loads((job / "status.json").read_text())
assert status["status"] == "SUCCEEDED" and status["exit_code"] == 0
assert status["cleanup"]["verified"] and not status["worker_may_be_live"]
manifest = json.loads((job / "export-manifest.json").read_text())
inventory = {f["path"]: f for f in manifest["files"]}
with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as stream:
    for name in json.loads(sys.argv[2]):
        path = job / name
        assert name == "export-manifest.json" or name in inventory, name
        assert path.resolve(strict=True).is_relative_to(job) and path.is_file() and not path.is_symlink()
        if name != "export-manifest.json":
            assert path.stat().st_size == inventory[name]["bytes"], name
        stream.add(path, arcname=name, recursive=False)
