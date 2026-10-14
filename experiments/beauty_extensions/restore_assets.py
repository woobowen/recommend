"""Recover only five indexed Beauty members from immutable archived chunks.

Offsets come from the preserved, independently reviewed TAR index. Full member
SHA-256 is mandatory; the archive is never rewritten or fully rehashed here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import os


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("plan", type=Path)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    root = Path(plan["output_root"])
    assert str(root).startswith("/home/bwwu/grid-audits/sid-extensions-p0/beauty-extensions/assets")
    root.mkdir(parents=True, exist_ok=True)
    assert shutil.disk_usage(root).free > 100 * 1024 ** 3
    batches = {b["batch"]: b for b in plan["batches"]}
    records = []
    for entry in plan["entries"]:
        original = Path(entry["destination"])
        output = root / entry["sha256"] / Path(entry["relative_path"]).name
        def digest(path):
            h = hashlib.sha256()
            with path.open("rb") as f:
                for block in iter(lambda:f.read(4 * 1024 ** 2), b""):
                    h.update(block)
            return h.hexdigest()
        if original.is_file() and original.stat().st_size == entry["bytes"] and digest(original) == entry["sha256"]:
            records.append({"entry": entry["relative_path"], "path": str(original), "sha256": entry["sha256"], "route": "reuse_existing_verified"})
            continue
        if output.exists():
            assert output.stat().st_size == entry["bytes"] and digest(output) == entry["sha256"]
        else:
            batch = batches[entry["batch"]]
            manifest = Path(batch["manifest_path"])
            assert digest(manifest) == batch["manifest_sha256"]
            chunks = sorted(Path(batch["chunk_directory"]).glob("*.tar.part"))
            offset = entry["tar_data_offset"]
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(output.suffix + ".partial")
            assert not temporary.exists()
            remaining = entry["bytes"]
            h = hashlib.sha256()
            with temporary.open("xb") as target:
                for chunk in chunks:
                    size = chunk.stat().st_size
                    if offset >= size:
                        offset -= size
                        continue
                    with chunk.open("rb") as source:
                        source.seek(offset); offset = 0
                        while remaining:
                            block = source.read(min(4 * 1024 ** 2, remaining))
                            if not block:
                                break
                            target.write(block); h.update(block); remaining -= len(block)
                    if not remaining:
                        break
            assert remaining == 0 and h.hexdigest() == entry["sha256"], "Indexed member did not match its bound hash"
            os.replace(temporary, output)
            output.chmod(0o444)
        records.append({"entry": entry["relative_path"], "path": str(output), "sha256": entry["sha256"], "route": "indexed_member_range_read_full_member_hash"})
    receipt = {"status": "PASS", "members": records, "archived_members_modified": False,
               "whole_archive_rehashed": False, "new_data_regeneration": False}
    target = Path(os.environ["GRID_JOB_DIR"]) / "output" / "assets.json"
    target.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
