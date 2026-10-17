"""Restore only hash-bound launch inputs from the preserved server archives.

No original/archive file is modified. Archive payloads are read through a
seekable concatenation; only the explicitly listed regular members are copied.
"""
from __future__ import annotations

import argparse
import bisect
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time


class ChunkReader(io.RawIOBase):
    def __init__(self, paths):
        self.paths = paths
        self.ends = []
        size = 0
        for path in paths:
            size += path.stat().st_size
            self.ends.append(size)
        self.size, self.position = size, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=os.SEEK_SET):
        base = {os.SEEK_SET: 0, os.SEEK_CUR: self.position, os.SEEK_END: self.size}[whence]
        position = base + offset
        if position < 0:
            raise ValueError("Negative archive seek")
        self.position = position
        return position

    def read(self, size=-1):
        if size < 0:
            size = self.size - self.position
        data = []
        while size > 0 and self.position < self.size:
            index = bisect.bisect_right(self.ends, self.position)
            begin = self.ends[index - 1] if index else 0
            count = min(size, self.ends[index] - self.position)
            with self.paths[index].open("rb") as stream:
                stream.seek(self.position - begin)
                block = stream.read(count)
            if len(block) != count:
                raise IOError("Truncated preserved archive chunk")
            data.append(block)
            self.position += count
            size -= count
        return b"".join(data)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verified(path, record):
    return path.is_file() and not path.is_symlink() and path.stat().st_size == record["bytes"] and sha(path) == record["sha256"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, default=Path(__file__).with_name("a2_sources.json"))
    parser.add_argument("--archive-root", type=Path, default=Path("/data/bwwu/archives/wmh-a800-audit-20261010"))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    records = json.loads(args.sources.read_text())["original_assets"]
    root = args.output_root
    if root.is_symlink():
        raise ValueError("Output root must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    needed = sum(r["bytes"] for r in records if not (root / r["archive_path"]).exists())
    if shutil.disk_usage(root).free < needed + 10 * 1024**3:
        raise RuntimeError("Insufficient launch asset space plus 10GiB reserve")
    lock = (root / ".restore.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    batches = {}
    for r in records:
        if r["type"] != "regular" or PurePosixPath(r["archive_path"]).is_absolute() or ".." in PurePosixPath(r["archive_path"]).parts:
            raise ValueError("Unsafe source member")
        batches.setdefault((r["capture_version"], r["batch_id"]), []).append(r)
    receipt = []
    for (version, batch), selected in sorted(batches.items()):
        manifest = args.archive_root / "metadata" / f"{version}-{batch}-capture-manifest.json"
        assert all(sha(manifest) == r["manifest_sha256"] for r in selected), "Source manifest changed"
        bound = json.loads(manifest.read_text())
        preserved = {q["archive_path"]: q for q in bound["preserved"]}
        for r in selected:
            assert all(preserved[r["archive_path"]][key] == r[key] for key in ("sha256", "bytes", "type"))
        chunk_root = args.archive_root / version / "batches" / batch / "chunks"
        paths = []
        for part in sorted(bound["parts"], key=lambda q: q["index"]):
            p = chunk_root / Path(part["path"]).name
            assert p.is_file() and not p.is_symlink() and p.stat().st_size == part["bytes"], str(p)
            paths.append(p)
        with tarfile.open(fileobj=ChunkReader(paths), mode="r:") as archive:
            for r in selected:
                target = root / r["archive_path"]
                assert root.resolve() in target.resolve().parents
                if target.exists():
                    if not verified(target, r):
                        raise RuntimeError("Existing target identity mismatch; retained: " + str(target))
                    receipt.append({**r, "path": str(target), "reused": True})
                    continue
                member = archive.getmember(r["archive_path"])
                assert member.isfile() and member.size == r["bytes"]
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + ".partial")
                with archive.extractfile(member) as source, temporary.open("xb") as dest:
                    shutil.copyfileobj(source, dest, 4 * 1024 * 1024)
                if not verified(temporary, r):
                    raise RuntimeError("Restored bytes mismatch; partial retained")
                os.chmod(temporary, 0o444)
                os.replace(temporary, target)
                receipt.append({**r, "path": str(target), "reused": False})
        if time.monotonic() - started > 21600:
            raise TimeoutError("CPU preparation hard time limit")
    result = {"status": "SUCCEEDED", "assets": receipt, "restored_logical_bytes": sum(r["bytes"] for r in receipt),
              "originals_modified": False, "full_archives_materialized": False,
              "member_content_sha256_verified": True, "archive_chunks_freshly_rehashed": False,
              "elapsed_seconds": time.monotonic() - started}
    (root / "asset-receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
