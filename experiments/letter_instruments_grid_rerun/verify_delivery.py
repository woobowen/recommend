"""Verify the published review bundle and optionally materialize every logical path."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
from zipfile import ZipFile


def stream_hash(stream):
    return hashlib.file_digest(stream, "sha256").hexdigest()


def safe(path):
    pp = PurePosixPath(path)
    return bool(path) and not pp.is_absolute() and ".." not in pp.parts and str(pp) == path and (
        "\\" not in path and ":" not in path and "\x00" not in path)


def verify_bundle(path, extract=None):
    with ZipFile(path) as z:
        entries = z.infolist()
        names = [entry.filename for entry in entries]
        assert len(names) == len(set(names)) and all(safe(name) for name in names)
        assert all(not entry.is_dir() and not stat.S_ISLNK(entry.external_attr >> 16) for entry in entries)
        assert z.testzip() is None
        manifest = json.loads(z.read("MANIFEST.json"))
        assert set(names) == set(manifest["stored"]) | {"MANIFEST.json"}
        assert set(manifest["aliases"]).isdisjoint(names)
        assert all(safe(alias) and target in manifest["stored"] for alias, target in manifest["aliases"].items())
        for entry in entries:
            if entry.filename == "MANIFEST.json":
                continue
            expected = manifest["stored"][entry.filename]
            with z.open(entry) as stream:
                assert stream_hash(stream) == expected["sha256"], entry.filename
            assert entry.file_size == expected["bytes"]
        assert len({entry["sha256"] for entry in manifest["stored"].values()}) == len(manifest["stored"])
        for required in (
            "TASK_PROMPT.md", "rules/AGENTS.md", "identity/letter/Instruments.item2id",
            "identity/letter/Instruments.inter.json", "identity/grid/item_embeddings.pt",
            "identity/common/item_identity.jsonl", "identity/common/user_axis.json",
            "snapshot/VERSION", "formal-plan/downstream/effective_config.yaml",
            "formal-plan/upstream/quantizer_train/resolved.yaml", "preview/status.json",
            "preview/output/run/upstream/semantic_ids.pt", "preview/output/run/training/best.json",
            "preview/output/run/evaluation/summary.json", "nonauthor/NONAUTHOR_REVIEW.md",
            "release/remote-readback.json", "formal-not-submitted.request.json",
        ):
            assert required in manifest["stored"] or required in manifest["aliases"], required

        def read_bytes(name):
            return z.read(manifest["aliases"].get(name, name))

        def read(name):
            return json.loads(read_bytes(name))

        terminal = read("preview/status.json")
        assert terminal["exit_code"] == 0 and terminal["status"] == "SUCCEEDED"
        assert b"formal" not in read_bytes("release/runner-prefix-readback.txt")
        best = read("preview/output/run/training/best.json")
        analysis = read("preview/output/run/evaluation/summary.json")
        assert best["sha256"] == analysis["best_sha256"] and best["mode"] == analysis["mode"] == "preview"
        best_path = "preview/output/run/training/" + best["file"]
        entry = manifest["stored"][manifest["aliases"].get(best_path, best_path)]
        assert best["sha256"] == entry["sha256"]
        for mode in ("Normal", "Shuffle3a", "Shuffle4"):
            directory = "preview/output/run/evaluation/" + mode
            native = read(directory + "/native_evaluation.json")
            assert native["reload"]["strict"] and native["reload"]["state_sha256"] == best["state_sha256"]
            assert native["reload"]["checkpoint_sha256"] == best["sha256"]
            raw = read_bytes(directory + "/candidates.jsonl")
            assert hashlib.sha256(raw).hexdigest() == native["candidate_sha256"]
            candidates = [json.loads(line) for line in raw.splitlines()]
            assert len(candidates) == 20 and all(len(c["ids"]) == 10 and c["mode"] == mode for c in candidates)
        assert read_bytes("preview/output/run/evaluation/Normal/candidates.jsonl") == read_bytes(
            "preview/output/run/evaluation/Normal/native_shadow.jsonl")
        sid = read("preview/output/run/upstream/sid_gate.json")
        assert best["mode"] == "preview" and sid["mode"] == "preview"
        for name in ("formal-not-submitted.request.json",):
            request = read(name)
            assert request["scope"]["mode"] == "formal" and request["scope"]["upstream_lightning_steps"] == 30
            assert request["capture_version"] == read_bytes("snapshot/VERSION").decode().strip()
        if extract:
            extract.mkdir(parents=True, exist_ok=False)
            for name in names:
                destination = extract / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                with z.open(name) as src, destination.open("xb") as dst:
                    shutil.copyfileobj(src, dst)
            for alias, target in manifest["aliases"].items():
                destination = extract / alias
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(extract / target, destination)
        sizes = Counter()
        for entry in entries:
            sizes[entry.filename.split("/")[0]] += entry.compress_size
        with path.open("rb") as stream:
            digest = stream_hash(stream)
        return {"zip_sha256": digest, "bytes": path.stat().st_size, "stored": len(entries),
                "aliases": len(manifest["aliases"]), "crc_and_sha256_verified": True,
                "compressed_bytes_by_prefix": dict(sizes)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("zip", type=Path)
    p.add_argument("--extract", type=Path)
    args = p.parse_args()
    print(json.dumps(verify_bundle(args.zip, args.extract), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
