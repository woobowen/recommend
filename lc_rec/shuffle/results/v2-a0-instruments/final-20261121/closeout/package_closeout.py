"""Repackage the reviewed Instruments A0 ZIP with fixed closeout documentation."""
import argparse
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import stat
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED, ZipInfo

REVIEWED = "457c1c60853582693633cdae741e883f2bf2d643"
PREVIOUS_NOTE = "831d830db396242d50dd1b8a55aa8838f81523a4"
RUN = "e2f54ebd2d7709821de1071693f91644f5ed9a30"
CONTROLLER = "4f5758ffb578f3a8d3d06a54262f59d3f63db370"
ORIGINAL_SHA = "3dc6a752625137d25fe18ebac9685acbca2fa00706d30c8815bb327cd8257f48"
ORIGINAL_SIZE = 34_894_063
RESULTS = "lc_rec/shuffle/results/v2-a0-instruments/final-20261121"
NOTE = RESULTS + "/closeout"
DOCS = {RESULTS + "/README.md": "repo/" + RESULTS + "/README.md",
        "lc_rec/shuffle_v2/README.md": "repo/lc_rec/shuffle_v2/README.md",
        NOTE + "/README.md": "repo/" + NOTE + "/README.md",
        NOTE + "/offline-README.md": "README.md",
        NOTE + "/package_closeout.py": "closeout/package_closeout.py"}
SOURCE_VERSIONS = {"formal": RUN, "controller": CONTROLLER}
SECRET = re.compile(
    rb"(?i)(-----BEGIN [A-Z ]*PRIVATE KEY-----|gh[pousr]_[0-9a-z]{30,}|"
    rb"github_pat_[0-9a-z_]{60,}|hf_[A-Za-z0-9]{30,}|Bearer [A-Za-z0-9._-]{25,}|AKIA[A-Z0-9]{16})"
)
BINARY_SUFFIXES = {".npy", ".npz"}


def git(*args):
    return subprocess.check_output(["git", *args], timeout=30)


def json_bytes(body):
    return (json.dumps(body, ensure_ascii=False, indent=2) + "\n").encode()


def digest(stream):
    h = sha256()
    for block in iter(lambda: stream.read(1 << 20), b""):
        h.update(block)
    return h.hexdigest()


def inspect_members(archive):
    names = set()
    for item in archive.infolist():
        name = item.filename
        path = PurePosixPath(name)
        assert path.as_posix() == name and not path.is_absolute() and path.parts, name
        assert not any(c in name for c in "\\:\x00"), name
        assert all(p not in (".", "..") and not p.endswith((" ", ".")) for p in path.parts), name
        assert stat.S_ISREG(item.external_attr >> 16), name
        assert name.casefold() not in names, name
        names.add(name.casefold())


def zip_write(archive, name, stream):
    info = ZipInfo(name)
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    info.compress_type = ZIP_DEFLATED
    with archive.open(info, "w") as out:
        shutil.copyfileobj(stream, out)


def check_extracted(archive, manifest):
    members = manifest["members"]
    assert set(archive.namelist()) == set(members) | {"MANIFEST.json"}
    assert archive.testzip() is None
    inspect_members(archive)
    assert all(a["member"] in members for a in manifest["aliases"].values())
    with TemporaryDirectory(prefix="instruments-a0-closeout-") as tmp:
        root = Path(tmp)
        archive.extractall(root)
        for name, record in members.items():
            path = root / name
            assert path.stat().st_size == record["bytes"], name
            with path.open("rb") as stream:
                assert digest(stream) == record["sha256"], name
            if path.suffix not in BINARY_SUFFIXES:
                assert SECRET.search(path.read_bytes()) is None, name
        sources = json.loads((root / "source-map.json").read_bytes())
        for role, commit in SOURCE_VERSIONS.items():
            assert sources[role]["commit"] == commit
            assert set(sources[role]["path_to_zip_member"].values()) <= members.keys()
        guide = (root / "README.md").read_text()
        links = re.findall(r"\]\(([^ )]+)\)", guide)
        assert all((root / link).is_file() for link in links) and len(links) >= 30
        commands = [shlex.split(line) for line in guide.splitlines() if "python3 -B " in line]
        assert len(commands) == 2
        stats = root / ("repo/" + RESULTS)
        for leaf in ("metrics.csv", "arrays.npz", "summary.json", "paired.csv",
                     "verify_offline_statistics.py", "final-stats-independent.py"):
            assert (stats / leaf).is_file(), leaf
        assert len(list((root / "data").iterdir())) == 19
        for command in commands:
            offset = command.index("python3")
            script = command[offset + 2]
            args = command[offset + 3:]
            assert (root / script).is_file(), script
            for arg in args:
                if not arg.startswith("--") and arg != "verify":
                    assert (root / arg).exists(), arg
            entry = ["verify"] if args[0] == "verify" else []
            completed = subprocess.run([sys.executable, "-B", script, *entry, "--help"],
                                       cwd=root, capture_output=True, check=True, timeout=15)
            assert b"usage:" in completed.stdout, script
    return len(links)


def build(original, output, commit, readback):
    from io import BytesIO

    assert output.suffix == ".zip" and not output.exists()
    assert not original.is_symlink() and original.stat().st_size == ORIGINAL_SIZE
    with original.open("rb") as stream:
        assert digest(stream) == ORIGINAL_SHA
    assert re.fullmatch(r"[a-f0-9]{40}", commit)
    assert set(git("diff", "--name-only", REVIEWED, commit).decode().splitlines()) == DOCS.keys()
    published = json.loads(readback.read_bytes())
    assert published["remote_head"] == commit and published["branch"] == "wbw-lc-rec-shuffle-v2"
    overlays = {}
    for path, member in DOCS.items():
        payload = git("show", f"{commit}:{path}")
        assert published["files"][path]["sha256"] == sha256(payload).hexdigest(), path
        overlays[member] = (payload, {"commit": commit, "path": path})
    overlays["closeout/changes.patch"] = (git("diff", "--no-color", PREVIOUS_NOTE, commit),
                                            {"base": PREVIOUS_NOTE, "head": commit})
    overlays["closeout/github-readback.json"] = (json_bytes(published), {"source": "fixed GitHub readback"})
    partial = output.with_suffix(".zip.partial")
    with ZipFile(original) as old, ZipFile(partial, "x", compression=ZIP_DEFLATED, compresslevel=6) as new:
        inspect_members(old)
        original_manifest = old.read("MANIFEST.json")
        source = json.loads(original_manifest)
        assert set(source["members"]) | {"MANIFEST.json"} == set(old.namelist())
        assert not (set(overlays) & source["aliases"].keys())
        moved = {name: "history/reviewed/" + name
                 for name in sorted((set(overlays) & source["members"].keys()) | {"MANIFEST.json"})}
        assert all(a["member"] not in moved for a in source["aliases"].values())
        records = {}
        for name, record in source["members"].items():
            member = moved.get(name, name)
            assert old.getinfo(name).file_size == record["bytes"], name
            with old.open(name) as stream:
                zip_write(new, member, stream)
            records[member] = record
        overlays[moved["MANIFEST.json"]] = (original_manifest, {"source": "reviewed ZIP manifest"})
        original_rows = len(records)
        overlays["closeout/checks.json"] = (json_bytes({
            "reviewed_zip": {"bytes": ORIGINAL_SIZE, "sha256": ORIGINAL_SHA},
            "research_run": RUN, "controller": CONTROLLER, "reviewed_result": REVIEWED,
            "closeout_commit": commit, "original_members_preserved_by_sha256": original_rows,
            "research_gpu_budget_used": 0, "formal_statistics_regenerated": False,
            "verification": "Member SHA-256 after extraction, CRC, safe paths, credential patterns, deployment mappings, and root direct links. Existing validation CLI and inputs checked in the extracted copy. Research review reused."
        }), {"source": "closeout package checks"})
        for name, (data, origin) in overlays.items():
            assert name not in records
            zip_write(new, name, BytesIO(data))
            records[name] = {"bytes": len(data), "sha256": sha256(data).hexdigest(), "origin": origin}
        manifest = {"members": records, "aliases": source["aliases"], "relocated_originals": moved,
                    "reviewed_result": REVIEWED, "original_zip_sha256": ORIGINAL_SHA,
                    "closeout_commit": commit}
        zip_write(new, "MANIFEST.json", BytesIO(json_bytes(manifest)))
    with ZipFile(partial) as verified:
        links = check_extracted(verified, manifest)
    partial.rename(output)
    with output.open("rb") as stream:
        report = {"file": str(output), "bytes": output.stat().st_size, "sha256": digest(stream),
                  "members": len(records) + 1, "original_members_preserved": original_rows + 1,
                  "direct_root_links": links, "aliases": len(source["aliases"]),
                  "closeout_commit": commit, "crc_extraction_member_hash_secret_scan": "PASS",
                  "existing_offline_entry_points": "PASS_HELP_AND_INPUT_PATHS"}
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--readback", required=True, type=Path)
    args = parser.parse_args()
    build(args.original, args.output, args.commit, args.readback)


if __name__ == "__main__":
    main()
