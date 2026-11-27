"""Transactional prediction storage for the fixed Beauty five-condition axis."""

from contextlib import ExitStack, contextmanager
import fcntl
import json
import math
import os
import re
from pathlib import Path
import shutil
from uuid import uuid4

from lc_rec.beauty_history_attention.identity import require, sha
from lc_rec.shuffle_v2.core import MODES, canonical_sha256

CONDITIONS = ("normal", *MODES)
PREDICTIONS = "predictions.rank0.jsonl"
SPEC = "generation-spec.json"
CHUNK_USERS = 32


def read_json(path):
    require(path.is_file() and not path.is_symlink(), f"missing regular file: {path}")
    return json.loads(path.read_bytes())


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _record(path):
    require(path.is_file() and not path.is_symlink(), f"missing regular file: {path}")
    return {"bytes": path.stat().st_size, "sha256": sha(path)}


def _check_directory(path):
    require(path.is_dir() and not path.is_symlink(), f"expected directory: {path}")


def atomically_write_json(path, value):
    pending = path.with_name(f".{path.name}.{uuid4().hex}")
    write_json(pending, value)
    os.replace(pending, path)
    fsync_dir(path.parent)


@contextmanager
def open_store(root, binding, assignment, chunk_users, *, metadata=None):
    """Take the writer lock before touching any resumable run state."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    _check_directory(root)
    fd = os.open(root / ".prediction.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "rb+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another prediction attempt holds the output lock") from error
        yield PredictionStore(root, binding, assignment, chunk_users, metadata=metadata)


def verify_final(root, binding=None):
    """Validate the atomic merged publication against its specification and all bytes."""
    root = Path(root)
    final = root / "final"
    _check_directory(root)
    _check_directory(final)
    spec = read_json(root / SPEC)
    require(spec.get("schema") == 1, "prediction specification schema mismatch")
    require(binding is None or binding == spec["binding"], "prediction binding mismatch")
    receipt = read_json(final / "completion.json")
    require(receipt["schema"] == 1 and receipt["spec_sha256"] == sha(root / SPEC),
            "final publication specification mismatch")
    expected = {"generation.json", *(f"{mode}/{name}" for mode in CONDITIONS
                                     for name in (PREDICTIONS, "rank0.json"))}
    require(set(receipt["files"]) == expected, "final prediction receipt member mismatch")
    require({p.name for p in final.iterdir()} == {"completion.json", "generation.json", *CONDITIONS},
            "unexpected final prediction member")
    for mode in CONDITIONS:
        _check_directory(final / mode)
        require({p.name for p in (final / mode).iterdir()} == {PREDICTIONS, "rank0.json"},
                "unexpected prediction condition member")
    for member, identity in receipt["files"].items():
        require(_record(final / member) == identity, f"published prediction changed: {member}")
    report = read_json(final / "generation.json")
    runtime_sha = sha(root / "generation-runtime.json")
    require(receipt["runtime_sha256"] == runtime_sha, "final runtime certificate changed")
    require(report["prediction_binding"] == spec["binding"], "final report binding mismatch")
    require(report["assignment_sha256"] == canonical_sha256(spec["assignment"]),
            "final report assignment mismatch")
    require(report["assigned_users"] == len(spec["assignment"]), "final report population mismatch")
    return report


def _empty_totals():
    return {mode: {"seconds": 0., "identical_to_normal_rows": 0} for mode in CONDITIONS}


def _validate_user(rows, row_idx, validate, totals):
    normal = None
    for mode, row in rows.items():
        require(type(row["row_idx"]) is int and row["row_idx"] == row_idx,
                "prediction transaction contains a wrong or duplicate row")
        validate(row, row_idx, mode)
        seconds = row["generation_seconds"]
        require(type(seconds) in (int, float) and seconds >= 0 and math.isfinite(seconds),
                "invalid prediction time")
        totals[mode]["seconds"] += seconds
        pair = row["candidates"], row["sorted_generation_order"]
        if mode == "normal":
            normal = pair
        elif not row["shuffle"]["input_changed"]:
            require(pair == normal, "identity prediction differs from B Normal")
            totals[mode]["identical_to_normal_rows"] += 1


class PredictionStore:
    """Commit every contiguous chunk only after all five condition streams are durable."""

    def __init__(self, root, binding, assignment, chunk_users, *, metadata):
        require(assignment and all(type(i) is int for i in assignment)
                and len(set(assignment)) == len(assignment), "bad prediction axis")
        require(type(chunk_users) is int and 1 <= chunk_users <= CHUNK_USERS, "bad checkpoint granularity")
        self.root = root
        self.binding = binding
        self.ranges = [assignment[i:i + chunk_users] for i in range(0, len(assignment), chunk_users)]
        spec = {"schema": 1, "binding": binding, "assignment": assignment, "chunk_users": chunk_users}
        spec_path = root / SPEC
        if os.path.lexists(spec_path):
            require(read_json(spec_path) == spec, "existing predictions are from a different run")
        else:
            # A killed initial write can leave its private, never-committed temp file.
            for item in root.iterdir():
                require(item.name == ".prediction.lock" or (
                    re.fullmatch(r"\.generation-spec\.json\.[0-9a-f]{32}", item.name)
                    and item.is_file() and not item.is_symlink()),
                    "refusing to adopt nonempty output without its specification")
            atomically_write_json(spec_path, spec)
        self.spec_sha = sha(spec_path)
        self.chunks = root / "chunks"
        self.attempts = root / "attempts"
        for p in (self.chunks, self.attempts):
            p.mkdir(exist_ok=True)
            _check_directory(p)
        if os.path.lexists(root / "final"):
            self.completed = verify_final(root, binding)
            self.attempt = None
            return
        self.completed = None
        self.attempt = self.attempts / f"attempt-{uuid4().hex}"
        self.attempt.mkdir()
        fsync_dir(self.attempts)
        if metadata is not None:
            self.note(metadata)

    def note(self, metadata):
        atomically_write_json(self.attempt / "attempt.json", metadata)

    def bind_runtime(self, runtime):
        """Match effective generation settings and backend identity across attempts."""
        path = self.root / "generation-runtime.json"
        if os.path.lexists(path):
            require(read_json(path) == runtime, "effective prediction runtime changed")
        else:
            require(not any(self.chunks.iterdir()), "runtime identity absent for existing chunks")
            atomically_write_json(path, runtime)

    def _chunk_name(self, number):
        return f"{number:06d}"

    def _check_chunk(self, number, validate):
        chunk = self.chunks / self._chunk_name(number)
        _check_directory(chunk)
        meta = read_json(chunk / "chunk.json")
        expected = self.ranges[number]
        require(meta["schema"] == 1 and meta["spec_sha256"] == self.spec_sha
                and meta["ordinal"] == number and meta["assignment"] == expected,
                f"chunk identity does not match assignment: {chunk}")
        require({p.name for p in chunk.iterdir()} == {"chunk.json", *(f"{mode}.jsonl" for mode in CONDITIONS)},
                f"unexpected prediction chunk members: {chunk}")
        for mode in CONDITIONS:
            require(_record(chunk / f"{mode}.jsonl") == meta["files"][mode],
                    f"chunk prediction changed: {chunk}/{mode}")
        totals = _empty_totals()
        with ExitStack() as stack:
            readers = [stack.enter_context((chunk / f"{mode}.jsonl").open("rb")) for mode in CONDITIONS]
            for row_idx, *raw_rows in zip(expected, *readers, strict=True):
                require(all(raw.endswith(b"\n") for raw in raw_rows),
                        f"unterminated prediction row: {chunk}/{row_idx}")
                rows = {mode: json.loads(raw) for mode, raw in zip(CONDITIONS, raw_rows, strict=True)}
                _validate_user(rows, row_idx, validate, totals)
        require(meta["totals"] == totals, f"prediction chunk totals disagree: {chunk}")
        return meta

    def committed(self, validate):
        names = sorted(p.name for p in self.chunks.iterdir())
        require(len(names) <= len(self.ranges)
                and names == [self._chunk_name(i) for i in range(len(names))],
                "committed prediction chunks do not form an exact prefix")
        return [self._check_chunk(i, validate) for i in range(len(names))]

    @contextmanager
    def chunk(self, number, validate):
        private = self.attempt / f"chunk-{self._chunk_name(number)}"
        target = self.chunks / self._chunk_name(number)
        require(not os.path.lexists(target), "refusing to replace committed predictions")
        private.mkdir()
        fsync_dir(self.attempt)
        expected = self.ranges[number]
        totals = _empty_totals()
        rows_seen = 0

        with ExitStack() as stack:
            streams = {
                mode: stack.enter_context((private / f"{mode}.jsonl").open("x", encoding="utf-8"))
                for mode in CONDITIONS
            }

            def write_user(row_idx, rows):
                nonlocal rows_seen
                require(rows_seen < len(expected) and expected[rows_seen] == row_idx and tuple(rows) == CONDITIONS,
                        "prediction transaction must include all conditions on the next assigned row")
                _validate_user(rows, row_idx, validate, totals)
                for mode, row in rows.items():
                    streams[mode].write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                rows_seen += 1

            yield write_user
            require(rows_seen == len(expected), "prediction transaction is incomplete")
            for stream in streams.values():
                stream.flush()
                os.fsync(stream.fileno())
        write_json(private / "chunk.json", {
            "schema": 1, "ordinal": number, "assignment": expected, "spec_sha256": self.spec_sha,
            "files": {mode: _record(private / f"{mode}.jsonl") for mode in CONDITIONS}, "totals": totals,
        })
        fsync_dir(private)
        os.rename(private, target)
        fsync_dir(self.chunks)
        fsync_dir(self.attempt)

    def publish(self, report, per_condition, *, clock=None):
        target = self.root / "final"
        require(not os.path.lexists(target), "refusing to overwrite final predictions")
        require(sorted(p.name for p in self.chunks.iterdir())
                == [self._chunk_name(i) for i in range(len(self.ranges))],
                "cannot publish an incomplete prediction run")
        private = self.attempt / "final.pending"
        private.mkdir()
        files = {}
        for mode in CONDITIONS:
            folder = private / mode
            folder.mkdir()
            with (folder / PREDICTIONS).open("xb") as joined:
                for i in range(len(self.ranges)):
                    with (self.chunks / self._chunk_name(i) / f"{mode}.jsonl").open("rb") as stream:
                        shutil.copyfileobj(stream, joined)
                joined.flush()
                os.fsync(joined.fileno())
            write_json(folder / "rank0.json", per_condition[mode])
            for name in (PREDICTIONS, "rank0.json"):
                files[f"{mode}/{name}"] = _record(folder / name)
            fsync_dir(folder)
        if clock is not None:
            report["prepublication_elapsed_seconds"] = clock()
        write_json(private / "generation.json", report)
        files["generation.json"] = _record(private / "generation.json")
        write_json(private / "completion.json", {
            "schema": 1, "spec_sha256": self.spec_sha, "files": files,
            "runtime_sha256": sha(self.root / "generation-runtime.json"),
        })
        fsync_dir(private)
        os.rename(private, target)
        fsync_dir(self.root)
        fsync_dir(self.attempt)
