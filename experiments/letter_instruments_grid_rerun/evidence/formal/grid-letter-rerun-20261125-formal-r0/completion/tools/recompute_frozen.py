"""Recompute every saved formal result from original, hash-bound candidate rows."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as fp:
        return hashlib.file_digest(fp, "sha256").hexdigest()


class CompareRows:
    def __init__(self, stream):
        self.stream = stream
        self.count = 0

    def write(self, line):
        self.count += 1
        assert self.stream.readline() == line.encode(), self.count


def replay(source, terminal, inputs):
    experiment = source / "experiments/letter_instruments_grid_rerun"
    sys.path.insert(0, str(experiment))
    from downstream import replay_candidates, sequences, summarize, user_axis

    run = terminal / "output/run"
    request = read(terminal / "request.json")
    scope = request["scope"]
    assert scope["mode"] == "formal" and scope["reuse_job"] is None
    for name in ("Instruments.inter.json", "user_axis.json"):
        assert scope["input_hashes"][name] == sha(inputs / name)
    series = sequences(inputs / "Instruments.inter.json")
    axis = user_axis(inputs / "user_axis.json")
    codefile = run / "upstream/item_sids.jsonl"
    gate = read(run / "upstream/sid_gate.json")
    assert gate["item_sids_sha256"] == sha(codefile) and gate["mode"] == "formal"
    sid_rows = [json.loads(line) for line in codefile.read_bytes().splitlines()]
    assert [row["item_id"] for row in sid_rows] == list(range(9922))
    codes = [tuple(row["sid"]) for row in sid_rows]
    assert len(set(codes)) == 9922
    users = list(range(24772))
    checks = {}
    with TemporaryDirectory(prefix="grid-instruments-result-replay-") as tmp:
        output = Path(tmp)
        for condition in ("Normal", "Shuffle3a", "Shuffle4"):
            stage = run / "evaluation" / condition
            target = output / condition
            target.mkdir()
            candidate = stage / "candidates.jsonl"
            native = stage / "native_evaluation.json"
            original = read(native)
            assert sha(candidate) == original["candidate_sha256"]
            users_metrics = stage / "per_user_metrics.jsonl"
            with users_metrics.open("rb") as fp:
                compare = CompareRows(fp)
                sample = replay_candidates(candidate, condition, series, codes, axis, users, original["metrics"], compare)
                assert compare.count == len(users) and fp.read(1) == b""
            assert sample == read(stage / "offline_evaluation.json")
            for file in (candidate, native):
                (target / file.name).symlink_to(file)
            (target / "offline_evaluation.json").write_text(json.dumps(sample, allow_nan=False))
            checks[condition] = {"users": compare.count, "candidates": sample["counts"]["candidates"],
                                 "candidate_sha256": original["candidate_sha256"],
                                 "per_user_sha256": sha(users_metrics), "per_user_exact_bytes": True,
                                 "native_abs_delta": sample["native_abs_delta"]}
        summarize(SimpleNamespace(output=output, mode="formal"))
        summary = read(output / "summary.json")
        assert summary == read(run / "evaluation/summary.json")
        return {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
                "execution_commit": request["capture_version"], "job": request["job_id"],
                "conditions": checks, "recomputed_summary": summary,
                "summary_source_sha256": sha(run / "evaluation/summary.json"),
                "candidate_count_total": sum(c["candidates"] for c in checks.values())}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "terminal", "inputs"):
        cli.add_argument("--" + name, required=True, type=Path)
    args = cli.parse_args()
    print(json.dumps(replay(args.source.resolve(), args.terminal.resolve(), args.inputs.resolve()),
                     indent=2, allow_nan=False))
