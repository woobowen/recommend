"""CPU tests for durable process interruption, handoff and B-only boundaries."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from lc_rec.beauty_history_attention.evaluate import preflight_paired, preflight_rows
from lc_rec.beauty_history_attention.prediction_store import (
    CONDITIONS, PREDICTIONS, SPEC, open_store, verify_final,
)
from lc_rec.shuffle_v2.core import canonical_sha256

BINDING = {"run_commit": "1" * 40, "backend": "eager"}
AXIS = [5, 2, 14, 6]
RUNTIME = {"backend": "eager", "generation": {"num_beams": 20}}


def validate(row, i, mode):
    if row["condition"] != mode or row["row_idx"] != i:
        raise ValueError("wrong condition or row")


def row(i, mode):
    return dict(row_idx=i, condition=mode, generation_seconds=0.125, candidates=[i],
                sorted_generation_order=[0], shuffle={"input_changed": False})


def write_rows(write_user, assigned):
    for i in assigned:
        write_user(i, {mode: row(i, mode) for mode in CONDITIONS})


def report():
    return dict(prediction_binding=BINDING, assignment_sha256=canonical_sha256(AXIS),
                assigned_users=len(AXIS))


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "predictions"

    def store(self, binding=BINDING):
        return open_store(self.root, binding, AXIS, 2)

    def spawn(self, code):
        return subprocess.Popen([sys.executable, "-B", "-c", code, str(self.root)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def test_process_exit_without_callback_preserves_private_files_and_reuses_only_committed(self):
        process = self.spawn("""
import os, sys
from pathlib import Path
from lc_rec.beauty_history_attention.test_prediction_store import BINDING, AXIS, RUNTIME, validate, write_rows
from lc_rec.beauty_history_attention.prediction_store import open_store
with open_store(Path(sys.argv[1]), BINDING, AXIS, 2) as store:
    store.bind_runtime(RUNTIME)
    with store.chunk(0, validate) as write_user:
        write_rows(write_user, AXIS[:2])
    with store.chunk(1, validate) as write_user:
        write_rows(write_user, AXIS[2:3])
        with (store.attempt / 'chunk-000001' / 'normal.jsonl').open('ab') as stream:
            stream.write(b'{"row_idx":')
            stream.flush()
            os.fsync(stream.fileno())
        os._exit(48)
""")
        _, err = process.communicate(timeout=10)
        self.assertEqual(process.returncode, 48, err)
        prior = (self.root / "chunks" / "000000" / "normal.jsonl").read_bytes()
        partial_paths = list((self.root / "attempts").glob("*/chunk-000001/normal.jsonl"))
        self.assertEqual(len(partial_paths), 1)
        self.assertEqual(partial_paths[0].read_bytes(), b'{"row_idx":')
        with self.store() as store:
            store.bind_runtime(RUNTIME)
            self.assertEqual([chunk["assignment"] for chunk in store.committed(validate)], [[5, 2]])
            with self.assertRaisesRegex(ValueError, "replace committed"):
                with store.chunk(0, validate):
                    self.fail("overwriting durable chunk")
            with store.chunk(1, validate) as write_user:
                write_rows(write_user, AXIS[2:])
            self.assertEqual([chunk["assignment"] for chunk in store.committed(validate)], [[5, 2], [14, 6]])
            store.publish(report(), {mode: {"condition": mode} for mode in CONDITIONS})
        self.assertEqual((self.root / "chunks" / "000000" / "normal.jsonl").read_bytes(), prior)
        self.assertTrue(partial_paths[0].is_file())
        with self.store() as store:
            self.assertEqual(store.completed, report())
        for mode in CONDITIONS:
            output = self.root / "final" / mode / PREDICTIONS
            self.assertEqual([json.loads(line)["row_idx"] for line in output.read_text().splitlines()], AXIS)
        wrong = self.root / "final" / "normal" / PREDICTIONS
        wrong.write_text(wrong.read_text().replace('"row_idx": 5', '"row_idx": 1'))
        with self.assertRaisesRegex(ValueError, "published prediction changed"):
            verify_final(self.root)

    def test_interrupted_initial_spec_can_retry_without_removing_failed_attempt(self):
        process = self.spawn("""
import os, sys
from pathlib import Path
from lc_rec.beauty_history_attention.test_prediction_store import BINDING, AXIS
from lc_rec.beauty_history_attention.prediction_store import open_store
os.replace = lambda *args: os._exit(49)
with open_store(Path(sys.argv[1]), BINDING, AXIS, 2):
    pass
""")
        _, err = process.communicate(timeout=10)
        self.assertEqual(process.returncode, 49, err)
        self.assertFalse((self.root / SPEC).exists())
        pending = list(self.root.glob(f".{SPEC}.*"))
        self.assertEqual(len(pending), 1)
        with self.store() as store:
            self.assertEqual(store.committed(validate), [])
        self.assertTrue(pending[0].exists())

    def test_concurrent_writer_refused_and_wrong_binding_cannot_resume(self):
        with self.store():
            process = self.spawn("""
import sys
from pathlib import Path
from lc_rec.beauty_history_attention.test_prediction_store import BINDING, AXIS
from lc_rec.beauty_history_attention.prediction_store import open_store
with open_store(Path(sys.argv[1]), BINDING, AXIS, 2):
    pass
""")
            _, err = process.communicate(timeout=10)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("another prediction attempt holds the output lock", err)
        with self.assertRaisesRegex(ValueError, "different run"):
            with self.store(BINDING | {"backend": "sdpa"}):
                self.fail("accepted different run")
        with self.store() as store:
            store.bind_runtime(RUNTIME)
            with self.assertRaisesRegex(ValueError, "runtime changed"):
                store.bind_runtime(RUNTIME | {"backend": "sdpa"})
            with self.assertRaisesRegex(ValueError, "incomplete"):
                with store.chunk(0, validate) as write_user:
                    write_rows(write_user, AXIS[:1])
            self.assertEqual(store.committed(validate), [])
            with self.assertRaisesRegex(ValueError, "incomplete"):
                store.publish(report(), {})

    def test_tampered_committed_chunk_fails_closed(self):
        with self.store() as store:
            store.bind_runtime(RUNTIME)
            with store.chunk(0, validate) as write_user:
                write_rows(write_user, AXIS[:2])
        victim = self.root / "chunks" / "000000" / "shuffle4.jsonl"
        victim.write_text(victim.read_text().replace('"row_idx": 2', '"row_idx": 1'))
        with self.store() as store, self.assertRaisesRegex(ValueError, "chunk prediction changed"):
            store.committed(validate)


class BoundedSampleTest(unittest.TestCase):
    def test_row_length_strata_and_paired_identity(self):
        rows = [
            dict(row_idx=i, encoded_length=12+i,
                 conditions={mode: {"input_changed": i != j} for j, mode in enumerate(CONDITIONS[1:], 1)})
            for i in range(40)
        ]
        fixed = preflight_rows(rows)
        sample = preflight_rows(rows, 20)
        self.assertEqual(sample[:len(fixed)], fixed)
        self.assertEqual(len(sample), 20)
        self.assertEqual(len(set(sample)), 20)
        self.assertEqual(sample, preflight_rows(rows, 20))
        keys = [f"m{i}" for i in range(16)]
        values = [{"row_idx": i, "metrics": dict.fromkeys(keys, float(i))} for i in sample]
        result = preflight_paired(dict.fromkeys(CONDITIONS, values), keys, sample)
        self.assertEqual(result["row_indices"], sample)
        self.assertEqual(result["status"], "PASS_BOUNDED_PAIRING")
        self.assertTrue(all(record["difference"] == record["interface_percentile_low"]
                            == record["interface_percentile_high"] == 0
                            for records in result["comparisons"].values() for record in records))


if __name__ == "__main__":
    unittest.main()
