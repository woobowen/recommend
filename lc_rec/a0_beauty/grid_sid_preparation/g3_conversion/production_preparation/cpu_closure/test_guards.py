"""Targeted guard checks, with temporary fixtures and no production work."""
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import patch

from common import digest, pin, read, sha, write
from closure import snapshot_paths, writable_handles
from launch import job_name, ledger


class Guards(TestCase):
    def test_input_change_and_write_handle(self):
        with TemporaryDirectory() as scratch:
            p = Path(scratch) / "input"
            p.write_bytes(b"sealed")
            original = sha(p)
            self.assertEqual(pin(p, original, 6), p)
            observed = snapshot_paths({p})
            with p.open("ab") as writer:
                self.assertIn({"pid": os.getpid(), "fd": writer.fileno()}, writable_handles({p}))
                writer.write(b"changed")
            self.assertNotEqual(observed, snapshot_paths({p}))
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                pin(p, original)

    def test_exclusive_evidence_write(self):
        with TemporaryDirectory() as scratch:
            p = Path(scratch) / "result.json"
            write(p, {"attempt": 0}, exclusive=True)
            with self.assertRaises(FileExistsError):
                write(p, {"attempt": 1}, exclusive=True)
            self.assertEqual(read(p), {"attempt": 0})

    def test_failed_attempt_charged_and_missing_runner_reserved(self):
        with TemporaryDirectory() as scratch:
            root = Path(scratch)
            intents = root / "requests"
            jobs = root / "jobs"
            intents.mkdir()
            jobs.mkdir()
            cfg = {"closure_id": "test", "plan_raw_sha256": "test-plan", "runner_job_prefix": "fixture",
                   "cpu_job_root": str(jobs), "cpu_runner": "/tmp/fixture",
                   "budget": {"cpu_worker_wall_max_seconds": 7200, "combined_max_seconds": 199800,
                              "cleanup_reserve_seconds": 120}}
            first = job_name(cfg, "compute", 0)
            next_job = job_name(cfg, "compute", 1)
            for job, n in ((first, 0), (next_job, 1)):
                req = {"job_id": job, "closure_kind": "compute", "closure_correction_round": n,
                       "closure_id": "test", "original_plan_sha256": "test-plan", "timeout_seconds": 1000}
                write(intents / f"{job}.json", req, exclusive=True)
            (jobs / first).mkdir()
            request = read(intents / f"{first}.json")
            write(jobs / first / "request.json", request)
            start = datetime(2026, 11, 20, tzinfo=timezone.utc)
            failed = {"request_sha256": digest(request), "status": "FAILED", "exit_code": 2,
                      "started_at": start.isoformat(), "ended_at": (start + timedelta(seconds=31)).isoformat(),
                      "cleanup_verified": True, "worker_alive": False, "supervisor_alive": False,
                      "workgroup_live": False}
            with patch("launch.runner", return_value=failed):
                receipt = ledger(cfg, intents, 124174.622703)
                self.assertEqual(receipt["closure_used_seconds"], 31)
                self.assertEqual(receipt["closure_reserved_seconds_unsettled"], 1000)
                self.assertEqual(receipt["remaining_closure_seconds"], 6169)
                self.assertEqual(receipt["highest_correction_round"], 1)
                self.assertEqual(receipt["attempts"][0]["runner"]["exit_code"], 2)
                (jobs / "fixture-unaccounted").mkdir()
                with self.assertRaisesRegex(ValueError, "unaccounted"):
                    ledger(cfg, intents, 124174.622703)


if __name__ == "__main__":
    main()
