"""CPU-only verification of formal ownership, recovery, and fail-closed gates.

Fixtures exercise file and controller mechanics in disposable directories. No
fixture issues a formal approval, contacts the deployed runner, or imports CUDA.
"""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import subprocess
import sys
import unittest
from unittest.mock import patch

from . import formal


class FormalChecks(unittest.TestCase):
    def setUp(self):
        self.work = TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        self.root = Path(self.work.name)
        self.plan = {"run_commit": "1" * 40, "job_name": "cpu-fixture",
                     "output_root": str(self.root / "out"),
                     "config_sha256": "2" * 64,
                     "train_sha256": "3" * 64, "snapshot_manifest_sha256": "4" * 64,
                     "mode": "bidirectional", "backend": "eager"}

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def test_formal_submit_and_worker_deny_missing_gate_before_touching_runner(self):
        args = SimpleNamespace(plan=self.root / "unsigned-plan.json", formal_authorization=None,
                               formal_authorization_sha256=None, attempt=1)
        with patch.object(formal, "check_runner") as runner:
            for command in (formal.submit, formal.worker):
                with self.assertRaisesRegex(ValueError, "separate user authorization"):
                    command(args)
            runner.assert_not_called()
        self.assertFalse(Path(self.plan["output_root"]).exists())

    def test_unsigned_placeholder_and_mismatched_digest_are_rejected(self):
        unsigned = self.write(self.root / "unsigned.json", {"formal_run_authorized": False,
                                                            "external_review_status": "PENDING"})
        planfile = self.root / "unsigned-plan.json"
        with patch.object(formal, "validate_plan", return_value=(self.plan, "5" * 64)):
            with self.assertRaisesRegex(ValueError, "record SHA-256 mismatch"):
                formal.verify_gate(planfile, unsigned, "6" * 64)
            with self.assertRaisesRegex(ValueError, "does not cover"):
                formal.verify_gate(planfile, unsigned, formal.sha(unsigned))

    def test_immutable_file_is_retryable_after_prepublication_interrupt(self):
        path = self.root / "receipt.json"
        with patch.object(formal.os, "fsync", side_effect=InterruptedError):
            with self.assertRaises(InterruptedError):
                formal.install_immutable(path, {"fixture": "old"})
        self.assertFalse(path.exists())
        formal.install_immutable(path, {"fixture": "old"})
        formal.install_immutable(path, {"fixture": "old"})
        with self.assertRaisesRegex(ValueError, "different identity|another identity"):
            formal.install_immutable(path, {"fixture": "other"})
        self.assertEqual(formal.read_record(path)[0], {"fixture": "old"})
        self.assertFalse(tuple(self.root.glob("*.partial")))

    def test_simultaneous_worker_lock_and_uncertain_runner_forbid_retry(self):
        with formal.lock(self.root / "worker.lock"):
            with self.assertRaises(BlockingIOError):
                with formal.lock(self.root / "worker.lock"):
                    self.fail("same-run work lock was shared")
        for status in (
            None,
            {"status": "CONTROL_LOST", "cleanup_verified": True},
            {"status": "FAILED", "cleanup_verified": True, "supervisor_alive": True},
            {"status": "FAILED", "cleanup_verified": False},
        ):
            with self.assertRaisesRegex(ValueError, "no clean terminal"):
                formal.check_attempt(self.plan, 2, "6" * 64, "5" * 64, lambda *_: status)

    def test_retry_accepts_only_clean_same_authorized_lineage(self):
        job = self.root / formal.job_id(self.plan, 1)
        request = self.write(job / "request.json", {
            "argv": ["x", "x", "x", "x", "worker", "--plan-sha256", "5" * 64,
                     "--formal-authorization-sha256", "6" * 64],
        })
        clean = {"status": "FAILED", "cleanup_verified": True,
                 "started_at": "2026-11-27T12:00:00+00:00", "ended_at": "2026-11-27T13:00:00+00:00"}
        with patch.object(formal, "RUNNER_JOBS", self.root):
            actual = formal.check_attempt(self.plan, 2, "6" * 64, "5" * 64, lambda *_: clean)
            self.assertEqual(actual, formal.BUDGETS["chain_seconds"] - 3600)
            with self.assertRaisesRegex(ValueError, "another plan"):
                formal.check_attempt(self.plan, 2, "6" * 64, "7" * 64, lambda *_: clean)
            with self.assertRaisesRegex(ValueError, "already completed"):
                formal.check_attempt(self.plan, 2, "6" * 64, "5" * 64,
                                     lambda *_: clean | {"status": "SUCCEEDED"})
        self.assertTrue(request.is_file())

    def test_same_id_queries_existing_request_without_submitting(self):
        root = Path(self.plan["output_root"])
        args = SimpleNamespace(attempt=1, plan=self.root / "plan",
                               formal_authorization=self.root / "absent",
                               formal_authorization_sha256="6" * 64)
        request = {"fixture_request": True}
        self.write(self.root / formal.job_id(self.plan, 1) / "request.json", request)
        seen = {"fixture_runner_state": "RUNNING"}
        with (patch.object(formal, "gate_records", return_value=(self.plan, "5" * 64, {}, (None, None))),
              patch.object(formal, "check_runner"),
              patch.object(formal, "RUNNER_JOBS", self.root),
              patch.object(formal, "runner_request", return_value=request),
              patch.object(formal, "get_status", return_value=seen),
              patch.object(formal, "check_idle") as check_idle,
              patch.object(formal, "runner") as runner):
            self.assertEqual(formal.submit(args), seen)
            check_idle.assert_not_called()
            runner.assert_not_called()
        self.assertFalse((root / "runner-request-r1.json").exists())

    def test_restart_does_not_overwrite_incomplete_training_attempt(self):
        old = self.root / "training-r1"
        checkpoint = old / "checkpoint-2053"
        checkpoint.mkdir(parents=True)
        self.write(self.root / "attempt-r1/training.json", {"path": str(old), "resume_from": None})
        fresh, resume = formal.training_resume(self.root, 2, self.plan)
        self.assertEqual(fresh, self.root / "training-r2")
        self.assertIsNone(resume)
        self.assertTrue(checkpoint.is_dir())

    def test_resume_uses_durable_checkpoint_and_quarantines_half_written_next(self):
        old = self.root / "training-r1"
        selected = self.write(old / "checkpoint-2053/complete.json", {})
        broken = self.write(old / "checkpoint-4107/partial-artifact", {"synthetic": True})
        self.write(self.root / "attempt-r1/training.json", {"path": str(old), "resume_from": None})
        future = self.root / "attempt-r2"
        future.mkdir()
        with patch.object(formal, "durable_checkpoint") as validate:
            self.assertEqual(formal.training_resume(self.root, 2, self.plan), (old, selected.parent))
            validate.assert_called_once_with(selected.parent, self.plan)
        formal.archive_partial(old, future)
        self.assertTrue((future / "prior-failed-output/checkpoint-4107/partial-artifact").is_file())
        self.assertFalse(broken.exists())
        self.assertTrue(selected.is_file())

    def test_final_checkpoint_choice_checks_all_four_losses_and_earliest_tie(self):
        old = self.root / "training-r1"
        latest = old / "checkpoint-8214"
        best = old / "checkpoint-4107"
        phases = [[2053, 0.5], [4107, 0.2], [6160, 0.2], [8214, 0.3]]
        state = {"global_step": 8214, "epoch": 4.0, "best_model_checkpoint": str(best), "best_metric": 0.2,
                 "log_history": [{"step": step, "eval_loss": loss} for step, loss in phases]}
        path = self.write(latest / "trainer_state.json", state)
        with patch.object(formal, "durable_checkpoint"):
            self.assertEqual(formal.validation_choice(latest, self.plan), (best, 0.2, phases))
            self.write(path, state | {"best_model_checkpoint": str(old / "checkpoint-6160")})
            with self.assertRaisesRegex(ValueError, "earliest minimum"):
                formal.validation_choice(latest, self.plan)

    def test_interrupted_after_step_8214_only_runs_strict_export_recovery(self):
        attempt = self.root / "attempt-r2"
        attempt.mkdir()
        old = self.root / "training-r1"
        latest = old / "checkpoint-8214"
        best = old / "checkpoint-4107"
        args = SimpleNamespace(attempt=2, plan=self.root / "plan.json", formal_authorization=self.root / "absent",
                               formal_authorization_sha256="6" * 64)
        self.write(attempt / "recovery-reload.json", {"checkpoint": str(best), "strictly_reloaded": True})
        with (patch.object(formal, "training_resume", return_value=(old, latest)),
              patch.object(formal, "validation_choice", return_value=(best, 0.2, [])),
              patch.object(formal, "launch") as launch,
              patch.object(formal, "result_for_adapter", return_value={"fixture": True})):
            candidate = formal.finish_training(self.plan, args, self.root, attempt)
        self.assertEqual(candidate, attempt / "recovered-best-adapter")
        self.assertEqual([call.args[:2] for call in launch.call_args_list], [("formal", "recovery-export")])

    def test_cpu_metrics_incomplete_summary_does_not_block_retry_or_get_reused(self):
        prior = self.write(self.root / "attempt-r1/metrics/summary.json", {"half": "done"})
        target = self.root / "attempt-r2"
        target.mkdir()
        args = SimpleNamespace(attempt=2)
        with (patch("lc_rec.beauty_history_attention.evaluate.verify_completed_metrics") as validate,
              patch.object(formal, "launch") as launch):
            got = formal.finish_metrics(self.plan, self.root, target, args, self.root / "generation")
            self.assertEqual(got, target / "metrics")
            launch.assert_called_once()
            self.assertEqual(validate.call_args_list[0].args[0], target / "metrics")
        self.assertEqual(formal.read_record(prior)[0], {"half": "done"})

    def test_complete_output_remains_readable_after_runner_cleanup_failure(self):
        root = Path(self.plan["output_root"])
        metrics = root / "attempt-r1/metrics"
        generation, adapter = root / "generation", root / "training-r1/best_adapter"
        handoff = self.write(root / "training-result.json", {
            "adapter": str(adapter), "run_commit": self.plan["run_commit"], "completed_training_step": 8214,
        })
        summary = self.write(metrics / "summary.json", {
            "status": "PASS_V2_EVALUATION", "run_commit": self.plan["run_commit"],
            "users_per_condition": 22363, "condition_count": 5,
        })
        result = {"status": "COMPLETE", "plan_sha256": "5" * 64, "run_commit": self.plan["run_commit"],
                  "training_sha256": formal.sha(handoff), "metrics_summary_sha256": formal.sha(summary),
                  "generation": str(generation), "metrics": str(metrics), "adapter": str(adapter)}
        self.write(root / "complete.json", result)
        with (patch.object(formal, "verify_selected_adapter") as check_adapter,
              patch("lc_rec.beauty_history_attention.evaluate.verify_completed_generation") as check_generation,
              patch("lc_rec.beauty_history_attention.evaluate.verify_completed_metrics") as check_metrics):
            self.assertEqual(formal.completed_output(self.plan, "5" * 64, verify_files=True), result)
            check_adapter.assert_called_once_with(self.plan, adapter)
            check_generation.assert_called_once_with(generation, self.plan["run_commit"], adapter, "eager")
            check_metrics.assert_called_once_with(metrics, generation, self.plan["run_commit"])
        summary.write_text('{"broken_fixture":true}')
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            formal.completed_output(self.plan, "5" * 64, verify_files=False)

    def test_direct_training_rejects_optimized_python(self):
        result = subprocess.run([sys.executable, "-OB", "-m", formal.PACKAGE + ".train", "--help"],
                                capture_output=True, text=True, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("training requires all runtime checks", result.stderr)


if __name__ == "__main__":
    unittest.main()
