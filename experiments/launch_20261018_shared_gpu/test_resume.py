"""CPU-only continuation safety fixtures; never launch a research command."""
import contextlib
import copy
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from experiments.launch_20261018_shared_gpu import resume


class ContinuationSafety(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_queue = subprocess.run(
            ["git", "show", "b44399926c98369191c2ff5c2425377c8c07d3c8:experiments/launch_20261017/queue.py"],
            cwd=resume._CONTROLLER_ROOT, capture_output=True, check=True).stdout

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.oldroot = self.root / "original-snapshot"
        self.oldroot.mkdir()
        self.state = self.root / "original-state"
        self.state.mkdir()
        (self.state / "supervisor.lock").touch()
        source = self.oldroot / "experiments/launch_20261017/queue.py"
        source.parent.mkdir(parents=True)
        source.write_bytes(self.original_queue)
        self.oldmodule = self.module(source)
        manifest = dict(source_commit="b44399926c98369191c2ff5c2425377c8c07d3c8",
                        files=[dict(path="experiments/launch_20261017/queue.py",
                                    sha256=resume.queue.sha(source), bytes=len(self.original_queue))])
        self.write(self.oldroot / "DEPLOYMENT.json", manifest)
        self.jobs = self.root / "jobs"
        self.jobs.mkdir()
        self.outer = self.jobs / "original-queue"
        self.outer.mkdir()
        self.oldconfig_path = self.root / "original-config.json"
        self.old = dict(schema=1, ready=True, queue_id="original-queue",
                        runner=dict(job_root=str(self.jobs)),
                        snapshot=dict(root=str(self.oldroot), commit=manifest["source_commit"]),
                        namespace_identity={}, state_dir=str(self.state),
                        gpu_locks_dir=str(self.root / "gpu-locks"), poll_seconds=1,
                        max_queue_seconds=172700, max_storage_bytes=53687091200,
                        gpu_budget_seconds=143041, budget_groups={"toys-5300": 9000},
                        prior_gpu_request_usage=dict(seconds=58.650573, jobs=["original-gpu-gate"]),
                        free_space=[dict(path=str(self.root), min_free_bytes=1)],
                        storage_roots=[str(self.root)], blocked_workflows={})
        self.old["stages"] = [self.stage("done", "GPU-aaaa"), self.stage("active", "GPU-bbbb"),
                              self.stage("pending", "GPU-aaaa", ["active"])]
        self.write(self.oldconfig_path, self.old)
        self.old_request = dict(job_id="original-queue", inputs=[
            dict(path=str(self.oldroot / "DEPLOYMENT.json"),
                 sha256=resume.queue.sha(self.oldroot / "DEPLOYMENT.json")),
            dict(path=str(self.oldconfig_path), sha256=resume.queue.sha(self.oldconfig_path))])
        self.write(self.outer / "request.json", self.old_request)
        self.oldstate = dict(queue_id="original-queue", config_sha256=resume.queue.sha(self.oldconfig_path),
                             started_at=(dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=10)).isoformat(),
                             status="STOPPED_NO_NEW_SUBMISSIONS", stages={})
        self.responses = {}
        for name, status, seconds in [("done", "SUCCEEDED", 3), ("active", "RUNNING", 5)]:
            stage = next(s for s in self.old["stages"] if s["id"] == name)
            job = self.jobs / stage["request"]["job_id"]
            job.mkdir()
            request = self.oldmodule.wrapped_request(self.old, self.oldconfig_path, stage)
            self.write(job / "request.json", request)
            self.write(job / "accepted.json", dict(request_sha256=resume._canonical_sha(request)))
            actual = self.status(stage["request"]["job_id"], status, seconds)
            actual["request_sha256"] = resume._canonical_sha(request)
            self.responses[stage["request"]["job_id"]] = actual
            self.oldstate["stages"][name] = dict(job_id=stage["request"]["job_id"], status=status,
                                                runner_status=copy.deepcopy(actual))
        self.oldstate["stages"]["pending"] = dict(job_id="original-pending", status="WAITING_GPU")
        self.write(self.state / "queue-status.json", self.oldstate)
        self.config = copy.deepcopy(self.old)
        self.config.update(queue_id="new-controller", state_dir=str(self.root / "new-state"),
                           approved_gpu_uuids=["GPU-aaaa", "GPU-bbbb"],
                           controller_snapshot=dict(root=str(resume._CONTROLLER_ROOT), commit="e" * 40))
        self.config["stages"][-1]["resource_policy"] = dict(allow_shared=True)
        self.config["continuation"] = dict(old_config_path=str(self.oldconfig_path),
            old_config_sha256=resume.queue.sha(self.oldconfig_path),
            old_queue_state_path=str(self.state / "queue-status.json"),
            old_queue_state_sha256=resume.queue.sha(self.state / "queue-status.json"),
            old_queue_job_id="original-queue", old_queue_request_path=str(self.outer / "request.json"),
            old_queue_request_sha256=resume.queue.sha(self.outer / "request.json"),
            original_absolute_queue_deadline=self.deadline(), submitted_stage_ids=["done", "active"])
        self.configpath = self.root / "new-config.json"
        self.write(self.configpath, self.config)
        self.calls = []
        self.patches = contextlib.ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(mock.patch.object(resume.queue, "runner_call", self.call))
        for name in ["wrapped_request", "accounting", "initial_stage_states", "queue_should_stop"]:
            if hasattr(resume.queue, name):
                value = getattr(resume.queue, name)
                self.addCleanup(setattr, resume.queue, name, value)

    def module(self, source):
        spec = importlib.util.spec_from_file_location("fixture_original_queue", source)
        value = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(value)
        return value

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def stage(self, name, gpu, dependencies=None):
        return dict(id=name, workflow="toys-a-5300", kind="evaluate", gpu_uuids=[gpu],
                    budget_group="toys-5300", depends_on=dependencies or [], gates=[],
                    request=dict(schema=1, job_id="original-" + name,
                                 argv=["/usr/bin/python3", "-c", "raise RuntimeError('NEVER EXECUTE')"],
                                 cwd=str(self.oldroot), env={}, inputs=[], timeout_seconds=100))

    def deadline(self):
        return (dt.datetime.fromisoformat(self.oldstate["started_at"]) +
                dt.timedelta(seconds=self.old["max_queue_seconds"])).isoformat()

    def status(self, job_id, status, seconds=1):
        begin = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=seconds)
        return dict(job_id=job_id, status=status, exit_code=0 if status == "SUCCEEDED" else None,
                    cleanup_verified=status == "SUCCEEDED", supervisor_alive=False,
                    worker_alive=False, workgroup_live=False, started_at=begin.isoformat(),
                    ended_at=(begin + dt.timedelta(seconds=seconds)).isoformat())

    def call(self, config, mode, argument):
        self.calls.append((mode, str(argument)))
        if mode == "query":
            if Path(argument) == self.outer:
                result = self.status("original-queue", "FAILED")
                result.update(cleanup_verified=True,
                              request_sha256=resume._canonical_sha(self.old_request))
                return result
            return copy.deepcopy(self.responses[Path(argument).name])
        request = json.loads(Path(argument).read_text())
        result = self.status(request["job_id"], "SUCCEEDED", 2)
        result["request_sha256"] = resume._canonical_sha(request)
        job = self.jobs / request["job_id"]
        self.write(job / "request.json", request)
        self.write(job / "accepted.json", dict(request_sha256=result["request_sha256"]))
        self.responses[request["job_id"]] = result
        return dict(result, status="ACCEPTED", exit_code=None, cleanup_verified=False)

    def save_state(self):
        self.write(self.state / "queue-status.json", self.oldstate)
        self.config["continuation"]["old_queue_state_sha256"] = resume.queue.sha(self.state / "queue-status.json")
        self.config["continuation"]["original_absolute_queue_deadline"] = self.deadline()
        self.write(self.configpath, self.config)

    def test_inherit_exact_original_wrappers_and_active_job(self):
        result = resume.install(self.config, self.configpath)
        self.assertEqual(result["stage_states"]["done"]["status"], "SUCCEEDED")
        self.assertEqual(result["stage_states"]["active"]["status"], "RUNNING")
        for stage in self.config["stages"][:2]:
            actual = resume.queue.wrapped_request(self.config, self.configpath, stage)
            self.assertEqual(actual, json.loads((self.jobs / stage["request"]["job_id"] / "request.json").read_text()))
        request = resume.queue.wrapped_request(self.config, self.configpath, self.config["stages"][-1])
        self.assertEqual(request["queue_stage"]["command"], self.old["stages"][-1]["request"]["argv"])
        self.assertEqual(request["cwd"], str(self.oldroot))
        self.assertEqual(request["argv"][2], str(Path(resume.__file__).resolve()))

    def test_finite_queue_only_submits_original_unsubmitted_stage(self):
        self.responses["original-active"].update(status="SUCCEEDED", exit_code=0, cleanup_verified=True)
        resume.install(self.config, self.configpath)
        self.patches.enter_context(mock.patch.object(resume.queue, "validate", return_value={"fixture": "CPU_ONLY"}))
        self.patches.enter_context(mock.patch.object(resume.queue, "gpu_snapshot", return_value={"GPU-aaaa": {}, "GPU-bbbb": {}}))
        self.patches.enter_context(mock.patch.object(resume.queue, "disk_snapshot", return_value={"ok": True}))
        self.patches.enter_context(mock.patch.object(resume.queue, "prelaunch_samples", return_value=(True, [])))
        self.assertEqual(resume.queue.supervise(self.config, self.configpath), 0)
        submissions = [Path(arg).stem for mode, arg in self.calls if mode == "submit"]
        self.assertEqual(submissions, ["original-pending"])
        saved = json.loads((Path(self.config["state_dir"]) / "queue-status.json").read_text())
        self.assertEqual(saved["status"], "TERMINAL_PENDING_RESULT_REVIEW")
        self.assertEqual(saved["accounting"]["used_gpu_request_seconds"], 10)
        self.assertAlmostEqual(saved["accounting"]["total_used_gpu_request_seconds_including_prior"], 68.650573)

    def test_unresolved_submitting_intent_refused(self):
        self.oldstate["stages"]["pending"]["status"] = "SUBMITTING"
        self.save_state()
        with self.assertRaisesRegex(ValueError, "SUBMITTING"):
            resume.bind(self.config, self.configpath)

    def test_request_without_acceptance_is_not_reused(self):
        self.write(self.jobs / "original-pending" / "request.json", {"ambiguous": True})
        with self.assertRaisesRegex(ValueError, "unresolved request identity"):
            resume.bind(self.config, self.configpath)

    def test_even_unexplained_empty_original_job_directory_is_not_reused(self):
        (self.jobs / "original-pending").mkdir()
        with self.assertRaisesRegex(ValueError, "unresolved request identity"):
            resume.bind(self.config, self.configpath)

    def test_original_active_controller_refused(self):
        original = self.call
        def active(config, mode, argument):
            result = original(config, mode, argument)
            if Path(argument) == self.outer:
                result.update(status="RUNNING", supervisor_alive=True)
            return result
        with mock.patch.object(resume.queue, "runner_call", active):
            with self.assertRaisesRegex(ValueError, "not verified terminal"):
                resume.bind(self.config, self.configpath)

    def test_unsubmitted_stage_cannot_claim_success_or_execution(self):
        for status in ["SUCCEEDED", "ACCEPTED", "RUNNING", "FAILED", "CONTROL_LOST"]:
            with self.subTest(status=status):
                self.oldstate["stages"]["pending"]["status"] = status
                self.save_state()
                with self.assertRaisesRegex(ValueError, "execution status"):
                    resume.bind(self.config, self.configpath)

    def test_original_outer_cleanup_must_be_verified(self):
        original = self.call
        def incomplete(config, mode, argument):
            result = original(config, mode, argument)
            if Path(argument) == self.outer:
                result.update(cleanup_verified=False)
            return result
        with mock.patch.object(resume.queue, "runner_call", incomplete):
            with self.assertRaisesRegex(ValueError, "not verified terminal"):
                resume.bind(self.config, self.configpath)

    def test_budget_and_scientific_command_cannot_change(self):
        self.config["gpu_budget_seconds"] += 1
        with self.assertRaisesRegex(ValueError, "original bound setting"):
            resume.bind(self.config, self.configpath)
        self.config["gpu_budget_seconds"] -= 1
        self.config["stages"][-1]["request"]["argv"].append("--new-method")
        with self.assertRaisesRegex(ValueError, "scientific request"):
            resume.bind(self.config, self.configpath)

    def test_absolute_original_deadline_cannot_reset(self):
        self.config["continuation"]["original_absolute_queue_deadline"] = (
            dt.datetime.fromisoformat(self.deadline()) + dt.timedelta(seconds=1)).isoformat()
        with self.assertRaisesRegex(ValueError, "absolute queue deadline"):
            resume.bind(self.config, self.configpath)
        self.oldstate["started_at"] = (dt.datetime.now(dt.timezone.utc) -
                                      dt.timedelta(seconds=self.old["max_queue_seconds"])).isoformat()
        self.save_state()
        resume.install(self.config, self.configpath)
        self.assertTrue(resume.queue.queue_should_stop(self.config, 0))

    def test_existing_terminal_failure_preserved_no_retry(self):
        self.responses["original-active"].update(status="FAILED", exit_code=1, cleanup_verified=True)
        result = resume.bind(self.config, self.configpath)
        self.assertEqual(result["stage_states"]["active"]["status"], "BLOCKED_NO_RETRY")
        self.oldstate["stages"]["pending"].update(status="BLOCKED_GATE", reason="original failure")
        self.save_state()
        result = resume.bind(self.config, self.configpath)
        self.assertEqual(result["stage_states"]["pending"]["status"], "BLOCKED_GATE")

    def test_original_lock_must_be_released(self):
        with (self.state / "supervisor.lock").open("r+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                resume.bind(self.config, self.configpath)


if __name__ == "__main__":
    unittest.main(verbosity=2)
