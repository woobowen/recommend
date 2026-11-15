"""Tests for persistent control behavior with an injected runner boundary."""
import tempfile
from pathlib import Path
import unittest
from formal_control import StageControl, digest


class FakeRunner:
    def __init__(self):
        self.jobs = {}
        self.calls = 0
        self.lose_reply = False
    def query(self, job_id):
        return self.jobs.get(job_id)
    def submit(self, path):
        import json
        request = json.loads(Path(path).read_text())
        self.calls += 1
        result = {'job_id': request['job_id'], 'request_sha256': digest(request),
                  'status': 'RUNNING'}
        self.jobs[request['job_id']] = result
        if self.lose_reply:
            raise TimeoutError('injected lost reply')
        return result


class ControlTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.runner = FakeRunner()
        self.plan = {'scope': 'SYNTHETIC_TEST', 'run_commit': 'a' * 40,
                     'config_sha256': 'b' * 64, 'input_identity': {'fixture': 'x'}}
        self.request = {'job_id': 'test-job', 'capture_version': 'a' * 40,
                        'env': {'CUDA_VISIBLE_DEVICES': ''}, 'beauty_stage': 'train',
                        'beauty_config_sha256': 'b' * 64,
                        'beauty_input_identity': {'fixture': 'x'}}
        self.ctl = StageControl(self.tmp.name, self.plan, self.runner,
                                lambda *args: {'verified': True})

    def test_lost_reply_queries_original_job(self):
        self.runner.lose_reply = True
        with self.assertRaises(TimeoutError):
            self.ctl.reconcile('train', self.request)
        control = StageControl(self.tmp.name, self.plan, self.runner,
                               lambda *args: {'verified': True})
        self.assertEqual(control.reconcile('train', self.request)['status'], 'RUNNING')
        self.assertEqual(self.runner.calls, 1)

    def test_dependency_and_formal_scope_rejected(self):
        with self.assertRaises(RuntimeError):
            self.ctl.reconcile('normal', dict(self.request, beauty_stage='normal'))
        with self.assertRaises(PermissionError):
            StageControl(self.tmp.name, dict(self.plan, scope='FORMAL'), self.runner, None)

    def test_cleanup_and_identity_are_required(self):
        self.ctl.reconcile('train', self.request)
        result = self.runner.jobs['test-job']
        result.update(status='SUCCEEDED', exit_code=0, cleanup_verified=False,
                      worker_alive=False, workgroup_live=False)
        with self.assertRaises(RuntimeError):
            self.ctl.reconcile('train', self.request)
        result['cleanup_verified'] = True
        self.assertEqual(self.ctl.reconcile('train', self.request)['status'], 'COMPLETE')
        with self.assertRaises(ValueError):
            self.ctl.reconcile('train', dict(self.request, extra='changed'))

if __name__ == '__main__':
    unittest.main()
