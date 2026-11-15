"""Beauty control entry point and legacy test compatibility primitives.

CLI execution delegates to production_control.Control, which owns formal
requests, artifact validation, authorization and cumulative recovery budgets.
StageControl remains only for inherited small tests; production does not use it.
"""
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

STAGES = ('train', 'verify_best_adapter', 'normal', 'normal_eval', 's3',
          's3_eval', 's4', 's4_eval', 'paired', 'collect')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def durable_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(canonical(value) + b'\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextlib.contextmanager
def locked(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def read(path):
    return json.loads(Path(path).read_text())


class Runner:
    """Call the fixed, independently supervised runner on the server."""
    def __init__(self, path, expected_sha256, job_root):
        self.path = Path(path).resolve(strict=True)
        self.job_root = Path(job_root).resolve(strict=True)
        if hashlib.sha256(self.path.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError('runner identity mismatch')

    def query(self, job_id):
        job = self.job_root / job_id
        if not (job / 'request.json').is_file():
            return None
        return json.loads(subprocess.check_output(
            ['/usr/bin/python3', '-I', '-S', '-B', str(self.path), 'query', str(job)],
            text=True, timeout=45, env={k:v for k,v in os.environ.items() if k not in ('GRID_JOB_ID','GRID_JOB_DIR')}))

    def submit(self, request_path):
        return json.loads(subprocess.check_output(
            ['/usr/bin/python3', '-I', '-S', '-B', str(self.path), 'submit',
             str(self.job_root), str(request_path)], text=True, timeout=60, env={k:v for k,v in os.environ.items() if k not in ('GRID_JOB_ID','GRID_JOB_DIR')}))


class StageControl:
    """Persist immutable requests and reconcile one sequential stage at a time.

    An ambiguous submit is never retried here: query the original identity and
    require manual reconciliation if no accepted job is visible.  The caller
    must provide a real stage-specific artifact verifier.  This primitive has no
    authority to construct or submit a formal GPU request.
    """
    def __init__(self, root, plan, runner, verifier):
        self.root = Path(root)
        self.plan = plan
        self.runner = runner
        self.verifier = verifier
        if plan.get('scope') != 'SYNTHETIC_TEST':
            raise PermissionError('formal launch is not authorized or implemented')
        if not re.fullmatch(r'[0-9a-f]{40}', plan.get('run_commit', '')):
            raise ValueError('full commit required')
        if not re.fullmatch(r'[0-9a-f]{64}', plan.get('config_sha256', '')):
            raise ValueError('configuration identity required')
        if not plan.get('input_identity'):
            raise ValueError('input identity required')
        self.root.mkdir(parents=True, exist_ok=True)
        with locked(self.root / '.control.lock'):
            identity = self.root / 'plan.json'
            if identity.exists() and read(identity) != plan:
                raise ValueError('plan identity conflict')
            if not identity.exists():
                durable_write(identity, plan)

    def reconcile(self, stage, request):
        if stage not in STAGES:
            raise ValueError('unknown stage')
        job_id = request.get('job_id', '')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,90}', job_id):
            raise ValueError('invalid job identity')
        if request.get('capture_version') != self.plan['run_commit']:
            raise ValueError('request commit mismatch')
        if request.get('beauty_stage') != stage or request.get('beauty_config_sha256') != self.plan['config_sha256'] or request.get('beauty_input_identity') != self.plan['input_identity']:
            raise ValueError('request stage or input identity mismatch')
        if request.get('env', {}).get('CUDA_VISIBLE_DEVICES') != '':
            raise PermissionError('only isolated CPU tests are authorized')
        request_path = self.root / (stage + '.request.json')
        state_path = self.root / (stage + '.state.json')
        with locked(self.root / '.control.lock'):
            for previous in STAGES[:STAGES.index(stage)]:
                state = self.root / (previous + '.state.json')
                if not state.is_file() or read(state).get('status') != 'COMPLETE':
                    raise RuntimeError('unverified dependency: ' + previous)
            for other in STAGES:
                other_path = self.root / (other + '.request.json')
                if other != stage and other_path.exists() and read(other_path).get('job_id') == job_id:
                    raise ValueError('job identity already bound to another stage')
            if request_path.exists() and read(request_path) != request:
                raise ValueError('stage request identity conflict')
            if not request_path.exists():
                durable_write(request_path, request)
            state = read(state_path) if state_path.exists() else None
            if state and state['request_sha256'] != digest(request):
                raise ValueError('persisted request identity conflict')
            result = self.runner.query(job_id)
            if result is None:
                if state is not None:
                    raise RuntimeError('submission outcome unknown: query original job; no restart')
                durable_write(state_path, {'status': 'SUBMITTING', 'job_id': job_id,
                                           'request_sha256': digest(request)})
                result = self.runner.submit(request_path)
            if result.get('job_id') != job_id or result.get('request_sha256') != digest(request):
                raise ValueError('runner request identity mismatch')
            status = result.get('status')
            record = {'status': status, 'job_id': job_id, 'request_sha256': digest(request),
                      'runner_observation': result}
            if status == 'SUCCEEDED':
                if (result.get('exit_code') != 0 or not result.get('cleanup_verified') or
                    result.get('worker_alive') is not False or result.get('workgroup_live') is not False):
                    raise RuntimeError('terminal cleanup or exit unverified')
                evidence = self.verifier(stage, request, result)
                if not isinstance(evidence, dict) or evidence.get('verified') is not True:
                    raise RuntimeError('stage artifacts unverified')
                record.update(status='COMPLETE', artifact_evidence=evidence)
            durable_write(state_path, record)
            return record


if __name__ == "__main__":
    from production_control import main
    main()
