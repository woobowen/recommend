"""Independent terminal-job identity, export and conservative budget audit.

Consumes original detached-runner exported directories. Does not submit/query
jobs or load models. Full result/attention numerical acceptance is a separate
non-author audit. Failed checks remain explicit, never turned into success.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

ALLOWED_GPUS = {'GPU-dfff0e95-1db6-e2f4-3431-55bb852fa0ae',
                'GPU-59cac3bd-3645-668b-79c8-839e0d878893'}
EVAL_IDS = {f'gs-20261017-{domain}-b-{condition}-v1'
            for domain in ('toys', 'sports') for condition in ('normal', 'shuffle3', 'shuffle4')}
ATTENTION_IDS = {f'gs-20261017-{domain}-b-{condition}-attention-v1'
                 for domain in ('toys', 'sports') for condition in ('normal', 'shuffle3', 'shuffle4')}
CALLBACK_FAILURE_IDS = {f'gs-20261017-{domain}-b-callback-audit-v1' for domain in ('toys', 'sports')}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_identity(path):
    data = path.read_bytes()
    return {'path': str(path), 'bytes': len(data), 'sha256': digest(data)}


def safe_name(name):
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and '..' not in path.parts and '\\' not in name and str(path) == name


def date(value):
    return dt.datetime.fromisoformat(value.replace('Z', '+00:00'))


def phase_of(request):
    args = request['argv']
    if 'experiments.shuffle_20261017.preflight' in args:
        return 'preflight'
    for phase in ('evaluate', 'attention', 'identity-check'):
        if phase in args:
            return phase
    return 'cpu_evidence'


def verify_directory(path):
    read = lambda name: json.loads((path / name).read_bytes())
    manifest = read('export-manifest.json')['files']
    assert len({row['path'] for row in manifest}) == len(manifest)
    expected = {row['path'] for row in manifest}
    actual = {str(p.relative_to(path)) for p in path.rglob('*') if p.is_file()}
    assert actual == expected | {'export-manifest.json'}, ('export file set', str(path))
    for row in manifest:
        assert safe_name(row['path'])
        member = path / row['path']
        assert not member.is_symlink()
        data = member.read_bytes()
        assert len(data) == row['bytes'] and digest(data) == row['sha256'], str(member)
    request, status, accepted = read('request.json'), read('status.json'), read('accepted.json')
    job = request['job_id']
    assert re.fullmatch(r'gs-20261017-[a-z0-9-]+', job)
    assert status['job_id'] == job
    request_fp = digest(json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode())
    assert accepted['request_sha256'] == request_fp
    assert status['status'] in ('SUCCEEDED', 'FAILED', 'TIMED_OUT', 'CANCELLED', 'INCOMPLETE'), ('nonterminal', job, status['status'])
    assert type(status['exit_code']) is int
    assert status['cleanup']['verified'] is True and not status.get('worker_may_be_live', False)
    phase = phase_of(request)
    cuda = request.get('env', {}).get('CUDA_VISIBLE_DEVICES', '')
    if cuda:
        assert cuda in ALLOWED_GPUS, (job, 'physical UUID')
        cap = {'preflight': 900, 'evaluate': 1800, 'attention': 1200}[phase]
        assert 0 < request['timeout_seconds'] <= cap
    conservative_seconds = max(0, (date(status['ended_at']) - date(status['started_at'])).total_seconds())
    invocation = read('output/invocation.json') if (path / 'output/invocation.json').exists() else None
    deployment = None
    if invocation is not None:
        deployment = invocation.get('code_identity')
        if deployment:
            assert request['capture_version'] == deployment['source_commit'] + '/' + deployment['snapshot_sha256']
            derived = digest(json.dumps({'source_commit': deployment['source_commit'], 'files': deployment['files']}, sort_keys=True).encode())
            assert derived == deployment['snapshot_sha256']
            manifest_sha = digest((json.dumps(deployment, indent=2) + '\n').encode())
            inputs = {row['path']: row['sha256'] for row in request['inputs']}
            assert inputs[request['cwd'] + '/DEPLOYMENT.json'] == manifest_sha
            assert invocation['spec_sha256'] in inputs.values()
    if status['status'] == 'SUCCEEDED' and phase in ('evaluate', 'attention'):
        assert status['exit_code'] == 0 and invocation is not None and deployment is not None
        assert invocation['phase'] == phase
        assert invocation['interpreter'] == '/data/bwwu/envs/grid-20261009-e61baf3/bin/python'
        spec, result = invocation['spec'], read('output/RESULT.json')
        assert spec['domain'] in ('toys', 'sports') and spec['route'] == 'B'
        assert spec['shuffle_n'] in (0, 3, 4) and spec['seed'] == 42 and spec['generation_k'] == 10
        assert result['formal'] is True and result['status'] == 'COMPLETE'
        assert all(result[key] == spec[key] for key in ('config', 'sid', 'checkpoint', 'shuffle_n'))
        if phase == 'evaluate':
            assert result['test_complete'] is True and result['metrics']['coverage_exact'] is True
            assert result['metrics']['samples'] == {'toys': 19412, 'sports': 35598}[spec['domain']]
        else:
            assert result['samples'] == 32 and len(result['batches']) == 4
    if job in CALLBACK_FAILURE_IDS:
        assert status['status'] == 'FAILED' and status['exit_code'] != 0
        assert not (path / 'output/RESULT.json').exists(), 'callback failure must not be labelled completed'
    evidence = [file_identity(path / name) for name in ('request.json', 'accepted.json', 'status.json', 'export-manifest.json')]
    return {'job_id': job, 'phase': phase, 'status': status['status'], 'exit_code': status['exit_code'],
            'cleanup_verified': True, 'worker_may_be_live': False, 'gpu_uuid': cuda or None,
            'timeout_seconds': request['timeout_seconds'],
            'conservative_status_interval_seconds_including_cleanup': conservative_seconds,
            'source_commit': deployment['source_commit'] if deployment else None,
            'snapshot_sha256': deployment['snapshot_sha256'] if deployment else None,
            'capture_version': request['capture_version'], 'snapshot_path': request['cwd'],
            'original_export_directory': str(path), 'export_files_verified': len(manifest), 'evidence': evidence}


def audit(args):
    paths = [Path(path) for path in json.loads(args.directories.read_text())]
    assert len(set(paths)) == len(paths), 'duplicate directory listing'
    jobs = [verify_directory(path) for path in paths]
    identifiers = [row['job_id'] for row in jobs]
    assert len(set(identifiers)) == len(identifiers), 'duplicate original job'
    by_id = {row['job_id']: row for row in jobs}
    if args.final:
        assert EVAL_IDS <= set(by_id), 'missing formal B evaluation condition'
        assert ATTENTION_IDS <= set(by_id), 'missing formal B attention condition'
        assert CALLBACK_FAILURE_IDS <= set(by_id), 'missing preserved optional callback failures'
        assert all(by_id[job]['status'] == 'SUCCEEDED' for job in EVAL_IDS | ATTENTION_IDS), 'formal condition incomplete'
    gpu_jobs = [row for row in jobs if row['gpu_uuid']]
    actual = sum(row['conservative_status_interval_seconds_including_cleanup'] for row in gpu_jobs)
    checks = sum(row['conservative_status_interval_seconds_including_cleanup'] for row in gpu_jobs if row['phase'] == 'preflight')
    requested_caps = sum(row['timeout_seconds'] for row in gpu_jobs)
    assert actual <= 5.5 * 3600, 'task hard budget exceeded'
    assert checks <= 0.5 * 3600, 'GPU check hard budget exceeded'
    return {'status': 'PASS', 'reviewer_task': '/root/delivery_audit',
            'reviewed_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'final_complete_B_matrix_required': args.final,
            'job_count': len(jobs), 'terminal_counts': dict(Counter(row['status'] for row in jobs)),
            'all_original_exports_identity_verified': True, 'jobs': jobs,
            'accounting': {'conservative_status_interval_seconds_including_cleanup': actual,
                           'gpu_check_conservative_seconds': checks,
                           'sum_requested_timeout_caps_seconds': requested_caps,
                           'request_worker_wall_hard_limit_seconds': 5.5 * 3600,
                           'gpu_check_wall_hard_limit_seconds': 0.5 * 3600,
                           'not_precise_GPU_utilization_time': True},
            'scope_limit': 'Original captured terminal statuses and byte/version chain; no fresh remote process query. Full prediction, beam, head and statistics semantics depend on separate independent result audits.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directories', required=True, type=Path)
    p.add_argument('--report', required=True, type=Path)
    p.add_argument('--final', action='store_true')
    args = p.parse_args()
    try:
        report = audit(args)
    except Exception as error:
        report = {'status': 'FAIL', 'reviewer_task': '/root/delivery_audit',
                  'exception': type(error).__name__, 'reason': str(error)}
    report['program'] = file_identity(Path(__file__).resolve())
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: report.get(key) for key in ('status', 'job_count', 'terminal_counts', 'reason')}))
    raise SystemExit(0 if report['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
