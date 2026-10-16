"""Read-only fixed-commit / local snapshot byte and manifest comparison."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def check(args):
    deployment = json.loads(args.deployment.read_bytes())
    manifest, receipt = deployment['manifest'], deployment['receipt']
    source_commit, records = manifest['source_commit'], manifest['files']
    assert len(source_commit) == 40 and len({r['path'] for r in records}) == len(records)
    expected_digest = digest(json.dumps({'source_commit': source_commit, 'files': records}, sort_keys=True).encode())
    assert manifest['snapshot_sha256'] == expected_digest
    assert receipt['files_verified'] == len(records)
    manifest_data = (json.dumps(manifest, indent=2) + '\n').encode()
    assert receipt['manifest_sha256'] == digest(manifest_data)
    checked = []
    with tarfile.open(args.tar, 'r:gz') as archive:
        members = archive.getmembers()
        assert len({m.name for m in members}) == len(members)
        assert {m.name for m in members} == {r['path'] for r in records} | {'DEPLOYMENT.json'}
        assert all(m.isfile() and not PurePosixPath(m.name).is_absolute()
                   and '..' not in PurePosixPath(m.name).parts and '\\' not in m.name for m in members)
        assert archive.extractfile('DEPLOYMENT.json').read() == manifest_data
        for row in records:
            data = archive.extractfile(row['path']).read()
            assert len(data) == row['bytes'] and digest(data) == row['sha256'], row['path']
            original = subprocess.check_output(['git', 'show', source_commit + ':' + row['path']], cwd=args.repo)
            assert data == original, ('source commit mismatch', row['path'])
            if args.published_commit:
                published = subprocess.check_output(['git', 'show', args.published_commit + ':' + row['path']], cwd=args.repo)
                if data != published:
                    # Configuration specs for subsequent observation may add references;
                    # runtime Python is the required immutable scientific source identity.
                    assert not row['path'].endswith('.py'), ('published Python differs', row['path'])
                    checked.append({'path': row['path'], 'change_after_runtime': True, 'source_sha256': digest(data), 'published_sha256': digest(published)})
    return {'status': 'PASS', 'reviewer_task': '/root/delivery_audit',
            'reviewed_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
            'source_commit': source_commit, 'snapshot_sha256': expected_digest,
            'snapshot_tar': {'path': str(args.tar), 'bytes': args.tar.stat().st_size, 'sha256': digest(args.tar.read_bytes())},
            'deployment_manifest': {'path': str(args.deployment), 'sha256': digest(args.deployment.read_bytes())},
            'remote_snapshot_path_in_receipt': receipt['root'], 'files_verified': len(records),
            'python_files_verified': sum(r['path'].endswith('.py') for r in records),
            'tar_matches_fixed_source_commit': True, 'published_commit_compared': args.published_commit,
            'post_runtime_nonpython_changes': checked,
            'scope_limit': 'Independent local archive-to-Git bytes; actual remote files need separate fresh readback evidence. No model/GPU execution or Git mutation.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deployment', required=True, type=Path)
    p.add_argument('--tar', required=True, type=Path)
    p.add_argument('--repo', required=True, type=Path)
    p.add_argument('--published-commit')
    p.add_argument('--report', required=True, type=Path)
    a = p.parse_args()
    try:
        result = check(a)
    except Exception as error:
        result = {'status': 'FAIL', 'reviewer_task': '/root/delivery_audit',
                  'exception': type(error).__name__, 'reason': str(error)}
    result['program'] = {'path': str(Path(__file__).resolve()), 'sha256': digest(Path(__file__).read_bytes())}
    a.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result.get(k) for k in ('status', 'source_commit', 'snapshot_sha256', 'files_verified', 'reason')}))
    raise SystemExit(0 if result['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
