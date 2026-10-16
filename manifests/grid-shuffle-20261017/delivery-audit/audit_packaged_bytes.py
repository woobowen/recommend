"""Independent packaged bytes vs fixed Git and original runner exports.

The archive/container audit is separate. This check ties every namespaced
repository/source/Beauty/evidence/external/applied-rule member to its real
origin and checks that the complete canonical raw fixed32 matrices survive.
"""
import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import zipfile

SUPPORT = Path('/home/addaswsw/recommend-support/grid-shuffle-20261017')
RULES = Path('/home/addaswsw/recommend')
PREFIX = 'manifests/grid-shuffle-20261017/'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git_bytes(repo, commit, path):
    return subprocess.check_output(['git', 'show', commit + ':' + path], cwd=repo)


def check(args):
    summary_bytes = git_bytes(args.repo, args.content_commit, PREFIX + 'EXECUTION_SUMMARY.json')
    summary = json.loads(summary_bytes)
    anchor = summary['inherited_Beauty_anchor']
    jobs = {row['job_id']: row for row in summary['jobs']}
    assert len(jobs) == len(summary['jobs'])
    originals = {}
    for job, row in jobs.items():
        directory = Path(row['local_evidence'])
        raw = (directory / 'export-manifest.json').read_bytes()
        assert sha(raw) == row['export_manifest']['sha256']
        entries = json.loads(raw)['files']
        assert len({entry['path'] for entry in entries}) == len(entries)
        originals[job] = {entry['path']: entry for entry in entries}
    checked = []
    raw_npz = []
    namespaces = Counter()
    with zipfile.ZipFile(args.zip) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = info.filename
            value = archive.read(info)
            namespace, _, relative = name.partition('/')
            origin = None
            if namespace in ('repository', 'fixed-source'):
                expected = git_bytes(args.repo, args.content_commit, relative)
                assert value == expected, ('Git content mismatch', name)
                origin = args.content_commit + ':' + relative
            elif namespace == 'inherited-beauty':
                path = relative if relative.startswith('experiments/') else 'manifests/grid-formal-20261015/' + relative
                assert value == git_bytes(args.repo, anchor, path), ('Beauty anchor mismatch', name)
                origin = anchor + ':' + path
            elif namespace == 'evidence':
                job, _, entry = relative.partition('/')
                assert job in jobs, ('unknown job in ZIP', job)
                directory = Path(jobs[job]['local_evidence'])
                if entry == 'export-manifest.json':
                    assert value == (directory / entry).read_bytes()
                    origin = str(directory / entry)
                elif entry == 'ALL_PNG_IDENTITIES.json':
                    expected = [row for row in originals[job].values() if row['path'].endswith('.png')]
                    assert json.loads(value) == expected, ('all PNG identity index', job)
                    assert len(expected) == 32
                    origin = str(directory / 'export-manifest.json') + '#all_PNG'
                else:
                    row = originals[job][entry]
                    assert len(value) == row['bytes'] and sha(value) == row['sha256'], ('raw export mismatch', name)
                    origin = str(directory / entry)
                    if entry.endswith('/bundle/arrays.npz'):
                        raw_npz.append((job, entry))
            elif namespace == 'external-receipts':
                path = SUPPORT / relative
                assert path.is_file() and not path.is_symlink()
                assert value == path.read_bytes(), ('external receipt byte mismatch', name)
                origin = str(path)
            elif namespace == 'applied-rules':
                path = RULES / relative
                assert value == path.read_bytes(), ('applied rules mismatch', name)
                origin = str(path)
            else:
                assert name in ('README.md', 'MANIFEST.json'), ('unexpected package namespace', name)
                origin = 'generated navigation or checked inner manifest'
            namespaces[namespace] += 1
            checked.append({'member': name, 'bytes': len(value), 'sha256': sha(value), 'origin': origin})
    conditions = {row['attention_job_id'] for row in summary['conditions']}
    assert len(conditions) == 6
    assert set(raw_npz) == {(job, f'output/batch-{batch:02d}/bundle/arrays.npz')
                            for job in conditions for batch in range(4)}, 'complete raw NPZ matrix'
    assert namespaces['repository'] > 0 and namespaces['fixed-source'] > 0 and namespaces['inherited-beauty'] > 0
    return {'status': 'PACKAGED_ORIGIN_BYTES_PASS', 'reviewer_task': '/root/delivery_audit',
            'reviewed_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
            'content_commit': args.content_commit, 'Beauty_anchor': anchor,
            'zip': {'path': str(args.zip), 'bytes': args.zip.stat().st_size, 'sha256': sha(args.zip.read_bytes())},
            'summary_sha256': sha(summary_bytes), 'files_checked': len(checked), 'namespace_counts': dict(namespaces),
            'original_complete_raw_NPZ_count': len(raw_npz), 'members': checked,
            'scope_limit': 'All packaged origin bytes checked against published Git or original export identity; full container CRC/path/secret/record validation is separate audit_archive.py. No models or GPU calls.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--zip', required=True, type=Path)
    p.add_argument('--repo', required=True, type=Path)
    p.add_argument('--content-commit', required=True)
    p.add_argument('--report', required=True, type=Path)
    a = p.parse_args()
    try:
        result = check(a)
    except Exception as error:
        result = {'status': 'FAIL', 'reviewer_task': '/root/delivery_audit',
                  'exception': type(error).__name__, 'reason': str(error)}
    result['program'] = {'path': str(Path(__file__).resolve()), 'sha256': sha(Path(__file__).read_bytes())}
    a.report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({key: result.get(key) for key in ('status', 'content_commit', 'files_checked', 'reason')}))
    raise SystemExit(0 if result['status'] == 'PACKAGED_ORIGIN_BYTES_PASS' else 1)


if __name__ == '__main__':
    main()
