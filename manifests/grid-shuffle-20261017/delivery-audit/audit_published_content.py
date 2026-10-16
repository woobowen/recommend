"""Actual independent content publication readback and evidence acceptance.

Adds fresh GitHub ref/tree/raw-byte reads, actual immutable-server byte reads,
and original-ID terminal/process checks to the completed content data audit.
No Git mutation, submission, model/GPU calls or original evidence rewriting.
"""
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from urllib.parse import quote

HERE = Path(__file__).resolve().parent


def identity(path):
    raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--commit', required=True)
    p.add_argument('--publication', required=True, type=Path)
    p.add_argument('--report', required=True, type=Path)
    args = p.parse_args()
    spec = importlib.util.spec_from_file_location('content_evidence_review', HERE / 'audit_content.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    report = module.check(args)
    assert report['status'] == 'CONTENT_ACCEPTANCE_PASS'
    publication = json.loads(args.publication.read_bytes())
    ref = json.loads(subprocess.check_output(['gh', 'api', 'repos/woobowen/recommend/git/ref/heads/wbw'], cwd=module.REPO))
    assert ref['object']['sha'] == args.commit
    commit = json.loads(subprocess.check_output(['gh', 'api', 'repos/woobowen/recommend/git/commits/' + args.commit], cwd=module.REPO))
    local_tree = subprocess.check_output(['git', 'rev-parse', args.commit + '^{tree}'], cwd=module.REPO, text=True).strip()
    assert commit['tree']['sha'] == local_tree
    readbacks = []
    for row in publication['actual_GitHub_file_readbacks']:
        raw = subprocess.check_output(['gh', 'api', '-H', 'Accept: application/vnd.github.raw+json',
                                       'repos/woobowen/recommend/contents/' + quote(row['path'], safe='/') + '?ref=' + args.commit],
                                      cwd=module.REPO, timeout=30)
        assert raw == module.git_bytes(args.commit, row['path'])
        assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
        readbacks.append(row)
    sys.path.insert(0, '/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
    from transport import Transport
    import gridctl
    transport = Transport('new-server'); transport.identity()
    roots = sorted(item['root'] for item in publication['server_snapshots'])
    source = '''from pathlib import Path
import hashlib,json
rows=[]
for root in ROOTS:
 p=Path(root);raw=(p/'DEPLOYMENT.json').read_bytes();manifest=json.loads(raw)
 assert hashlib.sha256(json.dumps({'source_commit':manifest['source_commit'],'files':manifest['files']},sort_keys=True).encode()).hexdigest()==manifest['snapshot_sha256']
 for row in manifest['files']:
  path=p/row['path'];assert path.is_file() and not path.is_symlink()
  data=path.read_bytes();assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
 rows.append({'root':root,'manifest_sha256':hashlib.sha256(raw).hexdigest(),'manifest':manifest,'files_verified':len(manifest['files'])})
print(json.dumps(rows))
'''.replace('ROOTS', repr(roots))
    actual_snapshots = json.loads(transport.python(source, timeout=60))
    original = {item['root']: item for item in publication['server_snapshots']}
    assert {item['root'] for item in actual_snapshots} == set(original)
    for item in actual_snapshots:
        assert item['manifest'] == original[item['root']]['manifest']
        assert item['manifest_sha256'] == original[item['root']]['manifest_sha256']
        for row in item['manifest']['files']:
            raw = module.git_bytes(item['manifest']['source_commit'], row['path'])
            assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
    summary = json.loads(Path(report['summary']['path']).read_bytes())
    current_jobs = {}
    for row in summary['jobs']:
        value = gridctl.query(transport, row['job_id'])
        assert (value['status'], value['exit_code']) == (row['status'], row['exit_code'])
        assert value['cleanup_verified'] and not value['worker_alive'] and not value['workgroup_live'] and not value.get('worker_may_be_live', False)
        current_jobs[row['job_id']] = {'status': value['status'], 'exit_code': value['exit_code'],
                                       'cleanup_verified': True, 'worker_alive': False, 'workgroup_live': False}
    data_subset = json.loads((HERE / 'data-subset-and-round3-package-review.json').read_bytes())
    assert data_subset['status'] == 'INHERITED_DATA_SHA_SUBSET_AND_PACKAGE_ROUND3_PASS'
    module.verify_identity(data_subset['subset'])
    module.verify_identity(data_subset['actual_current_readonly_stat_receipt'])
    report.update({'independent_fresh_publication_checks_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                   'actual_fresh_GitHub_ref_and_tree': {'ref': ref['object']['sha'], 'tree': local_tree},
                   'actual_fresh_GitHub_raw_bytes': readbacks,
                   'actual_fresh_server_snapshots': [{key: item[key] for key in ('root', 'manifest_sha256', 'files_verified')} for item in actual_snapshots],
                   'actual_fresh_original_ID_terminal_process_checks': current_jobs,
                   'inherited_data_SHA_and_current_loader_stat_independent_review': identity(HERE / 'data-subset-and-round3-package-review.json'),
                   'actual_visual_spotcheck': identity(HERE / 'final-visual-spotcheck.json'),
                   'base_evidence_review_program': identity(HERE / 'audit_content.py'),
                   'remaining_delivery_steps': ['Actual ZIP complete origin/container audit', 'Final delivery receipt commit publication and independent readback'],
                   'program': identity(Path(__file__).resolve())})
    assert not args.report.exists()
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'content_commit': args.commit, 'fresh_GitHub_files': len(readbacks),
                      'fresh_server_snapshots': len(actual_snapshots), 'fresh_terminal_jobs': len(current_jobs)}))


if __name__ == '__main__':
    main()
