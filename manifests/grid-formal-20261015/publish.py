"""Push only wbw, read actual GitHub blobs, and verify existing run snapshots."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from urllib.parse import quote

REPO = Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
SUPPORT = Path('/home/addaswsw/recommend-support/grid-formal-20261015')
REMOTE = '/home/bwwu/grid-audits/sid-extensions-p0/grid-formal-20261015'


def command(argv):
    return subprocess.check_output(argv, cwd=REPO, text=True).strip()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--commit', required=True)
    a = p.parse_args()
    assert command(['git', 'branch', '--show-current']) == 'wbw'
    assert command(['git', 'rev-parse', 'HEAD']) == a.commit
    assert not command(['git', 'diff', '--name-only']) and not command(['git', 'diff', '--cached', '--name-only'])
    summary = json.loads(command(['git', 'show', a.commit+':manifests/grid-formal-20261015/EXECUTION_SUMMARY.json']))
    assert summary['status'] in ('READY_FOR_REVIEW', 'PARTIAL', 'BLOCKED')
    subprocess.run(['git', 'push', 'origin', 'HEAD:refs/heads/wbw'], cwd=REPO, check=True)
    remote = command(['git', 'ls-remote', 'origin', 'refs/heads/wbw']).split()[0]
    assert remote == a.commit
    ref = json.loads(command(['gh', 'api', 'repos/woobowen/recommend/git/ref/heads/wbw']))
    assert ref['object']['sha'] == a.commit
    critical = ['manifests/grid-formal-20261015/FINAL_RECEIPT.md',
        'manifests/grid-formal-20261015/EXECUTION_SUMMARY.json',
        'docs/sid-extensions/experiments/shuffle.md', 'docs/sid-extensions/experiments/mlp.md',
        'docs/sid-extensions/experiments/attention.md',
        'experiments/formal_20261015/deploy.py', 'experiments/formal_20261015/run.py',
        'experiments/formal_20261015/train.py',
        'experiments/beauty_extensions/author/models.py',
        'experiments/beauty_extensions/observation.py',
        'experiments/formal_20261015/specs/toys-mlp.json',
        'experiments/formal_20261015/specs/sports-mlp.json']
    for route in ('a', 'b'):
        for n in (0, 3, 4):
            critical.append(f'experiments/formal_20261015/specs/attention-beauty-{route}-{n}.json')
    blobs = []
    for name in critical:
        record = json.loads(command(['gh', 'api', 'repos/woobowen/recommend/contents/'+quote(name, safe='/')+'?ref='+a.commit]))
        assert record.get('encoding') == 'base64', name
        data = base64.b64decode(record['content'])
        assert data == subprocess.check_output(['git', 'show', a.commit+':'+name], cwd=REPO), name
        blobs.append({'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'github_blob_sha': record['sha']})
    sys.path.insert(0, '/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
    from transport import Transport, atomic
    t = Transport('new-server')
    t.identity()
    snapshots = sorted({job['snapshot_sha256'] for job in summary['jobs']})
    code = '''import pathlib,json,hashlib
root=pathlib.Path(ROOT)
results=[]
for digest in DIGESTS:
 p=root/digest[:16]
 assert p.is_dir() and not p.is_symlink()
 mf=p/'DEPLOYMENT.json';assert not mf.is_symlink()
 d=json.loads(mf.read_text());assert d['snapshot_sha256']==digest
 for row in d['files']:
  f=p/row['path'];assert f.is_file() and not f.is_symlink()
  assert f.stat().st_size==row['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==row['sha256']
 results.append({'root':str(p),'manifest_sha256':hashlib.sha256(mf.read_bytes()).hexdigest(),'manifest':d})
print(json.dumps(results))
'''.replace('ROOT', repr(REMOTE+'/snapshots')).replace('DIGESTS', repr(snapshots))
    exports = json.loads(t.python(code, timeout=60))
    for export in exports:
        manifest = export['manifest']
        for record in manifest['files']:
            data = subprocess.check_output(['git', 'show', manifest['source_commit']+':'+record['path']], cwd=REPO)
            assert len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['sha256'], record['path']
        # Historical failed CPU entry retains its old run.py. Each actual run
        # is checked against its own fixed commit, not relabelled final HEAD.
        export['matches_own_fixed_run_commit'] = True
    result = {'commit': a.commit, 'branch': 'wbw', 'remote_ref_verified': True,
        'github_ref_verified': True, 'key_file_bytes_verified': True,
        'github_actual_blob_readbacks': blobs, 'server_snapshot_mapping_verified': True,
        'server_exports': exports, 'server_has_git': False,
        'no_new_snapshot_created_for_publication': True,
        'identity_kind': 'immutable exports verified against each original run commit; final GitHub contains those ancestor commits and published results',
        'other_branches_written': False}
    atomic(SUPPORT/'publication.json', result)
    print(json.dumps({'commit': a.commit, 'GitHub_blobs_verified': len(blobs),
        'existing_server_snapshots_verified': len(exports), 'new_deployments': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
