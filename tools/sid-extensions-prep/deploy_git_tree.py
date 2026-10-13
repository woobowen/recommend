"""Deploy a hash-bound ordinary Git tree into this task's isolated export root.

No Git hooks, filters, source imports or executable source calls. Only metadata
may change between C and D; any audited source/config change is refused.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile
import datetime

ROOT = Path('/home/bwwu/recommend-worktrees/sid-extensions-p0')


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def deploy(plan, output):
    assert plan['target_root'] == str(ROOT)
    assert len(plan['commit']) == 40 and all(c in '0123456789abcdef' for c in plan['commit'])
    assert all(not p.is_symlink() for p in [ROOT, *ROOT.parents])
    marker = ROOT / '.sid-extensions-deployment.json'
    if ROOT.exists():
        assert marker.is_file() and json.loads(marker.read_text())['stage_id'] == 'sid-extensions-p0', 'unowned existing target'
        assert json.loads(marker.read_text())['source_set_id_sha256'] == plan['source_set_id_sha256'], 'different audited source set'
    archive = Path(plan['archive'])
    support = Path('/home/bwwu/grid-audits/sid-extensions-p0')
    assert support in archive.parents and not archive.is_symlink()
    assert archive.stat().st_size == plan['archive_bytes'] and sha(archive) == plan['archive_sha256']
    expected = {e['path']: e for e in plan['files']}
    assert len(expected) == len(plan['files'])
    ownership = {'stage_id': 'sid-extensions-p0', 'status': 'DEPLOYING',
        'commit': plan['commit'], 'source_set_id_sha256': plan['source_set_id_sha256'],
        'archive_sha256': plan['archive_sha256']}
    staging = output / 'deployment-staging'
    staging.mkdir()
    ROOT.parent.mkdir(parents=True, exist_ok=True)
    assert output.stat().st_dev == ROOT.parent.stat().st_dev, 'atomic deployment requires same filesystem'
    if not ROOT.exists():
        # Initialize the directory with its ownership marker before making the
        # target name visible; interruption leaves a separate task init dir.
        initial = ROOT.with_name(ROOT.name + '.initializing-' + os.environ['GRID_JOB_ID'])
        initial.mkdir()
        with (initial / marker.name).open('x') as f:
            f.write(json.dumps(ownership, indent=2) + '\n')
            f.flush()
            os.fsync(f.fileno())
        assert not ROOT.exists()
        os.rename(initial, ROOT)
    else:
        q = staging / 'ownership-deploying.json'
        q.write_text(json.dumps(ownership, indent=2) + '\n')
        os.replace(q, marker)
    seen = set()
    rows = []
    with tarfile.open(archive, 'r:gz') as tf:
        for member in tf:
            relative = PurePosixPath(member.name)
            assert not relative.is_absolute() and '..' not in relative.parts
            if member.isdir():
                continue
            assert member.isfile() and member.name in expected and member.name not in seen
            e = expected[member.name]
            b = tf.extractfile(member).read(e['bytes'] + 1)
            assert len(b) == e['bytes'] and hashlib.sha256(b).hexdigest() == e['sha256']
            assert hashlib.sha1(b'blob ' + str(len(b)).encode() + b'\0' + b).hexdigest() == e['git_blob']
            p = ROOT / member.name
            assert all(not q.is_symlink() for q in [p, *p.parents])
            p.parent.mkdir(parents=True, exist_ok=True)
            route = 'REUSED_IDENTICAL_TREE_FILE'
            if p.exists() and sha(p) != e['sha256']:
                assert not e['audited_source_config'], 'audited source changed between deliveries'
                assert member.name.startswith(('docs/sid-extensions/', 'manifests/sid-extensions/')) or member.name in ['references/wmh-original/README.md', 'tools/sid-extensions-prep/README.md'], 'nonmetadata overwrite prohibited'
                q = staging / (hashlib.sha256(member.name.encode()).hexdigest() + '.partial')
                with q.open('xb') as f:
                    f.write(b)
                    f.flush()
                    os.fsync(f.fileno())
                assert sha(q) == e['sha256']
                os.replace(q, p)
                route = 'UPDATED_TASK_DELIVERY_METADATA'
            elif not p.exists():
                q = staging / (hashlib.sha256(member.name.encode()).hexdigest() + '.partial')
                with q.open('xb') as f:
                    f.write(b)
                    f.flush()
                    os.fsync(f.fileno())
                assert sha(q) == e['sha256']
                os.link(q, p)
                q.unlink()
                route = 'CREATED_TREE_FILE'
            assert p.stat().st_size == e['bytes'] and sha(p) == e['sha256']
            rows.append(dict(e, actual_path=str(p), actual_sha256=e['sha256'], route=route))
            seen.add(member.name)
    assert seen == set(expected)
    actual_files = {str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() or p.is_symlink()}
    assert actual_files - {marker.name} == seen, 'unexpected hidden/unrecorded tree file'
    receipt = {'stage_id': 'sid-extensions-p0', 'commit': plan['commit'], 'source_set_id_sha256': plan['source_set_id_sha256'],
               'files': len(rows), 'status': 'SUCCEEDED', 'source_code_execution': False, 'model_execution': False,
               'observed_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'rows': rows}
    (output / 'RESULT.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    q = staging / 'ownership-final.json'
    q.write_text(json.dumps({k: v for k, v in receipt.items() if k != 'rows'}, indent=2) + '\n')
    os.replace(q, marker)


if __name__ == '__main__':
    deploy(json.loads(Path(sys.argv[1]).read_text()), Path(os.environ['GRID_JOB_DIR']) / 'output')
