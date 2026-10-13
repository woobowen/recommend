"""Assemble task-transferred chunks and verify fixed LFS whole-file identities."""
import hashlib
import json
import os
from pathlib import Path
import sys
import datetime


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def assemble(plan, output):
    assert plan['baseline_sha'] == '563dc50b1c7e9e2e05cdfddd13e91a3d1888001e'
    root = Path('/data/bwwu/sid-extensions-assets/563dc50b1c7e')
    chunk_root = Path('/home/bwwu/grid-audits/sid-extensions-p0/inputs/lfs-chunks')
    rows = []
    for e in plan['entries']:
        p = Path(e['destination'])
        assert p.is_absolute() and '..' not in p.parts and root in p.parents
        assert all(not q.is_symlink() for q in [p, *p.parents])
        p.parent.mkdir(parents=True, exist_ok=True)
        v = os.statvfs(p.parent)
        assert v.f_bavail * v.f_frsize > 100 * 1024**3 + e['bytes']
        if not p.exists():
            q = p.with_name(p.name + '.partial')
            assert not q.exists(), 'retain prior incomplete assembly for inspection'
            cursor = 0
            whole = hashlib.sha256()
            with q.open('xb') as dst:
                for part in e['parts']:
                    assert part['object_sha256'] == e['sha256'] and part['start'] == cursor
                    assert part['bytes'] == part['end'] - part['start'] + 1
                    source = Path(part['remote'])
                    assert chunk_root in source.parents and '..' not in source.parts
                    assert not source.is_symlink() and source.stat().st_size == part['bytes']
                    h = hashlib.sha256()
                    with source.open('rb') as src:
                        for b in iter(lambda: src.read(4 * 1024**2), b''):
                            h.update(b)
                            whole.update(b)
                            dst.write(b)
                    assert h.hexdigest() == part['sha256']
                    cursor += part['bytes']
                dst.flush()
                os.fsync(dst.fileno())
            assert cursor == e['bytes'] and whole.hexdigest() == e['sha256']
            q.chmod(0o444)
            os.link(q, p)
            q.unlink()
        assert p.stat().st_size == e['bytes'] and sha(p) == e['sha256']
        row = dict(e, actual_path=str(p), actual_sha256=e['sha256'], nlink=p.stat().st_nlink,
                   mode=oct(p.stat().st_mode & 0o777), integrity='PAYLOAD_PRESENT_HASH_MATCHED',
                   route='FIXED_RANGE_RELAY_AND_HASH_VERIFIED_SERVER_ASSEMBLY', model_load='MODEL_LOAD_NOT_RUN')
        rows.append(row)
        with (output / 'files.jsonl').open('a') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    (output / 'RESULT.json').write_text(json.dumps({'status': 'SUCCEEDED', 'entries': len(rows),
        'bytes': sum(e['bytes'] for e in rows), 'rows': rows, 'observed_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'model_execution': False, 'source_code_execution': False}, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    assemble(json.loads(Path(sys.argv[1]).read_text()), Path(os.environ['GRID_JOB_DIR']) / 'output')
