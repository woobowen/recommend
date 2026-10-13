"""Fresh read-only hashes of explicit task tree/assets, no deserialization."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys


def verify(plan, output):
    roots = [Path('/home/bwwu/recommend-worktrees/sid-extensions-p0'),
             Path('/home/bwwu/grid-audits/sid-extensions-p0/restored'),
             Path('/data/bwwu/sid-extensions-assets/563dc50b1c7e')]
    assert len(plan['entries']) == len({e['path'] for e in plan['entries']})
    rows = []
    errors = []
    with (output / 'files.jsonl').open('w') as log:
        for e in plan['entries']:
            p = Path(e['path'])
            assert p.is_absolute() and '..' not in p.parts and any(r in p.parents for r in roots)
            before = p.stat()
            h = hashlib.sha256()
            with p.open('rb') as f:
                for b in iter(lambda: f.read(4 * 1024**2), b''):
                    h.update(b)
            after = p.stat()
            stable = tuple(getattr(before, k) for k in ['st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns']) == tuple(getattr(after, k) for k in ['st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns'])
            good = stable and after.st_size == e['bytes'] and h.hexdigest() == e['sha256']
            row = dict(e, actual_sha256=h.hexdigest(), actual_bytes=after.st_size, stable_during_read=stable,
                       result='PASS' if good else 'FAIL', symlink=p.is_symlink(), resolved_target=str(p.resolve()),
                       mode=oct(after.st_mode & 0o777), nlink=after.st_nlink)
            rows.append(row)
            log.write(json.dumps(row, ensure_ascii=False) + '\n')
            log.flush()
            if not good:
                errors.append(row)
    result = {'status': 'PASS' if not errors else 'FAIL', 'entries': len(rows),
              'bytes': sum(e['actual_bytes'] for e in rows), 'errors': errors,
              'source_set_id_sha256': plan.get('source_set_id_sha256'), 'commit': plan.get('commit'),
              'observed_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'source_code_execution': False, 'model_execution': False,
              'limits': 'Fresh non-atomic byte traversal; per-file stability checked, no runtime/semantic/model compatibility assertion.'}
    (output / 'RESULT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    assert not errors, 'delivery content mismatch'


if __name__ == '__main__':
    verify(json.loads(Path(sys.argv[1]).read_text()), Path(os.environ['GRID_JOB_DIR']) / 'output')
