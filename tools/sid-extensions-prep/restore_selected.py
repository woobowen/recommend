"""Restore only hash-bound ordinary files from selected preserved TAR batches.

Standard library only. Never imports, deserializes, or executes source material.
Existing matching data may be exposed through explicitly recorded read-only-use
symlinks; new payloads have independent inodes. All selected batch chunks and
whole TAR bytes are hashed. Other archives are never visited.
"""
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import tarfile
import datetime


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def atomic(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name + '.partial')
    q.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(q, p)


def target_path(raw, roots):
    p = Path(raw)
    assert p.is_absolute() and '..' not in p.parts
    assert any(p != r and r in p.parents for r in roots), 'target outside task roots'
    for parent in p.parents:
        assert not parent.is_symlink(), 'symlink target ancestor'
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


class Parts(io.RawIOBase):
    def __init__(self, manifest, root):
        self.parts = manifest['parts']
        self.root = Path(root)
        self.i = 0
        self.file = None
        self.part_hash = None
        self.whole_hash = hashlib.sha256()
        self.part_bytes = 0
        self.total = 0
        self.checked = []

    def readable(self):
        return True

    def read(self, n=-1):
        if n < 0:
            raise ValueError('unbounded archive read prohibited')
        result = bytearray()
        while len(result) < n and self.i < len(self.parts):
            meta = self.parts[self.i]
            if self.file is None:
                p = self.root / Path(meta['path']).name
                assert not p.is_symlink() and p.stat().st_size == meta['bytes']
                self.file = p.open('rb')
                self.part_hash = hashlib.sha256()
                self.part_bytes = 0
            b = self.file.read(n - len(result))
            if b:
                result.extend(b)
                self.part_hash.update(b)
                self.whole_hash.update(b)
                self.part_bytes += len(b)
                self.total += len(b)
            else:
                assert self.part_bytes == meta['bytes']
                assert self.part_hash.hexdigest() == meta['sha256'], 'archive chunk mismatch'
                self.checked.append({'index': meta['index'], 'bytes': self.part_bytes,
                                     'sha256': self.part_hash.hexdigest()})
                self.file.close()
                self.file = None
                self.i += 1
        return bytes(result)


def prepare(plan, output):
    roots = [Path('/home/bwwu/grid-audits/sid-extensions-p0/restored'),
             Path('/data/bwwu/sid-extensions-assets/563dc50b1c7e')]
    reserve = 100 * 1024**3
    for root in roots:
        root.mkdir(parents=True, exist_ok=True)
        v = os.statvfs(root)
        assert v.f_bavail * v.f_frsize > reserve + plan['maximum_new_bytes']
    rows = []
    pending = []
    journal = Path(output) / 'files.jsonl'
    with journal.open('a') as log:
        def record(e, p, route):
            assert p.stat().st_size == e['bytes'] and digest(p) == e['sha256']
            row = dict(e, actual_path=str(p), actual_sha256=e['sha256'],
                       integrity='PAYLOAD_PRESENT_HASH_MATCHED', route=route,
                       resolved_target=str(p.resolve()), mode=oct(stat.S_IMODE(p.stat().st_mode)),
                       nlink=p.stat().st_nlink, read_only_use=True)
            rows.append(row)
            log.write(json.dumps(row, ensure_ascii=False) + '\n')
            log.flush()
        for e in plan['entries']:
            p = target_path(e['destination'], roots)
            if p.exists() or p.is_symlink():
                record(e, p, 'REUSED_EXISTING_TASK_ENTITY')
                continue
            candidate = Path(e['candidate']) if e.get('candidate') else None
            if candidate and candidate.is_file():
                assert candidate.stat().st_size == e['bytes'] and digest(candidate) == e['sha256'], 'candidate identity mismatch'
                os.symlink(str(candidate.resolve()), p)
                record(e, p, 'READ_ONLY_USE_SYMLINK_TO_EXISTING_MATCHED_ENTITY')
            else:
                pending.append(e)
        for batch in plan['batches']:
            wanted = {e['member_path']: e for e in pending if e['batch'] == batch['batch']}
            if not wanted:
                continue
            mp = Path(batch['manifest_path'])
            assert digest(mp) == batch['manifest_sha256']
            m = json.loads(mp.read_text())
            bound = {x['archive_path']: x for x in m['preserved']}
            for name, e in wanted.items():
                pp = PurePosixPath(name)
                assert not pp.is_absolute() and '..' not in pp.parts
                assert bound[name]['type'] == 'regular'
                assert (bound[name]['bytes'], bound[name]['sha256']) == (e['bytes'], e['sha256'])
            stream = Parts(m, batch['chunk_directory'])
            found = set()
            with tarfile.open(fileobj=stream, mode='r|') as tf:
                for member in tf:
                    if member.name not in wanted:
                        continue
                    assert member.name not in found and member.isfile()
                    e = wanted[member.name]
                    assert member.size == e['bytes']
                    p = target_path(e['destination'], roots)
                    q = p.with_name(p.name + '.partial')
                    assert not p.exists() and not q.exists(), 'refuse overwrite or incomplete prior attempt'
                    h = hashlib.sha256()
                    count = 0
                    with tf.extractfile(member) as src, q.open('xb') as dst:
                        for b in iter(lambda: src.read(4 * 1024 * 1024), b''):
                            dst.write(b)
                            h.update(b)
                            count += len(b)
                        dst.flush()
                        os.fsync(dst.fileno())
                    assert count == e['bytes'] and h.hexdigest() == e['sha256']
                    q.chmod(0o444)
                    os.link(q, p)
                    q.unlink()
                    record(e, p, 'EXTRACTED_FROM_VERIFIED_ARCHIVE_SELECTED_MEMBER')
                    found.add(member.name)
            while stream.read(4 * 1024 * 1024):
                pass
            assert found == set(wanted)
            assert stream.total == m['archive_bytes'] and stream.whole_hash.hexdigest() == m['archive_sha256']
            atomic(Path(output) / (batch['batch'].replace('/', '-') + '.json'),
                   {'batch': batch['batch'], 'manifest_sha256': batch['manifest_sha256'],
                    'whole_tar_sha256': stream.whole_hash.hexdigest(), 'bytes': stream.total,
                    'chunks': stream.checked, 'selected_members': len(found),
                    'limit': 'Only selected member bodies compared; not a rerun of the old full-library audit.'})
    assert len(rows) == len(plan['entries'])
    atomic(Path(output) / 'RESULT.json', {'status': 'SUCCEEDED', 'entries': len(rows),
           'payload_bytes': sum(e['bytes'] for e in rows),
           'observed_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'model_execution': False, 'source_code_execution': False,
           'reserve_bytes': reserve, 'rows': rows})


if __name__ == '__main__':
    plan_path = Path(sys.argv[1])
    output = Path(os.environ['GRID_JOB_DIR']) / 'output'
    prepare(json.loads(plan_path.read_text()), output)
