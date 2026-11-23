"""Retrieve immutable release payloads and verify actual bytes before extraction."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import urllib.request
import zipfile


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--asset', required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    spec = json.loads(args.manifest.read_text(encoding='utf-8'))['assets'][args.asset]
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / args.asset
    if not archive.exists():
        part = archive.with_suffix('.zip.partial')
        with urllib.request.urlopen(spec['download_url'], timeout=180) as response, part.open('wb') as f:
            shutil.copyfileobj(response, f, 8*1024*1024)
        assert part.stat().st_size == spec['bytes'] and digest(part) == spec['sha256'], 'Download mismatch'
        part.replace(archive)
    assert archive.stat().st_size == spec['bytes'] and digest(archive) == spec['sha256'], 'Archive mismatch'
    dest = args.output / archive.stem
    with zipfile.ZipFile(archive) as z:
        assert len(z.namelist()) == len(set(z.namelist())), 'Duplicate member'
        for info in z.infolist():
            p = PurePosixPath(info.filename)
            assert not p.is_absolute() and '..' not in p.parts and '\\' not in info.filename and ':' not in info.filename
            assert (info.external_attr >> 16) & 0o170000 != 0o120000, 'Symlink member'
        assert z.testzip() is None, 'CRC mismatch'
        for row in spec['members']:
            target = dest / row['member']
            if target.exists():
                assert target.stat().st_size == row['bytes'] and digest(target) == row['sha256'], str(target)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(row['member']) as source, target.open('wb') as f:
                    shutil.copyfileobj(source, f, 8*1024*1024)
                assert target.stat().st_size == row['bytes'] and digest(target) == row['sha256'], str(target)
        if 'MANIFEST.json' in z.namelist():
            (dest / 'MANIFEST.json').write_bytes(z.read('MANIFEST.json'))
    print(json.dumps({'archive': str(archive), 'extracted': str(dest), 'sha256': spec['sha256'], 'verified_members': len(spec['members'])}))


if __name__ == '__main__': main()
