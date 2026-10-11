"""Restore original Qwen shards from Git LFS transport parts, checking SHA-256."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(16 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'codex/weights_manifest.json').read_text(encoding='utf-8'))
    required = sum(r['bytes'] for r in manifest['files'] if not (ROOT / r['path']).exists())
    if not args.verify_only and shutil.disk_usage(ROOT).free < required:
        raise RuntimeError(f'Need {required / 1024**3:.2f} GiB free to restore original weights')
    for item in manifest['files']:
        dest = ROOT / item['path']
        if dest.exists():
            if dest.stat().st_size != item['bytes'] or digest(dest) != item['sha256']:
                raise RuntimeError(f'Existing weight hash mismatch; refusing to overwrite {dest}')
            print('Verified:', dest.name, flush=True)
            continue
        for part in item['parts']:
            p = ROOT / part['path']
            if not p.is_file() or p.stat().st_size != part['bytes'] or digest(p) != part['sha256']:
                raise RuntimeError(f'Missing or mismatched LFS part: {p}; run git lfs pull first')
        if args.verify_only:
            print('Verified transport parts:', dest.name, flush=True)
            continue
        temporary = dest.with_suffix(dest.suffix + '.assembling')
        h = hashlib.sha256()
        with temporary.open('wb') as f:
            for part in item['parts']:
                with (ROOT / part['path']).open('rb') as src:
                    for chunk in iter(lambda: src.read(16 * 1024**2), b''):
                        f.write(chunk)
                        h.update(chunk)
        if temporary.stat().st_size != item['bytes'] or h.hexdigest() != item['sha256']:
            raise RuntimeError(f'Restored hash mismatch: {temporary}')
        temporary.replace(dest)
        print('Restored:', dest.name, flush=True)


if __name__ == '__main__':
    main()
