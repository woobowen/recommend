#!/usr/bin/env python3
"""Build or verify one bounded evidence ZIP; integrity is not scientific PASS."""
from pathlib import Path, PurePosixPath
import argparse
import datetime as dt
import hashlib
import json
import re
import stat
import unicodedata
import zipfile
import zlib

SECRET_PATTERNS = {
    'private_key_header': re.compile(rb'-----BEGIN (?:RSA |OPENSSH |EC |DSA |ENCRYPTED )?PRIVATE KEY-----'),
    'github_token': re.compile(rb'(?<![A-Za-z0-9])gh[pousr]_[A-Za-z0-9]{30,}'),
    'github_pat': re.compile(rb'github_pat_[A-Za-z0-9_]{40,}'),
    'huggingface_token': re.compile(rb'(?<![A-Za-z0-9])hf_[A-Za-z0-9]{30,}'),
    'aws_access_key': re.compile(rb'(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])'),
}

def safe_name(name):
    path = PurePosixPath(name)
    assert name == unicodedata.normalize('NFC', name), name
    assert path.as_posix() == name, name
    assert not path.is_absolute() and '..' not in path.parts and '.' not in path.parts, name
    assert not any(c in name for c in ['\\', ':', '\x00', '\n', '\r']), name
    assert name and not name.endswith('/')
    assert not any(p in {'.ssh', '.env', '.netrc', '.npmrc', '.pypirc', 'id_rsa', 'id_ed25519', 'credentials.json'}
                   or p.startswith('.env.') for p in path.parts), name

def scan_sensitive_bytes(data, name):
    hits = [label for label, pattern in SECRET_PATTERNS.items() if pattern.search(data)]
    assert not hits, (name, 'potential secret pattern', hits)

def digest_stream(stream):
    digest, crc, size = hashlib.sha256(), 0, 0
    for block in iter(lambda: stream.read(4*1024*1024), b''):
        digest.update(block)
        crc = zlib.crc32(block, crc)
        size += len(block)
    return dict(bytes=size, sha256=digest.hexdigest(), crc32=f'{crc & 0xffffffff:08x}')

def file_identity(path):
    with path.open('rb') as stream:
        return digest_stream(stream)

def verify(path, manifest_name='manifest.json'):
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        names = [i.filename for i in entries]
        assert len(names) == len(set(names)) == len({s.casefold() for s in names})
        for entry in entries:
            safe_name(entry.filename)
            assert not entry.is_dir() and not stat.S_ISLNK(entry.external_attr >> 16)
            assert entry.file_size <= 100_000_000, entry.filename
        assert archive.testzip() is None, 'ZIP CRC check failed'
        manifest = json.loads(archive.read(manifest_name))
        records = {r['path']: r for r in manifest['files']}
        assert len(records) == len(manifest['files'])
        assert set(names) == set(records) | {manifest_name}
        for entry in entries:
            data = archive.read(entry)
            scan_sensitive_bytes(data, entry.filename)
            if entry.filename == manifest_name:
                continue
            actual = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                          crc32=f'{zlib.crc32(data) & 0xffffffff:08x}')
            expected = records[entry.filename]
            assert all(actual[k] == expected[k] for k in ['bytes', 'sha256', 'crc32']), entry.filename
            assert entry.file_size == actual['bytes'] and f'{entry.CRC:08x}' == actual['crc32']
        manifest_sha = hashlib.sha256(archive.read(manifest_name)).hexdigest()
    return dict(status='PACKAGE_INTEGRITY_PASS', zip=str(path.resolve()), **file_identity(path),
                members=len(entries), payload_members=len(records),
                member_sha256_pass=True, member_sizes_pass=True, crc_pass=True,
                safe_paths_pass=True, duplicate_paths=0, windows_case_collisions=0,
                sensitive_pattern_scan_pass=True, scan_scope='all actual uncompressed member bytes, including manifest',
                manifest_sha256=manifest_sha, scientific_review_status=manifest['scientific_review_status'],
                limitation='Cryptographic/package integrity does not replace the separate real scientific reviews.')

def build(staging, output):
    assert staging.is_dir() and not staging.is_symlink() and not output.exists()
    manifest_path = staging/'manifest.json'
    assert not manifest_path.exists(), 'Preserve previous prepared package; do not silently regenerate.'
    final_review = json.loads((staging/'review/final-internal-review.json').read_text())
    assert final_review['status'] in ['FINAL_INTERNAL_REVIEW_PASS', 'PARTIAL', 'BLOCKED', 'FINAL_INTERNAL_REVIEW_FAILED']
    records, secret_scan, seen = [], [], set()
    for path in sorted(staging.rglob('*')):
        assert not path.is_symlink(), str(path)
        if not path.is_file():
            continue
        name = path.relative_to(staging).as_posix()
        safe_name(name)
        assert name.casefold() not in seen, name
        seen.add(name.casefold())
        assert path.suffix not in {'.ckpt', '.bin', '.safetensors', '.tfrecord', '.pyc'}, name
        assert path.name != 'item_embeddings.pt', name
        assert path.stat().st_size <= 100_000_000, (name, path.stat().st_size)
        content = path.read_bytes()
        scan_sensitive_bytes(content, name)
        secret_scan.append(name)
        records.append(dict(path=name, **file_identity(path)))
    assert 'manifest.json'.casefold() not in seen
    assert sum(r['bytes'] for r in records) <= 650_000_000
    manifest = dict(schema='grid-a2-fd-final-evidence-manifest-v1',
                    created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                    scientific_review_status=final_review['status'], files=records,
                    member_hash_scope='All payload members; manifest excludes itself to avoid circular self-hashing. ZIP verification records its SHA separately.',
                    secret_scan=dict(status='PASS', patterns=list(SECRET_PATTERNS),
                                     payload_members=len(secret_scan), scope='all payload bytes regardless of suffix',
                                     private_credentials_collected=False),
                    excluded_assets='Checkpoints, T5 weights, TFRecords and full environments remain on the server; see assets-not-copied.json for identity and recovery.')
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for name in sorted([r['path'] for r in records] + ['manifest.json']):
            archive.write(staging/name, arcname=name)
    report = verify(output)
    report['sensitive_pattern_scan_pass'] = True
    report['scan_patterns'] = list(SECRET_PATTERNS)
    report['verified_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
    receipt = output.with_suffix(output.suffix+'.verification.json')
    assert not receipt.exists()
    receipt.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    output.with_suffix(output.suffix+'.sha256').write_text(report['sha256']+'  '+output.name+'\n')
    return report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    command = sub.add_parser('build')
    command.add_argument('staging', type=Path)
    command.add_argument('output', type=Path)
    command = sub.add_parser('verify')
    command.add_argument('zip', type=Path)
    args = parser.parse_args()
    report = build(args.staging, args.output) if args.action == 'build' else verify(args.zip)
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
