#!/usr/bin/env python3
"""Package only fixed-commit LC-Rec sources/evidence and external publication receipts."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
import zipfile


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def safe(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or '..' in path.parts or '\\' in name:
        raise ValueError('Unsafe member path')
    return name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--receipts', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', args.commit):
        raise ValueError('Full fixed DELIVERY_COMMIT required')
    if args.output.exists() or args.output.is_symlink():
        raise ValueError('Preserve existing delivery; choose a new task-specific output')

    def git(*parts):
        return subprocess.check_output(['git', '-C', str(args.repo), *parts])

    if git('rev-parse', args.commit+'^{commit}').decode().strip() != args.commit:
        raise ValueError('Commit mismatch')
    wanted = ['lc_rec/offload', 'lc_rec/vendor/original',
              'lc_rec/evidence/cpu_input_asset_hashes.json',
              'lc_rec/evidence/aux_pairing/output/auxiliary_pairing.json',
              'lc_rec/evidence/assets/asset_manifest.json',
              'lc_rec/evidence/assets/hf_summary.json',
              'lc_rec/evidence/assets/hf_model.safetensors.index.json',
              'lc_rec/evidence/assets/hf_config.json',
              'lc_rec/evidence/assets/hf_tokenizer_config.json',
              'lc_rec/evidence/assets/hf_special_tokens_map.json',
              'lc_rec/evidence/assets/hf_LICENSE',
              'lc_rec/evidence/assets/hf_README.md',
              'lc_rec/evidence/official_manifest.json']
    entries = git('ls-tree', '-rz', args.commit, '--', *wanted).split(b'\0')
    members = {}
    source_rows = []
    for entry in entries:
        if not entry:
            continue
        fields, path = entry.split(b'\t', 1)
        mode, kind, oid = fields.decode().split()
        if mode not in ['100644', '100755'] or kind != 'blob':
            raise ValueError('Symlink/submodule is not allowed in audit archive')
        path = path.decode()
        name = safe('source/'+path)
        blob = git('cat-file', 'blob', oid)
        if len(blob) > 10*1024**2 or blob.startswith(b'version https://git-lfs.github.com/spec/v1'):
            raise ValueError('Large asset or LFS pointer is not review source')
        members[name] = blob
        source_rows.append({'member': name, 'git_blob': oid, 'commit': args.commit})
    for item in sorted(args.receipts.rglob('*')):
        if item.is_symlink():
            raise ValueError('Receipt symlink rejected')
        if not item.is_file():
            continue
        blob = item.read_bytes()
        if len(blob) > 5*1024**2:
            raise ValueError('Large external receipt rejected')
        name = safe('publication/'+item.relative_to(args.receipts).as_posix())
        if name in members:
            raise ValueError('Duplicate member')
        members[name] = blob
    required = ['source/lc_rec/offload/REVIEW_ENTRY.md',
                'source/lc_rec/offload/FAITHFULNESS.md',
                'source/lc_rec/offload/scripts/offload_probe.py',
                'source/lc_rec/offload/config/ds_optimizer_only.json',
                'source/lc_rec/offload/config/formal_candidate.json',
                'publication/DELIVERY_RECEIPT.json']
    if any(name not in members for name in required):
        raise ValueError('Required review material missing')
    patterns = [rb'-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----',
                rb'gh[pousr]_[A-Za-z0-9]{25,}', rb'hf_[A-Za-z0-9]{28,}',
                rb'AKIA[0-9A-Z]{16}',
                rb'(?i)Authorization\s*[:=]\s*[\"\x27]?Bearer\s+[A-Za-z0-9._-]{20,}']
    suspect = [{'path': name, 'rule': i} for name, blob in members.items()
               for i, pattern in enumerate(patterns) if re.search(pattern, blob)]
    if suspect:
        raise ValueError('Potential secret; inspect privately: '+json.dumps(suspect))
    manifest = {'delivery_commit': args.commit,
                'note': 'This manifest excludes itself; ZIP SHA-256 is outside ZIP.',
                'files': [{'path': name, 'bytes': len(blob), 'sha256': sha(blob)}
                          for name, blob in sorted(members.items())],
                'git_sources': source_rows}
    members['MEMBER_SHA256.json'] = (json.dumps(manifest, ensure_ascii=False, indent=2)+'\n').encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, blob in sorted(members.items()):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, blob)
    if args.output.stat().st_size > 50*1024**2:
        raise ValueError('Review archive exceeds 50MiB; preserve and report')
    with tempfile.TemporaryDirectory(prefix='lc-rec-offload-verify-') as tmp:
        dest = Path(tmp)
        with zipfile.ZipFile(args.output) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or set(names) != set(members):
                raise ValueError('Archive member set/duplicates mismatch')
            if archive.testzip() is not None:
                raise ValueError('CRC failed')
            for name in names:
                safe(name)
                target = dest/name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
        unpacked = json.loads((dest/'MEMBER_SHA256.json').read_text())
        if {row['path'] for row in unpacked['files']} | {'MEMBER_SHA256.json'} != set(names):
            raise ValueError('Unpacked manifest completeness mismatch')
        for row in unpacked['files']:
            blob = (dest/row['path']).read_bytes()
            if len(blob) != row['bytes'] or sha(blob) != row['sha256']:
                raise ValueError('Unpacked hash mismatch')
        if len(list(dest.rglob('*'))) < len(names):
            raise ValueError('Extraction incomplete')
    result = {'path': str(args.output.absolute()), 'bytes': args.output.stat().st_size,
              'sha256': sha(args.output.read_bytes()), 'members': len(members),
              'manifest_members_excluding_itself': len(manifest['files']),
              'crc_verified': True, 'actual_extraction_and_member_hashes_verified': True,
              'duplicate_paths': False, 'unsafe_paths': False, 'secret_pattern_hits': 0,
              'delivery_commit': args.commit}
    args.output.with_suffix(args.output.suffix+'.sha256').write_text(result['sha256']+'  '+args.output.name+'\n')
    args.output.with_suffix(args.output.suffix+'.receipt.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
