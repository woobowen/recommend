#!/usr/bin/env python3
"""Package fixed A0 Normal review content with externally verified receipts."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
import zipfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    assert not path.is_absolute() and '..' not in path.parts and '\\' not in name
    assert name and not name.endswith('/')


def selected(name):
    base = 'lc_rec/letter/'
    relative = name.removeprefix(base)
    return (
        relative.startswith(('scripts/', 'config/', 'evidence/formal_det_20261112/',
                             'review/formal_det_20261112/'))
        or relative in {
            'A0_NORMAL_FINAL_REVIEW_ENTRY.md',
            'ATTENTION_BACKWARD_DIAGNOSIS_REVIEW_ENTRY.md',
            'RESUME_DIAGNOSIS_REVIEW_ENTRY.md',
            'NORMAL_REVIEW_ENTRY.md', 'AUTOGRAD_REVIEW_ENTRY.md',
            'evidence/formal_20261111/precision_approval.json',
        }
        or relative.startswith('vendor/LETTER/LETTER-LC-Rec/')
        and relative.endswith(('.py', '.sh', '.json'))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('commit', 'receipts', 'output', 'windows'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    commit = subprocess.check_output(['git', 'rev-parse', args.commit], cwd=repo,
                                     text=True).strip()
    paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', commit,
                                    'lc_rec/letter'], cwd=repo, text=True).splitlines()
    payload = {name: subprocess.check_output(['git', 'show', commit + ':' + name],
                                             cwd=repo)
               for name in paths if selected(name)}
    entry = 'lc_rec/letter/A0_NORMAL_FINAL_REVIEW_ENTRY.md'
    assert entry in payload
    receipt_root = Path(args.receipts)
    receipts = sorted(p for p in receipt_root.rglob('*') if p.is_file())
    assert receipts, 'external publication and deployment receipts are required'
    for path in receipts:
        assert not path.is_symlink()
        name = 'external_receipts/' + str(path.relative_to(receipt_root))
        assert name not in payload
        payload[name] = path.read_bytes()
    payload['FIXED_IDENTITY.json'] = (json.dumps({
        'DELIVERY_COMMIT': commit, 'entry': entry,
        'source': 'git show fixed commit; publication/deployment receipts external',
    }, indent=2) + '\n').encode()
    patterns = [rb'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----',
                rb'\bgh[pousr]_[A-Za-z0-9]{25,}\b',
                rb'\bgithub_pat_[A-Za-z0-9_]{30,}\b',
                rb'\bsk-(?:proj-)?[A-Za-z0-9_-]{35,}\b',
                rb'\bhf_[A-Za-z0-9]{30,}\b', rb'\bAKIA[A-Z0-9]{16}\b']
    rows = []
    for name, data in sorted(payload.items()):
        safe_name(name)
        assert not name.endswith(('.pt', '.pth', '.bin', '.safetensors', '.model'))
        assert 'predictions.rank' not in name
        assert len(data) <= 16 * 1024**2
        assert not any(re.search(pattern, data) for pattern in patterns), name
        rows.append({'path': name, 'bytes': len(data), 'sha256': digest(data)})
    manifest = 'MEMBER_SHA256.json'
    assert manifest not in payload
    payload[manifest] = (json.dumps({
        'DELIVERY_COMMIT': commit, 'files': rows,
        'coverage': 'Every payload except this manifest; its SHA in external receipt',
    }, indent=2) + '\n').encode()
    output, windows = Path(args.output), Path(args.windows)
    assert output.resolve() != windows.resolve()
    assert not output.exists() and not windows.exists(), 'preserve existing archives'
    output.parent.mkdir(parents=True, exist_ok=True)
    windows.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as archive:
        for name, data in sorted(payload.items()):
            archive.writestr(name, data)
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        members = archive.infolist()
        assert len(members) == len({m.filename for m in members}) == len(payload)
        assert {m.filename for m in members} == set(payload)
        for member in members:
            safe_name(member.filename)
            assert (member.external_attr >> 16) & 0o170000 != 0o120000
        with tempfile.TemporaryDirectory(prefix='a0-normal-verify-',
                                         dir=output.parent) as temp:
            archive.extractall(temp)
            for name, data in payload.items():
                path = Path(temp) / name
                assert path.is_file() and not path.is_symlink()
                assert path.stat().st_size == len(data)
                assert digest(path.read_bytes()) == digest(data)
    shutil.copyfile(output, windows)
    sha256 = digest(output.read_bytes())
    assert digest(windows.read_bytes()) == sha256
    winpath = subprocess.check_output(['wslpath', '-w', str(windows.resolve())],
                                      text=True).strip()
    ps = ("$p='" + winpath.replace("'", "''") + "'; "
          "if (!(Test-Path -LiteralPath $p -PathType Leaf)) {exit 2}; "
          "[pscustomobject]@{bytes=(Get-Item -LiteralPath $p).Length; "
          "sha256=(Get-FileHash -Algorithm SHA256 -LiteralPath $p).Hash.ToLower()} "
          "| ConvertTo-Json -Compress")
    observed = json.loads(subprocess.check_output([
        '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe',
        '-NoProfile', '-NonInteractive', '-Command', ps,
    ], text=True).strip())
    assert observed == {'bytes': output.stat().st_size, 'sha256': sha256}
    receipt = {
        'status': 'PASS', 'DELIVERY_COMMIT': commit,
        'wsl_path': str(output.resolve()), 'windows_path': winpath,
        'windows_mount': str(windows.resolve()), 'bytes': output.stat().st_size,
        'sha256': sha256, 'members': len(payload),
        'uncompressed_payload_bytes': sum(map(len, payload.values())),
        'manifest_sha256': digest(payload[manifest]),
        'checks': ['CRC', 'full_extract', 'all_member_sizes_SHA256',
                   'unique_safe_paths', 'credential_patterns',
                   'Windows_Test-Path_size_Get-FileHash', 'retained_Windows_copy'],
        'limitations': 'Scoped membership and pattern scan do not prove absence of every secret.',
    }
    output.with_suffix(output.suffix + '.sha256').write_text(sha256 + '  ' + output.name + '\n')
    output.with_suffix(output.suffix + '.verification.json').write_text(
        json.dumps(receipt, indent=2) + '\n')
    shutil.copyfile(output.with_suffix(output.suffix + '.verification.json'),
                    windows.with_suffix(windows.suffix + '.verification.json'))
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
