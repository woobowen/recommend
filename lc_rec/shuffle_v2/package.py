"""Build and verify a light audit ZIP from a fixed Git tree and vetted receipts."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tempfile
import zipfile

BASE = '7368d301d0d928f77d50551c3b53a5b30c23d63f'
BRANCH = 'wbw-lc-rec-shuffle-v2'
REPO = Path(__file__).resolve().parents[2]
ZIP_FILES = ('lc_rec/shuffle_v2', 'lc_rec/shuffle/README.md', 'lc_rec/shuffle/core.py',
             'lc_rec/shuffle/entry.py', 'lc_rec/letter/scripts/formal_evaluate.py',
             'lc_rec/a1_qwen3_06b/scripts/evaluate.py', 'lc_rec/qwen_preflight_20261110/scripts/common.py',
             'lc_rec/letter/EVALUATION_PROTOCOL_V1.md', 'lc_rec/letter/config/a0_evaluation_v1.json',
             'lc_rec/letter/review/formal_det_20261112/independent_metrics.py',
             'lc_rec/a1_qwen3_06b/scripts/offline_metrics.py', 'lc_rec/a1_qwen3_06b/review/independent_extended.py',
             'lc_rec/letter/review/final_audit_20261113/external/official_cce.py',
             'lc_rec/letter/review/final_audit_20261113/statistics_verify.py')
SECRET = re.compile(rb'(?i)(-----BEGIN [A-Z ]*PRIVATE KEY-----|gh[pousr]_[0-9a-z]{30,}|github_pat_[0-9a-z_]{60,}|'
                    rb'hf_[A-Za-z0-9]{30,}|Bearer [A-Za-z0-9._-]{25,}|AKIA[A-Z0-9]{16})')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=REPO, timeout=45)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    return name == path.as_posix() and not path.is_absolute() and all(p not in ('', '.', '..') for p in path.parts) and not any(c in name for c in ('\\', ':', '\x00'))


def check_required(files, status):
    assert 'source/lc_rec/shuffle/README.md' in files
    if status == 'FIRST_GROUP_COMPLETE_PENDING_WEB_REVIEW':
        required = {'evidence/paired/summary.json', 'evidence/paired/metrics.csv', 'evidence/paired/paired.csv',
                    'evidence/paired/arrays.npz', 'evidence/nonauthor-final.md', 'evidence/remote-run-index.json',
                    'evidence/github-readback.json', 'evidence/preflight/summary.json'}
        missing = sorted(required - files.keys())
        if missing:
            raise FileNotFoundError(f'Final audit package is missing {missing}')


def create(commit, evidence, output, status):
    assert len(commit) == 40 and output.suffix == '.zip' and not output.exists()
    files = {}
    for entry in git('ls-tree', '-rz', commit, '--', *ZIP_FILES).split(b'\x00'):
        if not entry:
            continue
        mode_identity, name = entry.split(b'\t')
        mode, kind, object_id = mode_identity.decode().split()
        assert mode in ('100644', '100755') and kind == 'blob'
        files['source/' + name.decode()] = git('cat-file', 'blob', object_id)
    files['source/changes.patch'] = git('diff', BASE, commit, '--', 'lc_rec/shuffle_v2', 'lc_rec/shuffle/README.md')
    for path in sorted(evidence.rglob('*')):
        assert not path.is_symlink(), path
        if path.is_file():
            assert path.stat().st_size < 20 * 1024**2, path
            files['evidence/' + path.relative_to(evidence).as_posix()] = path.read_bytes()
    check_required(files, status)
    note = {'status': status, 'branch': BRANCH, 'source_commit': commit, 'base_commit': BASE,
            'entry': 'source/lc_rec/shuffle/README.md',
            'github_entry': f'https://github.com/woobowen/recommend/blob/{commit}/lc_rec/shuffle/README.md'}
    files['README.json'] = (json.dumps(note, indent=2) + '\n').encode()
    assert len({name.casefold() for name in files}) == len(files)
    for name, data in files.items():
        assert safe_name(name), name
        assert not SECRET.search(data), f'Secret scan flagged {name}'
    manifest = {name: {'bytes': len(data), 'sha256': sha(data)} for name, data in sorted(files.items())}
    files['MANIFEST.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    with tempfile.NamedTemporaryFile(prefix=output.stem, suffix='.partial', dir=output.parent) as temp:
        staged = Path(temp.name)
        with zipfile.ZipFile(staged, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name, data in sorted(files.items()):
                archive.writestr(name, data)
        with zipfile.ZipFile(staged) as archive:
            assert archive.testzip() is None
            names = archive.namelist()
            assert len(names) == len(set(names)) and set(names) == set(files)
            assert all(safe_name(n) for n in names)
            assert all(not stat.S_ISLNK(i.external_attr >> 16) for i in archive.infolist())
            assert json.loads(archive.read('MANIFEST.json')) == manifest
            for name, identity in manifest.items():
                data = archive.read(name)
                assert identity == {'bytes': len(data), 'sha256': sha(data)}
        contents = staged.read_bytes()
        with output.open('xb') as file:
            file.write(contents)
    assert output.read_bytes() == contents
    total = sha(contents)
    note.update(zip=str(output.resolve()), sha256=total, bytes=len(contents), members=len(files),
                crc_verified=True, hash_verified=True, paths_checked=True, secret_scan=True)
    output.with_suffix('.zip.sha256').write_text(f'{total}  {output.name}\n')
    output.with_suffix('.zip.verification.json').write_text(json.dumps(note, indent=2) + '\n')
    return note


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--status', required=True, choices=('PARTIAL_BLOCKED','FIRST_GROUP_COMPLETE_PENDING_WEB_REVIEW'))
    a = parser.parse_args()
    assert a.evidence.is_dir()
    print(json.dumps(create(a.commit, a.evidence, a.output, a.status), indent=2))


if __name__ == '__main__':
    main()
