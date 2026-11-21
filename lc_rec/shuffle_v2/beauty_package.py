"""Stream and verify the single-file Beauty audit delivery."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

SECRET = re.compile(rb'(?i)(-----BEGIN [A-Z ]*PRIVATE KEY-----|gh[pousr]_[0-9a-z]{30,}|github_pat_[0-9a-z_]{60,}|'
                    rb'hf_[A-Za-z0-9]{30,}|Bearer [A-Za-z0-9._-]{25,}|AKIA[A-Z0-9]{16})')
BLOCK = 2**20


def safe_name(name):
    path = PurePosixPath(name)
    return (path.as_posix() == name and not path.is_absolute() and '.' not in path.parts
            and '..' not in path.parts and not any(c in name for c in ('\\', ':', '\x00')))


def stream_identity(source, destination=None, scan=False):
    digest = hashlib.sha256()
    size = 0
    overlap = b''
    while block := source.read(BLOCK):
        digest.update(block)
        size += len(block)
        if scan:
            assert not SECRET.search(overlap + block), 'Credential-shaped content encountered'
            overlap = block[-256:]
        if destination is not None:
            destination.write(block)
    return {'bytes': size, 'sha256': digest.hexdigest()}


def verify(path, extract=None):
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        assert len({member.filename.casefold() for member in members}) == len(members)
        assert all(safe_name(member.filename) and not member.is_dir()
                   and not stat.S_ISLNK(member.external_attr >> 16) for member in members)
        manifest = json.loads(archive.read('MANIFEST.json'))
        assert set(manifest['members']) == set(archive.namelist()) - {'MANIFEST.json'}
        for member, expected in manifest['members'].items():
            with archive.open(member) as source:
                if extract is None:
                    actual = stream_identity(source)
                else:
                    destination = extract / member
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with destination.open('xb') as target:
                        actual = stream_identity(source, target)
                assert actual == expected, member
        if extract:
            (extract / 'MANIFEST.json').write_bytes(archive.read('MANIFEST.json'))
    with path.open('rb') as source:
        package = stream_identity(source)
    return {**package, 'status': manifest['status'], 'members': len(members),
            'path': str(path.resolve()), 'crc_ok': True, 'sha256_verified': True}


def create(spec, output):
    assert output.suffix == '.zip' and not output.exists()
    files = spec['files']
    assert 'README.md' in files and 'MANIFEST.json' not in files
    assert len({name.casefold() for name in files}) == len(files)
    assert all(safe_name(name) for name in files)
    manifest = {'status': spec['status'], 'members': {}}
    if spec['status'] == 'SECOND_GROUP_COMPLETE_PENDING_WEB_REVIEW':
        required = {'README.md', 'offline_review.sh', 'paired/summary.json', 'paired/arrays.npz',
                    'reviews/prelaunch.json', 'reviews/final.json', 'publication/readback.json',
                    'replay/inputs.jsonl', 'data/Beauty.inter.json', 'data/Beauty.index.json', 'data/Beauty.item.json'}
        for condition in ('normal', 'shuffle2', 'shuffle3a', 'shuffle3b', 'shuffle4'):
            required.update({f'conditions/{condition}/test/predictions.rank0.jsonl',
                             f'conditions/{condition}/primary/per_user_metrics.npz',
                             f'conditions/{condition}/primary/per_user.csv'})
        required.add('conditions/normal/test/predictions.rank1.jsonl')
        assert required <= files.keys(), sorted(required - files.keys())
    staged = output.with_suffix('.zip.partial')
    with zipfile.ZipFile(staged, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, original in sorted(files.items()):
            source_path = Path(original)
            assert source_path.is_file() and not source_path.is_symlink(), source_path
            before = source_path.stat()
            with source_path.open('rb') as source, archive.open(name, 'w', force_zip64=True) as destination:
                try:
                    identity = stream_identity(source, destination, scan=name.endswith(
                        ('.json', '.jsonl', '.py', '.sh', '.log', '.md', '.txt', '.diff', '.csv')))
                except AssertionError as error:
                    raise AssertionError(name) from error
            after = source_path.stat()
            assert (before.st_size, before.st_mtime_ns, before.st_ino) == (after.st_size, after.st_mtime_ns, after.st_ino)
            assert identity['bytes'] == before.st_size
            manifest['members'][name] = identity
        archive.writestr('MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
    receipt = verify(staged)
    staged.rename(output)
    receipt['path'] = str(output.resolve())
    return receipt


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--zip', type=Path, required=True)
    p.add_argument('--spec', type=Path)
    p.add_argument('--extract', type=Path)
    args = p.parse_args()
    if args.spec:
        assert args.extract is None
        receipt = create(json.loads(args.spec.read_text()), args.zip)
    else:
        if args.extract:
            args.extract.mkdir(parents=True, exist_ok=False)
        receipt = verify(args.zip, args.extract)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
