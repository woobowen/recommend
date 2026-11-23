"""Extract and verify a pinned official LETTER snapshot without changing its files."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    here = Path(__file__).resolve().parent
    manifest = json.loads((here / 'source_manifest.json').read_text(encoding='utf-8'))
    archive = here / 'LETTER-official-8d0154e.zip'
    assert sha(archive) == manifest['archive_sha256'], 'Archive checksum mismatch'
    dest = Path(manifest['server_source_root'])
    assert not dest.exists(), f'Refusing to overwrite existing source: {dest}'
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for info in z.infolist():
            path = PurePosixPath(info.filename)
            assert path.parts[0] == 'LETTER-' + manifest['commit']
            assert not path.is_absolute() and '..' not in path.parts
            assert ((info.external_attr >> 16) & 0o170000) != 0o120000, 'Unexpected symlink'
            if info.is_dir():
                continue
            output = dest.joinpath(*path.parts[1:])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(z.read(info))
    for entry in manifest['file_manifest']:
        path = dest / entry['path']
        assert path.stat().st_size == entry['bytes']
        assert sha(path) == entry['sha256'], entry['path']
    required = ['README.md', 'RQ-VAE/main.py', 'RQ-VAE/train_tokenizer.sh',
                'RQ-VAE/generate_indices.py', 'data_process/amazon_text_emb.py',
                'LETTER-LC-Rec/run_train.sh', 'LETTER-TIGER/run_train.sh']
    assert all((dest / path).is_file() for path in required)
    data = {p.name: sorted(f.name for f in p.iterdir() if f.is_file())
            for p in (dest / 'data').iterdir() if p.is_dir()}
    result = dict(complete=True, source_root=str(dest), commit=manifest['commit'],
                  archive_sha256=manifest['archive_sha256'],
                  verified_files=len(manifest['file_manifest']),
                  all_public_source_files_present=True, datasets_in_author_repository=data,
                  weights_downloaded=False, dependencies_installed=False,
                  training_started=False, gpus_used=False)
    (here / 'download_verified.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
