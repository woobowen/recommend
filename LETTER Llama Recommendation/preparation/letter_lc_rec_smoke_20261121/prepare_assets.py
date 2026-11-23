"""Stage pinned author code/data and LLaMA-1 7B metadata; weights are optional."""
from pathlib import Path
import argparse
import hashlib
import json
import urllib.request
import zipfile
import time
import concurrent.futures

HERE = Path(__file__).resolve().parent
COMMIT = '8d0154e28de37dbb6e24871c508ad8ddb1921cda'
MODEL_REVISION = '4782ad278652c7c71b72204d462d6d01eaaf7549'
MODEL_ID = 'huggyllama/llama-7b'
SOURCE_URL = f'https://codeload.github.com/HonghuiBao2000/LETTER/zip/{COMMIT}'
ARCHIVE_SHA = '7f865db94f21fa358c5e0c5f79bc0df88561327f246f445e187c9e31bbfa93c6'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def get(url, path, size=None, sha=None):
    if path.exists() and (size is None or path.stat().st_size == size):
        if sha and digest(path) != sha:
            raise RuntimeError(f'Existing file hash mismatch: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + '.partial')
    for attempt in range(12):
        offset = part.stat().st_size if part.exists() else 0
        if size is not None and offset == size:
            break
        headers = {'User-Agent': 'LETTER-reproduction'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=90) as r:
                append = offset > 0 and r.status == 206
                with part.open('ab' if append else 'wb') as f:
                    while chunk := r.read(8 * 1024 * 1024):
                        f.write(chunk)
            if size is None or part.stat().st_size == size:
                break
        except Exception as exc:
            print('download_retry', path.name, attempt, str(exc), flush=True)
            if attempt == 11:
                raise
            time.sleep(min(attempt + 1, 5))
    if size is not None and part.stat().st_size != size:
        raise RuntimeError(f'Incomplete file: {part}')
    if sha and digest(part) != sha:
        raise RuntimeError(f'Hash mismatch: {part}')
    part.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', action='store_true')
    p.add_argument('--model-dir', type=Path, default=HERE / 'model_metadata')
    p.add_argument('--relay-base')
    args = p.parse_args()
    archive = HERE / 'LETTER-official.zip'
    get(SOURCE_URL, archive, sha=ARCHIVE_SHA)
    vendor = HERE / 'vendor' / 'LETTER'
    source_paths = []
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            relative = '/'.join(name.split('/')[1:])
            if not relative or name.endswith('/'):
                continue
            target = vendor / relative
            if not target.resolve().is_relative_to(vendor.resolve()):
                raise RuntimeError('Unexpected archive path')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(name))
            source_paths.append(relative)
    api = HERE / 'model_api.json'
    get(f'{args.relay_base}/api' if args.relay_base else
        f'https://huggingface.co/api/models/{MODEL_ID}/revision/{MODEL_REVISION}?blobs=true', api)
    metadata = json.loads(api.read_text(encoding='utf-8'))
    assert metadata['sha'] == MODEL_REVISION
    allowed = {'config.json', 'generation_config.json', 'special_tokens_map.json',
               'tokenizer_config.json', 'tokenizer.json', 'tokenizer.model',
               'model.safetensors.index.json', 'model-00001-of-00002.safetensors',
               'model-00002-of-00002.safetensors', 'README.md'}
    model_manifest = []
    for f in metadata['siblings']:
        name = f['rfilename']
        if name not in allowed:
            continue
        expected = f.get('lfs', {}).get('sha256')
        entry = {'name': name, 'bytes': f['size'], 'sha256': expected}
        if args.weights or not name.endswith('.safetensors'):
            url = f'{args.relay_base}/files/{name}' if args.relay_base else f'https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{name}'
            get(url,
                args.model_dir / name, f['size'], expected)
            entry['downloaded'] = True
            entry['sha256'] = digest(args.model_dir / name)
        else:
            entry['downloaded'] = False
        model_manifest.append(entry)
    audit = {}
    for ds in ('Beauty', 'Instruments'):
        folder = vendor / 'data' / ds
        files = {key: folder / f'{ds}.{key}.json' for key in ('index', 'inter', 'item')}
        source = {key: json.loads(file.read_text(encoding='utf-8')) for key, file in files.items()}
        sid, seq, items = source['index'], source['inter'], source['item']
        assert set(sid) == set(items)
        assert all(len(code) == 4 for code in sid.values())
        assert all(len(history) >= 3 and all(str(i) in sid for i in history) for history in seq.values())
        audit[ds] = {'users': len(seq), 'items': len(sid),
                     'unique_sids': len({tuple(x) for x in sid.values()}),
                     'sid_tokens': len({t for x in sid.values() for t in x}),
                     'sha256': {key: digest(file) for key, file in files.items()}}
    report = {'source_commit': COMMIT, 'source_archive_sha256': digest(archive),
              'model_id': MODEL_ID, 'model_revision': MODEL_REVISION,
              'model_files': model_manifest, 'datasets': audit,
              'official_files': {relative: digest(vendor / relative) for relative in sorted(source_paths)}}
    (HERE / 'assets_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'datasets': audit, 'weights_downloaded': args.weights,
                      'vendor': str(vendor)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
