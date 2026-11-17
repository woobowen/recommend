"""Small file helpers; no SID or model data are loaded at import."""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path('/home/jcy/GRID_wmh_Qwen_TIGER_20261009')
HERE = ROOT / 'codex/llm2vec_sid_hierarchy_20261113'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n'
                            for row in rows), encoding='utf-8')


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line]


def status(path, stage, **extra):
    record = {'stage': stage, 'updated_unix': time.time(), **extra}
    write_json(path, record)
    print(json.dumps(record, ensure_ascii=False), flush=True)
