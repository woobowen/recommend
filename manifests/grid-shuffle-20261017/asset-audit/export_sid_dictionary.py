"""CPU read-only projection of existing SID tensors for independent auditing."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES', '') == ''
    spec = json.loads(args.spec.read_text())
    source = spec['sid']
    sid_path = Path(source['path'])
    assert sid_path.is_file() and not sid_path.is_symlink()
    assert sid_path.stat().st_size == source['bytes']
    assert sha(sid_path) == source['sha256']
    assert sha(__file__) == spec['script']['sha256']
    output = Path(os.environ['GRID_JOB_DIR']) / 'output'
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'invocation.json', {
        'argv': sys.argv, 'interpreter': sys.executable, 'cwd': os.getcwd(),
        'started_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'job_id': os.environ['GRID_JOB_ID'], 'source_commit': spec['source_commit'],
        'script': spec['script'], 'spec': {'path': str(args.spec), 'sha256': sha(args.spec)},
        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'purpose': 'Read-only projection; no new SID generation and no model or GPU execution',
    })
    import torch
    sid = torch.load(sid_path, map_location='cpu', weights_only=True)
    assert sid.ndim == 2 and sid.shape[0] == 4 and sid.dtype in (torch.int32, torch.int64)
    rows = sid.t().tolist()
    legal = sorted({tuple(row) for row in rows})
    assert all(len(row) == 4 and all(type(code) is int and code >= 0 for code in row) for row in rows)
    assert len(rows) == spec['expected_item_count']
    assert not torch.cuda.is_initialized()
    result = {
        'kind': 'existing SID raw four-code dictionary projection; no upstream regeneration',
        'source': source, 'shape': list(sid.shape),
        'row_count': len(rows), 'unique_four_code_count': len(legal),
        'duplicate_four_code_rows': len(rows) - len(legal),
        'legal_raw_sids': legal, 'model_constructions': 0, 'GPU_model_calls': 0,
        'cuda_initialized': torch.cuda.is_initialized(),
        'calculation_source': spec['calculation_source'],
        'identity': {'script': spec['script'], 'source_commit': spec['source_commit'],
                     'spec_sha256': sha(args.spec), 'job_id': os.environ['GRID_JOB_ID']},
    }
    write_json(output / 'sid_dictionary.json', result)
    write_json(output / 'RESULT.json', {
        'status': 'PASS', 'domain': spec['domain'], 'source': source,
        'dictionary': {'path': str(output / 'sid_dictionary.json'),
                       'sha256': sha(output / 'sid_dictionary.json'),
                       'bytes': (output / 'sid_dictionary.json').stat().st_size},
        'model_constructions': 0, 'GPU_model_calls': 0,
        'row_count': len(rows), 'unique_four_code_count': len(legal),
        'cuda_initialized': torch.cuda.is_initialized(),
    })
    print(json.dumps({'status': 'PASS', 'domain': spec['domain'], 'items': len(rows),
                      'unique_raw_sids': len(legal), 'model_calls': 0, 'GPU_calls': 0}))


if __name__ == '__main__':
    main()
