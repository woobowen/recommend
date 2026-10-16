"""Bind inherited full data SHA records to this task's real readonly loader files.

This is metadata-only final evidence collection. No model or data content read.
Full payload hashes remain explicitly inherited from the fixed prior manifest.
"""
import collections
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlparse

REPO = Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
SUPPORT = Path('/home/addaswsw/recommend-support/grid-shuffle-20261017')
JOBS = Path('/home/addaswsw/recommend-support/sid-extensions-p0/jobs/new-server')
ANCHOR = 'd0bcacfee4bab4276861573f7746405fb63c54f2'
SOURCE = 'manifests/sid-extensions/ASSET_MANIFEST.jsonl'


def identity(path):
    value = path.read_bytes()
    return {'path': str(path), 'bytes': len(value), 'sha256': hashlib.sha256(value).hexdigest()}


def main():
    source = subprocess.check_output(['git', 'show', ANCHOR + ':' + SOURCE], cwd=REPO)
    assert source == (REPO / SOURCE).read_bytes()
    records = []
    for number, line in enumerate(source.splitlines(), 1):
        row = json.loads(line)
        if (row.get('dataset') in ('toys', 'sports') and
                row.get('split') in ('training', 'evaluation', 'testing') and
                row.get('purpose') == 'fixed_baseline_data_and_embedded_item_user_indexes' and
                row.get('actual_server_path', '').endswith('.tfrecord.gz')):
            assert row['integrity'] == 'PAYLOAD_PRESENT_HASH_MATCHED'
            assert row['provenance'] == 'PROVENANCE_BOUND' and row['read_only_use']
            records.append({'source_physical_line_1based': number, 'original_record': row})
    assert len(records) == 1293
    by_path = {r['original_record']['actual_server_path']: r['original_record'] for r in records}
    assert len(by_path) == len(records)
    loader_inputs = []
    expected_loader = {}
    for domain in ('toys', 'sports'):
        base = JOBS / f'gs-20261017-{domain}-b-preflight-r2/retrieved/output'
        for name in ('test_coverage.json', 'fixed_attention_samples.json'):
            path = base / name
            document = json.loads(path.read_bytes())
            loader_inputs.append(identity(path))
            for row in document['loader']['files']:
                absolute = urlparse(row['path']).path
                assert absolute in by_path and row['bytes'] == by_path[absolute]['bytes']
                if absolute in expected_loader:
                    assert expected_loader[absolute] == row
                expected_loader[absolute] = row
    assert len(expected_loader) == 862
    sys.path.insert(0, '/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
    from transport import Transport
    transport = Transport('new-server')
    transport.identity()
    code = '''from pathlib import Path
import datetime,json,stat
rows=[]
for name in TASK_DATA_PATHS:
 p=Path(name);s=p.stat()
 rows.append({'path':name,'regular':stat.S_ISREG(s.st_mode),'symlink':p.is_symlink(),
              'mode':oct(stat.S_IMODE(s.st_mode)),'bytes':s.st_size,
              'mtime_ns':s.st_mtime_ns,'device':s.st_dev,'inode':s.st_ino})
print(json.dumps({'observed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'payload_content_read':False,'files':rows}))
'''.replace('TASK_DATA_PATHS', repr(sorted(by_path)))
    current = json.loads(transport.python(code, timeout=60))
    current_path = SUPPORT / 'asset-audit/data-current-stat.json'
    with current_path.open('x') as stream:
        json.dump(current, stream, indent=2); stream.write('\n')
    assert len(current['files']) == len(by_path)
    for row in current['files']:
        prior = by_path[row['path']]
        assert row['regular'] and not row['symlink'] and row['mode'] == '0o444'
        assert row['bytes'] == prior['bytes']
        if row['path'] in expected_loader:
            observed = expected_loader[row['path']]
            assert all(row[key] == observed[key] for key in ('bytes', 'mtime_ns', 'device', 'inode'))
    subset = {
        'status': 'INHERITED_DATA_SHA_SUBSET_AND_CURRENT_READONLY_STAT_BOUND',
        'source_manifest': {'commit': ANCHOR, 'path': SOURCE, 'bytes': len(source),
                            'sha256': hashlib.sha256(source).hexdigest()},
        'scope': 'Complete original Toys/Sports training/evaluation/testing records, unchanged with physical source line numbers. Payload SHA values inherited; no payload hashes recomputed in this final metadata query.',
        'historical_stage_labels': 'Original not_run/SEMANTIC_BINDING_UNVERIFIED fields describe the original preparation stage, are preserved verbatim, and do not override the separately accepted current B runtime evidence.',
        'current_B_runtime_semantic_admission': identity(SUPPORT / 'asset-audit/runtime-asset-admission.json'),
        'current_stat_observation': identity(current_path),
        'current_loader_inputs': loader_inputs,
        'readonly_regular_entities_checked': len(records),
        'test_and_fixed32_loader_file_stat_matches': len(expected_loader),
        'payload_hashes_freshly_recomputed': False,
        'dataset_split_counts': dict(collections.Counter(
            r['original_record']['dataset'] + '/' + r['original_record']['split'] for r in records)),
        'records': records
    }
    output = SUPPORT / 'asset-audit/inherited-data-sha-subset.json'
    with output.open('x') as stream:
        json.dump(subset, stream, indent=2, ensure_ascii=False); stream.write('\n')
    print(json.dumps({'status': subset['status'], 'path': str(output),
                      'records': len(records), 'loader_stat_matches': len(expected_loader),
                      'payload_hashes_freshly_recomputed': False, 'sha256': identity(output)['sha256']}))


if __name__ == '__main__':
    main()
