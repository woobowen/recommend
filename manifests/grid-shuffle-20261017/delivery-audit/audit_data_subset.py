"""Independent inherited data SHA subset and current stat identity audit."""
import ast
from collections import Counter
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.parse import urlparse

REPO = Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
SUPPORT = Path('/home/addaswsw/recommend-support/grid-shuffle-20261017')


def identity(path):
    data = path.read_bytes()
    return {'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def verify(row):
    assert identity(Path(row['path'])) == row


def main():
    subset_path = SUPPORT / 'asset-audit/inherited-data-sha-subset.json'
    subset = json.loads(subset_path.read_bytes())
    assert subset['status'] == 'INHERITED_DATA_SHA_SUBSET_AND_CURRENT_READONLY_STAT_BOUND'
    origin = subset['source_manifest']
    assert origin['commit'] == 'd0bcacfee4bab4276861573f7746405fb63c54f2'
    original = subprocess.check_output(['git', 'show', origin['commit'] + ':' + origin['path']], cwd=REPO)
    assert len(original) == origin['bytes'] and hashlib.sha256(original).hexdigest() == origin['sha256']
    expected = []
    for number, line in enumerate(original.splitlines(), 1):
        row = json.loads(line)
        if (row.get('dataset') in ('toys', 'sports') and row.get('split') in ('training', 'evaluation', 'testing')
                and row.get('purpose') == 'fixed_baseline_data_and_embedded_item_user_indexes'
                and row.get('actual_server_path', '').endswith('.tfrecord.gz')):
            expected.append({'source_physical_line_1based': number, 'original_record': row})
    assert subset['records'] == expected and len(expected) == subset['readonly_regular_entities_checked'] == 1293
    by_path = {entry['original_record']['actual_server_path']: entry['original_record'] for entry in expected}
    assert len(by_path) == 1293
    for row in by_path.values():
        assert row['integrity'] == 'PAYLOAD_PRESENT_HASH_MATCHED' and row['provenance'] == 'PROVENANCE_BOUND'
        assert row['lfs_oid'] == 'sha256:' + row['sha256'] and len(row['sha256']) == 64
    assert dict(Counter(row['dataset'] + '/' + row['split'] for row in by_path.values())) == subset['dataset_split_counts']
    verify(subset['current_stat_observation'])
    current = json.loads(Path(subset['current_stat_observation']['path']).read_bytes())
    assert current['payload_content_read'] is False and subset['payload_hashes_freshly_recomputed'] is False
    observed = {row['path']: row for row in current['files']}
    assert len(observed) == 1293 and observed.keys() == by_path.keys()
    loader_files = {}
    assert len(subset['current_loader_inputs']) == 4
    for row in subset['current_loader_inputs']:
        verify(row)
        document = json.loads(Path(row['path']).read_bytes())
        for entry in document['loader']['files']:
            path = urlparse(entry['path']).path
            if path in loader_files:
                assert loader_files[path] == entry
            loader_files[path] = entry
    assert len(loader_files) == subset['test_and_fixed32_loader_file_stat_matches'] == 862
    for path, stat in observed.items():
        assert stat['regular'] and not stat['symlink'] and stat['mode'] == '0o444'
        assert stat['bytes'] == by_path[path]['bytes']
        if path in loader_files:
            assert all(stat[key] == loader_files[path][key] for key in ('bytes', 'mtime_ns', 'device', 'inode'))
    verify(subset['current_B_runtime_semantic_admission'])
    correction = SUPPORT / 'correction-round-3.json'
    round3 = json.loads(correction.read_bytes())
    assert round3['maximum_task_correction_rounds'] == 3 and not round3['further_correction_attempts_authorized']
    package = REPO / 'manifests/grid-shuffle-20261017/package_review.py'
    collector = REPO / 'manifests/grid-shuffle-20261017/collect_publication.py'
    binder = REPO / 'manifests/grid-shuffle-20261017/bind_data_identities.py'
    for path in (package, collector, binder):
        ast.parse(path.read_bytes())
    assert 'correction-round-3.json' in package.read_text() and 'correction-round-3.json' in collector.read_text()
    report = {'status': 'INHERITED_DATA_SHA_SUBSET_AND_PACKAGE_ROUND3_PASS', 'reviewer_task': '/root/delivery_audit',
              'reviewed_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'subset': identity(subset_path),
              'source_manifest': origin, 'original_records_exact_with_line_numbers': 1293,
              'actual_current_readonly_stat_receipt': subset['current_stat_observation'],
              'all_inherited_record_byte_sizes_and_0444_regular_stat_matches': 1293,
              'formal_test_and_fixed32_loader_stat_matches': 862,
              'dataset_split_counts': subset['dataset_split_counts'], 'new_payload_rehash_claimed': False,
              'old_preparation_stage_flags_preserved_and_current_runtime_admission_separate': True,
              'correction_round3': identity(correction),
              'reviewed_programs': [identity(path) for path in (package, collector, binder)],
              'new_model_GPU_calls_or_original_payload_reads': 0,
              'scope_limit': 'Original full SHA values inherited from fixed earlier manifest, explicitly not fresh payload content verification. Current byte size/readonly/stat identity and actual loader sources independently compared to recorded real query.'}
    destination = SUPPORT / 'delivery-audit/data-subset-and-round3-package-review.json'
    report['program'] = identity(Path(__file__).resolve())
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'status': report['status'], 'records': 1293, 'loader_matches': 862}))


if __name__ == '__main__':
    main()
