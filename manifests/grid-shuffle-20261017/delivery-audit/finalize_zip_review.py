"""Aggregate completed real independent ZIP/origin/content audit receipts."""
from collections import Counter
import datetime as dt
import hashlib
import json
from pathlib import Path
import zipfile

SUPPORT = Path('/home/addaswsw/recommend-support/grid-shuffle-20261017')
HERE = SUPPORT / 'delivery-audit'


def identity(path):
    value = path.read_bytes()
    return {'path': str(path), 'bytes': len(value), 'sha256': hashlib.sha256(value).hexdigest()}


def main():
    read = lambda name: json.loads((HERE / name).read_bytes())
    container = read('zip-container-structure-review.json')
    origin = read('zip-origin-bytes-review.json')
    content = read('content-independent-review.json')
    fresh = read('content-fresh-independent-readback-review.json')
    data = read('data-subset-and-round3-package-review.json')
    raw = read('raw-attention-statistics-crosscheck.json')
    jobs = read('final-terminal-job-review.json')
    assert container['status'] == 'PASS' and origin['status'] == 'PACKAGED_ORIGIN_BYTES_PASS'
    assert content['status'] == fresh['status'] == 'CONTENT_ACCEPTANCE_PASS'
    assert data['status'] == 'INHERITED_DATA_SHA_SUBSET_AND_PACKAGE_ROUND3_PASS'
    assert raw['status'] == 'FINAL_RAW_ATTENTION_STATISTICS_CROSSCHECK_PASS'
    assert jobs['status'] == 'PASS' and jobs['job_count'] == 22
    assert origin['content_commit'] == content['content_commit'] == fresh['content_commit'] == container['inner_manifest_content_commit']
    archive = Path(container['zip']['path'])
    assert identity(archive) == container['zip'] == origin['zip']
    package = json.loads((SUPPORT / 'package-receipt.json').read_bytes())
    for field in ('path', 'bytes', 'sha256'):
        assert package[field] == container['zip'][field]
    assert package['member_count'] == container['files'] == origin['files_checked'] == 1509
    assert package['NPZ_members'] == container['numeric_NPZ_count'] == origin['original_complete_raw_NPZ_count'] == 24
    assert container['verified_record_count'] == 24
    assert container['all_outer_and_nested_NPZ_CRC_PASS'] and container['safe_unique_unencrypted_regular_members']
    assert container['obvious_secret_signatures'] == []
    inventory = Path(package['intended_members'])
    assert identity(inventory)['sha256'] == package['intended_members_sha256']
    assert container['intended_inventory_match']['sha256'] == package['intended_members_sha256']
    assert archive.with_suffix('.zip.sha256').read_text().strip() == package['sha256'] + '  ' + archive.name
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read('MANIFEST.json'))
        attention_receipts = [row for row in manifest['raw_evidence'] if row['all_original_PNG']]
        assert len(attention_receipts) == 6 and sum(row['all_original_PNG'] for row in attention_receipts) == 192
        raw_selected_png = sum(row['PNG_in_review_ZIP'] for row in attention_receipts)
        png_namespaces = Counter(row['path'].split('/')[0] for row in container['members'] if row['path'].endswith('.png'))
        assert png_namespaces['evidence'] == raw_selected_png
        fresh_inside = 'external-receipts/delivery-audit/content-fresh-independent-readback-review.json' in z.namelist()
    proof_names = ['zip-container-structure-review.json', 'zip-origin-bytes-review.json',
                   'content-independent-review.json', 'content-fresh-independent-readback-review.json',
                   'data-subset-and-round3-package-review.json', 'raw-attention-statistics-crosscheck.json',
                   'final-terminal-job-review.json', 'final-visual-spotcheck.json']
    report = {'status': 'PASS', 'acceptance_kind': 'INDEPENDENT_REVIEW_ZIP_PASS',
              'reviewer_task': '/root/delivery_audit', 'reviewed_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'task_overall_status': 'PARTIAL', 'content_commit': content['content_commit'],
              'zip': container['zip'], 'windows_path': package['windows_path'],
              'member_count': container['files'], 'uncompressed_bytes': container['uncompressed_bytes'],
              'intended_member_inventory': identity(inventory),
              'outer_and_all_nested_NPZ_CRC': 'PASS', 'all_1509_member_sizes_SHA_and_origins': 'PASS',
              'inner_manifest_and_external_intended_inventory': 'PASS',
              'safe_unique_regular_unencrypted_members': 'PASS', 'obvious_secret_signature_scan': 'PASS',
              'complete_raw_NPZ': 24, 'complete_record_JSON': 24,
              'all_numeric_array_record_references': container['verified_array_references'],
              'all_original_PNG_identities': 192, 'PNG_members_by_namespace': dict(png_namespaces),
              'prespecified_selected_raw_evidence_PNG_members': raw_selected_png,
              'actual_fresh_non_author_readback_receipt_inside_ZIP': fresh_inside,
              'original_jobs_terminal_counts': jobs['terminal_counts'], 'original_task_active_job_ids': [],
              'conservative_GPU_requested_worker_wall_hours': jobs['accounting']['conservative_status_interval_seconds_including_cleanup'] / 3600,
              'task_correction_rounds_used': 3, 'new_installations': [],
              'independent_checks': [identity(HERE / name) for name in proof_names],
              'program': identity(Path(__file__).resolve()),
              'scope_limits': ['A routes remain unexecuted with disclosed important asset/selection gaps; whole task stays PARTIAL.',
                               'Data payload SHA values explicitly inherited from fixed original manifest, with current stat/loader identity checks; no fresh payload rehash claimed.',
                               'Credential signature scan excludes obvious secrets, not a mathematical guarantee against every possible encoding.',
                               'Final delivery receipt commit publication/readback remains a separate pending check.',
                               'Internal review acceptance is not GPT webpage secondary review or scientific invariance proof.']}
    output = HERE / 'zip-independent-review.json'
    assert not output.exists()
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({key: report[key] for key in ('status', 'acceptance_kind', 'content_commit', 'zip', 'member_count', 'PNG_members_by_namespace')}))


if __name__ == '__main__':
    main()
