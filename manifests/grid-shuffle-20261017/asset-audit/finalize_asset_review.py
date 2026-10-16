"""Freeze scoped asset admission and preserve the optional callback failure."""
import datetime as dt
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def receipt(path):
    p = Path(path)
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def main():
    source = json.loads((HERE / 'asset-source-review.json').read_text())
    reviews = {}
    for domain in ['toys', 'sports']:
        p = HERE / (domain + '-runtime-asset-admission.json')
        review = json.loads(p.read_text())
        assert review['status'] == 'PASS' and review['domain'] == domain
        assert review['assets'] == source['B'][domain]['assets']
        assert review['original_source_review']['sha256'] == receipt(HERE / 'asset-source-review.json')['sha256']
        reviews[domain] = {'status': 'PASS', 'review': receipt(p), 'test_targets': review['full_index_unique_targets'],
                           'fixed32_visible_item_groups_independently_rechecked': review['fixed32_visible_item_groups_rechecked_against_original_sid'],
                           'strict_model_values_and_embedded_sid': True,
                           'selected_checkpoint_source': source['B'][domain]['checkpoint_selection'],
                           'assets': review['assets'], 'job_id': review['job_id']}
    independent_dictionary = HERE.parent / 'results-audit/sid-dictionary-independent-review.json'
    dictionary = json.loads(independent_dictionary.read_text())
    assert dictionary['status'] == 'INDEPENDENT_SID_DICTIONARY_PROJECTIONS_PASS'
    callbacks = json.loads((HERE / 'callback-v1-collection.json').read_text())
    assert callbacks['expected_terminal_status'] == 'FAILED'
    for row in callbacks['receipts']:
        assert row['status'] == 'FAILED' and row['cleanup_verified']
    result = {
        'status': 'PASS_B_CORE_ASSET_ADMISSION_WITH_A_UNRESOLVED',
        'reviewer': '/root/asset_audit', 'review_time_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'program': receipt(__file__),
        'scope': 'Non-author B original selection/source/payload identity and runtime strict/semantic asset admission; A gaps retained. Not full formal result acceptance.',
        'source_review': receipt(HERE / 'asset-source-review.json'),
        'original_B_item_axis_source_review': receipt(HERE / 'b-item-axis-source-review.json'),
        'runtime_review_program': receipt(HERE / 'review_runtime_assets.py'),
        'raw_sid_reader_program': receipt(HERE.parent / 'results-audit/verify_sid_dictionary_projection.py'),
        'B': reviews, 'A': source['A'],
        'SID_dictionary_projection': {
            'producer': '/root/asset_audit', 'self_check': receipt(HERE / 'sid-dictionary-collection-review.json'),
            'nonauthor': '/root/results_audit', 'independent_review': receipt(independent_dictionary),
            'all_original_raw_SID_tuple_values_independently_reconstructed': True,
            'no_new_SID_generation': True, 'model_or_GPU_calls': 0},
        'optional_checkpoint_callback_supplement': {
            'status': 'FAILED_UNVERIFIED_SUPPLEMENT_NOT_A_REQUIRED_CURRENT_B_GATE',
            'failed_original_jobs': [row['job_id'] for row in callbacks['receipts']],
            'failed_original_receipt': receipt(HERE / 'callback-v1-collection.json'),
            'failure': 'Standalone safe-global import order hit TokenizerConfig circular import before checkpoint load/model/GPU; original full entrypoint initializes src.utils first',
            'main_selection_evidence_already_passed': ['original config/CLI overrides', 'original CSV validation maximum', 'original stage_result selected checkpoint', 'fresh SHA/LFS identity', 'r2 actual strict weights and embedded SID exact'],
            'no_fabricated_callback_verification': True,
            'repair_policy': 'Preserve v1, do not submit again now; only main controller may include one minimal import-order repair in a remaining task-wide concentrated correction round',
            'no_new_GPU_execution': True},
        'failures_preserved': [
            {'kind': 'source CSV export', 'reason': 'Text read normalized CRLF; failed SHA guard', 'resolution': 'Original bytes/base64 export reverified against LFS OID; no research source change'},
            {'kind': 'B preflight v1', 'reason': 'src namespace import order', 'receipt': receipt(HERE / 'preflight-original-error-worklog-1.json')},
            {'kind': 'B preflight r1', 'reason': 'Unused Hydra metadata now resolver outside Hydra', 'receipt': receipt(HERE / 'preflight-r1-original-query.json')},
            {'kind': 'optional callback v1', 'reason': 'Standalone original import circular dependency', 'receipt': receipt(HERE / 'callback-v1-collection.json')},
        ],
        'new_packages_or_toolchains': [], 'environment_configuration_changes': [],
        'Git_operations': 0, 'original_asset_writes': 0,
        'main_task_state_limit': 'B formal evaluation has separate ongoing main-controlled jobs; this scoped audit does not claim all task jobs are complete',
    }
    output = HERE / 'runtime-asset-admission.json'
    assert not output.exists()
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'report': receipt(output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
