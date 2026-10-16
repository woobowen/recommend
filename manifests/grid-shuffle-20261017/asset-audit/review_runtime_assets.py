"""Non-author review of raw preflight pairing, item-axis and target evidence."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'results-audit'))
from verify_sid_dictionary_projection import read_sid


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain', choices=['toys', 'sports'], required=True)
    parser.add_argument('--job-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    expected = read(HERE / 'candidate_specs.json')[args.domain]
    source_review = read(HERE / 'asset-source-review.json')
    selection = source_review['B'][args.domain]['checkpoint_selection']
    directory = args.job_dir
    manifest = read(directory / 'export-manifest.json')
    entries = {row['path']: row for row in manifest['files']}
    assert len(entries) == len(manifest['files'])
    assert set(str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()) == set(entries) | {'export-manifest.json'}
    for rel, entry in entries.items():
        p = directory / rel
        assert not Path(rel).is_absolute() and '..' not in Path(rel).parts
        assert not p.is_symlink() and p.stat().st_size == entry['bytes'] and sha(p) == entry['sha256']
    request, accepted, status = [read(directory / name) for name in ['request.json', 'accepted.json', 'status.json']]
    fingerprint = hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    assert accepted['request_sha256'] == fingerprint
    assert status['status'] == 'SUCCEEDED' and status['exit_code'] == 0
    assert status['cleanup']['verified'] and not status['worker_may_be_live']
    invocation, loading, checks, coverage, cohort, result, used = [read(directory / 'output' / name) for name in [
        'invocation.json', 'loading.json', 'data_check.json', 'test_coverage.json', 'fixed_attention_samples.json', 'RESULT.json', 'used_config.json']]
    assert result['status'] == 'PASS' and result['engineering_only'] and not result['formal']
    assert result['domain'] == args.domain and result['route'] == 'B'
    spec = invocation['spec']
    for name in ['sid', 'checkpoint', 'config']:
        assert spec[name] == expected[name]
        inputs = {r['path']: r['sha256'] for r in request['inputs']}
        assert inputs[expected[name]['path']] == expected[name]['sha256']
    assert invocation['code_identity']['source_commit'] + '/' + invocation['code_identity']['snapshot_sha256'] == request['capture_version']
    strict = loading['strict_load']
    assert not strict['missing_keys'] and not strict['unexpected_keys']
    assert strict['state_tensor_count'] == 177 and strict['all_parameter_keys_shapes_values_equal'] and strict['embedded_sid_exact']
    assert strict['global_step'] == selection['selected_step']
    assert loading['shared_sid_table'] is True and loading['parameter_count'] == 13128064
    assert loading['model_class'] == 'experiments.beauty_extensions.baseline.BaselineShuffleModel'
    assert result['checkpoint_parameters_unchanged'] and result['input_mask_labels_unchanged'] and result['effective_config_roundtrip_exact']
    count = 11924 if args.domain == 'toys' else 18357
    raw_sid_path = HERE.parent / 'results-audit/raw-sid-for-audit' / f'{args.domain}-{expected["sid"]["sha256"][:16]}.pt'
    assert raw_sid_path.stat().st_size == expected['sid']['bytes'] and sha(raw_sid_path) == expected['sid']['sha256']
    sid, sid_storage = read_sid(raw_sid_path)
    assert list(sid.shape) == checks['sid_shape'] == [4, count]
    assert checks['range'] == [0, 255] and checks['fourth_collision_rule_verified']
    assert checks['source_data_modified'] is False
    legal = {tuple(map(int, row)) for row in sid.T}
    assert len(legal) == count
    # Independently recheck the collision-code convention on all source item IDs.
    groups = {}
    for item_id in range(count):
        groups.setdefault(tuple(map(int, sid[:3, item_id])), []).append(item_id)
    for ids in groups.values():
        fourth = [int(sid[3, index]) for index in ids]
        assert fourth == ([0] if len(ids) == 1 else list(range(1, len(ids) + 1)))
    assert {entry['split'] for entry in checks['split_checks']} == {'train', 'val', 'test'}
    for entry in checks['split_checks']:
        assert entry['num_workers'] == 0 and entry['sequence_length_tokens_including_target'] == 120
        assert entry['visible_history_item_limit'] == 29
        for sample in entry['semantic_examples']:
            assert tuple(sample['target_sid']) in legal
            assert all(tuple(row) in legal for row in sample['first_item_sids'])
    assert coverage['sid'] == cohort['sid'] == expected['sid']
    assert coverage['original_label_targets_verified'] and coverage['complete_target_hidden_groups_verified']
    assert coverage['duplicate_user_rows'] == 0
    assert coverage['rows'] == coverage['targets'] == coverage['unique_users'] == result['test_targets']
    assert coverage['item_range'] == [0, count - 1] and coverage['sid_item_count'] == count
    assert coverage['coverage_is_not_model_evaluation'] and not coverage['model_execution']
    assert coverage['loader']['batch_size_per_device'] == 8 and not coverage['loader']['drop_last']
    assert coverage['loader']['sequence_length'] == 120 and coverage['loader']['num_workers'] == 0
    index_path = directory / 'output/test_target_index.jsonl'
    assert sha(index_path) == coverage['target_index_sha256'] == result['test_index']['sha256']
    index = [json.loads(line) for line in index_path.open()]
    keys = [row['sample_key'] for row in index]
    assert len(index) == len(set(keys)) == coverage['targets']
    assert all(tuple(row['target']) in legal for row in index)
    assert [row['global_row_index'] for row in index] == list(range(len(index)))
    assert cohort['split'] == 'evaluation' and cohort['selected_users'] == 32
    assert cohort['model_outputs_inspected'] is False and 'First 32 distinct user_id' in cohort['selection_rule']
    assert len(cohort['sample_keys']) == len(set(cohort['sample_keys'])) == len(cohort['samples']) == 32
    assert cohort['sample_keys'] == [sample['sample_key'] for sample in cohort['samples']]
    assert cohort['loader']['batch_size_per_device'] == 8 and cohort['loader']['num_workers'] == 0
    history_checks = 0
    for idx, sample in enumerate(cohort['samples']):
        assert sample['target_hidden'] and sample['target'] == cohort['targets'][idx]
        assert tuple(sample['target']) in legal
        raw = cohort['input_ids'][idx]
        mask = cohort['attention_mask'][idx]
        assert len(raw) == len(mask) == 120 and len(sample['visible_history_item_ids']) == 30
        for pos, item_id in enumerate(sample['visible_history_item_ids']):
            values, flags = raw[4*pos:4*pos+4], mask[4*pos:4*pos+4]
            if item_id == -1:
                assert values == [-1]*4 and flags == [0]*4
            else:
                assert values == [int(v) for v in sid[:, item_id]] and flags == [1]*4
                history_checks += 1
    assert sha(directory / 'output/fixed_attention_samples.json') == result['attention_cohort']['sha256']
    assert used['model']['top_k_for_generation'] == 10 and used['model']['shuffle_n'] == 0
    assert used['trainer']['precision'] == '32-true'
    assert used['eval']['evaluator']['top_k_list'] == [5, 10]
    for split in ['train', 'val', 'test']:
        cfg = used['datamodule'][split + '_dataloader_config']
        assert cfg['dataset_config']['semantic_id_map']['sequence_data']['_args_'][0]['file_path'] == expected['sid']['path']
        assert cfg['labels']['sequence_data']['transform']['next_k'] == 4
        assert cfg['padding_token'] == cfg['masking_token'] == -1
        assert cfg['sequence_length'] == 120
    receipt = {
        'status': 'PASS', 'reviewer': '/root/asset_audit', 'domain': args.domain, 'route': 'B',
        'review_time_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'review_scope': 'Non-author source and runtime asset pairing, fixed cohort/item-axis and complete index legality/coverage; no new model/GPU execution',
        'job_id': status['job_id'], 'job_status': status['status'], 'request_sha256': fingerprint,
        'source_commit': invocation['code_identity']['source_commit'], 'snapshot': request['cwd'],
        'capture_version': request['capture_version'], 'assets': expected,
        'original_source_review': {'path': str(HERE / 'asset-source-review.json'), 'sha256': sha(HERE / 'asset-source-review.json')},
        'strict_load': strict, 'shared_sid_table': True, 'parameter_count': loading['parameter_count'],
        'independent_raw_SID_reader': sid_storage, 'all_item_collision_codes_independently_rechecked': count,
        'full_index_unique_targets': len(index), 'all_target_SIDs_legal': True,
        'complete_original_label_and_hidden_group_checks_recorded': True,
        'fixed32_sample_selection': cohort['selection_rule'], 'fixed32_split': cohort['split'],
        'fixed32_visible_item_groups_rechecked_against_original_sid': history_checks,
        'files_sha256_verified': len(entries),
        'essential_inputs': [{'path': str(directory / name), 'sha256': sha(directory / name), 'bytes': (directory / name).stat().st_size} for name in ['request.json', 'accepted.json', 'status.json', 'export-manifest.json', 'output/loading.json', 'output/data_check.json', 'output/test_coverage.json', 'output/test_target_index.jsonl', 'output/fixed_attention_samples.json', 'output/RESULT.json', 'output/used_config.json']],
        'limits': ['Not an independent rerun of the full raw TFRecord-to-target scan; original scanning code and full raw output were reviewed', 'This is formal asset admission, not acceptance of full recommendation or attention results', 'A routes remain excluded for the exact unresolved or missing conditions in the source report'],
    }
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': 'PASS', 'domain': args.domain, 'targets': len(index), 'receipt': str(args.output), 'sha256': sha(args.output)}))


if __name__ == '__main__':
    main()
