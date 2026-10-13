"""Derive manifest from immutable preparation results, never from model loads."""
import hashlib
import json
from pathlib import Path

S = Path('/home/addaswsw/recommend-support/sid-extensions-p0')
R = Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
M = R / 'manifests/sid-extensions'
B = '563dc50b1c7e9e2e05cdfddd13e91a3d1888001e'


def main():
    restore = json.loads((S / 'jobs/new-server/sid-extensions-p0-restore-v1/retrieved/output/RESULT.json').read_text())
    lfs_plan = json.loads((S / 'requests/fetch-lfs-selected-v1-plan.json').read_text())
    assemblies = list((S / 'jobs/new-server').glob('sid-extensions-p0-assemble-lfs-*/retrieved/output/RESULT.json'))
    actual = {}
    for p in assemblies:
        a = json.loads(p.read_text())
        if a['status'] == 'SUCCEEDED':
            actual.update({e['logical_path']: dict(e, actual_preparation_job=p.parents[2].name) for e in a['rows']})
    rows = []
    for e in restore['rows']:
        if e['purpose'] != 'dataset_shard':
            continue
        assert e['actual_sha256'] == e['sha256']
        rows.append({'dataset': e['dataset'], 'purpose': 'fixed_baseline_data_and_embedded_item_user_indexes',
            'category': 'REQUIRED', 'split': Path(e['source_path']).parent.name,
            'logical_path': e['logical_path'], 'actual_server_path': e['actual_path'],
            'resolved_target': e['resolved_target'], 'mode': e['mode'], 'read_only_use': True,
            'read_only_limit': 'Symlink targets remain original 0664; no chmod of shared originals' if 'SYMLINK' in e['route'] else 'New independent regular entity 0444',
            'source_version': B, 'source_original_path': e['source_path'],
            'matching_preserved_capture': e['capture_version'], 'archive_batch': e['batch'],
            'archive_member': e['member_path'], 'entry_id': e['entry_id'],
            'bytes': e['bytes'], 'sha256': e['sha256'], 'lfs_oid': 'sha256:' + e['sha256'],
            'integrity': 'PAYLOAD_PRESENT_HASH_MATCHED', 'provenance': 'PROVENANCE_BOUND',
            'config_result_reference': ['GRID baseline/codex/dataset_source_manifest.json', 'Baseline Results/test_results.json',
                                        'manifests/sid-extensions/baseline-run-records/' + e['dataset']],
            'existing_pair_evidence': 'review/A_RUN_RECORDS_STATIC.json establishes config/source identity only; old T5 pair evidence deliberately not inherited for Qwen',
            'semantic_binding': 'SEMANTIC_BINDING_UNVERIFIED',
            'semantic_limit': 'No content-hash-bound Qwen item/SID/checkpoint runtime pairing proof available; byte equivalence is not semantic validation',
            'model_load': 'MODEL_LOAD_NOT_RUN', 'experiment': 'EXPERIMENT_NOT_AUTHORIZED',
            'not_run': ['loader execution', 'item remapping', 'SID deserialization', 'model/checkpoint load', 'metrics'],
            'recovery': 'Reuse matching server entity or restore only this member from recorded manifest/chunks, verify SHA256',
            'preparation_job': 'sid-extensions-p0-restore-v1', 'route': e['route']})
    for e in lfs_plan['entries']:
        got = actual.get(e['logical_path'])
        if got:
            assert (got['bytes'], got['actual_sha256']) == (e['bytes'], e['sha256'])
        rows.append({'dataset': e['dataset'], 'purpose': e['purpose'], 'category': e['category'],
            'logical_path': e['logical_path'], 'actual_server_path': e['destination'],
            'source_version': B, 'source_original_path': e['repository_path'],
            'bytes': e['bytes'], 'sha256': e['sha256'], 'lfs_oid': e['lfs_oid'],
            'integrity': 'PAYLOAD_PRESENT_HASH_MATCHED' if got else 'PENDING_SERVER_ASSEMBLY_AND_WHOLE_OID',
            'provenance': 'PROVENANCE_BOUND', 'config_result_reference': [e['result_reference'], e['selected_checkpoint'],
                'GRID baseline/codex/server_manifest.json', 'manifests/sid-extensions/baseline-run-records/' + e['dataset']],
            'existing_pair_evidence': 'Fixed result -> selected checkpoint path + effective config/overrides -> SID path; LFS OID+server_manifest bind bytes. archive_validation strict-load boolean lacks exact per-asset hash pairing and is not reused as semantic PASS.',
            'semantic_binding': 'SEMANTIC_BINDING_UNVERIFIED', 'model_load': 'MODEL_LOAD_NOT_RUN',
            'experiment': 'EXPERIMENT_NOT_AUTHORIZED',
            'not_run': ['torch.load', 'pickle.loads', 'Hydra instantiate', 'model compatibility load', 'forward/backward', 'train/test/predict'],
            'recovery': 'Fetch fixed repository LFS OID by 32MiB Range, reuse committed chunk receipts, assemble on server and compare whole SHA256',
            'preparation_job': got['actual_preparation_job'] if got else None,
            'transfer_job_index': 'PREPARATION_JOB_SUMMARY.json; failed attempts and current relay are separate identities',
            'mode': got.get('mode') if got else None, 'nlink': got.get('nlink') if got else None,
            'route': got['route'] if got else 'IN_PROGRESS_FIXED_RANGE_RELAY'})
    with (M / 'ASSET_MANIFEST.jsonl').open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    grouped = {}
    for d in ['beauty', 'toys', 'sports']:
        subset = [e for e in rows if e['dataset'] == d]
        required = [e for e in subset if e['category'] == 'REQUIRED']
        grouped[d] = {'entries': len(subset), 'required_entries': len(required),
                      'required_payloads_present_hash_matched': sum(e['integrity'] == 'PAYLOAD_PRESENT_HASH_MATCHED' for e in required),
                      'missing_or_pending_required': [e['logical_path'] for e in required if e['integrity'] != 'PAYLOAD_PRESENT_HASH_MATCHED'],
                      'bytes': sum(e['bytes'] for e in subset),
                      'semantic_binding': 'SEMANTIC_BINDING_UNVERIFIED', 'model_load': 'MODEL_LOAD_NOT_RUN'}
    (M / 'ASSET_SUMMARY.json').write_text(json.dumps({'baseline_sha': B, 'datasets': grouped, 'entries': len(rows),
        'all_required_payloads_present_hash_matched': all(not e['missing_or_pending_required'] for e in grouped.values()),
        'no_model_execution': True, 'no_data_split_or_item_mapping_change': True,
        'asset_manifest_sha256': hashlib.sha256((M / 'ASSET_MANIFEST.jsonl').read_bytes()).hexdigest()}, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({d: {k: v for k, v in e.items() if k != 'missing_or_pending_required'} for d, e in grouped.items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
