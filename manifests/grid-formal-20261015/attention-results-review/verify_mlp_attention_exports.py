"""Independent CPU validation of completed Toys/Sports item-MLP attention."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess

import numpy as np

from verify_attention_exports import (REPO, SUPPORT, JOBS, digest, identity,
                                      load_bundle, verify_png_pixels)


def local_dependency(asset):
    parts = Path(asset['path']).relative_to('/home/bwwu/grid-audits/sid-extensions-p0/jobs').parts
    candidates = [JOBS/parts[0]/'retrieved'/Path(*parts[1:])]
    candidates += [p/parts[0]/Path(*parts[1:]) for p in SUPPORT.glob('collected-*') if p.is_dir()]
    matches = [p for p in candidates if p.is_file() and digest(p.read_bytes()) == asset['sha256']]
    assert matches, asset
    return matches[0]


def audit(job):
    directory = JOBS/job/'retrieved'
    read = lambda name: json.loads((directory/name).read_bytes())
    manifest = read('export-manifest.json')
    assert len({r['path'] for r in manifest['files']}) == len(manifest['files'])
    assert {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()} == {r['path'] for r in manifest['files']} | {'export-manifest.json'}
    for row in manifest['files']:
        p = directory/row['path']; data = p.read_bytes()
        assert not p.is_symlink() and len(data) == row['bytes'] and digest(data) == row['sha256']
    request, accepted, state = read('request.json'), read('accepted.json'), read('status.json')
    fingerprint = digest(json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode())
    assert accepted['request_sha256'] == fingerprint and request['job_id'] == state['job_id'] == job
    assert state['status'] == 'SUCCEEDED' and state['exit_code'] == 0 and state['cleanup']['verified']
    assert not state.get('worker_may_be_live', False) and request['timeout_seconds'] == 1200
    invocation, result = read('output/invocation.json'), read('output/RESULT.json')
    spec, deployment = invocation['spec'], invocation['code_identity']
    domain = spec['domain']; assert domain in ('toys', 'sports') and spec['route'] == 'MLP' and spec['shuffle_n'] == 0
    sp = REPO/f'experiments/formal_20261015/specs/attention-{domain}-mlp.json'
    assert json.loads(sp.read_bytes()) == spec and invocation['spec_sha256'] == digest(sp.read_bytes())
    assert invocation['phase'] == 'attention'
    assert request['capture_version'] == deployment['source_commit']+'/'+deployment['snapshot_sha256']
    assert digest(json.dumps({'source_commit': deployment['source_commit'], 'files': deployment['files']}, sort_keys=True).encode()) == deployment['snapshot_sha256']
    for row in deployment['files']:
        b = subprocess.check_output(['git', 'show', deployment['source_commit']+':'+row['path']], cwd=REPO)
        assert len(b) == row['bytes'] and digest(b) == row['sha256']
    code = {r['path']: r['sha256'] for r in deployment['files'] if r['path'].endswith('.py')}
    reviewp = SUPPORT/f'mlp-attention-admission/{domain}-gate-review.json'
    tokenp = SUPPORT/f'mlp-attention-admission/{domain}-authorization.json'
    review, token = json.loads(reviewp.read_bytes()), json.loads(tokenp.read_bytes())
    plan = digest(json.dumps({k: v for k, v in spec.items() if k not in ('formal_approved', 'nonauthor_gate')}, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode())
    assert review['status'] == 'PREFLIGHT_GATE_PASS' and review['plan_sha256'] == token['plan_sha256'] == plan
    assert review['source_python_sha256'] == token['source_python_sha256'] == code and len(code) == 171
    assert token['independent_review']['sha256'] == digest(reviewp.read_bytes()) and spec['nonauthor_gate']['sha256'] == digest(tokenp.read_bytes())
    assert token['assets'] == {k: spec[k] for k in ('config', 'sid', 'checkpoint')}
    trainp, trainstatep = local_dependency(spec['evaluation_result']), local_dependency(spec['evaluation_status'])
    train, trainstate = json.loads(trainp.read_bytes()), json.loads(trainstatep.read_bytes())
    assert trainstate['status'] == 'SUCCEEDED' and trainstate['exit_code'] == 0 and trainstate['cleanup']['verified']
    assert train['status'] == 'COMPLETE' and train['formal'] and train['test_complete']
    assert train['best_checkpoint']['sha256'] == spec['checkpoint']['sha256']
    assert result['status'] == 'COMPLETE' and result['formal'] and result['samples'] == 32
    assert result['shuffle_n'] == 0 and all(result[k] == spec[k] for k in ('checkpoint', 'sid', 'config'))
    cohortp = local_dependency(spec['attention_cohort']); cohort = json.loads(cohortp.read_bytes())
    assert cohort['selected_users'] == 32 and len(set(cohort['sample_keys'])) == 32 and not cohort['model_outputs_inspected']
    strict = read('output/loading.json')['strict_load']
    assert not strict['missing_keys'] and not strict['unexpected_keys'] and strict['all_parameter_keys_shapes_values_equal'] and strict['embedded_sid_exact']
    assert read('output/loading.json')['shared_sid_table']
    assert strict['global_step'] == (4400 if domain == 'toys' else 4900)
    elapsed = (datetime.datetime.fromisoformat(state['ended_at'])-datetime.datetime.fromisoformat(state['started_at'])).total_seconds()
    trained = (datetime.datetime.fromisoformat(trainstate['ended_at'])-datetime.datetime.fromisoformat(trainstate['started_at'])).total_seconds()
    allowance = spec['resource_allowance']
    assert abs(trained-allowance['actual_training_validation_and_test_worker_wall_seconds']) < 1e-6
    assert elapsed < 1200 and trained+elapsed < 43200 and trained+1200 == allowance['maximum_cumulative_worker_wall_with_capture_seconds']
    rows, batches = [], []
    row_error = padding = 0.0; pixel_cells = mask_cells = nonlocal_rows = 0
    for index in range(4):
        root = directory/'output'/f'batch-{index:02d}'; recordp = root/'bundle/record.json'; b = load_bundle(recordp)
        keys = cohort['sample_keys'][index*8:(index+1)*8]
        assert b['capture']['sample_keys'] == keys and b['capture']['selected_indices'] == list(range(8))
        assert b['capture']['encoder_calls'] == 1 and b['capture']['beam_size'] == 10 and b['capture']['backend']['name'] == 'eager'
        assert b['metadata']['spec'] == spec and b['metadata']['source_permutations'] is None and not b['metadata']['engineering_only']
        assert np.array_equal(b['input_sids'], cohort['input_ids'][index*8:(index+1)*8])
        assert np.array_equal(b['input_mask'], cohort['attention_mask'][index*8:(index+1)*8])
        groups = b['input_sids'].reshape(8, 30, 4); masks = b['input_mask'].reshape(8, 30, 4)
        assert np.all(masks == masks[..., -1:]) and np.array_equal(b['encoder_mask'], masks[..., -1])
        assert b['encoder_mask'].shape == (8, 30)
        for sample, positions in enumerate(b['position_map']):
            assert len(positions) == 30
            for item, p in enumerate(positions):
                assert p['role'] == 'item_aggregation' and p['aggregation_axis'] == 'H*E'
                assert p['encoder_position'] == p['history_slot'] == item and p['visible'] == masks[sample, item, -1]
                assert np.array_equal(p['original_sids'], groups[sample, item]) and np.array_equal(p['original_code_mask'], masks[sample, item])
                assert (p['item_id'] >= 0) is bool(p['visible'])
        final_heads = np.empty((8, 4, 6, 30), dtype=np.float32); local_heads = np.empty_like(final_heads)
        parents = np.zeros((8, 4), dtype=np.int64); children = np.zeros_like(parents)
        assert len(b['steps']) == 4
        for h, step in enumerate(b['steps']):
            assert step['hierarchy'] == h and step['cross_heads'].shape == (8, 1 if h == 0 else 10, 6, 30)
            local_heads[:, h] = step['cross_heads'][:, 0]
            for sample in range(8):
                previous = [()] if h == 0 else [tuple(t) for t in b['steps'][h-1]['generated_ids'][sample]]
                current = [tuple(t) for t in step['generated_ids'][sample]]
                assert len(set(current)) == len(current) == 10
                lookup = {prefix: p for p, prefix in enumerate(previous)}
                assert np.array_equal([lookup[prefix[:-1]] for prefix in current], step['parent_indices'][sample])
                target = tuple(b['final_generated_ids'][sample, 0])
                parent = lookup[target[:h]]; child = current.index(target[:h+1])
                parents[sample, h] = parent; children[sample, h] = child; final_heads[sample, h] = step['cross_heads'][sample, parent]
            assert np.isfinite(step['scores']).all() and np.all(np.diff(step['scores'], axis=1) <= 0)
        paths = b['paths']
        for name, actual in [('local_first_heads', local_heads), ('final_top1_heads', final_heads), ('final_parent_indices', parents), ('final_child_indices', children), ('backtracked_top1_sid', b['final_generated_ids'][:, 0])]:
            assert np.array_equal(paths[name], actual)
        assert np.array_equal(b['final_generated_ids'], b['steps'][-1]['generated_ids']) and np.array_equal(b['final_scores'], b['steps'][-1]['scores'])
        for name, heads in [('local_first', local_heads), ('final_top1', final_heads)]:
            assert np.isfinite(heads).all() and (heads >= 0).all() and np.array_equal(paths[name+'_cross'], heads.mean(axis=2))
            row_error = max(row_error, float(np.abs(heads.sum(axis=-1)-1).max()))
            padding = max(padding, float((heads*(1-b['encoder_mask'][:, None, None])).sum(axis=-1).max()))
        assert row_error < 1e-5 and padding == 0
        nonlocal_rows += int((parents != 0).sum())
        checks = json.loads((root/'off_on_checks.json').read_bytes()); batch_result = json.loads((root/'RESULT.json').read_bytes())
        assert checks['sid_exact'] and checks['scores']['exact'] and checks['scores']['max_absolute'] == 0
        assert all(checks[k] for k in ('cpu_rng_same', 'cuda_rng_same', 'parameter_sha256_unchanged', 'model_overrides_restored', 'stack_hooks_restored'))
        assert batch_result['exception_cleanup'] and batch_result['status'] == 'PASS' and not batch_result['engineering_only']
        assert result['batches'][index]['sample_keys'] == keys and result['batches'][index]['observer_off_on_checks'] == checks
        assert result['batches'][index]['bundle_sha256'] == digest(recordp.read_bytes())
        figures = json.loads((root/'figures/plot_receipt.json').read_bytes()); assert figures['source_record_sha256'] == digest(recordp.read_bytes()) and len(figures['products']) == 8
        for sample, product in enumerate(figures['products']):
            image = root/'figures'/product['file']; assert identity(image)['sha256'] == product['sha256'] and image.stat().st_size == product['bytes']
            pixel = verify_png_pixels(image, paths['local_first_cross'][sample], paths['final_top1_cross'][sample], b['encoder_mask'][sample], product)
            pixel_cells += pixel['heatmap_cells_checked']; mask_cells += pixel['encoder_mask_cells_checked']
            rows.append({'sample_key': keys[sample], 'final_Top1_SID': b['final_generated_ids'][sample, 0].tolist(),
                'final_parent_indices': parents[sample].tolist(), 'visible_aggregated_item_tokens': int(b['encoder_mask'][sample].sum()), 'PNG': identity(image)})
        batches.append({'batch': index, 'record': identity(recordp), 'arrays': identity(recordp.parent/'arrays.npz'),
            'off_on': identity(root/'off_on_checks.json'), 'plot_receipt': identity(root/'figures/plot_receipt.json'),
            'parameter_sha256': b['metadata']['parameter_sha256']})
    assert [r['sample_key'] for r in rows] == cohort['sample_keys'] and len({x['parameter_sha256'] for x in batches}) == 1
    receipt = {'status': 'INDEPENDENT_FORMAL_MLP_ATTENTION_RESULT_PASS', 'reviewer_task': '/root/implementation_review',
        'reviewed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'job_id': job, 'domain': domain,
        'scope': 'Independent attention observation and raw-result audit only. This reviewer authored formal train.py and does not self-sign its training; the already accepted training/best record and separate source/results reviewer gate are input dependencies.',
        'source_commit': deployment['source_commit'], 'snapshot_sha256': deployment['snapshot_sha256'], 'request_fingerprint': fingerprint,
        'original_export_manifest': identity(directory/'export-manifest.json'), 'original_export_members_verified': len(manifest['files']),
        'program': identity(Path(__file__)), 'PNG_audit_helper': identity(Path(__file__).parent/'verify_attention_exports.py'),
        'invocation': identity(directory/'output/invocation.json'), 'result': identity(directory/'output/RESULT.json'),
        'checkpoint': spec['checkpoint'], 'cohort': identity(cohortp), 'independent_attention_gate': identity(reviewp),
        'prior_accepted_training_dependency': identity(trainp), 'prior_training_status': identity(trainstatep),
        'budget': {'attention_requested_seconds': 1200, 'actual_attention_worker_wall_seconds': elapsed,
            'original_training_validation_test_wall_seconds': trained, 'actual_cumulative_domain_wall_seconds': trained+elapsed,
            'original_domain_cap_seconds': 43200, 'remaining_original_domain_budget_after_capture_seconds': 43200-trained-elapsed,
            'no_domain_transfer_or_new_budget': True},
        'samples': 32, 'batches': batches, 'sample_checks': rows, 'raw_path_head_shape': [32, 4, 6, 30],
        'original_120_SID_slots_preserved_and_30_item_level_masks_exact': True, 'SEP_or_user_token_in_actual_encoder': False,
        'unique_exact_prefix_parent_map_and_final_Top1_and_local_first_paths_verified': True,
        'actual_nonzero_final_ancestor_parent_rows': nonlocal_rows,
        'maximum_raw_row_sum_error': row_error, 'maximum_padding_attention_mass': padding,
        'PNGs_reopened_and_pixel_LUT_verified': 32, 'raw_attention_heatmap_cells_checked': pixel_cells, 'mask_cells_checked': mask_cells,
        'off_on_and_cleanup': 'Exact SID/scores, same completion CPU/CUDA RNG, unchanged parameters and observer hooks/overrides, intentional exception cleanup probe, and runner owned-process cleanup are recorded and source-bound. Outer finally cache reset is an unchanged source fact; no serialized post-cache/device object state or raw RNG bytes exist.',
        'scientific_similarity_or_invariance_claim': None, 'new_model_or_GPU_calls': 0, 'research_source_Git_or_original_evidence_writes': 0}
    out = SUPPORT/'attention-results-review'/(job+'-review.json'); assert not out.exists()
    out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(identity(out)|{'status': receipt['status'], 'job': job, 'pixel_cells': pixel_cells, 'row_error': row_error, 'padding_mass': padding}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('jobs', nargs='+')
    for job in p.parse_args().jobs:
        audit(job)
