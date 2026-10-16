"""Independent CPU audit of existing fixed32 A/B exports; never run a model."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
from matplotlib import colormaps

REPO = Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
SUPPORT = Path('/home/addaswsw/recommend-support/grid-formal-20261015')
JOBS = Path('/home/addaswsw/recommend-support/sid-extensions-p0/jobs/new-server')
sys.path.insert(0, str(REPO))
from experiments.beauty_extensions.attention.records import load_bundle


def digest(data):
    return hashlib.sha256(data).hexdigest()


def identity(path):
    data = path.read_bytes()
    return {'path': str(path), 'bytes': len(data), 'sha256': digest(data)}


def local_job_asset(asset):
    parts = Path(asset['path']).relative_to('/home/bwwu/grid-audits/sid-extensions-p0/jobs').parts
    path = JOBS/parts[0]/'retrieved'/Path(*parts[1:])
    assert digest(path.read_bytes()) == asset['sha256']
    return path


def verify_png_pixels(path, local, final, mask, product):
    """Check actual cell-center pixels against raw matrices and declared LUT.

    The long black spines locate the two untrimmed image panels independently
    of recorded pixel coordinates. This neither redraws nor alters a PNG.
    """
    with Image.open(path) as image:
        image.load()
        assert image.size == (product['width'], product['height']) == (2880, 1280)
        rgb = np.asarray(image.convert('RGB'))
    assert rgb.max() > rgb.min()
    black = rgb.max(axis=2) < 50
    groups = []
    for y in range(rgb.shape[0]):
        edges = np.flatnonzero(np.diff(np.r_[False, black[y], False]))
        runs = [(left, right) for left, right in zip(edges[::2], edges[1::2])
                if right-left > rgb.shape[1]*0.6]
        if not runs:
            continue
        left, right = max(runs, key=lambda pair: pair[1]-pair[0])
        row = (y, int(left), int(right))
        if not groups or y > groups[-1][-1][0]+1:
            groups.append([row])
        else:
            groups[-1].append(row)
    assert len(groups) >= 5, path
    scale = float(max(local.max(), final.max()))
    assert product['scale_min'] == 0 and product['scale_max'] == scale
    error = 0
    for panel, matrix in enumerate((local, final)):
        top, bottom = groups[2*panel][-1], groups[2*panel+1][0]
        left, right = top[1:]
        assert bottom[1:] == (left, right)
        ys = np.floor(top[0]+(np.arange(4)+0.5)*(bottom[0]-top[0])/4).astype(int)
        xs = np.floor(left+(np.arange(matrix.shape[1])+0.5)*(right-left)/matrix.shape[1]).astype(int)
        actual = rgb[ys[:, None], xs[None, :]]
        expected = colormaps['YlOrRd'](matrix/scale, bytes=True)[..., :3]
        difference = int(np.abs(actual.astype(int)-expected.astype(int)).max())
        assert difference <= 1, (path, panel, difference)
        error = max(error, difference)
    top, bottom = groups[4][0], groups[-1][-1]
    assert top[0] < bottom[0]
    xs = np.floor(top[1]+(np.arange(len(mask))+0.5)*(top[2]-top[1])/len(mask)).astype(int)
    actual_mask = rgb[(top[0]+bottom[0])//2, xs]
    expected_mask = np.repeat(np.where(mask, 0, 255)[:, None], 3, axis=1)
    assert np.abs(actual_mask.astype(int)-expected_mask).max() <= 1, path
    return {'heatmap_cells_checked': int(2*matrix.size), 'encoder_mask_cells_checked': int(len(mask)),
            'maximum_RGB_quantization_difference_0_to_255': error}


def audit(job):
    directory = JOBS/job/'retrieved'
    rows = json.loads((directory/'export-manifest.json').read_bytes())['files']
    manifest_names = {row['path'] for row in rows}
    assert len(manifest_names) == len(rows)
    actual_names = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}
    assert actual_names == manifest_names | {'export-manifest.json'}
    for row in rows:
        path = directory/row['path']
        assert not path.is_symlink()
        data = path.read_bytes()
        assert len(data) == row['bytes'] and digest(data) == row['sha256'], path
    read = lambda name: json.loads((directory/name).read_bytes())
    request, state, accepted = read('request.json'), read('status.json'), read('accepted.json')
    assert request['job_id'] == state['job_id'] == job
    fingerprint = digest(json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode())
    assert accepted['request_sha256'] == fingerprint
    assert state['status'] == 'SUCCEEDED' and state['exit_code'] == 0 and state['cleanup']['verified']
    assert not state.get('worker_may_be_live', False) and request['timeout_seconds'] <= 1200
    result, invocation = read('output/RESULT.json'), read('output/invocation.json')
    deployment = invocation['code_identity']
    assert request['capture_version'] == deployment['source_commit']+'/'+deployment['snapshot_sha256']
    assert digest(json.dumps({'source_commit': deployment['source_commit'], 'files': deployment['files']}, sort_keys=True).encode()) == deployment['snapshot_sha256']
    assert invocation['phase'] == 'attention' and invocation['interpreter'] == '/data/bwwu/envs/grid-20261009-e61baf3/bin/python'
    spec = invocation['spec']; route, n = spec['route'], spec['shuffle_n']
    assert route in ('A', 'B') and n in (0, 3, 4)
    spec_path = REPO/f'experiments/formal_20261015/specs/attention-beauty-{route.lower()}-{n}.json'
    assert json.loads(spec_path.read_bytes()) == spec and digest(spec_path.read_bytes()) == invocation['spec_sha256']
    for row in deployment['files']:
        data = subprocess.check_output(['git', 'show', deployment['source_commit']+':'+row['path']], cwd=REPO)
        assert len(data) == row['bytes'] and digest(data) == row['sha256'], row['path']
    manifest_sha = digest((json.dumps(deployment, indent=2)+'\n').encode())
    assert {row['path']: row['sha256'] for row in request['inputs']}[request['cwd']+'/DEPLOYMENT.json'] == manifest_sha
    assert result['status'] == 'COMPLETE' and result['formal'] and result['samples'] == 32
    assert result['shuffle_n'] == n and all(result[k] == spec[k] for k in ('config', 'sid', 'checkpoint'))
    cohort_path = local_job_asset(spec['attention_cohort'])
    cohort = json.loads(cohort_path.read_bytes())
    assert cohort['selected_users'] == 32 and not cohort['model_outputs_inspected']
    assert len(set(cohort['sample_keys'])) == 32
    loading = read('output/loading.json')
    strict = loading['strict_load']
    assert not strict['missing_keys'] and not strict['unexpected_keys']
    assert strict['all_parameter_keys_shapes_values_equal'] and strict['embedded_sid_exact']
    assert loading['shared_sid_table'] is (route == 'B')
    used = read('output/used_config.json')
    vocabulary = used['model']['decoder']['embed_tokens']['num_embeddings']
    sample_rows, batch_rows = [], []
    total_pixel_cells = total_mask_cells = nonlocal_rows = 0
    masked_max = row_sum_error = maximum_plot_error = 0.0
    for index in range(4):
        root = directory/'output'/f'batch-{index:02d}'
        record_path = root/'bundle/record.json'
        bundle = load_bundle(record_path)
        keys = cohort['sample_keys'][index*8:(index+1)*8]
        capture = bundle['capture']; metadata = bundle['metadata']; paths = bundle['paths']
        assert capture['sample_keys'] == keys and capture['selected_indices'] == list(range(8))
        assert capture['batch_size'] == 8 and capture['beam_size'] == 10 and capture['encoder_calls'] == 1
        assert capture['backend']['name'] == 'eager'
        assert metadata['spec'] == spec and metadata['engineering_only'] is False and metadata['shuffle_n'] == n
        assert result['batches'][index]['sample_keys'] == keys
        assert result['batches'][index]['bundle_sha256'] == digest(record_path.read_bytes())
        assert np.array_equal(bundle['input_sids'], np.asarray(cohort['input_ids'][index*8:(index+1)*8]))
        assert np.array_equal(bundle['input_mask'], np.asarray(cohort['attention_mask'][index*8:(index+1)*8]))
        grouped_mask = bundle['input_mask'].reshape(8, 30, 4)
        assert np.all(grouped_mask == grouped_mask[..., -1:])
        mask = bundle['encoder_mask']
        assert mask.shape == (8, 150) and np.isin(mask, [0, 1]).all()
        assert np.array_equal(mask.reshape(8, 30, 5), np.repeat(grouped_mask[..., -1:], 5, axis=-1))
        permutations = metadata['source_permutations']
        if n:
            permutations = np.asarray(permutations)
            assert permutations.shape == (8, 30, 4)
            assert np.array_equal(np.sort(permutations, axis=-1), np.broadcast_to(np.arange(4), permutations.shape))
            if n == 3:
                assert np.all(permutations[..., 3] == 3)
            visible_permutations = permutations[grouped_mask[..., -1].astype(bool)]
            permutation_summary = {'visible_occurrences': len(visible_permutations),
                'distinct_visible_permutations': len(np.unique(visible_permutations, axis=0)),
                'identity_visible_occurrences': int(np.all(visible_permutations == np.arange(4), axis=1).sum())}
            assert permutation_summary['distinct_visible_permutations'] > 1
        else:
            assert permutations is None
            permutation_summary = {'shuffle_off': True}
        for sample, positions in enumerate(bundle['position_map']):
            assert len(positions) == 150
            for slot, position in enumerate(positions):
                item, destination = divmod(slot, 5)
                visible = int(grouped_mask[sample, item, -1])
                assert position['encoder_position'] == slot and position['history_slot'] == item and position['visible'] == visible
                if destination == 4:
                    assert position['role'] == 'sep'
                    continue
                original = destination if not n or not visible else int(permutations[sample, item, destination])
                sampled = destination if not n else int(permutations[sample, item, destination])
                raw_sid = int(bundle['input_sids'][sample, item*4+original])
                assert position['role'] == 'sid_lookup_vector'
                assert position['lookup_code_position'] == original and position['destination_code_position'] == destination
                assert position['sampled_source_code_position'] == sampled and position['raw_sid_value'] == raw_sid
                assert position['lookup_table_index'] == (raw_sid+original*vocabulary if visible else 0)
                assert position['padding_group_kept_at_input_positions'] is (not bool(visible))
                assert (position['item_id'] >= 0) is bool(visible)
        assert len(bundle['steps']) == 4
        independent_final_heads = np.empty((8, 4, 6, 150), dtype=np.float32)
        independent_local_heads = np.empty_like(independent_final_heads)
        independent_parents = np.zeros((8, 4), dtype=np.int64)
        independent_children = np.zeros_like(independent_parents)
        for h, step in enumerate(bundle['steps']):
            parents_width = 1 if h == 0 else 10
            assert step['hierarchy'] == h and step['parent_beams'] == parents_width
            assert step['cross_heads'].shape == (8, parents_width, 6, 150)
            assert step['generated_ids'].shape == (8, 10, h+1)
            assert step['last_query_index'] == step['query_length_before_last_selection']-1
            assert np.isfinite(step['scores']).all() and np.all(np.diff(step['scores'], axis=1) <= 0)
            independent_local_heads[:, h] = step['cross_heads'][:, 0]
            for sample in range(8):
                current = step['generated_ids'][sample]
                assert len({tuple(row) for row in current}) == 10
                previous = [()] if h == 0 else [tuple(row) for row in bundle['steps'][h-1]['generated_ids'][sample]]
                lookup = {prefix: p for p, prefix in enumerate(previous)}
                expected_parents = [lookup[tuple(row[:-1])] for row in current]
                assert np.array_equal(expected_parents, step['parent_indices'][sample])
                final_sid = bundle['final_generated_ids'][sample, 0]
                parent = lookup[tuple(final_sid[:h])]
                child = [tuple(row) for row in current].index(tuple(final_sid[:h+1]))
                independent_parents[sample, h] = parent
                independent_children[sample, h] = child
                independent_final_heads[sample, h] = step['cross_heads'][sample, parent]
            expected_prefix = np.empty((8, 0), dtype=np.int64) if h == 0 else bundle['steps'][h-1]['generated_ids'][:, 0]
            assert np.array_equal(expected_prefix, step['local_first_prefix'])
        assert np.array_equal(bundle['final_generated_ids'], bundle['steps'][-1]['generated_ids'])
        assert np.array_equal(bundle['final_scores'], bundle['steps'][-1]['scores'])
        for name, value in [('local_first_heads', independent_local_heads), ('final_top1_heads', independent_final_heads),
                            ('final_parent_indices', independent_parents), ('final_child_indices', independent_children),
                            ('backtracked_top1_sid', bundle['final_generated_ids'][:, 0])]:
            assert np.array_equal(paths[name], value), name
        for path, values in [('local_first', independent_local_heads), ('final_top1', independent_final_heads)]:
            assert np.isfinite(values).all() and (values >= 0).all()
            assert np.array_equal(paths[path+'_cross'], values.mean(axis=2))
            row_sum_error = max(row_sum_error, float(np.abs(values.sum(axis=-1)-1).max()))
            masked_max = max(masked_max, float((values*(1-mask[:, None, None])).sum(axis=-1).max()))
        assert row_sum_error < 1e-5 and masked_max == 0
        nonlocal_rows += int((independent_parents != 0).sum())
        checks = json.loads((root/'off_on_checks.json').read_bytes())
        assert checks['sid_exact'] and checks['scores']['exact'] and checks['scores']['max_absolute'] == 0
        assert all(checks[k] for k in ('cpu_rng_same', 'cuda_rng_same', 'parameter_sha256_unchanged', 'model_overrides_restored', 'stack_hooks_restored'))
        assert result['batches'][index]['observer_off_on_checks'] == checks
        batch_result = json.loads((root/'RESULT.json').read_bytes())
        assert batch_result['status'] == 'PASS' and batch_result['exception_cleanup'] and not batch_result['engineering_only']
        assert batch_result['saved']['record_sha256'] == digest(record_path.read_bytes())
        assert batch_result['saved']['arrays_sha256'] == digest((record_path.parent/'arrays.npz').read_bytes())
        figures = json.loads((root/'figures/plot_receipt.json').read_bytes())
        assert figures['source_record_sha256'] == digest(record_path.read_bytes()) and len(figures['products']) == 8
        for sample, product in enumerate(figures['products']):
            path = root/'figures'/product['file']; image_bytes = path.read_bytes()
            assert len(image_bytes) == product['bytes'] and digest(image_bytes) == product['sha256']
            assert product['sample_record_index'] == sample and product['reopened']
            pixel = verify_png_pixels(path, paths['local_first_cross'][sample], paths['final_top1_cross'][sample], mask[sample], product)
            total_pixel_cells += pixel['heatmap_cells_checked']; total_mask_cells += pixel['encoder_mask_cells_checked']
            maximum_plot_error = max(maximum_plot_error, pixel['maximum_RGB_quantization_difference_0_to_255'])
            sample_rows.append({'sample_key': keys[sample], 'batch': index, 'index_in_batch': sample,
                'final_Top1_SID': bundle['final_generated_ids'][sample, 0].tolist(),
                'final_parent_indices': independent_parents[sample].tolist(), 'final_child_indices': independent_children[sample].tolist(),
                'visible_encoder_positions': int(mask[sample].sum()), 'figure': identity(path)})
        batch_rows.append({'batch': index, 'bundle': identity(record_path), 'arrays': identity(record_path.parent/'arrays.npz'),
            'figures': identity(root/'figures/plot_receipt.json'), 'off_on': identity(root/'off_on_checks.json'),
            'parameter_sha256': metadata['parameter_sha256'], 'actual_generate_call_identity': job+f'/batch-{index:02d}/observer-encoder-1',
            'query_lengths_before_last_selection': [step['query_length_before_last_selection'] for step in bundle['steps']],
            'permutation_summary': permutation_summary})
    assert [row['sample_key'] for row in sample_rows] == cohort['sample_keys']
    assert len({row['parameter_sha256'] for row in batch_rows}) == 1
    elapsed = (datetime.datetime.fromisoformat(state['ended_at'])-datetime.datetime.fromisoformat(state['started_at'])).total_seconds()
    assert elapsed < 1200
    receipt = {'schema': 1, 'status': 'INDEPENDENT_FORMAL_ATTENTION_RESULT_PASS', 'reviewer_task': '/root/implementation_review',
        'reviewed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'job_id': job, 'route': route, 'shuffle_n': n,
        'main_working_baseline': route == 'B' and n == 0,
        'actual_runner_status': state['status'], 'exit_code': state['exit_code'], 'cleanup_verified': True,
        'source_commit': deployment['source_commit'], 'snapshot_sha256': deployment['snapshot_sha256'],
        'manifest_sha256': manifest_sha, 'request_fingerprint': fingerprint, 'export_manifest': identity(directory/'export-manifest.json'),
        'original_export_member_count_verified': len(rows), 'program': identity(Path(__file__)),
        'invocation': identity(directory/'output/invocation.json'), 'result': identity(directory/'output/RESULT.json'),
        'timeout_seconds': request['timeout_seconds'], 'actual_worker_wall_seconds': elapsed,
        'cohort': identity(cohort_path), 'samples': 32, 'batches': batch_rows, 'sample_checks': sample_rows,
        'matrix_shape_per_path': [32, 4, 6, 150], 'head_mean_shape_per_path': [32, 4, 150],
        'four_steps_ten_beams_exact_unique_prefix_parent_reconstruction': True,
        'final_Top1_ancestor_rows_using_nonzero_parents': nonlocal_rows,
        'raw_head_mean_and_both_paths_exactly_recomputed': True,
        'maximum_raw_attention_row_sum_error': row_sum_error, 'maximum_padding_attention_mass': masked_max,
        'encoder_position_identity': 'all 30 items x (four looked-up SID vectors + SEP), mask preserved, offset applied before vector permutation; hidden groups stay at original slots',
        'off_on': 'Each real batch records exact all-K SID/scores, equal CPU/CUDA RNG completion, unchanged parameter hash and restored observer hooks/methods after replaying identical RNG starting states.',
        'figure_evidence': {'PNGs_reopened': 32, 'both_panel_raw_attention_cell_center_colors_checked': total_pixel_cells,
            'mask_cell_center_colors_checked': total_mask_cells, 'maximum_RGB_quantization_difference_0_to_255': maximum_plot_error,
            'plot_rule': 'YlOrRd LUT, vmin0, one shared per-figure raw maximum, four whole rows and all150 positions; no row normalization or cropping'},
        'cleanup_evidence_and_limits': {'normal_observer_hooks_and_overrides_restored_recorded': True,
            'intentional_observer_exception_cleanup_recorded': True, 'runner_owned_process_cleanup_verified': True,
            'outer_clear_runtime_finally_and_testing_flag_reset': 'verified unchanged source paths and successful completion; no serialized post-model-object cache/device snapshot exists',
            'RNG_scope': 'Runtime booleans and unchanged source confirm replay/start and end equality; raw CPU/CUDA RNG byte states are not stored.',
            'Shuffle_scope': 'Stored permutations belong to this separate fixed32 actual generate capture, not the previous full-test random draws; observer call ID is explicitly derived from job/batch.'},
        'scientific_similarity_or_invariance_claim': None, 'new_model_GPU_calls': 0, 'research_source_or_Git_writes': 0,
        'scope': 'Independent raw export/NPZ/prefix/mask/pixel validation; no self-signature of reviewer-authored MLP train.py and no repeated model evaluation.'}
    destination = SUPPORT/'attention-results-review'/(job+'-review.json')
    assert not destination.exists()
    destination.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    return {'review': identity(destination), 'status': receipt['status'], 'job': job, 'nonlocal_ancestor_rows': nonlocal_rows,
            'row_sum_error': row_sum_error, 'padding_mass': masked_max, 'raw_plot_cell_checks': total_pixel_cells}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('jobs', nargs='+')
    args = p.parse_args()
    for job in args.jobs:
        print(json.dumps(audit(job), ensure_ascii=False), flush=True)
