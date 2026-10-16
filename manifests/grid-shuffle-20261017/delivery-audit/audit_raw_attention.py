"""Independent final-delivery raw NPZ and descriptive-statistics crosscheck.

Reads all original numeric arrays directly with NumPy allow_pickle=False.
Rechecks original parents, selected heads/matrices, masks and all 2304 saved
descriptive pairs using the already approved equations. No model imports,
generation, recoloring, output rewrites or cross-Shuffle column comparison.
"""
import argparse
import datetime as dt
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

METRICS = ('cosine', 'pearson', 'total_variation', 'jensen_shannon_nats')
PATHS = ('final_top1_cross', 'local_first_cross')


def identity(path):
    raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def load_direct(record):
    document = json.loads(record.read_bytes())
    assert document['schema'] == 'grid-beauty-cross-attention-v1'
    assert document['arrays_file'] == 'arrays.npz'
    arrays_file = record.with_name('arrays.npz')
    assert identity(arrays_file)['sha256'] == document['arrays_sha256']
    seen = set()
    with np.load(arrays_file, allow_pickle=False) as arrays:
        def decode(value):
            if isinstance(value, dict):
                if '__array__' in value:
                    key = value['__array__']
                    data = arrays[key].copy()
                    assert data.dtype.kind in 'biuf'
                    assert list(data.shape) == value['shape'] and str(data.dtype) == value['dtype']
                    assert data.dtype.kind not in 'f' or np.isfinite(data).all()
                    seen.add(key)
                    return data
                return {key: decode(child) for key, child in value.items()}
            if isinstance(value, list):
                return [decode(child) for child in value]
            return value
        bundle = decode(document['bundle'])
        assert seen == set(arrays.files)
    return bundle, len(seen)


def same_number(actual, expected):
    if actual is None or expected is None:
        assert actual is expected
    else:
        assert abs(actual - expected) <= 2e-12, (actual, expected)


def pair_values(x, y):
    cosine = float(np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y))))
    a, b = x - x.mean(), y - y.mean()
    denominator = np.sqrt(np.dot(a, a) * np.dot(b, b))
    pearson = float(np.dot(a, b) / denominator) if denominator > 0 else None
    p, q = x / x.sum(), y / y.sum()
    middle = 0.5 * (p + q)
    positive_p, positive_q = p > 0, q > 0
    js = 0.5 * (float(np.dot(p[positive_p], np.log(p[positive_p] / middle[positive_p])))
                + float(np.dot(q[positive_q], np.log(q[positive_q] / middle[positive_q]))))
    return {'cosine': cosine, 'pearson': pearson, 'total_variation': float(np.abs(p - q).sum() / 2),
            'jensen_shannon_nats': max(0.0, js)}


def audit_job(row, directory):
    assert len(row['input_files']) == 8
    assert len({source['path'] for source in row['input_files']}) == 8
    for source in row['input_files']:
        assert identity(Path(source['path'])) == source
    result = json.loads((directory / 'output/RESULT.json').read_bytes())
    assert result['status'] == 'COMPLETE' and result['formal'] and result['samples'] == 32
    expected_keys = [key for batch in result['batches'] for key in batch['sample_keys']]
    assert len(set(expected_keys)) == len(expected_keys) == 32
    source_rows = row['per_sample_query_pairs']
    records_by_pair = {(entry['sample_key'], entry['path'], tuple(entry['query_pair'])): entry for entry in source_rows}
    assert len(source_rows) == len(records_by_pair) == 384
    keys, checked = [], []
    array_count = parent_count = pair_count = 0
    maximum_padding = maximum_row_error = 0.0
    independent = []
    for batch_index in range(4):
        record = directory / 'output' / f'batch-{batch_index:02d}' / 'bundle/record.json'
        bundle, arrays = load_direct(record)
        array_count += arrays
        checked.extend([identity(record), identity(record.with_name('arrays.npz'))])
        batch_keys = bundle['capture']['sample_keys']
        assert batch_keys == expected_keys[batch_index * 8:(batch_index + 1) * 8]
        keys.extend(batch_keys)
        assert bundle['capture']['selected_indices'] == list(range(8))
        mask = bundle['encoder_mask']
        assert mask.shape[0] == 8 and np.isin(mask, [0, 1]).all()
        width = mask.shape[1]
        assert len(bundle['steps']) == 4
        final_sid = bundle['final_generated_ids'][:, 0]
        assert final_sid.shape == (8, 4)
        final_heads = np.empty((8, 4, 6, width), dtype=np.float32)
        local_heads = np.empty_like(final_heads)
        for hierarchy, step in enumerate(bundle['steps']):
            assert step['cross_heads'].shape == (8, 1 if hierarchy == 0 else 10, 6, width)
            assert step['generated_ids'].shape == (8, 10, hierarchy + 1)
            local_heads[:, hierarchy] = step['cross_heads'][:, 0]
            for sample in range(8):
                earlier = [()] if hierarchy == 0 else [tuple(r) for r in bundle['steps'][hierarchy - 1]['generated_ids'][sample]]
                assert len(set(earlier)) == len(earlier)
                lookup = {prefix: index for index, prefix in enumerate(earlier)}
                children = [tuple(child) for child in step['generated_ids'][sample]]
                assert len(set(children)) == 10
                parents = [lookup[child[:-1]] for child in children]
                assert np.array_equal(step['parent_indices'][sample], np.asarray(parents))
                parent = lookup[tuple(final_sid[sample, :hierarchy])]
                child = children.index(tuple(final_sid[sample, :hierarchy + 1]))
                assert bundle['paths']['final_parent_indices'][sample, hierarchy] == parent
                assert bundle['paths']['final_child_indices'][sample, hierarchy] == child
                final_heads[sample, hierarchy] = step['cross_heads'][sample, parent]
                parent_count += len(parents)
        assert np.array_equal(bundle['paths']['backtracked_top1_sid'], final_sid)
        assert np.array_equal(bundle['paths']['final_top1_heads'], final_heads)
        assert np.array_equal(bundle['paths']['local_first_heads'], local_heads)
        for path, heads in zip(PATHS, (final_heads, local_heads)):
            matrix = bundle['paths'][path]
            assert matrix.shape == (8, 4, width)
            assert np.array_equal(matrix, heads.mean(axis=2))
            raw = matrix.astype(np.float64)
            assert np.isfinite(raw).all() and (raw >= 0).all()
            hidden = (raw * (1 - mask[:, None, :])).sum(axis=-1)
            maximum_padding = max(maximum_padding, float(hidden.max()))
            maximum_row_error = max(maximum_row_error, float(np.abs(raw.sum(axis=-1) - 1).max()))
            for sample, sample_key in enumerate(batch_keys):
                visible = raw[sample][:, mask[sample].astype(bool)]
                for first, second in itertools.combinations(range(4), 2):
                    token = (sample_key, path, (first + 1, second + 1))
                    saved = records_by_pair[token]
                    assert saved['visible_encoder_positions'] == int(mask[sample].sum())
                    assert saved['all_encoder_positions'] == width
                    assert saved['plot_relative'] == f'output/batch-{batch_index:02d}/figures/sample_{sample:03d}_cross_attention.png'
                    values = pair_values(visible[first], visible[second])
                    for metric in METRICS:
                        same_number(saved[metric], values[metric])
                    independent.append({'sample_key': sample_key, 'path': path, **values})
                    pair_count += 1
    assert keys == expected_keys and pair_count == 384
    assert {source['path']: source for source in row['input_files']} == {source['path']: source for source in checked}
    same_number(row['maximum_masked_mass'], maximum_padding)
    same_number(row['maximum_raw_row_sum_error'], maximum_row_error)
    for path in PATHS:
        rows = [entry for entry in independent if entry['path'] == path]
        assert len(rows) == 192
        for metric in METRICS:
            values = np.asarray([entry[metric] for entry in rows if entry[metric] is not None], dtype=np.float64)
            if values.size == 0:
                assert metric not in row['summary'][path]
                continue
            saved = row['summary'][path][metric]
            assert saved['n'] == len(values)
            for field, value in [('min', values.min()), ('median', np.median(values)), ('mean', values.mean()), ('max', values.max())]:
                same_number(saved[field], float(value))
        # Preserve the declared producer's deterministic tie rule exactly after
        # independently checking every saved constituent against raw arrays.
        saved_rows = [entry for entry in source_rows if entry['path'] == path]
        per_user = [{'sample_key': key,
                     'mean_pair_cosine': float(np.mean([entry['cosine'] for entry in saved_rows if entry['sample_key'] == key])),
                     'mean_pair_total_variation': float(np.mean([entry['total_variation'] for entry in saved_rows if entry['sample_key'] == key])),
                     'plot_relative': next(entry['plot_relative'] for entry in saved_rows if entry['sample_key'] == key)}
                    for key in keys]
        worst = sorted(per_user, key=lambda entry: (entry['mean_pair_cosine'], entry['sample_key']))[:3]
        saved_worst = row['least_similar_three_under_declared_mean_cosine'][path]
        assert saved_worst == worst
    assert row['similarity_acceptance_threshold'] is None and row['scientific_invariance_claim'] is None
    return {'job_id': row['job_id'], 'users': 32, 'NPZ': 4, 'arrays_read': array_count,
            'raw_parent_indices_independently_reconstructed': parent_count,
            'descriptive_pairs_independently_recomputed': pair_count,
            'maximum_padding_mass': maximum_padding, 'maximum_raw_row_sum_error': maximum_row_error,
            'raw_inputs': checked}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--statistics', required=True, type=Path)
    p.add_argument('--jobs-root', required=True, type=Path)
    p.add_argument('--report', required=True, type=Path)
    args = p.parse_args()
    try:
        statistics = json.loads(args.statistics.read_bytes())
        assert statistics['status'] == 'DESCRIPTIVE_STATISTICS_COMPLETE'
        expected = {f'gs-20261017-{domain}-b-{condition}-attention-v1'
                    for domain in ('toys', 'sports') for condition in ('normal', 'shuffle3', 'shuffle4')}
        assert {row['job_id'] for row in statistics['jobs']} == expected and len(statistics['jobs']) == 6
        outputs = [audit_job(row, args.jobs_root / row['job_id'] / 'retrieved') for row in statistics['jobs']]
        report = {'status': 'FINAL_RAW_ATTENTION_STATISTICS_CROSSCHECK_PASS', 'reviewer_task': '/root/delivery_audit',
                  'reviewed_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'statistics': identity(args.statistics),
                  'conditions': outputs, 'NPZ_count': sum(row['NPZ'] for row in outputs),
                  'all_pairs_checked': sum(row['descriptive_pairs_independently_recomputed'] for row in outputs),
                  'equations': 'Original approved Cosine/Pearson on visible raw values; TV/JS only probability-normalize statistic inputs; JS natural logarithm. No new similarity threshold.',
                  'new_GPU_or_model_calls': 0, 'original_input_writes': 0,
                  'scope_limit': 'Independent direct raw-parent/head-mean/mask/statistical crosscheck. Full PNG pixel and lifecycle/shuffle/parameter checks remain referenced to prior real non-author audits.'}
    except Exception as error:
        report = {'status': 'FAIL', 'reviewer_task': '/root/delivery_audit', 'exception': type(error).__name__, 'reason': str(error)}
    report['program'] = identity(Path(__file__).resolve())
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({key: report.get(key) for key in ('status', 'NPZ_count', 'all_pairs_checked', 'reason')}))
    raise SystemExit(0 if report['status'] == 'FINAL_RAW_ATTENTION_STATISTICS_CROSSCHECK_PASS' else 1)


if __name__ == '__main__':
    main()
