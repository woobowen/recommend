"""Independently verify existing descriptive statistics; read raw NPZ only."""
import argparse
import datetime
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np

REPO = Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
SUPPORT = Path('/home/addaswsw/recommend-support/grid-formal-20261015')
sys.path.insert(0, str(REPO))
from experiments.beauty_extensions.attention.records import load_bundle


def identity(path):
    data = path.read_bytes()
    return {'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def close(first, second):
    if first is None or second is None:
        assert first is second
    else:
        assert math.isfinite(first) and math.isfinite(second)
        assert abs(first-second) < 1e-12, (first, second)


def pair_values(x, y):
    cosine = float(np.dot(x, y)/math.sqrt(np.dot(x, x)*np.dot(y, y)))
    pearson = None if np.var(x) == 0 or np.var(y) == 0 else float(np.corrcoef(x, y)[0, 1])
    p, q = x/x.sum(), y/y.sum()
    entropy = lambda v: -math.fsum(float(t)*math.log(float(t)) for t in v if t > 0)
    js = entropy((p+q)/2)-(entropy(p)+entropy(q))/2
    return {'cosine': cosine, 'pearson': pearson,
            'total_variation': math.fsum(abs(float(a)-float(b)) for a, b in zip(p, q))/2,
            'jensen_shannon_nats': max(0.0, js)}


def audit(path):
    report = json.loads(path.read_bytes())
    assert report['status'] == 'DESCRIPTIVE_STATISTICS_COMPLETE' and report['formal_source_only']
    for key in ('script', 'plan'):
        row = report['invocation'][key]
        assert identity(Path(row['path'])) == row
    plan = json.loads(Path(report['invocation']['plan']['path']).read_bytes())
    assert set(plan['jobs']) == {row['job_id'] for row in report['jobs']}
    checks = []
    for job in report['jobs']:
        directory = Path(plan['jobs'][job['job_id']])
        for source in job['input_files']:
            assert identity(Path(source['path'])) == source
        assert len(job['input_files']) == 8
        expectations = {}
        keys = []
        masked_max = raw_sum_error = 0.0
        for record in sorted((directory/'output').glob('batch-*/bundle/record.json')):
            bundle = load_bundle(record)
            assert not bundle['metadata']['engineering_only']
            capture_keys = bundle['capture']['sample_keys']
            assert bundle['capture']['selected_indices'] == list(range(8))
            keys.extend(capture_keys)
            masks = np.asarray(bundle['encoder_mask'], dtype=bool)
            for name in ('final_top1_cross', 'local_first_cross'):
                matrix = np.asarray(bundle['paths'][name], dtype=np.float64)
                masked_max = max(masked_max, float((matrix*(~masks[:, None])).sum(axis=-1).max()))
                raw_sum_error = max(raw_sum_error, float(np.abs(matrix.sum(axis=-1)-1).max()))
                for sample, key in enumerate(capture_keys):
                    visible = matrix[sample][:, masks[sample]]
                    assert visible.shape[0] == 4 and np.isfinite(visible).all() and (visible >= 0).all()
                    for first, second in itertools.combinations(range(4), 2):
                        expectations[(key, name, first+1, second+1)] = {
                            'visible_encoder_positions': int(masks[sample].sum()),
                            'all_encoder_positions': int(masks.shape[-1]),
                            'plot_relative': str(record.parent.parent.relative_to(directory)/'figures'/f'sample_{sample:03d}_cross_attention.png'),
                            **pair_values(visible[first], visible[second])}
        assert len(keys) == len(set(keys)) == 32 and len(expectations) == 384
        assert job['samples'] == 32 and job['cohort_keys_exact']
        assert job['similarity_acceptance_threshold'] is None and job['scientific_invariance_claim'] is None
        actual_keys = set()
        for row in job['per_sample_query_pairs']:
            key = (row['sample_key'], row['path'], *row['query_pair'])
            assert key not in actual_keys
            actual_keys.add(key)
            expected = expectations[key]
            for name in ('visible_encoder_positions', 'all_encoder_positions', 'plot_relative'):
                assert row[name] == expected[name]
            for name in ('cosine', 'pearson', 'total_variation', 'jensen_shannon_nats'):
                close(row[name], expected[name])
        assert actual_keys == set(expectations)
        close(job['maximum_masked_mass'], masked_max)
        close(job['maximum_raw_row_sum_error'], raw_sum_error)
        for name in ('final_top1_cross', 'local_first_cross'):
            pairs = [row for row in job['per_sample_query_pairs'] if row['path'] == name]
            assert len(pairs) == 192
            for metric in ('cosine', 'pearson', 'total_variation', 'jensen_shannon_nats'):
                values = sorted(row[metric] for row in pairs if row[metric] is not None)
                if not values:
                    assert metric not in job['summary'][name]
                    continue
                stats = job['summary'][name][metric]
                assert stats['n'] == len(values)
                close(stats['min'], values[0]); close(stats['max'], values[-1])
                close(stats['mean'], math.fsum(values)/len(values))
                close(stats['median'], (values[(len(values)-1)//2]+values[len(values)//2])/2)
            per_user = []
            for key in keys:
                six = [row for row in pairs if row['sample_key'] == key]
                assert len(six) == 6
                # Exact declared NumPy mean for deterministic tie-breaking;
                # every constituent was independently checked above.
                per_user.append({'sample_key': key, 'mean_pair_cosine': float(np.mean([r['cosine'] for r in six])),
                    'mean_pair_total_variation': float(np.mean([r['total_variation'] for r in six])),
                    'plot_relative': six[0]['plot_relative']})
            assert sorted(per_user, key=lambda r: (r['mean_pair_cosine'], r['sample_key']))[:3] == job['least_similar_three_under_declared_mean_cosine'][name]
        checks.append({'job_id': job['job_id'], 'all_32_users_and_384_pair_path_rows_independently_checked': True,
            'main_and_secondary_192_pair_summaries_verified': True, 'least_three_per_path_verified': True,
            'maximum_padding_mass': masked_max, 'maximum_raw_row_sum_error': raw_sum_error})
    receipt = {'status': 'INDEPENDENT_ATTENTION_DESCRIPTIVE_STATISTICS_PASS', 'reviewer_task': '/root/implementation_review',
        'reviewed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'report': identity(path),
        'verification_program': identity(Path(__file__)), 'actual_statistics_program': report['invocation']['script'],
        'plan': report['invocation']['plan'], 'jobs': checks,
        'method': 'Independent raw NPZ visible-weight cosine and np.corrcoef, entropy-form JS(nats), scalar-summed TV; exhaustive all pair rows, summaries and declared least-three ranking. Numeric agreement tolerance 1e-12 is a computation check, not a similarity acceptance threshold.',
        'plot_scope': 'Only saved raw arrays read; no row-normalized heatmaps, file changes, or unaligned cross-condition comparisons.',
        'scientific_claim': None, 'new_GPU_or_model_calls': 0, 'source_Git_or_original_evidence_writes': 0}
    destination = SUPPORT/'attention-results-review/attention-statistics-runtime-review.json'
    assert not destination.exists()
    destination.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(identity(destination)|{'status': receipt['status'], 'jobs': len(checks)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('report', type=Path)
    audit(p.parse_args().report)
