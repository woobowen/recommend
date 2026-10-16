"""CPU descriptive comparisons of all six query pairs in each fixed32 capture.

This never compares raw columns across shuffled conditions. Probability-only
normalization for TV/JS does not alter stored matrices or plotted heatmaps.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
sys.path.insert(0, str(REPO))
from experiments.beauty_extensions.attention.records import load_bundle


def identity(path):
    return {'path': str(path), 'bytes': path.stat().st_size,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def summarize(values):
    values = np.asarray([v for v in values if v is not None], dtype=np.float64)
    assert values.size and np.isfinite(values).all()
    return {'n': int(values.size), 'min': float(values.min()),
        'median': float(np.median(values)), 'mean': float(values.mean()),
        'max': float(values.max())}


def pair_metrics(x, y):
    cosine = float(x@y/(np.linalg.norm(x)*np.linalg.norm(y)))
    xc, yc = x-x.mean(), y-y.mean()
    denominator = np.linalg.norm(xc)*np.linalg.norm(yc)
    correlation = float(xc@yc/denominator) if denominator > 0 else None
    p, q = x/x.sum(), y/y.sum()
    middle = (p+q)/2
    def kl(a):
        positive = a > 0
        return float(np.sum(a[positive]*np.log(a[positive]/middle[positive])))
    return {'cosine': cosine, 'pearson': correlation,
        'total_variation': float(np.abs(p-q).sum()/2),
        'jensen_shannon_nats': max(0.0, (kl(p)+kl(q))/2)}


def analyze(job, directory):
    result = json.loads((directory/'output/RESULT.json').read_text())
    assert result['formal'] and result['status'] == 'COMPLETE' and result['samples'] == 32
    expected = [key for batch in result['batches'] for key in batch['sample_keys']]
    rows, keys, inputs = [], [], []
    maximum_masked = maximum_sum_error = 0.0
    records = sorted((directory/'output').glob('batch-*/bundle/record.json'))
    assert len(records) == 4
    for record in records:
        bundle = load_bundle(record)
        assert bundle['metadata']['engineering_only'] is False
        inputs.extend([identity(record), identity(record.parent/'arrays.npz')])
        sample_keys = bundle['capture']['sample_keys']
        selected = bundle['capture']['selected_indices']
        selected_keys = [sample_keys[index] for index in selected]
        assert len(selected_keys) == 8
        keys.extend(selected_keys)
        mask = np.asarray(bundle['encoder_mask'], dtype=bool)
        for path in ('final_top1_cross', 'local_first_cross'):
            matrix = np.asarray(bundle['paths'][path], dtype=np.float64)
            assert matrix.shape[:2] == (8, 4) and np.isfinite(matrix).all() and (matrix >= 0).all()
            maximum_masked = max(maximum_masked, float((matrix*(~mask[:, None, :])).sum(-1).max()))
            maximum_sum_error = max(maximum_sum_error, float(np.abs(matrix.sum(-1)-1).max()))
            for sample, key in enumerate(selected_keys):
                visible = matrix[sample][:, mask[sample]]
                assert visible.shape[0] == 4 and (visible.sum(-1) > 0).all()
                for first, second in itertools.combinations(range(4), 2):
                    rows.append({'sample_key': key, 'path': path,
                        'query_pair': [first+1, second+1],
                        'visible_encoder_positions': int(mask[sample].sum()),
                        'all_encoder_positions': int(mask.shape[-1]),
                        'plot_relative': str(record.parent.parent.relative_to(directory)/'figures'/f'sample_{sample:03d}_cross_attention.png'),
                        **pair_metrics(visible[first], visible[second])})
    assert keys == expected and len(set(keys)) == 32 and len(rows) == 32*6*2
    summaries = {}
    least = {}
    for path in ('final_top1_cross', 'local_first_cross'):
        values = [r for r in rows if r['path'] == path]
        summaries[path] = {metric: summarize([r[metric] for r in values])
            for metric in ('cosine', 'pearson', 'total_variation', 'jensen_shannon_nats')
            if any(r[metric] is not None for r in values)}
        per_user = []
        for key in keys:
            pairs = [r for r in values if r['sample_key'] == key]
            per_user.append({'sample_key': key,
                'mean_pair_cosine': float(np.mean([r['cosine'] for r in pairs])),
                'mean_pair_total_variation': float(np.mean([r['total_variation'] for r in pairs])),
                'plot_relative': pairs[0]['plot_relative']})
        least[path] = sorted(per_user, key=lambda r: (r['mean_pair_cosine'], r['sample_key']))[:3]
    return {'job_id': job, 'samples': 32, 'cohort_keys_exact': True,
        'input_files': inputs, 'summary': summaries, 'per_sample_query_pairs': rows,
        'least_similar_three_under_declared_mean_cosine': least,
        'maximum_masked_mass': maximum_masked, 'maximum_raw_row_sum_error': maximum_sum_error,
        'similarity_acceptance_threshold': None, 'scientific_invariance_claim': None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    plan = json.loads(a.plan.read_text())
    result = {'status': 'DESCRIPTIVE_STATISTICS_COMPLETE', 'formal_source_only': True,
        'method': {'matrix': 'last decoder layer, head mean; raw saved values',
            'main_path': 'actual generation final Top-1 ancestry',
            'secondary_path': 'local first parent independently per step',
            'pairs': 'all six unordered pairs of four queries, for all 32 fixed users',
            'positions': 'all encoder positions with actual encoder_mask == 1, including visible special tokens',
            'cosine_and_pearson': 'raw visible weights; Pearson is null for a constant row',
            'TV_and_JS': 'visible rows divided by their visible sum only to form probability distributions; TV=0.5*L1, JS=0.5*(KL(p||m)+KL(q||m)), natural log',
            'figure_preprocessing': 'none; no row rescaling, clipping or image alteration',
            'worst_case_rule': 'lowest per-user mean of six pair cosines; tie by sample key; all users retained',
            'cross_condition_matrix_comparison': 'not performed; no unaligned Shuffle-column comparison',
            'similarity_threshold_or_invariance_proof': None},
        'jobs': [analyze(job, Path(directory)) for job, directory in plan['jobs'].items()],
        'invocation': {'argv': sys.argv, 'interpreter': sys.executable,
            'script': identity(Path(__file__)), 'plan': identity(a.plan), 'GPU_or_model_calls': 0}}
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(a.output), 'jobs': len(result['jobs']), 'samples_each': 32,
        'scientific_invariance_claim': None}))


if __name__ == '__main__':
    main()
