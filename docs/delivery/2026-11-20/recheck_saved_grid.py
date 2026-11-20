"""Check 24 GRID rankings and three MLP rankings from saved outputs."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path


def read_json(path):
    return json.loads(path.read_bytes())


def check(rows_path, expected):
    """Apply original top-k definitions to recorded SIDs, scores and labels."""
    users = set()
    totals = collections.Counter()
    digest = hashlib.sha256()
    ties = 0
    duplicate_count = 0
    with rows_path.open('rb') as stream:
        for line in stream:
            digest.update(line)
            row = json.loads(line)
            candidates = list(map(tuple, row['sids']))
            scores = row['scores']
            target = tuple(row['target'])
            key = row['sample_key']
            assert key not in users
            assert len(scores) == len(candidates) == 10 and all(map(math.isfinite, scores))
            duplicate_count += 10 - len(set(candidates))
            users.add(key)
            relevant_count = candidates.count(target)
            relevant = [candidates[i] == target for i in sorted(range(10), key=scores.__getitem__, reverse=True)]
            tied = any(scores[i] == scores[j] and (candidates[i] == target) != (candidates[j] == target)
                       for i in range(10) for j in range(i))
            ties += tied
            for k in (5, 10):
                totals[f'recall@{k}'] += sum(relevant[:k]) / max(1, min(relevant_count, k))
                discounted_hits = sum(1 / math.log2(rank + 2) for rank, hit in enumerate(relevant[:k]) if hit)
                ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, relevant_count)))
                totals[f'ndcg@{k}'] += discounted_hits / (ideal or 1)
    assert len(users) == expected['samples']==expected['unique_users']
    assert digest.hexdigest() == expected['prediction_sha256']
    assert duplicate_count == expected['duplicate_candidate_count']
    assert not ties, f'Tied scores straddle relevant and irrelevant candidates: {ties}'
    metrics = {name: value / len(users) for name, value in totals.items()}
    differences = {name: abs(value - expected['metrics'][name]) for name, value in metrics.items()}
    assert max(differences.values()) <= 1e-7, (rows_path, differences)
    return {'samples': len(users), 'prediction_sha256': digest.hexdigest(), 'duplicate_candidates': duplicate_count,
            'score_ties_mixed_relevance': ties, 'metrics': metrics, 'diff_abs': differences}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--raw', type=Path, required=True, help='Directory with one original predictions.jsonl per job')
    args = parser.parse_args()
    manifests = args.repo / 'manifests'
    cross = read_json(manifests / 'grid-a2-fd-recovery-20261018/final-results/cross-domain/cross-domain-evidence-index-with-a2.json')
    cases = {}
    for route in cross['routes']:
        for condition in route['conditions']:
            cases[condition['job_id']] = dict(condition, metrics=condition['metrics_original'])
    for folder, job in (
        ('beauty-extensions', 'beauty-mlp-formal-v1'),
        ('grid-formal-20261015', 'gf-20261015-toys-mlp-formal-v1'),
        ('grid-formal-20261015', 'gf-20261015-sports-mlp-formal-v1'),
    ):
        cases[job] = read_json(manifests / folder / 'runs' / job / 'output/test/recomputed_metrics.json') if folder != 'beauty-extensions' else read_json(args.raw / job / 'recomputed_metrics.json')
    assert len(cases) == 27
    results = {job: check(args.raw / job / 'predictions.jsonl', case) for job, case in sorted(cases.items())}
    print(json.dumps({'cases': len(results), 'rows': sum(r['samples'] for r in results.values()),
                      'max_absolute_difference': max(max(r['diff_abs'].values()) for r in results.values()),
                      'scope': 'Four raw ranking metrics and unique user count. No source-data reload, new inference or training.',
                      'results': results}, indent=2))


if __name__ == '__main__':
    main()
