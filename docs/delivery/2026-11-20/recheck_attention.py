"""Compare saved GRID fixed32 rows and A-route summaries to recomputed values."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--raw', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    from experiments.launch_20261017.statistics import analyze

    manifests = args.repo / 'manifests'
    pair_fields = ('cosine', 'pearson', 'total_variation', 'jensen_shannon_nats')
    summary_files = [
        manifests / family / 'attention-statistics/attention-descriptive-statistics.json'
        for family in ('grid-formal-20261015', 'grid-shuffle-20261017')
    ]
    observed = {}
    max_delta = 0.0
    for path in summary_files:
        for expected in json.loads(path.read_bytes())['jobs']:
            job = expected['job_id']
            actual = analyze(job, args.raw / job)
            for row, old in zip(actual['per_sample_query_pairs'], expected['per_sample_query_pairs'], strict=True):
                for field in ('sample_key', 'path', 'query_pair', 'visible_encoder_positions'):
                    assert row[field] == old[field], (job, field)
                for field in pair_fields:
                    if old[field] is None:
                        assert row[field] is None
                    else:
                        difference = abs(old[field] - row[field])
                        assert difference <= 1e-12, (job, field, difference)
                        max_delta = max(max_delta, difference)
            observed[job] = {'samples': actual['samples'], 'rows': len(actual['per_sample_query_pairs']), 'compared': 'all_pair_values'}

    a_root = manifests / 'grid-a2-fd-recovery-20261018/final-results'
    index = json.loads((a_root / 'cross-domain/cross-domain-evidence-index-with-a2.json').read_bytes())
    job_by_route = {(route['route_id'], item['condition']): item['job_id'].replace('-eval-', '-attention-')
                    for route in index['routes'][4:] for item in route['conditions']}
    with (a_root / 'attention-means.csv').open() as source:
        summary = list(csv.DictReader(source))
    for key, job in job_by_route.items():
        actual = analyze(job, args.raw / job)
        rows = [r for r in summary if (r['route'], r['condition']) == key]
        assert len(rows) == 2, (key, rows)
        for row in rows:
            for field in pair_fields:
                recorded = float(row[field])
                metric = actual['summary'][row['path']][field]['mean']
                assert math.isclose(recorded, metric, abs_tol=1e-8, rel_tol=0), (job, field, recorded, metric)
        observed[job] = {'samples': actual['samples'], 'rows': len(actual['per_sample_query_pairs']), 'compared': 'two path means for four metrics'}
    print(json.dumps({'jobs': observed, 'jobs_count': len(observed), 'rows_calculated': sum(r['rows'] for r in observed.values()), 'max_delta_per_pair': max_delta}, indent=2))


if __name__ == '__main__':
    main()
