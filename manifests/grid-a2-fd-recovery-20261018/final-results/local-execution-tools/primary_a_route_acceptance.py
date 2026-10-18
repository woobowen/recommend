#!/usr/bin/env python3
"""Primary arithmetic acceptance of original full A-route predictions.

No generation, model loading, candidate filtering, or prediction rewriting.
Independent reviewers perform the broader structural/attention audits separately.
"""
from collections import Counter
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent

def read(path):
    return json.loads(path.read_text())

def lines(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]

def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()

def accept_route(name, root, job_prefix, job_suffix, legal, count):
    metadata = root/(job_prefix+'metadata'+job_suffix)/'output'
    target_index = metadata/'test_target_index.jsonl'
    expected = Counter((r['sample_key'], tuple(r['target'])) for r in lines(target_index))
    assert len(expected) == sum(expected.values()) == count
    result = dict(route=name, samples=count, conditions={}, paired={})
    original_predictions = {}
    for condition, n in [('Normal', 0), ('Shuffle3', 3), ('Shuffle4', 4)]:
        job = root/(job_prefix+'eval-'+str(n)+job_suffix)
        status = read(job/'status.json')
        cleanup = read(job/'cleanup.json')
        assert status['status'] == 'SUCCEEDED' and status['exit_code'] == 0 and cleanup['verified']
        path = job/'output/test/predictions.jsonl'
        rows = lines(path)
        assert Counter((r['sample_key'], tuple(r['target'])) for r in rows) == expected
        assert len(rows) == len({r['sample_key'] for r in rows}) == count
        target_hash = sha(path)
        match = [r for r in status['artifacts'] if r['path'] == 'output/test/predictions.jsonl']
        assert len(match) == 1 and match[0]['sha256'] == target_hash and match[0]['bytes'] == path.stat().st_size
        ranks, invalid, duplicates = [], 0, 0
        predictions = {}
        for row in rows:
            candidates = [tuple(s) for s in row['sids']]
            scores = np.asarray(row['scores'], dtype=np.float32)
            target = tuple(row['target'])
            assert len(candidates) == len(scores) == 10
            assert all(len(s) == 4 for s in candidates) and np.isfinite(scores).all()
            assert np.all(scores[:-1] >= scores[1:])
            target_ranks = [j+1 for j, sid in enumerate(candidates) if sid == target]
            rank = min(target_ranks, default=11)
            # Relevant score ties could make the native topk evaluator ambiguous.
            if target_ranks:
                target_score = scores[rank-1]
                assert np.count_nonzero(scores == target_score) == 1
            invalid += sum(s not in legal for s in candidates)
            duplicates += len(candidates)-len(set(candidates))
            ranks.append(rank)
            predictions[row['sample_key']] = (target, candidates, rank)
        metrics = {}
        for k in (5, 10):
            metrics['recall@'+str(k)] = sum(rank <= k for rank in ranks)/count
            metrics['ndcg@'+str(k)] = sum(1/math.log2(rank+1) if rank <= k else 0 for rank in ranks)/count
        original = read(job/'output/test/recomputed_metrics.json')
        delta = {key: abs(value-original['metrics'][key]) for key, value in metrics.items()}
        assert max(delta.values()) < 1e-7
        assert invalid == original['invalid_candidate_count'] and duplicates == original['duplicate_candidate_count']
        result['conditions'][condition] = dict(job_id=job.name, original_metrics=original['metrics'],
                    independent_float64_metrics=metrics, maximum_absolute_delta=max(delta.values()),
                    invalid_candidates=invalid, duplicate_candidates=duplicates,
                    raw_prediction_sha256=target_hash, full_target_multiset_exact=True)
        original_predictions[n] = predictions
    for n in (3, 4):
        counts = Counter()
        for key, (target, normal, normal_rank) in original_predictions[0].items():
            changed_target, shuffled, shuffled_rank = original_predictions[n][key]
            assert target == changed_target
            counts['top1_changed'] += normal[0] != shuffled[0]
            counts['top10_order_changed'] += normal != shuffled
            counts['top10_set_changed'] += set(normal) != set(shuffled)
            for k in (5, 10):
                counts[f'hit{k}_{int(normal_rank <= k)}to{int(shuffled_rank <= k)}'] += 1
            counts['rank_improved_miss11'] += shuffled_rank < normal_rank
            counts['rank_worsened_miss11'] += shuffled_rank > normal_rank
            counts['rank_unchanged_miss11'] += shuffled_rank == normal_rank
        result['paired']['Shuffle'+str(n)] = dict(samples=count, **dict(counts))
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a2-raw', type=Path)
    args = parser.parse_args()
    toys_legal = set(map(tuple, read(BASE/'reviews/toys/legal_sid_cpu_export.json')['item_axis_sids']))
    a1_legal = set(map(tuple, read(BASE/'reviews/sports_a1/sid_rows.json')))
    results = []
    for label, workflow in [('Toys-A-5300', 'toys5300'), ('Toys-A-5000', 'toys5000')]:
        results.append(accept_route(label, BASE/'reviews/toys/raw', 'ga-20261017-'+workflow+'-', '-v1', toys_legal, 19412))
    results.append(accept_route('Sports-A1', BASE/'reviews/sports_a1/raw', 'ga-20261017-sportsa1-', '-v1', a1_legal, 35598))
    if args.a2_raw:
        with np.load(BASE/'primary-quantization-acceptance/quantization-numpy-arrays.npz', allow_pickle=False) as archive:
            legal = set(map(tuple, archive['sid'].T.tolist()))
        results.append(accept_route('Sports-A2', args.a2_raw, 'ga-20261018-sportsa2-', '-fd-recovery-v1', legal, 35598))
    report = dict(status='PRIMARY_AVAILABLE_A_ROUTE_ARITHMETIC_ACCEPTANCE_PASS', author='/root',
                  checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(), routes=results,
                  a2_included=bool(args.a2_raw), final_full_task_review_pass=False,
                  scope='All original full predictions and paired metrics of listed routes; no model or attention inference',
                  candidate_policy='Original candidates retained; no filtering, completion, rescoring or reordering')
    destination = BASE/'primary-result-acceptance'
    destination.mkdir(exist_ok=True)
    suffix = 'all-four' if args.a2_raw else 'existing-three'
    path = destination/(suffix+'.json')
    assert not path.exists(), 'Preserve previous acceptance; use a new explicitly named artifact if inputs change.'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(path=str(path), routes=len(results), conditions=sum(len(r['conditions']) for r in results),
                         maximum_metric_delta=max(c['maximum_absolute_delta'] for r in results for c in r['conditions'].values()),
                         status=report['status']), ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
