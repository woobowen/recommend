#!/usr/bin/env python3
"""Run the frozen LETTER evaluator on existing synthetic shards, never formal receipts."""
import argparse
import json
import math
import os
from pathlib import Path
import time
import formal_evaluate as evaluation


def main():
    p = argparse.ArgumentParser()
    for name in ('fixture-root', 'data-root', 'source-root', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    label = 'SYNTHETIC_CPU_INTEGRATION_ONLY'
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only environment required')
    if label not in str(a.fixture_root) or label not in str(a.output):
        raise ValueError('synthetic fixture and output identity required')
    if a.output.exists():
        raise FileExistsError(a.output)
    notice = json.loads((a.fixture_root / 'SYNTHETIC_NOTICE.json').read_text())
    if notice['designation'] != label or notice['real_model_predictions'] or notice['formal_job_receipt']:
        raise ValueError('fixture is not isolated synthetic output')
    summary = json.loads((a.fixture_root / 'summary.json').read_text())
    if summary['designation'] != label or summary['users'] != 22363:
        raise ValueError('unexpected fixture manifest')
    _, catalog, users, assets = evaluation.catalog_and_users(a.data_root)
    evaluator = evaluation.load_evaluator(evaluation.author_source(a.source_root))
    results = {}
    start = time.monotonic()
    for condition in ('Normal', 'S3', 'S4'):
        root = a.fixture_root / condition
        expected = summary['conditions'][condition]
        paths = [root / 'synthetic_predictions' / f'predictions.rank{rank}.jsonl' for rank in (0, 1)]
        report = evaluation.aggregate_predictions(paths, users, catalog, evaluator, require_rank_done=False)
        for actual, recorded in zip(report['prediction_files'], expected['prediction_files'], strict=True):
            if actual['sha256'] != recorded['sha256'] or actual['bytes'] != recorded['bytes']:
                raise ValueError('prediction identity differs from saved manifest')
        prior = json.loads((root / 'synthetic_stage_a.json').read_text())
        for key, value in report['metrics'].items():
            if not math.isclose(value, prior['metrics'][key], abs_tol=1e-13, rel_tol=1e-13):
                raise ValueError('original LETTER metric differs from synthetic fixture: ' + key)
        results[condition] = report
    a.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {'designation': label, 'formal_job_receipt': False, 'real_model_predictions': False,
               'author_evaluator_sha256': evaluation.sha(evaluation.author_source(a.source_root) / 'evaluate.py'),
               'asset_identities': assets, 'results': results, 'elapsed_seconds': time.monotonic() - start}
    with a.output.open('x') as f:
        json.dump(payload, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'designation': label, 'conditions': list(results), 'status': 'PASS',
                      'elapsed_seconds': payload['elapsed_seconds']}))


if __name__ == '__main__':
    main()
