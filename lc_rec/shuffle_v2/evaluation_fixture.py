"""CPU regression of single-rank evaluation using an isolated Normal topology fixture."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

from lc_rec.shuffle_v2 import core, evaluate as ev
from lc_rec.shuffle_v2.replay import CONFIG, dump, frozen


def main():
    import numpy as np
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    config = json.loads(CONFIG.read_text())
    base = config['baseline']
    src = Path(base['normal_test_dir'])
    a.output.mkdir(parents=True, exist_ok=False)
    test = a.output / 'normal-topology-fixture'
    test.mkdir()
    frozen_manifests = ev.normal_predictions(src, config)
    assignment = []
    prediction_path = test / 'predictions.rank0.jsonl'
    with ((src / 'predictions.rank0.jsonl').open() as left, (src / 'predictions.rank1.jsonl').open() as right,
          prediction_path.open('x') as merged):
        for pair in zip(left, right, strict=True):
            for rank, raw in enumerate(pair):
                row = json.loads(raw)
                assert row['rank'] == rank
                assignment.append(row['row_idx'])
                row['rank'] = 0
                merged.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    assert set(assignment) == set(range(config['users'])) and len(assignment) == config['users']
    report = json.loads((src / 'rank0.json').read_text())
    report.update(world_size=1, assigned_users=len(assignment), completed_users=len(assignment),
                  assignment_sha256=core.canonical_sha256(assignment), scope='CPU_SINGLE_TOPOLOGY_FIXTURE')
    dump(test / 'rank0.json', report)
    _, catalog, users, _ = frozen.catalog_and_users(base['data-root'])
    aggregate = frozen.aggregate_predictions([prediction_path], users, catalog, frozen.load_evaluator(base['source-root']),
                                             run_commit=base['run-commit'])
    aggregate['RUN_COMMIT'] = base['run-commit']
    dump(test / 'metrics.json', aggregate)
    original = ev.module('fixture_original', ev.single_evaluator(a.output, ev.LC / 'letter/review/formal_det_20261112/independent_metrics.py', 'original'))
    stage_a = a.output / 'independent_original.json'
    dump(stage_a, original.audit(Path(base['data-root']), test))
    primary = ev.module('fixture_primary', ev.single_evaluator(a.output, ev.LC / 'a1_qwen3_06b/scripts/offline_metrics.py', 'primary'))
    primary.RUN = base['run-commit']
    digest = frozen.sha(prediction_path)
    size = prediction_path.stat().st_size
    primary.PRED = [(size, digest)]
    official = ev.LC / 'letter/review/final_audit_20261113/external/official_cce.py'
    result = a.output / 'evaluation'
    primary.evaluate(SimpleNamespace(output=result, test_dir=test, config=ev.LC / 'letter/config/a0_evaluation_v1.json',
                                     official_cce=official, stage_a_report=stage_a, data_root=Path(base['data-root']),
                                     evaluation_commit=json.loads(Path('RUN_VERSION.json').read_text())['RUN_COMMIT']))
    manifest = a.output / 'prediction_manifest.json'
    dump(manifest, {'schema': 'a1_frozen_prediction_manifest_v1', 'gpu_run_commit': base['run-commit'],
                    'predictions': [{'rank': 0, 'path': str(prediction_path), 'bytes': size, 'sha256': digest}]})
    extended = ev.module('fixture_extended', ev.single_evaluator(a.output, ev.LC / 'a1_qwen3_06b/review/independent_extended.py', 'extended'))
    independent = extended.aggregate(Path(base['data-root']), test, stage_a, manifest, result / 'per_user.csv')
    dump(a.output / 'independent_extended.json', independent)
    with (np.load(result / 'per_user_metrics.npz', allow_pickle=False) as new,
          np.load(Path(config['normal_primary']) / 'per_user_metrics.npz', allow_pickle=False) as old):
        assert set(new.files) == set(old.files) and all(np.array_equal(new[k], old[k]) for k in new.files)
    values = json.loads((result / 'metrics.json').read_text())['metrics']
    assert max(abs(v - independent['metrics'][k]) for k, v in values.items()) <= 1e-13
    dump(a.output / 'summary.json', {'status': 'PASS_SINGLE_SHARD_EVALUATION_FIXTURE', 'gpu_used': False,
                                     'scope': 'Re-shard archived Normal predictions for CPU topology regression',
                                     'users': len(assignment), 'metrics': values, 'frozen_normal': frozen_manifests,
                                     'rank0_fixture_sha256': digest,
                                     'per_user_exact': True, 'source_hashes': {p.name: frozen.sha(p) for p in a.output.glob('evaluator-*.py')}})


if __name__ == '__main__':
    main()
