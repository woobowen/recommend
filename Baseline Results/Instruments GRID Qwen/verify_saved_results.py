"""Check saved originals and reproduce the four reported metrics; no model execution."""
from pathlib import Path
import csv, hashlib, json, math

HERE = Path(__file__).resolve().parent
KEYS = ['test/recall@5', 'test/recall@10', 'test/ndcg@5', 'test/ndcg@10']


def main():
    run = HERE / 'run'
    result = json.loads((run / 'result.json').read_text(encoding='utf8'))
    assert result == json.loads((run / 'formal/stage_result.json').read_text(encoding='utf8'))
    assert json.loads((run / 'status.json').read_text())['phase'] == 'complete'
    assert json.loads((run / 'formal/exit.json').read_text())['returncode'] == 0
    assert result['global_step'] == 7100 and '006100.ckpt' in result['checkpoint']
    audit = json.loads((run / 'data_audit.json').read_text())
    assert audit['users'] == 27530 and audit['items'] == 10611
    assert audit['splits']['testing']['users'] == 27530
    rows = list(csv.DictReader((run / 'formal/csv/version_0/metrics.csv').open()))
    vals = [r for r in rows if r.get('val/recall@5')]
    best = max(vals, key=lambda r: float(r['val/recall@5']))
    assert int(best['step']) + 1 == 6100
    metrics = result['test_metrics'][0]
    for k in KEYS:
        assert math.isfinite(metrics[k]) and 0 <= metrics[k] <= 1
        assert any(r.get(k) and float(r[k]) == metrics[k] for r in rows), k
    files = json.loads((run / 'source_manifest.json').read_text(encoding='utf8'))
    for rel, expected in files.items():
        assert hashlib.sha256((run / rel).read_bytes()).hexdigest() == expected, rel
    with (HERE / 'metrics.csv').open('w', newline='', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'upstream', 'downstream', 'condition', 'seed', 'test_users',
                         'recall@5', 'recall@10', 'ndcg@5', 'ndcg@10'])
        writer.writerow(['instruments', 'qwen_rq_kmeans', 'grid_tiger', 'normal', 42, 27530,
                         *[metrics[k] for k in KEYS]])
    table = '| 数据集 | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |\n|---|---:|---:|---:|---:|\n'
    table += '| Instruments | ' + ' | '.join(f'{metrics[k]:.6f}' for k in KEYS) + ' |\n'
    (HERE / 'metrics.md').write_text(table, encoding='utf8')
    print(json.dumps({'saved_result_consistency': True, 'source_files': len(files),
                      'metrics': {k: metrics[k] for k in KEYS}, 'test_users_from_data_audit': 27530,
                      'best_checkpoint_step': 6100, 'validation_rows': len(vals),
                      'scope': 'Saved aggregate output and source checks, not candidate-level recomputation or model reevaluation'}))


if __name__ == '__main__':
    main()
