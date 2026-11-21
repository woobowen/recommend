from pathlib import Path
import csv
import json

root = Path(__file__).resolve().parent
snapshot = json.loads((root / 'latest_three_dataset_metrics.json').read_text(encoding='utf-8'))
metrics = ['recall@5', 'recall@10', 'ndcg@5', 'ndcg@10']
papers = {
    'TIGER': {
        'beauty': [0.0454, 0.0648, 0.0321, 0.0384],
        'sports': [0.0264, 0.0400, 0.0181, 0.0225],
        'toys': [0.0521, 0.0712, 0.0371, 0.0432],
    },
    'GRID RK-Means': {
        'beauty': [0.0422, 0.0639, 0.0277, 0.0347],
        'sports': [0.0236, 0.0353, 0.0153, 0.0191],
        'toys': [0.0376, 0.0577, 0.0243, 0.0308],
    },
}
rows = []
for dataset in ['beauty', 'sports', 'toys']:
    run = snapshot['datasets'][dataset]
    assert run['pipeline_status']['state'] == 'completed'
    assert run['pipeline_status']['exit_code'] == 0
    test = run['result']['test_metrics'][0]
    for metric in metrics:
        assert test['test/' + metric] == run['latest_test']['test/' + metric]
    for paper, values in papers.items():
        for metric, reference in zip(metrics, values[dataset]):
            value = test['test/' + metric]
            rows.append({'dataset': dataset, 'metric': metric, 'test_value': value,
                         'paper': paper, 'paper_value': reference,
                         'relative_change_percent': (value / reference - 1) * 100})
with (root / '测试集与论文对比_20261011.csv').open('w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
lines = [f"核查时间：{snapshot['checked_at']}", '',
         '三个数据集均完成训练和最佳 checkpoint 测试，进程退出码均为 0。', '',
         '相对增减 =（本次测试指标 − 论文指标）÷ 论文指标 × 100%。使用未四舍五入的测试指标计算。', '',
         '| 数据集 | 指标 | 本次测试 | TIGER 原论文 | 增减 | GRID RK-Means | 增减 |',
         '|---|---|---:|---:|---:|---:|---:|']
for dataset in ['beauty', 'sports', 'toys']:
    for metric in metrics:
        selected = [r for r in rows if r['dataset'] == dataset and r['metric'] == metric]
        a, b = selected
        lines.append(f"| {dataset.title()} | {metric} | {a['test_value']:.5f} | {a['paper_value']:.5f} | {a['relative_change_percent']:+.2f}% | {b['paper_value']:.5f} | {b['relative_change_percent']:+.2f}% |")
lines += ['', 'TIGER：Table 1，https://arxiv.org/html/2305.05065',
          'GRID：Table 1 RK-Means，https://arxiv.org/html/2507.22224', '',
          '本次配置是 Qwen2.5-7B-Instruct + RQ-KMeans、单次运行，按验证 Recall@5 选最佳 checkpoint。',
          'TIGER 使用 Sentence-T5 + RQ-VAE；GRID 默认 Flan-T5-XL + RK-Means，报告五个种子的平均结果。',
          '不同文本模型、训练配置、checkpoint 选择和重复次数使这些数值对比不能视为同配置下的显著提升证明。', '',
          '| 数据集 | 测试完成时间（北京时间） | 测试 checkpoint | 最后训练步数 |',
          '|---|---|---|---:|']
for dataset in ['beauty', 'sports', 'toys']:
    run = snapshot['datasets'][dataset]
    lines.append(f"| {dataset.title()} | {run['progress']['time']} | {Path(run['result']['checkpoint']).name} | {run['result']['global_step']} |")
text = '\n'.join(lines) + '\n'
(root / '测试集与论文对比_20261011.md').write_text(text, encoding='utf-8')
print(text)
