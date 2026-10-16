"""CPU rendering of every finite original CSV point; no model or GPU use."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--domain', choices=('toys', 'sports'), required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    state = json.loads((a.directory/'status.json').read_text())
    result = json.loads((a.directory/'output/RESULT.json').read_text())
    assert state['status'] == 'SUCCEEDED' and state['exit_code'] == 0 and state['cleanup']['verified']
    assert result['formal'] and result['training_complete'] and result['test_complete']
    csv_files = list((a.directory/'output/csv').rglob('metrics.csv'))
    assert len(csv_files) == 1
    source = csv_files[0]
    names = ('train/loss_step', 'train/loss_epoch', 'val/loss',
        'val/recall@5', 'val/recall@10', 'val/ndcg@5', 'val/ndcg@10')
    points = {name: [] for name in names}
    count = 0
    with source.open(newline='') as stream:
        for row in csv.DictReader(stream):
            count += 1
            for name in names:
                if row.get(name) not in ('', None):
                    step, value = float(row['step']), float(row[name])
                    assert math.isfinite(step) and math.isfinite(value)
                    points[name].append([step, value])
    assert len(points['val/recall@5']) == result['completed_full_validations']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    a.output.mkdir(parents=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    for name in names:
        if points[name]:
            x, y = zip(*points[name])
            axes[0 if name in names[:3] else 1].plot(x, y, marker='.', label=name)
    axes[0].set_ylabel('Original logged loss')
    axes[1].set_ylabel('Validation retrieval metric')
    axes[1].set_xlabel('Original Lightning CSV step (unchanged)')
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.25)
    fig.suptitle(a.domain.title()+' formal MLP: all logged points')
    png = a.output/'training_validation_curves.png'
    fig.savefig(png, dpi=160)
    plt.close(fig)
    with Image.open(png) as saved:
        saved.load()
        dimensions = list(saved.size)
    value = {'status': 'PASS', 'formal_source': True,
        'source_CSV': {'path': str(source), 'bytes': source.stat().st_size, 'sha256': sha(source)},
        'job_id': state['job_id'], 'training_RESULT_sha256': sha(a.directory/'output/RESULT.json'),
        'best_checkpoint': result['best_checkpoint'], 'CSV_rows': count,
        'points': points, 'smoothing_or_selection': None,
        'x_axis': 'raw Lightning CSV step; no correction of the original logging index',
        'PNG': {'path': str(png), 'bytes': png.stat().st_size, 'sha256': sha(png),
            'dimensions': dimensions, 'actually_reopened': True},
        'invocation': {'argv': sys.argv, 'interpreter': sys.executable,
            'script_sha256': sha(Path(__file__)), 'model_or_GPU_calls': 0}}
    (a.output/'plot_receipt.json').write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'domain': a.domain, 'validation_points': result['completed_full_validations'],
        'output': str(a.output), 'PNG_reopened': True, 'model_or_GPU_calls': 0}))


if __name__ == '__main__':
    main()
