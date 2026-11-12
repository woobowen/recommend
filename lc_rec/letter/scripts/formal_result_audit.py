#!/usr/bin/env python3
"""Read terminal A0 evidence; never load a model or start GPU computation."""
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys

from formal_train import verify_checkpoint
from formal_ledger import reconcile


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(data)
    return result.hexdigest()


def seconds(start, end):
    return (dt.datetime.fromisoformat(end) - dt.datetime.fromisoformat(start)).total_seconds()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('job', 'run-commit', 'output'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    output = Path(args.output)
    assert not output.exists(), 'preserve existing audit'
    assert args.job.startswith('letter-a0-20261112-')
    base = Path('/data/bwwu/lc-rec/a0-normal-20261111')
    job = Path('/home/bwwu/grid-audits/sid-extensions-p0/jobs') / args.job
    status = read(job / 'status.json')
    assert status['status'] == 'SUCCEEDED' and status['exit_code'] == 0
    assert status['cleanup']['verified'], 'must audit terminal, empty task group'
    pipeline = read(job / 'output/pipeline.json')
    assert pipeline['status'] == 'SUCCESS' and pipeline['run_commit'] == args.run_commit
    request = read(job / 'request.json')
    assert request['capture_version'] == args.run_commit
    assert request['argv'][request['argv'].index('--kind') + 1] == 'formal'
    assert not any(v in request['argv'] for v in
                   ['--diagnosis', '--probe-stop', '--preflight-interfaces'])
    release = Path(request['cwd'])
    version = read(release / 'RUN_VERSION.json')
    assert version['RUN_COMMIT'] == args.run_commit and version['not_a_git_checkout']
    source_files = []
    for row in version['files']:
        if row['path'].startswith(('lc_rec/letter/scripts/', 'lc_rec/letter/config/')):
            path = release / row['path']
            assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
            source_files.append(row)
    config = release / 'lc_rec/letter/config/formal_instruments_seed42.json'
    training = base / 'formal_seed42/training'
    selected = read(training / 'best_selection.json')
    assert selected['run_commit'] == args.run_commit
    assert selected['completed_training_step'] == 8240
    best = Path(selected['checkpoint'])
    assert best.resolve().parent == training.resolve()
    best_step = int(best.name.removeprefix('checkpoint-'))
    assert selected['loaded_best_checkpoint_step'] == best_step
    assert selected['metric'] == 'eval_loss' and selected['greater_is_better'] is False
    assert sha(training / 'best_adapter/adapter_model.bin') == selected['adapter_sha256']
    reports, summaries = [], []
    for rank in (0, 1):
        report = read(training / ('rank%d.json' % rank))
        assert report['status'] == 'TRAINING_COMPLETE'
        assert report['global_step'] == 8240 and report['epoch'] == 4.0
        assert report['run_commit'] == args.run_commit and report['config_sha256'] == sha(config)
        assert report['best_checkpoint'] == str(best) and report['best_metric'] == selected['value']
        fa = report['formal_attention']
        assert fa['enabled'] and not fa['tensor_observers'] and not fa['torch_global_deterministic']
        assert fa['cublas_workspace_config'] is None
        assert fa['calls']['native_varlen'] > 0
        assert fa['calls']['native_varlen'] == fa['calls']['extension_varlen']
        events = [json.loads(line) for line in
                  (training / ('events.rank%d.jsonl' % rank)).read_text().splitlines()]
        epoch = [v for v in events if v['event'] == 'epoch_end']
        assert [(v['step'], v['epoch']) for v in epoch] == [
            (2060, 1.0), (4120, 2.0), (6180, 3.0), (8240, 4.0)]
        validation = [v for v in events if v['event'] == 'log' and 'eval_loss' in v['values']]
        assert [v['step'] for v in validation] == [2060, 4120, 6180, 8240]
        assert all(math.isfinite(v['values']['eval_loss']) for v in validation)
        winner = min(validation, key=lambda v: v['values']['eval_loss'])
        assert winner['step'] == best_step and winner['values']['eval_loss'] == selected['value']
        restores = [v for v in events if v['event'] == 'strict_ds_restored']
        assert restores and Path(restores[-1]['path']) == best
        assert restores[-1]['global_step'] == best_step
        dispatch = [v for v in events if v['event'] == 'formal_fa_backward_dispatch']
        assert {'native_varlen', 'extension_varlen'} <= {v['boundary'] for v in dispatch}
        assert all(v['deterministic'] is True for v in dispatch)
        updates = [v for v in events if v['event'] == 'optimizer_step']
        assert updates and updates[-1]['step'] == 8240
        assert all(math.isfinite(v['grad_norm']) for v in updates)
        checkpoints = [v for v in events if v['event'] == 'checkpoint_complete']
        assert [v['step'] for v in checkpoints] == [2060, 4120, 6180, 8240]
        summaries.append({'rank': rank, 'epochs': epoch, 'validation': validation,
                          'restores': restores, 'checkpoints': checkpoints,
                          'train_peak_allocated_bytes': report['peak_allocated_bytes'],
                          'train_peak_reserved_bytes': report['peak_reserved_bytes'],
                          'elapsed_seconds': report['elapsed_seconds'],
                          'formal_attention': fa})
        reports.append(report)
    assert [v['values']['eval_loss'] for v in summaries[0]['validation']] == [
        v['values']['eval_loss'] for v in summaries[1]['validation']]
    checkpoints = sorted(training.glob('checkpoint-*'),
                         key=lambda p: int(p.name.split('-')[-1]))
    assert len(checkpoints) <= 2 and best in checkpoints
    assert checkpoints[-1].name == 'checkpoint-8240'
    for path in checkpoints:
        verify_checkpoint(path, args.run_commit, sha(config))
    # CPU adapter-only equality check after terminal job; no base-model load.
    import torch
    saved = torch.load(best / 'adapter_model.bin', map_location='cpu', weights_only=False)
    exported = torch.load(training / 'best_adapter/adapter_model.bin',
                          map_location='cpu', weights_only=False)
    assert set(saved) == set(exported) and len(saved) == 450
    assert all(torch.equal(saved[key], exported[key]) for key in saved)
    test = base / args.job / 'test'
    independent = read(test / 'independent_metrics.json')
    assert independent['status'] == 'PASS' and independent['run_commit'] == args.run_commit
    metrics = read(test / 'metrics.json')
    assert metrics['test_complete'] and metrics['actual_test_users'] == 24772
    assert metrics['adapter_model_sha256'] == selected['adapter_sha256']
    test_reports = [read(test / ('rank%d.json' % rank)) for rank in (0, 1)]
    assert all(v['status'] == 'PASS' and v['completed_users'] == 12386 for v in test_reports)
    resources = [json.loads(line) for line in (job / 'output/resources.jsonl').read_text().splitlines()]
    ledger = reconcile(base / 'formal_seed42_ledger.json')
    cumulative = sum(v['wall_seconds'] for v in ledger['attempts'])
    assert cumulative <= 345600
    stages = [{**row, 'wall_seconds': seconds(row['started_utc'], row['ended_utc'])}
              for row in pipeline['stages'] if 'ended_utc' in row]
    result = {
        'status': 'PASS', 'audit': 'A0_TERMINAL_RESULT_EVIDENCE',
        'RUN_COMMIT': args.run_commit, 'job_id': args.job, 'runner_status': status['status'],
        'source_files': source_files, 'selected_best': selected,
        'selected_checkpoint_export_450_tensors_exact': True,
        'rank_summaries': summaries, 'retained_checkpoints': [str(p) for p in checkpoints],
        'metrics': metrics, 'independent_metrics': independent,
        'stages': stages, 'formal_ledger': ledger,
        'formal_cumulative_wall_seconds': cumulative,
        'formal_gpu_device_seconds': sum(v['gpu_device_seconds'] for v in ledger['attempts']),
        'runner_wall_seconds': seconds(status['started_at'], status['ended_at']),
        'resource_samples': len(resources),
        'min_host_available_bytes': min(v['mem_available'] for v in resources),
        'min_disk_available_bytes': min(v['disk_available'] for v in resources),
        'max_swap_used_bytes': max(v['swap_used'] for v in resources),
        'test_resources': [{key: v[key] for key in (
            'rank', 'elapsed_seconds', 'elapsed_generation_seconds',
            'gpu_peak_allocated_bytes', 'gpu_peak_reserved_bytes',
            'cpu_process_peak_rss_bytes')} for v in test_reports],
        'limits': ['Resource samples are not continuous peaks',
                   'Checkpoint large members checked by recorded size; adapter tensors checked exactly',
                   'Training stage wall includes validation and checkpoint work; do not add it twice',
                   'Controller CPU audit; no independent GPU or training rerun'],
        'argv': sys.argv, 'script_sha256': sha(__file__),
        'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': 'PASS', 'output': str(output), 'sha256': sha(output)}))


if __name__ == '__main__':
    main()
