"""Copy bounded terminal receipts; retain original full versus selective identities."""
import json
from pathlib import Path
import shutil

REPO = Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
SUPPORT = Path('/home/addaswsw/recommend-support/grid-formal-20261015')
JOBS = Path('/home/addaswsw/recommend-support/sid-extensions-p0/jobs/new-server')
ROOT = REPO/'manifests/grid-formal-20261015'


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert target.read_bytes() == source.read_bytes(), target
    else:
        shutil.copyfile(source, target)


def main():
    path = ROOT/'EXECUTION_SUMMARY.json'
    summary = json.loads(path.read_text())
    records = {r['job_id']: r for r in summary['jobs']}
    candidates = {d.name: (d/'retrieved', False) for d in JOBS.glob('gf-20261015-*')
        if (d/'retrieved/status.json').is_file()}
    for collection in sorted(SUPPORT.glob('collected-*')):
        manifest_path = collection/'COLLECTION_MANIFEST.json'
        if not collection.is_dir() or not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text())
        for job in manifest['jobs']:
            assert job.startswith('gf-20261015-') and '/' not in job
            directory = collection/job
            if job not in candidates:
                candidates[job] = (directory, True)
    for job, (directory, selective) in sorted(candidates.items()):
        status = json.loads((directory/'status.json').read_text())
        request = json.loads((directory/'request.json').read_text())
        assert status['status'] in ('SUCCEEDED', 'FAILED', 'TIMED_OUT', 'CANCELLED')
        assert status['cleanup']['verified'] and request['job_id'] == status['job_id'] == job
        assert request['cwd'].startswith('/home/bwwu/grid-audits/sid-extensions-p0/grid-formal-20261015/snapshots/')
        for f in directory.rglob('*'):
            assert not f.is_symlink(), f
            if not f.is_file():
                continue
            relative = f.relative_to(directory)
            if f.suffix in ('.json', '.yaml', '.yml', '.csv', '.log', '.py') and f.stat().st_size <= 1024**2:
                copy(f, ROOT/'runs'/job/relative)
            # A single prespecified first-cohort figure is readable on GitHub;
            # all 32 per-condition figures remain in raw evidence and final ZIP.
            if str(relative) == 'output/batch-00/figures/sample_000_cross_attention.png':
                copy(f, ROOT/'figures'/job/f.name)
        if selective:
            copy(directory.parent/'COLLECTION_MANIFEST.json', ROOT/'runs'/job/'COLLECTION_MANIFEST.json')
        phase = request['argv'][4] if 'experiments.formal_20261015.run' in request['argv'] else (
            'CPU_TRAINING_CURVES' if 'experiments.beauty_extensions.plot_training' in request['argv'] else 'CPU_CONFIGURATION_REGRESSION')
        record = {'job_id': job, 'phase': phase, 'status': status['status'],
            'exit_code': status['exit_code'], 'fetched': True,
            'source_commit': request['capture_version'].split('/')[0],
            'snapshot_sha256': request['capture_version'].split('/')[1],
            'physical_gpu': request['env']['CUDA_VISIBLE_DEVICES'], 'cleanup_verified': True,
            'started_at': status['started_at'], 'ended_at': status['ended_at'],
            'evidence': 'runs/'+job, 'full_export_remote': '/home/bwwu/grid-audits/sid-extensions-p0/jobs/'+job}
        record['selective_export_local' if selective else 'full_export_local'] = str(directory)
        if selective:
            record['fetch_scope'] = 'verified original job metadata and lightweight outputs; model/data payloads remain server-side'
        result_file = directory/'output/RESULT.json'
        if result_file.is_file():
            result = json.loads(result_file.read_text())
            record['result_status'] = result['status']
            if phase == 'attention':
                record.update({'attention_samples': result.get('samples'),
                    'formal_attention': result.get('formal'),
                    'display_sample_selection': 'first sample of pre-existing fixed cohort; no output-based selection'})
            if phase == 'train':
                record.update({'training_complete': result.get('training_complete'),
                    'test_complete': result.get('test_complete'), 'best_checkpoint': result.get('best_checkpoint')})
        records[job] = record
    summary['jobs'] = list(records.values())
    # Overall status and research conclusions are set by main control after
    # actual independent result review, never inferred from runner exit zero.
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'terminal_evidence_jobs': len(candidates), 'total_records': len(records),
        'overall_status_unchanged': summary['status']}))


if __name__ == '__main__':
    main()
