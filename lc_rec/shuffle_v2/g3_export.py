"""Export only the G3 evidence required for the complete review ZIP."""
import argparse
import datetime
import json
from pathlib import Path
import tarfile

from lc_rec.shuffle_v2 import core, g3_assets

JOB_FILES = ('request.json', 'accepted.json', 'attempt.json', 'status.json', 'cleanup.json', 'work.log', 'supervisor.log')
DATA_ROOT = Path('/data/bwwu/lc-rec/g3-beauty-v2')
PREFIX = 'g3-beauty-v2-'


def export(args):
    config, _, normal_rank, adapter, frozen_paths = g3_assets.load()
    data = Path(config['data_root']) / 'Beauty'
    record = g3_assets.read(data / 'Beauty.g3.upstream.json')
    selected = {}

    def add(path, member):
        path = Path(path)
        assert path.is_file() and member not in selected, (path, member)
        selected[member] = path

    for kind, digest in config['assets'].items():
        name = f'Beauty.{kind}.json'
        add(g3_assets.verified(data / name, digest), 'data/Beauty/' + name)
    add(data / 'Beauty.g3.upstream.json', 'historical/Beauty.g3.upstream.json')
    add(g3_assets.verified(record['g3']['path'], record['g3']['sha256']), 'historical/upstream-semantic_ids.pt')
    for relative, path in frozen_paths.items():
        if relative.endswith('/Beauty.g3.index.json'):
            continue  # The same bytes appear as data/Beauty/Beauty.index.json.
        add(path, f'frozen-source/{relative}')
    formal = Path(config['formal_root'])
    for name in ('best_selection.json', 'rank0.json', 'checkpoint-6160/complete.json'):
        add(formal / 'training' / name, f'historical/training/{name}')
    for name in ('g3_identity.json', 'adapter_config.json', 'tokenizer.model', 'tokenizer_config.json',
                 'added_tokens.json', 'special_tokens_map.json'):
        add(g3_assets.verified(adapter / name, normal_rank['adapter_files'][name]['sha256']), f'historical/adapter/{name}')
    for name in ('generation_config.json', 'config.json', 'model.safetensors.index.json', 'tokenizer_config.json', 'special_tokens_map.json'):
        add(g3_assets.verified(Path(config['model']) / name, normal_rank['model']['files'][name]['sha256']), f'historical/base/{name}')
    normal = formal / 'attempts/normal/a0/test/metrics.json'
    add(g3_assets.verified(normal, config['normal_metrics_sha256']), 'historical/normal.metrics.json')
    steps = [args.replay, args.normal, f'cpu-paired-{args.preflight_attempt}']
    for mode in core.MODES:
        steps += [f'gpu-preflight-{mode}-{args.preflight_attempt}', f'cpu-eval-{mode}-{args.preflight_attempt}']
    for case in ('decoy', 'normal', 'orphan', 'timeout'):
        steps.append(f'cpu-lifecycle-{case}-r0')
    for step in steps:
        root = DATA_ROOT / step
        assert root.is_dir()
        for path in sorted(root.rglob('*')):
            if path.is_file():
                assert not path.is_symlink()
                if path.name == 'normal_preflight.jsonl':
                    continue  # The 32 rows are selected offline from the original full JSONL.
                if path.name in ('derived.rank0.py', 'derived.rank0.diff') and step.startswith('gpu-preflight-'):
                    first = DATA_ROOT / f'gpu-preflight-shuffle2-{args.preflight_attempt}' / path.name
                    assert g3_assets.sha(first) == g3_assets.sha(path)
                    if root != first.parent and path.suffix == '.diff':
                        continue
                add(path, f'remote/{step}/{path.relative_to(root)}')
    add(DATA_ROOT / 'cpu-lifecycle-review-r0.json', 'remote/cpu-lifecycle-review-r0.json')
    job_root = args.job_root
    job_ids, requests, boundaries = [], [], {}
    active_export = PREFIX + args.output.stem.removesuffix('.tar')
    for path in sorted(job_root.glob(PREFIX + '*')):
        job_ids.append(path.name)
        if path.name == active_export:
            continue
        assert not any(word in path.name for word in ('formal', 'group-series'))
        status = g3_assets.read(path / 'status.json')
        assert status['ended_at'] and status['cleanup']['verified']
        boundaries[path.name] = {key: status.get(key) for key in ('status', 'exit_code', 'started_at', 'ended_at')}
        for name in JOB_FILES:
            add(path / name, f'jobs/{path.name}/{name}')
    for path in sorted(args.requests_root.glob(PREFIX + '*')):
        requests.append(path.name)
    outputs = sorted(path.name for path in DATA_ROOT.iterdir())
    assert all('formal' not in name and 'group-series' not in name for name in [*outputs, *requests])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    inventory_path = args.output.with_name('server-inventory.json')
    with inventory_path.open('x') as stream:
        json.dump({'captured_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   'jobs': job_ids, 'requests': requests, 'outputs': outputs, 'formal_submitted': False,
                   'job_root': str(job_root), 'requests_root': str(args.requests_root), 'timing_cutoff': boundaries},
                  stream, indent=2, allow_nan=False)
    add(inventory_path, 'server-inventory.json')
    summary = {}
    with tarfile.open(args.output, 'x:gz', compresslevel=3) as archive:
        for member, path in sorted(selected.items()):
            before = path.stat()
            digest = g3_assets.sha(path)
            archive.add(path.resolve(), arcname=member, recursive=False)
            after = path.stat()
            assert (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
            summary[member] = {'remote': str(path), 'bytes': before.st_size, 'sha256': digest}
    with args.output.with_name('export-manifest.json').open('x') as stream:
        json.dump({'path': str(args.output), 'bytes': args.output.stat().st_size,
                   'sha256': g3_assets.sha(args.output), 'members': summary}, stream, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for flag in ('output', 'job-root', 'requests-root'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--replay', required=True)
    parser.add_argument('--normal', required=True)
    parser.add_argument('--preflight-attempt', required=True)
    export(parser.parse_args())


if __name__ == '__main__':
    main()
