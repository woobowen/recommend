"""Export this completed run without changing originals or using a GPU."""
from pathlib import Path
import datetime, hashlib, json, zipfile

ROOT = Path('/home/jcy/GRID_wmh_Qwen_TIGER_20261009')
RUN = ROOT / 'experiments/instruments_grid_qwen_20261125_005542'
FAILED = ROOT / 'experiments/instruments_grid_qwen_20261125_005511'
OLD = ROOT / 'codex/instrument_semantic_hierarchy_20261115'
UP = OLD / 'runs/20261115_104723'
OUT = ROOT / 'codex/instruments_grid_publication_20261125'


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    assert json.loads((RUN / 'status.json').read_text())['phase'] == 'complete'
    frozen = json.loads((RUN / 'source_manifest.json').read_text())
    for rel, expected in frozen.items():
        assert sha(RUN / rel) == expected, rel
    metadata = {}
    assets = {}
    for f in RUN.rglob('*'):
        if not f.is_file() or '__pycache__' in f.parts or f.suffix in ('.pyc', '.lock'):
            continue
        rel = f.relative_to(RUN).as_posix()
        if '/checkpoints/' in rel:
            if rel.startswith('formal/'):
                assets['run/' + rel] = f
            continue
        if rel.startswith('data/') or rel in ('sequences.jsonl', 'user_mapping.json'):
            assets['run/' + rel] = f
        else:
            metadata['run/' + rel] = f
    for rel in ['launch.json', 'status.json', 'supervisor.log', 'source_manifest.json',
                'data_audit.json', 'gpu_start.json', 'runner/run_experiment.py',
                'runner/prepare_data.py', 'prepare/exit.json', 'prepare/command.json',
                'prepare/console.log']:
        f = FAILED / rel
        if f.is_file():
            metadata['attempts/failed_005511/' + rel] = f
    for rel in ['qwen_config.json', 'input_audit.json', 'download_manifest.json',
                'upstream/upstream_audit.json', 'upstream/item_embeddings.json',
                'upstream/encoding_checkpoint.json', 'upstream/qwen_status.json']:
        f = UP / rel
        if f.is_file():
            metadata['upstream_metadata/' + rel] = f
    for sub in ['quantizer_train', 'sid_generation', 'write_item_records']:
        for f in (UP / sub).rglob('*'):
            if f.is_file() and f.suffix not in ('.pt', '.ckpt', '.pyc') and '__pycache__' not in f.parts:
                metadata['upstream_metadata/' + f.relative_to(UP).as_posix()] = f
    for rel in ['item_mapping.jsonl', 'items_qwen.jsonl']:
        assets['inputs/' + rel] = OLD / 'inputs/instrument' / rel
    assets['inputs/interactions_5core_aligned.jsonl'] = Path('/home/jcy/data/amazon2018_instruments/interactions_5core_aligned.jsonl')
    for rel in ['upstream/semantic_ids.pt', 'upstream/item_embeddings.pt',
                'quantizer_train/checkpoints/checkpoint_000_000030.ckpt']:
        assets[rel] = UP / rel
    for name in ['run_instrument.py', 'download_prepare.py', 'write_item_records.py']:
        f = OLD / name
        if f.is_file():
            metadata['upstream_control/' + name] = f
    for sub in ['source']:
        for f in (UP / sub).rglob('*'):
            if f.is_file() and f.suffix in ('.py', '.yaml', '.json') and '__pycache__' not in f.parts:
                metadata['upstream_metadata/' + f.relative_to(UP).as_posix()] = f
    for f in (ROOT / 'codex/grid_instruments_20261125').glob('*.md'):
        metadata['historical_preparation/' + f.name] = f
    outputs = []
    for name, files in [('instruments-grid-metadata.zip', metadata),
                        ('instruments-grid-original-assets.zip', assets)]:
        dest = OUT / name
        members = []
        with zipfile.ZipFile(dest, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as z:
            for member, f in sorted(files.items()):
                assert f.is_file(), str(f)
                row = {'member': member, 'source': str(f), 'bytes': f.stat().st_size,
                       'sha256': sha(f)}
                z.write(f, member)
                members.append(row)
            z.writestr('MANIFEST.json', json.dumps(members, indent=2))
        with zipfile.ZipFile(dest) as z:
            assert z.testzip() is None
        spec = {'name': name, 'bytes': dest.stat().st_size, 'sha256': sha(dest),
                'members': members}
        (OUT / (name + '.json')).write_text(json.dumps(spec, indent=2))
        outputs.append({k: v for k, v in spec.items() if k != 'members'})
    record = {'time_beijing': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
              'run': str(RUN), 'source_files_verified': len(frozen), 'outputs': outputs,
              'scope': 'CPU export and CRC/hash checks; no model reevaluation/training',
              'omitted': ['smoke checkpoint: disposable preflight model, metadata retained',
                          'locks and bytecode caches']}
    (OUT / 'export_receipt.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
