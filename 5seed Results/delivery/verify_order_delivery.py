"""CPU-only source, checkpoint pairing and raw-attention identity verification."""
import hashlib
import json
from pathlib import Path


def load(p): return json.loads(p.read_text(encoding='utf-8'))


def main():
    here = Path(__file__).resolve().parent
    runs = load(here/'run_index.json'); assert len(runs) == 20
    assets = load(here/'assets.json')['assets']
    models = [r for name, spec in assets.items() if name.startswith('order-checkpoints') for r in spec['members']]
    assert len(models) == 20
    checked = 0
    for r in runs:
        manifest = load(here/'runtime_records'/r['key']/'source_manifest.json')
        assert manifest['sha256'] == r['source_signature']
        assert hashlib.sha256(json.dumps(manifest['files'], sort_keys=True).encode()).hexdigest() == manifest['sha256']
        for name, sha in manifest['files'].items():
            assert hashlib.sha256((here/'source_snapshots'/r['source_signature']/name).read_bytes()).hexdigest() == sha, name
            checked += 1
        assert any(m['source'] == r['checkpoint'] and m['sha256'] == r['checkpoint_sha256'] for m in models)
        stage = here/'runtime_records'/r['key']/'stage_result.json'
        assert hashlib.sha256(stage.read_bytes()).hexdigest() == r['stage_result_sha256']
    old = load(here.parent/'attention/source_manifest.json')
    raw = assets['order-attention-originals.zip']['members']
    assert len(old) == len(raw) == 4000
    by_source = {r['source']: r['sha256'] for r in raw}
    assert all(by_source.get(r['source_file']) == r['sha256'] for r in old)
    report = {'runs': 20, 'distinct_source_snapshots': len({r['source_signature'] for r in runs}),
              'source_file_comparisons': checked, 'checkpoint_pairs': 20, 'raw_attention_identities': 4000,
              'gpu_work': False, 'scope': 'Saved original identities; no model evaluation'}
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
