"""Final CPU-only audit. Requires the synthetic GPU smoke test to have passed."""
import ast
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

from common import HERE, roots, sha, write_json, read_jsonl


def main():
    cfg, root, prep, out = roots()
    model_dir = root / cfg['model']
    downloaded = json.loads((prep / 'model_download_verified.json').read_text())
    assert downloaded['complete']
    for f in downloaded['files']:
        path = model_dir / f['file']
        assert path.stat().st_size == f['bytes'] and sha(path) == f['sha256'], f['file']
    model_config = json.loads((model_dir / 'config.json').read_text())
    assert model_config['model_type'] == 'llama' and model_config['hidden_size'] == 4096
    assert model_config['num_hidden_layers'] == 32 and model_config['num_key_value_heads'] == 8
    index = json.loads((model_dir / 'model.safetensors.index.json').read_text())
    for name in set(index['weight_map'].values()):
        assert (model_dir / name).exists()
    audit = json.loads((prep / 'data_preparation_verified.json').read_text())
    token_audit = json.loads((prep / 'text_token_audit.json').read_text())
    assert audit['config_sha256'] == sha(HERE / 'config.json')
    for dataset, spec in cfg['datasets'].items():
        rows = read_jsonl(prep / dataset / 'classifier_items.jsonl')
        assert len(rows) == spec['expected_items']
        assert all(set(row) == {'item_id', 'text'} for row in rows)
        assert sha(prep / dataset / 'classifier_items.jsonl') == audit['datasets'][dataset]['classifier_items_sha256']
        assert token_audit['datasets'][dataset]['classifier_items_sha256'] == audit['datasets'][dataset]['classifier_items_sha256']
    subprocess.run([sys.executable, str(HERE / 'selftest.py')], check=True)
    # Verify that real-data entry points refuse to run without explicit execution.
    checks = [[sys.executable, str(HERE/'run_stage.py'), 'discover', '--dataset', 'beauty'],
              [sys.executable, str(HERE/'launch_stage.py'), 'discover', '--dataset', 'beauty'],
              [sys.executable, str(HERE/'evaluate.py')]]
    for command in checks:
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 2 and '--execute' in result.stderr
    smoke = json.loads((prep / 'model_smoke_verified.json').read_text())
    assert smoke['passed'] and smoke['synthetic_only'] and smoke['real_items_classified'] == 0
    assert smoke['config_sha256'] == sha(HERE / 'config.json')
    versions = {}
    for name in ['torch', 'transformers', 'accelerate', 'numpy', 'safetensors', 'tokenizers', 'tensorflow', 'requests']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'not installed under this distribution name'
    for path in HERE.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), filename=path.name)
    # No real annotation, taxonomy discovery, pilot or evaluation is allowed during preparation.
    assert not out.exists() or not any(out.rglob('*.jsonl')), 'Unexpected real experiment output exists'
    assert not out.exists() or not any(out.rglob('taxonomy*.json')), 'Unexpected generated taxonomy exists'
    ready = {'ready': True, 'state': 'WAITING_FOR_USER_CONFIRMATION', 'experiment_started': False,
        'prepared_unix': time.time(), 'python': sys.version, 'executable': sys.executable,
        'environment': 'isolated venv with read-only access to existing grid packages; no package upgrades',
        'packages': versions, 'model_path': str(model_dir),
        'model_download_manifest_sha256': sha(prep/'model_download_verified.json'),
        'source_sha256': {p.name: sha(p) for p in HERE.glob('*.py') if p.name != 'transfer_model.py'},
        'config_sha256': sha(HERE/'config.json'),
        'data_preparation_manifest_sha256': sha(prep/'data_preparation_verified.json'),
        'synthetic_model_smoke_sha256': sha(prep/'model_smoke_verified.json'),
        'text_token_audit_sha256': sha(prep/'text_token_audit.json'),
        'datasets': {d: {'items': a['items'], 'blank_texts': a['blank_texts'],
                         'prefix_summary': a['prefix_summary']} for d, a in audit['datasets'].items()}}
    write_json(prep / 'readiness.json', ready)
    write_json(prep / 'environment_versions.json', versions)
    print(json.dumps({'ready': True, 'experiment_started': False, 'model': str(model_dir)}, indent=2))


if __name__ == '__main__':
    main()
