"""Fixed asset binding for the G3 Beauty V2 replay and generation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

CONFIG = Path(__file__).with_name('a0_beauty_g3.json')
ENTRY = 'lc_rec/a0_beauty/formal_evaluate.py'
VENDOR = 'lc_rec/letter/vendor/LETTER/LETTER-LC-Rec'
MANIFEST = 'lc_rec/a0_beauty/config/preflight_g3_users_seed42.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verified(path, expected):
    path = Path(path)
    if sha(path) != expected:
        raise ValueError(f'G3 frozen content identity mismatch: {path}')
    return path


def load():
    config = read(CONFIG)
    release = Path(config['release'])
    formal = Path(config['formal_root'])
    paths = {name: verified(release / name, digest) for name, digest in config['source_files'].items()}
    normal = formal / 'attempts/normal/a0/test'
    normal_rank = read(verified(normal / 'rank0.json', config['normal_rank_sha256']))
    adapter = formal / 'training/best_adapter'
    identity = read(verified(adapter / 'g3_identity.json', config['model_identity_sha256']))
    assert identity['scope'] == 'FORMAL' and identity['run_commit'] == config['training_commit']
    assert identity['adapter_sha256'] == config['adapter_sha256']
    assert identity == normal_rank['adapter_identity']
    for member in ('tokenizer.model', 'added_tokens.json', 'tokenizer_config.json', 'special_tokens_map.json'):
        verified(adapter / member, normal_rank['adapter_files'][member]['sha256'])
    source = paths[ENTRY]
    sys.path[:0] = [str(source.parent), str(release / VENDOR)]
    spec = importlib.util.spec_from_file_location('frozen_g3_formal', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return config, module, normal_rank, adapter, paths
