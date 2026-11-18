"""Pinned G3 Beauty experiment identity. No bypass or asset-selection mode."""
import hashlib
import json
from pathlib import Path

EXPERIMENT_ID = 'grid-g3-beauty-lcrec-seed42-20261118'
INDEX_SHA = '420fc060caba3a22b0f03900dfe26abc67da8753f3bac2803f7f3c8703a05c22'
BRIDGE_SHA = '10a7ca66dba43b69232a7c96b48d3287df9314a70a9ec9083ff00acfcf2a8c4f'
G3_SHA = 'd41adcfb22a9a47d5e0424c068f4c74f33cc9fdbaf5db08540fbfa0e3d720e18'
ASSETS = {'index': INDEX_SHA, 'inter': '8809584b8b78fa6561771c88911ed65bd716fc9926fca1e581e6a1ab578ccfd5',
          'item': '74aeddf911548aadfcd1dab326c16678ed4ac74457e874da300d63a94bdb2330'}
VOCAB_SIZE = 32786
NEW_TOKEN_COUNT = 786
TRAINABLE_PARAMETERS = 288571392
TOKEN_MAPPING_SHA = 'dba960c3166b49d9098c713095273db3b6faa856ae2485ec8eda408ae66e82bc'
MODEL_REVISION = '4782ad278652c7c71b72204d462d6d01eaaf7549'
HERE = Path(__file__).resolve().parent
MODEL_FILES = {
 'config.json': 'e4c3dea29aefc16c0e63825f9ae097dbf5cb5d7c320843a84257e11be3b452c8',
 'model-00001-of-00002.safetensors': 'd43476fdd2fca0c44d55ee930039dd5dafb6331764dc0b5e5f89c60b551fcc12',
 'model-00002-of-00002.safetensors': '94712c4b457815e0f8aaa4f78170aebba2b9ee3339ff2b40fceb983338fbafee',
 'model.safetensors.index.json': '217d527037a95cd5cd30384afb425987824373b94cc7206beb48ec0c9499c98f',
 'tokenizer.model': '9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347',
 'tokenizer_config.json': 'fc6996ba04b235235cf9c7bd00fd8817c9b8abf445fe14b0a36ef900e3b9cfe8',
 'special_tokens_map.json': 'ff3b4a612c4e447acb02d40071bddd989fe0da87eb5b7fe0dbadfc4f74de7531',
 'generation_config.json': 'fd7ff399e5568cc21a0a8414f43df88ef7c424995b9b97a90563165d2cf79efd'}


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def source_identity(paths):
    return [{'path': str(Path(p).resolve()), 'bytes': Path(p).stat().st_size, 'sha256': sha(p)} for p in paths]


def verify_assets(data_root):
    root = Path(data_root) / 'Beauty'
    rows = {k: source_identity([root / ('Beauty.' + k + '.json')])[0] for k in ASSETS}
    need({k: r['sha256'] for k, r in rows.items()} == ASSETS, 'G3 data asset identity')
    upstream = json.loads((root / 'Beauty.g3.upstream.json').read_text())
    need(upstream['experiment_id'] == EXPERIMENT_ID, 'G3 upstream experiment')
    for k, expected in [('g3', G3_SHA), ('bridge', BRIDGE_SHA)]:
        need(sha(upstream[k]['path']) == upstream[k]['sha256'] == expected, 'G3 upstream ' + k + ' identity')
    need(sha(HERE / 'grid_sid_preparation/identity_audit_abfe/bridge.csv') == BRIDGE_SHA, 'release bridge identity')
    need(sha(HERE / 'grid_sid_preparation/g3_conversion/Beauty.g3.index.json') == INDEX_SHA, 'release index identity')
    index = json.loads((root / 'Beauty.index.json').read_text())
    need(set(index) == set(map(str, range(12101))), 'G3 item axis')
    need(len({tuple(x) for x in index.values()}) == 12101, 'G3 full SID uniqueness')
    need([len({x[i] for x in index.values()}) for i in range(4)] == [256, 256, 256, 18], 'G3 position sets')
    return rows


def tokenizer_identity(tokenizer, index=None):
    index = index or json.loads((HERE / 'grid_sid_preparation/g3_conversion/Beauty.g3.index.json').read_text())
    tokens = sorted({t for row in index.values() for t in row})
    lookup = {t: tokenizer.encode(t, add_special_tokens=False) for t in tokens}
    need(len(tokenizer) == VOCAB_SIZE and len(tokens) == NEW_TOKEN_COUNT, 'G3 tokenizer vocabulary')
    need(all(len(v) == 1 for v in lookup.values()), 'G3 SID must be single tokens')
    mapping = {k: v[0] for k, v in lookup.items()}
    need(len(set(mapping.values())) == NEW_TOKEN_COUNT and not set(mapping.values()) & set(tokenizer.all_special_ids), 'G3 token IDs/special collision')
    need(digest(mapping) == TOKEN_MAPPING_SHA, 'G3 token to ID mapping mismatch')
    return {'vocab_size': VOCAB_SIZE, 'new_token_count': NEW_TOKEN_COUNT,
            'token_to_id_sha256': digest(mapping), 'full_vocab_sha256': digest(tokenizer.get_vocab())}


_BASE_CACHE = {}


def verify_base_model(model):
    root = Path(model).resolve()
    need(root.name == MODEL_REVISION, 'G3 base model revision')
    # Cache only while all underlying file identities remain unchanged.
    signature = tuple((n, (root/n).stat().st_size, (root/n).stat().st_mtime_ns, (root/n).stat().st_ctime_ns) for n in MODEL_FILES)
    key = (str(root), signature)
    if key not in _BASE_CACHE:
        rows = {n: source_identity([root/n])[0] for n in MODEL_FILES}
        need(all(rows[n]['sha256'] == h for n, h in MODEL_FILES.items()), 'G3 base model content identity')
        _BASE_CACHE[key] = {'path': str(root), 'revision': MODEL_REVISION, 'files': rows}
    return _BASE_CACHE[key]


def write_adapter_identity(adapter, *, scope, run_commit, config_sha256, tokenizer, data_root, model, **extra):
    root = Path(adapter)
    need(scope in ('FORMAL', 'PREFLIGHT'), 'adapter scope')
    value = {'experiment_id': EXPERIMENT_ID, 'scope': scope, 'assets': ASSETS,
             'upstream': {'g3': G3_SHA, 'bridge': BRIDGE_SHA}, 'data_identity': verify_assets(data_root),
             'run_commit': run_commit, 'config_sha256': config_sha256,
             'tokenizer_identity': tokenizer_identity(tokenizer), 'base_model_identity': verify_base_model(model),
             'files': source_identity([p for p in sorted(root.iterdir()) if p.is_file() and p.name != 'g3_identity.json']), **extra}
    value['adapter_sha256'] = sha(root/'adapter_model.bin')
    (root/'g3_identity.json').write_text(json.dumps(value, indent=2)+'\n')
    return value


def verify_adapter_identity(adapter, *, scope, run_commit=None, tokenizer=None, model=None):
    root = Path(adapter)
    value = json.loads((root/'g3_identity.json').read_text())
    need(value['experiment_id'] == EXPERIMENT_ID and value['assets'] == ASSETS, 'G3 adapter experiment/assets')
    need(value['scope'] == scope and scope in ('FORMAL', 'PREFLIGHT'), 'G3 adapter scope mismatch')
    need(value['upstream'] == {'g3': G3_SHA, 'bridge': BRIDGE_SHA}, 'G3 adapter upstream identity')
    base = value['base_model_identity']
    need(base.get('revision') == MODEL_REVISION and
         {n: r['sha256'] for n, r in base['files'].items()} == MODEL_FILES,
         'G3 adapter base model provenance')
    if run_commit is not None:
        need(value['run_commit'] == run_commit, 'G3 adapter source commit')
    need(value['tokenizer_identity']['token_to_id_sha256'] == TOKEN_MAPPING_SHA, 'G3 adapter token mapping')
    names = set()
    for row in value['files']:
        name = Path(row['path']).name
        need(name not in names, 'duplicate adapter member')
        names.add(name)
        p = root/name
        need(p.is_file() and not p.is_symlink() and p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], 'G3 adapter member identity: '+name)
    need({'adapter_model.bin','adapter_config.json','tokenizer.model','tokenizer_config.json','added_tokens.json','special_tokens_map.json'} <= names, 'G3 adapter missing members')
    need(sha(root/'adapter_model.bin') == value['adapter_sha256'], 'G3 adapter weight identity')
    if tokenizer is not None:
        need(tokenizer_identity(tokenizer) == value['tokenizer_identity'], 'G3 reloaded tokenizer identity')
    if model is not None:
        need(verify_base_model(model) == value['base_model_identity'], 'G3 reloaded model identity')
    return value


def verify_launch_authorization(path, expected_sha, run_commit, config_sha256):
    need(path and expected_sha and sha(path) == expected_sha, 'new G3 formal launch authorization required')
    r = json.loads(Path(path).read_text())
    need(r.get('experiment_id') == EXPERIMENT_ID and r.get('asset_sha256') == ASSETS,
         'authorization belongs to a different experiment')
    need(r.get('authorized_by') == 'user' and r.get('formal_launch_authorized') is True and
         r.get('run_commit') == run_commit and r.get('config_sha256') == config_sha256 and
         r.get('scope') == 'FORMAL' and r.get('authorization_kind') == 'USER_LAUNCH_AUTHORIZATION',
         'explicit new G3 user formal launch authorization required')
    return r
