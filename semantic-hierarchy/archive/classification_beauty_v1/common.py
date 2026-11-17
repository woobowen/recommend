import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_config():
    return json.loads((HERE / 'config.json').read_text(encoding='utf-8'))


def roots():
    cfg = load_config()
    root = Path(cfg['root'])
    return cfg, root, HERE / 'preparation', root / 'experiments' / cfg['experiment']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    tmp.replace(path)


def prefix_groups(sids):
    from collections import defaultdict
    result = {}
    for length in (1, 2, 3):
        groups = defaultdict(list)
        for item_id, sid in enumerate(sids):
            groups[tuple(int(x) for x in sid[:length])].append(item_id)
        result[length] = dict(groups)
    return result


def canonical_taxonomy(value):
    """IDs depend only on the named text taxonomy, never on SID statistics."""
    categories = value.get('categories')
    if not isinstance(categories, list) or not categories:
        raise ValueError('Expected a nonempty categories array')
    required = ['coarse_name', 'name', 'definition']
    for c in categories:
        if any(not isinstance(c.get(k), str) or not c[k].strip() for k in required):
            raise ValueError('Invalid taxonomy entry: ' + str(c))
    keys = [(c['coarse_name'].strip().casefold(), c['name'].strip().casefold()) for c in categories]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate category names; merge them before freezing')
    coarse_names = sorted({c['coarse_name'].strip() for c in categories})
    coarse_ids = {name: f'C{i+1:02d}' for i, name in enumerate(coarse_names)}
    output = []
    for i, c in enumerate(sorted(categories, key=lambda c: (c['coarse_name'], c['name']))):
        output.append(dict(c, id=f'F{i+1:03d}', coarse_id=coarse_ids[c['coarse_name'].strip()]))
    return {'coarse_categories': [{'id': coarse_ids[name], 'name': name} for name in coarse_names],
            'categories': output,
            'unknown': {'id': 'UNKNOWN', 'coarse_id': 'UNKNOWN', 'definition': 'Insufficient text to identify the main product type.'}}
