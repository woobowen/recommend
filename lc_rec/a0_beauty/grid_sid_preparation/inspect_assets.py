#!/usr/bin/env python3
"""Read-only identity and structure inspection, not an item-mapping proof."""
import collections
import hashlib
import json
from pathlib import Path
import torch

ASSETS = {
    'G1': '/home/bwwu/grid-audits/sid-extensions-p0/beauty-extensions/assets/188af0341d8c232de73acd9aa7fc3390db8d430c8c71a94ec9816962d3d94f68/merged_predictions_tensor.pt',
    'G2': '/home/bwwu/grid-audits/sid-extensions-p0/beauty-extensions/assets/2a9513a5d7c2a45ee8ccbd923647699350bffe41c804379d6c48e3ddd27645ba/merged_predictions_tensor.pt',
    'G3': '/data/bwwu/sid-extensions-assets/563dc50b1c7e/experiments/beauty_rkmeans_qwen_20261010_012253/semantic_ids.pt',
}
DATA = Path('/home/bwwu/grid-audits/sid-extensions-p0/transfers/letter-a0-beauty-author-8d0154e28de3/Beauty')

def identity(path):
    b = Path(path).read_bytes()
    return {'path': str(path), 'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()}

out = {'warning': 'Tensor axes and matching counts alone do not establish GRID-to-LETTER item correspondence.', 'candidates': {}, 'letter': {}}
for key, path in ASSETS.items():
    t = torch.load(path, map_location='cpu', weights_only=True)
    row = identity(path)
    row.update(shape=list(t.shape), dtype=str(t.dtype), minimum=t.min().item(), maximum=t.max().item())
    if t.ndim == 2 and t.shape[0] == 4:
        rows = [tuple(x) for x in t.T.tolist()]
        counts = collections.Counter(rows)
        row.update(assumed_item_axis=1, unique_rows=len(counts), collision_groups=sum(v > 1 for v in counts.values()), per_position_range=[[t[i].min().item(), t[i].max().item()] for i in range(4)])
    out['candidates'][key] = row
for name in ('Beauty.index.json', 'Beauty.item.json', 'Beauty.inter.json'):
    path = DATA / name
    if path.is_file():
        data = json.loads(path.read_text())
        row = identity(path)
        row.update(type=type(data).__name__, length=len(data))
        if isinstance(data, dict) and data:
            k = next(iter(data)); row['example_key'] = k; row['example_value_type'] = type(data[k]).__name__
        out['letter'][name] = row
print(json.dumps(out, indent=2, sort_keys=True))
