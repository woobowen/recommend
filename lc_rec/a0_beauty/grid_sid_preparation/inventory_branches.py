#!/usr/bin/env python3
"""Inventory candidate-looking paths at fixed refs; not an exhaustive provenance audit."""
import json
import re
import subprocess

REFS = {
 'SID-Order-Invariance': '48861a5688ff10908e4f564917e060c52da8b871',
 'main': '563dc50b1c7e9e2e05cdfddd13e91a3d1888001e',
 'sid-extensions': 'a2bbdec09eb88f10ea22aed641d389d445039220',
 'sid-semantic-hierarchy': 'e8f4603ddf5e7783162efc747b2b7cc60d7da2f3',
 'wbw-grid': '9767729a5114e21b9c949892af3d759baca6a98e',
 'wbw-lc-rec': '2a7d193af47a533a8f0e1ab5691902ae3040970b',
 'wbw-sid-order': '7f7697e4baf28fb252b45e3afd356c333c455dbb',
 'wmh-grid': '7e943ea9a7d92f16eb10c6c1b27c27761d3f6c20',
 'yjc-grid': 'c92e1576210b3daf2ce6f63f77b65f651f6f7127',
}
out = {}
for name, sha in REFS.items():
    raw = subprocess.check_output(['git', 'ls-tree', '-rz', '--full-tree', sha])
    rows = []
    for entry in raw.split(b'\0'):
        if not entry: continue
        meta, bpath = entry.split(b'\t', 1); path = bpath.decode()
        if re.search(r'(semantic.?ids?|merged_predictions|item.*sid|sid.*item|rqvae|rkmeans)', path, re.I) and not re.search(r'\.(py|pyc|tfrecord\.gz)$', path, re.I):
            mode, kind, oid = meta.decode().split()
            rows.append({'path': path, 'blob': oid})
    out[name] = {'commit': sha, 'path_count': len(rows), 'paths': rows}
print(json.dumps(out, ensure_ascii=False, indent=2))
