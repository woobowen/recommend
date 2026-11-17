"""Read-only audit of saved Qwen embeddings and RQ-KMeans checkpoints on ECNU."""
from pathlib import Path
import hashlib
import json
import pickletools
import zipfile

import torch


ROOT = Path('/home/jcy/GRID_wmh_Qwen_TIGER_20261009')
RUNS = ['beauty_rkmeans_qwen_20261010_012253',
        'toys_rkmeans_qwen_20261010_093036']


def checkpoint_flag(path, key):
    # Inspect the serialized scalar without importing project classes or unpickling.
    with zipfile.ZipFile(path) as archive:
        member = next(n for n in archive.namelist() if n.endswith('/data.pkl'))
        operations = list(pickletools.genops(archive.read(member)))
    hyper_start = next(pos for _, arg, pos in operations if arg == 'hyper_parameters')
    matches = []
    for index, (_, argument, position) in enumerate(operations):
        if argument != key or position < hyper_start:
            continue
        following = next(op for op, _, _ in operations[index + 1:]
                         if op.name not in {'BINPUT', 'LONG_BINPUT', 'MEMOIZE'})
        if following.name not in {'NEWTRUE', 'NEWFALSE'}:
            raise ValueError(f'Expected boolean after {key}, got {following.name}')
        matches.append({'value': following.name == 'NEWTRUE', 'pickle_offset': position})
    if len(matches) != 1:
        raise ValueError(f'Expected one {key} in checkpoint hyperparameters: {matches}')
    return matches[0]


result = {'root': str(ROOT), 'runs': [], 'source_evidence': {}}
for name in RUNS:
    run = ROOT / 'experiments' / name
    path = run / 'item_embeddings.pt'
    vectors = torch.load(path, map_location='cpu', weights_only=True)
    norms = vectors.double().norm(dim=-1)
    checkpoint = next((run / 'quantizer_train/checkpoints').glob('*.ckpt'))
    result['runs'].append({
        'run': name, 'embedding_path': str(path),
        'embedding_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'shape': list(vectors.shape), 'dtype': str(vectors.dtype),
        'all_finite': bool(torch.isfinite(vectors).all()),
        'norm_min': norms.min().item(), 'norm_mean': norms.mean().item(),
        'norm_median': norms.median().item(), 'norm_max': norms.max().item(),
        'norm_std_population': norms.std(unbiased=False).item(),
        'unit_norm_fraction_at_1e-4': ((norms - 1).abs() < 1e-4).double().mean().item(),
        'checkpoint': str(checkpoint),
        'checkpoint_normalize_residuals': checkpoint_flag(checkpoint, 'normalize_residuals'),
    })
    del vectors, norms

for relative, intervals in {
    '_运行/generate_qwen_embeddings.py': [(60, 88)],
    'src/models/components/network_blocks/aggregation_strategy.py': [(55, 75)],
    'src/modules/clustering/residual_quantization.py': [(18, 32), (204, 214), (246, 258), (287, 301)],
    'configs/experiment/rkmeans_train_flat.yaml': [(138, 152)],
    'configs/experiment/rkmeans_inference_flat.yaml': [(11, 25)],
    '_运行/run_pipeline.py': [(129, 147)],
}.items():
    source = (ROOT / relative).read_text().splitlines()
    result['source_evidence'][relative] = [
        {'line': number, 'text': source[number - 1]}
        for start, end in intervals for number in range(start, min(end, len(source)) + 1)
    ]

print(json.dumps(result, ensure_ascii=False, indent=2))
