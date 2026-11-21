"""Assemble a new derived runtime without modifying frozen snapshots or assets."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--run-key', required=True)
    ap.add_argument('--baseline-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args(); here = Path(__file__).resolve().parent
    rows = json.loads((here/'run_index.json').read_text(encoding='utf-8'))
    row = next(r for r in rows if r['key'] == a.run_key)
    manifest = json.loads((here/'runtime_records'/a.run_key/'source_manifest.json').read_text(encoding='utf-8'))
    source = here/'source_snapshots'/row['source_signature']
    assert not a.output.exists(), 'Use a fresh output directory'
    baseline = a.baseline_root.resolve(); assert baseline.is_dir()
    for name, sha in manifest['files'].items():
        p = source/name
        assert hashlib.sha256(p.read_bytes()).hexdigest() == sha, name
    a.output.mkdir(parents=True)
    for name in manifest['files']:
        target = a.output/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/name, target)
    for name in ['original_assets', 'experiments', 'models']:
        if (baseline/name).exists(): (a.output/name).symlink_to(baseline/name, target_is_directory=True)
    (a.output/'.project-root').touch()
    for name in ['requirements.txt', 'pyproject.toml']:
        if (baseline/name).exists(): shutil.copy2(baseline/name, a.output/name)
    p = a.output/'codex/plan_v1_1/run_task.py'; original = p.read_text(encoding='utf-8')
    old = "quantizer = Path(qm['checkpoint'].replace('/home/jcyin/', str(Path.home())+'/')).resolve()"
    new = "quantizer = (baseline / 'quantizer_train/checkpoints' / Path(qm['checkpoint']).name).resolve()"
    assert original.count(old) == 1
    adapted = original.replace(old, new).replace("baseline = ROOT / 'experiments' / BASELINES[args.dataset]", "baseline = (ROOT / 'experiments' / BASELINES[args.dataset]).resolve()")
    p.write_text(adapted, encoding='utf-8', newline='\n')
    (a.output/'path_adaptation.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True), adapted.splitlines(True), fromfile='historical/run_task.py', tofile='derived/run_task.py')), encoding='utf-8')
    receipt = {'historical_source_signature': row['source_signature'], 'run_key': a.run_key,
               'baseline_root': str(baseline), 'derived_run_task_sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
               'scope': 'Path adaptation only; not the historical runtime identity. No experiment launched.'}
    (a.output/'assembly_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt))


if __name__ == '__main__': main()
