"""Create a separate portable downstream runtime from exact archived sources/assets."""
from pathlib import Path
import argparse, difflib, hashlib, json, shutil

HERE = Path(__file__).resolve().parent
OLD_ROOT = '/home/jcy/GRID_wmh_Qwen_TIGER_20261009'
RUN_ID = 'instruments_grid_qwen_20261125_005542'


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--assets', required=True, type=Path, help='Extracted instruments-grid-original-assets directory')
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    args.output = args.output.resolve()
    assert not args.output.exists(), 'Choose a new output directory'
    assets = args.assets.resolve()
    spec = json.loads((HERE / 'assets.json').read_text())['assets']['instruments-grid-original-assets.zip']
    expected = {r['member']: r['sha256'] for r in spec['members']}
    needed = ['upstream/semantic_ids.pt', 'run/formal/checkpoints/checkpoint_epoch=000_step=006100.ckpt']
    needed += ['run/data/' + split + '/part-00000.tfrecord.gz' for split in ['training', 'evaluation', 'testing']]
    for rel in needed:
        assert hashlib.sha256((assets / rel).read_bytes()).hexdigest() == expected[rel], rel
    src = HERE / 'run/source'
    manifest = json.loads((HERE / 'run/source_manifest.json').read_text(encoding='utf8'))
    for rel, h in manifest.items():
        assert hashlib.sha256((HERE / 'run' / rel).read_bytes()).hexdigest() == h, rel
    shutil.copytree(src, args.output)
    for rel in needed:
        dest = args.output / ('data/' + rel.removeprefix('run/data/') if rel.startswith('run/data/')
                              else 'best.ckpt' if rel.endswith('.ckpt') else 'semantic_ids.pt')
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(assets / rel, dest)
    original = (HERE / 'run/formal/.hydra/config.yaml').read_text(encoding='utf8')
    text = original.replace(OLD_ROOT + '/experiments/' + RUN_ID + '/source', args.output.as_posix())
    text = text.replace(OLD_ROOT + '/experiments/' + RUN_ID + '/data', (args.output / 'data').as_posix())
    old_sid = json.loads((HERE / 'run/data_audit.json').read_text())['sid_path']
    text = text.replace(old_sid, (args.output / 'semantic_ids.pt').as_posix())
    (args.output / 'configs/instruments_reproduction.yaml').write_text(text, encoding='utf8')
    # The new evaluator calls test only; the historical grid_stage.py still calls fit first.
    evaluator = '''from pathlib import Path
import sys, json
sys.path.insert(0,str(Path(__file__).resolve().parent))
import rootutils
rootutils.setup_root(__file__,indicator='.project-root',pythonpath=True)
import hydra,torch
from src.utils.custom_hydra_resolvers import *
from src.utils.launcher_utils import pipeline_launcher
@hydra.main(version_base='1.3',config_path='configs',config_name='instruments_reproduction')
def main(cfg):
    assert cfg.ckpt_path, 'Pass ckpt_path=/absolute/path/best.ckpt'
    with pipeline_launcher(cfg) as p:
        metrics=p.trainer.test(p.model,datamodule=p.datamodule,ckpt_path=cfg.ckpt_path)
        (Path(cfg.paths.output_dir)/'reevaluation_result.json').write_text(json.dumps(metrics,indent=2))
if __name__=='__main__':
    torch.set_float32_matmul_precision('medium')
    main()
'''
    (args.output / 'evaluate_checkpoint.py').write_text(evaluator, encoding='utf8')
    receipt = {'source': 'Exact frozen run/source; originals in the publication stay unchanged',
               'adaptations': 'New instruments_reproduction config maps root/data/SID paths only; separate test-only evaluator',
               'config_diff': ''.join(difflib.unified_diff(original.splitlines(True), text.splitlines(True),
                                                          fromfile='historical', tofile='portable')),
               'scope': 'Assembly and file hash checks only; no training/evaluation execution'}
    (args.output / 'assembly_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf8')
    print(json.dumps({'runtime': str(args.output), 'selected_assets_verified': len(needed)}))


if __name__ == '__main__':
    main()
