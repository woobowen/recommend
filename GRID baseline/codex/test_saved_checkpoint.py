"""Evaluate an archived best checkpoint after relocating this repository."""
from pathlib import Path
import argparse
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
OLD_ROOT = '/home/jcyin/GRID_wmh_Qwen_TIGER_20261009'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['beauty', 'sports', 'toys'], required=True)
    parser.add_argument('--gpu', required=True, help='One idle physical GPU index')
    parser.add_argument('--output', type=Path, required=True, help='A new result directory')
    parser.add_argument('--summary', type=Path, default=ROOT.parent / 'Baseline Results/test_results.json')
    parser.add_argument('--validate-only', action='store_true', help='Load model and checkpoint on CPU, without evaluation')
    args = parser.parse_args()
    if not args.gpu.isdigit():
        parser.error('--gpu requires a numeric physical GPU index')
    output = args.output.resolve()
    if output.exists() and not args.validate_only:
        parser.error('--output must be a new directory')
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS='4', TF_CPP_MIN_LOG_LEVEL='3',
                      TF_NUM_INTRAOP_THREADS='2', TF_NUM_INTEROP_THREADS='2')
    for key in ('LOCAL_RANK', 'RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT', 'METADATA_DIR'):
        os.environ.pop(key, None)
    sys.path.insert(0, str(ROOT))
    import rootutils
    rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)
    import torch
    from omegaconf import OmegaConf
    import src.utils.custom_hydra_resolvers
    from src.utils.launcher_utils import pipeline_launcher
    # Importing custom_hydra_resolvers registers the project's resolver names.
    summary = json.loads(args.summary.read_text(encoding='utf-8'))
    relative = summary['datasets'][args.dataset]['checkpoint'].split('GRID baseline/', 1)[1]
    checkpoint = ROOT / relative
    recorded_config = checkpoint.parent.parent / '.hydra/config.yaml'
    text = recorded_config.read_text(encoding='utf-8').replace(OLD_ROOT, ROOT.as_posix())
    cfg = OmegaConf.create(text)
    cfg.paths.root_dir = ROOT.as_posix()
    cfg.paths.output_dir = output.as_posix()
    cfg.paths.log_dir = output.as_posix()
    cfg.paths.work_dir = ROOT.as_posix()
    cfg.paths.profile_dir = (output / 'profile_output').as_posix()
    cfg.paths.metadata_dir = (output / 'metadata').as_posix()
    cfg.ckpt_path = checkpoint.as_posix()
    cfg.callbacks.restart_job = None
    cfg.callbacks.model_checkpoint = None
    cfg.trainer.devices = 1
    cfg.trainer.strategy = 'auto'
    for split in ['train', 'val', 'test']:
        loader = cfg.data_loading[f'{split}_dataloader_config'].dataloader
        loader.num_workers = 0
        loader.timeout = 0
        loader.persistent_workers = False
    if args.validate_only:
        import hydra
        model = hydra.utils.instantiate(cfg.model)
        checkpoint_data = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint_data['state_dict'], strict=True)
        print(json.dumps({'dataset': args.dataset, 'checkpoint': str(checkpoint),
                          'state_dict_strict_load': True, 'checkpoint_step': checkpoint_data['global_step']}))
        return
    output.mkdir(parents=True)
    torch.set_float32_matmul_precision('medium')
    with pipeline_launcher(cfg) as p:
        metrics = p.trainer.test(p.model, datamodule=p.datamodule, ckpt_path=cfg.ckpt_path)
    (output / 'test_result.json').write_text(json.dumps({'checkpoint': str(checkpoint), 'test_metrics': metrics}, indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
