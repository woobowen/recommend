"""Train a fresh recommendation model on this user's existing baseline SIDs."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASELINES = {
    'beauty': 'beauty_rkmeans_qwen_20261010_012253',
    'sports': 'sports_rkmeans_qwen_20261010_093035',
    'toys': 'toys_rkmeans_qwen_20261010_093036',
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=list(BASELINES), required=True)
    parser.add_argument('--variant', choices=['mlp', 'shuffle'], required=True)
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        parser.error('Output must be a new directory; no implicit checkpoint reuse')
    out.mkdir(parents=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS='4',
                      PYTHONUNBUFFERED='1', TF_CPP_MIN_LOG_LEVEL='3',
                      TF_NUM_INTRAOP_THREADS='2', TF_NUM_INTEROP_THREADS='2')
    for key in ('PROJECT_ROOT', 'METADATA_DIR', 'LOCAL_RANK', 'RANK', 'WORLD_SIZE',
                'MASTER_ADDR', 'MASTER_PORT'):
        os.environ.pop(key, None)
    sys.path.insert(0, str(ROOT))
    import rootutils
    rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)
    import torch
    import lightning as L
    from hydra import compose, initialize_config_dir
    from lightning.pytorch.callbacks import Callback
    from omegaconf import OmegaConf, open_dict
    import src.utils.custom_hydra_resolvers
    from src.utils.launcher_utils import pipeline_launcher

    baseline = ROOT / 'experiments' / BASELINES[args.dataset]
    sid = baseline / 'semantic_ids.pt'
    vectors = baseline / 'item_embeddings.pt'
    metadata = json.loads(vectors.with_suffix('.json').read_text())
    quantizer_info = json.loads((baseline / 'quantizer_train/stage_result.json').read_text())
    quantizer = Path(quantizer_info['checkpoint']).resolve()
    assert quantizer.is_relative_to(baseline), 'Quantizer must be from this baseline run'
    assert metadata['dataset'] == args.dataset and metadata['architecture'] == 'qwen2'
    assert digest(vectors) == metadata['sha256'], 'Baseline vector hash mismatch'
    ids = torch.load(sid, map_location='cpu', weights_only=False)
    assert list(ids.shape) == [4, metadata['shape'][0]]
    assert ids.min() >= 0 and ids.max() < 256
    assert torch.unique(ids.T, dim=0).size(0) == ids.size(1)
    provenance = {
        'dataset': args.dataset, 'variant': args.variant, 'seed': 42,
        'recommendation_fit_checkpoint': None, 'train': True, 'test': True,
        'upstream_reused_from_own_baseline': str(baseline),
        'inputs': {name: {'path': str(path), 'sha256': digest(path)}
                   for name, path in [('text_vectors', vectors), ('quantizer', quantizer), ('semantic_ids', sid)]},
        'sid_shape': list(ids.shape), 'source_commit': '800b5a49e97b96aa4d7ca64bcdf83a7f91e81f1c',
        'shuffle_shared_embedding_table': args.variant == 'shuffle',
        'smoke_only': args.smoke,
    }
    write_json(out / 'input_provenance.json', provenance)
    del ids

    overrides = [
        'experiment=tiger_train_flat', f'data_dir={ROOT}/original_assets/data/amazon_data/{args.dataset}',
        f'semantic_id_path={sid}', f'paths.root_dir={ROOT}', f'paths.output_dir={out}',
        f'paths.work_dir={ROOT}', f'paths.profile_dir={out}/profile_output',
        f'paths.metadata_dir={out}/metadata', 'num_hierarchies=4',
        f'model._target_=codex.sid_order_launch.sid_order_{args.variant}_model.SemanticIDEncoderDecoder',
        'trainer.devices=1', '++trainer.strategy=auto', 'trainer.precision=32-true',
        'trainer.val_check_interval=1600', 'trainer.log_every_n_steps=10',
        '++trainer.enable_progress_bar=false', 'train=true', 'test=true', 'ckpt_path=null',
        '++callbacks.restart_job=null', 'extras.print_config=false', 'seed=42',
        '++model.top_k_for_generation=10',
    ]
    for split in ('train', 'val', 'test'):
        prefix = f'data_loading.{split}_dataloader_config.dataloader.'
        overrides += [f'++{prefix}num_workers=0', f'++{prefix}timeout=0', f'++{prefix}persistent_workers=false']
    if args.variant == 'shuffle':
        overrides += ['++model.shuffle_embedding_on_test=false', '++model.hierarchies_to_shuffle=3',
                      '++model.save_attentions_on_test=true',
                      '++model.save_attentions_on_predict=false',
                      f'++model.attention_save_dir={out}/attention_weights']
    if args.smoke:
        overrides += ['trainer.max_steps=2', 'trainer.max_epochs=1', 'trainer.accumulate_grad_batches=1',
                      'trainer.val_check_interval=2', '++trainer.limit_train_batches=2',
                      '++trainer.limit_val_batches=2', '++trainer.limit_test_batches=2',
                      'data_loading.train_dataloader_config.dataloader.batch_size_per_device=2',
                      'data_loading.train_dataloader_config.dataloader.collate_fn.max_batch_size=4',
                      'data_loading.val_dataloader_config.dataloader.batch_size_per_device=2',
                      'data_loading.test_dataloader_config.dataloader.batch_size_per_device=2']
    with initialize_config_dir(config_dir=str(ROOT / 'configs'), version_base='1.3'):
        cfg = compose(config_name='train', overrides=overrides)
    # Resolve filesystem fields explicitly because this runner is not a Hydra CLI task.
    with open_dict(cfg):
        cfg.paths.output_dir = str(out)
        cfg.paths.profile_dir = str(out / 'profile_output')
        cfg.paths.metadata_dir = str(out / 'metadata')
        cfg.trainer.default_root_dir = str(out)
        cfg.callbacks.model_checkpoint.dirpath = str(out / 'checkpoints')
        cfg.logger.csv.save_dir = str(out)
    (out / '.hydra').mkdir()
    OmegaConf.save(cfg, out / '.hydra/config.yaml')
    write_json(out / '.hydra/overrides.json', overrides)
    torch.set_float32_matmul_precision('medium')

    class Recorder(Callback):
        def __init__(self):
            self.last = 0
            self.test_mode = None
        def record(self, trainer, state, **values):
            write_json(out / 'progress.json', dict(state=state, time=time.time(),
                       global_step=trainer.global_step, epoch=trainer.current_epoch,
                       test_mode=self.test_mode, **values))
        def on_train_start(self, trainer, pl_module):
            assert trainer.global_step == 0, 'Production training must start at step zero'
            self.record(trainer, 'training_started')
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if time.monotonic() - self.last < 10:
                return
            loss = outputs.get('loss') if isinstance(outputs, dict) else outputs
            value = float(loss.detach().float().cpu())
            if not math.isfinite(value):
                raise ValueError('Non-finite training loss')
            self.record(trainer, 'training', loss=value, loss_finite=True, batch_idx=batch_idx)
            self.last = time.monotonic()
        def on_train_end(self, trainer, pl_module):
            self.record(trainer, 'training_completed')
        def on_test_start(self, trainer, pl_module):
            self.record(trainer, 'test_started')
        def on_test_end(self, trainer, pl_module):
            self.record(trainer, 'test_completed')
        def on_exception(self, trainer, pl_module, exception):
            self.record(trainer, 'failed', error=str(exception))

    recorder = Recorder()
    with pipeline_launcher(cfg) as p:
        p.trainer.callbacks.append(recorder)
        if args.variant == 'mlp':
            assert hasattr(p.model, 'item_aggregation_mlp') and p.model.sep_token is None
        else:
            assert p.model.get_embedding_table('encoder') is p.model.get_embedding_table('decoder')
            p.model.shuffle_calls = 0
            import types
            original_shuffle = p.model._shuffle_hierarchical_embeddings
            def counted_shuffle(self, **kwargs):
                self.shuffle_calls += 1
                return original_shuffle(**kwargs)
            p.model._shuffle_hierarchical_embeddings = types.MethodType(counted_shuffle, p.model)
        write_json(out / 'model_verified.json', {
            'class': type(p.model).__module__ + '.' + type(p.model).__name__,
            'parameter_count': sum(v.numel() for v in p.model.parameters()),
            'ckpt_path': cfg.ckpt_path, 'test': cfg.test,
            'trainer_max_steps': p.trainer.max_steps, 'trainer_max_epochs': p.trainer.max_epochs,
            'val_check_interval': p.trainer.val_check_interval,
            'save_attentions_on_test': getattr(p.model, 'save_attentions_on_test', False),
            'save_attentions_on_predict': getattr(p.model, 'save_attentions_on_predict', False),
            'attention_save_dir': getattr(p.model, 'attention_save_dir', None),
            'source_sha256': digest(Path(__file__).with_name(f'sid_order_{args.variant}_model.py')),
        })
        p.trainer.fit(p.model, datamodule=p.datamodule, ckpt_path=None)
        if args.variant == 'shuffle':
            assert p.model.shuffle_calls == 0, 'No shuffle allowed during training or validation'
        checkpoint = Path(p.trainer.checkpoint_callback.best_model_path).resolve()
        assert checkpoint.is_file() and checkpoint.is_relative_to(out)
        result = {'checkpoint': str(checkpoint), 'global_step': p.trainer.global_step,
                  'trained_from_scratch': True, 'test_enabled': True, 'test_metrics': {}}
        modes = [('mlp', False, 0)] if args.variant == 'mlp' else [
            ('unshuffled', False, 3), ('shuffle_3', True, 3), ('shuffle_4', True, 4)]
        for mode, enabled, count in modes:
            recorder.test_mode = mode
            L.seed_everything(42, workers=True)
            before = getattr(p.model, 'shuffle_calls', 0)
            if args.variant == 'shuffle':
                p.model.shuffle_embedding_on_test = enabled
                p.model.hierarchies_to_shuffle = count
            metrics = p.trainer.test(p.model, datamodule=p.datamodule, ckpt_path=str(checkpoint))
            if args.variant == 'shuffle':
                calls = p.model.shuffle_calls - before
                assert (calls > 0) == enabled, f'Shuffle activation mismatch for {mode}'
                attention_dir = out / 'attention_weights/test' / mode
                attention_files = list(attention_dir.glob('*.pt.gz'))
                assert attention_files, f'Attention files missing for {mode}'
                import gzip
                matrices = sorted(attention_dir.glob('cross_attention_full_batch_*.pt.gz'))
                assert matrices, f'Cross attention matrices missing for {mode}'
                with gzip.open(matrices[0], 'rb') as stream:
                    saved_matrix = torch.load(stream, map_location='cpu', weights_only=False)
                assert saved_matrix.ndim == 4 and saved_matrix.shape[1] == 4
                assert torch.isfinite(saved_matrix).all()
            else:
                calls = 0
                attention_files = []
            assert metrics and all(math.isfinite(float(x)) for x in metrics[0].values())
            result['test_metrics'][mode] = metrics
            write_json(out / f'test_{mode}.json', {'checkpoint': str(checkpoint), 'metrics': metrics,
                       'shuffle_embedding_on_test': enabled, 'hierarchies_to_shuffle': count,
                       'shuffle_calls': calls, 'seed': 42, 'full_test_set': not args.smoke,
                       'attention_file_count': len(attention_files)})
            write_json(out / 'stage_result.json', result)
        recorder.record(p.trainer, 'completed')
        print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
