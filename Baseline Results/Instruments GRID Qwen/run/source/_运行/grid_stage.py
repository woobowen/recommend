"""Invoke the unchanged WMH quantizer/TIGER pipeline and record its outputs."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import rootutils
rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)
import hydra
import torch
from omegaconf import DictConfig
from src.utils.custom_hydra_resolvers import *
from src.utils.launcher_utils import pipeline_launcher
from lightning.pytorch.callbacks import Callback


class ProgressRecorder(Callback):
    def __init__(self, path):
        self.path = path
        self.last_write = 0

    def write(self, trainer, state, **extra):
        value = dict(state=state, time=time.strftime('%Y-%m-%d %H:%M:%S %z'),
                     global_step=trainer.global_step, epoch=trainer.current_epoch, **extra)
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
        temporary.replace(self.path)
        self.last_write = time.monotonic()

    def on_train_start(self, trainer, pl_module):
        self.write(trainer, 'training_started')

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if time.monotonic()-self.last_write < 10:
            return
        loss = outputs.get('loss') if isinstance(outputs, dict) else outputs
        extra = dict(batch_idx=batch_idx)
        if isinstance(loss, torch.Tensor):
            extra.update(loss=float(loss.detach().float().mean().cpu()), loss_finite=bool(torch.isfinite(loss).all()))
        self.write(trainer, 'training', **extra)

    def on_train_end(self, trainer, pl_module):
        self.write(trainer, 'training_completed')

    def on_test_start(self, trainer, pl_module):
        self.write(trainer, 'test_started')

    def on_test_end(self, trainer, pl_module):
        self.write(trainer, 'test_completed')

    def on_exception(self, trainer, pl_module, exception):
        self.write(trainer, 'failed', error=str(exception))


@hydra.main(version_base='1.3', config_path='../configs', config_name='train')
def main(cfg: DictConfig):
    out = Path(cfg.paths.output_dir)
    with pipeline_launcher(cfg) as p:
        p.trainer.callbacks.append(ProgressRecorder(out/'progress.json'))
        if cfg.task_name == 'inference':
            p.trainer.predict(p.model, datamodule=p.datamodule, ckpt_path=cfg.ckpt_path,
                              return_predictions=False)
            result = {'predictions': str(out/'pickle/merged_predictions_tensor.pt')}
        else:
            p.trainer.fit(p.model, datamodule=p.datamodule, ckpt_path=cfg.get('ckpt_path'))
            callback = p.trainer.checkpoint_callback
            checkpoint = callback.best_model_path if callback else ''
            if not checkpoint:
                checkpoint = str(out/'checkpoints/final.ckpt')
                Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
                p.trainer.save_checkpoint(checkpoint)
            result = {'checkpoint': checkpoint, 'global_step': p.trainer.global_step}
            if cfg.get('test', False):
                metrics = p.trainer.test(p.model, datamodule=p.datamodule, ckpt_path=checkpoint)
                result['test_metrics'] = metrics
        (out/'stage_result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    main()
