"""Exercise real Lightning test hooks and predict output on a private smoke model."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2',
                      TF_CPP_MIN_LOG_LEVEL='3', TF_NUM_INTRAOP_THREADS='1',
                      TF_NUM_INTEROP_THREADS='1')
    sys.path.insert(0, str(ROOT))
    import torch
    import lightning as L
    from omegaconf import OmegaConf
    import src.utils.custom_hydra_resolvers
    from src.utils.launcher_utils import pipeline_launcher

    torch.set_num_threads(2)
    source = args.source
    cfg = OmegaConf.load(source / '.hydra/config.yaml')
    cfg.id = 'attention-switch-verification'
    checkpoint = json.loads((source / 'stage_result.json').read_text())['checkpoint']
    cfg.paths.output_dir = str(out)
    cfg.paths.profile_dir = str(out / 'profile_output')
    cfg.paths.metadata_dir = str(out / 'metadata')
    cfg.callbacks.model_checkpoint = None
    cfg.callbacks.early_stopping = None
    cfg.callbacks.model_summary = None
    cfg.trainer.accelerator = 'cpu'
    cfg.trainer.devices = 1
    cfg.trainer.default_root_dir = str(out)
    cfg.trainer.enable_model_summary = False
    cfg.logger.csv.save_dir = str(out)
    cfg.ckpt_path = None
    cfg.model.save_attentions_on_test = False
    cfg.model.attention_save_dir = str(out / 'attention_weights')
    report = {'uses_private_smoke_checkpoint': checkpoint, 'production_jobs_interrupted': False}
    with pipeline_launcher(cfg) as p:
        model = p.model
        data = p.datamodule
        trainer = p.trainer
        class Capture(L.Callback):
            def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
                self.batch = batch[0]
        capture = Capture()
        trainer.callbacks.append(capture)
        L.seed_everything(42, workers=True)
        disabled = trainer.test(model, datamodule=data, ckpt_path=checkpoint)
        assert not (out / 'attention_weights').exists()
        report['test_disabled_writes_nothing'] = True

        model.save_attentions_on_test = True
        L.seed_everything(42, workers=True)
        enabled = trainer.test(model, datamodule=data, ckpt_path=checkpoint)
        assert disabled == enabled, 'Attention recording changed unshuffled metrics'
        attention = out / 'attention_weights/test/unshuffled'
        for batch in (0, 1):
            import gzip
            with gzip.open(attention / f'cross_attention_full_batch_{batch}.pt.gz', 'rb') as stream:
                matrix = torch.load(stream, weights_only=False)
            assert list(matrix.shape[:3]) == [2, 4, 6]
            assert torch.isfinite(matrix).all() and matrix.device.type == 'cpu'
        report['test_enabled_saves_distinct_batch_indices'] = True
        report['cross_attention_shape'] = list(matrix.shape)
        report['recording_preserves_test_metrics'] = True

        trainer.validate(model, datamodule=data, ckpt_path=checkpoint)
        assert len(list((out / 'attention_weights').rglob('*.pt.gz'))) == 8
        report['validation_does_not_save_test_attentions'] = True

        model.shuffle_embedding_on_test = True
        model.hierarchies_to_shuffle = 3
        trainer.test(model, datamodule=data, ckpt_path=checkpoint)
        assert (out / 'attention_weights/test/shuffle_3/cross_attention_full_batch_0.pt.gz').is_file()
        report['test_modes_save_to_separate_directories'] = True

        batch = capture.batch
        # Test collate does not retain prediction writer keys. Supply keys
        # for this isolated predict_step check without changing model inputs.
        batch.user_id_list = list(range(batch.mask.size(0)))
        model.eval()
        with torch.inference_mode():
            model.save_attentions_on_predict = False
            prediction_off = model.predict_step(batch, batch_idx=5)
            assert not (out / 'attention_weights/predict').exists()
            model.save_attentions_on_predict = True
            prediction_on = model.predict_step(batch, batch_idx=5)
        assert torch.equal(prediction_off.predictions, prediction_on.predictions)
        assert (out / 'attention_weights/predict/default/cross_attention_full_batch_5.pt.gz').is_file()
        report['predict_has_independent_switch'] = True
        report['recording_preserves_predictions'] = True

    (out / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
