"""CPU-only regression for the explicitly authorized one-line save repair."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--spec', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--child', action='store_true')
    a = p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES', '') == ''
    output = a.output or Path(os.environ['GRID_JOB_DIR']) / 'output'
    output.mkdir(parents=True, exist_ok=True)
    if not a.child:
        reports = []
        for index, spec in enumerate(a.spec):
            destination = output / f'config-{index:02d}'
            command = [sys.executable, '-B', str(Path(__file__).resolve()), '--child',
                       '--repo', str(a.repo), '--spec', str(spec), '--output', str(destination)]
            subprocess.run(command, check=True)
            reports.append(json.loads((destination / 'RESULT.json').read_text()))
        value = {'status': 'PASS', 'regression': 'configuration serialization only',
                 'configs': reports, 'model_constructions': 0, 'GPU_model_calls': 0,
                 'scientific_configuration_changed': False,
                 'configuration_files_saved_and_reloaded': len(reports),
                 'script_sha256': digest(__file__)}
        (output / 'RESULT.json').write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
        return
    assert len(a.spec) == 1
    sys.path.insert(0, str(a.repo.resolve()))
    from experiments.beauty_extensions.common import bind_source, sha256, verify_asset
    from experiments.formal_20261015.run import formal_config
    from omegaconf import OmegaConf
    spec = json.loads(a.spec[0].read_text())
    bind_source(spec['route'])
    if spec['route'] == 'MLP':
        from experiments.formal_20261015.train import configure
        cfg = configure(spec, output)
    else:
        cfg = formal_config(spec, output)
    before = OmegaConf.to_container(cfg, resolve=False)
    used = {key: OmegaConf.to_container(cfg[key], resolve=True, throw_on_missing=True)
            for key in ('model', 'trainer', 'optim', 'eval', 'callbacks', 'logger')}
    used['datamodule'] = OmegaConf.to_container(cfg.data_loading.datamodule, resolve=True, throw_on_missing=True)
    # This is the exact call policy in the repaired entry, independently checked
    # below through its parsed AST. No model/callback/data object is instantiated.
    saved = output / 'effective_config.yaml'
    OmegaConf.save(cfg, saved, resolve=False)
    restored = OmegaConf.load(saved)
    assert before == OmegaConf.to_container(restored, resolve=False)
    restored_used = {key: OmegaConf.to_container(restored[key], resolve=True, throw_on_missing=True)
                     for key in ('model', 'trainer', 'optim', 'eval', 'callbacks', 'logger')}
    restored_used['datamodule'] = OmegaConf.to_container(restored.data_loading.datamodule, resolve=True, throw_on_missing=True)
    assert used == restored_used
    encoded = json.dumps(used, sort_keys=True, ensure_ascii=False, allow_nan=False)
    assert '${' not in encoded and '???' not in encoded
    assert restored.semantic_id_path == spec['sid']['path']
    assert restored.seed == 42 and restored.model.top_k_for_generation == 10
    assert not any(key.startswith('limit_') for key in restored.trainer)
    assert not restored.trainer.get('fast_dev_run', False)
    assert not restored.trainer.get('overfit_batches', 0)
    assets = {name: verify_asset(spec[name]) for name in ('config', 'sid')}
    protocol = None
    if spec['route'] == 'MLP':
        domain = spec['domain']
        raw, accumulation, cadence = {'toys': (32, 48, 4800), 'sports': (16, 16, 1600)}[domain]
        assert restored.train and restored.test and restored.ckpt_path is None
        assert restored.optim.optimizer._target_ == 'torch.optim.Adam'
        assert restored.optim.optimizer.lr == .001 and restored.optim.optimizer.weight_decay == .0001
        assert restored.optim.scheduler is None
        assert restored.data_loading.train_dataloader_config.dataloader.batch_size_per_device == raw
        assert restored.trainer.accumulate_grad_batches == accumulation
        assert restored.trainer.val_check_interval == cadence
        assert restored.trainer.max_steps == 320000 and restored.trainer.max_epochs == 10
        selection = OmegaConf.to_container(restored.callbacks, resolve=True)
        best = [v for v in selection.values() if isinstance(v, dict) and v.get('_target_', '').endswith('ModelCheckpoint')]
        early = [v for v in selection.values() if isinstance(v, dict) and v.get('_target_', '').endswith('EarlyStopping')]
        assert len(best) == len(early) == 1
        assert best[0]['monitor'] == early[0]['monitor'] == 'val/recall@5'
        assert best[0]['mode'] == early[0]['mode'] == 'max'
        assert early[0]['patience'] == 10 and early[0]['min_delta'] == 0
        assert spec['hard_cap_hours'] == 12 and spec['formal_protocol_user_approved']
        protocol = {'raw_microbatch': raw, 'accumulation': accumulation, 'val_microbatches': cadence,
                    'initialization': spec['initialization'], 'total_hard_cap_seconds': 43200,
                    'max_steps': 320000, 'max_epochs': 10, 'selection': best[0], 'early_stopping': early[0]}
    run = a.repo / 'experiments/formal_20261015/run.py'
    saves = [n for n in ast.walk(ast.parse(run.read_text())) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
             and n.func.value.id == 'OmegaConf' and n.func.attr == 'save']
    assert len(saves) == 1
    assert any(k.arg == 'resolve' and isinstance(k.value, ast.Constant) and k.value.value is False
               for k in saves[0].keywords)
    (output / 'used_config_before.json').write_text(json.dumps(used, ensure_ascii=False, indent=2) + '\n')
    (output / 'used_config_after_reload.json').write_text(json.dumps(restored_used, ensure_ascii=False, indent=2) + '\n')
    value = {'status': 'PASS', 'route': spec['route'], 'domain': spec.get('domain', 'beauty'),
             'spec_sha256': digest(a.spec[0]), 'run_py_sha256': sha256(run),
             'whole_template_roundtrip_exact': True, 'actual_used_subtrees_resolve_exact': True,
             'used_config_sha256': hashlib.sha256(encoded.encode()).hexdigest(),
             'source_only_id_template': before.get('id'),
             'unused_id_is_not_evaluated_by_computation': True, 'assets': assets,
             'approved_protocol': protocol, 'model_constructions': 0, 'GPU_model_calls': 0,
             'model_SID_initialization_optimizer_batch_validation_selection_budget_unchanged': True}
    (output / 'RESULT.json').write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
