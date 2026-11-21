"""Read-only protocol audit; write evidence without touching model outputs."""
from pathlib import Path
import hashlib
import json
from omegaconf import OmegaConf

ROOT=Path(__file__).resolve().parents[2]
group=ROOT/'codex/first_round_20261104'
evidence=[]
for task in ['T01','T02','T03','T04']:
    status=json.loads((group/f'{task}_status.json').read_text())
    out=Path(status['output'])
    cfg=OmegaConf.load(out/'.hydra/config.yaml')
    provenance=json.loads((out/'input_provenance.json').read_text())
    source=json.loads((out/'source_manifest.json').read_text())
    assert cfg.train and cfg.test and cfg.ckpt_path is None
    assert cfg.seed==42 and cfg.model.eval_seed==2026
    assert cfg.trainer.devices==1 and cfg.trainer.strategy=='auto'
    assert cfg.trainer.precision=='32-true' and cfg.trainer.deterministic
    assert cfg.trainer.max_steps==320000 and cfg.trainer.max_epochs==10
    assert cfg.trainer.accumulate_grad_batches==16 and cfg.trainer.val_check_interval==1600
    assert cfg.callbacks.restart_job is None
    assert cfg.callbacks.model_checkpoint.monitor=='val/recall@5'
    assert cfg.callbacks.model_checkpoint.mode=='max' and cfg.callbacks.model_checkpoint.save_top_k==1
    assert cfg.callbacks.early_stopping.patience==10
    assert cfg.optim.optimizer.lr==0.001 and cfg.optim.optimizer.weight_decay==0.0001
    assert cfg.optim.scheduler is None
    for key in ('limit_train_batches','limit_val_batches','limit_test_batches'):
        assert key not in cfg.trainer or cfg.trainer[key]==1.0
    assert cfg.data_loading.train_dataloader_config.dataloader.batch_size_per_device==16
    assert cfg.data_loading.train_dataloader_config.dataloader.collate_fn.max_batch_size==128
    for split in ('val','test'):
        data=cfg.data_loading[f'{split}_dataloader_config'].dataloader
        assert data.batch_size_per_device==8 and not data.drop_last and not data.should_shuffle_rows
    assert provenance['fit_ckpt_path'] is None and not provenance['preflight_only']
    assert not provenance['full_test_attention_export']
    unchanged=all(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==hash_value
                  for path,hash_value in source['files'].items())
    assert unchanged
    sample=json.loads((out/'sample_manifest.json').read_text())
    assert len(sample['users'])==len({u['user_id'] for u in sample['users']})==100
    evidence.append({'task_id':task,'config_matches_v1_1':True,'runtime_source_unchanged':True,
                     'source_sha256':source['sha256'],'fixed_users':100})
assert len({entry['source_sha256'] for entry in evidence})==1
report={'all_four_passed':True,'tasks':evidence}
(group/'configuration_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
