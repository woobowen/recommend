#!/usr/bin/env python3
"""A1 native Qwen3 Int8/BF16 LETTER training with complete strict recovery.

Bounded acceptance uses this same four-epoch scheduler and native loader;
callbacks stop computation without shortening the scheduler horizon.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata as md
import inspect
import json
import math
import os
from pathlib import Path
import random
import sys
import time

from checkpoint import (ReplayLoader, cpu_clone, int8_dynamic_state, rng_capture,
                        seal_checkpoint, sha, strict_hot_int8_alias,
                        verify_checkpoint, write)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['letter', 'model', 'output', 'config', 'run-commit']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--resume')
    parser.add_argument('--probe-stop', type=int, default=0)
    parser.add_argument('--probe-save', type=int, default=2)
    parser.add_argument('--preflight-interfaces', action='store_true')
    args = parser.parse_args()
    assert args.probe_stop in (0, 12)
    assert args.probe_save == 2
    assert not args.preflight_interfaces or args.probe_stop == 12
    assert len(args.run_commit) == 40 and all(c in '0123456789abcdef' for c in args.run_commit)
    config = json.loads(Path(args.config).read_text())
    expected = json.loads((Path(__file__).parents[1] / 'config/formal_instruments_seed42.json').read_text())
    assert config == expected, 'Unapproved formal config drift'
    config_sha = sha(args.config)
    rank, local, world = (int(os.environ[key]) for key in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])
    assert world == 2
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / f'rank{rank}.json'
    if report_path.exists():
        assert args.resume, 'Existing run requires explicit same-seed checkpoint resume'
        saved_report = output / 'attempt_reports' / f'{sha(report_path)[:16]}-rank{rank}.json'
        saved_report.parent.mkdir(parents=True, exist_ok=True)
        if saved_report.exists():
            assert saved_report.read_bytes() == report_path.read_bytes()
        else:
            saved_report.write_bytes(report_path.read_bytes())
    started = time.monotonic()
    events = (output / f'events.rank{rank}.jsonl').open('a', buffering=1)

    def event(kind, **fields):
        record = {'event': kind, 'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                  'elapsed_seconds': time.monotonic() - started,
                  'attempt_job_id': os.environ.get('GRID_JOB_ID'), **fields}
        events.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        if kind != 'batch':
            print(json.dumps({'rank': rank, **record}, ensure_ascii=False, allow_nan=False), flush=True)

    import numpy as np
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from transformers import Trainer, TrainerCallback, TrainingArguments
    import transformers.trainer as trainer_module
    from bitsandbytes.nn import Linear8bitLt
    from deepspeed import DeepSpeedEngine
    from accelerate.data_loader import DataLoaderShard
    from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict, prepare_model_for_kbit_training
    sys.path.insert(0, str(Path(__file__).parents[2] / 'qwen_preflight_20261110/scripts'))
    from common import Collator, TARGETS, datasets, load_model, loss as author_loss, sharing, tokenizer_for
    torch.cuda.set_device(local)
    dist.init_process_group('nccl', timeout=dt.timedelta(minutes=15))
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    versions = {name: md.version(name) for name in ['torch', 'transformers', 'peft', 'accelerate', 'deepspeed', 'bitsandbytes']}
    assert versions['transformers'] == '4.57.6' and versions['peft'] == '0.21.2'
    assert versions['accelerate'] == '1.12.0' and versions['deepspeed'] == '0.16.9'
    assert versions['bitsandbytes'] == '0.45.5' and versions['torch'].split('+')[0] == '2.6.0'
    report = {'status': 'INITIALIZING', 'rank': rank, 'local_rank': local, 'world_size': world,
              'argv': sys.argv, 'cwd': os.getcwd(), 'python': sys.executable,
              'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'run_commit': args.run_commit, 'config_sha256': config_sha, 'versions': versions,
              'probe': bool(args.probe_stop), 'config': config,
              'gpu': {'name': torch.cuda.get_device_name(local),
                      'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                      'total_bytes': torch.cuda.get_device_properties(local).total_memory},
              'runtime_sources': {name: {'path': inspect.getsourcefile(callable_),
                                        'sha256': sha(inspect.getsourcefile(callable_))}
                                  for name, callable_ in [('Trainer', Trainer), ('Linear8bitLt', Linear8bitLt),
                                                         ('DeepSpeedEngine', DeepSpeedEngine),
                                                         ('DataLoaderShard', DataLoaderShard)]}}
    write(report_path, report)
    try:
        train, valid, test = datasets(args.letter)
        tokenizer, tokenmap = tokenizer_for(args.model, train.get_new_tokens())
        assert tokenmap['original_tokenizer_size'] == 151669 and len(tokenizer) == 152587
        assert len(train.prompts) == 17 and len(train.indices) == 9922
        for sid in train.indices.values():
            text = ''.join(sid)
            ids = tokenizer.encode(text, add_special_tokens=False)
            assert len(ids) == 4 and tokenizer.decode(ids, skip_special_tokens=False) == text
        if rank == 0:
            tokenizer.save_pretrained(output)
            from transformers import AutoTokenizer
            restored = AutoTokenizer.from_pretrained(output, local_files_only=True)
            assert restored.get_vocab() == tokenizer.get_vocab()
        dist.barrier()
        report['tokenizer'] = tokenmap
        report['data'] = {'train_rows': len(train), 'valid_rows': len(valid), 'test_users': len(test),
                          'catalog_items': len(train.indices), 'unique_catalog_sids': len(train.get_all_items()),
                          'validation_text_sha256': hashlib.sha256(json.dumps(valid.valid_text_data,
                              ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()}
        assert report['data']['unique_catalog_sids'] == 9897
        model_root = Path(args.model)
        assert model_root.name == config['model_revision']
        weight = model_root / 'model.safetensors'
        assert weight.stat().st_size == config['model_weight_bytes']
        # The prelaunch asset audit supplies full payload SHA verification. Each
        # GPU process checks the pinned path, manifest and sizes without rereading
        # the 1.2 GB model payload twice at every invocation.
        manifest = json.loads((model_root / 'ASSET_MANIFEST.json').read_text())
        assert manifest['revision'] == config['model_revision'] and manifest['model_id'] == config['model_id']
        report['model_asset'] = {'path': str(model_root.resolve()), 'revision': config['model_revision'],
                                'config_sha256': sha(model_root / 'config.json'),
                                'manifest_sha256': sha(model_root / 'ASSET_MANIFEST.json'),
                                'weight_bytes': weight.stat().st_size,
                                'weight_mtime_ns': weight.stat().st_mtime_ns,
                                'expected_weight_sha256': config['model_weight_sha256']}
        event('load_model')
        model, loading = load_model(args.model, 'qwen3', True, torch.device('cuda', local))
        assert type(model).__name__ == 'Qwen3ForCausalLM'
        assert model.config.num_hidden_layers == 28 and model.config._attn_implementation == 'sdpa'
        report['model_class'] = type(model).__module__ + '.' + type(model).__name__
        report['model_source_sha256'] = sha(inspect.getsourcefile(type(model)))
        report['strict_official_loading_info'] = loading
        assert sharing(model)['same_parameter_object'] and model.get_input_embeddings().weight.shape[0] == 151936
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        assert sharing(model)['same_parameter_object']
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={'use_reentrant': False})
        targets = [name for name, module in model.named_modules() if name.rsplit('.', 1)[-1] in TARGETS]
        assert len(targets) == 196 and TARGETS == config['lora_target_modules']
        model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=.05,
                    target_modules=TARGETS, modules_to_save=['embed_tokens', 'lm_head'],
                    ensure_weight_tying=True, bias='none', task_type=TaskType.CAUSAL_LM))
        for name, parameter in model.named_parameters():
            if '.original_module.' in name:
                parameter.requires_grad = False
        model.config.use_cache = False
        model.config.pad_token_id = tokenizer.pad_token_id
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        named = dict(model.named_parameters())
        trainable = {name: parameter for name, parameter in named.items() if parameter.requires_grad}
        assert len(trainable) == 393 and sum(parameter.numel() for parameter in trainable.values()) == 161295360
        assert sum(parameter.numel() for name, parameter in trainable.items() if 'lora_' in name) == 5046272
        assert sum('lora_' in name for name in trainable) == 392
        assert all(parameter.requires_grad == ('lora_' in name or '.modules_to_save.default.' in name)
                   for name, parameter in named.items())
        tied = model.get_input_embeddings().weight
        assert tied is model.get_output_embeddings().weight and tied.requires_grad and tied.numel() == 156249088
        report['sharing'] = sharing(model)
        report['lora'] = {'target_names': targets, 'targets': 196, 'lora_tensors': 392,
                          'lora_parameters': 5046272, 'active_tied_parameters': tied.numel(),
                          'independent_trainable_objects': 393, 'independent_trainable_parameters': 161295360,
                          'config': model.peft_config['default'].to_dict()}
        report['parameter_inventory'] = {name: {'shape': list(parameter.shape), 'dtype_before_ds': str(parameter.dtype),
                                               'requires_grad': parameter.requires_grad} for name, parameter in named.items()}
        ds_config = json.loads((Path(args.config).parent / 'ds_z2_bf16.json').read_text())
        assert ds_config['zero_optimization']['allgather_bucket_size'] == config['deepspeed_communication_bucket_elements'] == 50000000
        assert ds_config['zero_optimization']['reduce_bucket_size'] == 50000000
        training_args = TrainingArguments(output_dir=str(output), seed=42, data_seed=42,
            per_device_train_batch_size=8, per_device_eval_batch_size=8,
            gradient_accumulation_steps=4, num_train_epochs=4, learning_rate=2e-4,
            weight_decay=.01, warmup_ratio=.01, lr_scheduler_type='cosine', optim='adamw_torch',
            bf16=True, fp16=False, gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant': False},
            eval_strategy='epoch', save_strategy='epoch', eval_delay=1,
            save_total_limit=2, load_best_model_at_end=True, metric_for_best_model='eval_loss',
            greater_is_better=False, save_safetensors=False,
            deepspeed=ds_config, ddp_find_unused_parameters=False,
            dataloader_num_workers=0, dataloader_drop_last=False, dataloader_pin_memory=True,
            remove_unused_columns=False, prediction_loss_only=True,
            report_to=[], logging_steps=10, disable_tqdm=True, local_rank=local,
            accelerator_config={'use_seedable_sampler': True})
        report['training_arguments'] = training_args.to_dict()
        report['loss_scaling'] = {'model_accepts_loss_kwargs': False, 'compute_loss_func': None,
            'micro_loss': 'FP32 full-vocabulary causal CE(logits/0.8), mean over five labels per row',
            'Trainer': 'divide once by current_gradient_accumulation_steps=4',
            'DeepSpeed': 'scale_wrt_gas=False in native Transformers 4.57.6 training_step'}
        saved_loader = None
        if args.resume:
            verify_checkpoint(args.resume, args.run_commit, config_sha)
            saved_loader = torch.load(Path(args.resume) / f'loader_state_{rank}.pth', map_location='cpu', weights_only=False)
        old_skip = trainer_module.skip_first_batches

        def replay_skip(loader, num_batches=0):
            if isinstance(loader, ReplayLoader):
                assert loader.saved is not None and loader.saved['yielded'] == num_batches
                return loader
            return old_skip(loader, num_batches)

        trainer_module.skip_first_batches = replay_skip
        trainer = None

        def strict_ds_load(engine, path, load_module_strict=True):
            meta = verify_checkpoint(path, args.run_commit, config_sha)
            event('strict_ds_restore_begin', path=str(path), native_hf_requested_strict=load_module_strict,
                  enforced_strict=True, checkpoint_step=meta['global_step'])
            with strict_hot_int8_alias(engine.module, event):
                loaded, client = engine.load_checkpoint(str(path), load_module_strict=True,
                                load_optimizer_states=True, load_lr_scheduler_states=True)
            assert loaded and engine.global_steps == meta['global_step']
            scheduler = torch.load(Path(path) / 'external_scheduler.pth', map_location='cpu', weights_only=False)
            trainer.lr_scheduler.load_state_dict(scheduler)
            assert sharing(engine.module)['same_parameter_object']
            event('strict_ds_restored', path=str(path), global_step=engine.global_steps,
                  scheduler=trainer.lr_scheduler.state_dict(), client_keys=sorted(client))

        trainer_module.deepspeed_load_checkpoint = strict_ds_load
        first_ce = True
        epoch_started = None
        gradient_records, handles = {}, []
        initial_probe = {}
        if args.probe_stop:
            for name, parameter in trainable.items():
                gradient_records[name] = {'calls': 0, 'nonzero_calls': 0, 'max_abs': 0., 'finite': True}
                def observe(gradient, name=name, parameter=parameter):
                    row = gradient_records[name]
                    row['calls'] += 1
                    finite = bool(torch.isfinite(gradient).all())
                    maximum = float(gradient.abs().max())
                    row['finite'] &= finite
                    row['max_abs'] = max(row['max_abs'], maximum)
                    row['nonzero_calls'] += int(maximum > 0)
                    assert finite, name
                    if parameter is tied:
                        values = gradient[tokenmap['sid_ids']].abs().amax(dim=1)
                        row['all_918_sid_rows_have_gradient'] = bool((values > 0).all())
                    return gradient
                handles.append(parameter.register_hook(observe))

        class QwenTrainer(Trainer):
            _writing_checkpoint = False

            def compute_loss(self, current, inputs, return_outputs=False, num_items_in_batch=None):
                nonlocal first_ce
                assert num_items_in_batch is None and self.model_accepts_loss_kwargs is False
                labels = inputs['labels']
                outputs = current(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], use_cache=False)
                value = author_loss(outputs.logits, labels)
                assert torch.isfinite(value).all()
                if first_ce:
                    with torch.no_grad():
                        shifted = labels[:, 1:]
                        mask = shifted != -100
                        assert int(mask.sum()) == labels.shape[0] * 5
                        logits = outputs.logits.detach().float()[:, :-1, :][mask] / .8
                        independent = -F.log_softmax(logits, dim=-1).gather(1, shifted[mask].unsqueeze(1)).mean()
                        difference = float((independent - value.detach()).abs())
                        assert difference < 2e-5
                    event('independent_ce', abs_difference=difference, supervised_tokens=int(mask.sum()),
                          rows=labels.shape[0], temperature=.8, label_padding_by_attention_mask=True)
                    first_ce = False
                return (value, outputs) if return_outputs else value

            def get_train_dataloader(self):
                loader = ReplayLoader(super().get_train_dataloader(), saved_loader, event)
                self.replay_loader = loader
                expected_microbatches = math.ceil(len(train) / (8 * world))
                expected_steps = math.ceil(expected_microbatches / 4)
                assert len(loader) == expected_microbatches == 8240
                assert expected_steps == 2060
                report['effective_loader'] = {'microbatches_per_epoch_per_rank': len(loader),
                    'updates_per_epoch': expected_steps, 'total_updates': expected_steps * 4,
                    'presentations_per_epoch': len(loader) * 8 * world,
                    'original_rows_per_epoch': len(train), 'tail_repeated_presentations': len(loader) * 8 * world - len(train),
                    'batch_sampler': type(loader.batch_sampler).__name__,
                    'inner_sampler': type(loader.batch_sampler.batch_sampler.sampler).__name__}
                event('loader', **report['effective_loader'])
                return loader

            def _load_rng_state(self, checkpoint):
                # Native HF restores after collecting the first resumed group;
                # ReplayLoader must restore before data/prompt construction.
                if checkpoint:
                    event('native_late_rng_restore_suppressed', checkpoint=str(checkpoint))

            def training_step(self, current, inputs, num_items_in_batch=None):
                step = self.state.global_step
                if args.probe_stop or step < 3 or step % 250 == 0:
                    event('batch', step=step, loader_yielded=self.replay_loader.yielded,
                          input_sha256=hashlib.sha256(inputs['input_ids'].cpu().numpy().tobytes()).hexdigest(),
                          label_sha256=hashlib.sha256(inputs['labels'].cpu().numpy().tobytes()).hexdigest(),
                          shape=list(inputs['input_ids'].shape))
                assert self.current_gradient_accumulation_steps == 4
                value = super().training_step(current, inputs, num_items_in_batch)
                assert torch.isfinite(value).all()
                if args.probe_stop:
                    event('micro_loss', step=step, loss=float(value),
                          current_gradient_accumulation_steps=self.current_gradient_accumulation_steps)
                return value

            def _save_optimizer_and_scheduler(self, directory):
                # HF4.57 defaults to exclude_frozen_parameters=True for PEFT.
                # Full frozen Int8 base/scales are required for strict recovery.
                assert self.is_deepspeed_enabled
                self.model_wrapped.save_checkpoint(directory, exclude_frozen_parameters=False)
                if rank == 0:
                    torch.save(self.lr_scheduler.state_dict(), Path(directory) / 'scheduler.pt')

            def _save(self, output_dir=None, state_dict=None):
                # PEFT auto-detection sees expanded embeddings and otherwise
                # adds duplicate wrapper keys beyond the 394 adapter entries.
                # The complete active tied matrix already belongs to
                # modules_to_save; use the same explicit policy as Q06-L.
                directory = Path(output_dir or self.args.output_dir)
                directory.mkdir(parents=True, exist_ok=True)
                self.model.save_pretrained(directory, state_dict=state_dict,
                    safe_serialization=False, save_embedding_layers=False)
                self.processing_class.save_pretrained(directory)
                torch.save(self.args, directory / 'training_args.bin')

            def _rotate_checkpoints(self, use_mtime=False, output_dir=None):
                if self._writing_checkpoint:
                    return
                return super()._rotate_checkpoints(use_mtime=use_mtime, output_dir=output_dir)

            def _save_checkpoint(self, current, trial):
                begin = time.monotonic()
                self._writing_checkpoint = True
                try:
                    super()._save_checkpoint(current, trial)
                finally:
                    self._writing_checkpoint = False
                checkpoint = output / f'checkpoint-{self.state.global_step}'
                torch.save(self.replay_loader.checkpoint(self.state.global_step), checkpoint / f'loader_state_{rank}.pth')
                if rank == 0:
                    torch.save(self.lr_scheduler.state_dict(), checkpoint / 'external_scheduler.pth')
                dist.barrier()
                if rank == 0:
                    seal_checkpoint(checkpoint, self.state.global_step, args.run_commit, config_sha)
                dist.barrier()
                if rank == 0:
                    self._rotate_checkpoints(use_mtime=False, output_dir=str(output))
                dist.barrier()
                event('checkpoint_complete', step=self.state.global_step, path=str(checkpoint),
                      best=self.state.best_model_checkpoint, best_metric=self.state.best_metric,
                      seconds=time.monotonic() - begin)

            def evaluate(self, *positional, **keyword):
                begin = time.monotonic()
                metrics = super().evaluate(*positional, **keyword)
                assert math.isfinite(float(metrics['eval_loss']))
                event('validation_complete', step=self.state.global_step, metrics=metrics,
                      records=len(keyword.get('eval_dataset', self.eval_dataset)), seconds=time.monotonic() - begin)
                return metrics

        class Callback(TrainerCallback):
            def on_train_begin(self, args_, state, control, **keyword):
                engine = trainer.model_wrapped
                assert state.max_steps == 8240 and engine.zero_optimization_stage() == 2
                assert engine.train_micro_batch_size_per_gpu() == 8
                assert engine.gradient_accumulation_steps() == 4 and engine.train_batch_size() == 64
                parameters = [parameter for group in engine.optimizer.bit16_groups for parameter in group]
                assert len(parameters) == 393 and {id(parameter) for parameter in parameters} == {id(parameter) for parameter in trainable.values()}
                assert len({id(parameter) for parameter in parameters}) == len(parameters)
                assert sharing(model)['same_parameter_object'] and model.is_gradient_checkpointing
                assert model.config._attn_implementation == 'sdpa' and model.config.use_cache is False
                if args.probe_stop:
                    # Snapshot after DS casts and any strict checkpoint load.
                    # Initial dtype rounding or loading a resumed adapter must
                    # never count as an optimizer-produced parameter update.
                    for name, parameter in trainable.items():
                        initial_probe[name] = (parameter.detach().cpu().clone() if parameter is not tied
                            else parameter.detach()[tokenmap['sid_ids']].cpu().clone())
                    report['probe_initial_state'] = {
                        'global_step': state.global_step,
                        'deepspeed_global_step': engine.global_steps,
                        'after_deepspeed_initialization_and_strict_restore': True,
                        'before_first_update_of_this_process': True,
                        'parameter_dtypes': {name: str(parameter.dtype) for name, parameter in trainable.items()},
                    }
                    event('probe_initial_state', global_step=state.global_step,
                          deepspeed_global_step=engine.global_steps,
                          parameter_dtypes=sorted({str(parameter.dtype) for parameter in trainable.values()}))
                report.update(status='TRAINING', effective_ds=engine._config._param_dict,
                              initial_global_step=state.global_step, optimizer_membership_exact=True,
                              scheduler_class=type(trainer.lr_scheduler).__module__ + '.' + type(trainer.lr_scheduler).__name__,
                              deepspeed_owns_scheduler=engine.lr_scheduler is not None)
                write(report_path, report)
                event('train_begin', global_step=state.global_step, max_steps=state.max_steps,
                      optimizer=type(engine.optimizer.optimizer).__name__, scheduler=trainer.lr_scheduler.state_dict(),
                      deepspeed_owns_scheduler=engine.lr_scheduler is not None)

            def on_step_end(self, args_, state, control, **keyword):
                engine = trainer.model_wrapped
                assert engine.global_steps == state.global_step
                norm = engine.get_global_grad_norm()
                assert norm is None or math.isfinite(float(norm))
                if args.probe_stop or state.global_step % 10 == 0:
                    event('optimizer_step', step=state.global_step, epoch=state.epoch,
                          lr=[float(group['lr']) for group in engine.optimizer.param_groups],
                          grad_norm=None if norm is None else float(norm))
                if args.probe_stop and state.global_step in [args.probe_save, args.probe_stop]:
                    control.should_save = True
                if args.probe_stop and state.global_step >= args.probe_stop:
                    control.should_training_stop = True
                return control

            def on_epoch_begin(self, args_, state, control, **keyword):
                nonlocal epoch_started
                epoch_started = time.monotonic()
                event('epoch_begin', step=state.global_step, epoch=state.epoch)

            def on_epoch_end(self, args_, state, control, **keyword):
                if args.probe_stop:
                    control.should_evaluate = False
                    control.should_save = False
                event('epoch_end', step=state.global_step, epoch=state.epoch,
                      seconds=time.monotonic() - epoch_started)
                return control

            def on_log(self, args_, state, control, logs=None, **keyword):
                if logs:
                    event('log', step=state.global_step, epoch=state.epoch, values=logs)

        trainer = QwenTrainer(model=model, args=training_args, train_dataset=train, eval_dataset=valid,
                              processing_class=tokenizer, data_collator=Collator(tokenizer), callbacks=[Callback()])
        # Qwen forward accepts **kwargs. Explicitly retain author's per-microbatch
        # mean and one GAS division, rather than newer num_items_in_batch semantics.
        trainer.model_accepts_loss_kwargs = False
        assert trainer.compute_loss_func is None
        result = trainer.train(resume_from_checkpoint=args.resume)
        for handle in handles:
            handle.remove()
        engine = trainer.model_wrapped
        if args.probe_stop:
            assert trainer.state.global_step == args.probe_stop
            changed = {}
            for name, parameter in trainable.items():
                current = parameter.detach().cpu() if parameter is not tied else parameter.detach()[tokenmap['sid_ids']].cpu()
                changed[name] = int((current != initial_probe[name]).sum())
                assert changed[name] > 0, name
                assert gradient_records[name]['finite'] and gradient_records[name]['nonzero_calls'] > 0
            assert gradient_records[next(name for name, parameter in trainable.items() if parameter is tied)]['all_918_sid_rows_have_gradient']
            report['probe_gradients'] = gradient_records
            report['probe_changed_elements'] = changed
            state = {'trainable': {name: parameter.detach().cpu().clone() for name, parameter in trainable.items()},
                     'optimizer': cpu_clone(engine.optimizer.state_dict()),
                     'scheduler': cpu_clone(trainer.lr_scheduler.state_dict()),
                     'global_step': trainer.state.global_step, 'rng': rng_capture(trainer.replay_loader.loader),
                     'loader': trainer.replay_loader.checkpoint(trainer.state.global_step),
                     'int8_dynamic': int8_dynamic_state(model)}
            torch.save(state, output / f'acceptance_state_{rank}.pt')
            del state, initial_probe
            event('acceptance_state_saved', step=trainer.state.global_step)
            if args.preflight_interfaces:
                from torch.utils.data import Subset
                metrics = trainer.evaluate(eval_dataset=Subset(valid, range(32)))
                checkpoint = output / f'checkpoint-{args.probe_stop}'
                trainer.state.best_model_checkpoint = str(checkpoint)
                trainer.state.best_metric = metrics['eval_loss']
                strict_ds_load(engine, checkpoint)
                trainer.save_model(str(output / 'preflight_adapter'))
                event('preflight_best_restore', checkpoint=str(checkpoint),
                      adapter=str(output / 'preflight_adapter'), selection_scope='interface only; not formal selection')
        else:
            assert trainer.state.global_step == 8240 and trainer.state.epoch == 4.0
            assert trainer.state.best_model_checkpoint and math.isfinite(float(trainer.state.best_metric))
            adapter = output / 'best_adapter'
            trainer.save_model(str(adapter))
            trainer.save_state()
            if rank == 0:
                disk = torch.load(adapter / 'adapter_model.bin', map_location='cpu', weights_only=False)
                active = get_peft_model_state_dict(model, save_embedding_layers=False)
                assert set(active) == set(disk)
                for name, value in active.items():
                    assert torch.equal(value.detach().cpu(), disk[name]), name
                assert sharing(model)['same_parameter_object']
                write(output / 'best_selection.json', {'run_commit': args.run_commit,
                    'checkpoint': trainer.state.best_model_checkpoint, 'metric': 'eval_loss',
                    'greater_is_better': False, 'value': trainer.state.best_metric,
                    'completed_training_step': trainer.state.global_step,
                    'loaded_best_checkpoint_step': engine.global_steps,
                    'adapter_sha256': sha(adapter / 'adapter_model.bin'),
                    'adapter_bytes': (adapter / 'adapter_model.bin').stat().st_size,
                    'adapter_keys': len(disk), 'sharing': sharing(model)})
            dist.barrier()
        report.update(status='PROBE_COMPLETE' if args.probe_stop else 'TRAINING_COMPLETE',
            global_step=trainer.state.global_step, epoch=trainer.state.epoch,
            best_checkpoint=trainer.state.best_model_checkpoint, best_metric=trainer.state.best_metric,
            metrics=result.metrics, elapsed_seconds=time.monotonic() - started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
        write(report_path, report)
        event('training_complete', global_step=trainer.state.global_step, status=report['status'])
    except BaseException as error:
        import traceback
        report.update(status='FAILED', error=repr(error), traceback=traceback.format_exc(),
                      elapsed_seconds=time.monotonic() - started)
        write(report_path, report)
        raise
    finally:
        events.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        import traceback
        if '--output' in sys.argv:
            output = Path(sys.argv[sys.argv.index('--output') + 1])
            write(output / f'failure.rank{os.environ.get("RANK", "unknown")}.json',
                  {'status': 'FAILED', 'error': repr(error), 'traceback': traceback.format_exc(), 'argv': sys.argv})
        raise
