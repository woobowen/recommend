#!/usr/bin/env python3
"""A0 author Trainer entry with full, strict DS checkpoints and loader replay.

The vendor tree is immutable. A bounded acceptance run uses the same four-epoch
scheduler and loader, stopping by callback; it is not a formal seed.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import inspect
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.partial')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def nosync_source_identity(callable_):
    # contextmanager's outer wrapper is defined in contextlib.py. Inspect the
    # original decorated Accelerator method, retaining the full-file guard.
    original=inspect.unwrap(callable_)
    path=inspect.getsourcefile(original)
    return {'source_file':path,'file_sha256':sha(path),
            'method_sha256':hashlib.sha256(inspect.getsource(original).encode()).hexdigest(),
            'signature':str(inspect.signature(original))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['source-root', 'data-root', 'model', 'output', 'config', 'run-commit']:
        p.add_argument('--' + name, required=True)
    p.add_argument('--resume')
    p.add_argument('--probe-stop', type=int, default=0)
    p.add_argument('--probe-save', type=int, default=2)
    p.add_argument('--precision-approval', required=True)
    a = p.parse_args()
    rank = int(os.environ['RANK']); local = int(os.environ['LOCAL_RANK'])
    assert int(os.environ['WORLD_SIZE']) == 2
    cfg = json.loads(Path(a.config).read_text())
    author=json.loads(Path(a.config).with_name('author_instruments.json').read_text())
    for k,v in author.items():
        if k not in ['base_model','output_dir','resume_from_checkpoint','deepspeed','data_path','gradient_accumulation_steps']:
            assert cfg[k]==v, 'unapproved author setting drift: '+k
    assert (cfg['gradient_accumulation_steps'],cfg['global_batch'],cfg['world_size'])==(4,64,2)
    approval = json.loads(Path(a.precision_approval).read_text())
    assert approval['formal_a_only_bf16_authorized'] is True
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    import torch
    import torch.distributed as dist
    import numpy as np
    from transformers import Trainer, TrainingArguments, TrainerCallback, LlamaTokenizer
    import transformers.trainer as trainer_module
    import peft
    from peft import LoraConfig, TaskType, get_peft_model
    from peft import get_peft_model_state_dict
    torch.cuda.set_device(local)
    dist.init_process_group('nccl', timeout=dt.timedelta(minutes=15))
    sys.path.insert(0, str(Path(a.source_root).resolve()))
    from modeling_letter import LETTER
    from fastchat.train.llama2_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
    replace_llama_attn_with_flash_attn()
    from compat import (adapt_fastchat_unpad_arity, restore_legacy_bnb_state_field,
                        install_peft05_out_of_place, install_peft05_lora_a_bf16_autocast)
    from utils import load_datasets, set_seed
    from collator import Collator
    set_seed(42)
    from formal_evaluate import catalog_and_users
    _, _, _, data_identity = catalog_and_users(a.data_root)
    report = {'status': 'INITIALIZING', 'run_commit': a.run_commit, 'rank': rank,
              'argv': sys.argv, 'cwd': os.getcwd(), 'python': sys.executable,
              'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'config_sha256': sha(a.config), 'precision_approval': approval,
              'versions': {x: importlib.metadata.version(x) for x in
                           ['torch','transformers','accelerate','deepspeed','peft','bitsandbytes','fschat','flash-attn']},
              'unpad_compat': adapt_fastchat_unpad_arity(), 'probe': bool(a.probe_stop)}
    report['data_identity']=data_identity
    report['base_model_identity']={'path':str(Path(a.model).resolve()),
          'revision':'4782ad278652c7c71b72204d462d6d01eaaf7549',
          'config_sha256':sha(Path(a.model)/'config.json'),
          'source':'previous official LFS verification; current file sizes/receipt rechecked in preflight'}
    assert Path(a.model).name==report['base_model_identity']['revision']
    assert report['base_model_identity']['config_sha256']=='e4c3dea29aefc16c0e63825f9ae097dbf5cb5d7c320843a84257e11be3b452c8'
    report['base_model_identity']['weight_files']={}
    for name,size in [('model-00001-of-00002.safetensors',9976579217),('model-00002-of-00002.safetensors',3500297444)]:
        f=Path(a.model)/name
        assert f.stat().st_size==size
        report['base_model_identity']['weight_files'][name]={'bytes':size,'mtime_ns':f.stat().st_mtime_ns}
    rp = out / ('rank%d.json' % rank)
    if rp.exists() and not a.probe_stop:
        assert a.resume, 'existing formal training output requires explicit full resume'
        previous=out/'attempt_reports'/('%s-rank%d.json'%(sha(rp)[:16],rank))
        previous.parent.mkdir(exist_ok=True)
        if previous.exists():assert previous.read_bytes()==rp.read_bytes()
        else:previous.write_bytes(rp.read_bytes())
    events = (out / ('events.rank%d.jsonl' % rank)).open('a', buffering=1)
    write(rp,report)

    def event(kind, **kw):
        record = {'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                  'elapsed_seconds': time.monotonic()-started, 'event': kind,
                  'attempt_job_id':os.environ.get('GRID_JOB_ID'), **kw}
        events.write(json.dumps(record, ensure_ascii=False) + '\n')
        if kind != 'batch': print(json.dumps({'rank': rank, **record}), flush=True)

    da = SimpleNamespace(**cfg)
    da.data_path = str(Path(a.data_root).resolve())
    train, valid = load_datasets(da)
    assert (len(train), len(valid)) == (131837, 49544)
    tok = LlamaTokenizer.from_pretrained(a.model, model_max_length=2048, padding_side='right', local_files_only=True)
    tok.pad_token_id = 0
    added = tok.add_tokens(train.datasets[0].get_new_tokens())
    assert added == 918 and len(tok) == 32918
    if rank == 0: tok.save_pretrained(out)
    collator = Collator(da, tok)
    model = LETTER.from_pretrained(a.model, load_in_8bit=True, device_map={'':local}, local_files_only=True)
    model.set_hyper(.8); model.resize_token_embeddings(len(tok))
    model = peft.prepare_model_for_int8_training(model)
    report['bnb_compat'] = restore_legacy_bnb_state_field(model)
    model = get_peft_model(model, LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,
                  target_modules=cfg['lora_target_modules'].split(','),
                  modules_to_save=['embed_tokens','lm_head'],bias='none',
                  inference_mode=False,task_type=TaskType.CAUSAL_LM))
    report['out_of_place'] = install_peft05_out_of_place(model)
    assert report['out_of_place']['target_count']==224
    for n, v in model.named_parameters():
        if 'original_module' in n and any(s in n for s in ['embed_tokens','lm_head']): v.requires_grad=False
    model.config.use_cache=False
    inventory = {n:{'shape':list(v.shape),'dtype_before_ds':str(v.dtype),'requires_grad':v.requires_grad}
                 for n,v in model.named_parameters()}
    names = [n for n,v in model.named_parameters() if v.requires_grad]
    assert len(names)==450 and sum(v.numel() for v in model.parameters() if v.requires_grad)==289652736
    assert all(v.requires_grad == ('lora_' in n or 'modules_to_save.default' in n) for n,v in model.named_parameters())
    report['parameter_inventory'] = inventory
    report['data']={'train_rows':len(train),'valid_rows':len(valid),'tokenizer_vocab':len(tok)}
    ds=json.loads((Path(a.source_root)/'config/ds_z2_bf16.json').read_text())
    ds['flops_profiler']['enabled']=False
    ds['zero_allow_untested_optimizer']=True
    # Resolve the same auto fields through the native HF integration.
    ta=TrainingArguments(output_dir=str(out),seed=42,per_device_train_batch_size=8,
         per_device_eval_batch_size=8,gradient_accumulation_steps=4,warmup_ratio=.01,
         num_train_epochs=4,learning_rate=2e-4,weight_decay=.01,lr_scheduler_type='cosine',
         report_to=[],fp16=False,bf16=True,logging_steps=10,optim='adamw_torch',
         gradient_checkpointing=True,evaluation_strategy='epoch',save_strategy='epoch',
         eval_steps=1309,save_steps=1309,save_total_limit=1,load_best_model_at_end=True,
         deepspeed=ds,ddp_find_unused_parameters=False,eval_delay=1,dataloader_num_workers=0,
         local_rank=local,disable_tqdm=True)
    assert ta.metric_for_best_model=='loss' and ta.greater_is_better is False
    report['training_arguments']=ta.to_dict()
    report['deepspeed_requested']=ds

    def rng_capture(loader=None):
        result={'python':random.getstate(),'numpy':np.random.get_state(),
                'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
        gen=getattr(loader,'synchronized_generator',None)
        if gen is not None: result['loader_generator']=gen.get_state()
        return result

    def rng_restore(state, loader=None):
        random.setstate(state['python']);np.random.set_state(state['numpy'])
        torch.set_rng_state(state['cpu']);torch.cuda.set_rng_state_all(state['cuda'])
        if 'loader_generator' in state:
            gen=getattr(loader,'synchronized_generator',None)
            assert gen is not None
            gen.set_state(state['loader_generator'])

    class ReplayLoader:
        """Replay CPU input construction to the saved yield, including prefetch.

        Restore epoch-start global AND sampler-generator RNG before replay; then
        restore checkpoint RNG before requesting the next yield. The original
        DataLoaderShard retains its prefetch and batching behavior. Only resume
        performs CPU replay; it never runs a forward/backward for skipped rows.
        """
        def __init__(self, loader, saved):
            self.loader=loader;self.saved=saved;self.counter=0;self.yielded=0
            self.epoch_start=None;self.epoch_index=-1
        def __getattr__(self,key): return getattr(self.loader,key)
        def __len__(self): return len(self.loader)
        def __iter__(self):
            epoch=self.counter;self.counter+=1;self.epoch_index=epoch;self.yielded=0
            saved=self.saved
            active=saved is not None and epoch==saved['resume_epoch']
            skip=0
            if active:
                if saved['yielded'] < len(self):
                    rng_restore(saved['epoch_start'], self.loader)
                    skip=saved['yielded']
                else:
                    rng_restore(saved['checkpoint_rng'], self.loader)
            self.epoch_start=rng_capture(self.loader)
            iterator=iter(self.loader)
            if active:
                for _ in range(skip): next(iterator)
                rng_restore(saved['checkpoint_rng'], self.loader)
                self.yielded=skip
                event('loader_replayed',epoch=epoch,replayed_batches=skip,
                      checkpoint_step=saved['global_step'])
                self.saved=None
            for batch in iterator:
                self.yielded+=1
                yield batch
        def checkpoint(self,step):
            resume_epoch=self.epoch_index+(self.yielded==len(self))
            return {'schema':1,'epoch_index':self.epoch_index,'yielded':self.yielded,
                    'resume_epoch':resume_epoch,'epoch_start':self.epoch_start,
                    'checkpoint_rng':rng_capture(self.loader),'global_step':step}

    old_skip=trainer_module.skip_first_batches
    def checkpoint_skip(loader, num_batches=0):
        if isinstance(loader,ReplayLoader):
            assert loader.saved and loader.saved['yielded']==num_batches
            return loader
        return old_skip(loader,num_batches)
    trainer_module.skip_first_batches=checkpoint_skip

    # Make the native HF/DS strict module load explicit and verify its step.
    def strict_ds_load(engine, path):
        saved=json.loads((Path(path)/'trainer_state.json').read_text())
        loaded,client=engine.load_checkpoint(path,load_module_strict=True,
                 load_optimizer_states=True,load_lr_scheduler_states=True)
        assert loaded and engine.global_steps==saved['global_step']
        # HF4.31/Accelerate0.21 keeps LambdaLR outside the DS engine. Persist it
        # separately rather than attaching it to engine.step (which double-steps
        # the scheduler). This also covers native best-checkpoint restoration.
        scheduler_path=Path(path)/'external_scheduler.pth'
        assert scheduler_path.is_file()
        trainer.lr_scheduler.load_state_dict(torch.load(scheduler_path,weights_only=False,map_location='cpu'))
        event('strict_ds_restored',path=str(path),global_step=engine.global_steps,
              scheduler=trainer.lr_scheduler.state_dict(),client_keys=sorted(client))
    trainer_module.deepspeed_load_checkpoint=strict_ds_load

    saved_loader=None
    if a.resume:
        cp=Path(a.resume)
        meta=json.loads((cp/'complete.json').read_text())
        assert meta['run_commit']==a.run_commit and meta['config_sha256']==sha(a.config)
        for row in meta['files']:
            f=cp/row['path'];assert f.is_file() and f.stat().st_size==row['bytes']
            if row['sha256'] is not None: assert sha(f)==row['sha256']
        saved_loader=torch.load(cp/('loader_state_%d.pth'%rank),weights_only=False,map_location='cpu')

    class AuditTrainer(Trainer):
        def get_train_dataloader(self):
            result=ReplayLoader(super().get_train_dataloader(), saved_loader)
            self.replay_loader=result
            assert len(result)==8240
            event('loader',length=len(result),batch_sampler=type(result.batch_sampler).__name__,
                  sampler_generator=type(result.synchronized_generator).__name__)
            return result
        def _load_rng_state(self, checkpoint):
            # ReplayLoader restores before resumed input is consumed. Native HF
            # restores after fetching the first resumed input, which is too late.
            if checkpoint: event('native_late_rng_restore_suppressed',checkpoint=str(checkpoint))
        def training_step(self, current, inputs):
            step=self.state.global_step
            before=current.global_steps
            boundary=current.is_gradient_accumulation_boundary()
            if a.probe_stop or step<3 or step%250==0 or (step+1)%2060==0:
                event('batch',step=step,loader_yielded=self.replay_loader.yielded,
                      input_sha256=hashlib.sha256(inputs['input_ids'].cpu().numpy().tobytes()).hexdigest(),
                      label_sha256=hashlib.sha256(inputs['labels'].cpu().numpy().tobytes()).hexdigest(),
                      shape=list(inputs['input_ids'].shape))
            loss=super().training_step(current,inputs)
            if not torch.isfinite(loss).all(): raise FloatingPointError('nonfinite training loss')
            if a.probe_stop:
                assert current.global_steps==before+int(boundary)
                event('micro_loss',step=step,loss=float(loss),ds_global_step_before=before,
                      ds_global_step_after=current.global_steps,ds_accumulation_boundary=boundary,
                      scheduler_last_epoch=self.lr_scheduler.state_dict()['last_epoch'])
            return loss
        def _save_checkpoint(self,current,trial,metrics=None):
            if metrics is not None and 'eval_loss' in metrics:
                assert math.isfinite(float(metrics['eval_loss'])), 'nonfinite validation loss'
            super()._save_checkpoint(current,trial,metrics)
            cp=out/('checkpoint-%d'%self.state.global_step)
            # Native _save_checkpoint ran collectively, including DS optimizer,
            # scheduler, module states and rank-specific RNG. Add loader state.
            torch.save(self.replay_loader.checkpoint(self.state.global_step),cp/('loader_state_%d.pth'%rank))
            if rank==0:torch.save(self.lr_scheduler.state_dict(),cp/'external_scheduler.pth')
            dist.barrier()
            if rank==0:
                rows=[]
                for f in sorted(cp.rglob('*')):
                    if f.is_file() and f.name!='complete.json':
                        rows.append({'path':str(f.relative_to(cp)),'bytes':f.stat().st_size,
                                     'sha256':sha(f) if f.stat().st_size<8*1024**2 else None})
                assert any('optim_states.pt' in r['path'] for r in rows)
                write(cp/'complete.json',{'run_commit':a.run_commit,'config_sha256':sha(a.config),
                     'global_step':self.state.global_step,'files':rows,'collective_ranks':2})
            dist.barrier()
            event('checkpoint_complete',path=str(cp),step=self.state.global_step,
                  best=self.state.best_model_checkpoint,best_metric=self.state.best_metric)

    class Callback(TrainerCallback):
        def on_train_begin(self,args,state,control,**kw):
            engine=trainer.model_wrapped
            assert state.max_steps==8240
            named=dict(model.named_parameters());ids={id(v) for v in named.values() if v.requires_grad}
            original=engine.optimizer.bit16_groups
            assert {id(v) for g in original for v in g}==ids
            assert sum(len(g) for g in original)==450
            # Historical original failure is preserved and pinned, not retried
            # with a partially executed forward before the formal first batch.
            proof={'matched_dtype_conflict':True,'source':'prior audited A0 FP32/BF16 error',
                   'historical_evidence':'lc_rec/letter/evidence/jobs/letter-20261110-gpu-r1'}
            report['a_only_bf16']=install_peft05_lora_a_bf16_autocast(model,proof)
            assert len(report['a_only_bf16']['targets'])==224
            assert not torch.is_autocast_enabled()
            report['status']='TRAINING'
            report['effective_ds']=engine._config._param_dict
            report['optimizer_membership_exact']=True
            report['initial_global_step']=state.global_step
            write(rp,report)
            event('train_begin',global_step=state.global_step,max_steps=state.max_steps,
                  optimizer=type(engine.optimizer).__name__,scheduler=trainer.lr_scheduler.state_dict(),
                  deepspeed_owns_scheduler=engine.lr_scheduler is not None)
        def on_step_end(self,args,state,control,**kw):
            engine=trainer.model_wrapped
            assert engine.global_steps==state.global_step
            norm=engine.get_global_grad_norm()
            if norm is not None and not math.isfinite(float(norm)):raise FloatingPointError('nonfinite gradient norm')
            if a.probe_stop or state.global_step%10==0:
                event('optimizer_step',step=state.global_step,epoch=state.epoch,
                      lr=engine.get_lr(),grad_norm=None if norm is None else float(norm))
            if a.probe_stop and state.global_step in [a.probe_save,a.probe_stop]: control.should_save=True
            if a.probe_stop and state.global_step>=a.probe_stop: control.should_training_stop=True
            return control
        def on_epoch_end(self,args,state,control,**kw):
            if a.probe_stop: control.should_evaluate=False;control.should_save=False
            event('epoch_end',step=state.global_step,epoch=state.epoch)
            return control
        def on_log(self,args,state,control,logs=None,**kw):
            if logs: event('log',step=state.global_step,epoch=state.epoch,values=logs)

    trainer=AuditTrainer(model=model,args=ta,train_dataset=train,eval_dataset=valid,
                         tokenizer=tok,data_collator=collator,callbacks=[Callback()])
    # Accelerate 0.21 predates DS 0.16's explicit prohibition of no_sync for
    # partitioned gradients. Match official Accelerate's stage>=2 nullcontext
    # behavior on this one engine; DS retains its native loss scaling/steps.
    from contextlib import contextmanager
    from deepspeed import DeepSpeedEngine
    from accelerate import Accelerator
    from accelerate.utils import DistributedType
    original_no_sync=trainer.accelerator.no_sync
    assert report['versions']['accelerate']=='0.21.0' and report['versions']['deepspeed']=='0.16.9'
    assert original_no_sync.__func__ is Accelerator.no_sync
    nosync_identity=nosync_source_identity(original_no_sync)
    assert nosync_identity['file_sha256']=='933a19cd7411a44e46d23c54dbfc8c2ab94b3f29225e66aceed429e51dd43f8a'
    report['zero2_no_sync_compat']={'scope':'current Trainer accelerator instance and exact stage2 engine only',
               'upstream_reference_blob':'e666b8d7a07c789245bed88c9e68f220a81eb3e7',
               'calls':0,'source_identity':nosync_identity}
    @contextmanager
    def zero2_native_accumulation(current):
        if current is trainer.model_wrapped and type(current) is DeepSpeedEngine:
            assert trainer.accelerator.distributed_type==DistributedType.DEEPSPEED
            assert current.zero_optimization_stage()==2 and current.gradient_accumulation_steps()==4
            assert current.train_micro_batch_size_per_gpu()==8 and current.train_batch_size()==64 and dist.get_world_size()==2
            assert not current.inside_no_sync_ctxt and current.enable_backward_allreduce
            assert current._is_gradient_accumulation_boundary is None
            report['zero2_no_sync_compat']['calls']+=1
            yield
        else:
            with original_no_sync(current):yield
    trainer.accelerator.no_sync=zero2_native_accumulation
    # Preserve intermediate acceptance checkpoints; formal uses author limit=1.
    if a.probe_stop: trainer.args.save_total_limit=3
    try:
        result=trainer.train(resume_from_checkpoint=a.resume)
        if a.probe_stop:
            assert trainer.state.global_step==a.probe_stop
            engine=trainer.model_wrapped
            state={'trainable':{n:v.detach().cpu().clone() for n,v in model.named_parameters() if v.requires_grad},
                   'optimizer':engine.optimizer.state_dict(),
                   'scheduler':trainer.lr_scheduler.state_dict(),
                   'global_step':trainer.state.global_step,'rng':rng_capture(trainer.replay_loader.loader)}
            torch.save(state,out/('acceptance_state_%d.pt'%rank))
            event('acceptance_state_saved',step=trainer.state.global_step)
        else:
            assert trainer.state.global_step==8240 and trainer.state.epoch==4.0
            assert trainer.state.best_model_checkpoint and trainer.state.best_metric is not None
            assert math.isfinite(float(trainer.state.best_metric))
            # Native Trainer already restored strict best DS state on all ranks.
            best=out/'best_adapter'
            trainer.save_model(str(best));trainer.save_state()
            if rank==0:
                weights=torch.load(best/'adapter_model.bin',map_location='cpu',weights_only=False)
                active=get_peft_model_state_dict(model)
                assert set(active)==set(weights) and len(weights)==450
                for n,v in active.items():assert torch.equal(v.detach().cpu(),weights[n])
                write(out/'best_selection.json',{'checkpoint':trainer.state.best_model_checkpoint,
                      'metric':'eval_loss','greater_is_better':False,'value':trainer.state.best_metric,
                      'adapter_sha256':sha(best/'adapter_model.bin'),'run_commit':a.run_commit})
            dist.barrier()
        report.update(status='PROBE_COMPLETE' if a.probe_stop else 'TRAINING_COMPLETE',
                      global_step=trainer.state.global_step,epoch=trainer.state.epoch,
                      best_checkpoint=trainer.state.best_model_checkpoint,best_metric=trainer.state.best_metric,
                      metrics=result.metrics,elapsed_seconds=time.monotonic()-started,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        write(rp,report)
    except BaseException as e:
        import traceback
        report.update(status='FAILED',error=repr(e),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-started)
        write(rp,report);raise
    finally: events.close()
    dist.destroy_process_group()


if __name__=='__main__':
    try:main()
    except BaseException as error:
        import traceback
        if '--output' in sys.argv:
            output=Path(sys.argv[sys.argv.index('--output')+1])
            write(output/('failure.rank%s.json'%os.environ.get('RANK','unknown')),
                  {'status':'FAILED','error':repr(error),'traceback':traceback.format_exc(),'argv':sys.argv})
        raise
