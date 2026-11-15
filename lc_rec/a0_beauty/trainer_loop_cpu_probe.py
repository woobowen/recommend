#!/usr/bin/env python3
"""Exercise the pinned patched Trainer on four short CPU-only toy epochs."""
import json
import os
import sys
import tempfile
import torch
from transformers import Trainer, TrainerCallback, TrainingArguments
import transformers.trainer as trainer_module
from trainer_loop_patch import build


class Dataset:
    def __len__(self): return 10
    def __getitem__(self, i): return {'x': torch.tensor([i/10], dtype=torch.float32)}


class Model(torch.nn.Module):
    def __init__(self): super().__init__();self.layer=torch.nn.Linear(1,1)
    def forward(self, x):
        logits = self.layer(x)
        return {'loss': logits.square().mean(), 'logits': logits}


class AuditTrainer(Trainer):
    def _beauty_capture_pre_best(self): pass
    def training_step(self, model, inputs):
        self.consumed += 1
        return super().training_step(model, inputs)

    def evaluate(self, *args, **kwargs):
        before = [None if p.grad is None else p.grad.detach().clone() for p in self.model.parameters()]
        result = super().evaluate(*args, **kwargs)
        after = [None if p.grad is None else p.grad.detach().clone() for p in self.model.parameters()]
        assert all((a is None and b is None) or (a is not None and b is not None and torch.equal(a,b))
                   for a,b in zip(before,after)), 'validation changed accumulated gradients'
        self.validation.append({'consumed':self.consumed,'updates':self.state.global_step})
        return result


class Epochs(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        events.append({'updates':state.global_step})


events=[]
output = tempfile.mkdtemp(prefix='letter-beauty-trainer-cpu-')
args = TrainingArguments(output_dir=output, no_cuda=True, per_device_train_batch_size=1,
    per_device_eval_batch_size=2, gradient_accumulation_steps=4, num_train_epochs=4,
    max_steps=10, evaluation_strategy='epoch', save_strategy='no', report_to=[],
    disable_tqdm=True, remove_unused_columns=False, seed=42, logging_steps=100, eval_delay=1)
AuditTrainer._inner_training_loop, patched_sha = build(trainer_module)
trainer = AuditTrainer(model=Model(), args=args, train_dataset=Dataset(), eval_dataset=Dataset(),
                       callbacks=[Epochs()])
trainer.accelerator.gradient_state.plugin_kwargs['sync_with_dataloader']=False
trainer._beauty_epochs=4; trainer._beauty_resume_state=None
trainer.consumed=0; trainer.validation=[]
trainer.train()
assert trainer.consumed == 40 and trainer.state.global_step == 10 and trainer.state.epoch == 4.0
assert [v['consumed'] for v in trainer.validation] == [10,20,30,40]
assert [v['updates'] for v in trainer.validation] == [2,5,7,10]
assert [v['updates'] for v in events] == [2,5,7,10]
print(json.dumps({'status':'PASS', 'output':output, 'patched_loop_sha256':patched_sha,
                  'training_microbatches':trainer.consumed, 'validations':trainer.validation,
                  'updates':trainer.state.global_step, 'epoch':trainer.state.epoch,
                  'scope':'CPU toy Trainer, no ZeRO, resume, distributed or Beauty model'}, indent=2))
