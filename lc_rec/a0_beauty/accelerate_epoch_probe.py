#!/usr/bin/env python3
"""Inspect installed Accelerate accumulation state across short CPU epochs."""
import json
import os
import torch
import accelerate
from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin
from torch.utils.data import DataLoader, TensorDataset

sync = os.environ['PROBE_SYNC_WITH_LOADER'] == '1'
plugin = GradientAccumulationPlugin(num_steps=4, sync_with_dataloader=sync)
acc = Accelerator(cpu=True, gradient_accumulation_plugin=plugin)
loader = acc.prepare(DataLoader(TensorDataset(torch.arange(10)), batch_size=1))
model = torch.nn.Linear(1, 1)
events = []
for epoch in range(2):
    for batch, _ in enumerate(loader):
        with acc.accumulate(model):
            events.append({'epoch': epoch, 'batch': batch, 'step': acc.step,
                           'end_of_dataloader': acc.gradient_state.end_of_dataloader,
                           'sync': acc.sync_gradients})
actual = [(e['epoch'], e['batch']) for e in events if e['sync']]
expected = ([(0, 3), (0, 7), (0, 9), (1, 3), (1, 7), (1, 9)] if sync
            else [(0, 3), (0, 7), (1, 1), (1, 5), (1, 9)])
assert actual == expected, (actual, expected)
print(json.dumps({'accelerate_version': accelerate.__version__, 'torch_version': torch.__version__,
                  'sync_with_dataloader': sync, 'batches_per_epoch': 10,
                  'events': events, 'sync_positions': actual, 'status': 'PASS',
                  'scope': 'Accelerate CPU state only; no DeepSpeed optimizer or model training'}, indent=2))
