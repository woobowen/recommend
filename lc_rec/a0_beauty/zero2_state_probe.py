#!/usr/bin/env python3
"""Observe small-model ZeRO-2 partial-accumulation state on two real GPUs.

This is an observational GPU diagnostic, not the Beauty acceptance or a
checkpoint-recovery test. Run only under the task's bounded guarded runner.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import deepspeed
import torch
import torch.distributed as dist

rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
assert int(os.environ['WORLD_SIZE']) == 2
torch.cuda.set_device(local)
torch.manual_seed(42)
model = torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.Tanh(), torch.nn.Linear(128, 16)).cuda(local)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=.01)
cfg = {'train_micro_batch_size_per_gpu': 8, 'gradient_accumulation_steps': 4,
       'train_batch_size': 64, 'bf16': {'enabled': True}, 'zero_allow_untested_optimizer': True,
       'zero_optimization': {'stage': 2, 'allgather_partitions': True,
           'allgather_bucket_size': 500000, 'overlap_comm': True, 'reduce_scatter': True,
           'reduce_bucket_size': 500000, 'contiguous_gradients': True},
       'steps_per_print': 100000}
engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer,
                                        model_parameters=model.parameters(), config=cfg)
torch.cuda.synchronize()
processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                                     '--format=csv,noheader,nounits'], text=True)
actual = [line.split(', ') for line in processes.strip().splitlines() if line.strip()]
expected_uuids = ['GPU-dfff0e95-1db6-e2f4-3431-55bb852fa0ae', 'GPU-59cac3bd-3645-668b-79c8-839e0d878893']
assert [str(os.getpid()), expected_uuids[local]] in actual, 'GPU process mapping mismatch'
task_pids = [None, None]
dist.all_gather_object(task_pids, str(os.getpid()))
assert all(pid in task_pids for pid, _ in actual), 'another GPU process appeared after launch'


def describe(value):
    if isinstance(value, torch.Tensor):
        return {'shape': list(value.shape), 'dtype': str(value.dtype),
                'device': str(value.device), 'nonzero': int(torch.count_nonzero(value).item())}
    if isinstance(value, dict): return {str(k): describe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [describe(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)): return value
    return type(value).__name__


events = []
for micro in range(10):
    x = torch.full((8, 64), (micro + rank + 1)/16, device=torch.device('cuda', local), dtype=next(engine.module.parameters()).dtype)
    loss = engine(x).float().square().mean()
    engine.backward(loss)
    was_boundary = bool(engine.is_gradient_accumulation_boundary())
    engine.step()
    if micro in (0, 1, 3, 7, 9):
        torch.cuda.synchronize()
        opt = engine.optimizer
        events.append({'consumed': micro+1, 'global_steps': engine.global_steps,
                       'micro_steps': engine.micro_steps, 'was_boundary': was_boundary,
                       'optimizer_type': type(opt).__name__,
                       'optimizer_micro_step_id': getattr(opt, 'micro_step_id', None),
                       'averaged_gradients': describe(getattr(opt, 'averaged_gradients', None)),
                       'parameter_gradients': {n: describe(p.grad) for n, p in engine.module.named_parameters()},
                       'loss': float(loss.detach().cpu())})
assert engine.global_steps == 2 and engine.micro_steps == 10
out = Path(os.environ['GRID_JOB_DIR'])/'output'/f'zero2_state_rank{rank}.json'
out.write_text(json.dumps({'rank':rank, 'pid': os.getpid(), 'gpu_uuid': expected_uuids[local], 'argv':sys.argv, 'events':events,
    'scope':'observational toy model; no checkpoint or resume proof'}, indent=2)+'\n')
dist.barrier()
dist.destroy_process_group()
