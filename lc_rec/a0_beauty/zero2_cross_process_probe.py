#!/usr/bin/env python3
"""Two-rank toy ZeRO-2 checkpoint experiment across an incomplete GAS cycle.

Not a Beauty model, HF Trainer, DataLoader, validation or formal preacceptance.
"""
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import deepspeed
import numpy as np
import torch
import torch.distributed as dist

rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
assert int(os.environ['WORLD_SIZE']) == 2
torch.cuda.set_device(local)
device = torch.device('cuda', local)
out = Path(os.environ['GRID_JOB_DIR'])/'output'
checkpoint = out/'zero2_resume_checkpoint'
cfg = {'train_micro_batch_size_per_gpu': 8, 'gradient_accumulation_steps': 4,
       'train_batch_size': 64, 'bf16': {'enabled': True}, 'zero_allow_untested_optimizer': True,
       'zero_optimization': {'stage': 2, 'allgather_partitions': True,
           'allgather_bucket_size': 500000, 'overlap_comm': True, 'reduce_scatter': True,
           'reduce_bucket_size': 500000, 'contiguous_gradients': True},
       'steps_per_print': 100000}


def new_engine():
    torch.manual_seed(42)
    model = torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.Dropout(.05),
                                torch.nn.Tanh(), torch.nn.Linear(128, 16)).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=.01)
    engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optim,
                                           model_parameters=model.parameters(), config=cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda step: 0.5*(1+np.cos(np.pi*step/10)))
    return engine, scheduler


def clone_cpu(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: clone_cpu(v) for k,v in value.items()}
    if isinstance(value, list): return [clone_cpu(v) for v in value]
    if isinstance(value, tuple): return tuple(clone_cpu(v) for v in value)
    return value


def move_cuda(value):
    if isinstance(value, torch.Tensor): return value.to(device)
    if isinstance(value, dict): return {k: move_cuda(v) for k,v in value.items()}
    if isinstance(value, list): return [move_cuda(v) for v in value]
    if isinstance(value, tuple): return tuple(move_cuda(v) for v in value)
    return value


def same(a, b, name):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b), name
    elif isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), name
        for k in a: same(a[k], b[k], name+'/'+str(k))
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), name
        for i, (x,y) in enumerate(zip(a,b)): same(x,y,name+'/'+str(i))
    else: assert a == b, (name, a, b)


def rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(device)}


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state(state['cuda'], device)


def batch(engine, scheduler, i):
    x = torch.randn((8, 64), device=device, dtype=next(engine.module.parameters()).dtype)
    xhash = hashlib.sha256(x.float().cpu().numpy().tobytes()).hexdigest()
    loss = engine(x).float().square().mean()
    engine.backward(loss)
    boundary = bool(engine.is_gradient_accumulation_boundary())
    engine.step()
    if boundary: scheduler.step()
    return {'i':i, 'input_sha256':xhash, 'loss':float(loss.detach().cpu()),
            'global_steps':engine.global_steps, 'micro_steps':engine.micro_steps,
            'scheduler_epoch':scheduler.last_epoch}


from types import SimpleNamespace
from zero2_partial_state import snapshot, save_exclusive, restore

mode = os.environ['BEAUTY_PROBE_MODE']
assert mode in ('continuous', 'resume')
run_commit = os.environ['BEAUTY_PROBE_COMMIT']
cfg_sha = hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
# This lightweight stand-in only exercises helper serialization. It does not
# test Accelerator's actual gradient state or a real Trainer/DataLoader.
accelerator = SimpleNamespace(step=0, gradient_state=SimpleNamespace(sync_with_dataloader=False))
engine, scheduler = new_engine()
if mode == 'continuous':
    random.seed(100+rank); np.random.seed(100+rank)
    torch.manual_seed(100+rank); torch.cuda.manual_seed_all(100+rank)
    for i in range(10): batch(engine, scheduler, i)
    accelerator.step = 10
    assert engine.global_steps == 2 and engine.micro_steps == 10
    assert engine.save_checkpoint(str(checkpoint), tag='partial10')
    state = snapshot(engine, scheduler, accelerator, {'consumed_total':10, 'epoch':1, 'batch':10},
                     run_commit=run_commit, config_sha256=cfg_sha)
    save_exclusive(out/f'cross_partial_rank{rank}.pt', state)
else:
    loaded, _ = engine.load_checkpoint(str(checkpoint), tag='partial10', load_module_strict=True,
                                       load_optimizer_states=True, load_lr_scheduler_states=True)
    assert loaded
    state = torch.load(out/f'cross_partial_rank{rank}.pt', map_location='cpu', weights_only=False)
    cursor = restore(engine, scheduler, accelerator, state, run_commit=run_commit, config_sha256=cfg_sha)
    assert cursor == {'consumed_total':10, 'epoch':1, 'batch':10} and accelerator.step == 10

events = [batch(engine, scheduler, i) for i in range(10,20)]
torch.cuda.synchronize()
result = {'weights':clone_cpu(engine.module.state_dict()),
          'optimizer':clone_cpu(engine.optimizer.optimizer.state_dict()),
          'scheduler':scheduler.state_dict(), 'rng':rng(), 'events':events,
          'global_steps':engine.global_steps, 'micro_steps':engine.micro_steps}
if mode == 'continuous':
    torch.save(result, out/f'cross_reference_rank{rank}.pt')
else:
    expected = torch.load(out/f'cross_reference_rank{rank}.pt', map_location='cpu', weights_only=False)
    for key in ('weights','optimizer','scheduler','events','global_steps','micro_steps'):
        same(expected[key], result[key], key)
    assert expected['rng']['python'] == result['rng']['python']
    assert expected['rng']['numpy'][0] == result['rng']['numpy'][0]
    assert np.array_equal(expected['rng']['numpy'][1], result['rng']['numpy'][1])
    assert expected['rng']['numpy'][2:] == result['rng']['numpy'][2:]
    same(expected['rng']['torch'], result['rng']['torch'], 'rng/torch')
    same(expected['rng']['cuda'], result['rng']['cuda'], 'rng/cuda')
    assert result['global_steps'] == 5 and result['micro_steps'] == 20
    (out/f'cross_resume_rank{rank}.json').write_text(json.dumps(
        {'status':'PASS', 'rank':rank, 'separate_process_launches':True,
         'checkpoint_microsteps':10, 'final_microsteps':20, 'final_updates':5,
         'exact_checks':['weights','optimizer','scheduler','rng','inputs','loss'],
         'scope':'toy ZeRO-2; Accelerator is a stand-in; no real Trainer, validation or DataLoader'}, indent=2)+'\n')
dist.barrier()
dist.destroy_process_group()


