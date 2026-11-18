"""Strict rank-local supplemental ZeRO-2 accumulation state.

This is a draft helper: it must be integrated with the native checkpoint,
Trainer's actual sample cursor, and verified across process restarts before use.
"""
import os
from pathlib import Path
import random
from g3_identity import EXECUTION_TOPOLOGY, EXECUTION_PROFILE, verify_execution_profile


def _map(value, fn):
    import torch
    if isinstance(value, torch.Tensor): return fn(value)
    if isinstance(value, dict): return {k: _map(v, fn) for k, v in value.items()}
    if isinstance(value, list): return [_map(v, fn) for v in value]
    if isinstance(value, tuple): return tuple(_map(v, fn) for v in value)
    return value


def snapshot(engine, scheduler, accelerator, loader_state, *, run_commit, config_sha256):
    import numpy as np
    import torch
    rank = torch.distributed.get_rank()
    verify_execution_profile(EXECUTION_PROFILE)
    gas=EXECUTION_PROFILE['gradient_accumulation_steps'];world=EXECUTION_PROFILE['world_size']
    assert torch.distributed.get_world_size() == world
    assert engine.zero_optimization_stage() == 2 and engine.gradient_accumulation_steps() == gas
    assert type(engine.optimizer).__name__ == 'DeepSpeedZeroOptimizer'
    assert not accelerator.gradient_state.sync_with_dataloader
    assert len(run_commit) == 40 and len(config_sha256) == 64
    consumed = int(loader_state['consumed_total'])
    assert engine.micro_steps == consumed and engine.global_steps == consumed // gas
    assert engine.optimizer.micro_step_id == ((consumed % gas) - 1 if consumed % gas else -1)
    device = torch.cuda.current_device()
    torch.cuda.synchronize(device)
    clone = lambda t: t.detach().cpu().clone()
    params = {name: clone(p.grad) for name, p in engine.module.named_parameters() if p.grad is not None}
    return {'schema': 2, 'rank': rank, 'world_size': world,
            'execution_topology':EXECUTION_TOPOLOGY,'execution_profile':EXECUTION_PROFILE,
            'run_commit': run_commit,
            'config_sha256': config_sha256, 'micro_steps': int(engine.micro_steps),
            'global_steps': int(engine.global_steps),
            'optimizer_micro_step_id': int(engine.optimizer.micro_step_id),
            'averaged_gradients': _map(engine.optimizer.averaged_gradients, clone),
            'parameter_gradients': params, 'scheduler': scheduler.state_dict(),
            'accelerator_step': int(accelerator.step), 'loader_state': loader_state,
            'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
                    'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(device)}}


def save_exclusive(path, state):
    import torch
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    if path.exists() or temporary.exists(): raise FileExistsError(path)
    with temporary.open('xb') as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def restore(engine, scheduler, accelerator, state, *, run_commit, config_sha256):
    import numpy as np
    import torch
    rank = torch.distributed.get_rank()
    verify_execution_profile(state['execution_profile'])
    gas=EXECUTION_PROFILE['gradient_accumulation_steps'];world=EXECUTION_PROFILE['world_size']
    assert state['schema'] == 2 and state['rank'] == rank and state['world_size'] == world
    assert state['execution_topology']==EXECUTION_TOPOLOGY
    assert state['run_commit'] == run_commit and state['config_sha256'] == config_sha256
    assert engine.zero_optimization_stage() == 2 and engine.gradient_accumulation_steps() == gas
    assert type(engine.optimizer).__name__ == 'DeepSpeedZeroOptimizer'
    assert not accelerator.gradient_state.sync_with_dataloader
    assert engine.global_steps == state['global_steps'] == state['micro_steps'] // gas
    assert state['loader_state']['consumed_total'] == state['micro_steps']
    device = torch.device('cuda', torch.cuda.current_device())
    engine.micro_steps = state['micro_steps']
    engine.optimizer.micro_step_id = state['optimizer_micro_step_id']
    engine.optimizer.averaged_gradients = _map(state['averaged_gradients'], lambda t: t.to(device))
    named = dict(engine.module.named_parameters())
    assert set(state['parameter_gradients']) <= set(named)
    for name, p in named.items():
        saved = state['parameter_gradients'].get(name)
        if saved is None: p.grad = None
        else:
            assert tuple(saved.shape) == tuple(p.shape) and saved.dtype == p.dtype
            p.grad = saved.to(device)
    scheduler.load_state_dict(state['scheduler'])
    accelerator.step = state['accelerator_step']
    random.setstate(state['rng']['python']); np.random.set_state(state['rng']['numpy'])
    torch.set_rng_state(state['rng']['torch']); torch.cuda.set_rng_state(state['rng']['cuda'], device)
    return state['loader_state']
