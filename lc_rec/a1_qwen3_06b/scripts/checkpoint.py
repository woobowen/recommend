"""Qwen A1 complete ZeRO-2 checkpoint integrity and deterministic input replay.

This module does not replace attention, autograd, PEFT, or Int8 arithmetic.
The temporary hot SCB alias handles the public bitsandbytes 0.45.5 state_dict
contract: forward moves weight.SCB into state.SCB, while loading requires it
on weight. The exact existing storage is borrowed only during strict loading.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import random


DEEPSPEED_LOGGING_SENTINEL = (
    'Infinity [Accelerate 1.12.0 DeepSpeedPlugin.__post_init__: '
    'steps_per_print stdout logging sentinel]'
)


def deepspeed_metadata(config):
    """Encode only Accelerate's positive-inf stdout logging sentinel for JSON.

    Accelerate 1.12.0 utils/dataclasses.py DeepSpeedPlugin.__post_init__ sets
    config['steps_per_print'] = float('inf'). The engine/config is untouched;
    a shallow report copy changes that single metadata value only. Every other
    nonfinite value, including nested config values and negative inf/NaN at
    steps_per_print, remains an error under strict JSON validation.
    """
    assert isinstance(config, dict), 'Expected an actual DeepSpeed config dictionary'
    result = dict(config)
    value = result.get('steps_per_print')
    if type(value) is float and math.isinf(value) and value > 0:
        result['steps_per_print'] = DEEPSPEED_LOGGING_SENTINEL
    # Validate this one config subtree. This is deliberately not a general
    # nonfinite sanitizer and does not touch training loss/gradient evidence.
    json.dumps(result, allow_nan=False)
    return result


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    def default(value):
        if isinstance(value, (set, frozenset)):
            return sorted(value)
        raise TypeError(type(value).__name__)
    with temporary.open('w') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=default) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def required_files(step):
    prefix = f'global_step{step}/'
    return {
        'adapter_model.bin', 'adapter_config.json', 'trainer_state.json',
        'training_args.bin', 'external_scheduler.pth', 'latest',
        'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
        'loader_state_0.pth', 'loader_state_1.pth',
        'rng_state_0.pth', 'rng_state_1.pth',
        prefix + 'mp_rank_00_model_states.pt',
        prefix + 'bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt',
        prefix + 'bf16_zero_pp_rank_1_mp_rank_00_optim_states.pt',
    }


def seal_checkpoint(path, step, run_commit, config_sha):
    path = Path(path)
    rows = []
    for file in sorted(path.rglob('*')):
        if file.is_file() and file.name != 'complete.json':
            assert not file.is_symlink()
            size = file.stat().st_size
            assert size > 0
            rows.append({'path': str(file.relative_to(path)), 'bytes': size,
                         'sha256': sha(file) if size < 8 * 1024**2 else None})
    assert required_files(step) <= {row['path'] for row in rows}
    write(path / 'complete.json', {'schema': 'a1_full_checkpoint_v1',
          'global_step': step, 'run_commit': run_commit, 'config_sha256': config_sha,
          'collective_ranks': 2, 'exclude_frozen_parameters': False, 'files': rows})
    return verify_checkpoint(path, run_commit, config_sha)


def verify_checkpoint(path, run_commit, config_sha):
    path = Path(path)
    meta = json.loads((path / 'complete.json').read_text())
    assert meta['run_commit'] == run_commit and meta['config_sha256'] == config_sha
    assert meta['collective_ranks'] == 2 and meta['exclude_frozen_parameters'] is False
    assert required_files(meta['global_step']) <= {row['path'] for row in meta['files']}
    for row in meta['files']:
        relative = Path(row['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        file = path / relative
        assert file.is_file() and not file.is_symlink() and file.stat().st_size == row['bytes'] > 0
        if row['sha256'] is not None:
            assert sha(file) == row['sha256']
    assert (path / 'latest').read_text().strip() == f"global_step{meta['global_step']}"
    assert json.loads((path / 'trainer_state.json').read_text())['global_step'] == meta['global_step']
    return meta


def loader_generators(loader):
    """Capture actual generator objects without assuming the old HF loader."""
    import torch
    candidates = {'synchronized_generator': getattr(loader, 'synchronized_generator', None),
                  'base_generator': getattr(loader, 'generator', None)}
    sampler = getattr(loader, 'batch_sampler', None)
    while sampler is not None:
        if hasattr(sampler, 'sampler'):
            sampler = sampler.sampler
            break
        sampler = getattr(sampler, 'batch_sampler', None)
    if sampler is not None:
        # SeedableRandomSampler lazily creates this on its first iteration.
        if getattr(sampler, 'generator', None) is None and hasattr(sampler, 'initial_seed'):
            sampler.generator = torch.Generator().manual_seed(sampler.initial_seed)
        candidates['sampler_generator'] = getattr(sampler, 'generator', None)
    result, seen = {}, set()
    for name, generator in candidates.items():
        if generator is not None and id(generator) not in seen:
            assert isinstance(generator, torch.Generator)
            result[name] = generator
            seen.add(id(generator))
    assert result, 'No actual loader/sampler generator was captured'
    return result


def rng_capture(loader):
    import numpy as np
    import torch
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all(),
            'loader_generator': {name: gen.get_state() for name, gen in loader_generators(loader).items()}}


def rng_restore(state, loader):
    import numpy as np
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'])
    torch.cuda.set_rng_state_all(state['cuda'])
    generators = loader_generators(loader)
    assert generators.keys() == state['loader_generator'].keys()
    for name, generator in generators.items():
        generator.set_state(state['loader_generator'][name])


class ReplayLoader:
    """Recreate the native sharded loader and its one-batch lookahead on resume.

    HF 4.57 obtains four microbatches before each accumulation group. Checkpoints
    occur only at optimizer boundaries, so no Trainer-prefetched group remains.
    CPU replay restores epoch-start RNG, reconstructs every yielded batch and the
    DataLoaderShard lookahead, then restores checkpoint RNG before next yield.
    """
    def __init__(self, loader, saved, event):
        self.loader, self.saved, self.event = loader, saved, event
        self.epoch_index, self.yielded, self.epoch_start = 0, 0, None
        loader_generators(loader)

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def __len__(self):
        return len(self.loader)

    def set_epoch(self, epoch):
        self.epoch_index = epoch
        self.loader.set_epoch(epoch)

    def __iter__(self):
        saved, self.yielded = self.saved, 0
        active = saved is not None and self.epoch_index == saved['resume_epoch']
        skip = 0
        if active:
            if saved['yielded'] < len(self):
                rng_restore(saved['epoch_start'], self.loader)
                skip = saved['yielded']
            else:
                rng_restore(saved['checkpoint_rng'], self.loader)
        self.epoch_start = rng_capture(self.loader)
        iterator = iter(self.loader)
        if active:
            for _ in range(skip):
                next(iterator)
            rng_restore(saved['checkpoint_rng'], self.loader)
            self.yielded = skip
            self.saved = None
            self.event('loader_replayed', epoch=self.epoch_index,
                       replayed_batches=skip, checkpoint_step=saved['global_step'])
        for batch in iterator:
            self.yielded += 1
            yield batch

    def checkpoint(self, step):
        return {'schema': 1, 'epoch_index': self.epoch_index, 'yielded': self.yielded,
                'resume_epoch': self.epoch_index + int(self.yielded == len(self)),
                'epoch_start': self.epoch_start, 'checkpoint_rng': rng_capture(self.loader),
                'global_step': step}


@contextmanager
def strict_hot_int8_alias(model, event):
    """Borrow existing SCB storage for native strict hot loading, then restore."""
    from bitsandbytes.nn import Linear8bitLt
    aliases = []
    for name, module in model.named_modules():
        if isinstance(module, Linear8bitLt) and module.weight.SCB is None:
            assert module.state.SCB is not None, f'Uninitialized Int8 SCB: {name}'
            module.weight.SCB = module.state.SCB
            assert module.weight.SCB.data_ptr() == module.state.SCB.data_ptr()
            aliases.append((name, module))
    event('strict_int8_load_layout', hot_scb_alias_count=len(aliases),
          implementation='existing SCB storage alias only during native strict load')
    try:
        yield
    finally:
        for name, module in aliases:
            assert module.weight.SCB is module.state.SCB, name
            module.weight.SCB = None


def int8_dynamic_state(model):
    """Complete quantized weight/scales plus dynamic flags, only for probes."""
    from bitsandbytes.nn import Linear8bitLt
    import torch
    result = {}
    for name, module in model.named_modules():
        if not isinstance(module, Linear8bitLt):
            continue
        fields = {'weight': module.weight.detach().cpu().clone()}
        for owner_name, owner in [('weight', module.weight), ('state', module.state)]:
            for field in ['CB', 'SCB', 'CxB', 'SB', 'idx', 'subB', 'has_fp16_weights',
                          'threshold', 'is_training', 'use_pool', 'force_no_igemmlt']:
                if hasattr(owner, field):
                    value = getattr(owner, field)
                    if isinstance(value, torch.Tensor):
                        value = value.detach().cpu().clone()
                    elif not isinstance(value, (str, int, float, bool, tuple, list, type(None))):
                        raise TypeError(f'Unsupported Int8 dynamic field {name}.{owner_name}.{field}')
                    fields[f'{owner_name}.{field}'] = value
        result[name] = fields
    assert len(result) == 196
    return result


def cpu_clone(value):
    import copy
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_clone(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_clone(child) for child in value)
    if isinstance(value, list):
        return [cpu_clone(child) for child in value]
    return copy.deepcopy(value)


def cpu_int8_state_contract():
    """Bounded native bnb cold/hot load reproducer, requiring no CUDA context.

    Synthetic Int8 weights/scales exercise serialization only. This does not
    claim any training, numerical-kernel or full-checkpoint acceptance.
    """
    import inspect
    import torch
    from bitsandbytes.nn import Int8Params, Linear8bitLt
    from torch import nn
    source = inspect.getsourcefile(Linear8bitLt)
    records = []
    holder = nn.Module()
    holder.projection = Linear8bitLt(4, 3, bias=False, has_fp16_weights=False)
    module = holder.projection
    payload = torch.arange(-6, 6, dtype=torch.int8).reshape(3, 4)
    scales = torch.tensor([.25, .5, .75], dtype=torch.float32)
    module.weight = Int8Params(payload.clone(), requires_grad=False, has_fp16_weights=False,
                               CB=payload.clone(), SCB=scales.clone())
    module.state.CB = None
    module.state.SCB = None
    state = cpu_clone(holder.state_dict())
    cold = holder.load_state_dict(state, strict=True)
    assert not cold.missing_keys and not cold.unexpected_keys
    assert torch.equal(module.weight, payload) and torch.equal(module.weight.SCB, scales)
    module.init_8bit_state()
    assert module.weight.SCB is None and module.state.SCB is not None
    before_weight = module.weight.detach().clone()
    before_scales = module.state.SCB.detach().clone()
    before_storage = module.state.SCB.data_ptr()
    try:
        holder.load_state_dict(cpu_clone(state), strict=True)
    except RuntimeError as error:
        assert 'Loading a quantized checkpoint into non-quantized Linear8bitLt' in str(error)
        native_failure = str(error)
    else:
        raise AssertionError('Native hot SCB load did not exhibit the audited bnb contract')
    with strict_hot_int8_alias(holder, lambda kind, **fields: records.append({'event': kind, **fields})):
        loaded = holder.load_state_dict(cpu_clone(state), strict=True)
        assert not loaded.missing_keys and not loaded.unexpected_keys
        assert module.weight.SCB is module.state.SCB
        assert module.state.SCB.data_ptr() == before_storage
    assert module.weight.SCB is None and module.state.SCB.data_ptr() == before_storage
    assert torch.equal(module.weight, before_weight) and torch.equal(module.state.SCB, before_scales)
    assert not torch.cuda.is_initialized()
    return {'status': 'PASS_CPU_SERIALIZATION_CONTRACT', 'cuda_context_created': False,
            'scope': 'Synthetic CPU serialization; not a GPU/checkpoint recovery result',
            'source': {'path': source, 'sha256': sha(source),
                       'save_to_state_dict_line': inspect.getsourcelines(Linear8bitLt._save_to_state_dict)[1],
                       'load_from_state_dict_line': inspect.getsourcelines(Linear8bitLt._load_from_state_dict)[1],
                       'init_8bit_state_line': inspect.getsourcelines(Linear8bitLt.init_8bit_state)[1]},
            'cold_native_strict': True, 'hot_native_error': native_failure,
            'hot_aliased_strict': True, 'SCB_storage_unchanged': True,
            'weight_and_scale_values_unchanged': True, 'hot_layout_restored': True,
            'events': records}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu-int8-contract', action='store_true', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    write(args.output, cpu_int8_state_contract())
