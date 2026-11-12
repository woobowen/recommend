"""Approved A0 FA2 backward adaptation; no tensor observers or global controls."""
import functools
import hashlib
import importlib.metadata
import inspect
import os
from pathlib import Path

INTERFACE_SHA = 'd8eaf989a5401fedd23d11a336a7f38e67596d0a4d5a279df14cbe2d7de2ed75'
FASTCHAT_SHA = 'de6b3211ea8a7a860f3bea6fc1529c5f0f58732421a156f1b8e62881b58afe6e'
EXTENSION_SHA = '49bd9d3367051b74dd86039532a3786cdf324fd88befcd0b7b3f3aeaae1abbd2'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def install(config, event):
    import torch
    import flash_attn.flash_attn_interface as fa
    import fastchat.train.llama2_flash_attn_monkey_patch as fastchat
    assert config['flash_backward_deterministic'] is True
    assert importlib.metadata.version('flash-attn') == '2.7.4.post1'
    assert digest(fa.__file__) == INTERFACE_SHA
    assert digest(fastchat.__file__) == FASTCHAT_SHA
    assert digest(fa.flash_attn_gpu.__file__) == EXTENSION_SHA
    assert not torch.are_deterministic_algorithms_enabled()
    assert not os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    report = {'enabled': True, 'recipe': config['recipe_identity'],
              'interface_sha256': INTERFACE_SHA, 'fastchat_sha256': FASTCHAT_SHA,
              'extension_sha256': EXTENSION_SHA, 'version': '2.7.4.post1',
              'torch_global_deterministic': False, 'cublas_workspace_config': None,
              'tensor_observers': False, 'calls': {}, 'signatures': {}}

    for name in ('flash_attn_func', 'flash_attn_varlen_kvpacked_func'):
        original = getattr(fastchat, name)
        assert original is getattr(fa, name)
        signature = inspect.signature(original)
        assert signature.parameters['deterministic'].default is False
        report['signatures'][name] = str(signature)
        report['calls'][name] = 0

        def forward_wrapper(fn, label, sig):
            @functools.wraps(fn)
            def forward(*args, **kwargs):
                assert 'deterministic' not in sig.bind(*args, **kwargs).arguments
                kwargs['deterministic'] = True
                result = fn(*args, **kwargs)
                report['calls'][label] += 1
                if report['calls'][label] == 1:
                    event('formal_fa_forward', function=label, deterministic=True)
                return result
            return forward
        setattr(fastchat, name, forward_wrapper(original, name, signature))

    # Guard both actual FastChat branches. Check every dispatch, record only the
    # first successful call of each ABI. No synchronization or tensor copying.
    for owner, name, count, position, label in (
        (fa, '_wrapped_flash_attn_backward', 17, 16, 'native_dense'),
        (fa, '_wrapped_flash_attn_varlen_backward', 21, 20, 'native_varlen'),
        (fa.flash_attn_gpu, 'bwd', 19, 16, 'extension_dense'),
        (fa.flash_attn_gpu, 'varlen_bwd', 24, 21, 'extension_varlen'),
    ):
        original = getattr(owner, name)
        report['calls'][label] = 0

        def backward_wrapper(fn, key, argc, index):
            @functools.wraps(fn)
            def backward(*args, **kwargs):
                assert len(args) == argc and args[index] is True, key
                result = fn(*args, **kwargs)
                report['calls'][key] += 1
                if report['calls'][key] == 1:
                    event('formal_fa_backward_dispatch', boundary=key,
                          deterministic=args[index], positional_arguments=len(args),
                          deterministic_position=index, extension_sha256=EXTENSION_SHA)
                return result
            return backward
        setattr(owner, name, backward_wrapper(original, label, count, position))
    event('formal_fa_installed', **report)
    return report


def assert_active(report):
    assert report['enabled'] is True
    assert report['calls']['native_varlen'] > 0
    assert report['calls']['extension_varlen'] == report['calls']['native_varlen']
    assert report['calls']['extension_dense'] == report['calls']['native_dense']
