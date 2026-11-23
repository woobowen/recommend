"""Disable only optional bnb SwitchBack Triton modules on a compilerless host.

Int8 Linear8bitLt uses the installed bitsandbytes CUDA binaries.
No author code or installed package is edited.
"""
import sys
import types


def disable_optional_bnb_triton():
    name = 'bitsandbytes.triton.triton_utils'
    if 'bitsandbytes' in sys.modules:
        raise RuntimeError('Apply this import guard before importing bitsandbytes')
    guard = types.ModuleType(name)
    guard.is_triton_available = lambda: False
    sys.modules[name] = guard
    # This harness never uses a DeepSpeed engine. Accelerate otherwise imports
    # an unrelated inherited installation while saving an ordinary PEFT model.
    import accelerate.utils.other
    accelerate.utils.other.is_deepspeed_available = lambda: False


def restore_legacy_bnb_state(model):
    """PEFT 0.12 reads a flag removed by bnb 0.46; preserve default False."""
    import bitsandbytes as bnb
    count = 0
    for layer in model.modules():
        if isinstance(layer, bnb.nn.Linear8bitLt) and not hasattr(layer.state, 'memory_efficient_backward'):
            layer.state.memory_efficient_backward = False
            count += 1
    return count
