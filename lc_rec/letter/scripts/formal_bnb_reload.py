"""Guarded hot-cache SCB alias for native strict bnb 0.45.5 checkpoint load.

No package code, quantization, weights or arithmetic is replaced. The native
loader expects SCB on weight although native forward moved it to state. Make
that existing buffer visible for the duration of the original strict load.
"""
from contextlib import contextmanager
import hashlib
import importlib.metadata
import inspect
from pathlib import Path

MODULES_SHA='e97af1a69794f192bf58e0154efb660f3d2ed0a00dade672c7d8f9a3d14b9e53'


def layout(t):
    return (t.data_ptr(),tuple(t.shape),tuple(t.stride()),t.storage_offset(),t.dtype,t.device)


@contextmanager
def hot_scb_alias(model, event, expected_modules=224):
    import torch
    import bitsandbytes as bnb
    assert importlib.metadata.version('bitsandbytes')=='0.45.5'
    source=inspect.getsourcefile(bnb.nn.Linear8bitLt)
    assert hashlib.sha256(Path(source).read_bytes()).hexdigest()==MODULES_SHA
    targets=[(n,m) for n,m in model.named_modules() if isinstance(m,bnb.nn.Linear8bitLt)]
    assert len(targets)==expected_modules
    changes=[]
    identities={n:(id(m.weight),m.weight.dtype,tuple(m.weight.shape),m.weight.requires_grad) for n,m in targets}
    try:
        for name,module in targets:
            assert module.state.has_fp16_weights is False
            assert module.weight.dtype==torch.int8 and not module.weight.requires_grad
            if module.weight.SCB is None:
                assert module.weight.CB is None
                assert module.state.CB is not None and module.state.SCB is not None
                assert layout(module.state.CB)==layout(module.weight.data)
                scale=module.state.SCB
                assert tuple(scale.shape)==(module.weight.shape[0],) and tuple(scale.stride())==(1,)
                assert scale.dtype==torch.float32 and scale.device==module.weight.device and not scale.requires_grad
                changes.append((name,module,module.weight.SCB))
                module.weight.SCB=module.state.SCB
        event('bnb_hot_scb_alias_begin',modules=len(targets),hot_modules=len(changes),
              modules_source_sha256=MODULES_SHA,scope='native strict checkpoint load only; no requantization')
        yield
        for name,module in targets:
            assert identities[name]==(id(module.weight),module.weight.dtype,tuple(module.weight.shape),module.weight.requires_grad)
        for name,module,_ in changes:
            assert layout(module.state.CB)==layout(module.weight.data)
            assert module.state.SCB is module.weight.SCB
        event('bnb_hot_scb_alias_loaded',modules=len(targets),hot_modules=len(changes),
              parameter_identity_and_cache_aliases_preserved=True)
    finally:
        for _,module,original in changes:module.weight.SCB=original


def cpu_regression():
    """Actual installed native loader; synthetic buffers, no CUDA execution."""
    import torch
    import bitsandbytes as bnb
    assert not torch.cuda.is_initialized()
    from peft.tuners.lora import Linear8bitLt as Peft8
    assert Peft8._load_from_state_dict is bnb.nn.Linear8bitLt._load_from_state_dict
    def make(hot):
        m=bnb.nn.Linear8bitLt(8,8,bias=False,has_fp16_weights=False)
        m.weight.data=torch.zeros((8,8),dtype=torch.int8)
        if hot:
            m.weight.CB=None;m.weight.SCB=None
            m.state.CB=m.weight.data;m.state.SCB=torch.ones(8,dtype=torch.float32)
        else:
            m.weight.CB=m.weight.data;m.weight.SCB=torch.ones(8,dtype=torch.float32)
        return m
    broken=make(True)
    saved={k:v.clone() for k,v in broken.state_dict().items()}
    saved['weight'].fill_(3);saved['SCB'].fill_(2)
    try:broken.load_state_dict(saved,strict=True)
    except RuntimeError as error:
        assert 'Loading a quantized checkpoint into non-quantized Linear8bitLt' in str(error)
        original_error=str(error)
    else:raise AssertionError('Native hot-SCB failure was not reproduced')
    m=make(True)
    events=[]
    with hot_scb_alias(m,lambda kind,**kw:events.append({'event':kind,**kw}),expected_modules=1):
        result=m.load_state_dict(saved,strict=True)
        assert not result.missing_keys and not result.unexpected_keys
        assert torch.equal(m.weight.data,saved['weight']) and torch.equal(m.state.CB,saved['weight'])
        assert torch.equal(m.state.SCB,saved['SCB'])
    assert m.weight.SCB is None and m.weight.CB is None and not torch.cuda.is_initialized()
    cold=make(False)
    with hot_scb_alias(cold,lambda kind,**kw:events.append({'event':kind,**kw}),expected_modules=1):
        result=cold.load_state_dict(saved,strict=True)
        assert not result.missing_keys and not result.unexpected_keys
    assert torch.equal(cold.weight,saved['weight']) and torch.equal(cold.weight.SCB,saved['SCB'])
    assert cold.state.SCB is None and cold.weight.CB is not None
    return {'status':'PASS','synthetic_cpu_only':True,'original_error_reproduced':original_error,
            'strict_complete_weight_SCB_exact':True,'hot_cache_aliases_and_parameter_identity_preserved':True,
            'independent_instances_for_failure_and_success':True,'cold_native_behavior_unchanged':True,
            'actual_peft_subclass_inherits_identical_native_loader':True,
            'events':events,'modules_source_sha256':MODULES_SHA}
