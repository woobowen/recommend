"""Narrow, observable bnb compatibility for PEFT 0.5 and 0.10 APIs.

No author file is changed. Legacy API repairs do not change the model definition.
The separately approved PEFT 0.5 precision shim is confined to LoRA A linear
calls after an observed FP32-input/BF16-weight failure. All changes are reported.
"""
from __future__ import annotations

import inspect
import functools
import hashlib
import importlib.metadata
import types


def restore_legacy_bnb_state_field(model):
    """Supply an obsolete False flag only if this PEFT version reads it.

    Both inspected PEFT releases read the flag from old bnb state. PEFT 0.5
    additionally forwards it into the bnb constructor; PEFT 0.10 keeps the
    existing base layer instead. Modern bnb no longer stores/uses the flag.
    """
    import bitsandbytes as bnb
    try:
        from peft.tuners.lora.bnb import dispatch_bnb_8bit, Linear8bitLt as PeftLinear8bitLt
        reader = dispatch_bnb_8bit
    except (ImportError, ModuleNotFoundError):
        from peft.tuners.lora import LoraModel, Linear8bitLt as PeftLinear8bitLt
        reader = LoraModel._create_new_module
    source = inspect.getsource(reader)
    changes = []
    if "memory_efficient_backward" not in source:
        return changes
    changes.append({"legacy_reader": reader.__module__ + "." + reader.__qualname__,
                    "reader_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                    "state_field_compatibility_required": True})
    peft_constructor = inspect.getsource(PeftLinear8bitLt.__init__)
    original_constructor = bnb.nn.Linear8bitLt.__init__
    signature = inspect.signature(original_constructor)
    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
    if ("memory_efficient_backward" in peft_constructor
            and "memory_efficient_backward" not in signature.parameters and not has_kwargs):
        record = {"callable": "bitsandbytes.nn.Linear8bitLt.__init__",
                  "original_signature": str(signature),
                  "original_source_sha256": hashlib.sha256(inspect.getsource(original_constructor).encode()).hexdigest(),
                  "peft_constructor_source_sha256": hashlib.sha256(peft_constructor.encode()).hexdigest(),
                  "change": "Remove only the explicitly False legacy memory_efficient_backward keyword; reject every other value",
                  "calls_with_false_keyword_removed": 0}

        @functools.wraps(original_constructor)
        def compatible_constructor(self, *args, **kwargs):
            if "memory_efficient_backward" in kwargs:
                value = kwargs.pop("memory_efficient_backward")
                if value is not False:
                    raise ValueError("Legacy memory_efficient_backward compatibility accepts only the boolean False")
                record["calls_with_false_keyword_removed"] += 1
            return original_constructor(self, *args, **kwargs)

        compatible_constructor.__letter_legacy_kw_shim__ = True
        bnb.nn.Linear8bitLt.__init__ = compatible_constructor
        changes.append(record)
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear8bitLt) and not hasattr(module.state, "memory_efficient_backward"):
            module.state.memory_efficient_backward = False
            changes.append({"module": name, "field": "state.memory_efficient_backward", "value": False,
                            "reason": "Inspected PEFT legacy read; bnb removed unused constructor-default False field"})
    return changes


def peft05_lora_a_targets(model):
    """Identify only the inspected PEFT 0.5 int8 branch's ordinary LoRA A."""
    import torch
    from peft.tuners.lora import Linear8bitLt

    if importlib.metadata.version("peft") != "0.5.0":
        raise RuntimeError("This approved precision adaptation is pinned to PEFT 0.5.0")
    source = inspect.getsource(Linear8bitLt.forward)
    if "x = x.float()" not in source or "not torch.is_autocast_enabled()" not in source:
        raise RuntimeError("PEFT 0.5 int8 forward does not match the inspected FP32 branch")
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, Linear8bitLt) and "default" in module.lora_A:
            linear = module.lora_A["default"]
            if type(linear) is not torch.nn.Linear or "input" not in inspect.signature(linear.forward).parameters:
                raise RuntimeError(f"Unexpected LoRA A implementation/signature: {name}")
            targets.append((name + ".lora_A.default", linear))
    if not targets:
        raise RuntimeError("No inspected PEFT 0.5 int8 LoRA A modules found")
    identity = {"class": Linear8bitLt.__module__ + "." + Linear8bitLt.__qualname__,
                "forward_signature": str(inspect.signature(Linear8bitLt.forward)),
                "forward_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "target_count": len(targets)}
    return targets, identity


def install_peft05_lora_a_bf16_autocast(model, original_failure):
    """Apply the user's approved probe-only adaptation to necessary A calls.

    Parent PEFT forward, bnb base projection, LoRA B and LETTER CE retain their
    original contexts. The instance forward override has no state_dict entries
    and therefore is not embedded in the saved adapter/model parameters.
    """
    import torch

    if not original_failure.get("matched_dtype_conflict"):
        raise RuntimeError("A-only BF16 adaptation requires a recorded original dtype failure")
    targets, identity = peft05_lora_a_targets(model)
    records = []
    for name, linear in targets:
        if linear.weight.dtype != torch.bfloat16 or linear.weight.device.type != "cuda":
            raise RuntimeError(f"Approved A-only path requires a CUDA BF16 weight: {name}")
        original = linear.forward
        record = {"module": name, "weight_dtype": str(linear.weight.dtype),
                  "original_forward_signature": str(inspect.signature(original)),
                  "original_forward_source_sha256": hashlib.sha256(inspect.getsource(original).encode()).hexdigest(),
                  "amp_scope": "Only this LoRA A nn.Linear call; base, parent PEFT forward, LoRA B and CE excluded",
                  "calls": 0, "adapted_calls": 0, "observed_dtype_combinations": []}

        def make_forward(original_forward, evidence):
            @functools.wraps(original_forward)
            def forward(this, input, *args, **kwargs):
                evidence["calls"] += 1
                outside_amp = torch.is_autocast_enabled()
                necessary = input.dtype == torch.float32 and this.weight.dtype == torch.bfloat16
                if necessary:
                    if outside_amp:
                        raise RuntimeError("Unexpected external autocast: approved scope is LoRA A only")
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        output = original_forward(input, *args, **kwargs)
                        inner_amp = torch.is_autocast_enabled()
                    evidence["adapted_calls"] += 1
                else:
                    output = original_forward(input, *args, **kwargs)
                    inner_amp = outside_amp
                combination = {"input_dtype": str(input.dtype), "weight_dtype": str(this.weight.dtype),
                               "output_dtype": str(output.dtype), "outer_autocast": outside_amp,
                               "linear_autocast": inner_amp, "autocast_after_call": torch.is_autocast_enabled()}
                if necessary and (output.dtype != torch.bfloat16 or not inner_amp or torch.is_autocast_enabled()):
                    raise RuntimeError("LoRA A-only BF16 precision adaptation did not preserve its asserted scope/dtype")
                if "first_call" not in evidence:
                    evidence["first_call"] = combination.copy()
                evidence["last_call"] = combination.copy()
                if combination not in evidence["observed_dtype_combinations"]:
                    evidence["observed_dtype_combinations"].append(combination)
                return output
            return forward

        linear.forward = types.MethodType(make_forward(original, record), linear)
        records.append(record)
    return {"adaptation": "User-approved probe-only PEFT 0.5 LoRA A BF16 autocast after original dtype error",
            "parent_forward_identity": identity, "targets": records,
            "state_dict_effect": "No keys/parameters added; runtime instance forward override only"}
