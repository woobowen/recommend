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


PEFT05_FORWARD_SHA256 = "df705367a1512d2f8bc3ff067a2617218a0a69ca5f151f2d2d1e233462d06088"
PEFT05_LORA_FILE_SHA256 = "cbcab4d64b5254d0e1a0018f5e870f7bd93f0178ed1b14d2b750e033f871aa01"
BNB0455_FORWARD_SHA256 = "d3f4146d34e7100027694d30fa36bac6c8691c244f865a15ff55e3c462b12665"
_PEFT05_ORIGINAL_FORWARD = None
_PEFT05_TARGET_RECORDS = {}


def peft05_out_of_place_forward(self, x):
    """Derived PEFT 0.5 int8 forward; preserve both differentiable branches."""
    import torch
    import bitsandbytes as bnb

    evidence = _PEFT05_TARGET_RECORDS.get(id(self))
    if evidence is None:
        raise RuntimeError("Unregistered PEFT 0.5 int8 module reached the guarded forward")
    evidence["calls"] += 1
    evidence["grad_enabled_calls" if torch.is_grad_enabled() else "no_grad_calls"] += 1
    # The pinned MRO's immediate original parent is bnb.nn.Linear8bitLt.
    # An explicit call preserves its implementation without a __class__ closure.
    result = bnb.nn.Linear8bitLt.forward(self, x)

    if self.disable_adapters or self.active_adapter not in self.lora_A.keys():
        evidence["base_only_calls"] += 1
        return result
    elif self.r[self.active_adapter] > 0:
        if not torch.is_autocast_enabled():
            expected_dtype = result.dtype

            if x.dtype != torch.float32:
                x = x.float()
            output = (
                self.lora_B[self.active_adapter](
                    self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
                ).to(expected_dtype)
                * self.scaling[self.active_adapter]
            )
        else:
            output = (
                self.lora_B[self.active_adapter](
                    self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
                )
                * self.scaling[self.active_adapter]
            )
        result = result + output
        evidence["lora_add_calls"] += 1
    else:
        evidence["rank_zero_calls"] += 1
    return result


def install_peft05_out_of_place(model):
    """Guard the frozen wheel and replace only its int8 class forward in memory.

    Installed packages and author files remain unchanged. Parameter identities,
    shapes, dtypes, requires_grad and state_dict keys are checked before/after.
    The registry makes actual calls inspectable, including checkpoint replays.
    """
    global _PEFT05_ORIGINAL_FORWARD
    import bitsandbytes as bnb
    from peft.tuners.lora import Linear8bitLt

    if importlib.metadata.version("peft") != "0.5.0":
        raise RuntimeError("Out-of-place compatibility is limited to PEFT == 0.5.0")
    if (importlib.metadata.version("bitsandbytes") != "0.45.5"
            or hashlib.sha256(inspect.getsource(bnb.nn.Linear8bitLt.forward).encode()).hexdigest() != BNB0455_FORWARD_SHA256):
        raise RuntimeError("bitsandbytes parent differs from the inspected 0.45.5 forward")
    original = _PEFT05_ORIGINAL_FORWARD or Linear8bitLt.forward
    source = inspect.getsource(original)
    source_file = inspect.getsourcefile(original)
    from pathlib import Path
    file_sha = hashlib.sha256(Path(source_file).read_bytes()).hexdigest()
    mro = [c.__module__ + "." + c.__qualname__ for c in Linear8bitLt.__mro__]
    expected_mro = ["peft.tuners.lora.Linear8bitLt", "bitsandbytes.nn.modules.Linear8bitLt",
                    "torch.nn.modules.linear.Linear", "torch.nn.modules.module.Module",
                    "peft.tuners.lora.LoraLayer", "peft.tuners.tuners_utils.BaseTunerLayer",
                    "abc.ABC", "builtins.object"]
    if (hashlib.sha256(source.encode()).hexdigest() != PEFT05_FORWARD_SHA256
            or file_sha != PEFT05_LORA_FILE_SHA256
            or str(inspect.signature(original)) != "(self, x: torch.Tensor)"
            or mro != expected_mro or Linear8bitLt.__bases__[0] is not bnb.nn.Linear8bitLt
            or source.count("result += output") != 1):
        raise RuntimeError("Installed PEFT source/signature/class/MRO differs from the frozen 0.5 wheel")
    if _PEFT05_ORIGINAL_FORWARD is not None and Linear8bitLt.forward is not peft05_out_of_place_forward:
        raise RuntimeError("A different runtime forward override is already present")
    targets = [(name, module) for name, module in model.named_modules() if isinstance(module, Linear8bitLt)]
    if not targets or any(type(module) is not Linear8bitLt for _, module in targets):
        raise RuntimeError("Expected exact PEFT 0.5 int8 target classes")

    def parameter_identity():
        return {name: (id(p), tuple(p.shape), str(p.dtype), p.requires_grad)
                for name, p in model.named_parameters()}

    before = parameter_identity()
    state_keys = set(model.state_dict())
    records = {}
    for name, module in targets:
        records[name] = {"calls": 0, "grad_enabled_calls": 0, "no_grad_calls": 0,
                         "base_only_calls": 0, "lora_add_calls": 0, "rank_zero_calls": 0}
        _PEFT05_TARGET_RECORDS[id(module)] = records[name]
    _PEFT05_ORIGINAL_FORWARD = original
    Linear8bitLt.forward = peft05_out_of_place_forward
    if before != parameter_identity() or state_keys != set(model.state_dict()):
        raise AssertionError("Forward installation changed parameter identity or state keys")
    return {"adaptation": "Probe-only PEFT 0.5 int8 differentiable out-of-place LoRA residual",
            "original_source_file": source_file, "original_file_sha256": file_sha,
            "original_source_sha256": PEFT05_FORWARD_SHA256,
            "original_signature": str(inspect.signature(original)),
            "derived_source_file": inspect.getsourcefile(peft05_out_of_place_forward),
            "derived_source_sha256": hashlib.sha256(inspect.getsource(peft05_out_of_place_forward).encode()).hexdigest(),
            "derived_signature": str(inspect.signature(peft05_out_of_place_forward)),
            "class": mro[0], "mro": mro,
            "parent_forward": "bitsandbytes.nn.Linear8bitLt.forward",
            "parent_forward_source_sha256": hashlib.sha256(inspect.getsource(bnb.nn.Linear8bitLt.forward).encode()).hexdigest(),
            "semantic_diff": ["super().forward(x) -> explicit pinned bnb.nn.Linear8bitLt.forward(self, x)",
                              "result += output -> result = result + output"],
            "target_count": len(targets), "module_calls": records,
            "parameter_and_state_identity_unchanged": True,
            "parameter_tensors_before_after": len(before), "state_dict_keys_before_after": len(state_keys)}


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
    original = _PEFT05_ORIGINAL_FORWARD or Linear8bitLt.forward
    source = inspect.getsource(original)
    if hashlib.sha256(source.encode()).hexdigest() != PEFT05_FORWARD_SHA256:
        raise RuntimeError("PEFT 0.5 precision path source differs from frozen forward")
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


def adapt_fastchat_unpad_arity():
    """Retain FastChat's four-result API when flash-attn adds redundant seqused.

    FastChat passes no unused_mask. The first four outputs retain identical
    tensor objects/values. Only the extra per-sequence used length is dropped.
    This does not replace any attention kernel or change the padding mask.
    """
    import torch
    import fastchat.train.llama2_flash_attn_monkey_patch as patch
    original = patch.unpad_input
    source = inspect.getsource(original)
    record = {"target": "fastchat.train.llama2_flash_attn_monkey_patch.unpad_input",
              "original_signature": str(inspect.signature(original)),
              "original_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
              "calls": 0, "observed_arities": [],
              "change": "Return original first four results; unused_mask is forbidden; extra used_seqlens checked once then ignored"}

    @functools.wraps(original)
    def four_results(hidden_states, attention_mask, *args, **kwargs):
        if args or kwargs:
            raise RuntimeError("This compatibility scope accepts only FastChat's original two positional arguments")
        result = original(hidden_states, attention_mask)
        if len(result) not in (4, 5):
            raise RuntimeError(f"Unexpected unpad_input return arity {len(result)}")
        record["calls"] += 1
        if len(result) not in record["observed_arities"]:
            record["observed_arities"].append(len(result))
            if len(result) == 5:
                expected = attention_mask.sum(dim=-1, dtype=torch.int32)
                if not torch.equal(result[4], expected) or not torch.equal(result[2][1:] - result[2][:-1], expected):
                    raise RuntimeError("Extra seqused is not redundant with original mask/cu_seqlens")
                record["extra_seqused_equals_mask_sum_and_cu_deltas"] = True
                record["extra_shape"] = list(result[4].shape)
                record["extra_dtype"] = str(result[4].dtype)
        return result[:4]

    patch.unpad_input = four_results
    return record
