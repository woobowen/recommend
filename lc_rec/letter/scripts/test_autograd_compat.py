#!/usr/bin/env python3
"""CPU synthetic tests of the actual guarded PEFT class forward, not CUDA acceptance."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-commit", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result = {"status": "RUNNING", "scope": "CPU synthetic custom-Function base with real PEFT class/derived method",
              "RUN_COMMIT": args.run_commit, "argv": sys.argv, "tests": []}
    started = time.monotonic()
    original_base = None
    try:
        import torch
        import torch.nn.functional as F
        from torch.utils.checkpoint import checkpoint
        import bitsandbytes as bnb
        from bitsandbytes.autograd._functions import MatmulLtState
        from peft.tuners.lora import Linear8bitLt
        import compat
        if torch.cuda.is_initialized():
            raise AssertionError("This is deliberately a CPU-only synthetic test")
        torch.manual_seed(42)

        class BaseView(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, weight):
                ctx.save_for_backward(weight)
                value = F.linear(x, weight)
                return value.reshape(*x.shape[:-1], weight.shape[0])

            @staticmethod
            def backward(ctx, grad):
                (weight,) = ctx.saved_tensors
                return grad.matmul(weight), None

        # Bypass only the CUDA-dependent constructor for a CPU test fixture.
        # The class, original super closure and actual derived forward are real.
        layer = Linear8bitLt.__new__(Linear8bitLt)
        torch.nn.Module.__init__(layer)
        layer.weight = bnb.nn.Int8Params(torch.randn(5, 7), requires_grad=False, has_fp16_weights=False)
        layer.state = MatmulLtState()
        layer.register_parameter("bias", None)
        layer.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(7, 3, bias=False)})
        layer.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(3, 5, bias=False)})
        layer.lora_dropout = torch.nn.ModuleDict({"default": torch.nn.Dropout(.05)})
        layer.scaling = {"default": 2.0}
        layer.r = {"default": 3}
        layer.active_adapter = "default"
        layer.disable_adapters = False
        model = torch.nn.ModuleDict({"target": layer})
        rng = torch.get_rng_state().clone()
        result["patch"] = compat.install_peft05_out_of_place(model)
        assert torch.equal(rng, torch.get_rng_state()), "Installation consumed initialization RNG"
        original_base = bnb.nn.Linear8bitLt.forward
        result["real_parent_source_sha256"] = hashlib.sha256(inspect.getsource(original_base).encode()).hexdigest()

        def synthetic_base(self, x):
            return BaseView.apply(x, self.weight)

        bnb.nn.Linear8bitLt.forward = synthetic_base
        x0 = torch.randn(2, 4, 7)
        upstream = torch.randn(2, 4, 5)
        x = x0.clone().requires_grad_(True)
        try:
            compat._PEFT05_ORIGINAL_FORWARD(layer, x)
        except RuntimeError as error:
            assert "view" in str(error) and "inplace" in str(error), str(error)
            result["tests"].append({"test": "original_custom_function_view_inplace", "status": "EXPECTED_ERROR", "error": str(error)})
        else:
            raise AssertionError("Original residual failed to reproduce the protected view mutation")

        def reference(x):
            return F.linear(x, layer.weight) + layer.scaling["default"] * layer.lora_B["default"](
                layer.lora_A["default"](layer.lora_dropout["default"](x)))

        def compare(checkpointed=False, zero_b=False):
            if zero_b:
                with torch.no_grad():
                    layer.lora_B["default"].weight.zero_()
            x = x0.clone().requires_grad_(True)
            torch.manual_seed(1907)
            y = checkpoint(layer, x, use_reentrant=True) if checkpointed else layer(x)
            params = [x, layer.lora_A["default"].weight, layer.lora_B["default"].weight]
            if checkpointed:
                layer.zero_grad(set_to_none=True)
                (y * upstream).sum().backward()
                gradients = [x.grad.clone(), params[1].grad.clone(), params[2].grad.clone()]
            else:
                gradients = torch.autograd.grad((y * upstream).sum(), params)
            xr = x0.clone().requires_grad_(True)
            torch.manual_seed(1907)
            yr = reference(xr)
            reference_gradients = torch.autograd.grad((yr * upstream).sum(), [xr, *params[1:]])
            differences = [float((a-b).abs().max()) for a,b in zip(gradients, reference_gradients)]
            # FP32: same linear/add graph and fixed dropout RNG; ordinary roundoff tolerance.
            assert torch.allclose(y, yr, atol=1e-6, rtol=1e-6)
            for a,b in zip(gradients, reference_gradients):
                assert torch.allclose(a,b,atol=1e-6,rtol=1e-6)
            assert gradients[0].abs().max() > 0 and gradients[2].abs().max() > 0
            assert (gradients[1].abs().max() == 0) if zero_b else (gradients[1].abs().max() > 0)
            result["tests"].append({"test": "reference_forward_and_gradients", "checkpointed": checkpointed,
                "B_zero_initialized": zero_b, "dropout": .05, "scale": 2.0,
                "output_max_abs_difference": float((y-yr).abs().max()),
                "input_A_B_gradient_max_abs_differences": differences,
                "input_A_B_gradient_max_abs": [float(g.abs().max()) for g in gradients],
                "tolerance": {"atol": 1e-6, "rtol": 1e-6, "dtype": "float32"}, "status": "PASS"})

        compare()
        compare(checkpointed=True)
        compare(checkpointed=True, zero_b=True)
        for branch in ("disable_adapters", "missing_active_adapter", "rank_zero"):
            layer.disable_adapters = branch == "disable_adapters"
            layer.active_adapter = "missing" if branch == "missing_active_adapter" else "default"
            layer.r["default"] = 0 if branch == "rank_zero" else 3
            calls = []
            handles = [m.register_forward_hook(lambda *a: calls.append(1)) for m in [layer.lora_A["default"], layer.lora_B["default"]]]
            x = x0.clone().requires_grad_(True)
            y = layer(x)
            gradient = torch.autograd.grad((y*upstream).sum(), x)[0]
            assert torch.equal(y, F.linear(x, layer.weight)) and not calls
            assert torch.allclose(gradient, upstream.matmul(layer.weight), atol=1e-6, rtol=1e-6)
            for h in handles:
                h.remove()
            result["tests"].append({"test": branch, "base_output_and_input_gradient": "PASS", "lora_calls": len(calls), "status": "PASS"})
        record = result["patch"]["module_calls"]["target"]
        assert record["grad_enabled_calls"] > 0 and record["no_grad_calls"] >= 2
        result["tests"].append({"test": "checkpoint_recomputation_observed", "calls": record, "status": "PASS"})
        result["cuda_initialized"] = torch.cuda.is_initialized()
        result["CUDA_bnb_and_BF16_acceptance"] = "NOT_TESTED_REQUIRES_REAL_GPU_PROBE"
        result["status"] = "PASS_CPU_SYNTHETIC"
    except BaseException as error:
        result["status"] = "FAIL"
        result["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        print(result["error"]["traceback"], flush=True)
    finally:
        if original_base is not None:
            bnb.nn.Linear8bitLt.forward = original_base
        result["elapsed_seconds"] = time.monotonic()-started
        (output/"autograd_cpu.json").write_text(json.dumps(result,indent=2)+"\n")
    return 0 if result["status"] == "PASS_CPU_SYNTHETIC" else 1


if __name__ == "__main__":
    raise SystemExit(main())
