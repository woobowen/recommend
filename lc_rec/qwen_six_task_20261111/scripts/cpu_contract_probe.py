"""Exercise the exact probe tensor digest on CPU scalar and BF16 contracts."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import struct
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    import torch
    assert not torch.cuda.is_initialized()
    source = Path(__file__).with_name("probe.py")
    tree = ast.parse(source.read_text())
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "tensor_digest"]
    assert len(functions) == 1
    namespace = {"torch": torch, "hashlib": hashlib}
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), str(source), "exec"), namespace)
    examples = [
        ("float32_scalar", torch.tensor(1.25), struct.pack("=f", 1.25)),
        ("int64_scalar", torch.tensor(7, dtype=torch.int64), struct.pack("=q", 7)),
        ("bf16_vector", torch.tensor([1., -2.], dtype=torch.bfloat16), struct.pack("=HH", 0x3f80, 0xc000)),
        ("float32_vector", torch.tensor([1., -2.]), struct.pack("=ff", 1., -2.)),
    ]
    results = []
    for name, tensor, expected in examples:
        actual = namespace["tensor_digest"](tensor)
        assert actual == hashlib.sha256(expected).hexdigest(), name
        results.append({"case": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype), "sha256": actual, "pass": True})
    assert not torch.cuda.is_initialized()
    report = {"status": "PASS_CPU_CONTRACT", "run_commit": os.environ["RUN_COMMIT"], "argv": sys.argv,
              "probe_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "torch": torch.__version__,
              "cuda_initialized": False, "test_method": "Compile the exact nested tensor_digest AST and compare with independent struct-packed known bytes", "cases": results}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "contracts.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
