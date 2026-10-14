"""Read pickle opcode names without executing checkpoint objects."""
import argparse
import json
from pathlib import Path
import pickletools
import zipfile


def globals_in_checkpoint(path):
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.endswith("/data.pkl") or n == "data.pkl"]
        if len(names) != 1:
            raise ValueError("Expected a unique PyTorch metadata pickle")
        data = z.read(names[0])
    globals_ = sorted({arg.replace("\n", " ") for op,arg,_ in pickletools.genops(data) if op.name == "GLOBAL"})
    if any(op.name == "STACK_GLOBAL" for op,_,_ in pickletools.genops(data)):
        raise ValueError("STACK_GLOBAL requires separate static review")
    return {"path": str(path), "pickle_bytes": len(data), "global_names": globals_,
            "objects_executed": False}


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("checkpoint", type=Path)
    args = p.parse_args()
    print(json.dumps(globals_in_checkpoint(args.checkpoint), indent=2))
