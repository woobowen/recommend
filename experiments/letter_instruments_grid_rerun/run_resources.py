"""Physical-device admission without loading CUDA or changing system state."""
import importlib.metadata
import os
import shutil
import subprocess

from pipeline import BASE, JOBS


def resources():
    gpu = os.environ["CUDA_VISIBLE_DEVICES"]
    assert gpu.startswith("GPU-") and "," not in gpu
    fields = subprocess.check_output(
        ["nvidia-smi", "-i", gpu, "--query-gpu=index,name,uuid,memory.total,memory.used",
         "--format=csv,noheader,nounits"], text=True).strip().split(", ")
    assert len(fields) == 5 and fields[2] == gpu and "4090" in fields[1] and int(fields[4]) <= 256, fields
    free = {str(path): shutil.disk_usage(path).free for path in (BASE, JOBS)}
    assert min(free.values()) > 2 * 1024**3
    names = ("torch", "lightning", "hydra-core", "tensorflow-cpu", "transformers", "numpy", "torchmetrics")
    return {"gpu": dict(zip(("index", "name", "uuid", "total_mib", "used_mib"), fields)), "free_bytes": free,
            "dependencies": {name: importlib.metadata.version(name) for name in names}}
