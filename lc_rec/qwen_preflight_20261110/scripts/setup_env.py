"""Install preverified local wheels into a new task-owned CPU-tested overlay.

Network access and dependency resolution are deliberately absent here. Shared
Torch/CUDA, DeepSpeed, and bitsandbytes files are inherited read-only and their
versions and source hashes are checked before and after the operation.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

TASK = Path(__file__).resolve().parents[1]
CONFIG = TASK / "config"
BASE = "/data/bwwu/envs/grid-20261009-e61baf3/bin/python"
OFFLOAD = "/data/bwwu/envs/lc-rec-offload-20261108/lib/python3.10/site-packages"
LETTER = "/data/bwwu/envs/letter-20261110/lib/python3.10/site-packages"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 ** 2), b""):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=["qwen3", "qwen35"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path,
                        default=Path("/data/bwwu/lc-rec/qwen-preflight-20261110/wheels"))
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Environment installation must hide all CUDA devices"
    config_path = CONFIG / ("dependencies_" + args.family + ".json")
    config = json.loads(config_path.read_text())
    assert config["family"] == args.family and config["target_python"] == "3.10.22"
    assert config["inventory_sha256"] == sha(CONFIG / "dependencies_remote_inventory.json")
    env = Path(config["environment"])
    assert env == Path("/data/bwwu/envs/qwen-preflight-20261110-" + args.family)
    assert not env.exists(), "Do not replace an existing overlay; query the original setup job"
    assert Path(BASE).is_file() and Path(OFFLOAD).is_dir() and Path(LETTER).is_dir()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    steps = []
    process_env = os.environ.copy()
    process_env.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", USE_TF="0", USE_FLAX="0",
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PIP_NO_INDEX="1",
                       PIP_DISABLE_PIP_VERSION_CHECK="1", TOKENIZERS_PARALLELISM="false")

    def run(argv):
        start = time.monotonic()
        result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, env=process_env)
        steps.append({"argv": argv, "returncode": result.returncode, "output": result.stdout,
                      "seconds": time.monotonic() - start})
        atomic(out / "install-log.json", steps)
        print(result.stdout, flush=True)
        result.check_returncode()
        return result.stdout

    # Inspect files through distribution metadata without importing CUDA packages.
    shared_code = '''import sys,json,importlib.metadata as m,hashlib
from pathlib import Path
sys.path[:0]=[OFFLOAD,LETTER]
rows={}
for package,expected,module in [('torch','2.6.0','torch'),('bitsandbytes','0.45.5','bitsandbytes'),('deepspeed','0.16.9','deepspeed')]:
 d=m.distribution(package);assert d.version==expected,(package,d.version)
 p=Path(d.locate_file(module+'/__init__.py'))
 rows[package]={'version':d.version,'source_path':str(p),'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
print(json.dumps(rows,sort_keys=True))
'''.replace("OFFLOAD", repr(OFFLOAD)).replace("LETTER", repr(LETTER))
    shared_before = json.loads(run([BASE, "-B", "-c", shared_code]).splitlines()[-1])
    wheels = []
    for row in config["packages"]:
        assert not row["name"].startswith(("torch", "nvidia", "triton", "deepspeed", "bitsandbytes"))
        assert row["url"].startswith("https://files.pythonhosted.org/")
        wheel = args.wheelhouse / row["filename"]
        assert wheel.is_file() and not wheel.is_symlink()
        assert wheel.stat().st_size == row["bytes"] and sha(wheel) == row["sha256"], str(wheel)
        wheels.append({"path": str(wheel), "bytes": row["bytes"], "sha256": row["sha256"]})
    atomic(out / "verified-local-wheels.json", wheels)
    run([BASE, "-m", "venv", "--system-site-packages", str(env)])
    site = env / "lib/python3.10/site-packages"
    pth = site / "qwen-readonly-framework.pth"
    pth.write_text(OFFLOAD + "\n" + LETTER + "\n")
    interpreter = str(env / "bin/python")
    requirements = out / "hashed-local-requirements.txt"
    requirements.write_text("\n".join(row["path"] + " --hash=sha256:" + row["sha256"] for row in wheels) + "\n")
    # --ignore-installed prevents pip from attempting uninstall of an inherited
    # distribution. Every replacement is installed only in this new overlay.
    run([interpreter, "-m", "pip", "install", "--no-index", "--no-deps", "--ignore-installed",
         "--require-hashes", "--no-cache-dir", "--disable-pip-version-check", "-r", str(requirements)])
    expected = {row["name"]: row["version"] for row in config["packages"]}
    validation = '''import json,sys,inspect,importlib.metadata as m,hashlib
import torch,transformers,peft,accelerate,bitsandbytes
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.version import Version
from peft import LoraConfig
from transformers import MODEL_IMPORT
expected=EXPECTED
for name,version in expected.items():
 assert m.version(name)==version,(name,m.version(name),version)
 for raw in m.requires(name) or []:
  req=Requirement(raw);markers=default_environment();markers['extra']=''
  if req.marker is None or req.marker.evaluate(markers):
   assert Version(m.version(req.name)) in req.specifier,(name,raw,m.version(req.name))
assert 'ensure_weight_tying' in inspect.signature(LoraConfig).parameters
assert not torch.cuda.is_initialized()
assert not torch.cuda.is_available()
rows={name:{'version':m.version(name),'root':str(m.distribution(name).locate_file(''))} for name in expected}
shared={}
for package,module in [('torch','torch'),('bitsandbytes','bitsandbytes'),('deepspeed','deepspeed')]:
 d=m.distribution(package);p=d.locate_file(module+'/__init__.py')
 shared[package]={'version':d.version,'source_path':str(p),'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
print(json.dumps({'packages':rows,'shared':shared,'python':sys.version,'sys_path':sys.path,
 'cuda_initialized':torch.cuda.is_initialized(),'cuda_available':torch.cuda.is_available(),
 'ensure_weight_tying_available':True,'model_classes':str([MODEL_IMPORT])},sort_keys=True))
'''
    model_classes = "Qwen3ForCausalLM" if args.family == "qwen3" else "Qwen3_5ForCausalLM,Qwen3_5TextConfig"
    validation = validation.replace("MODEL_IMPORT", model_classes).replace("EXPECTED", repr(expected))
    checked = json.loads(run([interpreter, "-B", "-c", validation]).splitlines()[-1])
    assert checked["shared"] == shared_before, "Shared framework version/path/source changed"
    atomic(out / "environment.json", {"family": args.family, "path": str(env), "status": "PASS",
           "dependency_config_sha256": sha(config_path), "new_packages": config["packages"],
           "readonly_framework_sites": [OFFLOAD, LETTER], "shared_before": shared_before,
           "validation": checked, "task_owned_pth": {"path": str(pth), "sha256": sha(pth)},
           "system_installs": [], "global_shell_config_changes": [],
           "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
    print("ENVIRONMENT_PASS", flush=True)


if __name__ == "__main__":
    main()
