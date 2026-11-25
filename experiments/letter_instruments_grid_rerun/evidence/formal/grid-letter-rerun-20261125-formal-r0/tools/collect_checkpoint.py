"""Copy only quantizer state or best checkpoint metadata from the existing job."""
import json
from pathlib import Path
import shlex
import sys

from collect import HERE, archive

PYTHON = "/data/bwwu/envs/grid-20261009-e61baf3/bin/python"
mode = sys.argv[1]
if mode == "best":
    source = json.loads((HERE / "cutoff/sample.json").read_text())["checkpoints"]
    assert len(source) == 1, source
    name, evidence = next(iter(source.items()))
    args = [name, json.dumps(evidence)]
else:
    assert mode == "quantizer"
    args = []
out = HERE / (mode + "-checkpoint")
out.mkdir(exist_ok=False)
command = shlex.join([PYTHON, "-B", "-", mode, *args])
archive(mode, out, command, (HERE / "checkpoint_remote.py").read_bytes())
print(out)
