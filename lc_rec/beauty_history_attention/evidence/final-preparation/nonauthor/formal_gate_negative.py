"""Negative formal launch review only. No authorization/test approval file is issued."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

snap = Path(sys.argv[1]).resolve()
commit = sys.argv[2]
package = "lc_rec.beauty_history_attention"
based = snap / "lc_rec" / "beauty_history_attention"
vendor = snap / "lc_rec" / "letter" / "vendor" / "LETTER" / "LETTER-LC-Rec"
env = os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONOPTIMIZE": "0", "PYTHONDONTWRITEBYTECODE": "1"}
for key in ("GRID_JOB_DIR","GRID_JOB_ID","RANK","WORLD_SIZE","LOCAL_RANK"):
    env.pop(key,None)
results={}
with tempfile.TemporaryDirectory(prefix="beauty-review-gate-") as path:
    tmp=Path(path)
    generation=tmp/"generation-has-no-authority"
    training=tmp/"train-has-no-authority"
    wrongscope=tmp/"formal-b-no-authorization"/"training"
    train_args=("--source-root", str(vendor), "--data-root", "/nonexistent-review-data",
                "--model", "/nonexistent-review-model", "--config", str(based/"config_bidirectional.json"),
                "--run-commit", commit)
    cases={
        "direct_generation": (generation,["evaluate","generate","--purpose","formal","--adapter","/nonexistent-review-adapter",
                          "--run-commit",commit,"--max-seconds","108000","--output",str(generation)],
                          "requires the reviewed plan and distinct user authorization"),
        "direct_training": (training,["train","--output",str(training),*train_args],
                          "requires the frozen plan and separate user authorization"),
        "preflight_cannot_write_formal_output": (wrongscope,["train","--output",str(wrongscope),"--boundary-preflight",*train_args],
                          "cannot write a formal B output"),
    }
    for name,(target,argv,reason) in cases.items():
        command=[sys.executable,"-B","-m",package+"."+argv[0],*argv[1:]]
        finished=subprocess.run(command,cwd=snap,env=env,text=True,capture_output=True,timeout=20)
        assert finished.returncode!=0 and reason in finished.stderr, finished.stderr
        assert not target.exists()
        results[name]=dict(returncode=finished.returncode,selected_error=finished.stderr.strip().splitlines()[-1],
                           output_absent=not target.exists())
print(json.dumps(dict(status="PASS",no_test_authorization_created=True,source_commit=commit,results=results,
                       source_files_sha256={name:hashlib.sha256((based/name).read_bytes()).hexdigest()
                                            for name in ("train.py","evaluate.py","formal.py")}),indent=2))
