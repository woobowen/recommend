#!/bin/bash
set -euo pipefail
support=/home/addaswsw/recommend-support/beauty-history-attention-20261127
commit="$1"
stage_name="$2"
attempt="$3"
timeout="$4"
shift 4
stage="$(cat "$support/windows-stage")"
remote=/data/bwwu/lc-rec/beauty-history-attention-20261127
snapshot="$remote/releases/$commit"
job_id="letter-beauty-v2-history-attn-$stage_name-$attempt"
jobroot=/home/bwwu/grid-audits/sid-extensions-p0/jobs
runner=/home/bwwu/grid-audits/sid-extensions-p0/scripts/runner-beauty-cbb963fab68c34cc.py
ssh=/mnt/c/Windows/System32/OpenSSH/ssh.exe
scp=/mnt/c/Windows/System32/OpenSSH/scp.exe
output="$remote/$stage_name-$attempt"
request="$stage/$job_id.json"
python3 - "$commit" "$stage_name" "$timeout" "$job_id" "$snapshot" "$output" "$support" "$request" "$@" <<'PY'
from pathlib import Path
import json
import sys

commit, stage, timeout, job, snapshot, output, support, request, *options = sys.argv[1:]
members = Path(support, f"export-{commit}", "SOURCE.sha256").read_text().splitlines()
gpu = "" if stage in ("cpu", "metrics") else "GPU-59cac3bd-3645-668b-79c8-839e0d878893"
package = "lc_rec.beauty_history_attention.preflight"
program = "/data/bwwu/envs/letter-20261110/bin/python"
request_object = dict(
    schema=1, host="zwlab-01", uid=1005, user="bwwu", namespace="sid-extensions-p0/new-server",
    deployment_sha256="e610be95a47894472c3f6b8396af8eb48fb1a998696a3a4914af2d886cc00877",
    capture_version=commit, job_id=job, cwd=snapshot, timeout_seconds=int(timeout),
    argv=[program, "-B", "-m", package, "--stage", stage, "--commit", commit, "--output", output, *options],
    env=dict(CUDA_VISIBLE_DEVICES=gpu, CUDA_DEVICE_ORDER="PCI_BUS_ID", PYTHONOPTIMIZE="0",
             PYTHONDONTWRITEBYTECODE="1", TORCHINDUCTOR_COMPILE_THREADS="1", PYTHONUNBUFFERED="1", USE_TF="0", USE_FLAX="0",
             HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
             OPENBLAS_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false",
             CUDA_HOME="/data/bwwu/lc-rec/offload-20261108/cuda-12.4",
             LD_LIBRARY_PATH="/data/bwwu/lc-rec/offload-20261108/cuda-12.4/lib64:/data/bwwu/envs/grid-20261009-e61baf3/lib/python3.10/site-packages/torch/lib"),
    inputs=[dict(path=f"{snapshot}/{path}", sha256=digest) for digest, path in (m.split("  ", 1) for m in members)],
)
Path(request).write_text(json.dumps(request_object, indent=2) + "\n")
PY
cp "$request" "$support/evidence/$job_id.request.json"
"$scp" -q "$(wslpath -w "$request")" "new-server:$remote/inputs/$job_id.json"
"$ssh" -T -o BatchMode=yes -o StrictHostKeyChecking=yes new-server "/usr/bin/python3 -I -S -B '$runner' submit '$jobroot' '$remote/inputs/$job_id.json'" | tee "$support/evidence/$job_id.submission.json"
