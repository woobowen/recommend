#!/usr/bin/env bash
set -euo pipefail
subject=${1:-/home/addaswsw/recommend-worktrees/shuffle-v2/lc_rec/beauty_history_attention/attention.py}
probe="$(dirname "$(realpath "$0")")/attention_cpu_probe.py"
python3 - "$subject" "$probe" <<'PY' | /mnt/c/Windows/System32/OpenSSH/ssh.exe -T -o BatchMode=yes -o StrictHostKeyChecking=yes new-server 'CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=3 /data/bwwu/envs/letter-20261110/bin/python -B -'
from pathlib import Path
import base64, hashlib, sys
subject, probe = (Path(entry) for entry in sys.argv[1:])
content = subject.read_bytes()
print('import base64, sys, types')
print('subject=types.ModuleType("review_subject"); sys.modules[subject.__name__]=subject')
print(f'exec(compile(base64.b64decode({base64.b64encode(content).decode()!r}), "review_subject", "exec"), subject.__dict__)')
print(f'__subject_sha={hashlib.sha256(content).hexdigest()!r}')
print(probe.read_text())
PY
