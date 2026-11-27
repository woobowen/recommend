#!/bin/bash
set -euo pipefail
support=/home/addaswsw/recommend-support/beauty-history-attention-20261127
kind="$1"
attempt="$2"
job_id="letter-beauty-v2-history-attn-$kind-$attempt"
runner=/home/bwwu/grid-audits/sid-extensions-p0/scripts/runner-beauty-cbb963fab68c34cc.py
job_dir=/home/bwwu/grid-audits/sid-extensions-p0/jobs/$job_id
snapshot="$support/evidence/$job_id.query.$(date -u +%Y%m%dT%H%M%SZ).json"
/mnt/c/Windows/System32/OpenSSH/ssh.exe -T -o BatchMode=yes -o StrictHostKeyChecking=yes new-server "/usr/bin/python3 -I -S -B '$runner' query '$job_dir'" > "$snapshot"
python3 - "$snapshot" <<'PY'
from pathlib import Path
import json
import sys
s = json.loads(Path(sys.argv[1]).read_text())
print(s["job_id"], s["status"], s.get("exit_code"), s.get("worker_alive"), s.get("workgroup_live"))
PY
