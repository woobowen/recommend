#!/bin/bash
set -euo pipefail
s=/home/addaswsw/recommend-support/beauty-history-attention-final-20261127
stage="$1"
attempt="$2"
job="letter-beauty-v2-history-attn-final-$stage-$attempt"
runner=/home/bwwu/grid-audits/sid-extensions-p0/scripts/runner-beauty-cbb963fab68c34cc.py
jobdir=/home/bwwu/grid-audits/sid-extensions-p0/jobs/$job
record="$s/evidence/$job.query.$(date -u +%Y%m%dT%H%M%SZ).json"
/mnt/c/Windows/System32/OpenSSH/ssh.exe -T -o BatchMode=yes -o StrictHostKeyChecking=yes new-server "/usr/bin/python3 -I -S -B '$runner' query '$jobdir'" > "$record"
python3 - "$record" <<'PY'
import json,sys
s=json.load(open(sys.argv[1]));print(s['job_id'],s['status'],s.get('exit_code'),s.get('worker_alive'),s.get('workgroup_live'))
PY
