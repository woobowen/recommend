#!/bin/bash
set -u
D=/data/bwwu/lc-rec/a0-beauty-20261115/production-closure-20261115/formal-candidate-5b2917d
J="$D/jobs/letter-a0-beauty-formal-beauty-a0-seed42-formal-train-a0"
date -u -Is
hostname
ps -p 3280445,3280453,3280454,3280455,3280460,3280461 -o pid,ppid,sid,lstart,args
python3 - "$D" "$J" <<'PY'
import json,sys,os
from pathlib import Path
d,j=map(Path,sys.argv[1:]); s=json.loads((d/'formal-run/state.json').read_text()); print('STATE',json.dumps(s))
t=json.loads((j/'status.json').read_text()); t.pop('worker_group',None); print('STATUS',json.dumps(t))
for p in [j/'work.log',j/'supervisor.log',d/'formal-run/controller.log']:
 print('LOG',p,'size',p.stat().st_size,'mtime',p.stat().st_mtime)
 with p.open('rb') as f: f.seek(max(0,p.stat().st_size-10000)); print(f.read().decode('utf8',errors='replace'))
PY
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
free -b
df -B1 "$D"
