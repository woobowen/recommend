#!/bin/bash
set -euo pipefail
RELEASE=/data/bwwu/lc-rec/a0-beauty-20261115/production-closure-20261115/releases/6e1b211d9c37f358c43b1540b40b17f6703150d2
ROOT=/data/bwwu/lc-rec/a0-beauty-20261115/production-closure-20261115/formal-candidate-5b2917d/formal-run
PLAN="$ROOT/plan.json"
AUTH="$ROOT/user-launch-authorization-20261116.json"
PY=/data/bwwu/envs/letter-20261110/bin/python
EVIDENCE=/data/bwwu/lc-rec/a0-beauty-20261115/production-closure-20261115/formal-candidate-5b2917d/launch-evidence-20261116
mkdir "$EVIDENCE"
exec > >(tee "$EVIDENCE/launch.log") 2>&1
cd "$RELEASE"
export PYTHONOPTIMIZE=0 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
printf 'UTC '; date -u -Is
printf 'HOST '; hostname
id
pwd
sha256sum "$PLAN" "$AUTH"
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
free -b
df -B1 "$ROOT"
"$PY" - "$PLAN" "$AUTH" <<'PY'
import sys,json,hashlib,os,subprocess
sys.path.insert(0,os.path.join(os.getcwd(),'lc_rec/a0_beauty'))
from formal_control import digest
from production_control import validate_plan,approval_documents,authorize,experiment_binding
plan,auth=sys.argv[1:]; p=json.load(open(plan)); state=json.load(open(os.path.join(p['root'],'state.json')))
assert hashlib.sha256(open(plan,'rb').read()).hexdigest()=='f5a3e7cb81269f637b92d29e021462aa72a3dd3ee86ff1ec8dfb95e93e1a08fc'
assert digest(p)=='0597958bcf852be1537252507a6c20e153256d3135f990b7362e45e867f95206'
assert state['status']=='PREPARED_NOT_SUBMITTED' and not state['attempts']
assert not os.listdir(p['job_root'])
assert not os.path.exists('/data/bwwu/lc-rec/a0-beauty-20261115/formal-experiment-registry')
assert len(p['inputs'])==96
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==p['gpu_uuids']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
mem=int(next(x.split()[1] for x in open('/proc/meminfo') if x.startswith('MemAvailable:')))*1024
v=os.statvfs(p['root']); assert mem>=40*1024**3 and v.f_bavail*v.f_frsize>=max(100*1024**3,v.f_blocks*v.f_frsize//10)
validate_plan(p); approval_documents(p,require_pass=True); authorize(p,auth,'ca733239d9c95a35a4c5504f8dc205c1039d91399df29630ef0b48d5caaa1963'); experiment_binding(p)
print('PRELAUNCH_CHECKS PASS',flush=True)
PY
printf 'START_ARGV: %s %s start --plan %s --authorization %s --authorization-sha256 %s\n' "$PY" "$RELEASE/lc_rec/a0_beauty/formal_control.py" "$PLAN" "$AUTH" ca733239d9c95a35a4c5504f8dc205c1039d91399df29630ef0b48d5caaa1963
"$PY" "$RELEASE/lc_rec/a0_beauty/formal_control.py" start --plan "$PLAN" --authorization "$AUTH" --authorization-sha256 ca733239d9c95a35a4c5504f8dc205c1039d91399df29630ef0b48d5caaa1963
printf 'START_RETURNED UTC '; date -u -Is
