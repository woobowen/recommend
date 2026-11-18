#!/usr/bin/env bash
# Saved first-launch invocation. Run only against the original experiment.
set -euo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1
release=/home/bwwu/grid-audits/sid-extensions-p0/g3-production-20261118/21d495b0be5b/release
python=/data/bwwu/envs/letter-20261110/bin/python
root=/data/bwwu/lc-rec/g3-formal-20261118/21d495b0be5b
record="$root/launch-20261118"
plan="$root/plan.json"
receipt="$record/authorization.json"
receipt_sha="${1:?Supply the recorded, fixed authorization SHA-256}"
cd "$release/lc_rec/a0_beauty"
set -o noclobber
exec > >(tee "$record/launch.stdout") 2> >(tee "$record/launch.stderr" >&2)
trap 'rc=$?; printf "launch_wrapper_exit=%s\n" "$rc" > "$record/launch.exit"; exit "$rc"' EXIT
date -u +%FT%TZ
printf 'cwd=%s\n' "$PWD"
PYTHONPATH="$PWD" "$python" "$record/precheck.py" "$plan"
set -x
"$python" -c 'import sys; from production_control import authorize; from formal_control import read; assert authorize(read(sys.argv[1]), sys.argv[2], sys.argv[3]); print("authorize=PASS")' "$plan" "$receipt" "$receipt_sha"
"$python" production_control.py start --plan "$plan" --authorization "$receipt" --authorization-sha256 "$receipt_sha"
set +x
date -u +%FT%TZ
