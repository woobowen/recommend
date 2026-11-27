#!/bin/bash
set -euo pipefail
s=/home/addaswsw/recommend-support/beauty-history-attention-20261127
r=/data/bwwu/lc-rec/beauty-history-attention-20261127
train_commit=b9dfa8f8adad6ee89c15bda9d978c5465ac7e772
diagnostic_commit=a26a9e9698fb5c66f04e93dd062f8beef8bade8b
ssh=/mnt/c/Windows/System32/OpenSSH/ssh.exe

wait_terminal() {
    local result
    while true; do
        result="$("$s/query.sh" "$1" "$2")"
        case "$result" in
            *" SUCCEEDED 0 False False"|*" WORKERS_TERMINATED 0 False False")
                printf '%s\n' "$result"
                return 0
                ;;
            *" RUNNING "*|*" ACCEPTED "*) sleep 20 ;;
            *) printf '%s\n' "$result"; return 1 ;;
        esac
    done
}
wait_terminal train resume-r1
"$ssh" -T -o BatchMode=yes -o StrictHostKeyChecking=yes new-server "test -f '$r/train-r1/training/preflight_adapter/history_attention.json' && python3 -c 'import json; s=json.load(open(\"$r/train-r1/training/rank0.json\")); assert s[\"status\"] == \"BOUNDARY_PREFLIGHT_COMPLETE\" and s[\"global_step\"] == 10'"
"$s/collect.sh" train resume-r1
"$s/collect.sh" train r1
"$s/submit.sh" "$diagnostic_commit" generate r1 720 --training "$r/train-r1/training" >/dev/null
if wait_terminal generate r1; then
    "$s/submit.sh" "$diagnostic_commit" metrics r1 300 --predictions "$r/generate-r1/predictions" >/dev/null
fi
"$s/collect.sh" generate r1
"$s/submit.sh" "$diagnostic_commit" regression r1 720 >/dev/null
wait_terminal regression r1 || true
"$s/collect.sh" regression r1
"$s/submit.sh" "$diagnostic_commit" training-regression r1 720 >/dev/null
wait_terminal training-regression r1 || true
"$s/collect.sh" training-regression r1
if "$s/query.sh" metrics r1 2>/dev/null; then
    wait_terminal metrics r1 || true
    "$s/collect.sh" metrics r1
fi
"$s/submit.sh" "$diagnostic_commit" cuda-check r2 180 >/dev/null
wait_terminal cuda-check r2 || true
"$s/collect.sh" cuda-check r2
