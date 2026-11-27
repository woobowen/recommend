#!/bin/bash
set -euo pipefail
s=/home/addaswsw/recommend-support/beauty-history-attention-final-20261127
commit="$(cat "$s/generation-commit")"
gpu=GPU-59cac3bd-3645-668b-79c8-839e0d878893
remote=/data/bwwu/lc-rec/beauty-history-attention-final-20261127
ssh=/mnt/c/Windows/System32/OpenSSH/ssh.exe
wait_clean() {
    local result
    while true; do
        result="$("$s/query.sh" "$1" "$2")"
        case "$result" in
            *" SUCCEEDED 0 False False") printf '%s\n' "$result"; return 0 ;;
            *" RUNNING "*|*" ACCEPTED "*) sleep 20 ;;
            *) printf '%s\n' "$result"; return 1 ;;
        esac
    done
}
wait_clean b-mask r0
"$s/collect.sh" b-mask r0
"$s/submit.sh" "$commit" generate-stop r0 700 "$gpu" >/dev/null
wait_clean generate-stop r0
"$ssh" -T -o BatchMode=yes -o StrictHostKeyChecking=yes new-server "cd '$remote/pilot' && test ! -e final && test ! -e chunks/000001 && sha256sum chunks/000000/*" > "$s/evidence/first-chunk-before-resume.sha256"
"$s/collect.sh" generate-stop r0
"$s/submit.sh" "$commit" generate-resume r0 700 "$gpu" >/dev/null
wait_clean generate-resume r0
"$ssh" -T -o BatchMode=yes -o StrictHostKeyChecking=yes new-server "cd '$remote/pilot' && test -f final/completion.json && sha256sum chunks/000000/* final/completion.json" > "$s/evidence/first-chunk-after-resume.sha256"
diff -u "$s/evidence/first-chunk-before-resume.sha256" <(head -6 "$s/evidence/first-chunk-after-resume.sha256")
"$s/collect.sh" generate-resume r0
"$s/submit.sh" "$commit" generate-resume duplicate-r0 180 "$gpu" >/dev/null
wait_clean generate-resume duplicate-r0
"$ssh" -T -o BatchMode=yes -o StrictHostKeyChecking=yes new-server "cd '$remote/pilot' && sha256sum chunks/000000/* final/completion.json" > "$s/evidence/first-chunk-after-duplicate.sha256"
cmp "$s/evidence/first-chunk-after-resume.sha256" "$s/evidence/first-chunk-after-duplicate.sha256"
"$s/collect.sh" generate-resume duplicate-r0
"$s/submit.sh" "$commit" metrics r0 300 '' >/dev/null
wait_clean metrics r0
"$s/collect.sh" metrics r0
"$s/collect.sh" pilot
