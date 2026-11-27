#!/bin/bash
set -euo pipefail
s=/home/addaswsw/recommend-support/beauty-history-attention-final-20261127
wstage="$(cat "$s/windows-stage")"
root=/data/bwwu/lc-rec/beauty-history-attention-final-20261127
stage="$1"
attempt="${2:-}"
jobs=/home/bwwu/grid-audits/sid-extensions-p0/jobs
ssh=/mnt/c/Windows/System32/OpenSSH/ssh.exe
scp=/mnt/c/Windows/System32/OpenSSH/scp.exe
if [ "$stage" = pilot ]; then
    name=pilot
    paths=(-C "$root" pilot)
else
    name="$stage-$attempt"
    paths=(-C "$jobs" "letter-beauty-v2-history-attn-final-$name" -C "$root" "$name")
fi
bundle="$name-evidence.tar.gz"
args="$(printf ' %q' "${paths[@]}")"
"$ssh" -T -o BatchMode=yes -o StrictHostKeyChecking=yes new-server "mkdir -p '$root/review-exports' && tar --exclude='*.lock' --exclude='*.safetensors' --exclude='*.bin' --exclude='*.partial' -czf '$root/review-exports/$bundle' $args && sha256sum '$root/review-exports/$bundle'" > "$s/evidence/$name.export.sha256"
"$scp" -q "new-server:$root/review-exports/$bundle" "$(wslpath -w "$wstage/$bundle")"
test "$(awk '{print $1}' "$s/evidence/$name.export.sha256")" = "$(sha256sum "$wstage/$bundle" | cut -d' ' -f1)"
mkdir -p "$s/collected"
tar -xzf "$wstage/$bundle" -C "$s/collected"
