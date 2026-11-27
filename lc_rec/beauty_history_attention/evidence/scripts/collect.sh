#!/bin/bash
set -euo pipefail
s=/home/addaswsw/recommend-support/beauty-history-attention-20261127
stage="$(cat "$s/windows-stage")"
kind="$1"
attempt="$2"
job="letter-beauty-v2-history-attn-$kind-$attempt"
jobs=/home/bwwu/grid-audits/sid-extensions-p0/jobs
remote=/data/bwwu/lc-rec/beauty-history-attention-20261127
name="$kind-$attempt"
bundle="$name-evidence.tar.gz"
ssh=/mnt/c/Windows/System32/OpenSSH/ssh.exe
scp=/mnt/c/Windows/System32/OpenSSH/scp.exe
archive_dir="$remote/review-exports"
"$ssh" -T -o BatchMode=yes -o StrictHostKeyChecking=yes new-server "set -eu; mkdir -p '$archive_dir'; tar --exclude='.run.lock' --exclude='.state.lock' --exclude='.submit.lock' --exclude='.observation.lock' --exclude='*.bin' --exclude='*.pt' --exclude='*.pth' --exclude='*.safetensors' --exclude='*.partial' -czf '$archive_dir/$bundle' -C '$jobs' '$job' -C '$remote' '$name'; sha256sum '$archive_dir/$bundle'" > "$s/evidence/$name.export.sha256"
"$scp" -q "new-server:$archive_dir/$bundle" "$(wslpath -w "$stage/$bundle")"
digest="$(awk '{print $1}' "$s/evidence/$name.export.sha256")"
actual="$(sha256sum "$stage/$bundle" | awk '{print $1}')"
test "$digest" = "$actual"
mkdir -p "$s/collected"
tar -xzf "$stage/$bundle" -C "$s/collected"
