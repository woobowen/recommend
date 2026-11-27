#!/bin/bash
set -euo pipefail
support=/home/addaswsw/recommend-support/beauty-history-attention-20261127
worktree=/home/addaswsw/recommend-worktrees/shuffle-v2
stage="$(cat "$support/windows-stage")"
remote=/data/bwwu/lc-rec/beauty-history-attention-20261127
commit="$1"
ssh=/mnt/c/Windows/System32/OpenSSH/ssh.exe
scp=/mnt/c/Windows/System32/OpenSSH/scp.exe
snapshot="$remote/releases/$commit"
export_dir="$support/export-$commit"
cd "$worktree"
test "$(git rev-parse "$commit^{commit}")" = "$commit"
mkdir "$export_dir"
git archive "$commit" lc_rec/beauty_history_attention lc_rec/letter/vendor/LETTER/LETTER-LC-Rec   lc_rec/letter/scripts lc_rec/letter/evidence/formal_20261111/precision_approval.json   lc_rec/letter/config/author_instruments.json lc_rec/letter/review/final_audit_20261113/external/official_cce.py   lc_rec/letter/review/final_audit_20261113/statistics_verify.py lc_rec/shuffle/core.py lc_rec/shuffle_v2   lc_rec/a0_beauty/{single_gpu_training.py,trainer_loop_patch.py,training_rows.py,zero2_partial_state.py,checkpoint_receipt.py,g3_identity.py} |
  tar -xf - -C "$export_dir"
(
    cd "$export_dir"
    find lc_rec -type f -print0 | sort -z | xargs -0 sha256sum > SOURCE.sha256
    tar -czf "$stage/$commit.tar.gz" SOURCE.sha256 lc_rec
)
sha256sum "$stage/$commit.tar.gz" > "$support/evidence/deploy-$commit.tar.sha256"
"$scp" -q "$(wslpath -w "$stage/$commit.tar.gz")" "new-server:$remote/inputs/$commit.tar.gz"
"$ssh" -T -o BatchMode=yes -o StrictHostKeyChecking=yes new-server "mkdir -p '$remote/releases' && mkdir '$snapshot' && cd '$snapshot' && tar -xzf '$remote/inputs/$commit.tar.gz' && sha256sum -c SOURCE.sha256" > "$support/evidence/deploy-$commit.log"
wc -l "$support/evidence/deploy-$commit.log"
