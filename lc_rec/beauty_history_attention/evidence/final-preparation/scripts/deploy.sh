#!/bin/bash
set -euo pipefail
s=/home/addaswsw/recommend-support/beauty-history-attention-final-20261127
commit="$1"
worktree=/home/addaswsw/recommend-worktrees/shuffle-v2
wstage="$(cat "$s/windows-stage")"
root=/data/bwwu/lc-rec/beauty-history-attention-final-20261127
ssh=/mnt/c/Windows/System32/OpenSSH/ssh.exe
scp=/mnt/c/Windows/System32/OpenSSH/scp.exe
snapshot="$root/releases/$commit"
export_dir="$s/export-$commit"
cd "$worktree"
test "$(git rev-parse "$commit^{commit}")" = "$commit"
mkdir "$export_dir"
mapfile -t experiment < <(git ls-tree -r --name-only "$commit" lc_rec/beauty_history_attention |
  awk '!/\/evidence\//')
git archive "$commit" "${experiment[@]}" lc_rec/letter/vendor/LETTER/LETTER-LC-Rec   lc_rec/letter/scripts lc_rec/letter/evidence/formal_20261111/precision_approval.json   lc_rec/letter/config/author_instruments.json lc_rec/letter/review/final_audit_20261113/external/official_cce.py   lc_rec/letter/review/final_audit_20261113/statistics_verify.py   lc_rec/shuffle/core.py lc_rec/shuffle_v2   lc_rec/a0_beauty/{single_gpu_training.py,trainer_loop_patch.py,training_rows.py,zero2_partial_state.py,checkpoint_receipt.py,g3_identity.py} |
  tar -xf - -C "$export_dir"
(
    cd "$export_dir"
    find lc_rec -type f -print0 | sort -z | xargs -0 sha256sum > SOURCE.sha256
    tar -czf "$wstage/$commit.tar.gz" SOURCE.sha256 lc_rec
)
sha256sum "$wstage/$commit.tar.gz" > "$s/evidence/deploy-$commit.tar.sha256"
"$ssh" -T -o BatchMode=yes -o StrictHostKeyChecking=yes new-server "mkdir -p '$root/inputs' '$root/releases'"
"$scp" -q "$(wslpath -w "$wstage/$commit.tar.gz")" "new-server:$root/inputs/$commit.tar.gz"
"$ssh" -T -o BatchMode=yes -o StrictHostKeyChecking=yes new-server "mkdir '$snapshot' && cd '$snapshot' && tar -xzf '$root/inputs/$commit.tar.gz' && sha256sum -c SOURCE.sha256" > "$s/evidence/deploy-$commit.log"
wc -l "$s/evidence/deploy-$commit.log"
