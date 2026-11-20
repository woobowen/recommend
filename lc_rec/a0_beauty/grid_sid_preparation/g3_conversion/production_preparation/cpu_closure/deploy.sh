#!/usr/bin/env bash
# Deploy committed bytes to the isolated CPU closure in the existing SSH setup.
set -euo pipefail
commit="${1:?full implementation commit}"
[[ "$commit" =~ ^[0-9a-f]{40}$ ]]
repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
prefix="$(realpath --relative-to="$repo" "$(dirname "$0")")"
cd "$repo"
local_stage=$(mktemp -d /tmp/g3-closure-deploy.XXXXXX)
trap 'rm -f -- "$local_stage"/{closure.py,closure_config.json,common.py,launch.py,paired_stats.py,predictions.py,review_raw.py,test_guards.py,deployment.json}; rmdir -- "$local_stage"' EXIT
for name in closure.py closure_config.json common.py launch.py paired_stats.py predictions.py review_raw.py test_guards.py; do
    git show "$commit:$prefix/$name" > "$local_stage/$name"
done
python3 -B - "$local_stage" "$commit" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

directory, commit = Path(sys.argv[1]), sys.argv[2]
files = {p.name: {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
         for p in sorted(directory.iterdir())}
(directory/"deployment.json").write_text(json.dumps({"implementation_commit":commit, "files":files},indent=2)+"\n")
PY
root=$(python3 -c 'import sys,json;print(json.load(open(sys.argv[1]))["closure_root"])' "$local_stage/closure_config.json")
[[ "$root" == /data/bwwu/lc-rec/g3-formal-20261118/cpu-closure-21d495b0be5b ]]
ssh=/mnt/c/Windows/System32/OpenSSH/ssh.exe
tar -C "$local_stage" -cf - . | base64 -w 0 | "$ssh" -T -o BatchMode=yes -o ConnectTimeout=12 new-server \
    "set -euo pipefail; umask 077; mkdir -p '$root/releases'; exec 7>'$root/.deployment.lock'; flock 7; dest='$root/releases/$commit'; if test -e \"\$dest\"; then cat >/dev/null; else staged=\$(mktemp -d '$root/releases/.receiving.XXXXXX'); base64 -d | tar -xf - -C \"\$staged\"; chmod u-w \"\$staged\"/*; mv -T \"\$staged\" \"\$dest\"; fi; /usr/bin/python3 -I -S -B -c 'import json, pathlib,hashlib,sys; p=pathlib.Path(sys.argv[1]); m=json.loads((p/\"deployment.json\").read_bytes()); assert set(m[\"files\"])|{\"deployment.json\"}=={x.name for x in p.iterdir()}; assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==v[\"sha256\"] and (p/n).stat().st_size==v[\"bytes\"] for n,v in m[\"files\"].items()); print(json.dumps(m,indent=2))' \"\$dest\""
