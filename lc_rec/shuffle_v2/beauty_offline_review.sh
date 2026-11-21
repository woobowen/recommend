#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
destination="${1:?Pass a path for the independent review receipt}"
python_bin="${PYTHON_BIN:-python3}"
export CUDA_VISIBLE_DEVICES=''
export OPENBLAS_NUM_THREADS=1

if [[ -f "$root/paired/summary.json" ]]; then
    run_commit="$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["RUN_COMMIT"])' "$root/publication/run.json")"
    additional=(--v2-run-commit "$run_commit" --paired-dir "$root/paired")
    for mode in shuffle2 shuffle3a shuffle3b shuffle4; do
        additional+=(--condition "$mode" "$root/conditions/$mode/test" "$root/conditions/$mode/primary")
    done
else
    additional=()
fi

"$python_bin" -B "$root/source/lc_rec/shuffle_v2/independent_beauty_review.py" \
    --data-dir "$root/data" \
    --condition normal "$root/conditions/normal/test" "$root/conditions/normal/primary" \
    "${additional[@]}" --output "$destination"

if [[ -f "$root/preflight-evaluation/summary.json" ]]; then
    run_commit="$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["RUN_COMMIT"])' "$root/publication/run.json")"
    check_dir="${destination%.json}-preflight"
    PYTHONPATH="$root/source:$root/source/support" "$python_bin" -B -m lc_rec.shuffle_v2.beauty_preflight_eval \
        --archive-root "$root" --preflight-dir "$root/preflight" --cpu-audit "$root/replay/audit.json" \
        --single-shard-audit "$root/replay/single-shard.json" --run-commit "$run_commit" --output "$check_dir"
    "$python_bin" -B - "$check_dir" "$root/preflight-evaluation" <<'PY'
import csv
import json
from pathlib import Path
import sys
import numpy as np

actual, recorded = map(Path, sys.argv[1:])
a, b = [json.loads((folder / 'summary.json').read_text()) for folder in (actual, recorded)]
assert a['status'] == b['status'] == 'PASS_BEAUTY_PREFLIGHT_EVALUATION_INTERFACE'
assert not a['formal_results'] and not b['formal_results']
for name in ('metrics.csv', 'paired.csv', 'changes_from_normal.csv'):
    with (actual / name).open() as f, (recorded / name).open() as g:
        assert list(csv.DictReader(f)) == list(csv.DictReader(g)), name
with np.load(actual / 'arrays.npz', allow_pickle=False) as x, np.load(recorded / 'arrays.npz', allow_pickle=False) as y:
    assert x.files == y.files and all(np.array_equal(x[key], y[key]) for key in x.files)
print('Verified archived 32-user engineering arrays and all 80 interface records')
PY
fi
