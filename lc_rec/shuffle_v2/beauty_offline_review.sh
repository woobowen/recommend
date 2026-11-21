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
