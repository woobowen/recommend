#!/usr/bin/env bash
# Recheck the extracted G3 V2 audit using only the enclosed files and CPU.
set -euo pipefail

if [[ $# != 9 ]]; then
    echo "Usage: $0 DATA_ROOT NORMAL_TEST CPU_AUDIT V2_RUN_COMMIT NEW_OUTPUT S2_TEST S3A_TEST S3B_TEST S4_TEST" >&2
    exit 2
fi

data_root=$1
normal_test=$2
cpu_audit=$3
run_commit=$4
output=$5
shift 5

src_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
python=${G3_REVIEW_PYTHON:-python3}
export PYTHONPATH="$src_root${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=
export PYTHONDONTWRITEBYTECODE=1

mkdir -- "$output"
"$python" -m lc_rec.shuffle_v2.g3_evaluate replay-verify \
    --data-root "$data_root" --test-dir "$normal_test" --cpu-audit "$cpu_audit" \
    --output "$output/replay"
"$python" -m lc_rec.shuffle_v2.g3_evaluate normal \
    --data-root "$data_root" --test-dir "$normal_test" --output "$output/normal"

derived_source="$1/derived.rank0.py"
eval_roots=()
for mode in shuffle2 shuffle3a shuffle3b shuffle4; do
    "$python" -m lc_rec.shuffle_v2.g3_evaluate condition \
        --data-root "$data_root" --test-dir "$1" --cpu-audit "$cpu_audit" \
        --derived-source "$derived_source" --mode "$mode" --scope PREFLIGHT --run-commit "$run_commit" --output "$output/$mode"
    eval_roots+=("$output/$mode")
    shift
done

exec "$python" -m lc_rec.shuffle_v2.g3_evaluate paired --scope PREFLIGHT \
    --normal "$output/normal" --condition-roots "${eval_roots[@]}" --output "$output/paired"
