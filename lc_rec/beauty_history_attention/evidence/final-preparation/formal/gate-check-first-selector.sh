#!/bin/bash
# CPU-only checks against the exact deployed candidate, with no approval issued.
set -eu
commit="$1"
root=/data/bwwu/lc-rec/beauty-history-attention-final-20261127
source="$root/releases/$commit"
inputs="$root/inputs"
run="$inputs/formal-gate-check-${commit:0:12}"
python=/data/bwwu/envs/letter-20261110/bin/python
name=letter-beauty-final-20261127
gpu=GPU-59cac3bd-3645-668b-79c8-839e0d878893
package=lc_rec.beauty_history_attention
train="$source/lc_rec/beauty_history_attention"
vendor="$source/lc_rec/letter/vendor/LETTER/LETTER-LC-Rec"
data=/home/bwwu/grid-audits/sid-extensions-p0/transfers/letter-a0-beauty-author-8d0154e28de3
model=/data/bwwu/lc-rec/assets/llama-7b/4782ad278652c7c71b72204d462d6d01eaaf7549
prev=/data/bwwu/lc-rec/beauty-history-attention-20261127/train-r1/training/preflight_adapter
job=/home/bwwu/grid-audits/sid-extensions-p0/jobs/letter-beauty-v2-history-attn-formal-b-$name-r1
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=1 USE_TF=0 USE_FLAX=0
test -d "$source"
mkdir -p "$inputs"
mkdir "$run"
cd "$source"
plan="$run/b-formal-plan.unsigned.json"
python3 -m json.tool /home/bwwu/grid-audits/sid-extensions-p0/scripts/runner-beauty-cbb963fab68c34cc.py.deployment.json > "$run/runner-deployment.json"
{
  sha256sum SOURCE.sha256
  sha256sum /home/bwwu/grid-audits/sid-extensions-p0/scripts/runner-beauty-cbb963fab68c34cc.py
} > "$run/runtime-sha256.txt"
"$python" -B -m "$package.formal" plan --run-commit "$commit" --job-name "$name" --gpu-uuid "$gpu" \
    --backend eager --output "$plan" > "$run/plan-created.json"
"$python" -B -m "$package.formal" observe --plan "$plan" --verify-completion > "$run/observe-unsubmitted.json"
"$python" -B -m unittest -v "$package.test_formal" > "$run/fixtures.log" 2>&1
output="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_root"])' "$plan")"
test ! -e "$output"
test ! -e "$job"
negative="$run/negative-no-authorization.json"
printf '%s\n' '{"test_fixture":true,"formal_run_authorized":false,"external_review_status":"PENDING"}' > "$negative"
digest="$(sha256sum "$negative" | cut -d ' ' -f1)"
failures="$run/negative-exits.txt"
expect_failure() {
    case_name="$1"
    shift
    status=0
    "$@" > "$run/$case_name.log" 2>&1 || status="$?"
    printf '%s %s\n' "$case_name" "$status" >> "$failures"
    test "$status" -ne 0
}
train_args=(--source-root "$vendor" --data-root "$data" --model "$model" --output "$output/training-r1"
            --config "$train/config_bidirectional.json" --run-commit "$commit")
generate_args=(--adapter "$prev" --run-commit "$commit" --output "$output/generation"
               --purpose formal --max-seconds 108000 --backend eager)
expect_failure submit-no-authorization "$python" -B -m "$package.formal" submit --plan "$plan"
expect_failure submit-negative-fixture "$python" -B -m "$package.formal" submit --plan "$plan" \
  --formal-authorization "$negative" --formal-authorization-sha256 "$digest"
expect_failure worker-negative-fixture "$python" -B -m "$package.formal" worker --plan "$plan" \
  --plan-sha256 "$(sha256sum "$plan" | cut -d ' ' -f1)" --formal-authorization "$negative" \
  --formal-authorization-sha256 "$digest"
expect_failure direct-train-missing-authorization "$python" -B -m "$package.train" "${train_args[@]}"
expect_failure direct-train-negative-fixture "$python" -B -m "$package.train" "${train_args[@]}" \
  --formal-plan "$plan" --formal-authorization "$negative" --formal-authorization-sha256 "$digest"
expect_failure direct-train-preflight-cannot-write-formal-output "$python" -B -m "$package.train" "${train_args[@]}" \
  --boundary-preflight
expect_failure direct-train-optimized "$python" -OB -m "$package.train" "${train_args[@]}"
expect_failure direct-generation-missing-authorization "$python" -B -m "$package.evaluate" generate "${generate_args[@]}"
expect_failure direct-generation-negative-fixture "$python" -B -m "$package.evaluate" generate "${generate_args[@]}" \
  --plan "$plan" --formal-authorization "$negative" --formal-authorization-sha256 "$digest"
grep -q 'formal B requires the frozen plan and separate user authorization' "$run/direct-generation-missing-authorization.log"
grep -q 'user authorization does not cover the exact B plan' "$run/submit-negative-fixture.log"
grep -q 'user authorization does not cover the exact B plan' "$run/direct-train-negative-fixture.log"
grep -q 'the bounded preflight cannot write a formal B output' "$run/direct-train-preflight-cannot-write-formal-output.log"
grep -q 'training requires all runtime checks' "$run/direct-train-optimized.log"
grep -q 'user authorization does not cover the exact B plan' "$run/direct-generation-negative-fixture.log"
test ! -e "$output"
test ! -e "$job"
python3 - "$run" "$output" "$job" <<'PY'
import json, sys
from pathlib import Path
run, work, job = map(Path, sys.argv[1:])
observation = json.loads((run / "observe-unsubmitted.json").read_text())
assert observation["runner"] is None and observation["stage"] is None
assert observation["recommendation_metrics"] == "尚未产生"
gate = {name: int(code) for name,code in [line.split() for line in (run/"negative-exits.txt").read_text().splitlines()]}
assert len(gate)==9 and all(gate.values())
print(json.dumps({"formal_run_started":False, "formal_submit_accepted":False,
                  "formal_job_path":str(job),"formal_job_exists":job.exists(),
                  "formal_output_path":str(work),"formal_output_exists":work.exists(),
                  "plan_is_unsigned":True,"negative_exit_codes":gate,
                  "plan_sha256":observation["plan_sha256"]},indent=2))
PY
echo "$run"
