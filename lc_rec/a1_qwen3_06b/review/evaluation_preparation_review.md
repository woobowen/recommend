# A1 independent evaluation preparation review

Reviewer: actual sub-agent `a0_protocol_review`, nonauthor of the primary A1
evaluation and paired-comparison implementations. Date: 2026-11-13.

**INDEPENDENT_EVALUATOR_PREPARATION: PASS (source/synthetic checks only).**
**A1_FULL_METRICS: NOT_YET_REVIEWED. PAIRED_ENTRY_SOURCE_REVIEW: PASS (current source only).**

No actual A1 formal predictions were available to this reviewer. This report
contains no A1 measured recommendation metric, full-data PASS or formal-launch
approval. Local CPU preparation must not be reported as experiment completion.

## Independent implementation and fixed basis

The reviewer added only `review/independent_extended.py` and this report for this
subtask. The new program was derived from the fixed A0 independent reviewer
implementation, not the production evaluator:

- Source DELIVERY: `55a1371da28a45a2ec81ceee7bd82fd42b7192d5`.
- Original path:
  `lc_rec/letter/review/final_audit_20261113/independent_extended.py`.
- Original SHA-256:
  `c779b97e40afff9b907dc8a2bb439fe6fb42979fb0151e6498caee565b9ea617`.
- New path: `lc_rec/a1_qwen3_06b/review/independent_extended.py`.
- New size: **24920 bytes**; SHA-256
  `ee193edd584a987093a254bb6b2886e6ba3b09fda8832b5bf6421a1b798ec20f`.

The program imports neither the primary A1 evaluator nor Torch/NumPy/vendor
ranking metrics. It independently parses original saved strings, stable-sorts
the original sequence scores, reconstructs target relevance, computes ranking
metrics and explicitly expands CCE item positions. It can compare all sixteen
metric values for every user against the primary CSV.

The reviewer performed no SSH, GPU operation, model load, package installation,
primary-code edit or Git write. Source reads of the fixed commit were read-only.
Later main-controller execution of this reviewer-owned script is a bridge and
must be described as such, not as this reviewer's independent SSH/GPU run.

## Minimal A1 identity adaptation

Full audit now requires `--manifest` with schema
`a1_frozen_prediction_manifest_v1`, a complete lowercase 40-hex
`gpu_run_commit` and exactly two unique rank records under `predictions`.
Each record provides rank, absolute prediction path, positive integer byte size
and lowercase 64-hex SHA-256. Each path must resolve to the corresponding
`--test-dir/predictions.rank{rank}.jsonl`.

The separate Stage-A report must be PASS for the same RUN and full 24,772 users.
Its two prediction records must match the frozen manifest's paths, sizes and
hashes. The independent program additionally computes actual SHA-256 during its
own necessary complete prediction stream, with no extra hashing read. It checks
that both shards remain unchanged while reading and each contains 12,386 users.

No A0 RUN or prediction-size/hash constants remain. The A0-specific observed
assertions `invalid==978` and `collision_candidates==4384` were removed. A1
diagnostics are independently measured instead. Fixed Instruments asset hashes,
24,772-user coverage, twenty candidates per user, target/user/SID pairing,
all sixteen formulas, full relevance checks and existing numerical tolerances
remain unchanged.

The result binds its actual frozen manifest path/SHA and actual RUN. As before,
existing result output is protected from overwrite. A manifest or Stage-A
identity failure is an error, not permission to fall back to A0 identities.

## Actual bounded CPU checks

The reviewer parsed the new file and compared these six function ASTs against
the original fixed A0 blob: `normalize`, `sorted_sids`, `ranking`,
`expanded_cce`, `compare_derived_row`, and `self_test`. All six are identical,
excluding source locations. This specifically protects stable tie sorting,
duplicate-target additive NDCG, first-hit MRR, target/prefix collision expansion,
invalid-prefix occupancy and CSV metric comparisons.

The CLI self-test used the fixed original official CCE file:

```text
python3 lc_rec/a1_qwen3_06b/review/independent_extended.py --self-test
  --official-cce lc_rec/letter/review/final_audit_20261113/external/official_cce.py
  --official-sha256 173c257a8711a885e8f6d5abcdb0ad2510ad4fab01c1700631b5ec261f50cd2d
```

Actual result: exit 0; **1694 numerical assertions PASS**, **266** named/random
CCE cases compared against the official function at unchanged `1e-14` absolute
tolerance. Official source is 4544 bytes/SHA
`173c257a8711a885e8f6d5abcdb0ad2510ad4fab01c1700631b5ec261f50cd2d`.
This is formula/fixture evidence, not a full A1 prediction audit.

Ten in-memory manifest checks also passed: accept valid out-of-order rank
records; reject wrong schema, incomplete RUN SHA, duplicate rank, bool rank,
bool byte size, invalid SHA characters, relative path, wrong absolute path and
missing shard. These synthetic checks did not create A1 prediction files or
metrics. A source scan confirmed the A0 RUN/PREDICTIONS and observed anomaly
assertions are absent.

## Frozen statistics verifier reuse

The existing independent
`lc_rec/letter/review/final_audit_20261113/statistics_verify.py` remains unmodified,
SHA-256 `c17e47746751146eb4c50245e88616811981031dbd64501eb205e1615774ab96`.
Its local fixed-asset census and seven directed tests were actually run and
recorded in `a0_protocol_basis.md`. It has no A0 model/prediction identity
constants and can verify the same Instruments axes, group metadata, CSV/NPZ
equality, group aggregates and 5000-replicate statistics for A1.

The verifier checks bootstrap statistics from derived user metrics; independent
raw-prediction ranking/CCE comparison is a separate prerequisite. A self-
consistent matrix alone is insufficient evidence of correct raw ranking.

## Read-only paired-entry findings

Inspected author file: `lc_rec/a1_qwen3_06b/scripts/paired.py`. The reviewer has
not modified this file. Initial findings below describe its first read before
the author's revision. That first source read did not capture a simultaneous
byte hash. The reviewer initially attached a subsequently captured hash to those
old findings, after the author had already changed the file concurrently. That
review-record identity error is corrected here: the current hash stated in the
follow-up section binds the corrected source, not the old deficiency.

The bootstrap formula uses per-user `A1-A0` differences and identical PCG64
indices for both models, with 5000 draws, seed20261113 and linear percentile
intervals. This is the correct paired construction. It also pins A0 CSV/NPZ
hashes, checks matrix size/row index and compares per-row identity/group fields.
Zero A0 denominators produce null relative differences.

The concrete remaining input-binding defect is:

1. It does not compare each model's NPZ metric matrix against the corresponding
   CSV's sixteen numeric metric columns. Thus correct CSV identities can coexist
   with a row-permuted or otherwise wrong A1 NPZ value matrix while the entry
   still reports `PASS_PAIRED_USER_AXIS`.

Additional defensive checks would make the contract more explicit: require
the complete frozen sixteen names/order, the literal frozen protocol version
and finite matrix values before generating samples. These are not separate
demonstrated false-PASS defects here: the trusted fixed A0 hashes and key/metadata
equality already transitively bind the column names and protocol, while
`allow_nan=False` rejects a nonfinite JSON result at output. The earlier message
to the author listed these checks together; this distinction avoids overstating
their effect under the actually pinned A0 inputs.

The required recommendation is a CSV-to-NPZ full-matrix check at the paired entry;
explicit fixed-key/protocol/finite checks are useful defensive additions.
A primary pipeline's separately verified statistics PASS can be an additional
gate, but invocation order should not conceal the standalone input-binding gap.
No experimental formula or research-method change is needed for these checks.
Revised bytes must be reread before declaring this source issue closed; that
follow-up read is recorded immediately below.

## Paired-entry follow-up: source issue closed

The reviewer subsequently reread the complete corrected `paired.py` and parsed
its AST successfully. Current source identity is **3790 bytes**, SHA-256
`99d06cc3f78e74727b18e10da248c519d9af5e82c98e5ad7f23345bf9369a4d0`.
It explicitly constructs the frozen sixteen metric names/order, requires NPZ
keys equal that list and values finite, reconstructs each input CSV's full
float64 metric matrix and requires `np.array_equal(csv_values, v)` with the
expected `(24772,16)` shape. It also requires every CSV row's literal
`letter_instruments_beam20_v1` protocol version. These checks occur separately
for A0 and A1 before difference construction or resampling.

The previous concrete input-binding defect is therefore closed in these bytes.
The reviewer also reread the unchanged row/user/target/group correspondence and
paired bootstrap construction: the 5000 PCG64 seed20261113 replicate-major int64
draws resample the per-user difference matrix, with linear percentile endpoints.

**PAIRED_ENTRY_SOURCE_REVIEW: PASS** for this source identity. This follow-up is
a read-only source and syntax check. No formal evaluation, full A1 input scan or
bootstrap execution was performed. Actual paired runtime acceptance remains
pending the complete real A1 prediction/derived-data evidence and independent
statistical verification.

No paired A0/A1 runtime result has been reviewed, and no significance conclusion
is made. Future full-data acceptance still requires original-user/target axes,
all raw predictions, independent rankings/CCE, grouping/statistics and actual
paired bootstrap receipts. The comparison cannot isolate an SID causal effect
across LLaMA-7B and Qwen3-0.6B model configurations.

GPT_UPLOAD_REQUIRED: NO for this preparation subtask; publish the independent
source/report and later actual receipts in the final fixed A1 delivery/ZIP.
