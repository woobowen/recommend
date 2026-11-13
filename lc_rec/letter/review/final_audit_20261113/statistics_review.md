# A0 supplementary statistics: independent review

Reviewer: `statistics_review` (non-author of the production evaluation implementation).

Final conclusion: **PASS — supplementary grouping, fixed user axis and user-level
bootstrap independently verified on all 24,772 A0 users.**

This conclusion covers the statistical subsystem. It does not replace the
separate original-training, original-metric, generation and CCE reviews.

## Scope and access

This reviewer has direct local filesystem and CPU execution access. The reviewer
read the project rules from `/home/addaswsw/recommend/`, the actual vendored
`LETTER-LC-Rec/data.py`, and the frozen Instruments assets. No Git operations,
GPU calls, model loading, training, generation, or original asset writes were
performed. Server-only result review, if bridged by the root agent, is identified
as such and does not imply an independent server or GPU reproduction.

The reviewer owns only this report and `statistics_verify.py`; production
evaluation code is authored separately.

## Definitions frozen before reading model scores

- The author train constructor first takes each original sequence `items[:-2]`,
  then creates supervised prefixes. Popularity counts each item occurrence in
  the former sequence once, avoiding multiplication through overlapping prefixes.
- Test history is `items[:-1]`, retaining at most the final 20 entries. Its valid
  history length excludes padding. Fixed bins are 0–5, 6–10, 11–20.
- Popularity uses all catalog items with positive training-visible counts for
  inverse empirical-CDF cutpoints at 1/3 and 2/3 (one-based ranks
  `ceil(n/3)` and `ceil(2*n/3)`). Zero-count catalog items form `UNSEEN`.
  `LONG_TAIL` has `0 < f <= q1`, `MID` has `q1 < f <= q2`, and `POPULAR`
  has `f > q2`. Equal frequencies are never split. Coincident cutpoints may
  produce an empty group and must be reported without changing the rule.
- User bootstrap is conditional on one fixed trained model, not training-seed
  variance. The planned protocol is 5,000 samples of 24,772 users with replacement,
  seed 20261113, PCG64, fixed ascending `row_idx` axis, replicate-major integer
  sampling, and linear-interpolated percentile endpoints at 2.5%/97.5%.

These are `SUPPLEMENTARY_ANALYSIS`; no direct LETTER paper comparison is claimed.

## Independent checks already executed

The three original assets were loaded locally and their complete SHA-256 hashes
matched the task's fixed SID, interaction, and item hashes. The reviewer independently
derived the following counts without accessing prediction scores:

| Quantity | Count |
|---|---:|
| Catalog items | 9,922 |
| Training-visible item occurrences | 156,609 |
| Positive-frequency catalog items | 9,913 |
| Zero-frequency catalog items | 9 |
| First/second positive-frequency cutpoint | 5 / 11 |
| LONG_TAIL / MID / POPULAR / UNSEEN catalog items | 3,530 / 3,387 / 2,996 / 9 |
| LONG_TAIL / MID / POPULAR / UNSEEN test targets | 3,990 / 4,132 / 16,615 / 35 |
| Test history 0–5 / 6–10 / 11–20 users | 12,556 / 8,652 / 3,564 |

Command (local CPU, no package installation):

```sh
python3 lc_rec/letter/review/final_audit_20261113/statistics_verify.py \
  --self-test --data-root lc_rec/letter/vendor/LETTER/data/Instruments
```

Exit code 0. NumPy 1.26.4. Seven directed checks passed: history bin edges,
inverse-CDF quantile values, tied frequencies, unseen/popularity bin edges,
numeric percentile interpolation, bootstrap block-size invariance, and
constant/finite/ordered confidence intervals. Synthetic examples are not evidence
that the full A0 bootstrap has passed.

Independent canonical user axis SHA-256:
`1daf63ce4c77b0265777a46b61939006aafab0276b9106c742de602c8c7e867a`.
This hashes compact JSON of the insertion-order user-ID list; another serialization
may have a different byte hash and must not be called an axis mismatch without
checking values.

## Acceptance gates

The source, complete derived-row, subgroup-aggregate, independent-bootstrap and
runtime-identity gates described below have now passed. Earlier source-review
and synthetic-test observations remain historical facts; they were not treated
as substitutes for the actual full-data execution.

## Production implementation inspection

The reviewer directly read the actual new `scripts/offline_metrics.py` and
`config/a0_evaluation_v1.json` before execution. The production implementation
matches the definitions above: training frequencies use original `seq[:-2]` once;
the clipped test-history length uses `min(20, len(seq)-1)`; positive catalog
nearest-rank cutpoints preserve ties; each test user contributes exactly one
row on the ascending `row_idx` axis; the metric matrix is float64; all 16
overall intervals use 5,000 user resamples from PCG64 with the prescribed seed
and linear percentile interpolation. Its subgroup report explicitly distinguishes
descriptive group estimates from overall confidence intervals.

No blocking statistical-definition issue was found in that pre-execution read.
That source review alone did not establish that runtime outputs matched. The
following full-data checks supply the runtime evidence.

The reviewer implementation now has a full-result CLI. It independently rebuilds
all 24,772 metadata rows from original assets, verifies CSV/NPZ value and axis
identity, checks all group denominators and aggregates, and generates an
independent set of 5,000 × 16 replicate means using **column-wise gather**.
The production implementation uses a joint three-dimensional gather. Every
replicate mean and each interval endpoint is compared at fixed absolute
tolerance `1e-13`; the raw maximum differences and execution costs are retained.

The CLI intentionally does not rescan the 381 MB prediction files. Its metric
inputs are the already independently ranking/CCE-verified per-user values.
Reading the source, checking derived data, and verifying bootstrap address
different gates and are not substituted for one another.

## Actual full-data execution and terminal review

The root agent executed this reviewer-authored script through the accepted CPU
runner, then bridged the unedited lightweight receipts to the shared WSL
filesystem. The reviewer directly read those collected receipts, the primary
results, the independent ranking/CCE per-user comparison, request/status/cleanup,
and the crosscheck. No independent SSH access or GPU rerun is claimed.

- Original GPU RUN: `4c6a83d9ba42dc2fe737792759e410ef1874da76`.
- Actual CPU evaluation source: `3d11258daa6aa4fb2bed0cd013bca50536fe351d`.
- CPU job: `letter-a0-20261112-final-audit-20261113-offline-r1`.
- Runner terminal state: `SUCCEEDED`, exit 0, cleanup verified with no live
  owned/unconfirmed/unsupported processes.
- Actual Python: `/data/bwwu/envs/letter-20261110/bin/python`; NumPy **1.24.4**.
  The local synthetic test used 1.26.4; those are separate recorded environments.
- Reviewer script SHA-256:
  `c17e47746751146eb4c50245e88616811981031dbd64501eb205e1615774ab96`.
- Actual primary script SHA-256:
  `e16aee62f3ebdad137c960f63ff4d8b9b6f7ee8de6c8fb8b3f14d1f5807ef45e`.
- Actual evaluation config SHA-256:
  `1d4cc625d0147f0db43980cad2aa66faa5be7a41deb5fa01e90167d37e4187c3`.

The reviewer independently rehashed the local script/config and collected
bootstrap/groups JSON. They match the remote execution receipts' byte identities.

The [raw independent statistics receipt](../../evidence/final_audit_20261113/jobs/letter-a0-20261112-final-audit-20261113-offline-r1/results/independent_statistics.json)
shows:

| Check | Actual result |
|---|---:|
| Metadata rows reconstructed and compared | 24,772 |
| CSV to companion NPZ metric values | exact equality, all 16 columns |
| Per-user metrics separately checked against raw predictions | 396,352 values; maximum difference 0 |
| Subgroup aggregate maximum absolute difference | 0 |
| Independently generated replicate means compared | 80,000 |
| Maximum replicate-mean absolute difference | 6.938893903907228e-16 |
| Maximum interval-endpoint absolute difference | 4.996003610813204e-16 |
| Point-estimate absolute difference | 0 |
| Fixed acceptance tolerance | 1e-13 |
| Independent bootstrap time | 12.681113 seconds |
| Complete independent statistics check time | 14.065299 seconds |
| CPU user/system time | 14.017301 / 0.081967 seconds |
| Independent check peak RSS | 100,909,056 bytes |

The small differences are floating-point summation-order effects between the
production multivariate gather and independent contiguous column gather. They
are disclosed, not called bytewise equality, and the original tolerance was not
relaxed. The primary overall bootstrap took 8.405447 seconds. The entire three-stage
CPU pipeline took 36.892866 seconds; it is not the original GPU experiment cost.

The input [CSV/NPZ manifest](../../evidence/final_audit_20261113/jobs/letter-a0-20261112-final-audit-20261113-offline-r1/results/primary/metrics.json)
and independent receipt bind:

| Server-derived file | Bytes | SHA-256 |
|---|---:|---|
| `primary/per_user.csv` | 6,186,926 | `423309c6765395e42fc6a18a43706a4653d71692b184c65fe5755f78a34df6fb` |
| `primary/per_user_metrics.npz` | 67,315 | `d46590f04dcd0f8438b67ccea636b3310a61ff081e146305032c614f444a1689` |
| `primary/bootstrap_draw_means.npy` | 640,128 | `9c74c2cb7a05a953de4ce12c1d87583ef958e9fd7534d56f16cd81406058e502` |

These files remain in the new server result directory:
`/data/bwwu/lc-rec/a0-final-audit-20261113/letter-a0-20261112-final-audit-20261113-offline-r1/`.
The original prediction, checkpoint, adapter, SID and GPU output paths were not
written by either statistics implementation. The collected crosscheck records
identical before/after original metrics, rank reports, training reports, selected
checkpoint record and formal ledger.

## Reviewed statistical results

The full subgroup result is preserved in [groups.json](../../evidence/final_audit_20261113/jobs/letter-a0-20261112-final-audit-20261113-offline-r1/results/primary/groups.json).
Representative descriptive values follow; these are not subgroup confidence intervals.

| Visible history | Users | HR@10 | NDCG@10 |
|---|---:|---:|---:|
| 0–5 | 12,556 | 0.1332430710 | 0.1006006984 |
| 6–10 | 8,652 | 0.1026352288 | 0.0777290409 |
| 11–20 | 3,564 | 0.0693041526 | 0.0482535834 |

| Training-frequency group | Users | HR@10 | NDCG@10 |
|---|---:|---:|---:|
| UNSEEN (0) | 35 | 0 | 0 |
| LONG_TAIL (1–5) | 3,990 | 0.0017543860 | 0.0011152236 |
| MID (6–11) | 4,132 | 0.0031461762 | 0.0015634367 |
| POPULAR (>11) | 16,615 | 0.1678001806 | 0.1261943990 |

Group boundaries and counts were determined from training frequencies and raw
history lengths before observing model scores. The very small UNSEEN group is
not evidence of stable zero performance on unseen-item populations. The strong
descriptive association with popularity does not identify a causal mechanism
or authorize selecting a different model, checkpoint, grouping, or experiment.

The primary [bootstrap report](../../evidence/final_audit_20261113/jobs/letter-a0-20261112-final-audit-20261113-offline-r1/results/primary/bootstrap.json)
contains all 16 reviewed 95% percentile intervals. Selected endpoints below use
the primary report's summation convention; the independent endpoints match to
the differences reported above.

| Metric | Estimate | User-bootstrap 95% interval |
|---|---:|---:|
| HR@5 | 0.0903035685 | [0.0866704344, 0.0938559664] |
| HR@10 | 0.1133537865 | [0.1093169708, 0.1171897707] |
| NDCG@5 | 0.0776631039 | [0.0743723357, 0.0808411509] |
| NDCG@10 | 0.0850811320 | [0.0817787923, 0.0882141564] |
| ItemHit@10 | 0.1130308413 | [0.1090343937, 0.1168658162] |
| ItemNDCG@10 | 0.0843175088 | [0.0810063330, 0.0874239909] |

## Limits and reusable user axis

The fixed user axis and target identities are verified and suitable for future
paired evaluation once another authorized run exists. No A1 or Shuffle samples
or comparisons were created. Future paired bootstrap must resample the same user
indices jointly across models. An A1 result cannot automatically be attributed
to SID alone when model size and architecture differ; Shuffle should use its own
model's Normal control. This report does not determine those future methods.

Overall confidence intervals describe sampling variability over these test users
conditional on the one frozen A0 model. They do not estimate between-training-seed
variation, future-item generalization, subgroup confidence bounds, or upstream
SID training reproducibility. No multiplicity-adjusted significance or causal
claims are made. These limits do not prevent PASS of the specified supplementary
statistical computation.
