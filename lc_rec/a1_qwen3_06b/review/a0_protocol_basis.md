# A1 evaluation basis: nonauthor review of fixed A0 evidence

Reviewer: actual sub-agent `a0_protocol_review`, independent of the A1 production
evaluation implementation. Review date: 2026-11-13.

**A0_PROTOCOL_BASIS: PASS. A1_EVALUATION: NOT_YET_REVIEWED.**

This review establishes the fixed protocol, reusable source and user/data axes.
The main controller has not yet supplied A1 formal predictions or result receipts.
None of the checks below certifies A1 training, full testing, metrics, CCE,
statistics, paired comparison or final acceptance.

## Scope, rules and independence

The first shell command was `ls -la` in `/home/addaswsw/recommend`. Its output
showed the three parallel rule files and historical reference material. The
reviewer explicitly read `/home/addaswsw/recommend/AGENTS.md`, `PROJECT_RULES.md`
and `REMOTE_EXECUTION.md`; the rule entry is separate from the Git worktree.
The worktree's `lc_rec/rules/AGENTS.md` was also read. The user explicitly
authorized this one-time A1 task and real nonauthor review.

Worktree: `/home/addaswsw/recommend-worktrees/lc-rec`. Historical source reads used
read-only `git show` at the complete fixed DELIVERY SHA below. No refs, index,
branch, remote or history were changed. No SSH, GPU invocation, model loading,
A0 generation/training, dependency installation or original-asset write was
performed. CPU checks loaded fixed source bytes in memory and read local assets;
they did not write result files. This report is the reviewer's only authorized
file addition. The reviewer has not authored or changed the A1 primary evaluator.

The reviewer has direct local filesystem/CPU access. Historical server outcomes
below are read from fixed published receipts and reports, not a claim of an
independent SSH session or a fresh server/GPU reproduction.

## Fixed historical evidence actually read

- A0 DELIVERY: `55a1371da28a45a2ec81ceee7bd82fd42b7192d5`.
- A0 GPU RUN: `4c6a83d9ba42dc2fe737792759e410ef1874da76`.
- A0 CPU evaluation RUN: `3d11258daa6aa4fb2bed0cd013bca50536fe351d`.
- [Fixed A0 final review entry](https://github.com/woobowen/recommend/blob/55a1371da28a45a2ec81ceee7bd82fd42b7192d5/lc_rec/letter/A0_NORMAL_FINAL_REVIEW_ENTRY.md).
- `lc_rec/letter/EVALUATION_PROTOCOL_V1.md` and
  `lc_rec/letter/config/a0_evaluation_v1.json` at that DELIVERY.
- Actual primary, pipeline, independent original/extended/statistics source,
  official CCE source identity and the evaluation/statistics review reports.
- A0 derived manifest and primary result identities, including the fixed
  per-user CSV/NPZ and canonical ordered-user-axis SHA.

The A0 entry records four epochs/8240 updates and four full validations, selecting
`checkpoint-8240` by minimum `eval_loss=1.5409067869186401`, followed by complete
24,772-user Beam20 evaluation. Its model was `huggyllama/llama-7b` revision
`4782ad278652c7c71b72204d462d6d01eaaf7549`. These are historical A0 facts;
they are not identities or expected outcomes for A1.

## Reusable code and A0-specific dependencies

All paths in this table are relative to the worktree and were inspected at the
fixed A0 DELIVERY.

| Purpose | Path | A1 use |
| --- | --- | --- |
| Protocol | `lc_rec/letter/EVALUATION_PROTOCOL_V1.md` | Preserve frozen definitions and user axes. |
| Machine definition | `lc_rec/letter/config/a0_evaluation_v1.json` | Preserve formulas; use A1's actual RUN/model/prediction identity. |
| Primary evaluator | `lc_rec/letter/scripts/offline_metrics.py` | Adapt `RUN`, `PRED` and identity gates; do not reuse A0 prediction constants. |
| CPU pipeline | `lc_rec/letter/scripts/offline_audit_pipeline.py` | Adapt A0 release/data/training/test/stage-A/protected/config paths and RUN/hash constants. |
| Independent four paper metrics | `lc_rec/letter/review/formal_det_20261112/independent_metrics.py` | Can remain unchanged for compatible A1 output schema; actual RUN is read dynamically. |
| Independent expanded metrics/CCE | `lc_rec/letter/review/final_audit_20261113/independent_extended.py` | Adapt `RUN`, `PREDICTIONS` and A0-specific historical anomaly assertions. |
| Independent grouping/bootstrap | `lc_rec/letter/review/final_audit_20261113/statistics_verify.py` | Same fixed Instruments axes; no A0 model/prediction constants. |
| Terminal training audit | `lc_rec/letter/scripts/formal_result_audit.py` | LLaMA FA, 450 tensors, job prefix and paths are A0-specific; cannot be copied as Qwen acceptance. |
| Official CCE | `lc_rec/letter/review/final_audit_20261113/external/official_cce.py` | Preserve exact official bytes and license. |

Specific hazards:

- `offline_metrics.py` pins A0 `RUN` and two `(bytes,SHA)` prediction identities;
  its protocol and Stage-A gates require the same RUN. Its derived manifest must
  instead bind the real A1 RUN and prediction files.
- `independent_extended.py` pins A0 `RUN` and `PREDICTIONS`; its final assertion
  `invalid == 978` and `collision_candidates == 4384` is an A0 observation,
  not an A1 acceptance condition. A1 diagnostics must be independently measured.
- `independent_metrics.py` dynamically verifies complete RUN SHA, rank completion,
  adapter/data/author manifests and the raw complete prediction SHA; it expects
  the A1 rank/schema to retain these records and
  `original_callback_verified_all_tokens`. It does not independently rerun the
  callback or decode saved token IDs.
- `formal_result_audit.py` assumes 32-layer LLaMA, FA deterministic counters and
  the A0 adapter shape/key counts. Qwen must use its actual architecture and tied
  embedding/head evidence, never fake the A0 tensor count.

These are identity/architecture adaptations, not authorization to change the
frozen ranking, generation, CCE, grouping or statistical definitions.

## Frozen evaluation definitions

Raw parsing is `raw_decoded.split('Response:')[-1].strip().replace(' ', '')`.
Sort by descending original sequence score using stable ordering; ties retain
generation order. Preserve all twenty candidates, including invalid and repeated
SIDs. Complete SID equality defines relevance; no item filtering, deduplication
or collision disambiguation is applied.

The sixteen metric columns are ordered as:

`hit@1, hit@5, hit@10, hit@20, ndcg@5, ndcg@10, ndcg@20, mrr@5, mrr@10, mrr@20, ItemHit@5, ItemHit@10, ItemHit@20, ItemNDCG@5, ItemNDCG@10, ItemNDCG@20`.

Single-target HR equals Recall. Original NDCG adds the discounts of **all** target
SID occurrences within K, including repeated targets; MRR uses the first target
occurrence. A0's zero duplicate observation does not eliminate this distinction
for a future result.

CCE uses only the first target occurrence. Let `P` be the sum of the group sizes
of all preceding SID occurrences, `g` the target group size and
`m=min(g,max(0,K-P))`. Then `ItemHit=m/g` and
`ItemNDCG=sum(i=1..m,1/log2(P+i+1))/g`. Absent targets score zero. Unknown SID
groups occupy one slot; preceding repeated SID occurrences each expand. A unique
target alone does not guarantee equality with SID scores when preceding groups
collide. CCE is averaged identity credit, not recovery of a unique item or ZCR.

Official basis: `Nishikata97/CollisionGenRec` commit
`6dd5d314f62d3379e664a52c9007e7645556970c`, `model/cce.py`, 4544 bytes,
SHA-256 `173c257a8711a885e8f6d5abcdb0ad2510ad4fab01c1700631b5ec261f50cd2d`.
Its fixed original file and MIT license are preserved in the external directory.

Top-5/10/20 SID diagnostics include invalid, duplicate and collision candidate
occurrences and affected users. Duplicates count `K - len(set(top_k_strings))`,
including invalid strings. Coverage is unique valid SIDs divided by **9897**,
not the 9922-item catalog. First-hit ranks 1–20 plus miss partition all users.

## Data axes, grouping and paired-comparison inputs

Full local SHA-256 checks actually passed:

| Asset | Bytes | SHA-256 |
| --- | ---: | --- |
| `Instruments.index.json` | 518041 | `33c87e28c52ae367985bdf494cb49d676d181e13a39497b4ad035f46111546dd` |
| `Instruments.inter.json` | 3312848 | `0a11d47791d90483005da93c36d826679e20e4592d1d2b86518d5f56f218e743` |
| `Instruments.item.json` | 7526890 | `1d9938e576a8d964e6b977a2b63aa56402840a788eaa6685f26f0d910a7133f2` |

The reviewer independently reconstructed 24,772 rows from the fixed assets using
the frozen statistics verifier. Canonical compact JSON of the insertion-order
user-ID list has SHA-256
`1daf63ce4c77b0265777a46b61939006aafab0276b9106c742de602c8c7e867a`.
The broader original test user/target-record serialization has a different scope
and is not required to have this same byte hash.

Test-visible history is `items[:-1][-20:]`, with no padding contribution. Its
0–5 / 6–10 / 11–20 groups contain **12556 / 8652 / 3564** users.

Popularity counts every original training-visible `items[:-2]` occurrence once,
not expanded supervised prefixes: **156609** occurrences. Positive-frequency
catalog nearest-rank tertiles yield `q1=5`, `q2=11`; ties stay together.

| Group | Training frequency | Catalog items | Test users |
| --- | --- | ---: | ---: |
| UNSEEN | 0 | 9 | 35 |
| LONG_TAIL | 1–5 | 3530 | 3990 |
| MID | 6–11 | 3387 | 4132 |
| POPULAR | >=12 | 2996 | 16615 |

These counts and boundaries are fixed independently of A1 predictions. Group
results are supplementary/descriptive; the 35-user UNSEEN group is small.

A0's already-reviewed pairable files remain under:

`/data/bwwu/lc-rec/a0-final-audit-20261113/letter-a0-20261112-final-audit-20261113-offline-r1/primary/`.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `per_user.csv` | 6186926 | `423309c6765395e42fc6a18a43706a4653d71692b184c65fe5755f78a34df6fb` |
| `per_user_metrics.npz` | 67315 | `d46590f04dcd0f8438b67ccea636b3310a61ff081e146305032c614f444a1689` |

These server-file identities were cross-checked in the fixed DELIVERY's derived
manifest, primary result and nonauthor statistics report. They have not been
freshly rehashed on the server by this reviewer.

Before paired bootstrap, compare A1/A0 `row_idx`, ordered user ID, target item,
target SID, metric column order/definition and grouping metadata one-to-one.
For every replicate, resample identical user indices and compute `A1-A0`.
Subtracting separately computed CI endpoints is not a paired interval. Undefined
relative changes when A0 is zero must be represented explicitly, not as a finite
percentage. Model architecture/scale differ, so this comparison cannot isolate a
causal effect of SID.

## Bootstrap and cost semantics

Overall bootstrap uses 5000 replacement samples of all 24772 users, PCG64 seed
20261113, replicate-major int64 draws and float64 metric values. Use linear
2.5/97.5 percentiles. It describes fixed-model test-user uncertainty, not
between-training-seed variance or subgroup CIs. The production implementation
uses joint multivariate gather; the independent verifier uses column-wise gather
with unchanged `1e-13` absolute comparison tolerance.

Generation latency is synchronized `model.generate` time per user, excluding
model loading and CPU decoding/serialization. Stage wall time and GPU-device
time must be reported separately. Training runtime includes validation/checkpoint
overheads unless separately measured; do not sum overlapping intervals twice.

## Actual bounded CPU checks and source identities

The reviewer read fixed source bytes through `git show`, compiled the required
modules in memory and performed read-only CPU verification. Command exit code 0.
No A0 prediction rescan or model computation was performed.

- Primary ranking/parser/CCE self-test: **PASS, 127 numerical assertions**,
  including target cutoff, repeated target, stable ties, invalid prefix, repeated
  prefix, target collisions and the Figure 2 partial-credit fixture.
- Independent data census: **PASS**, all three asset hashes, user axis, training
  frequencies, cutpoints and group populations above.
- Statistics self-test: **PASS, seven directed tests**, local NumPy **1.26.4**.
  This is separate from A0's historical server NumPy 1.24.4 execution.
- The five inspected source/config files below were byte-identical between the
  fixed DELIVERY blob and current worktree at review time.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `scripts/offline_metrics.py` | 17794 | `e16aee62f3ebdad137c960f63ff4d8b9b6f7ee8de6c8fb8b3f14d1f5807ef45e` |
| `review/final_audit_20261113/independent_extended.py` | 22480 | `c779b97e40afff9b907dc8a2bb439fe6fb42979fb0151e6498caee565b9ea617` |
| `review/final_audit_20261113/statistics_verify.py` | 17462 | `c17e47746751146eb4c50245e88616811981031dbd64501eb205e1615774ab96` |
| `review/final_audit_20261113/external/official_cce.py` | 4544 | `173c257a8711a885e8f6d5abcdb0ad2510ad4fab01c1700631b5ec261f50cd2d` |
| `config/a0_evaluation_v1.json` | 3503 | `1d4cc625d0147f0db43980cad2aa66faa5be7a41deb5fa01e90167d37e4187c3` |

The table's paths are relative to `lc_rec/letter/`. These checks establish that
the inspected A0 formulas and data axes are usable as the A1 basis. They are not
runtime evidence for A1. A1 needs its own fixed source/manifest, full predictions,
actual derived results and nonauthor complete-data verification before PASS.

No conflicting frozen evaluation definition was identified. Important A1-specific
training/generation/identity gates remain for their respective reviewers.

GPT_UPLOAD_REQUIRED: NO for this basis-only review; the supporting A0 evidence is
already at the fixed GitHub DELIVERY above. Final A1 publication and ZIP remain
the main controller's pending delivery work.
