# A0 final ranking and CCE: nonauthor review

Reviewer: `/root/final_metrics_review`.

Conclusion: **PASS for the original-four-metric audit, extended ranking metrics,
CCE, full-user alignment, and SID diagnostics within the scope below.** This is
the nonauthor metric review; final publication/ZIP validation and the separate
statistics/training reviews remain their own delivery gates.

## Access and independence

This reviewer has actual local filesystem, CPU, and web tools. The first shell
command was `ls -la` in `/home/addaswsw/recommend`; the project `AGENTS.md`,
`PROJECT_RULES.md`, and `REMOTE_EXECUTION.md` were explicitly read from that rules
directory. It is distinct from the Git worktree at
`/home/addaswsw/recommend-worktrees/lc-rec`.

The reviewer read fixed GPU RUN
`4c6a83d9ba42dc2fe737792759e410ef1874da76` sources `formal_evaluate.py`,
vendored `evaluate.py`, `test_ddp.py`, `data.py`, `utils.py`, and the existing
`review/formal_det_20261112/independent_metrics.py`. The primary new evaluator is
authored separately. This reviewer's independent program imports neither it nor
the original/vendor metric functions.

No Git mutations, GPU calls, model loading, training, generation, or writes to
original inputs were performed. A root-controlled remote execution of this
reviewer-authored program will be identified as bridge execution, not independent
SSH or GPU reproduction.

## Original policy verified from fixed sources

The original evaluator parses the final `Response:` suffix, strips surrounding
whitespace, removes literal spaces, and stably sorts descending by sequence score.
With `filter_items=False`, invalid complete SIDs remain ranked. Candidates are not
deduplicated. Hit is any exact complete-target-SID match in the prefix. NDCG sums
the log discount at **all** matching target positions. MRR in the supplementary
protocol uses the first match only.

The unchanged Stage-A nonauthor program is suitable for the first full four-metric
gate: it derives raw relevance and values before comparing the saved aggregates,
checks all rows/targets/Beam20 scores and manifest SHA values, and does not obtain
the answer by reformatting `metrics.json`. Its established aggregate tolerances are
`1e-13` for means and `1e-9` for sums. The sequential original sum and independent
`math.fsum` need not be bitwise identical; actual differences must be disclosed.

## CCE source and formula check

Independently read [arXiv v2, Section 3.1](https://arxiv.org/html/2605.25330v2),
equations 4–7 and Figure 2, and the actual public
[`model/cce.py`](https://github.com/Nishikata97/CollisionGenRec/blob/main/model/cce.py).
The reviewer additionally read the fixed raw URL at official full commit
`6dd5d314f62d3379e664a52c9007e7645556970c`. The locally saved `official_cce.py`
is 4,544 bytes, SHA-256
`173c257a8711a885e8f6d5abcdb0ad2510ad4fab01c1700631b5ec261f50cd2d`.

Paper start rank `p` is one-based; the task's `P = p - 1` counts all preceding
expanded items. The target group of width `g` contributes its uniformly averaged
Hit/NDCG at expanded positions `P+1` through `P+g`, clipped by K. The code uses
first target occurrence, `mapping.get(sid, 1)` for preceding SIDs, and does not
deduplicate repeated preceding SIDs. Therefore invalid preceding SIDs consume one
position, and a colliding prefix can push even a singleton target beyond K.

The independent implementation explicitly constructs the target's expanded item
positions, then averages the positions inside the cutoff. This is a separate
implementation from the direct `min(g, max(0, K-P))` formula.

No-collision reduction to original metrics additionally requires no target
duplicates and no expansion before the target. Original LETTER NDCG can count
multiple target occurrences whereas official CCE intentionally counts the first.
Any assertion of equivalence must retain that scope.

## Directed tests

The independent program passed local syntax checking and 94 numerical assertions:
hits at ranks 1/5/6/10/11/20, no hit, stable ties, exact parsing, repeated-target
additive original NDCG versus first-hit MRR, and ten named CCE examples. These
include singleton/size-2/size-3 targets, partial Top-K crossing, full exclusion,
absent target, preceding collision, invalid preceding SID, repeated preceding SID,
repeated target, and singleton target pushed out by preceding collisions.

Figure 2 is tested numerically as `ItemHit@5 = 2/3` and
`ItemNDCG@5 = (1/log2(5) + 1/log2(6))/3`.

The program supports a pinned-byte official-function comparison over the ten
named cases and 256 deterministically generated collision/repetition/invalid-SID
examples. This comparison was actually executed locally on CPU: 266 cases,
1,690 total numerical assertions, all PASS; per-value absolute tolerance
`1e-14`. The immutable receipt is `independent_extended_selftest.json` and binds
the official bytes and independent script SHA
`11a8b1c4916a0e90eaadb3cfa6e5c465b71c72e75ace0b5b3036e124681223b3`.
The later full-data results have now been reviewed below.

An initial draft contained one multiline-`if` syntax error, detected by local
`py_compile` and corrected before any formal execution or source freeze. No
numerical tolerance or acceptance criterion changed.

## Primary implementation review before source freeze

The reviewer read `scripts/offline_metrics.py` and `config/a0_evaluation_v1.json`
directly. No ranking/CCE or full-population execution blocker was identified.
The code preserves the original sort/parser/relevance, checks the original rank
assignment SHA, compares all users' CCE values with the fixed official function,
uses the 9,897-SID coverage denominator, and derives popularity solely from
`items[:-2]`. The bootstrap uses bounded blocks of eight replicates. Input
predictions are opened only for reading, while outputs use a new directory.

An additional reviewer-authored `test_production_formulas.py` actually compared
the primary formulas with this independent explicit-rank implementation and the
official CCE function: 1,024 deterministic random cases, 22,528 scalar numerical
comparisons, all PASS at fixed absolute tolerance `1e-14`. The receipt
`production_formula_check.json` binds primary script SHA
`e16aee62f3ebdad137c960f63ff4d8b9b6f7ee8de6c8fb8b3f14d1f5807ef45e`.
This is numerical formula evidence, not a substitute for the complete-user audit.

The final executed protocol now explicitly carries the three frozen input-asset
SHAs and canonical user-axis hash, addressing the review recommendation. The
source constants and runtime manifest enforce the same actual identities.

## Actual complete-data execution and final gates

The reviewer directly read the collected raw Stage-A receipt, its runner status,
the full independent extended receipt, primary metrics, crosscheck, pipeline,
immutable request and terminal runner status. These were obtained through the
main controller's server bridge. The reviewer did not personally initiate SSH or
claim an independent server/GPU reproduction.

Original Stage-A job `letter-a0-20261112-final-audit-20261113-stageA-r1` completed
SUCCEEDED, exit 0, cleanup verified. It used unchanged original independent script
SHA `d21d0eb6479564cd83c534a6a7dd926e48c8d50af6e72578280e94c87fc10473`
from GPU RUN `4c6a83d9ba42dc2fe737792759e410ef1874da76`. All 24,772 raw users,
20 beams each, 12,386 rows per rank, raw target/relevance and original prediction
size/SHA identities passed. HR@5/@10 agree exactly. Relative to the original
sequentially accumulated report, independent NDCG mean absolute differences are
`4.3021142204224816e-16` / `3.469446951953614e-16`; maximum sum difference is
`1.0686562745831907e-11`. These satisfy the unchanged `1e-13` mean / `1e-9` sum
tolerances. This is numerical agreement with disclosed summation differences,
not a claim of bitwise floating-point equality.

Full CPU job `letter-a0-20261112-final-audit-20261113-offline-r1` completed
SUCCEEDED, exit 0, cleanup verified, using fixed evaluation commit
`3d11258daa6aa4fb2bed0cd013bca50536fe351d`. The reviewer independently read Git
blobs at that commit and confirmed the executed source/config SHA identities:

| File | SHA-256 |
|---|---|
| `scripts/offline_metrics.py` | `e16aee62f3ebdad137c960f63ff4d8b9b6f7ee8de6c8fb8b3f14d1f5807ef45e` |
| `review/final_audit_20261113/independent_extended.py` | `c779b97e40afff9b907dc8a2bb439fe6fb42979fb0151e6498caee565b9ea617` |
| `config/a0_evaluation_v1.json` | `1d4cc625d0147f0db43980cad2aa66faa5be7a41deb5fa01e90167d37e4187c3` |

The independent implementation performed a necessary separate complete prediction
stream, reusing the successful Stage-A original-prediction checksums without a
redundant large hashing pass. It checked all row/target identities, raw parsing,
stable ranking and saved relevance. It then compared its independently derived
values against the primary CSV at every `row_idx`: **24,772 rows and 396,352
metric values, maximum absolute difference 0 for each of all 16 metrics**.
First-hit positions and miss=0 representation also matched. The primary code
separately compared all 24,772 users with the fixed official CCE function.

The CSV pairing receipt is 6,186,926 bytes, SHA
`423309c6765395e42fc6a18a43706a4653d71692b184c65fe5755f78a34df6fb`.
The full independent result is
[`independent_extended.json`](../../evidence/final_audit_20261113/jobs/letter-a0-20261112-final-audit-20261113-offline-r1/results/independent_extended.json),
with the main/independent aggregate and diagnostic comparison in
[`crosscheck.json`](../../evidence/final_audit_20261113/jobs/letter-a0-20261112-final-audit-20261113-offline-r1/results/crosscheck.json).

| K | SID Hit | SID NDCG | MRR | CCE ItemHit | CCE ItemNDCG |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.06450831584046504 | — | — | — | — |
| 5 | 0.0903035685451316 | 0.07766310391224163 | 0.07351176058991334 | 0.08973841433876958 | 0.07683032838933833 |
| 10 | 0.11335378653318262 | 0.08508113200832922 | 0.07655129062766718 | 0.11303084127240433 | 0.08431750877895493 |
| 20 | 0.13858388503148716 | 0.09146554910635438 | 0.07830539761333742 | 0.138139835297917 | 0.09066989688786065 |

Every primary/independent metric aggregate, Top-5/10/20 SID count/coverage, and
first-target rank count matched. Catalog checks found 9,922 items, 9,897 SIDs,
21 collision groups of size 2 and 2 of size 3, maximum 3, and 122 test targets
whose SID belongs to a collision group. Top-20 has 978 invalid candidate
occurrences, zero duplicates and 4,384 collision-SID candidate occurrences;
724 users have an invalid SID and 3,660 have a collision-SID candidate.
Top-5/10/20 unique valid SID counts are 1,739 / 2,495 / 3,305, divided by 9,897.
The first-hit counts sum to 3,433 hits and 21,339 misses, partitioning all users.
The original Hit@5/@10 integer hit counts are 2,237 / 2,808. CCE fractional-credit
sums are not generally counts of uniquely retrieved items and must not be labeled
as such.

The actual independent phase took 7.325976 seconds, CPU user 7.083845 seconds,
system 0.239043 seconds, peak RSS 144,380 KiB. The complete primary + independent
ranking + independent statistics pipeline took 36.892866 seconds. The immutable
request sets `CUDA_VISIBLE_DEVICES=""`; the pipeline asserts it, imports no model,
and reports GPU use false. Protected original metrics, rank records, best-selection
record and ledger have equal before/after size, mtime and SHA fingerprints.

The original Stage-A wrong data-root failure and the source-archive transport-path
failure were preserved and handled as task correction rounds 1 and 2 respectively.
Neither changed a metric formula, original prediction, SID, model or acceptance
tolerance. No failed numerical result was removed or replaced to obtain PASS.

## Limits and conclusion

- Original saved predictions were evaluated, not regenerated. This does not
  independently decode token IDs or rerun the original token constraint callback.
- No GPU computation, model or checkpoint tensor reload was performed in this
  metric review. Training correctness is covered by its separate actual-record
  review, and old unchanged assets are inherited with their fixed manifests.
- CCE expresses uncertainty within a SID group; it does not recover a unique
  item identity from the generator. The original SID mapping and paper metrics
  remain unchanged. ZCR was not executed.
- New MRR, Top-20, CCE and diagnostics have no directly comparable LETTER
  Instruments paper reference. They must retain their supplemental labels.
- The grouping/bootstrap review and its conditional single-model interpretation
  belong to the separate statistics report; this metric PASS does not by itself
  certify cross-seed stability or future A1/Shuffle results.
- Fixed protocol definitions can be reused, but future model runs need their own
  original-run/prediction manifests. The present A0-specific SHA constants must
  not be presented as the identities of future A1/Shuffle predictions.

**ORIGINAL_METRICS_INDEPENDENT_AUDIT: PASS. EXTENDED_METRICS: PASS.
CCE_EVALUATION: PASS. NONAUTHOR_METRIC_REVIEW: PASS**, within these limits.
