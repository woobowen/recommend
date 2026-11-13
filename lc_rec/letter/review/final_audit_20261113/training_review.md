# A0 training and original evaluation: nonauthor review

Reviewer: `/root/final_training_review`, a separate agent from the offline metric
implementation author.

**A0 FORMAL TRAINING: PASS.**
**A0 ORIGINAL FULL TEST: PASS.**
**ORIGINAL METRICS INDEPENDENT AUDIT: PASS.**

These conclusions cover the original A0 run and its original four metrics.
They do not certify the separate extended metrics, CCE, bootstrap or final
publication, whose own completed execution and review receipts are required.

## Evidence access and identity

The reviewer had working execution tools and directly read local Git objects at
`4c6a83d9ba42dc2fe737792759e410ef1874da76`, the published raw terminal evidence
captured at 2026-11-13 08:27 UTC, and the actual model-identity receipt. This was
direct local file/source inspection. At 2026-11-13 09:30:21 UTC the reviewer also
executed a separate lightweight read-only SSH query through the existing
Windows Transport, reading two final rank JSON reports and four small source
files. Their hashes matched the fixed RUN and original captured rank reports.
There was no GPU rerun, model load, or rehash of the multi-GB model/checkpoint
payloads.

The applicable rules were explicitly read from `/home/addaswsw/recommend/`;
the Git worktree is `/home/addaswsw/recommend-worktrees/lc-rec`, branch
`wbw-lc-rec`. The formal job is
`letter-a0-20261112-normal-det-seed42-r0`.

The source review covered fixed `formal_train.py`, `formal_attention.py`,
`formal_evaluate.py`, `formal_instruments_seed42.json`, the independent
`independent_metrics.py`, and author `data.py`, `test_ddp.py`, `evaluate.py`,
`utils.py`, `collator.py`, and `modeling_letter.py`. The machine-readable
[source identities and receipt checks](training_receipt_check.json) retain their
SHA-256 values. Its independently written stdlib checker executed 53 assertions
successfully; the preserved [checker](training_receipt_check.py) imports no
producer evaluator, Torch, or model code.

The separate [effective-setting checker](terminal_effective_check.py) executed
26 assertions against the independently SSH-captured
[raw effective records](terminal_effective_raw.json), with
[all checks PASS](terminal_effective_check.json). It confirms the final
TrainingArguments, native DeepSpeed stage/batch/precision, exact optimizer
membership, 450 trainable tensors / 289,652,736 elements, 224 A-only adaptation
targets and frozen original embedding/head. The published raw extraction is
limited to relevant configuration and compatibility fields; source rank SHA
values identify the complete server records.

## Findings already supported by original receipts

- Two ranks completed 8,240 updates and Epoch 4.0, starting at global step 0.
  Epoch-end events are exactly 2060, 4120, 6180, and 8240.
- The four Validation Loss values are respectively `1.840831995010376`,
  `1.6257708072662354`, `1.5417826175689697`, and `1.5409067869186401`.
  Both ranks report identical losses. These are complete author validation
  evaluations, not the old 32-example preflight.
- The minimum validation loss selects `checkpoint-8240`. Both ranks have an
  actual `strict_ds_restored` event for that checkpoint and scheduler step 8240.
  This is evidence of a real restore, rather than merely an asserted best path.
- The formal configuration and source preserve seed 42, four epochs,
  Int8/BF16, LoRA r8/alpha16/dropout .05, the seven projection families,
  response-only shifted causal CE at temperature .8, microbatch 8, two ranks,
  gradient accumulation 4, effective batch 64, ZeRO-2, AdamW and cosine schedule.
  The best/latest retention limit 2 is a disclosed storage adaptation from the
  author value 1, with collective-complete checkpoint marking before rotation.
- FA2 `2.7.4.post1` source/extension guards are present in the formal source.
  Both ranks record `deterministic=True` at actual native and CUDA-extension
  backward dispatch boundaries. Each final report records 1,054,720 matching
  native/extension varlen calls. No global Torch deterministic mode, cuBLAS
  workspace expansion, or diagnostic tensor observers were enabled.
- Both retained checkpoints (6180 and 8240) have `complete.json`, expected RUN
  and configuration identities, rank-specific loader/RNG, external scheduler,
  full DeepSpeed model state and two optimizer shards. The captured check
  verifies existing sizes and small-file hashes. It does not claim newly
  rehashed large payloads or a new recovery experiment.
- Original inference uses the exported best adapter with an unquantized BF16
  LLaMA base, as the author LoRA inference branch does. Both real rank reports
  record all 450 adapter keys / 289,652,736 elements strictly reloaded and exact
  after explicit dtype conversion; the adapter hash agrees with best selection.
- Original Prompt0, Beam20, ten-token generation, fixed per-position allowed
  token callback, stable descending score sort, unfiltered complete-SID equality,
  and additive duplicate-target NDCG are preserved. There is no catalog Trie or
  additional deduplication. The formal implementation refuses beam reduction.
- Each test rank completed 12,386 users. Producer evidence reports 24,772 total,
  no failures/skips, 978 invalid complete-SID candidates, no duplicate candidates,
  and 4,384 collision-SID candidate occurrences. These anomaly counts still
  require the separate full-prediction independent receipt for final acceptance.
- Original runner is `SUCCEEDED`, pipeline `SUCCESS`, both stage exit codes 0,
  cleanup verified. The persistent formal ledger has one successful attempt and
  no resume, within 96 hours. Recorded original process identities were absent
  at the captured terminal check. No interrupted attempt is being relabelled.

Two harmless metadata details are retained: DeepSpeed `steps_per_print=Infinity`
is its logging-disable sentinel, not a nonfinite loss. The compatibility record's
`a_only_bf16.adaptation` description inherits the old string `probe-only` from
the earlier helper; actual formal source, approved precision receipt, normal
argv and effective 224-module instrumentation establish its approved formal use.
Neither original record was edited to hide these details.

## Original metrics: full Stage A accepted

The preserved independent script recomputes parsing, ordering, every row's
relevance, user/target pairing, full coverage, finite scores, anomaly counts,
prediction byte SHA-256 and all four metric sums without importing the author
or formal evaluator. It reads original `metrics.json` only for comparison.
Its compensated summation can differ from producer floating-point accumulation
at roundoff scale; acceptance limits are mean `1e-13` and sum `1e-9`.

The script checks internally consistent RUN identities but does not hard-code
the task's expected RUN. The reviewer therefore separately checked the actual
completed output against `4c6a83d9ba42dc2fe737792759e410ef1874da76` and these two
frozen prediction hashes:

- Rank 0: `bdcbdde15d3ac6850708888e055aa67ae61212dbb675d9f915b9947b2a208bde`
- Rank 1: `f5d2ce538bfe86b024c38070f74cce3411a457554ae08ec610ec26987570c629`

The original independent script SHA is
`d21d0eb6479564cd83c534a6a7dd926e48c8d50af6e72578280e94c87fc10473`.
The successful actual job
`letter-a0-20261112-final-audit-20261113-stageA-r1` ran it unchanged, using the
existing `/data/bwwu/envs/letter-20261110/bin/python`. Its runner started at
09:33:40.718387 UTC and ended at 09:33:47.643074 UTC, `SUCCEEDED`, exit 0,
cleanup verified and no live worker/supervisor. The server output is
`/data/bwwu/lc-rec/a0-final-audit-20261113/stage_a_r1/independent_original.json`.
The reviewer directly read the retrieved
[full independent output](../../evidence/final_audit_20261113/independent_original.json)
and [runner status](../../evidence/final_audit_20261113/stage_a_status.json).
The separately written [receipt checker](stage_a_receipt_check.py) then passed
[all 16 checks](stage_a_receipt_check.json), binding executed script, real worker
argv, full RUN, original metrics byte SHA, both prediction sizes/SHA values,
24,772 users / 495,440 candidates and unchanged tolerances.

| Original metric | Original saved value | Independent full recompute | Absolute mean difference |
|---|---:|---:|---:|
| HR/Recall@5 | 0.0903035685451316 | 0.0903035685451316 | 0 |
| HR/Recall@10 | 0.11335378653318262 | 0.11335378653318262 | 0 |
| NDCG@5 | 0.07766310391224206 | 0.07766310391224163 | 4.3021142204224816e-16 |
| NDCG@10 | 0.08508113200832956 | 0.08508113200832922 | 3.469446951953614e-16 |

All user identities, targets, original decoded-text parsing, stable score order
and stored relevance were checked by the full independent scan. HR hit counts
are exactly 2237 and 2808. NDCG sums differ by `1.0686562745831907e-11` and
`8.640199666842818e-12`, far below the original `1e-9` sum tolerance; compensated
summation accounts for the roundoff. The mathematical evaluation protocol and
per-user relevance agree; **aggregate NDCG floating-point values are not claimed
bitwise identical**. The original `metrics.json` was retained unchanged at SHA
`7cbbc077385923f284ee800b4318f2aefd4fd561b13c5925a24ae4e3429d471e`.

The independent anomaly recompute exactly reproduces 978 invalid candidates,
724 users with an invalid candidate, zero duplicate candidates/users, 4384
collision-SID candidate occurrences, 3660 users with a collision candidate,
and 122 users whose target SID collides. There are no skipped users.

The first CPU attempt failed before prediction reading because its `--data-root`
incorrectly pointed inside `LETTER-LC-Rec/data`, rather than `LETTER/data`.
The [original failure](../../evidence/final_audit_20261113/independent_original_round0_failure.json)
and [failed runner status](../../evidence/final_audit_20261113/stage_a_status_round0_failure.json)
are preserved. One task-level engineering correction fixed the input path;
neither the independent script, original metrics, prediction files nor numeric
tolerances changed. The separate successful `r1` attempt retains its own identity.

The full prediction scan was executed on the server through the controller's
accepted CPU runner. This reviewer independently read its source and real
receipts, but did not perform a redundant second scan or independently decode
token IDs. The earlier narrow SSH query verified final training records directly;
it must not be confused with independent execution of the full prediction scan.

## Interpretation limits

The numerical adaptation changes actual FA backward execution and is not proof
of bitwise identity with the author's unknown historical CUDA/four-GPU run.
This single seed does not establish cross-seed stability. Reusing released SID
assets establishes downstream reproduction, not reproduction of upstream SID
training. Similarity to paper metrics alone does not prove implementation
correctness. There is no Shuffle/A1 result or comparative causal claim here.

No training, inference, prediction/SID mutation, model selection, environment
installation, or GPU work was performed by this reviewer.
