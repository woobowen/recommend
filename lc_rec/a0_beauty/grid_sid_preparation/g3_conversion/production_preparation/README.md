# Fixed G3 Beauty production preparation — BLOCKED on dual-GPU availability

This continues the historical conversion-only release without changing its old
reports. It is **not READY_FOR_REVIEW**: the mandatory actual-model checks have
not run. No formal experiment was started and no G3 formal launch authorization
was created. The complete acceptance matrix is [acceptance.json](acceptance.json).

## Implemented and actually checked

The production training, strict loading, generation, evaluation and controller
now share `g3_identity.py`. It pins the G3 tensor, bridge, converted index,
LETTER interactions/items, original LLaMA revision and full model file hashes.
The tokenizer must have 786 added tokens, vocabulary 32786 and the fixed token
mapping. The original three quantization digits and fourth deduplication digit
are preserved. Old LETTER adapters and tokenizers cannot satisfy these gates.

The original training and metric methods remain: int8 LLaMA training, BF16,
DeepSpeed stage 2, original attention compatibility, full embedding/head
modules_to_save plus LoRA, batch 8/rank and accumulation 4, seed 42/four epochs,
original loss/best selection, beam 20 and native position callback, unchanged
normalization, score sorting, duplicate-candidate and invalid-SID treatment.
The new config differs in experiment/SID description and this task's 240
GPU-device-minute preparation budget. Static inventory expectations remain
**expectations**, not a model measurement.

The saved finite training driver uses four ten-microbatch epochs and update
boundaries `[2,5,7,10]`. Continuous and cross-process recovery reuse production
logic and the established `atol=1e-6, rtol=1e-3`; RNG, loader, counters and
scheduler stay exact. To respect the 80 GiB limit, verified continuous temporary
checkpoints are removed only after their manifests, adapter and comparison state
are retained; early/resumed training shares one output directory. **This cleanup
path has not executed**, since no G3 GPU job started.

Finite generation loads one actual trained adapter per rank and performs
Normal/OFF/S3/S4 in one process, records actual model prefill and embedding
lookup, and requires exact Normal/OFF outputs. The old fixed 32 user IDs are
retained with G3-derived fields. All 22363 real visible histories contain no
repeated item: repeat-history boundary fixtures are explicitly synthetic and
excluded from user metrics. `g3_evaluate_preflight.py` consumes genuine future
outputs through production aggregation, offline metrics and paired statistics.
Formal mode retains the full 22363-user guards.

## Direct runtime evidence

Runtime snapshot: `9d93d01dd5fccf2e667a6a5c9486f2539a50390b`.

- [CPU r2 result](jobs/g3-lcrec-cpu-20261118-r2/output/cpu/result.json): PASS,
  all 131413 training rows and 22363 validation/test targets; 1148 label positions;
  ten actual callback invocations; 26 targeted negative cases; tokenizer reload.
  23.951 s, peak RSS 1428463616 bytes, no CUDA use.
- [Formal dry-run r2](jobs/g3-lcrec-dryrun-20261118-r2/output/dry_run.json): PASS,
  actual paths, source/config identities, argv and dependencies; no formal job
  submission. Draft preparation does not seal the eventual launch plan. Missing
  G3 authorization and old-experiment authorization are rejected. Future
  checkpoints/predictions are planned dependencies, not fabricated files.
- Both final jobs are SUCCEEDED/exit 0 with runner-owned process-group cleanup.
  Requests, accepted digests, logs and original status/cleanup records accompany
  the results. Fresh SSH queries confirmed the accepted jobs after submission.
- The failed dry-run r0 is retained: deployment incorrectly excluded vendor
  `test_ddp.py`. Round 1 repaired deployment closure and draft/provenance issues;
  round 2 tightened generation RUN/config binding and revalidated it. Cumulative
  correction round: **2 of 3**. CPU runner wall across five jobs: **166.682025 s**.
- **GPU usage: 0 GPU-device-minutes.** No G3 model or optimizer inventory,
  update, save/reload/resume, real four-condition prediction or real paired
  result is claimed. Previous LETTER GPU results are not reused as G3 PASS.

## Direct non-author review

Three capable agents read source and raw evidence directly. Their reports bind
paths/bytes/SHA and distinguish authorship:

- [Generation/metric review by training agent](review_training_agent.json).
- [Training/control/CPU evidence review by generation agent](review_generation_agent.json).
- [Control and raw evidence review](../../../evidence/g3-production-20261118/internal-review/cpu_validation_static.json).

All final CPU and dry-run import/source identities were independently compared
to the complete runtime Git commit. Findings and fixes are recorded, including
an explicitly withdrawn manifest-path finding. The internal review is **PARTIAL**
because actual-model evidence is missing. No overall PASS or external GPT
second-review approval is claimed.

## Resource blocker and bounded continuation

[Resource observations](resource_observations/) show GPU0 occupied by another
account's persistent `VLLM::EngineCore` process (PID 3304681, about 20 GiB), while
GPU1 is free. The process has run for over 25 hours. Repeated fresh observations
do not establish a simultaneous two-card window. No other user's process was
stopped, migrated or changed. The six-hour wait ceiling was not exhausted and
is not claimed as exhausted; the current blocker is persistent external
occupation without an authorized way to release it.

All source/CPU/dry-run/review work independent of this blocker is delivered.
Minimum unblock condition: both fixed RTX4090s become available. Existing
authorization covers the remaining bounded PREFLIGHT; it does **not** cover
FORMAL. Use the existing local deployment metadata
`/home/addaswsw/recommend-support/sid-extensions-p0/packages/g3-current-deployment.json`,
whose pinned runtime/source is also in [run_deployment.json](run_deployment.json).
Do not deploy a new HEAD or rerun a known job blindly.

```bash
python3 lc_rec/a0_beauty/g3_remote.py probe
python3 lc_rec/a0_beauty/g3_remote.py submit --stage train --job-id g3-lcrec-train-20261118-r2
python3 lc_rec/a0_beauty/g3_remote.py query --job-id g3-lcrec-train-20261118-r2
```

After the original train job exits and artifacts are checked, use distinct
`compare`, `generate`, and `evaluate` jobs, then direct non-author review.
`g3_preflight_driver.py` requires the passed CPU receipt, checks unchanged import
bytes, refreshes both devices before each model subprocess, enforces cumulative
240 GPU-device-minutes, bounded wall time and space/host-memory reserves. The
server output root is `/data/bwwu/lc-rec/g3-preparation-20261118/9d93d01dd5fc/`.
There is no active GPU job to resume at this delivery.

Formal preparation is a separate draft at
`/data/bwwu/lc-rec/g3-formal-20261118/9d93d01dd5fc/preparation-plan.json`.
Its budget is planning-only. After the missing preflights and webpage second
review, prepare a final submission plan with that review and obtain a new user
launch authorization bound to its complete plan/source/config/budget identity.
The old LETTER authorization cannot launch G3.

## Delivery and environment

Small raw evidence and actual source are published in `wbw-lc-rec`; large models
stay on new-server. Runtime and delivery commits may differ only in the
transport collection guard, package builder and review materials. See the
delivery readback receipt for exact comparison. ZIP member hashes, CRC, path,
duplicate and secret-pattern checks are written outside the ZIP together with
the whole archive SHA; the archive does not contain its own hash.

No system, pip/npm or toolchain package was installed; no global environment,
SSH, driver or network setting changed. Existing per-job CUDA variables fixed
an initial dependency probe. [installation_log.json](installation_log.json)
records this. Original assets, historical reports, failed receipts and all
unrelated untracked files were retained.
