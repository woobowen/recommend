# A0 Beauty formal engineering review — partial

`FORMAL_ENGINEERING_READINESS: PARTIAL`
`TECHNICAL_ENGINEERING_CHECKS: FAIL`
`NONAUTHOR_REVIEW: PENDING`
`FORMAL_READY_CANDIDATE: NO`
`FORMAL_LAUNCH_AUTHORIZED: NO`

This delivery contains useful engineering changes and real CPU evidence, but
not a complete, safe formal controller or full formal evaluation acceptance.
Do not submit formal training, Normal or Shuffle jobs based on this delivery.
No GPU job was submitted for this task. No synthetic score is a research result.

## Identities and inherited evidence

- Training/recovery preflight RUN: `f54a2b505fd2d8767109a11d330a7064aff8af5e`.
- Inference preflight RUN: `63f13def364d0852ad8bc0a20f10e41fd434446c`.
- Prior partial delivery: `5f6910054368fc3938c0533e5e91966de77b048d`.
- This CPU evaluation code snapshot: `271df1e33054c19b3ad3a3330a7a50949d633073`; this is **not** a formal training RUN.
- Three earlier GPU training/resume trajectories and the 32-user inference
  preflight were not repeated. Their prior runner failure is not rewritten.
- CPU source export: `/home/bwwu/grid-audits/sid-extensions-p0/transfers/beauty-engineering-cpu-271df1e33054/release`. Its runner request records each file and input SHA; no Git checkout is claimed.

## What changed

`offline_metrics.py` now independently binds each rank's complete ordered
assignment to the previously verified seed-0 22363-user sampler SHA and checks
the fixed bootstrap parameters. `paired_beauty.py` checks the canonical 16
columns, finite values, formal rank reports, adapter equality, input hashes,
per-user artifact hashes and frozen data identities; it rejects the known
preflight Adapter and Python optimization mode. It permits different prediction
RUN commits when the recorded model identity matches. This is **not** a proof
that an Adapter came from the formal training best checkpoint: the missing
outer controller must establish that binding.

`derive_formal_runner.py` derives a candidate from the exact historical A0
runner SHA, changing only two worker-session ownership namespace anchors. It
does not deploy or authorize a runner. `test_beauty_runner_cpu.py` and
`engineering_dry_run.py` are isolated CPU tests, not a production controller.
`synthetic_full_cpu.py` exercises the Beauty primary evaluator with fixed,
explicitly synthetic predictions in an isolated output directory. The small
negative tests are in `test_formal_identity_gates.py` and
`test_prediction_rejections.py`.

## Synthetic full-user CPU evidence

CPU job `letter-beauty-synthetic-cpu-20261116-r0` used the existing protected
runner with `CUDA_VISIBLE_DEVICES=''`, exited 0 and recorded verified cleanup.
Elapsed CPU job wall time was approximately 82.5 seconds; the three primary
metric reports recorded peak RSS up to 585,785,344 bytes. The server output was
approximately 213 MiB and remains at
`/data/bwwu/lc-rec/a0-beauty-20261115/SYNTHETIC_CPU_INTEGRATION_ONLY_271df1e33054_r0`.
The request and raw light results are in `evidence/formal-engineering-cpu-r0/`.

The actual author Beauty asset hashes and 22,363-user axis were checked.
Normal, S3 and S4 synthetic files each contained 11,182 / 11,181 rows and
447,260 candidates. The primary evaluator calculated 16 metrics, coverage,
groups and 5,000 bootstrap draws per condition, and compared its CCE formula
with the frozen official implementation for all 22,363 users per condition.
This exercise's stage-A metrics were created by the fixture program, **not**
by running the original LETTER evaluator, so it does not satisfy the requested
independent original-evaluator integration.

For S3 and S4 the synthetic paired differences were calculated in the fixture
program and compared with the frozen independent column-wise bootstrap; maximum
absolute draw differences were `3.99854e-15` and `3.93956e-15`, respectively.
The formal `paired_beauty.py` entry and its formal provenance gate were **not**
run against fabricated formal receipts. An independent nonauthor's separate
program re-derived 10 ranking metrics over the full axis, maximum absolute
difference `5.21805e-14`, and recomputed six CCE values for 100 users per
condition (600 comparisons each), maximum difference zero. Its source and
output are included as `independent_statistics.py` and `.json`. The
nonauthor did not independently recompute full-user CCE or bootstrap.

Twelve prediction rejection cases were tested on three actual Beauty user
identities with synthetic candidates. The identity test separately exercises
bootstrap and prediction/per-user artifact binding. These tests do **not**
cover all required 15 failures at full scale.

## Runner and simulated plan

Current server shared runner SHA:
`398768a732bbfebde4308486a0cedd924595157fda6cbff6a88f0bfaa512beff`,
with an 86,400-second single-job cap. The existing long A0 runner is
`/home/bwwu/grid-audits/sid-extensions-p0/scripts/runner-a0-96h-fff2df040295bf9f.py`
SHA `fff2df040295bf9f342b8bb3144d05e28b5b2cc5aff41f8d47ae312847489bb5`;
its deployment sidecar SHA is
`e610be95a47894472c3f6b8396af8eb48fb1a998696a3a4914af2d886cc00877`.
The original 96h runner's cross-session ownership recognizes only historical
A0 job prefixes, not Beauty. The derived candidate SHA is
`cc1a1e277faa9bc713a4d0d1b620248d47c398e8ed17d4ed3d7ab52805cc657e`.

The server CPU-only isolated probe tested 345600/345601 validation, marked
cross-session sleeper ownership, pidfd termination, an unmarked control,
actual synthetic submit→timeout→cleanup, and duplicate-request no-restart.
The candidate has not been formally deployed or tested with real torchrun
workers, disconnected SSH, real checkpoint recovery, or a cumulative 96h
multi-stage budget. A new formal resource budget and user authorization remain
necessary. No public runner or server system configuration was modified.

The in-memory `engineering_dry_run.py` tests dependency ordering, failed and
unknown states, no duplicate simulated completion and cleanup checks. It
cannot submit jobs, maintain a persistent lock, enforce real authorization or
perform checkpoint recovery. It is not a formal scheduler. The required
formal chain remains train → best Adapter → Normal → Normal evaluation → S3 →
S3 evaluation → S4 → S4 evaluation → paired statistics; it is not implemented
and verified end-to-end. Historical Instruments controllers cannot serve as a
Beauty formal controller without the requested additional adaptation.

## Method-difference classification

The following table describes the intended and inspected data-domain adaptation;
it is not a claim that the incomplete formal orchestration has been verified.

| Component | Classification | Basis / current scope |
|---|---|---|
| Data and split | BEAUTY_DATA_ADAPTATION | Frozen author Beauty files and user axis. |
| SID | BEAUTY_DATA_ADAPTATION | Original Beauty four-position author SIDs. |
| Tokenizer | BEAUTY_DATA_ADAPTATION | 965 Beauty symbols; vocabulary 32965. |
| Model | UNCHANGED | Frozen LLaMA-7B revision and LoRA family. |
| Trainable parameters | UNCHANGED | Existing seven projections plus embedding/head. |
| Loss | UNCHANGED | Existing response-only CE and temperature. |
| Optimizer | UNCHANGED | Existing AdamW settings. |
| Scheduler | UNCHANGED | Existing cosine/warmup settings. |
| Gradient accumulation | PREVIOUSLY_APPROVED_TRAINING_ADAPTATION | Cross-epoch accumulation retained; no training code edited this turn. |
| Precision | UNCHANGED | Existing int8/BF16 training and BF16 inference. |
| Checkpoint | PREVIOUSLY_APPROVED_TRAINING_ADAPTATION | Existing partial-gradient save/restore; not rerun this turn. |
| Prompt | UNCHANGED | Existing Prompt0 path. |
| Beam | UNCHANGED | Existing Beam20 path. |
| Generation callback | UNCHANGED | Existing positional callback. |
| Shuffle core | UNCHANGED | Existing protocol and seed; no core edit. |
| Evaluation | BEAUTY_DATA_ADAPTATION | Beauty axis/catalog and added identity checks; ranking/CCE mathematics unchanged. |
| Runner | BEAUTY_DATA_ADAPTATION | Candidate changes ownership namespace only; full safe deployment remains unverified. |

`NEW_METHOD_CHANGE: NONE` for the changes in this delivery; this is not an
end-to-end method correctness claim.

## Independent review and remaining gates

Two nonauthor agents directly read code and ran CPU checks. One reviewed the
runner ownership and actual identity tests; another independently checked
synthetic full-axis ranking and bounded CCE. Their scope is limited to those
checks, not a nonauthor PASS of the missing complete engineering chain.
Findings about per-user hashes, disabled assertions and unknown dry-run states
were corrected and retested. The updated runner test uses isolated synthetic
sleep processes, not real torchrun.

Still required: real formal orchestration and authorization gate; trustworthy
formal best-checkpoint/Adapter provenance; complete formal CPU evaluator
wiring and full negative-injection suite; formal paired entry testing; 96-hour
cumulative budget, lock, real runner ownership and resume checks; and final
non-author review. Formal launch also needs separate user authorization. No
new packages, drivers, tools or persistent system configuration were installed.
