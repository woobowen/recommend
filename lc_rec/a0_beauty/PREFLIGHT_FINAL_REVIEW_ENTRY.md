# A0 Beauty end-to-end preflight: partial delivery

`BEAUTY_END_TO_END_PREFLIGHT: PARTIAL`
`TECHNICAL_PREFLIGHT: PARTIAL`
`NONAUTHOR_REVIEW: PENDING`
`READY_FOR_FORMAL: NO`

The requested generation and independent audit are not complete. Neither
formal Beauty training nor full-user Normal/Shuffle was launched. This is a
partial delivery, not a final acceptance. Historical failure is retained in
`BLOCKED_BOUNDARY_PREFLIGHT.md` and its evidence.

## Identities and CPU preparation

Preflight RUN commit: `f54a2b505fd2d8767109a11d330a7064aff8af5e`.
The server export is
`/home/bwwu/grid-audits/sid-extensions-p0/transfers/beauty-preflight-f54a2b505fd2/release`;
220 files matched its SHA-256 manifest, whose SHA-256 is
`7b7c160687b91a3e9a23448a84081cc0b82211ee0877b320c1d483b00b95be5c`.
The outcome-blind 32-user manifest at `config/preflight_users_seed42.json`
has SHA-256 `714450a0e9479ed272266a5b4d3c2150d83742e4b70c38f80da888d4fa0aaa39`.
Budget, subprocess and manifest tests plus the actual Beauty tokenizer bind
are under `evidence/users-cpu-r0/`. The binding used the failed historical
run's saved tokenizer, not a trained adapter. Normal and Shuffle are untested.

The new budget ledger includes the failed model job, rejects missing or
ambiguous entries, and retains a 60-second cleanup reserve. CPU tests cover
missing failed receipt, invalid identity, unknown job, budget exhaustion,
nonzero subprocess propagation and timeout/reaping. No GPU work is launched
by these tests. CPU tests do not constitute model-level generation acceptance.

## Actual training and recovery

One protected runner job `letter-a0-beauty-e2e-20261116-r0` launched separate
Continuous, Early and Resumed processes. Their individual stage exit codes were
0. At the fixed guard cap the subsequent CPU comparison was terminated; the
runner reports **FAILED, exit 1**, while its own guard reports **TIMED_OUT**.
The runner's final observation verified no owned/unconfirmed/unsupported live
processes. The original result is not rewritten as PASS.

| Short epoch | cumulative microbatches/rank | optimizer updates | Continuous eval_loss | Resumed eval_loss |
|---|---:|---:|---:|---:|
| 1 | 10 | 2 | 12.54404067993164 | Early: 12.54404067993164 |
| 2 | 20 | 5 | 8.671497344970703 | 8.671497344970703 |
| 3 | 30 | 7 | 8.422368049621582 | 8.422368049621582 |
| 4 | 40 | 10 | 8.116425514221191 | 8.116425514221191 |

Early wrote `checkpoint-2` including both partial rank states and `complete.json`.
New Resumed rank processes each recorded `partial_restore_exact`, micro_steps
10, global_steps 2, optimizer_micro_step_id 1. See the rank event streams and
checkpoint manifests in `evidence/e2e-20261116-r0/`. This is a bounded real
model validation, not evidence that formal 131,413-row epochs will succeed.

After the timeout, CPU-only comparison job `letter-beauty-cpu-compare-20261116-r0`
failed because CUDA was hidden while the DeepSpeed CUDA accelerator tried to
query its device. Its failure is preserved. The same unchanged fixed comparator
then ran as `letter-beauty-cpu-compare-20261116-r1` with CUDA hidden and
`DS_ACCELERATOR=cpu`, exit 0. It compared both rank input/label hashes, row
order, losses, optimizer steps, learning rates, gradients, validation and
terminal state under the previously fixed `atol=1e-6, rtol=1e-3` rule;
identity, RNG, scheduler and counters are exact. The diagnostic copy
`boundary_compare_detailed.py` reran the same comparison in CPU-only job
`letter-beauty-cpu-detail-20261116-r0`: each rank compared 458 state tensors,
all 458 were exactly equal, with zero unequal elements, zero tolerance
violations and maximum absolute difference 0. Its per-tensor diagnostics and
runner receipts are included. The separate `preflight_early_compare.py` checked
Early versus Continuous first-epoch input/label hashes, ten loss values per
rank and first validation: PASS. These are author-executed checks, not an
independent review.

Both completed tracks selected `checkpoint-10` on the bounded eval loss and
saved adapter files with identical SHA-256:
`e56abd90bd3cc55d3786df04971a670bc12c54578110d255b5e7f3f4ec76af07`,
580,231,186 bytes. The full weight is retained on the server under
`/data/bwwu/lc-rec/a0-beauty-20261115/letter-a0-beauty-e2e-20261116-r0/{continuous,resumed}/preflight_adapter/adapter_model.bin`.
The training-side strict selected checkpoint restore and 450-key saved-versus-
active adapter check passed. Independent BF16 inference reload is not checked.

## Budget, gaps and reviewer

The four historical charges total 315.1950616836548 GPU-device-seconds. This
attempt conservatively charged 3001.351771686226, including time spent in the
CPU comparison before timeout. Total: 3316.546833369881; remaining:
283.4531666301191 of the 3600 ceiling. That is about 141.7 two-device wall
seconds before cleanup reserve and cannot reliably cover both remaining model
loads plus all required Normal/Shuffle generations. No further GPU attempt was
started. Normal Beam20, OFF equality, Shuffle3/4, model-level embedding, RoPE,
mask and cache checks remain outstanding. The preflight run timed out before
reaching even the post-adapter 32-user bind.

Three nonauthor agents were actually launched; each reported no direct access
to files or execution tools. They did not review source, raw receipts or run
checks. No independent PASS is claimed. Completing the task requires the
remaining bounded generation with an adequate authorized budget and a real
non-author review; formal training remains separately unauthorized.

Historical three correction rounds and the later authorized Collator round
remain unchanged; this turn used the user's new one-round authorization for
the budget and sample interfaces. The CPU-only compare retry changed the
accelerator setting only and did not change trained inputs or results. No
packages, toolchains or configuration files were installed or modified.
