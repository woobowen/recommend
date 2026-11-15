# A0 Beauty: real-model boundary preflight blocked

Status: **BLOCKED**, not a formal training RUN or final experimental delivery.
The task's three concentrated correction rounds had already been counted
before this preflight. The failed acceptance below is a new necessary
correctness blocker; the limit has not been reset.

## Execution and result

Preflight source snapshot `a7b2cd23a661600687bc143c37a995802cfdb016`,
published to `wbw-lc-rec`, was exported to the server with 186 files checked
against the Git archive. This snapshot is not the formal experiment RUN.

Job `letter-a0-beauty-boundary-model-20261115-r0` ran on `zwlab-01`, two
RTX 4090 GPUs, from 2026-11-15 04:15:21 UTC to 04:17:00 UTC. The independent
runner reported `FAILED`, exit 1, and verified no remaining owned processes.
The guarded upper-bound charge was 197.9677 GPU-device-seconds; together with
the three earlier probes the cumulative conservative charge is approximately
315.1951 GPU-device-seconds. Both GPUs subsequently reported 1 MiB in use and
no compute processes. The job was preflight only.

Both ranks loaded the Beauty LLaMA/LoRA path and emitted finite micro-losses;
the deterministic FlashAttention forward/backward hooks fired. The first short
epoch consumed 10 microbatches per rank, made 2 optimizer updates, and reached
its intended validation boundary with 2 pending accumulation microbatches.
The native sampler reconstruction passed for this short epoch. These facts do
not establish that validation, checkpointing, cross-process recovery, best
adapter restoration or inference work correctly.

Validation failed on both ranks before evaluation or checkpoint creation.
`formal_train.py` wraps the training dataset in `IndexedDataset` and gives the
Trainer `IndexedCollator` for *all* data loaders; the actual validation
dataset still yields dictionaries. `training_rows.py` attempts to unpack each
validation dictionary as `(index, sample)`, and the original LETTER collator
receives a string instead of a dictionary, raising `TypeError: string indices
must be integers`. See `evidence/boundary-model-r0/work.log` for both full
tracebacks. This is an engineering bug, not evidence of a research-method
failure or an acceptable reason to skip validation.

The preferred next correction is to ensure that the native, unwrapped LETTER
collator is used for validation while retaining `IndexedCollator` only on the
training loader, then execute a CPU test covering both actual dataset item
types and the same bounded real-model validation/recovery comparison. This
must preserve the original distributed validation and `eval_loss` aggregation.
That proposal is not implemented or validated here: the authorized task-level
correction limit has been reached. The expected cost is a code and CPU
regression followed by a new bounded GPU preflight within the remaining budget;
its actual duration and success cannot be inferred from this failed attempt.

## Evidence and uncompleted dependencies

Runner evidence is under `evidence/boundary-model-r0/` and the verified full
archive is at
`/home/bwwu/grid-audits/sid-extensions-p0/jobs/letter-a0-beauty-boundary-model-20261115-r0/evidence.tar.gz`,
8,276 bytes, SHA-256
`05e8cc473784b7a3526190e8ed203e17b663585d33408042da9f28f2660f5142`.
Original preflight outputs and rank logs remain at
`/data/bwwu/lc-rec/a0-beauty-20261115/letter-a0-beauty-boundary-model-20261115-r0/continuous`.
The guard's own `status` remained `STARTED` because its exception path did not
rewrite that field; the authoritative runner status and exit code are FAILED/1.

No Beauty formal training, best checkpoint, Normal Beam20, Shuffle3, Shuffle4,
16-metric results, bootstrap, independent result recomputation or non-author
final review exists. No result table or final experiment ZIP can honestly be
provided. A partial ZIP may be made for inspection but is not a final ZIP.
The non-author agent only inspected bridged excerpts and did not directly read
the server evidence or independently execute the final audit.
