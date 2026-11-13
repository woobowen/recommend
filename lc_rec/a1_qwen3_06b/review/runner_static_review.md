# A1 runner, pipeline and CPU isolation: nonauthor review

Reviewer: real sub-agent `a0_protocol_review`, nonauthor of the implementation.
Date: 2026-11-13.

**RUNNER_STATIC_REVIEW: PASS within the scope below.**

**NOT_FORMAL_PRELAUNCH_PASS.** This report does not approve formal training.
Actual asset/tokenizer CPU execution, the same-path GPU preflight, strict
fresh-process checkpoint recovery, validation/best/inference checks, budget
receipts and the complete nonauthor prelaunch gate remain required. No A1 formal
prediction or final metric is reviewed here.

## Scope and independence

The reviewer read `scripts/control.py`, `pipeline.py`, `runner.py`, `ledger.py`
and `cpu.py` under `lc_rec/a1_qwen3_06b/`, the unchanged A0 accepted runner and
ledger, reused CPU tokenizer/data audit, and the new CPU Int8 serialization
function called by the wrapper. The applicable project rules were explicitly
read from `/home/addaswsw/recommend/` earlier in this task.

Access was local filesystem plus bounded stdlib CPU checks. The reviewer did not
initiate SSH, submit jobs, run GPUs, load models, install packages, modify the
implementation or perform Git writes. The actual timeout receipt below was
collected through the main controller's existing server bridge. Reading and
validating that receipt is not an independent SSH/GPU reproduction. The only
new file authorized for this subtask is this report. The reviewer has not
authored the production A1 evaluator and remains independent for its later audit.

## Accepted runner comparison

The full textual comparison with `lc_rec/letter/scripts/formal_runner.py` showed
exactly two changes: append `letter-a1-q06-20261113-` to the prefix tuples in
`observe_group` and `signal_member`. No other runner code changed at review time.

Consequently, the accepted immutable request/ID mechanism, exclusive submit/run
locks, durable `attempt.json` claim, detached supervisor, dual task markers,
same-account process scope, PID/start/boot/session checks, pidfd-bound signals,
bounded TERM/KILL cleanup and refusal to infer success from leader exit remain
the existing implementation. The legacy `A0_MARKED_WORKER_SESSION` proof text is
a retained label for the same mechanism; it does not identify this A1 job as A0.

The client restricts job IDs to the A1 prefix. An identical accepted request
queries the existing job without restart; the same ID with different immutable
request content raises `REQUEST_ID_CONFLICT`. Querying an unresolved original
job does not create a new seed or infer terminal success.

## Pipeline, ledger and resource gates

Separate A1 paths are used for release, data outputs, GPU lock, preflight ledger
and formal seed ledger. A0 pairable CSV/NPZ files in `cpu.py` are opened only for
reading and identity verification. No inspected write targets an A0 model,
adapter, checkpoint, prediction, metric, ledger or source directory.

`ledger.reconcile` requires every recorded prior runner to be terminal, to have
an ending timestamp and to have verified cleanup. ACCEPTED, RUNNING and
CONTROL_LOST records are rejected, as is unverified cleanup. Pipeline `finally`
charges are provisional: reconciliation takes the maximum of that value and
the runner's start-to-terminal elapsed time, which includes worker cleanup.
Every such interval is conservatively charged to two devices.

The preflight budget is 3600 GPU-device-seconds, equivalent to 1800 seconds of
dual-device reservation. The formal seed budget is 345600 cumulative wall
seconds (96 hours). Pipeline stage limits subtract accumulated prior charges
and reserve 20 wall seconds for cleanup; client runner timeouts reserve 10 wall
seconds. The runner still owns identity-checked termination. Final ledger
reconciliation and actual budget receipts remain necessary after a real job;
the provisional pipeline `finally` value alone is not the final charge.

The GPU pipeline takes a nonblocking A1 GPU lock, reconciles prior attempts,
rejects duplicate job IDs, checks for existing GPU compute processes and records
the actual device UUIDs. It monitors host memory, disk reserve, bounded log
progress and cumulative time. An exception exits nonzero and leaves child
termination to the accepted runner; it does not issue broad process signals.

Formal launch additionally requires a prelaunch PASS document bound to the
complete RUN SHA, actual config SHA and every current script SHA. A subsequent
formal attempt cannot be a second fresh seed: it requires `--resume` or
`--resume-evaluation`. Training resume is restricted to the latest sealed
checkpoint in the formal training directory. Evaluation-only resume verifies
both rank completion records, step 8240/epoch 4, RUN/config identity and best
adapter SHA, then reuses completed training instead of retraining. The actual
training/checkpoint implementation and runtime evidence need separate review.

After initial inspection, the author added a GPU-entry CPU asset gate. The
reviewer read the updated block: GPU paths require the existing CPU report's
PASS, exact Qwen revision, expected weight size/SHA and CPU Int8 serialization
contract PASS. The asset report path/SHA is recorded. This static check does not
assert that the CPU report or GPU preflight has actually passed.

## CPU isolation

The client sets `CUDA_VISIBLE_DEVICES=""` for CPU, comparison and timeout
fixture jobs. The reused CPU tokenizer/data audit asserts that environment
before importing Torch inside its execution path and checks at the end that
CUDA was not initialized. Its model inventory uses meta tensors, not GPU model
weights. The newly called `cpu_int8_state_contract` constructs tiny CPU synthetic
Int8 tensors, exercises strict cold/hot serialization and explicitly checks
that CUDA is uninitialized. It does not claim to test GPU arithmetic or complete
DeepSpeed recovery.

The wrapper reads the official model's actual local identity, pinned dependency
versions and the fixed A0 paired assets. These statements describe inspected
source behavior; actual runtime manifests still need to be read before PASS.

## Actual local bounded checks

All five inspected scripts parsed successfully with Python `ast.parse`, without
importing the runtime modules, Torch, DeepSpeed or the transport.

The reviewer executed these nine directed stdlib fixtures against AST-extracted
implementation functions, with an in-memory ledger filesystem and tiny temporary
local accepted-job records:

1. Terminal runner elapsed charge replaces the shorter provisional pipeline
   value and includes cleanup; device charge is exactly twice wall charge.
2. ACCEPTED prior job is rejected by ledger reconciliation.
3. RUNNING prior job is rejected.
4. CONTROL_LOST prior job is rejected.
5. Unverified prior cleanup is rejected.
6. A larger existing conservative charge is preserved with its doubled device
   charge.
7. An identical accepted request returns `EXISTING_NO_RESTART` without launching
   a process.
8. A changed request under the same accepted job ID raises the immutable-ID
   conflict.
9. The original request bytes remain unchanged after that conflict.

**Result: 9/9 PASS, final fixture command exit 0.** The first version of the test
harness omitted the runner's `read` helper and failed with `NameError`; adding
that dependency to the harness fixed the fixture. No production implementation
was changed, and the failure is not evidence of a runner defect. This fixture
does not substitute for integration testing of namespace, transport or pidfds.

## Actual bridged timeout/cleanup receipt

The reviewer directly read and independently asserted the fields in:

`/home/addaswsw/recommend-support/a1-qwen3-06b-instruments-20261113/receipts/query-letter-a1-q06-20261113-runner-timeout-r0.json`.

Receipt identity: **6043 bytes**, SHA-256
`2db3b6fb6eff422f85831037c014ea3f9662277e5684d301ad749135d9005243`.

- Job: `letter-a1-q06-20261113-runner-timeout-r0`.
- Actual fixture release from worker argv:
  `65a36ad38eae6a5b23dababa223135784a1587ae`.
- Runner path:
  `/home/bwwu/grid-audits/sid-extensions-p0/scripts/runner-a1-96h-1fefc09a15c1cb73.py`.
- Terminal status **TIMED_OUT**, exit **-9**; `cleanup.verified` and queried
  `cleanup_verified` are both true.
- Started `2026-11-13T11:07:24.733316+00:00`; terminal
  `2026-11-13T11:07:32.027391+00:00`; elapsed **7.294075 seconds**.
- Cleanup elapsed **2.229330 seconds**. Both the original leader PID3233941 and
  marked separate-session child PID3233942 received recorded identity-scoped
  TERM and KILL signals. Final query has no live supervisor, worker or workgroup,
  no unconfirmed/unsupported live member and no scan error. Its observed group
  is verified empty; `worker_may_be_live=false`.
- `local_request_verified=true`; canonical request SHA is
  `246383f22d40a625130fd8a0d6b2a1790fd5216b0c49eab2657d7d5a727c219a`.

The reviewer's receipt assertions exited 0. This is real server-run fixture
evidence supplied through the main controller, distinct from the local synthetic
fixtures. Its deliberately timed-out status is the expected timeout test result,
not a failed A1 training run. No GPU training/checkpoint claim follows from it.

## Source identities and remaining limits

Latest post-CPU-gate files read for this report:

| File under `lc_rec/a1_qwen3_06b/scripts/` | Bytes | SHA-256 |
| --- | ---: | --- |
| `control.py` | 8982 | `b56eb02191bcdb35b52f552526d84a7f8edd6ebac52e5710734299221e346c09` |
| `pipeline.py` | 12096 | `48f17adf782aba693e8ee0d9754e7823ddfbbc68a0cc27f9cb2ab042e74aa3ae` |
| `runner.py` | 24826 | `1fefc09a15c1cb73878417f9eb50d0fe9e8fa1b2f60cc70b27f7cc4ed9cb0947` |
| `ledger.py` | 1684 | `51e643630008d7b4a724a23def9722fae0bad471cf30c967e9524d20ec7d954e` |
| `cpu.py` | 2725 | `fa14305915901f5068cf1e40425c5173c6bc1b051fcdf492f37dd0bcb036b5bb` |

The earlier pre-gate inspection used pipeline SHA
`32d1f5843d89f0eea237e331e1eb94b7b17018faf11866cfdfdad3906b8646f8`
and CPU wrapper SHA
`f55ae256ad5cda7b4b0a369a4a10a463bb49101937357483af82cf13e4dc30b2`.
Ledger and accepted-ID fixtures exercise unchanged functions. Subsequent source
changes require binding the new bytes to their applicable review/evidence.

One low-impact monitoring recommendation remains: at inspection, `observe`
recursively collected all eligible JSON/JSONL/log files up to 8 MiB, like
`collect`. Regular long-run monitoring should read status and bounded log tails;
full collection belongs to milestones/termination. The author stated this would
be changed; this report does not certify uninspected future bytes.

No blocking issue was found within this runner/static scope. Actual deployment,
GPU recovery/inference and final prelaunch reviews remain pending.

GPT_UPLOAD_REQUIRED: NO for this sub-review; the main controller must include
the report and required bridged receipt in the fixed A1 publication/ZIP.
