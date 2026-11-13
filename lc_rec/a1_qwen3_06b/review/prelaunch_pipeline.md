# A1 final pipeline and CPU handoff: nonauthor prelaunch review

Reviewer: actual sub-agent `a0_protocol_review`, independent of the production
control, training, inference, primary evaluation and paired implementations.
Date: 2026-11-13.

**PIPELINE_PRELAUNCH_CODE_AND_CPU: PASS.** No necessary code/CPU-interface blocker
was found in the exact core bytes listed below. **FORMAL_R_E_DEPLOYMENT_BINDING:
PENDING** at report creation. The main controller must bind the frozen GPU RUN
R and independent evaluation RUN E to those reviewed core bytes, deployment
receipts and accepted immutable request before launch. This report alone is not
an overall A1 prelaunch PASS and is not a formal experiment/result PASS.

## Independence and evidence scope

The reviewer explicitly read the project rules from `/home/addaswsw/recommend/`
earlier in this task, directly read the actual local implementation and fixed
historical blobs, and executed bounded local CPU/source fixtures. No transport
module, Torch, DeepSpeed, model, SSH or GPU was used in this review's fixture.
No production implementation or Git state was changed by the reviewer.

The reviewer owns independent expanded/CCE and paired-statistical verifier code,
not the primary metric or paired implementation. Bridge-supplied server
receipts are labeled as such. This is not an independent SSH/GPU reproduction.
The earlier `runner_static_review.md` remains unchanged as historical evidence.

## Frozen GPU/evaluation identity and automatic handoff

`control.py` requires a formal evaluation commit, resolves its full Git identity,
loads its separate deployment receipt and forwards the exact E release path.
The accepted request's `capture_version` and working directory remain R; its
explicit `--evaluation-commit` and `--evaluation-release` identify E. All R
script/config/vendor inputs and E script/review/config inputs are added to the
accepted runner's actual size-independent SHA verification list. The prelaunch
gate binds R, E, config SHA and every current R script SHA.

The server requires E's release-directory name, `RUN_VERSION.json` identity and
prelaunch approval to agree before GPU computation. Actual E export/core bytes
must also match this review's hash table through the deployment/readback receipt;
the commit string alone cannot replace that check. The E/R commits were not yet
assigned when this review was written, so no invented fixed RUN identity is
claimed here.

After training/best export and complete full-user testing, the same persistent
server pipeline invokes E's `offline_pipeline.py` directly with explicit R and
E, the original fixed training directory, completed test directory, frozen data
and a new per-attempt `offline_evaluation` output directory. This transition
requires no resumed Codex/WSL/SSH session or local file service.

The CPU child receives an inherited task-marker environment with
`CUDA_VISIBLE_DEVICES=""`; it is not started with torchrun or a new unmanaged
session. Its grandchildren inherit the same markers and session. The accepted
runner's existing identity/pidfd cleanup scope therefore continues to own the
whole pipeline. CPU nonzero exit propagates through the server pipeline; an
offline result must actually contain PASS and the requested E identity before
the outer job can declare SUCCESS.

## Original-to-derived identity chain

The inspected chain consistently binds:

1. Both complete formal training records and best selection to R and the actual
   best adapter SHA.
2. Both prediction rank reports to that adapter, the fixed Qwen revision and the
   same data/author/generation metadata.
3. Actual full prediction SHA/size/path records to the immutable A1 manifest.
4. Complete Stage-A original metric/user reconstruction to that RUN and those
   shards; primary and independent expanded evaluation use that same manifest.
5. All 24772 by 16 independently reconstructed metrics to the primary per-user
   CSV, then the unchanged independent grouping/bootstrap verifier to the full
   CSV/NPZ matrices and frozen data axes.
6. Paired main output to the fixed A0 and A1 matrices, followed by the independent
   paired verifier's complete 5000 by 16 replicate/CI recomputation and actual
   PASS gate.

The current offline pipeline includes the independent paired stage, closing
the earlier source-only audit-coverage gap. No raw predicted SID is filtered,
deduplicated or collision-disambiguated by these interfaces. Formula and fixed
asset evidence remains in `a0_protocol_basis.md` and
`evaluation_preparation_review.md`; it was not unnecessarily re-audited here.

## Evaluation-only recovery and preservation

`--resume-evaluation` requires a prior attempt, both R/config-bound
TRAINING_COMPLETE records at step8240/epoch4 and the selected adapter's actual
SHA. The train stage is recorded as reused and is not invoked. Within prior
attempts, a completed full test must match R, the selected adapter and 24772
users with both rank PASS reports. At most one such test may be found. It is
reused read-only; the fresh frozen-manifest/Stage-A chain then verifies complete
SHA/user/target coverage rather than trusting the initial summary alone.

If no completed full test exists, testing uses a new attempt's output; existing
failed or complete prediction directories are not overwritten. If training is
already completed, the ordinary training path rejects a fresh training start
instead of retraining from an older best. Subsequent formal attempts still
require resume or evaluation-only resume, and unresolved prior runner cleanup
blocks restart through the existing ledger reconciliation.

A0 pairable files and original assets are read-only inputs. Write destinations
remain under the new A1 job/output paths; primary and reviewer outputs reject
pre-existing destinations. The independent paired verifier also forbids writing
inside either primary input directory. This review found no A0 overwrite path.

## Budget and resource ownership

The automatic CPU stage runs through the same `run` function and resource guard
as the GPU stages, retaining the original attempt's cumulative deadline, GPU
lock and persistent ledger. The unique formal seed remains bounded by 345600
wall seconds (96 hours), including training, validation, checkpoint, recovery,
test and now the chained CPU work. Client/pipeline cleanup reserves remain
10/20 wall seconds; terminal runner reconciliation includes actual cleanup.

The ledger conservatively charges both devices for the whole reserved job,
including CPU audit time. This is reserved-device accounting, not a claim that
CPU evaluation executed GPU kernels. The preflight ledger is still the same
3600-device-second ledger; inference-only checking does not reset it. Resource
failure or time exhaustion exits nonzero and leaves bounded termination to the
accepted runner.

The current inference-only branch skips training, checks complete CP12 and both
PROBE_COMPLETE12 source records, and now compares all 394 checkpoint/adapter
key sets, BF16 dtypes, shapes and tensor values on CPU before consuming the
adapter. It records separate training and inference source identities and
adapter hashes. This closes the previously reported source-level adapter
binding gap. The main controller reported inference-r3 SUCCEEDED, but its raw
server receipt/predictions had not yet been supplied for this report, so no new
runtime inference PASS is claimed here.

## Actual local CPU verification

Saved reviewer fixture: `review/cpu_handoff_fixture.py`. It parses current source
and executes the actual extracted pipeline branches with fake subprocesses and
small temporary metadata explicitly marked synthetic. It imports neither the
production runtime modules nor any GPU framework. Command:

```text
python3 -B lc_rec/a1_qwen3_06b/review/cpu_handoff_fixture.py
```

Actual exit 0; **13 directed checks PASS**, status
`PASS_SYNTHETIC_CPU_HANDOFF_ONLY`:

- CPU command hides CUDA and retains accepted-runner session ownership.
- GPU command construction remains separate; neither path spawns a real process.
- CPU child failure propagates, and the shared resource/budget guard executes
  before any CPU spawn.
- R/E, config and script gates pass for correct inputs and reject wrong E/config/
  script identities and inconsistent E export version.
- Evaluation-only recovery reuses verified training without invoking training.
- Existing complete full test is reused; E receives the exact R/E and original
  test/training paths.
- Outer status consumes the offline PASS/E identity.
- Incomplete training is rejected for evaluation-only recovery.

These are interface/control-flow fixtures. Their synthetic metadata and mock
offline PASS are not A1 results. Actual prediction/metric/bootstrap PASS still
requires the full future server execution and independent raw-data review.

The earlier real runner timeout/cleanup fixture was also directly reviewed via
the local controller bridge: job `letter-a1-q06-20261113-runner-timeout-r0`,
TIMED_OUT/-9, identity-scoped cleanup verified, no live task members. Its full
receipt/source identity and capability limits are preserved in
`runner_static_review.md`. No redundant server/GPU fixture was run here.

## Reviewed byte identities and final binding gate

Paths are relative to `lc_rec/a1_qwen3_06b/`. The first three hashes match the
exact source bytes loaded and tested by the 13-check fixture; remaining core
files were read/bound independently. Any core change requires a corresponding
review/binding update, not reuse of a stale PASS.

| Path | Bytes | SHA-256 |
| --- | ---: | --- |
| `scripts/control.py` | 12351 | `d31bd030314986e4faa8e71bd8dc7fa9642afda2a6342c144d45d7fe1519f6b2` |
| `scripts/pipeline.py` | 17009 | `721c72f5f48ba03d72ac474fa07f8e0934dcd89cc5d4a0c8739c98f9022da225` |
| `scripts/offline_pipeline.py` | 5195 | `ed05ab904f708e22bb9c54c8ce81e8e8e19c89fb839de4c66d9f30ffaabf94ea` |
| `scripts/freeze_manifest.py` | 2977 | `3ee7214c5d025bc08b80d7cde8376ade93e44b8d9e41b828fcb37d2950df016e` |
| `scripts/paired.py` | 3790 | `99d06cc3f78e74727b18e10da248c519d9af5e82c98e5ad7f23345bf9369a4d0` |
| `scripts/runner.py` | 24826 | `1fefc09a15c1cb73878417f9eb50d0fe9e8fa1b2f60cc70b27f7cc4ed9cb0947` |
| `scripts/ledger.py` | 1684 | `51e643630008d7b4a724a23def9722fae0bad471cf30c967e9524d20ec7d954e` |
| `review/independent_extended.py` | 24920 | `ee193edd584a987093a254bb6b2886e6ba3b09fda8832b5bf6421a1b798ec20f` |
| `review/paired_verify.py` | 17013 | `bb9bbcce7426e8f673891208fced71459aabae05b7572aef288c215611f366e7` |
| `review/cpu_handoff_fixture.py` | 10102 | `7ff2c005bdf176627ad7f1aa1e05d10a3661a545ed6e438388daf02607100269` |

The unchanged original statistics verifier is
`lc_rec/letter/review/final_audit_20261113/statistics_verify.py`, SHA-256
`c17e47746751146eb4c50245e88616811981031dbd64501eb205e1615774ab96`.
The fixed original Stage-A verifier remains separately source-bound by its
accepted input/deployment map.

**Necessary source/CPU handoff gates: PASS.** Required before actual formal
launch: publish/freeze R and E, verify their actual deployment/core maps against
these hashes, bind both in the final prelaunch receipt/request, and combine this
pipeline review with the separate actual training/recovery/inference and budget
reviews. The main controller will append the concrete E/R evidence; this report
does not invent it. Formal execution and final A1 scientific audit remain pending.

GPT_UPLOAD_REQUIRED: NO for this sub-review; include this source/report, actual
deployment/request receipts and eventual runtime evidence in the fixed A1
GitHub delivery and lightweight review ZIP.
