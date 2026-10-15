# GRID formal stage receipt — 2026-10-15

## A. Overall status

**PARTIAL — authorized exception repair undergoing CPU regression.** No new formal evaluation, training or attention job has started. The original three correction rounds remain exhausted; the user explicitly approved one strictly scoped additional round, and its one-line serialization fix has now been applied. This receipt records completed preparation and an actual failed CPU entry check, rather than completed formal experiments.

## B. Work completed and inherited

The stage resumed `wbw` at `93567ea7f61e970c448838443595242d0b9beef0`, with no unpublished commits or tracked modifications. Existing untracked Python caches were retained. The server source directory was confirmed to be an export without `.git`; no related old worker was active. Existing A/B preflight, Beauty MLP formal best2600/full test/fixed32 attention and original failure history were inherited without rerunning them. The previous temporary cleanup and the user's verified preflight ZIP remain valid.

The new formal orchestration and approved domain specifications were committed and pushed as `ab056ffbb16f2303fe048c668ad275368ccc7237`. Its immutable server deployment is `/home/bwwu/grid-audits/sid-extensions-p0/grid-formal-20261015/snapshots/73074010b1acf951`, full snapshot identity `73074010b1acf9510e2acf7ecfcdf9ba0931a22c1b176c144d3c4c6c17ce5db1`. All 207 deployment files, including 171 Python files, were verified. Model, Shuffle sampler, beam search and evaluator source remained unchanged from the approved preflight.

Toys and Sports research settings and their separate 12-hour total allowances, including full testing, are explicitly approved by the user. They do not need renewed research approval. Toys uses the paired author RQ-VAE+STE bundle, scratch/all parameters, raw32/acc48/val4800. Sports uses its own Qwen/R-Kmeans SID plus the author's per-item MLP, scratch/all parameters, raw16/acc16/val1600, explicitly a new structure experiment rather than a recovered historical Sports MLP.

## C. Results by line

| Task | Actual action and result | Evidence | Status / remaining dependency |
|---|---|---|---|
| Beauty Shuffle A | Three exact plans, asset bindings and independent admission reviews prepared. CPU identity entry failed before model construction. | [Approval records](approval/), [failed job](runs/gf-20261015-a-identity-v1/), [diagnosis](a-identity-v1-failure-diagnosis.json) | Formal Normal/Shuffle3/Shuffle4 NOT_RUN; requires authorized serialization repair and CPU recheck. |
| Beauty Shuffle B | Three exact plans, asset bindings and independent admission reviews prepared; old shared-table lifecycle PASS inherited. | [Six-condition review index](beauty-six-gate-review-index.json), [specifications](../../experiments/formal_20261015/specs/) | Formal Normal/Shuffle3/Shuffle4 NOT_RUN; same new wrapper dependency. |
| Toys MLP | Approved original bundle/protocol, saved training driver and nonauthor source review. Current asset/complete split identity and cost scenarios checked. | [Source admission](source-formal-admission.json), [resource review](source-resource-review.json) | Research conditions CONFIRMED; full multiset metadata index and formal training NOT_RUN. |
| Sports MLP | Approved new structure experiment/protocol, same independently reviewed training driver; own assets and full split identity checked. | [Source admission](source-formal-admission.json), [resource review](source-resource-review.json) | Research conditions CONFIRMED; full multiset metadata index and formal training NOT_RUN. |
| Beauty baseline attention | Main baseline is B Qwen/R-Kmeans/shared table; existing fixed32 cohort retained. | [Attention protocol](../../docs/sid-extensions/experiments/attention.md), [formal wrapper](../../experiments/formal_20261015/run.py) | New formal figure NOT_RUN; shared wrapper dependency. |
| Beauty Shuffle attention | Separate capture path prepared, requiring corresponding completed formal evaluation identity; no preflight figure relabeled formal. | [Formal wrapper](../../experiments/formal_20261015/run.py) | A/B formal fixed32 NOT_RUN; evaluation prerequisites unmet. |
| MLP attention | Existing completed Beauty MLP fixed32 inherited without recollection. | [Existing formal result](../beauty-extensions/runs/beauty-mlp-formal-attention32-v1/RESULT.json) | Toys/Sports NOT_RUN because formal training has not started. |

No new Recall/NDCG result, best checkpoint or formal attention matrix is claimed. No test-set configuration search occurred.

## D. Actual new job and failure

| Job | Phase | Input commit / snapshot | Status | Exit | Fetch |
|---|---|---|---|---:|---|
| `gf-20261015-a-identity-v1` | CPU identity check, not formal evaluation | `ab056ffbb16f2303fe048c668ad275368ccc7237` / `73074010b1acf951` | FAILED | 1 | All nine export members verified and included [here](runs/gf-20261015-a-identity-v1/). |

Request fingerprint `8cf2fe1e425c839b6210ca836a21c2bbb0515636714562359f12155d156208a5` matched acceptance. Runner wall time was 8.784242 seconds, `CUDA_VISIBLE_DEVICES` was empty, cleanup was verified and no worker remained. Only invocation output was written. Source, manifest and request identities closed correctly: this failure was neither the old deployment-manifest conflict nor a CPU/CUDA model fault.

The new wrapper attempted `OmegaConf.save(..., resolve=True)` on the entire source configuration, including an unused Hydra `${now:...}` run-id field. Without a Hydra runtime this raised `UnsupportedInterpolationType: now` before the mode-specific model or index path. The concrete proposed repair changes only this whole-config save to `resolve=False`, matching the existing successful common/MLP entries; actual computation subtrees continue to be resolved through the original code. See the [exact unapplied patch](proposed-config-serialization-fix.patch) and [independent review](config-serialization-review.json). Original FAILED records are preserved.

## E. Resource and review evidence

Toys evaluation/testing each have 19,412 rows in 152 files; Sports each have 35,598 rows in 279 files. The 862 files match the already completed full CRC/schema/hash evidence. This stage refreshed directory names/sizes/readability and small SID/config hashes, without rescanning 2.3 GB of TFRecords or loading models. Full target multisets still require one saved CPU metadata job per domain.

Using the earlier Beauty formal run only as a 3,600-update/36-full-validation planning scenario, the conservative scenarios are about 7.381 hours for Toys and 9.092 hours for Sports, including one complete test, a stated 25% engineering allowance and startup time. These are assumptions, not statistical bounds or new stopping steps. Early stopping is not predictable; original max_steps320000 remains unchanged. No evidence currently shows that this reference scenario clearly exceeds 12 hours. The total hard limit will not be extended; exhaustion must remain INCOMPLETE. No new GPU job was started and no package/environment configuration was installed or changed.

Real review roles and limits:

- `/root/source_conditions_review`: approved domain source/configuration, user protocol and training-driver review; current asset/split and cost evidence. Did not sign missing runtime indexes or formal results. [Source](source-formal-admission.json), [resources](source-resource-review.json).
- `/root/implementation_review`: reviewed root-authored wrapper/deployment and the exact configuration serialization failure/proposal. Its own training-driver implementation was independently reviewed by the source reviewer. The proposal has static approval only, no runtime PASS. [Initial scope](implementation-formal-admission.json), [failure/proposal](config-serialization-review.json).
- `/root/results_publication_review`: six exact admission plans/source maps/assets/indexes and actual failed request/manifest/output identity. Did not sign unrun formal evaluation or attention. [Admission index](beauty-six-gate-review-index.json), [actual failure](identity-v1-failure-review.json).

## F. Only current blocking decision

The inherited preflight used all three concentrated correction rounds. The user subsequently explicitly approved one additional focused repair: apply the linked one-line configuration serialization patch, regenerate exact source/gate bindings and verify CPU-only save/reload/actual-interpolation/effective-configuration consistency followed by a new identity attempt. No request is made to reconsider the already approved Toys/Sports research conditions or 12-hour budgets. The patch has been applied only after this explicit exception approval. The original three rounds are not reset; no further exception is assumed.

The independent work above is preserved. Runtime work shares the new wrapper dependency, so submitting it through this known failing entry would waste jobs. After authorization, the old failed job must remain FAILED, and the new attempt must have a fresh immutable source/request identity. Existing successful preflights must not be rerun.

## G. Publication and delivery

All necessary evidence for this partial receipt, including the original failed export, is included in this publication on `wbw`. The actual delivery commit and remote readback are recorded outside Git in `/home/addaswsw/recommend-support/grid-formal-20261015/FINAL_PUBLICATION_RECEIPT.json`, avoiding a self-referential commit. The initial execution commit remains `ab056ffbb16f2303fe048c668ad275368ccc7237`; publication of additional evidence does not change the immutable failed deployment.

`GPT_UPLOAD_REQUIRED: NO` for this partial receipt. The already verified prior preflight ZIP remains at `/home/addaswsw/recommend-support/grid-shuffle-mlp-preflight-20261015/deliveries/grid-shuffle-mlp-preflight-20261015.zip`, SHA-256 `956b7488296f14fb5cb446f46ca482949b8296ba9f346a6dc444328b7f17c8e3`; it is not rebuilt or replaced. No cleanup, Parallel, additional method, baseline retraining or other experiment was performed.
