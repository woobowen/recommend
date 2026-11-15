# A0 Beauty inference preflight and formal readiness

`INFERENCE_TECHNICAL_PREFLIGHT: PASS`
`FORMAL_STATIC_READINESS: FAIL`
`INDEPENDENT_EXECUTABLE_CHECKS: FAIL` (full formal chain remains unverified;
the inference-specific independent replay passed)
`NONAUTHOR_REVIEW: PENDING`
`WEB_GPT_REVIEW: PENDING`
`FORMAL_READY_CANDIDATE: NO`
`FORMAL_LAUNCH_AUTHORIZED: NO`

## Real inference preflight

The user-authorized inference-only RUN was
`63f13def364d0852ad8bc0a20f10e41fd434446c`; the previously trained Adapter
remains bound to training RUN `f54a2b505fd2d8767109a11d330a7064aff8af5e`.
The server source export at
`/home/bwwu/grid-audits/sid-extensions-p0/transfers/beauty-infer-63f13def364d/release`
had 300 files verified against its manifest (SHA-256
`7f3d9a4515d7e538c9408cd5fc718272b505f63b3bdc9da220ee56b7842351fd`).
Training and recovery were not rerun. Their prior evidence remains at
`PREFLIGHT_FINAL_REVIEW_ENTRY.md` and `evidence/e2e-20261116-r0/`.

Job `letter-a0-beauty-infer-20261116-r0` exited 0, with runner-verified cleanup.
The separate Normal and Shuffle stages each strictly loaded BF16 LLaMA-7B and
verified all 450 expected saved/active PEFT keys, shapes and values after
explicit dtype conversion, including embedding/head. Adapter SHA-256:
`e56abd90bd3cc55d3786df04971a670bc12c54578110d255b5e7f3f4ec76af07`.
Its fixed 32-user manifest SHA-256 is
`714450a0e9479ed272266a5b4d3c2150d83742e4b70c38f80da888d4fa0aaa39`.

Normal produced 16 users per rank and 20 candidates per user, 640 total.
In Shuffle acceptance each rank processed those same 16 users with the
original, OFF, S3 and S4 calls. Runtime checks recorded exact same-process OFF
sequences/scores, input-only permutations, embedding lookup equivalence,
unchanged parameter versions, native mask and positions, empty initial cache,
Beam20, legal positional callback and EOS. The external saved Normal comparison
uses the frozen `atol=1e-5, rtol=1e-5` score criterion and exact sequence/order.

`verify_inference_independent.py` independently reconstructs protocol
permutations from the hashed author data and recorded token IDs, checks all 32
users, 128 conditions, prefill IDs, recorded positions/cache, candidates,
score order and Normal/OFF outputs. Its v2 output is PASS, with maximum
cross-process Normal/OFF score absolute difference 0. This program reads saved
records; it cannot independently observe the in-memory embedding or weight
hooks, which remain runtime checks. Both programs were executed by the same
main agent; this is an independent implementation, not a nonauthor review.
See `evidence/inference-model-r0/` for original candidate and acceptance files,
runner receipts, source manifest and verifier outputs. The 32 users are a
technical interface sample, not a Beauty performance estimate.

## Resources and historical state

The original five GPU preflight jobs conservatively charged
3316.546833369881 GPU-device-seconds. This inference job charged
635.9556972142309 under the same dual-device wall upper-bound convention.
Total is 3952.502530584112, below the newly approved project ceiling 7200.
The new 3600 allowance has 2964.044302785769 device-seconds remaining; the
project ceiling has 3247.497469415888 remaining. No budget is transferred to
formal training. The earlier training runner remains FAILED/1 because its
wall guard expired in CPU comparison; its three training stages and subsequent
CPU recovery comparisons are separately recorded and were not rewritten.

## Formal static review and blockers

`formal_static_audit.py` and `evidence/formal-static-r0/` capture code identity
and static checks. The training entry sets four epochs, 8214 optimizer updates,
GAS 4, microbatch 8, and limits the 32-row Validation subset and ten-step
schedule to `boundary_preflight`. The derived arithmetic has epoch updates
2053/4107/6160/8214 and pending microbatches 2/0/2/0. The Normal entry checks
11182/11181 rank assignments, 22363 users and Beam20; a preflight manifest is
forbidden in formal mode. Beauty `offline_metrics.py` defines 16 metrics,
12088 SID coverage denominator, dynamically derived Beauty popularity groups
and 5000 PCG64 bootstrap draws; synthetic ranking/CCE tests pass.

A real formal-chain blocker remains. The historical
`lc_rec/shuffle/evaluate_condition.py` and its controller are Instruments
specific (12386 users/rank, Instruments paths and A1 evaluation). They cannot
be used as a Beauty full-population evaluation/series runner. The historical
`paired_and_changes.py` also hardcodes 24772 and the Instruments axis. This
turn adds `paired_beauty.py`, a minimal Beauty-axis adaptation with a small
synthetic paired bootstrap regression, but no full-population Beauty run or
end-to-end independent evaluation has verified that adaptation. There is no
verified Beauty formal training→Normal→Shuffle3→Shuffle4 orchestrator with
fixed inputs, budget, stage dependencies and cleanup. The shared runner's
request validation caps one job at 86400 seconds while the draft formal config
contains a 345600-second limit; a compatible, reviewed scheduling/budget plan
is still needed. No formal GPU work was submitted.

The available nonauthor agents previously reported they could not directly
read files or execute tools; no nonauthor PASS is claimed. Web GPT review is
pending. Formal launch requires completion and review of the missing Beauty
controller/evaluation chain, a separately confirmed formal resource budget,
and the user's explicit subsequent authorization. No new system packages,
language packages, toolchains or persistent configuration were installed.
