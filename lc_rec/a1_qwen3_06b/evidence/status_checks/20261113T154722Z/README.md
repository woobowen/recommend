# A1 one-time status check — 2026-11-13

**A1 STATUS: RUNNING_HEALTHY.** Initial observation `2026-11-13T15:47:22.143088+00:00`; cross-check completed `2026-11-13T15:50:45.786953+00:00`. This is a non-atomic snapshot of a live job, not final scientific approval.

- Job: `letter-a1-q06-20261113-normal-seed42-r0`.
- GPU RUN: `0e3e2c6f5c1a76152ea0228390d94ffc1ff75ba5`; CPU evaluation RUN: `445f1475cd1dd70d19881dbc1639bb311f7cccf7`.
- Both ranks progressed 2290→2330/8240; latest epoch1.131067961/4, total28.2767%. Last Train Loss1.7883, optimizer LR0.000164832714196, Grad Norm5.286310196, all finite.
- Actual latest20 within-epoch update intervals, steps2130–2330: 4.843513s/update. Validation/checkpoint pause excluded explicitly.
- Epoch1/step2060 Validation: all49,544 records, loss1.9656445980072021; completed15:28:30 UTC, duration1836.886s. Current best=`checkpoint-2060`.
- CP2060 complete seal:23 members,4,417,952,648 logical bytes; all member sizes and18 sealed small-file hashes match. Large payload contents were neither rehashed nor loaded. No new save/restore test performed.
- Original supervisor/worker/rank process PID/start/UID/boot identities match. One formal ledger attempt with no resume; no interruption/recovery or rotation anomaly observed. Only one epoch/checkpoint has completed.
- Full Beam20 and original/extended/CCE/Bootstrap metrics have not started or been produced. Existing server pipeline continues independently.
- GPU0/1:16,594/17,302 MiB,40/38℃; utilization14%/8% at the instant sampled. Resource-window minimum available RAM251,177,275,392 bytes; current disk free2,953,046,949,888 bytes. Swap stable152,190,976 bytes. No scanned NaN/Inf/OOM/traceback/protection error. The historical30.6min log gap corresponds to real Validation, below the2h watchdog.
- Elapsed wall3.895h. Remaining train+3 Validation+checkpoint linear proxy9.488h; full pipeline planning11–13h conditional on current speed, first-validation cost and historical small-sample inference estimate. Not a promised finish time.

## Evidence

[Machine summary](status.json), [exact control observation](control_observation.raw.json), [runner query response](runner_observe.raw.json), [bounded server response](server_crosscheck.raw.json), [process identities](process_identity.excerpt.json), [checkpoint checks](checkpoint_integrity.json), [rank0 events](rank0_events.excerpt.jsonl), [rank1 events](rank1_events.excerpt.jsonl), [pipeline](pipeline.excerpt.json), [resources](resources.excerpt.json), [worklog tail](worklog.tail.txt), [scope and source identities](context.json), [member hashes](MEMBER_SHA256.json).

The server response records original paths, observed byte hashes/read ranges and non-atomic timestamps. Excerpts are parsed views or unchanged event lines, not claims that normalized JSON is a byte-for-byte copy of the original server file. Read-only collector source and local summary derivation are included for review. No model/adapter/checkpoint payload, full prediction file, credential or unrelated file is published.

The earlier15:37 connection timeout remains preserved in commit`e394357e9166311f2e9e6471f934d17e7e8f3924`; this successful check reached the same original process, without any resubmit or recovery. The two observation timestamps/steps are expected for an advancing job; no identity conflict was found.

This request performed only a bounded observation plus cross-check and GitHub evidence publication. No experiment stop, pause, restart, repair, extra inference/evaluation, source/config/checkpoint edit, package install or branch synchronization occurred. Existing untracked files remain outside this commit. No ZIP was created. Required final scientific audit remains pending.
