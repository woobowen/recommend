# A1 one-time latest status check — 2026-11-14 UTC

**A1 STATUS: COMPLETE_PENDING_AUDIT.** Existing server computation has completed successfully. Final nonauthor scientific review has not been performed in this status-check task; **FINAL AUDIT PASS is not asserted**.

Initial actual observation: `2026-11-14T03:40:33.569108+00:00`. Bounded artifact cross-check: `2026-11-14T03:42:24.660385+00:00` → `2026-11-14T03:42:25.029179+00:00`. These UTC times correspond to the evening of2026-11-13 in America/New_York. Historical step2330 is not reused as current state.

- Job: `letter-a1-q06-20261113-normal-seed42-r0`.
- GPU RUN: `0e3e2c6f5c1a76152ea0228390d94ffc1ff75ba5`.
- CPU evaluation RUN: `445f1475cd1dd70d19881dbc1639bb311f7cccf7`.
- Runner SUCCEEDED/exit0; cleanup verified. No supervisor, worker, rank or owned workgroup remains alive. The stored pipeline phase label `offline_evaluation` identifies its last completed stage; pipeline SUCCESS and all three stage exit0 establish terminal computation, not an active CPU evaluation.
- Both ranks:8240/8240 updates,4.0/4 Epoch,100%. Last ten-update Train Loss1.507; optimizer LR0 at final step, pre-scheduler logged LR7.4167e-12, Grad Norm3.239900827. Reported full-training mean loss1.894351399. Final within-epoch20-interval speed4.843837s/update; no current throughput/ETA is needed after terminal success.
- Best=`checkpoint-8240`,eval_loss1.5451041460037231. Both existing rank events show `strict_ds_restored` at8240 before `training_complete`; best adapter SHA `fe96ea74fc3c7253b0c3687239266b0f073e1a51ef9158e7d97bf30b4527bd03` matches both test rank reports. This check did not load or rehash the635MB adapter.
- Actual full test:24,772/24,772 users,12,386/rank,20 candidates/user,495,440 total; failed0,skipped0,each row exactly once. Test metadata and frozen prediction manifest identities are supplied; raw prediction files remain on server and were not downloaded/rehashed.

## All completed validations and checkpoints

| Epoch | Step | Validation Loss | Status |
|---|---:|---:|---|
| 1 | 2060 | 1.965644598007 | COMPLETE |
| 2 | 4120 | 1.674856305122 | COMPLETE |
| 3 | 6180 | 1.570463418961 | COMPLETE |
| 4 | 8240 | 1.545104146004 | COMPLETE |

Each pass covers49,544 records. Existing current CP6180 andCP8240 each have23 members and PASS_SEAL_CONTRACT: required roles,all member sizes,all18 pre-sealed small-file hashes match. Large tensor files were not rehashed or loaded. Four saved-checkpoint events2060/4120/6180/8240 are preserved; retaining only the latest two matches limit2. Latest complete recovery point isCP8240. One original formal ledger attempt,`resume:null`; no interruption/retry or rotation anomaly observed.

## Existing formal results

Original evaluator four metrics (unmodified existing file): HR@5=0.087195220410,HR@10=0.106935249475,NDCG@5=0.074377046646,NDCG@10=0.080685962145. Tiny original-vs-offline NDCG accumulation differences are preserved in the files, not rewritten.

The following16 point estimates, user CIs and paired differences are read from existing frozen server outputs. Bootstrap5000,PCG64,seed20261113,95%percentile. A0/A1 ordered user axis=`1daf63ce4c77b0265777a46b61939006aafab0276b9106c742de602c8c7e867a`. These are fixed-trained-model user uncertainties, not training-seed variance or isolated evidence of the SID effect.

| Metric | A1 | A1 95% CI | A0 | A1−A0 | Paired difference95% CI |
|---|---:|---|---:|---:|---|
| hit@1 | 0.0613595995 | [0.0582906104, 0.0643064751] | 0.0645083158 | -0.0031487163 | [-0.0045626110, -0.0017358308] |
| hit@5 | 0.0871952204 | [0.0836014452, 0.0907476183] | 0.0903035685 | -0.0031083481 | [-0.0053689650, -0.0009274584] |
| hit@10 | 0.1069352495 | [0.1029388019, 0.1107308655] | 0.1133537865 | -0.0064185371 | [-0.0090020991, -0.0037138705] |
| hit@20 | 0.1309139351 | [0.1265945422, 0.1350718553] | 0.1385838850 | -0.0076699499 | [-0.0105360891, -0.0046827063] |
| ndcg@5 | 0.0743770466 | [0.0711628297, 0.0774634350] | 0.0776631039 | -0.0032860573 | [-0.0047701032, -0.0018243834] |
| ndcg@10 | 0.0806859621 | [0.0774494082, 0.0837856429] | 0.0850811320 | -0.0043951699 | [-0.0058016747, -0.0029724568] |
| ndcg@20 | 0.0867903619 | [0.0835363476, 0.0899301312] | 0.0914655491 | -0.0046751872 | [-0.0060746771, -0.0032987887] |
| mrr@5 | 0.0701800420 | [0.0670352111, 0.0732177458] | 0.0735117606 | -0.0033317186 | [-0.0046901071, -0.0019800413] |
| mrr@10 | 0.0727408249 | [0.0695809293, 0.0757726141] | 0.0765512906 | -0.0038104657 | [-0.0051208117, -0.0024955026] |
| mrr@20 | 0.0744371969 | [0.0712867034, 0.0774534864] | 0.0783053976 | -0.0038682007 | [-0.0051527405, -0.0025861660] |
| ItemHit@5 | 0.0867108025 | [0.0831180365, 0.0902632004] | 0.0897384143 | -0.0030276118 | [-0.0052478605, -0.0008477313] |
| ItemHit@10 | 0.1064373755 | [0.1025082082, 0.1102319823] | 0.1130308413 | -0.0065934657 | [-0.0091501157, -0.0038887992] |
| ItemHit@20 | 0.1306717261 | [0.1263927014, 0.1348296464] | 0.1381398353 | -0.0074681092 | [-0.0103342483, -0.0044808655] |
| ItemNDCG@5 | 0.0735519714 | [0.0703657769, 0.0766316468] | 0.0768303284 | -0.0032783570 | [-0.0047563008, -0.0018049748] |
| ItemNDCG@10 | 0.0798634206 | [0.0766544162, 0.0829230324] | 0.0843175088 | -0.0044540882 | [-0.0058420424, -0.0030212507] |
| ItemNDCG@20 | 0.0860270396 | [0.0827877051, 0.0891409467] | 0.0906698969 | -0.0046428573 | [-0.0060286482, -0.0032647314] |

### SID diagnostics

| K | Valid/candidates | Valid rate | Invalid/users | Duplicate/users | Collision candidates/users | Unique valid SID coverage |
|---|---|---:|---|---|---|---|
| 5 | 123848/123860 | 99.9903116422% | 12/12 | 0/0 | 2245/2051 | 1817/9897 (18.35909872%) |
| 10 | 247687/247720 | 99.9866785080% | 33/33 | 0/0 | 3284/2912 | 2533/9897 (25.59361423%) |
| 20 | 495201/495440 | 99.9517600517% | 239/205 | 0/0 | 3745/3177 | 3375/9897 (34.10124280%) |

Coverage denominator9897; catalog9922 items/23 collision groups/48 affected items;122 target-collision users. First target rank1–20 and miss counts are in[status.json](status.json) and the unchanged primary metrics file. No filtering/deduplication/disambiguation was added.

### Existing fixed groups

| Group axis | Group | Users | HR@10 | NDCG@10 | ItemHit@10 | ItemNDCG@10 |
|---|---|---:|---:|---:|---:|---:|
| history_group | 0-5 | 12556 | 0.1234469576 | 0.0947315720 | 0.1231814803 | 0.0937747924 |
| history_group | 6-10 | 8652 | 0.1005547850 | 0.0748234227 | 0.1000924642 | 0.0741178355 |
| history_group | 11-20 | 3564 | 0.0642536476 | 0.0454351243 | 0.0628507295 | 0.0448015850 |
| popularity_group | UNSEEN | 35 | 0.0000000000 | 0.0000000000 | 0.0000000000 | 0.0000000000 |
| popularity_group | LONG_TAIL | 3990 | 0.0027568922 | 0.0013899105 | 0.0026733500 | 0.0012722128 |
| popularity_group | MID | 4132 | 0.0038722168 | 0.0020883198 | 0.0036302033 | 0.0019737022 |
| popularity_group | POPULAR | 16615 | 0.1578092085 | 0.1194449578 | 0.1571471562 | 0.1182753650 |

History is`items[:-1][-20:]`; popularity uses original training-visible interactions`items[:-2]`,fixed q1=5/q2=11. Full16 metrics per group remain in[groups.json](existing_results/offline_evaluation/primary/groups.json). Overall CIs are not subgroup CIs; UNSEEN has35 users.

## Existing offline program receipts and final-audit boundary

Original pipeline automatically ran existing independent_original,primary,independent_extended,independent_statistics,paired,independent_paired stages; all exit0,overall existing offline pipeline PASS. All16 main-vs-independent mean deltas0. Existing original/extended/statistics/paired verifier files each reportPASS. This check only reads those previous computations and their source/input identities; it neither reruns them nor converts their program status into final scientific nonauthor approval.

## Time, resources and budget

Runner start `2026-11-13T11:57:04.861057+00:00`,end `2026-11-14T02:50:46.514479+00:00`. Actual runner wall53621.653422s=14.894904h,within96h. Ledger preserved unchanged at53621.398429s; the0.255s difference comes from runner cleanup/start coverage. The read-only conservative max includes runner cleanup; no ledger reconciliation write was performed. Remaining execution time0; final scientific review/delivery remains outside this one-time check.

Train stage including Validation/checkpoint/best restore49,039.915303s; full test4,506.200103s; offline stage73.182916s. Generation P50=0.352987s/P95=0.357067s. User time sums are not end-to-end wall time.

Current GPUs are both1MiB/0%,28℃/27℃; compute process list empty after verified cleanup. UUIDs:GPU-dfff0e95-1db6-e2f4-3431-55bb852fa0ae andGPU-59cac3bd-3645-668b-79c8-839e0d878893. Current available RAM256,560,930,816bytes and disk2,947,599,441,920bytes. Runtime minimum available RAM250,379,694,080bytes; swap constant152,190,976bytes. No scanned NaN/Inf/OOM/Traceback/protection error. The longest1833.6s log silence corresponds to the completed1836.9s first Validation and is below7200s watchdog. No new log progression is expected from an ended job.

## Evidence and scope

- [Machine status](status.json),[current control observation](control_observation.raw.json),[runner query/cleanup observation](runner_query.raw.json).
- [Exact original persisted runner state](runner_status.persisted.raw.json),[cleanup](cleanup.raw.json),[pipeline](pipeline.raw.json),[ledger](formal_ledger.raw.json),[best selection](best_selection.raw.json).
- [Rank0 original event lines](rank0_events.excerpt.jsonl),[rank1](rank1_events.excerpt.jsonl),[training reports](rank_training.excerpt.json),[test reports](rank_test.excerpt.json),[checkpoint seal check](checkpoint_integrity.json).
- [Existing original four metrics](existing_results/test/metrics.json),[primary16/SID](existing_results/offline_evaluation/primary/metrics.json),[bootstrap](existing_results/offline_evaluation/primary/bootstrap.json),[paired result](existing_results/offline_evaluation/paired/paired_comparison.json).
- [Previously executed original verifier](existing_results/offline_evaluation/independent_original.json),[extended verifier](existing_results/offline_evaluation/independent_extended.json),[statistics verifier](existing_results/offline_evaluation/independent_statistics.json),[paired verifier](existing_results/offline_evaluation/independent_paired.json).
- [Resource snapshot](resource_snapshot.json),[source-file identities](SOURCE_FILES.json),[bounded raw response](server_crosscheck.raw.json),[query context](context.json),[member SHA list](MEMBER_SHA256.json).

The18 existing JSON files above were copied byte-for-byte and retain original SHA identities. Other excerpts are explicitly parsed views or unchanged event lines, with original path/read-range/observed-byte hash in the raw response. No model/adapter/checkpoint tensors,raw predictions,large CSV/NPZ arrays or credentials are published. Only this new timestamped directory is committed; prior evidence and untracked work remain protected. This task performed no experiment change,stop/restart/recovery,new GPU/CPU evaluation,repair,seed/domain expansion,package install or ZIP creation. Original final review entry outside this snapshot is not rewritten by this check.
