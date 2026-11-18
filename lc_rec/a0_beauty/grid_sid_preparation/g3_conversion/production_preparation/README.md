# Fixed G3 Beauty production preparation

## 最终收口与网页增量审核

本轮采用用户批准的单卡上限。训练 30 小时，Normal 含 OFF 对照 12 小时，
S3/S4 各 6 小时，CPU 合计 5,400 秒，流水线累计 199,800 秒（55.5 小时）。
阶段请求仍从剩余阶段、组、总量的最小值扣 120 秒清收预留。
[批准依据](single_gpu/final_closure/budget_approval.json)、
[有效预算](single_gpu/final_closure/approved_budget.json)与
[上一轮技术二审及来源](single_gpu/final_closure/external_technical_review.json)
分别记录。下文关于旧草案及当时待批准预算的文字为历史背景。

[最终计划](single_gpu/final_closure/final_plan.json)、
[计划身份与验收摘要](single_gpu/final_closure/final_plan_summary.json)、
[真实 CPU dry-run](single_gpu/final_closure/jobs/g3-final-plan-cpu-20261118-r2/output/dry_run.json)、
[非作者直接审查](single_gpu/final_closure/non_author_review.json)
供当前网页增量复核。既有网页技术二审对应交付提交
`6db8ab4ec98f5692014eeac02b03f3f55015d1eb` 与运行提交
`21d495b0be5b606853ec7cec68a3eaa1b58acbd7`。当前增量仍待网页复核。
正式训练、生成及控制器尚未启动。

收口复核脚本固定于 `780040a4a3743d124004811495f9c45e6a3273ef`。
[首次 CPU 作业](single_gpu/final_closure/jobs/g3-final-plan-cpu-20261118-r0/work.log)
于成功生成计划后遇到检查脚本的 CPU 阶段加总错误。第一轮修正按批准值
5,400 秒重新校验。直接训练 CLI 的拒绝路径会写诊断，首轮复核在正式根目录
留下了一个 2,428 字节的拒绝回执。第二轮修正先
[按原字节归档该诊断](single_gpu/final_closure/jobs/g3-final-plan-cpu-20261118-r1/diagnostic_relocation.json)，
移除归档后空目录，再将 CLI 否定测试输出指向隔离目录。最终复核比较同一计划
的前后字节，检查正式根目录、注册绑定、进程和 job。失败重试扣费通过使用
模拟 runner 状态及停止于请求创建前的隔离函数调用验证。

正式根目录为 `/data/bwwu/lc-rec/g3-formal-20261118/21d495b0be5b`，
计划位于其中的 `plan.json`，旧 `preparation-plan.json` 独立保留。
最终计划原始 SHA-256 为
`a56076cf07d67bc088442dc35f863134081b455f551fc45d2c6bfb2bfd246407`，
生产 canonical digest 为
`526987fa87f0feb565a7a9f42bc58f42e3121b9d9a13cea4613e47f842c394b6`。
计划固定使用源导出
`/home/bwwu/grid-audits/sid-extensions-p0/g3-production-20261118/21d495b0be5b/release`、
解释器 `/data/bwwu/envs/letter-20261110/bin/python` 与 Beauty 隔离 runner
`/data/bwwu/lc-rec/a0-beauty-20261115/production-closure-20261115/formal-candidate-5b2917d/runner-beauty.py`。
物理卡 1 对应逻辑 CUDA 0，UUID
`GPU-59cac3bd-3645-668b-79c8-839e0d878893`。

正式流程为 train → verify_best_adapter → Normal/OFF → normal_eval →
S3 → s3_eval → S4 → s4_eval → paired → collect。训练从原模型初始化，
正式生成共享本实验的 `training/best_adapter` 并覆盖 22,363 用户。

网页增量复核完成后，唯一新增的用户决定是明确启动这份固定计划。正式授权回执须
如实绑定该用户指令，包含 canonical digest、运行提交、配置 SHA、预算、task
`g3-beauty-seed42`、本实验资产、单卡 profile、有效期，以及对新独立 G3
实验的明确授权和旧 series `beauty-a0-seed42-original`。正式启动入口要求用户
确认的授权文件 SHA。启动时刷新 GPU UUID/占用、磁盘和主存。服务器执行入口如下，
其中 `USER_RECEIPT` 和 `USER_PINNED_SHA` 由届时明确授权提供。

```bash
release=/home/bwwu/grid-audits/sid-extensions-p0/g3-production-20261118/21d495b0be5b/release
python=/data/bwwu/envs/letter-20261110/bin/python
plan=/data/bwwu/lc-rec/g3-formal-20261118/21d495b0be5b/plan.json
cd "$release/lc_rec/a0_beauty" &&
  PYTHONDONTWRITEBYTECODE=1 "$python" -c 'import sys; from production_control import authorize; from formal_control import read; authorize(read(sys.argv[1]), sys.argv[2], sys.argv[3])' "$plan" "$USER_RECEIPT" "$USER_PINNED_SHA" &&
  PYTHONDONTWRITEBYTECODE=1 "$python" production_control.py start --plan "$plan" --authorization "$USER_RECEIPT" --authorization-sha256 "$USER_PINNED_SHA"
```

正式开始后以 `controller.log`、`state.json`、`requests/` 和 runner 下的原
job 目录跟踪，在线每 15 分钟轻量观察。控制器的 `query --plan "$plan"` 仅在
已经出现 `plan.identity.json` 和 `state.json` 后使用。断线先核对原状态、
控制器进程及已登记 job 的请求 SHA，再用原 runner 的
`query <job_root>/<job_id>` 检查终态和清收。未知提交结果时查原 ID。
预算不足或阶段失败时保留原计划、attempt、日志和累计账本。后续符合条件的显式
`--retry <stage>` 沿用该 plan 和原账本，train 恢复仅使用该实验最新完整
checkpoint 并通过 `--resume` 明确指定。

## 上一轮已验收的固定实现与预跑

The user explicitly replaced the earlier two-GPU execution requirement with
**one GPU, batch 8, gradient accumulation 8, global batch 64**, including the
future formal run. Physical GPU1 is the authorized device and becomes logical
CUDA0. GPU0 belongs to an unrelated task and is neither used nor required to
become available. The previous dual-GPU requirement is
`NOT_APPLICABLE_THIS_RUN`; its historical receipts remain unchanged.

The current acceptance authority is [single_gpu/acceptance.json](single_gpu/acceptance.json),
now sealed by the main controller as **`READY_FOR_REVIEW`** after actual
runtime verification, complementary direct non-author reviews, fixed GitHub
content readback and audit-package prechecks. The [delivery verification](single_gpu/delivery_verification.json)
records the published content commit; the final sealed delivery commit and ZIP
are independently read back and hashed outside Git to avoid self-reference.
The older [acceptance.json](acceptance.json) records the earlier dual-GPU
preparation delivery and remains unchanged. `READY_FOR_REVIEW` means ready for
the pending webpage second review; it grants neither external GPT approval nor
formal launch permission. No formal G3 experiment has been started or authorized.

## Fixed assets and research conditions

The [converted index](../Beauty.g3.index.json) and
[complete bridge](../../identity_audit_abfe/bridge.csv) retain all 12,101 items.
The fixed G3 tensor has shape `[4,12101]`, int64: three quantization positions
and the **original fourth deduplication position**, with value-set sizes
256/256/256/18. Position identities remain `<a_N>`, `<b_N>`, `<c_N>`, `<d_N>`.
No fifth token, new deduplication, item-axis reorder or title matching is used.
The full four-token catalog has 12,101 unique SIDs and zero collision groups.

[`g3_identity.py`](../../../g3_identity.py) pins tensor, bridge, index,
interactions/items, original base revision and full base-model file hashes.
It verifies 786 new tokens, vocabulary 32,786 and the exact token→ID map,
including tokenizer files saved with checkpoints/adapters. An isolated G3 data
entry uses the original LETTER interaction/item bytes and the fixed G3 index;
old LETTER indices and result directories are protected. An old adapter, an
identically shaped foreign model or a same-size reordered token map cannot
satisfy these identity guards.

The [effective G3 configuration](../../../config/formal_g3_beauty_seed42.json)
retains LLaMA-7B revision `4782ad278652c7c71b72204d462d6d01eaaf7549`,
int8 training base, BF16, DeepSpeed stage 2 and the existing attention
compatibility. It retains seed 42/four epochs, learning rate 2e-4, weight decay
0.01, warmup 0.01/cosine, original split/prompts/history 20/max length 2048,
response-only labels, LoRA r8/alpha16/dropout0.05/seven target module classes,
and **full** embedding/head `modules_to_save`. Frozen `original_module` copies
remain frozen. The storage retention override remains `save_total_limit=2`;
the inherited template field is not mistaken for the effective override.
Validation loss still selects the best checkpoint before export. The original
beam 20, return count, EOS/length, position callback, normalization, stable
score ordering, duplicate-candidate/invalid-SID treatment and metrics remain.
The explicit execution-layout change is the authorized single-GPU profile.

## Actual production chain and execution layout

Production training/loading, prediction, aggregation, offline metrics,
paired analysis and the formal controller share G3 asset, source/config,
execution-profile and PREFLIGHT/FORMAL identities. Relevant entries are
[`formal_train.py`](../../../formal_train.py),
[`formal_evaluate.py`](../../../formal_evaluate.py),
[`production_identity.py`](../../../production_identity.py),
[`production_pipeline.py`](../../../production_pipeline.py),
[`production_control.py`](../../../production_control.py),
[`offline_metrics.py`](../../../offline_metrics.py) and
[`paired_beauty.py`](../../../paired_beauty.py). Runtime snapshot
`21d495b0be5b606853ec7cec68a3eaa1b58acbd7` is mapped file by file in
[single_gpu/run_deployment.json](single_gpu/run_deployment.json).
Server source is an export with verified bytes, not a claimed Git checkout.

[`single_gpu_training.py`](../../../single_gpu_training.py) serializes the
historical two logical sampler shards by alternating their eight-row blocks.
A complete epoch contains 16,428 eight-row microbatches, 131,424 sampled
occurrences and the original 11 repeated tail indices. The four full-epoch
optimizer boundaries are `[2053,4107,6160,8214]`; accumulation continues across
epoch boundaries. Actual CPU checks compare the entire loader stream with an
independent reconstruction and the first 160 sampled indices with the retained
historical epoch-row evidence. DeepSpeed scales each eight-example microbatch
loss by 8. The [actual installed ZeRO averaging source](single_gpu/installed_zero_gradient_averaging.json)
and recorded effective configs establish `reduce_scatter=True`, no explicit
prescale/predivide overrides, and the ordinary non-sequence-parallel branch
(sequence size 1, derived from the initialization call chain). Its single
rank divisor is 1; the historical two-rank divisor was 2. Thus the original
microbatch weight `(1/4) × (1/2)` and the authorized single-rank weight
`(1/8) × (1/1)` both equal `1/8`. This is a source/config derivation using
historical evidence, not a new dual-GPU measurement. This does **not** assert a global token-mean loss, identical
single/dual floating-point trajectories or identical per-row randomized prompt
realizations across the former two processes. Prompt construction and its
selection algorithm were not replaced.

The finite production boundary mode uses **20 microbatches per short epoch**,
four short epochs and ten optimizer updates with boundaries `[2,5,7,10]`.
Stopping after epoch 1 leaves four accumulated microbatches pending out of eight.
Native checkpoint loading checks the single-rank optimizer/master state,
external HF scheduler, counters, pending gradients, Python/NumPy/CPU/one CUDA
RNG state, loader generator/order and consumption position. Saved tokenizer,
adapter and selected checkpoint identities must agree; export tensors are
compared by key with the selected checkpoint. New processes perform strict
loading, without ignoring missing/unexpected keys. Continuous/resumed tensor
comparison retains `atol=1e-6, rtol=1e-3`; scalar counters and RNG/loader states
are checked exactly. These are acceptance requirements, not inferred successes
from the existence of a mode or checkpoint file.

## CPU and formal dry-run evidence

The final source snapshot's actual bounded CPU jobs are independently accepted
runner jobs with original argv/cwd/interpreter, immutable deployed inputs,
actual import paths/bytes/SHA, stdout, exit and process-group cleanup:

- [CPU r3-final result](single_gpu/jobs/g3-lcrec-single-cpu-20261118-r3-final/output/cpu/result.json):
  PASS, all 131,413 training examples and the real validation/test user axes,
  exact token mapping/save-reload, 1,148 collator label positions, ten actual
  four-position/EOS callback calls and 45 targeted negative cases. Checks also
  exercise the installed DataLoaderShard stream and actual DeepSpeed loss-scale
  method, single-rank checkpoint/profile guards and prediction provenance.
- [Formal dry-run r3-final](single_gpu/jobs/g3-lcrec-single-dryrun-20261118-r3-final/output/dry_run.json):
  PASS, valid input/source/config bindings and single-process stage argv;
  **zero formal jobs submitted**. Future checkpoint and prediction paths are
  dependency outputs, not fabricated current assets. The preparation state is
  `DRAFT_PREPARED_NOT_SEALED` with no attempts or submitted stages.
- Actual controller and direct-training negatives reject an internal review
  labeled PASS as a webpage review. A synthetic external-kind receipt with a
  corrupted evidence SHA is also rejected. These fixtures live in explicitly
  synthetic CPU-only directories, create no user launch authorization and do
  not represent a real external review.

Real split/history boundary inputs and synthetic cases are identified separately.
All 22,363 real visible histories contain no repeated item; repeated-SID input
fixtures therefore do not pretend to be real repeated-item users. CPU metric
cases retain the distinction between repeated target hits in original NDCG
and first-hit CCE. Zero catalog collisions neither makes every position-legal
combination a catalog SID nor makes NDCG/CCE identical for arbitrary duplicate
candidate lists. No candidate deduplication or formula change forces equality.

## Model preflight, generation and evaluation

[`g3_preflight_driver.py`](../../../g3_preflight_driver.py) submits isolated
production-based train/compare/generate/evaluate stages through the existing
runner. It checks the actual CPU receipt/import bytes, refreshes authorized
GPU1 UUID/free memory before each model subprocess, checks host memory and
space, and charges model loading/saving/generation to the same cumulative
240 GPU-device-minute budget. New attempts keep old evidence and frozen
running source; a fresh connection queries the accepted job rather than
blindly restarting it. Per-job requests and terminal status/cleanup carry the
actual attempt and parent relationships. The latest machine acceptance gives
verified usage and terminal state; job submission alone is not model PASS.

The same selected, strictly loaded G3 preflight adapter supplies Normal, OFF,
S3 and S4 for the original fixed 32 user IDs in a single rank/process. G3 target
and history fields are rebuilt from the fixed index. Actual prefill inputs,
masks and embedding-call evidence establish that OFF preserves input, S3 moves
only the first three history tokens and S4 moves all four original IDs. Item
order, target, non-history text and padding/masks remain fixed; the generation
callback still constrains output by its original positions. Exact same-process
Normal/OFF comparison is required. S3/S4 scores need not change to pass an
input-intervention check.

A second independent inference process strictly reloads that same adapter and
runs Normal on the same 32 users, including the same three-left-pad boundary.
[`g3_evaluate_preflight.py`](../../../g3_evaluate_preflight.py) checks distinct
PID/start time, actual reference-file hashes, source/config/tokenizer/model
identity, exact prefill, generated suffixes and ranking/relevance. Score
comparison was fixed **before generation** to the inherited
`torch.float32` + `torch.allclose(atol=1e-5, rtol=1e-5)` rule from
`8f36720a024be671a3ba893370a98126bc0f3b4a` `shuffle_runtime.py`.
CPU synthetic comparator fixtures test this rule and rejection cases; they do
not substitute for genuine separate-process model outputs.

The saved CPU evaluation stage consumes genuine predictions through the actual
production aggregator, original offline metrics and original paired bootstrap
logic. PREFLIGHT metrics are explicitly 32-user interface results. FORMAL
retains all 22,363 users, one exact no-duplicate/no-missing shard and full
same-process Normal/OFF verification. Empty collision statistics have explicit
zero/empty handling; invalid complete SID combinations remain recorded.

The fresh `21d495b0be5b606853ec7cec68a3eaa1b58acbd7`
[round-3 training job](single_gpu/jobs/g3-lcrec-single-train-20261118-r3/status.json)
and its independent
[boundary comparison](single_gpu/jobs/g3-lcrec-single-compare-20261118-r3/output/boundary_compare.json)
have now actually SUCCEEDED/exit0 with owned process groups empty. Continuous,
early and new-process resumed training completed steps 10/2/10. Actual inventory
is 450 trainable tensors/288,571,392 parameters, all four embedding/head copies
`[32786,4096]`, exact optimizer membership and 291 byte-stable frozen tensors.
The early two-update observation has LoRA A unchanged, B224/embedding1/head1
changed; the complete continuous and resumed trajectories change all A224/B224/
embedding1/head1 families. The new process strictly restores step2, microstep20,
pending4/8 gradients, external scheduler `last_epoch=2`, and optimizer microstep3.
The subsequent 60 raw input/label/order records and micro-losses, eight optimizer
step/LR/gradient-norm observations, all four epoch row sets and validation losses
match exactly. All 457 full-state tensor leaves are exact; both 4,040,238,192-byte
pre-best state files share SHA-256
`523fac9a448e0eb9ce79e8c7bc360eac45621ce43bd652fa57db0a52b48d7670`.
Native validation selects checkpoint10 (bounded 32-example validation only);
its exported adapter matches every selected tensor key/value and has SHA-256
`cf1550ce915b132c1f9ae089a7ac2f75d47ff756a7297cb94a22743bb329e1c8`.
These fresh training/save/reload/resume checks have direct non-author signoff in
[the raw-evidence review](single_gpu/review_cpu_controls_agent.json).
The genuine [generation runner](single_gpu/jobs/g3-lcrec-single-generation-20261118-r3/status.json)
and [evaluation runner](single_gpu/jobs/g3-lcrec-single-evaluation-20261118-r3/status.json)
have also SUCCEEDED/exit0 with owned process groups empty. Both actual inference
processes strictly loaded all 450 adapter keys / 288,571,392 elements and checked
all active embedding/head rows against the selected saved weights, with explicit
saved-to-loaded dtype conversion. Four-condition PID 3327064 recorded exact
Normal/OFF input, suffixes, scores and rankings for all 32 users/640 candidates.
OFF/S3/S4 each cover 268 real visible-history item groups; S3 and S4 change the
actual prefill for every selected user and preserve their specified fourth-slot,
non-history, mask and original-ID semantics. The independent Normal reload uses
PID 3327203 and a distinct start time; source/classes/config/model identities and
both baseline reference-file hashes match. All 640 FP32 scores are exactly equal
(maximum absolute difference 0), satisfying the inherited predeclared allclose
criterion without changing its tolerance. The separately labeled repeated-item
synthetic fixture also reaches all four genuine model calls and is excluded
from the 32-user predictions/metrics.

The [real evaluation summary](single_gpu/jobs/g3-lcrec-single-evaluation-20261118-r3/output/evaluation/summary.json)
binds production aggregation, original offline metric output, both paired
analyses, CSV/NPY/NPZ and the original four-condition prediction files. These
short-run interface samples have all-zero hit/NDCG/MRR/CCE metrics; no formal
research result is claimed. Of 640 candidates, invalid complete SIDs are
Normal/OFF: 633, S3: 631 and S4: 632, while all candidate token positions remain legal.
The zero-collision catalog and its 12,101-item coverage denominator remain
correct. Direct non-author raw/source checks are in
[single_gpu/review_cpu_controls_agent.json](single_gpu/review_cpu_controls_agent.json);
the metric-consumer author's implementation is checked by the separate
[training-agent review](single_gpu/review_training_agent.json).
All 76 generation and 52 evaluation transfer members were rehashed locally
against remote collection records and verified per-file transfer receipts.
The sole later `g3_remote.py` diff adds `.csv/.npy/.npz` to the existing bounded
collection suffix list, keeping terminal/empty-workgroup, symlink exclusion,
16MiB member-size and transfer SHA checks. The fixed 21d computation source is
unchanged. These completed internal runtime checks do not grant external
webpage review approval or user permission to launch formal training.

## Preserved failure and diagnostic scope

The original [single-GPU r2 model job](single_gpu/jobs/g3-lcrec-single-train-20261118-r2/status.json)
remains **FAILED**: a generic all-family byte-change observation guard fired at
the early two-update boundary. Its original early report/exception/events are
retained. No failure status is rewritten as success. The next concentrated
correction, cumulative **round 3 of 3**, separates the early real-update probe
from the unchanged full-ten-update family probe and adds the direct formal
web-review guard. It does not change the optimizer/model recipe or relax the
full-run frozen/family checks.

The separate old-source resume-diagnostic and
[full diagnostic comparison](single_gpu/jobs/g3-lcrec-single-resume-compare-diagnostic-20261118-r2/output/diagnostic_compare.json)
are `DIAGNOSTIC_ONLY`: all 457 compared tensor leaves, scalar/RNG/loader state,
subsequent raw batch/loss/LR/gradient-norm and validation observations matched
exactly. The actual old-source G3 inventory measured 450 trainable tensors,
288,571,392 parameters, embedding/head `[32786,4096]`, exact optimizer membership
and unchanged frozen copies. These observations establish what was measured
in that run; they do not replace the required fresh round-3 lifecycle check.

After both diagnostic jobs had SUCCEEDED/exit0 and their owned process groups
were empty, the separately reviewed
[retirement job](single_gpu/jobs/g3-lcrec-single-retire-diagnostic-20261118-r2/output/retirement.json)
verified all identities and archived complete manifests before deleting only
old diagnostic `recovery/checkpoint-7` and `checkpoint-10`. It recorded
37,880,590,336 bytes of allocated file blocks and a separate filesystem free
space delta of 37,880,360,960 bytes; concurrent writes can affect the latter.
Both full comparison states, the selected adapter, failed originals, logs and
complete manifests remain available. No formal/active checkpoint or original
asset was a cleanup target. See the
[actual source/evidence review](single_gpu/review_retirement_source_agent.json).

## Independent review, formal launch and delivery

Non-authors directly read fixed production source and actual original job
receipts, rather than receiving only the author's summary. Their current
scope, findings, fixes, byte identities and limits are recorded in
[single_gpu/review_cpu_controls_agent.json](single_gpu/review_cpu_controls_agent.json),
[single_gpu/review_generation_agent.json](single_gpu/review_generation_agent.json),
[single_gpu/review_training_agent.json](single_gpu/review_training_agent.json) and
[the diagnostic review](single_gpu/review_diagnostic_resume_agent.json).
Each excludes its own implementation from independent signoff. Large states
stay on the server; the reviews distinguish direct lightweight-file rehashes
from actual job-produced large-tensor hash/comparison evidence. The genuine fresh training/reload/resume and generated/evaluated outputs are
now directly reviewed in their actual scopes. The main controller verifies the
combined non-author implementation coverage and the final GitHub/ZIP delivery
gates before writing the authoritative preparation status.

The formal entry builds **one new G3 training run**, then the same best adapter's
Normal/OFF/S3/S4, evaluation and collection. New output locations isolate old
LETTER results and all preparation artifacts. Query/continuation, duplicate
submission prevention and failure propagation remain in the existing
controller. Preparing or querying a draft does not seal a final launch plan.
A real `EXTERNAL_GPT_REVIEW` with its evidence-file identities is required by
both controller and direct training entry; internal non-author review cannot
satisfy it. A **new explicit user launch authorization**, bound to this G3
plan/source/config/profile/budget, is separately required. Old LETTER or
preflight authorization cannot launch FORMAL. This preparation task grants
neither that external approval nor formal launch permission.

The [saved immutable preparation draft](single_gpu/formal_preparation_plan.json)
is the exact 214,758-byte file bound by the actual dry-run (SHA-256
`44e6cc1a03f232f1d62a81ae0bde8fa1162ffd18830497bea369390c285efcbe`).
A separate [formal single-GPU budget proposal](single_gpu/formal_single_gpu_budget_proposal.json)
is **NOT_AUTHORIZED** and changes no running configuration or existing draft.
The second Normal process observed 25.030819 s for 32 fixed users; simple linear
extrapolation gives 4.859064 h per 22,363-user condition before model loading and
full-run variability. These selected boundary users are not a throughput
benchmark or a completion guarantee. The proposal increases S3/S4 from 4.5 h to
6 h each, giving a 55.5 h total within the inherited 96 h ceiling, while keeping
model/data/beam/metric settings fixed. Its arithmetic and scope were reviewed;
**the new budget has not been approved or dry-run**. After webpage second review,
the user must separately confirm the budget and launch authorization before a
new immutable formal plan is derived and checked. The completed dry-run applies
to the original saved draft only.

The runtime content commit and final delivery commit may differ for evidence,
reviews and this explanation; their exact relationship and GitHub byte
readback belong in the final delivery receipt. Small original logs/configs/
predictions and review scripts are published for webpage inspection; base
models, checkpoints and large comparison tensors remain server-side with
paths/size/content identities. The required lightweight ZIP is validated for
member SHA, CRC, duplicate/unsafe paths and sensitive content, with its whole
archive SHA outside the ZIP. Final acceptance and packaging receipts determine
`GPT_UPLOAD_REQUIRED`; this document makes no unsupported archive claim.

No system/pip/npm/toolchain package was installed and no global shell, SSH,
network or driver configuration was changed. Existing user environments and
per-job CUDA variables were reused. Historical dual-GPU/conversion reports,
failed receipts, fixed assets and unrelated files were retained.
