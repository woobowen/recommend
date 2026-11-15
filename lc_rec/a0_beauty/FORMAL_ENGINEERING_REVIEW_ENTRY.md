# A0 Beauty production closure — technical PASS, independent review PENDING

本轮完成生产执行实现与启动前 CPU 验收；**整体 PARTIAL 的剩余项是具有源码/执行工具的非作者完整审核**。未启动正式 Beauty 训练或 GPU 生成，也未填写网页 GPT 已审核通过。

```text
TECHNICAL_ENGINEERING_CHECKS: PASS
NONAUTHOR_REVIEW: PENDING
FORMAL_ENGINEERING_READINESS: PARTIAL
FORMAL_READY_CANDIDATE: NO
FORMAL_LAUNCH_AUTHORIZED: NO
NEW_METHOD_CHANGE: NONE
```

## 固定身份和真实运行范围

- 本轮工程源码最终提交：`e9a3a631929600757b7a99b5763540c6dd9d2b99`。DELIVERY SHA 置于包外交付回执，避免自引用。
- 原始未跟踪请求草稿 SHA-256 `05a9f0b2c722ddd22e6c7b8a05783973e3e35c32301ba2fd493061b36f7fc65c` 已核验并保存在 [source/formal_requests.original.py](evidence/production-closure/source/formal_requests.original.py)。
- 最终固定源码的完整十阶段与 Runner TEST RUN：`e9a3a631929600757b7a99b5763540c6dd9d2b99`，外层退出 **0**，237.997387秒。十阶段均真实提交、退出0并完成产物核验；最终 state 为 COMPLETE。
- 同一源码的独立客户端竞争/退出后重连、最新恢复点选择、exit0缺产物拒绝、真正 FORMAL prepare/缺授权拒绝，另一个 Runner 作业退出 **0**。最终 export 的95个文件全部逐文件核对 Git 内容与 SHA；发布前修正了作者 source_root 规范化和训练/Shuffle 的固定依赖清单，依赖导入检查亦退出0，未加载模型或 Trainer。
- 历史修正事实保留：`a6ef277` 的重型 Trainer 导入遗留进程触发 WORKERS_TERMINATED；`f617927` 十阶段成功后，外层进程测试因 PyTorch Python 无 `os.pidfd_open` 退出1。改为系统 Python 的受身份校验 pidfd 清理后，`e5fba09` 故障复核退出0；最终e9a3a63完整重验再确认通过。未降清收标准。
- 最终 worker admission 绑定真实 accepted request、阶段、输出路径和恢复来源；此前专项检查核验十个已接受请求并拒绝20个错误output，最终完整十阶段也经过同一准入函数。`paired_beauty.process()` 缺少正向来源context拒绝，成功context及完整数学在最终RUN实测。
- 以上均为 `SYNTHETIC_CPU_INTEGRATION_ONLY` 工程 TEST。外部训练/生成计算由明确标记的 fixture 替代；文件锁、持久状态、Runner 进程、来源检查、原 LETTER、16 指标/CCE、完整 CSV/NPZ 配对、统计输出均使用生产函数。

## 实际生产入口和依赖

入口：[formal_control.py](formal_control.py) → [production_control.py](production_control.py)。`prepare_production_plan.py` 根据固定 export、真实资产、隔离 Runner 和用户明确给定的预算 JSON 构造蓝图；`prepare`/`query` 不提交作业。

```bash
python formal_control.py prepare --plan /ABS/FORMAL_ROOT/plan.json
python formal_control.py query --plan /ABS/FORMAL_ROOT/plan.json
# 以下只供将来用户单独正式批准后使用；本轮未调用：
python formal_control.py start --plan /ABS/FORMAL_ROOT/plan.json \
  --authorization /ABS/USER_ISSUED_RECEIPT.json \
  --authorization-sha256 USER_MESSAGE_PINNED_SHA256
```

授权文件必须来自后续用户明确批准，SHA 必须由用户授权消息独立指定，不能让控制器自产批准。门槛绑定完整 plan digest、commit、配置、预算、task_id、范围、来源引用及有效期，并另查技术审核。软件不会证明一段自填文字来自人类；操作人须核对外部用户授权这一信任入口。本轮没有批准文件。正式预算及运行时资源仍待后续用户确认，不属于当前未实现的代码功能。

`start` 让服务器端控制器脱离 SSH，独立阶段由固定 Runner 接收和监督。控制器没有早于子任务预算的单 job 寿命限制；阶段各有 <=345600 秒上限，train+Normal 共享组预算，S3/S4 独立组预算，CPU 阶段另计，所有 attempt 继承总预算与清收余量。文件锁串行管理阶段，SUBMITTING 先落盘后外部调用；失联、CONTROL_LOST、未清收均不盲重投。

| 阶段 | 真实实现与验证 |
|---|---|
| train | 既有 `formal_train.py` 双 rank；检查 4 Epoch/8214、131413/44726、两 rank、四次 Validation 和 receipt |
| verify_best_adapter | [production_identity.py](production_identity.py)：原生严格最小值/并列保留、best/latest 完整清单、严格恢复事件、checkpoint→实际导出张量/dtype/shape、Tokenizer；正式分支再绑定 accepted 训练请求和 Runner 完成 |
| normal | 既有 `formal_evaluate.py` 双 rank，正向绑定 best；完整 11182/11181 分片、每用户20候选；调用原LETTER全轴审计后才允许完成，后续核对首次保存的预测SHA |
| normal_eval | 原 `load_evaluator/aggregate_predictions`→stage-A→`offline_metrics.evaluate` |
| s3 / s4 | 原 `shuffle_entry.py`/`shuffle_runtime.py`，Beauty 全轴 CPU 输入审计，固定 protocol/seed 和同一 best |
| s3_eval / s4_eval | 同一生产评价函数，预测 RUN 与评价 RUN 分别保存 |
| paired | `paired_beauty.process` 的完整来源/CSV/NPZ/22363×16/5000次 Bootstrap/转移与列表变化处理 |
| collect | 汇总固定产物及 SHA；退出0本身不构成完成 |

失败的 CPU 阶段可显式 `--retry STAGE`；完成训练及预测保留。训练恢复仅接受已验证的最新完整 checkpoint，不能从旧 best 重新启动同一 seed；旧 attempt 和费用保留。机器重启、存储故障及未知 supervisor 状态不承诺无损恢复。

## 原始回执导航和实际命令

全部最终轻量证据：[final-server/](evidence/production-closure/final-server/)。其中原始 request 给出完整 argv、解释器、env、输入 SHA；Runner `status.json`/`work.log`/`supervisor.log` 保留真实退出与清收。

令 `B=/data/bwwu/lc-rec/a0-beauty-20261115/production-closure-20261115`，`P=$B/SYNTHETIC_CPU_INTEGRATION_ONLY_e9a3a6319296/plan.json`。以下命令实际通过持久 Runner request 执行，完整绝对 argv 在对应回执中：

| 验收 | 实际命令核心 | 退出码/证据 |
|---|---|---|
| 完整十阶段、故障/防重/重试、真实torchrun | `/data/bwwu/envs/letter-20261110/bin/python $B/releases/e9a3a631929600757b7a99b5763540c6dd9d2b99/lc_rec/a0_beauty/closure_acceptance.py --plan $P` | **0**；[主作业原始status](evidence/production-closure/final-server/cpu-jobs/letter-a0-beauty-cpu-runner-check-main-e9a3a6319296/status.json)、[acceptance](evidence/production-closure/final-server/SYNTHETIC_CPU_INTEGRATION_ONLY_e9a3a6319296/acceptance.json)；negative_tests有26个条目，其中包括CPU重试及完成后防重观察 |
| 两客户端/退出后重连/正式拒绝 | `/data/bwwu/envs/letter-20261110/bin/python $B/releases/e9a3a631929600757b7a99b5763540c6dd9d2b99/lc_rec/a0_beauty/closure_conformance.py $P` | **0**；[conformance](evidence/production-closure/final-server/SYNTHETIC_CPU_INTEGRATION_ONLY_e9a3a6319296/concurrent_clients/conformance.json)；其中正式缺授权调用按预期退出1 |
| 真实依赖闭合/作者源码哈希 | `python $B/releases/890dcf1645868f8c94882cf19d0173f10f68646f/lc_rec/a0_beauty/closure_dependencies.py OLD_P`（该依赖代码未改，完整argv见request） | **0**；[原始status](evidence/production-closure/final-server/cpu-jobs/letter-a0-beauty-cpu-runner-check-dependencies-890dcf164586/status.json) |
| 反向时钟及十阶段重验（无后续受影响数学变化） | `python releases/abf9b71.../lc_rec/a0_beauty/closure_finalize_check.py P`（历史P） | **0**；[原始check](evidence/production-closure/final-server/SYNTHETIC_CPU_INTEGRATION_ONLY_f6179273dd27/finalize_check.json) |

[最终 state](evidence/production-closure/final-server/SYNTHETIC_CPU_INTEGRATION_ONLY_e9a3a6319296/state.json) 为 COMPLETE。该 RUN 训练、Normal、S3、S4 各接受一次；collect 的 CPU 重试另有记录，未重复训练或生成。

[配对 S3](evidence/production-closure/final-server/SYNTHETIC_CPU_INTEGRATION_ONLY_e9a3a6319296/attempts/paired/a0/s3/paired_comparison.json) 与 [配对 S4](evidence/production-closure/final-server/SYNTHETIC_CPU_INTEGRATION_ONLY_e9a3a6319296/attempts/paired/a0/s4/paired_comparison.json) 包含完整16列差值、相对变化、区间、Hit transitions、Top1/Top20顺序和集合变化。冻结列式独立复算的最大 draw 差分别为 `3.99853761212654e-15` / `3.939557013943329e-15`，低于 `1e-13`；NPY/NPZ 及各自 SHA 一并保留。三个条件各22363用户/447260候选；每组官方 CCE 比较覆盖全部22363用户。数学、分组、Coverage分母12088、PCG64 seed20261113/5000次/linear95% CI 均沿用旧实现。

本轮最后将完整候选/用户轴审计从紧邻的评价阶段提前接入生成完成门槛，并增加保存SHA对照；原LETTER轴散列与扩展指标轴散列采用各自原始schema，不混用。新增rank/target/score/候选数/保存身份负例已在最终RUN通过。

故障覆盖包括：正式授权缺失、exit0缺产物、checkpoint损坏、错误best、receipt错RUN/冲突/写失败、Adapter错配、合成进正式、评价验证失败、用户轴/列漂移、提交回执丢失、Runner不可达、活 worker、未知 supervisor、清收不确定、重复/冲突请求、累计预算耗尽、跨 attempt 费用及CPU重试。注入只发生在隔离测试状态/外部边界，具体输入快照与异常见回执，不能将注入状态当作真实 GPU 事实。

## Runner、空间与时间

实际私有 Runner：`$B/runner-beauty-cc1a1e277faa.py`，SHA-256 `cc1a1e277faa9bc713a4d0d1b620248d47c398e8ed17d4ed3d7ab52805cc657e`；[固定字节与 sidecar](evidence/production-closure/runner/)。namespace `beauty-production-closure-cpu-only`，job root `$B/cpu-jobs`。公共 Runner 未改。

真实 `torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 --max-restarts=0`：正常 SUCCEEDED/0、rank失败 FAILED/1、短超时 TIMED_OUT/1，均确认 agent/两 worker PID、start ticks、session、任务标记及清收。无关进程保持存活，最后只用其自身 pidfd 清理。345600接受、345601拒绝。独立 SSH 提交连接结束后另连查询；独立控制客户端退出时 worker 仍在，新客户端未重复提交。

[预算原始账](evidence/production-closure/final-server/budget.json)：所有相关 Runner 墙钟相加 **1658.314926秒**，包含历史费用且父子作业重复计入，是保守上界；本地小测试再保守预留60秒，合计上界1718.314926秒，低于7200。新增服务器临时逻辑量 **961602976字节**，低于8GiB。日期以服务器 UTC 回执为准（2026-11-15）。没有新增 GPU 使用、框架安装或系统配置修改。

旧合成大输入保持只读：`/data/bwwu/lc-rec/a0-beauty-20261115/SYNTHETIC_CPU_INTEGRATION_ONLY_271df1e33054_r0`；[逐文件路径/大小/SHA](evidence/production-closure/final-server/source_inputs.json)。测试输出是独立字节副本和 sidecar，不生成新候选，也未改原文件。server export 明确不是 Git checkout，[最终95文件RUN_VERSION](evidence/production-closure/final-server/releases/e9a3a631929600757b7a99b5763540c6dd9d2b99/RUN_VERSION.json) 给出逐文件 Git/export 映射。

## 方法差异与审核边界

| 范围 | 分类 / 结论 |
|---|---|
| Beauty作者数据/SID、22363轴、965专用Token、32965词表 | BEAUTY_DATA_ADAPTATION；沿用已审核配置 |
| 模型/LoRA家族、Loss、优化器、LR/scheduler、精度、Prompt0/Beam20 | UNCHANGED；Embedding/Head行数随Beauty词表变化，不声称参数总数与Instruments相同 |
| micro8×GAS4×2、连续跨Epoch累积与严格恢复 | PREVIOUSLY_APPROVED_TRAINING_ADAPTATION |
| receipt原始manifest字节、来源门槛、调度、持久状态与评价包装 | ENGINEERING_EVIDENCE_ONLY |
| 训练入口、Normal生成入口、offline_metrics、shuffle/core.py相对e8fc | 字节未改；配对数学未改，仅函数提取/来源门槛 |
| 新方法 | NEW_METHOD_CHANGE: NONE；未声称未经实测的数值等价 |

[非作者真实范围](evidence/production-closure/NONAUTHOR_REVIEW.md)：两代理无工具，只对主控提供的逻辑给出桥接意见，未直接读文件/执行。主控已执行内部检查，仍不能代替独立完整审查。网页GPT后续须检查最终源码、原始回执、授权信任入口及正式来源分支；目前没有真正正式Beauty模型，未来正式产物必须按这些门槛再验收。

本轮为用户追加的一次集中实现—修正—复核授权；历史三轮及后续专项授权不清零。保留了 transport 根目录拒绝、无效 HEAD export、重型 Trainer 导入遗留进程、测试解释器 pidfd 接口差异等失败事实，未以降低清收标准解决。必要研究变更为零。

依据以安装版本和真实源码为准：[Transformers v4.31.0 Trainer](https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/trainer.py)、[DeepSpeed checkpointing](https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html)。[安装 Trainer 选模证据](evidence/production-closure/installed_trainer_selection.json) 与轮转 fixture 的实际源文件 SHA 均保留。

审核复算入口：[INTERNAL_REVIEW.json](evidence/production-closure/INTERNAL_REVIEW.json)、[方法不变性](evidence/production-closure/method_invariance.json)、[工程diff](evidence/production-closure/engineering.diff)、[证据核验程序](evidence/production-closure/finalize_delivery_evidence.py)。轻量统计NPY/NPZ已提交；大型预测、CSV及CPU checkpoint fixture留服务器，以stage_result/metrics/paired文件内的路径、大小和SHA核对。原合成数据本轮只读复用，不能称为正式Beauty指标。

---

# 以下全部为历史 PARTIAL 记录（保留原文，不代表本轮状态）

# A0 Beauty formal engineering review — partial

## Production closure work in progress — NOT READY

The current work adds fail-closed persistent control primitives and a checkpoint
receipt hook. These are not a complete production controller: stage-specific
request construction and artifact verification, the formal best-source proof,
the unified evaluation path, real CPU torchrun runner tests and full pipeline
acceptance remain incomplete. The control primitive rejects formal scope; it
must not be used to launch a formal job. Receipt code changes training-file
bytes solely to record completion evidence before rotation; it is not yet
validated in a full GPU training run. The earlier sections below are historical
evidence, not proof of the new work.

`TECHNICAL_ENGINEERING_CHECKS: FAIL`
`NONAUTHOR_REVIEW: PENDING`
`FORMAL_ENGINEERING_READINESS: PARTIAL`
`FORMAL_READY_CANDIDATE: NO`
`FORMAL_LAUNCH_AUTHORIZED: NO`


## Follow-up CPU evidence — still partial

Code snapshot `420d09c40f4435a2061c57223e7c1db066beb696` adds an isolated
driver for `formal_evaluate.load_evaluator`, `audit_prediction`, and
`aggregate_predictions`. The original LETTER evaluator was actually called on
the existing synthetic Normal, S3 and S4 candidate shards, each with 22,363
users. The author evaluator and all six input shards were bound by SHA in the
server request. Job `letter-beauty-letter-evaluator-cpu-420d09c40f44-r0` exited
0 and reported verified cleanup; the raw request, status, log and result are in
`evidence/formal-engineering-cpu-r1/`. The output is synthetic, not model
inference or research results. This isolated evaluator does not establish a
production controller or end-to-end production evaluation.

The same snapshot extracts the existing paired Bootstrap mathematics into
`paired_bootstrap`, which the formal CLI calls only after its existing source
gate. `test_paired_bootstrap_full_cpu.py` used SHA-checked per-user matrices
from the prior synthetic fixture and independently recomputed both comparisons.
The largest draw discrepancies were `3.99853761212654e-15` for S3 and
`3.939557013943329e-15` for S4. The raw result is in the same evidence directory.
This tests the production mathematical function, not the full paired CLI or
formal best-Adapter provenance.

No production controller, training provenance proof, actual CPU torchrun
runner acceptance, cumulative recovery budget, authorization gate, full
end-to-end controller dry-run, or new nonauthor review was completed in this
follow-up. Three attempted nonauthor agents reported they had no file or
execution tools; no independent review PASS is claimed. The configured
`storage_save_total_limit=2` and checkpoint rotation also require a durable
per-epoch completeness proof before rotation if all four epochs are to be
verified; simply requiring four checkpoint directories after training conflicts
with the existing retention behavior. No formal GPU job was submitted.

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
