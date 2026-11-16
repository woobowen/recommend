# A0 Beauty 启动前工程收口与审核入口

本文件对应工程源码及 CPU RUN `6e1b211d9c37f358c43b1540b40b17f6703150d2`。正式 Beauty GPU 训练和全量生成未运行；预算未获用户授权。真实独立审核文件、用户授权及其外部 SHA 均尚未取得。本文件是主控技术审核，不是非作者审核。

## 审批与实验身份

- [production_control.py](production_control.py) 中 `approval_documents` 将审核、精度批准的路径、长度及 SHA 纳入 plan，重新读取同一字节并验证；Runner request 的 inputs 绑定审批文件、导出 manifest、Runner 和 sidecar 及审核证据。Worker 重新执行 plan/授权验证；[formal_train.py](formal_train.py) 在训练代码导入前使用 `O_NOFOLLOW` 从打开的普通文件读取并校验实际解析的字节。
- `authorize` 要求后续用户独立提供授权收据 SHA，校验收据作用域、动作、plan digest、源码、配置、预算、任务、系列、过期时间和独立审核。软件只能校验这些字节与字段，**不能证明用户消息或审核者身份**；操作人必须将 SHA 与真实外部消息核对。
- 固定的服务器 Beauty registry 独立于输出 root 和 Runner job_root，按 Beauty 资产、配方、模型 revision、seed、条件及实验系列确定身份。首次获授权后锁定原 root、task_id 和 plan；同系列更换 root/task_id/job_root/commit 对应的 plan 将被拒绝，不得到新预算。原 root 的 state 丢失也拒绝重新创建。失败 attempt 的 Runner 时间在原 state 中累计；新系列还需单独的用户授权收据标记及先前系列引用。该机制不防同账户有意篡改/删除固定 registry、绕过控制器运行或伪造外部消息；部署和权限是信任边界。新源码修复后的跨版本迁移未实现，需保留原任务并单独处理，不能重置账本。
- 历史本地未推送提交 `508bc53e76ff520fa97ea5bd324d48e53e19ecbc` 已保留并作为本轮起点。后续三个集中工程修正分别为 `aafbe275ea11f51105d4b2f54775ee348a256c10`（固定 registry、文件防 symlink）、`5b2917d39c6b0e190907908871945962b204fa5c`（正式授权 scope）、`6e1b211d9c37f358c43b1540b40b17f6703150d2`（新系列独立授权）。`3994c4edc9d15a027912790e29a4d53a615bb2f4` 仅增加测试。

## 候选方案及封存

[正式候选 JSON](evidence/prelaunch-hardening/formal-candidate.json) SHA-256 `cd5e7ab6b15f27a506079fea8786d6e6e1dbfdcd3bfb6c5a49ddaf9f43f1ec5e`。这是 `FORMAL_CANDIDATE_UNSEALED_NOT_SUBMITTABLE`：明确固定96个源码文件、输入资产、GPU UUID、解释器、Runner、sidecar、任务根、精度批准、预算与封存 argv；独立审核 SHA 为空是其尚未存在的真实状态。服务器同字节副本位于 `/data/bwwu/lc-rec/a0-beauty-20261115/production-closure-20261115/formal-candidate-5b2917d/formal-candidate-v2.json`。`PLAN_SEALED: NO`。未来正式 best Adapter 与预测 SHA 动态产生和验证，不能提前填预验收权重。

获得真实独立审核后，将其作为候选中指定的 `independent-prelaunch-review.json`。它应为 JSON，包含 `status=PASS`、`RUN_COMMIT=6e1b211d9c37f358c43b1540b40b17f6703150d2`、`config_sha256=24b568161f2d71d3c5962372a38683b2d33ee5d6d5064de0612215946a8106fd`、`formal_train_sha256=a0b29536f914c98e57dbfeaefc0d94ce38fcc4e29747f121b5e3aead054f93da`、`review_kind`（`INDEPENDENT_NONAUTHOR` 或 `EXTERNAL_GPT_REVIEW`）、真实 reviewer 和 `evidence_files`（实际可读文件的 path、bytes、sha256）。审核者必须真实读取并审核，不能由主控生成虚假 PASS。候选中的 `seal_argv_after_real_independent_review` 调用现有 [prepare_production_plan.py](prepare_production_plan.py)，真实 builder 会固定审核文件和 precision 的 SHA。封存后用户另行审阅最终 plan 和预算，提供绑定 plan canonical digest、明确范围和到期时间的用户授权文件及其独立 SHA；不得将用户给出的 SHA 从该文件自算后冒充用户消息。正式入口是 `formal_control.py start --plan <sealed-plan> --authorization <user-receipt> --authorization-sha256 <user-message-pinned-sha>`；本轮未调用。

[预算 JSON](evidence/prelaunch-hardening/budget-candidate.json) 为待批准候选：Train＋Normal 108000 秒、S3/S4 各16200 秒、CPU 3600 秒、总144000 秒；双卡上限280800 GPU-device-seconds＝78 GPU-device-hours。训练阶段上限108000、Normal 21600、S3/S4 各16200 秒；CPU verify/三个 eval 各900、paired 1200、collect 600 秒，但 CPU 阶段共享3600秒，阶段上限不可相加。每次请求从阶段/组/总剩余最小值扣120秒清收余量，剩余必须大于零，全部 attempt 共享预算；Runner TERM/KILL 代码分别2秒/3秒，120秒是保守规划，并非 OS 故障下清收必然完成的保证。

历史 Instruments Train＋Normal 实测45322.209秒（12.59小时）；30小时是其2.38倍。历史每个 Shuffle 的4小时为上限而非测量值。将12.59小时与两个4小时上限相加得到约20.59小时双卡规划参考、约41.18 GPU-device-hours；这只是跨数据域和上限外推，不能当作 Beauty 实测或可靠预测。30小时共享组内没有单独保证的恢复时间：训练超过约24小时会压缩Normal的6小时阶段上限空间，失败恢复同样消耗该组余额；缩短相较旧96小时有提前终止风险。预算上限不是预期耗时，也不是正式许可。

## 方法与来源

[实际源码 diff](evidence/prelaunch-hardening/production.diff) 显示 formal_train 仅增加正式审批字节读取及参数。`config/formal_beauty_seed42.json`、`formal_evaluate.py`、`checkpoint_receipt.py`、`production_identity.py`、`shuffle_entry.py`、`shuffle_runtime.py`、`offline_metrics.py`、`paired_beauty.py`、`lc_rec/shuffle/core.py` 相比历史工程 RUN e9a3a63 字节不变。

- `UNCHANGED`：原 LoRA/CE/AdamW/cosine、Int8/BF16、Prompt0/Beam20、回调、Shuffle、评价和 Bootstrap 定义。
- `BEAUTY_DATA_ADAPTATION`：作者 Beauty 数据、SID、tokenizer、22363 用户，数据 SHA 在候选中固定；Embedding/Head 随 Beauty 词表变化。
- `PREVIOUSLY_APPROVED_TRAINING_ADAPTATION`：双卡 micro8、GAS4、连续跨 Epoch 累积、严格部分状态恢复及已批准 BF16/Flash backward 工程适配。
- `ENGINEERING_EVIDENCE_ONLY`：审批字节、权限与 ledger、来源/阶段校验和封存。
- `NEW_METHOD_CHANGE: NONE`。

`formal_train.py` 实际设置4 epoch、8214 steps、AdamW 2e-4、weight decay .01、warmup .01、cosine、ZeRO-2、gradient checkpointing，训练时完整 Validation。`production_identity.py` 核查双 rank 四轮 validation/receipt、最早严格最小 eval_loss、best/latest 两 checkpoint、严格恢复、Adapter 每个 tensor 的 shape/dtype/value 和 Tokenizer；正式分支要求关联 Runner 接受的训练请求及完成状态。Normal/S3/S4 分别验证所用 Adapter 与 best 一致；22363 用户在两 rank 分为11182/11181，Beam20、Prompt0，不过滤/去重。Shuffle 为 seed42 的 S3/S4 非恒等排列。评价包括原 LETTER、16列、CCE、Coverage、分组及同用户配对 PCG64 seed20261113 的5000次 linear percentile Bootstrap。历史真实短训练与32用户预验收不在本轮重跑；CPU合成 fixture 不能证明未来真实正式权重或结果。

## 测试、资源与局限

[审批/授权/registry 测试](test_prelaunch_guards.py) 在本机使用真实生产函数，包含两个审批缺失/改变/互换、符号链接、训练入口字节读取、request inputs、Runner `verify` 输入漂移、授权 digest/过期/plan 变更、独立新系列与跨 root/task/job_root 约束；Runner namespace 在该定向测试中被 mock，并未在该测试里提交 Runner 作业。[测试输出](evidence/prelaunch-hardening/test_guards.stdout)；[服务器四项输出](evidence/prelaunch-hardening/server-targeted.stdout) 实际使用固定 Python 及源码。 [计划构造与负例](evidence/prelaunch-hardening/test_seal_integration.py) 在服务器使用真实资产和 builder；仅对标记为 SYNTHETIC 的目录补丁掉根目录 guard 来准备一个 PENDING fixture，未创建假 PASS；未打补丁的生产校验拒绝该目录，缺授权拒绝，空 GPU/配置 SHA 漂移拒绝。[预算重试](evidence/prelaunch-hardening/test_budget_step.py) 用真实 `Control.step`，模拟 Runner 外部边界和 WSL 内存读取；200 秒旧 attempt 从后续600秒阶段限额扣除，另扣120秒清收余量，故下一请求为280秒。[初始测试失败及处理](evidence/prelaunch-hardening/initial-test-failures.txt) 保留。测试未证明操作系统层面恶意篡改不可行，也不能验证真实用户来源。

CPU 十阶段结果与原始 request/status/log 将见本目录 `evidence/production-closure/server/6e1b211d9c37/`；注意训练和生成仍分别为合成 checkpoint 与已有合成预测替代，不是正式模型。双 rank CPU Runner 成功/失败/超时与清收、配对处理受这一验收及历史证据范围约束。

[服务器快照](evidence/prelaunch-hardening/server-snapshot.txt) 是 2026-11-16 02:46 UTC；两张 RTX4090 分别 UUID `GPU-dfff0e95-1db6-e2f4-3431-55bb852fa0ae`、`GPU-59cac3bd-3645-668b-79c8-839e0d878893`，各1 MiB、0%利用率、无 compute app；启动当天必须刷新。私有 Runner SHA `cc1a1e277faa9bc713a4d0d1b620248d47c398e8ed17d4ed3d7ab52805cc657e`，sidecar SHA `eafd22574dc2d07fd253822732a572200032fcd7e88654140021e2e045087894`。[RUN_VERSION](evidence/prelaunch-hardening/RUN_VERSION.json) 为服务器96个文件逐项对照 Git 的 SHA，服务器 export 不是 checkout。固定解释器 torch 2.6.0 / deepspeed 0.16.9 / transformers 4.31.0 / accelerate 0.21.0 / flash-attn 2.7.4.post1。未安装依赖、更改公共系统或使用 GPU。

两名先前审核代理明确无法读取文件或执行程序，按用户要求未反复启用同类不可用代理；`NONAUTHOR_REVIEW: PENDING`。需网页 GPT 读取真实源码、证据和最终版本完成独立审核。正式设备空闲、真实最佳 Adapter、全量预测、正式指标均只能在后续取得授权后实际核验。

实际最终 CPU RUN `letter-a0-beauty-cpu-runner-check-main-6e1b211d9c37` 的 [acceptance](evidence/production-closure/server/6e1b211d9c37/pipeline/acceptance.json) 为 PASS，内层总耗时241.3895秒、26项负例；[Runner status](evidence/production-closure/server/6e1b211d9c37/runner/status.json) 为 SUCCEEDED/0 且清收通过。成功、失败、超时均实际经过双 CPU rank，结果分别为 SUCCEEDED、FAILED、TIMED_OUT。原始 `request`/`accepted`/`status`/`work.log` 和阶段证据见同一目录。目录收集226个轻量文件。新RUN15个父子job墙钟加总414.545811秒，父子重叠故为保守上界，非 CPU 核秒；此前保守累计1718.314926秒，合计2132.860737秒，另有本轮非 Runner 轻量测试，仍在原7200秒额度内。局部服务器测试墙钟分别约0.01/0.12秒；本地有界测试计时见 `test_*.time`，完整 CPU 核秒未记录。没有新增GPU-device-seconds。服务器 export 实际成员96个，与 manifest 相同，无额外或缺失文件。

```text
APPROVAL_SHA_BINDING: PASS
CROSS_ROOT_BUDGET_GUARD: PASS
PRODUCTION_CONTROL_AUDIT: PASS
METHOD_INVARIANCE: PASS
PROVENANCE_AUDIT: PASS
RUNNER_AND_BUDGET_AUDIT: PASS
NONAUTHOR_REVIEW: PENDING
PLAN_CANDIDATE_COMPLETE: YES
PLAN_SEALED: NO
FORMAL_BUDGET_USER_APPROVED: NO
FORMAL_PLAN_SUBMITTED: NO
FORMAL_LAUNCH_AUTHORIZED: NO
TECHNICAL_PRELAUNCH_AUDIT: PASS
FORMAL_READY_CANDIDATE: YES
```

这些 PASS 仅指本轮源码、受控 fixture 与历史证据支持的技术准备，不能替代真实非作者审查、后续用户授权或正式产物核验。`FORMAL_READY_CANDIDATE` 表示可交付独立审查的未授权候选，**不是允许启动**。
