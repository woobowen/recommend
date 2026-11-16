# 2026-11-15：预算已批准，独立审核与正式封存待完成

用户在本次任务消息明确批准 Train＋Normal 108000 秒、S3/S4 各16200 秒、CPU 3600 秒、总144000 秒及120秒清收余量；双GPU上限280800 GPU-device-seconds。此批准不包含正式启动。新的 [预算批准记录](evidence/prelaunch-hardening/budget-user-approved.json) SHA-256 `2bcd27d8b7c6dbdfe373e3fe97d9bbcc8ca9b2626e54d482241fea46f7936deb`，服务器同字节副本位于 `/data/bwwu/lc-rec/a0-beauty-20261115/production-closure-20261115/formal-candidate-5b2917d/budget-user-approved.json`。历史候选文件未更改，其旧的“未批准”标记表示生成时状态，以本段用户新批准为准。该记录不是用户启动授权收据。

独立审核仍为 PENDING：此前审核代理明确不能读取或执行文件；本次用户没有提供独立审核 PASS 文件，也没有自称出具该记录。本轮主控检查不能替代非作者审核。正式审核文件缺失，因此尚无封存 Plan，也无正式 GPU 任务。固定候选仍是 [formal-candidate.json](evidence/prelaunch-hardening/formal-candidate.json)，SHA-256 `cd5e7ab6b15f27a506079fea8786d6e6e1dbfdcd3bfb6c5a49ddaf9f43f1ec5e`。本轮没有修改工程源码、计算方法或已有历史文件。

重核服务器 [资源](evidence/final-budget-review/server-resource-snapshot.txt) 和 [身份](evidence/final-budget-review/server-identity.json)：96文件 SHA 与服务器清单匹配，且 [Git 逐文件对照](evidence/final-budget-review/git-export-check.json) 96/96 一致；Runner SHA `cc1a1e277faa9bc713a4d0d1b620248d47c398e8ed17d4ed3d7ab52805cc657e`，sidecar SHA `eafd22574dc2d07fd253822732a572200032fcd7e88654140021e2e045087894`，正式 jobs 目录为空、正式输出与 registry 尚未创建。两张GPU UUID未变、各1 MiB、0%利用率、无 compute app；这是当前快照，启动时重新检查。时区 `2026-11-16 04:48 UTC` 即 `2026-11-15 23:48 America/New_York`。

[本轮有界负向测试](evidence/final-budget-review/test_start_denial.py) 使用真实服务器资产、已批准预算及生产 builder/Controller，在 SYNTHETIC 目录中仅对测试根路径 guard 打补丁，并将 Popen 替换为遇调用即失败的监视器；未修改正式源码。实际 `start` 缺少授权时抛 PermissionError、Popen未调用、attempts为空，退出0；[原始输出](evidence/final-budget-review/test_start_denial.stdout)。未打补丁的生产验证拒绝该合成路径；PENDING review 不能满足实际 `authorize`。测试执行0.163秒，不是正式训练或正式GPU。

## 取得真实外部审核后的单次封存入口

**先决条件：** 独立审核者已实际读取固定源码与证据并出具其真实 PASS；操作人从真实渠道核验身份和文件，而非自行编造。审核 JSON 的 `candidate_plan_sha256` 必须等于上述原始候选 SHA，其他必需字段见下方历史封存说明。将真实文件放到候选固定路径 `independent-prelaunch-review.json`。以下在服务器运行，只构造与核验 Plan，不含启动动作；审核来源真实性仍需人工核验。

关键区别：现有 `prepare_production_plan.py` 的 `validate_plan` 仅校验审核字节，**不会在构造时要求审核 PASS**；正式 `authorize` 才要求 PASS。本次未改生产源码（历史任务已有三轮集中修正），因此必须先后调用现有 `approval_documents(..., require_pass=True)` 并比对审核 SHA，只有所有检查通过才将产物称为封存计划。若构造后检查失败，保留产生的文件并报告未封存，不得启动或以删除账本解决。

```bash
cd /data/bwwu/lc-rec/a0-beauty-20261115/production-closure-20261115/releases/6e1b211d9c37f358c43b1540b40b17f6703150d2/lc_rec/a0_beauty
PYTHONOPTIMIZE=0 PYTHONDONTWRITEBYTECODE=1 /data/bwwu/envs/letter-20261110/bin/python - <<'PYSEAL'
import hashlib, json, subprocess
from pathlib import Path
from production_identity import identity
from production_control import approval_documents, validate_plan
from formal_control import digest
base=Path('/data/bwwu/lc-rec/a0-beauty-20261115/production-closure-20261115/formal-candidate-5b2917d')
candidate=base/'formal-candidate-v2.json'
budget=base/'budget-user-approved.json'
review=base/'independent-prelaunch-review.json'
assert hashlib.sha256(candidate.read_bytes()).hexdigest()=='cd5e7ab6b15f27a506079fea8786d6e6e1dbfdcd3bfb6c5a49ddaf9f43f1ec5e'
assert hashlib.sha256(budget.read_bytes()).hexdigest()=='2bcd27d8b7c6dbdfe373e3fe97d9bbcc8ca9b2626e54d482241fea46f7936deb'
c=json.loads(candidate.read_text()); b=json.loads(budget.read_text()); r=json.loads(review.read_text())
assert b['budget']==c['budget']['budget'] and b['stage_seconds']==c['budget']['stage_seconds']
assert r['candidate_plan_sha256']=='cd5e7ab6b15f27a506079fea8786d6e6e1dbfdcd3bfb6c5a49ddaf9f43f1ec5e'
args=c['seal_argv_after_real_independent_review']; precision=Path(args[args.index('--precision-approval')+1]); release=Path(args[args.index('--deployment')+1]).parent
assert hashlib.sha256(precision.read_bytes()).hexdigest()==c['precision_approval_sha256']
pr={'prelaunch_review':str(review),'precision_approval':str(precision),'approval_files':{'prelaunch_review':identity(review),'precision_approval':identity(precision)},'run_commit':c['engineering_commit'],'config_sha256':c['effective_config_sha256'],'release':str(release)}
review_sha=pr['approval_files']['prelaunch_review']['sha256']
approval_documents(pr,require_pass=True)
args[args.index('--budget')+1]=str(budget)
subprocess.run(args,check=True)
plan_path=Path(args[args.index('--root')+1])/'plan.json'; raw=plan_path.read_bytes(); plan=json.loads(raw)
validate_plan(plan); approval_documents(plan,require_pass=True)
assert plan['approval_files']['prelaunch_review']['sha256']==review_sha
assert plan['budget']==b['budget'] and plan['stage_seconds']==b['stage_seconds']
print(json.dumps({'status':'SEALED_NOT_SUBMITTED','path':str(plan_path),'file_sha256':hashlib.sha256(raw).hexdigest(),'canonical_plan_digest':digest(plan),'review_sha256':review_sha}))
PYSEAL
```

`PLAN_SEAL_READY` 表示上面既有生产函数与实际 builder 的准备及缺授权负例均已验证，且提供了不改生产源码的操作顺序；完整真实 PASS 文件的端到端封存尚无法实测。用户仍需另行批准最终 Plan、预算及执行范围并独立提供授权收据 SHA。`formal_control.py start` 不在本轮运行。

```text
INDEPENDENT_PRELAUNCH_REVIEW: PENDING
EXTERNAL_GPT_REVIEW: PENDING
METHOD_INVARIANCE: PASS
PROVENANCE_AUDIT: PASS
APPROVAL_SHA_BINDING: PASS
CROSS_ROOT_BUDGET_GUARD: PASS
RUNNER_DEPLOYMENT: PASS
BUDGET_CONFIGURATION: PASS
FORMAL_BUDGET_USER_APPROVED: YES
PLAN_CANDIDATE_COMPLETE: YES
PLAN_SEAL_READY: YES
PLAN_SEALED: NO
FORMAL_PLAN_SUBMITTED: NO
FORMAL_LAUNCH_AUTHORIZED: NO
TECHNICAL_PRELAUNCH_AUDIT: PASS
FORMAL_READY_CANDIDATE: YES
```

下方为固定 DELIVERY `db00b1fc6d16d8fcbb7d347fc25f09a64ef311ef` 的历史工程收口记录，其预算批准状态是当时事实，不覆盖本段新授权。

---

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
