# A0 Beauty 最终启动前审核：PARTIAL，禁止提交

本记录是对工程 RUN `e9a3a631929600757b7a99b5763540c6dd9d2b99`、此前 DELIVERY `b225fe65ef51aae169264767c8d26f6904a07a95` 的本轮部分审核；不是非作者 PASS、正式预算或用户启动授权。未启动模型、训练、生成或新增 GPU 计算。此记录不声称本轮完成完整最终终审。

## 阻塞与证据

1. 审批文件内容漂移：`production_control.py:74-88` 的 `authorize` 验证独立提供的授权收据 SHA 和 plan digest，但以路径读取 `prelaunch_review`；`request` 第 111–115 行只列出 plan 输入及授权收据。`prepare_production_plan.py:31` 只在 plan 中保存 `prelaunch_review` 和 `precision_approval` 路径；`formal_train.py:97-112` 再按路径读取它们。若两文件未另外列入受验证 inputs，plan 和 request 不固定这些文件字节。审核文件被替换后，同一已批准 plan 不必变化；精度审批亦有同类风险。`status=PASS` 和 `authorized_by=user` 字符串本身不能证明独立审核或真实用户同意。建议将具体文件 SHA 绑定入 plan、提交的 Runner inputs、控制器和训练启动时验证；须定向测试替换/缺失拒绝以及原有输入兼容性。此处未完成修复。
2. 预算身份范围：`production_control.py:119-131` 将身份与账本放在 `root`，第 133–148、186–196 行按本 root 的 attempt 累计，限制单阶段、组及总量并留清收余量。固定根目录下更改 plan 会被拒绝；现有证据没有证明相同实验换 root/task_id 后仍共享账本。不同 plan 的确需要不同授权 digest，不能据此断言能绕过真实用户授权，但尚未满足任务要求的跨身份预算防重。需要在不影响既存任务和不重置历史账目的前提下设计并验证稳定实验身份约束。
3. 旧 `FORMAL_PREPARED_NOT_SUBMITTED` 是测试草案，不是本轮候选：使用旧源码身份、1200 秒阶段占位、空 GPU UUID 及缺失审批。未生成新的完整正式 plan、SHA 或提交请求。后续需要完成审核修复与受影响函数定向回归，固定新源码、export、审批文件身份及实际资源计划后，才能冻结可执行候选。

## 已复核事实和局限

- 主控直接读取源码与历史记录；两名被指派审核者均反馈没有文件/执行工具，故 `NONAUTHOR_REVIEW: PENDING`。历史主控 `INTERNAL_REVIEW.json` 自述为主控检查，不能用作非作者通过。
- 生产路径 `closure_acceptance.py` 使用 `Control` 和生产执行/验证函数，外部计算由 `closure_fixture.py` 替代：人工构建合成 checkpoint、复制合成预测；不能证明真实正式 best Adapter 或真实 GPU 生成已产生。历史 [acceptance.json](evidence/production-closure/final-server/SYNTHETIC_CPU_INTEGRATION_ONLY_e9a3a6319296/acceptance.json) 报告真实 CPU Runner 和 26 条负例；历史 request/status/log 在同目录树。该证据证明范围不得扩张为正式训练正确性结论。
- 训练配置文件列出 seed 42、4 epoch、microbatch 8、GAS 4、world size 2、LR 0.0002、BF16、七个 LoRA 投影和 Prompt0/Beam20；配置 SHA `24b568161f2d71d3c5962372a38683b2d33ee5d6d5064de0612215946a8106fd`。历史 [method_invariance.json](evidence/production-closure/method_invariance.json) 给出源码对照，本轮未完成全部方法及生效路径独立审查，不宣称 METHOD_INVARIANCE PASS。
- `production_identity.py` 实现四轮记录、严格 best 和导出 Adapter/Tokenizer 检查；正式模型和正式结果目前不存在是正常事实。尚未完成真实完整来源与每项评价协议终审，不能声称通过。
- 未授权拒绝：历史 acceptance 记录缺失授权时 `PermissionError`；`production_control.py:74-83` 核对收据摘要、plan digest、版本/配置/预算/任务和过期时间。但用户消息是否真实提供 SHA 由操作人验证，不能由程序自证。文件内容漂移问题仍未修复。
- 控制器 root 内账本累加 attempts，训练及 Normal 共组、S3/S4 各自一组、其余为 CPU。预算为墙钟而非设备秒；清收余量从每次可用时间中扣除。完整 Runner 和断线证据尚需针对最终版本继续审查。
- 2026-11-16 00:50–00:51 UTC 只读查询 `zwlab-01`，UID 1005；GPU0 `GPU-dfff0e95-1db6-e2f4-3431-55bb852fa0ae`，GPU1 `GPU-59cac3bd-3645-668b-79c8-839e0d878893`，各 24564 MiB，总显存各使用 1 MiB、利用率 0%，无 compute app；当前 shell CUDA_VISIBLE_DEVICES 空。80 逻辑 CPU；RAM 247 GiB、可用 238 GiB，Swap 63 GiB、已用 145 MiB；/data 可用 2.6T、/home 所在 / 可用 549G。启动时重新检查。Python 环境版本 torch 2.6.0、deepspeed 0.16.9、transformers 4.31.0、accelerate 0.21.0、flash-attn 2.7.4.post1。Runner SHA `cc1a1e277faa9bc713a4d0d1b620248d47c398e8ed17d4ed3d7ab52805cc657e`，sidecar SHA `24e46291cbc2770833647364b66fc7cb1cfe80dd5c9b530465a0432dfc8a0c97`，现有 sidecar namespace 明确是 CPU-only，不能冒称正式部署。服务器 e9 export 清单 95 文件本轮逐个重新哈希，与清单一致；历史 Git 映射证据见 [RUN_VERSION.json](evidence/production-closure/final-server/releases/e9a3a631929600757b7a99b5763540c6dd9d2b99/RUN_VERSION.json)。
- 历史 Instruments train+Normal 墙钟 45322.209 秒（12.59 h），并非 Beauty 实测，见 [A0 实验记录](../letter/A0_NORMAL_FINAL_REVIEW_ENTRY.md)。旧未批准草案上限：train+Normal 345600 秒，S3 16200，S4 16200，CPU 3600，总 381600 秒（106 h），前三级双卡最大 756000 GPU-device-seconds = 210 GPU-device-hours，未扣清收余量。96 小时约历史实测的 7.63 倍，实际可能保留大量资源；这不是预计耗时或已批准预算。历史单 Shuffle 4 小时仅为上限，不能推出实测耗时。清收余量、逐阶段上限、预测耗时及实用推荐未完成冻结。

## 状态

```text
TECHNICAL_PRELAUNCH_AUDIT: FAIL
METHOD_INVARIANCE: FAIL
PROVENANCE_AUDIT: FAIL
RUNNER_AND_BUDGET_AUDIT: FAIL
NONAUTHOR_REVIEW: PENDING
FORMAL_PLAN_PREPARED: NO
FORMAL_PLAN_SUBMITTED: NO
FORMAL_BUDGET_USER_APPROVED: NO
FORMAL_READY_CANDIDATE: NO
FORMAL_LAUNCH_AUTHORIZED: NO
```

这里的 FAIL 表示本轮未达到完整 PASS 门槛，不表示已证实训练数学必然错误。未安装依赖或修改系统；本轮只做本地源码/记录读取与服务器只读查询，未计量 CPU 秒数，新增 GPU 使用为 0。原工作树未跟踪文件保留。
