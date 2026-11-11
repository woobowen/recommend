# A0 正式运行前执行审核：NO-GO / BLOCKED

审核者：真实 Codex 子代理 `/root/a0_execution_review`，非正式训练入口、runner、恢复实现及比较器作者。本审核只写本文件；未修改训练代码、安装依赖、提交 Git、启动或终止 GPU 作业。

**结论：不得提交正式 A0。** 跨进程恢复确实加载了完整 DeepSpeed checkpoint 并继续执行，但在预先固定的数值标准下，模型、Adam 状态、FP32 master、loss 和 gradient norm 的一致性验收未通过。round 0 之后三轮集中修正已耗尽。进程退出 0 与正式运行前置条件通过是两个结论。

## 1. 直接读取的范围与固定身份

首次执行命令为 `ls -la`，工作树是 `/home/addaswsw/recommend-worktrees/lc-rec`，实际分支为 `wbw-lc-rec`。显式读取规则来源 `/home/addaswsw/recommend/{AGENTS.md,PROJECT_RULES.md,REMOTE_EXECUTION.md}`，未声称规则由同级目录自动发现。

本审核直接读取以下内容，而非仅阅读主控摘要：

- `scripts/formal_control.py`、`formal_pipeline.py`、`formal_runner.py`、`formal_ledger.py`、`formal_train.py`、`formal_compare.py`；同时检查 evaluator 的完成条件。
- 原有 accepted runner、`gridctl.py` 与 Windows SSH/SCP `transport.py`，以及已收集的实际 HF Trainer、Accelerate DataLoader/Accelerator 和 DeepSpeed engine 源码。
- 本轮两个 r3 作业的双 rank `events.rank*.jsonl`、`rank*.json`，服务器 `request.json`、`status.json`、`cleanup.json`、pipeline/resource 记录及 checkpoint `complete.json`。
- `evidence/formal_20261111/acceptance_comparison.json` 全文，`cpu_replay_round3.json`，`final_resource_and_ledger.json` 和 `corrections.json`。
- 无 GPU TERM-resistant 新 session fixture 的真实 runner/cleanup 回执。

最后实际 GPU 验收内容提交是 `87ba6556ce7237f8ef32c690165449d0d6a28294`，不是已完成正式 Normal 训练的 RUN_COMMIT。服务器源码导出位置为：

`/home/bwwu/lc-rec/letter-a0-normal-20261111/releases/87ba6556ce7237f8ef32c690165449d0d6a28294/`

审核者使用 `git show <完整提交>:<路径>` 独立重算并核对了记录的服务器部署清单中 10 项 formal 源码、配置及 precision approval 的大小/SHA；全部一致。此为读取服务器部署回执与固定 Git 字节的复核，审核者没有亲自重新执行服务器部署或实时 SSH 查询。

| 内容 | SHA-256 |
|---|---|
| formal_train.py | `e6616dbb8ece1b66db3a8500b3a84870b4cbbbf4b7d25bdd85e0f9f9d173076c` |
| formal_instruments_seed42.json | `9e058cecb7151ebee3ab714ab8328af0008537156cd96684c988d89e7ee10600` |
| A0 专用 runner | `d5a324e16f4ec36b3e9351bd275eb70ad301fdb6d4cb4e859994d3b0041471d7` |
| acceptance_comparison.json（1,014,752 bytes） | `1819c4b076e4e88414534fa32f450370ceb6f04aa3f17c876613ee4df785b4bf` |
| cpu_replay_round3.json | `9e54c92be806ece277a3962b72ea727c65a26ea0b3d4532f4f8dcd593f61095f` |
| final_resource_and_ledger.json | `4dfc1cd3566ae3ebefee9bbbdd1620b78ae97bcd614dd8527ebb0edd08059e5d` |

完整比较报告的下载回执在 `acceptance_comparison.json.receipt.json`，其远端路径、大小与 SHA 与上述实际本地字节一致。

## 2. 已通过的限定执行检查

1. 稳定 job_id、不可变 request、durable attempt claim 和文件锁沿用 accepted runner 机制；同 job 查询不会再次提交。正式 seed ledger 累计预算不按 attempt 重置，旧 attempt 仅依据 terminal runner 与 verified cleanup 封账，包含清理尾段。
2. 正式训练输出改为稳定 `formal_seed42/training`，避免跨 attempt 时 HF 的绝对 best checkpoint 路径落在旧目录，导致 `_sorted_checkpoints().index(old_best)` 失败。原缺陷已用实际 HF 函数 AST 只读复现。完成训练后的评价接续需双 rank 四轮/8240 step、源码/配置和 adapter hash 门槛，不重训较早 best 后的 epoch。固定训练目录的轻量证据已纳入 collect。
3. 正式 gate 在本地与服务器绑定 commit、config、training source，并作为 hashed runner input；pipeline 传播阶段非零退出。SUCCESS 路径要求训练完成报告、best selection 和完整 test 产物。该成功路径尚未完成真实全流程验收。
4. A0 runner 对新 session 的准入限于本任务命名空间、same UID、双 marker/先前证明的进程身份及有效启动时刻；发送信号仍使用 pidfd，重核 boot/PID/start/UID/PGID/session 和 known proof。无 GPU fixture 的 parent 3201337 与 escaped child 3201338 均拒绝 TERM，随后各收到 identity-checked KILL；最终 `TIMED_OUT/-9`、`cleanup.verified=true`、verified empty。这是故障 fixture 的预期 PASS，不是正式 A0 超时。
5. 外部 HF LambdaLR 单独完整保存/恢复，未挂入 DS engine 引起双步进。实际版本为 Accelerate 0.21.0、DeepSpeed 0.16.9。实例级 ZeRO-2 `no_sync` 适配保留原 backward/reduction/GAS/step/_do_sync 路径，限定 stage2、micro8、GAS4、world2、global64。r3 `inspect.unwrap` 后的 Accelerator 全文件 SHA、方法 SHA 和 `(self, model)` 签名均有实际 CPU 回执；未放宽源码 guard。
6. CPU 合成 DataLoaderShard 检查覆盖两 rank 的 mid-epoch 与 next-epoch，输入/动态 prompt/NumPy 路径 exact。它明确标记 `synthetic_cpu_only`，不能替代 GPU 完整状态验收。

## 3. 真正跨进程恢复已发生，但一致性未通过

| 项目 | 连续路径 | 恢复路径 |
|---|---|---|
| job_id | `letter-a0-20261111-acceptance-cont-r3` | `letter-a0-20261111-acceptance-resume-r3` |
| 真实 worker PID | 3202757 | 3203140 |
| 开始 UTC | 2026-11-11 17:59:37.942296 | 2026-11-11 18:04:07.473041 |
| 结束 UTC | 2026-11-11 18:02:44.670968 | 2026-11-11 18:07:00.629656 |
| 监督墙钟 | 186.728672 s | 173.156615 s |
| 进程结果 | SUCCEEDED / 0 / verified cleanup | SUCCEEDED / 0 / verified cleanup |
| 双 rank 最终报告 | PROBE_COMPLETE / global step 4 | PROBE_COMPLETE / initial step 2 / final step 4 |

原进程完整退出并清理后，新进程才启动。两 rank 各有一次严格 DS restore（step2）和一次 loader replay（8 batches）。`load_module_strict=True`、optimizer states、外部 scheduler 和 loader/RNG 恢复路径实际执行；不是 adapter-only 加载。

审核者用独立的标准库 JSON 读取与算术，重新复核原始事件：

- 连续路径每 rank 16 microbatches，恢复路径每 rank 8 microbatches。
- DS accumulation boundary 均严格按 `[false,false,false,true]` 重复；连续 DS 更新为 `[1,2,3,4]`，恢复为 `[3,4]`，before/after 步数符合原生边界。
- 恢复区间全部 8 个 batch 的 step、yielded、shape、input hash、label hash 完全相同；LR 序列也完全相同。
- 首个 LR0 boundary 主要初始化 Adam 状态；on_step_end 记录的是 scheduler 推进后的下一次 LR，不能据此称首步已经有效改变参数。恢复后的 step3/4 处于非零 LR 区间。
- 两 rank 恢复区间前 4 个 micro-loss exact，但下一次更新后开始出现差异。

固定判据为每个浮点元素满足 `abs(resumed-reference) <= 1e-6 + 1e-3 * abs(reference)`，离散状态、scheduler、step 与 RNG 要求 exact；本审核未扩大容差。

| 原始事件独立复算 | rank0 | rank1 |
|---|---:|---:|
| 8 个 micro-loss 中超限数 | 3 | 1 |
| 最大 micro-loss 绝对差 | 0.004361867904663086 | 0.004642486572265625 |
| step4 连续 grad_norm | 67.86307525634766 | 67.86307525634766 |
| step4 恢复 grad_norm | 64.69631958007812 | 64.69631958007812 |
| step4 grad_norm 绝对差 | 3.1667556762695312 | 3.1667556762695312 |
| step4 grad_norm 相对差 | 4.666389880369185% | 4.666389880369185% |

step4 gradient norm 的固定容差约为 0.0678640753，明显小于实际差值。完整比较器在 micro-loss 检查处抛错，因此报告中没有成功写出的 `events` 结果；上述 gradient norm、LR 和 batch 计数是本审核从原始事件另行复算，未误称报告已完成该分支。

## 4. 完整状态报告的精确分类

完整 CPU 比较报告为 `FAIL`，执行约 115.228 s，比较器退出 2。每 rank 载入的 payload 包含 450 个 trainable tensors / 289,652,736 元素，FP32 master partition 为 144,826,368 元素，两个 Adam moments 各完整覆盖该 partition。报告所检查的模型/optimizer 张量均 finite。

每 rank 的 454 项失败必须区分：

- **453 项浮点数值超限**：450 个 trainable tensors，以及 `exp_avg`、`exp_avg_sq`、FP32 master partition 各一项。
- **1 项比较器类型覆盖缺口**：`optimizer.zero_stage.__dict__.__objclass__` 为 `EnumMeta`，未被比较器支持。此项不能称为数值差异，也不能将其解释为 optimizer 状态已丢失。

| 超限元素数（完整报告） | rank0 | rank1 |
|---|---:|---:|
| Adam exp_avg | 710,242 | 153,828 |
| Adam exp_avg_sq | 28 | 2 |
| FP32 master partition | 144,681 | 46,414 |

报告中的 Torch CPU RNG、loader generator RNG、两 visible CUDA RNG 张量及 NumPy RNG array exact；未报告 Python RNG、scheduler 或 optimizer step 的不一致。不能因此推导全部训练状态或优化轨迹已通过：453 项数值超限和独立复算的 loss/gradient 差异仍然存在。

本审核没有在本地重新读取服务器上的巨大 `.pt` payload；完整状态张量结论来自直接阅读比较器源码、全部 JSON 比较结果和 payload hash 回执，独立复算则直接使用双 rank 原始事件。不能将差异唯一归因于 FlashAttention/GPU nondeterminism、量化缓存或 checkpoint 实现错误；现有证据尚未排除这些不同原因，也未证明某个确定根因。

## 5. 存储、预算与停止状态

实测 checkpoint-2 清单为 19 项 / **18,673,205,984 bytes**，含两 rank optimizer 文件、约 14.62 GB model states、约 0.579 GB adapter、外部 scheduler、两 rank RNG/loader 等状态。常驻 best+last 约 37.35 GB，保存轮换的三份暂态约 56.02 GB，仍须另留序列化、导出与预测余量。大 payload 本轮清单主要核大小；不能称其每个大文件都重新计算了 SHA。

根据直接读取的 2026-11-11 18:15:02.460641 UTC 资源回执，审核者独立求和五个 GPU 验收 attempt：监督墙钟累计 **606.164091 s**，按两卡保守计费 **1,212.328182 device-seconds = 20.2054697 device-minutes**，低于 60 device-minutes。此为两设备保留时间的保守计费，不是瞬时利用率积分。

该回执中无 formal ledger、无 `formal_seed42` 目录，双 GPU 均约 1 MiB、无 compute processes；数据盘约 3.231 TB free。审核者没有另行实时 SSH 重查，以上状态仅绑定该实际采样时刻。

三轮集中修正均有真实失败与源码记录：round1 外部 scheduler，round2 ZeRO-2 `no_sync`，round3 decorated source guard。第三轮两个进程路径虽然执行成功，最终固定数值验收仍失败；没有剩余的自主修正轮次。

## 6. 最终边界与交付建议

**执行预审：NO-GO。A0 NORMAL：BLOCKED。** 维持原始失败、checkpoint 与日志，完成 GitHub/审核 ZIP 的阻塞态交付，不启动正式四轮作业，不通过放宽容差、忽略 EnumMeta、改精度或新增 GPU 探针包装 PASS。

尚未验证/完成：正式四 epoch、完整作者 validation、实际 best checkpoint 选模及恢复、24,772 用户 prompt0/beam20 全量 test、真实 HR/Recall/NDCG 和论文指标重现程度。短探针的 epoch 约 0.00194175，不能写成完整 baseline。96 小时正式预算未开始。

本审核不替代论文方法审核，也尚未审核之后才会产生的 DELIVERY_COMMIT、GitHub 回读或 ZIP；这些应由主控在交付时按实际事实另行核验。不得据本文件声称最终交付已完成。本审核未启动 A1、Shuffle、额外 seed 或其他数据集。
