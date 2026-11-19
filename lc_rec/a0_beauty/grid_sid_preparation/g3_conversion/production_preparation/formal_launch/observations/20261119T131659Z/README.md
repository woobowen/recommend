# G3 Beauty 正式单卡状态同步

**训练和最佳 adapter 验证均已完成，Normal/OFF 正式生成正常推进。** 服务器采样于
`2026-11-19T13:16:59.534145Z`。rank 记录和生成日志共同确认 **3,000 / 22,363 名用户**
（13.42%）。采样前最近五个进度点为 2,900、2,925、2,950、2,975、3,000，间隔约
30–32 秒。Normal 全量生成回执、Normal 评价、S3、S4 和最终配对汇总仍待该流水线推进。
[生成日志](normal.work.tail.log) · [rank 记录摘录](normal-rank0-extract.json) · [全量轴和进度取证](observation.json)

## 训练与选模

已完成 **8,214 / 8,214 次 optimizer update** 和 **4 / 4 epoch**，比上次记录多
4,424 次更新。四次验证各使用 44,726 条样本。最后一个定期训练日志点为 step 8,210，
loss 1.3221，LR 1.19427e−10，梯度范数 2.38831，时间为 11:31:07 UTC。
全程汇总 train loss 为 1.70750436。四个边界均有 epoch-end、验证事件和 checkpoint 完成回执。
[训练终态](train-rank0-extract.json) · [原始训练事件摘录](train-events.extract.jsonl) · [训练日志末尾](train.work.tail.log)

| Epoch | 累计更新 | Validation loss | Checkpoint |
|---:|---:|---:|---|
| 1 | 2,053 | 1.9063153267 | [回执](checkpoint-2053-receipt.json)，按保留规则轮转 |
| 2 | 4,107 | 1.7372068167 | [回执](checkpoint-4107-receipt.json)，按保留规则轮转 |
| 3 | 6,160 | **1.6667720079** | 保留，验证 loss 最优，[回执](checkpoint-6160-receipt.json) |
| 4 | 8,214 | 1.6976389885 | 保留，最新完整保存点，[回执](checkpoint-8214-receipt.json) |

训练以原 attempt 0 从 step 0 完成。训练结束后严格加载第 3 轮最优 checkpoint 并
导出 `training/best_adapter`。本轮核对了两份保留清单和在盘成员大小，对不超过
2 MiB 的成员复核 SHA-256，对大权重沿用生产验证回执。Normal rank 报告了严格加载的
450 个张量，包括 448 个 LoRA 张量和形状均为 `[32786, 4096]` 的完整 embedding 和 head。
已运行进程的 adapter、tokenizer、G3 数据身份与正式验证回执一致。
[选模](best_selection.json) · [adapter 身份](best-adapter-g3-identity.json) · [生产验证回执](verify_best_adapter-stage_result.json)

## 作业、进程和预算

| 阶段 | Job ID | Attempt | Runner | 退出码 | 账本用量（秒） |
|---|---|---:|---|---:|---:|
| train | `letter-a0-beauty-formal-g3-beauty-seed42-train-a0` | 0 | SUCCEEDED | 0 | 64,854.723 |
| verify_best_adapter | `letter-a0-beauty-formal-g3-beauty-seed42-verify-best-adapter-a0` | 0 | SUCCEEDED | 0 | 160.404 |
| normal | `letter-a0-beauty-formal-g3-beauty-seed42-normal-a0` | 0 | RUNNING | N/A | 4,104.285 |

两个完成阶段均通过清收核验。原控制器 PID 3334030 的 boot ID、启动 tick、UID 和运行命令
与上次观测一致。控制器汇总 `WAITING_OR_UNKNOWN`、阶段记录 `SUBMITTING` 是其等待
现有 runner 的状态。runner 实查为 `RUNNING`，监督进程、worker 和 GPU rank 都在运行，
PID 分别为 3339348、3339349、3339390。当前 Normal 使用 beam 20，并采用同一进程执行 OFF
精确一致性检查。已确认的进度点在对应 OFF 检查之后写入。全量 OFF 验收仍待阶段终态。
[查询前原账本](state-before-query.json) · [控制器查询](control-query.json) · [当前账本](state.json) · [当前 runner 查询](runner-query.json)

- GPU1 `GPU-59cac3bd-3645-668b-79c8-839e0d878893` 使用 21,870 / 24,564 MiB，
  瞬时利用率 41%，当前 rank 使用 21,860 MiB。主存可用 233.04 GiB，输出盘可用 2.46 TiB。
- 截至账本 `13:16:44.469366Z`，累计用量 69,119.411 秒，总剩余 130,680.589 秒
  （批准 199,800 秒）。train/Normal 组剩余 82,240.992 秒，Normal runner 距该阶段
  硬时限 38,975.715 秒，已计入 120 秒清收预留。
- 原控制器继续推进依赖链，当前无需用户处理。本轮到此结束主动监视。
  [资源、预算和请求身份](observation.json)

## 固定身份与恢复入口

任务 `g3-beauty-seed42`，`SINGLE_GPU`，batch 8、accumulation 8、global batch 64。
运行提交为 `21d495b0be5b606853ec7cec68a3eaa1b58acbd7`，
配置 SHA-256 为 `8e4c68c45b200d81624f25dd1c9bc204b5324f5da98e46ede6c68bc82a834853`。
两种计划摘要与冻结计划一致，七个相关服务器源码/配置文件与运行提交逐字节匹配，
三份 Beauty 资产的实测 SHA-256 也匹配原计划。正式根目录为
`/data/bwwu/lc-rec/g3-formal-20261118/21d495b0be5b`。
[冻结运行配置](https://github.com/woobowen/recommend/blob/21d495b0be5b606853ec7cec68a3eaa1b58acbd7/lc_rec/a0_beauty/config/formal_g3_beauty_seed42.json) · [计划封存回执](plan.identity.json) · [实际查询命令](query-commands.json) · [Normal 实际 argv](normal-compute-argv.json) · [上轮观测](../20261119T015346Z/README.md)
