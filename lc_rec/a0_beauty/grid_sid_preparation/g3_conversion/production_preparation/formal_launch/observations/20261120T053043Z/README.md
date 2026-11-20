# G3 Beauty 状态同步：paired 超时

**原流水线部分完成并已停止推进。** 2026-11-20 05:30:43.467451 UTC 的服务器取证确认，
训练、最佳 adapter 验证、Normal/OFF、S3、S4 及三项正式离线评价共八个阶段完成。
`paired` attempt 0 超时，`collect` 尚未提交。与上轮相比，Normal 从 11,775 推进到
22,363 名用户，新增 10,588 名，并已完成 S3、S4 各 22,363 名用户的生成和评价。
[本轮观测与阶段账本](observation.json) · [上轮观测](../20261119T161930Z/README.md)

## 停止原因与恢复边界

- 首个阻塞阶段为 `paired`。原 job `letter-a0-beauty-formal-g3-beauty-seed42-paired-a0`，
  request SHA-256 `93adddac739894f2c11331b1f25e160ac77955e4f487a9662c836489acebda59`。
  Runner 从 04:25:48.444839 UTC 运行至 04:43:48.877658 UTC，硬时限为 1,080 秒。
  状态 `TIMED_OUT`，退出码 `-15`；runner 记录向已确认身份的 worker 发送 SIGTERM，
  并核实工作组清空。原控制器随后报告 `failed stage requires explicit retry; downstream stopped`
  并退出。账本的原始 `WAITING_OR_UNKNOWN`、`SUBMITTING` 是其退出前留下的值。
  [原控制器日志](controller.log) · [原 runner 状态](paired.runner-status.json) · [本轮 runner 查询](runner-query.json)
- S3 配对文件已于 04:37:39.875164 UTC 落盘，覆盖 22,363 用户、16 项和 5,000 次 bootstrap。
  本轮重新读取保存的 `float64[5000,16]` 抽样均值，复算其中的线性百分位，与 S3 部分结果
  的全部已保存区间精确相符。超时时 S4 配对输出目录尚未创建，paired 阶段完成回执与
  collect 产物均缺失。该次 runner 的唯一计算日志是 S3 对应的 PASS 行。
  [S3 部分配对](paired-s3-partial-comparison.json) · [S3 原始 bootstrap 数组](paired-s3-bootstrap.npy) · [原 worker 日志](paired.work.log)
- 冻结实现每次配对前重新调用 `verify_prediction_stage`，其中经 `verify_training` 分别
  对两份保留 checkpoint 的清单成员做 SHA-256。一次验证涉及约 35.28 GiB 的 checkpoint
  载荷逻辑读取，另有预测身份和用户轴检查。由 S4 输出目录缺失可定位到 S4 结果写入前，
  日志没有精确到被终止时执行的函数。该结构是 CPU 阶段耗时风险，单个子步骤的实际耗时未采集。
  [冻结配对调度](https://github.com/woobowen/recommend/blob/21d495b0be5b606853ec7cec68a3eaa1b58acbd7/lc_rec/a0_beauty/production_pipeline.py) · [冻结 checkpoint 验证](https://github.com/woobowen/recommend/blob/21d495b0be5b606853ec7cec68a3eaa1b58acbd7/lc_rec/a0_beauty/formal_train.py)
- 总计费 **124,174.623 / 199,800 秒**，剩余 **75,625.377 秒**。CPU 组还剩
  **3,517.910 秒**，但 paired 单阶段上限 1,200 秒，已计费 1,080.433 秒并需留
  120 秒清收，按原公式可用 **−0.433 秒**，低于控制器的 1 秒准入条件。
  需要用户确认新的恢复授权。建议在独立 CPU 审计目录复用已在盘的预测和用户级评价产物，
  首次验证各输入身份并在同一审计中复用验证结果，沿原指标、种子和抽样次数补齐配对与收集。
  原封存账本和失败回执作为执行历史保留。
  [资源和预算](observation.json) · [原预算准入逻辑](https://github.com/woobowen/recommend/blob/21d495b0be5b606853ec7cec68a3eaa1b58acbd7/lc_rec/a0_beauty/production_control.py)

## 已完成阶段与产物

| 阶段 | attempt | Runner / 退出码 | 计费秒数 |
|---|---:|---|---:|
| train | 0 | SUCCEEDED / 0 | 64,854.723 |
| verify_best_adapter | 0 | SUCCEEDED / 0 | 160.404 |
| normal | 0 | SUCCEEDED / 0 | 28,369.489 |
| normal_eval | 0 | SUCCEEDED / 0 | 192.361 |
| s3 | 0 | SUCCEEDED / 0 | 14,522.128 |
| s3_eval | 0 | SUCCEEDED / 0 | 224.347 |
| s4 | 0 | SUCCEEDED / 0 | 14,546.193 |
| s4_eval | 0 | SUCCEEDED / 0 | 224.545 |
| paired | 0 | TIMED_OUT / −15 | 1,080.433 |
| collect | N/A | NOT_SUBMITTED | 0 |

九个已提交作业均已清收，同一任务的控制器和 worker 已退出。原 GPU1 采样为 **1 / 24,564 MiB、0%**。
固定任务 `g3-beauty-seed42`，实际运行提交 `21d495b0be5b606853ec7cec68a3eaa1b58acbd7`，
SINGLE_GPU`，batch 8、gradient accumulation 8、global batch 64。运行源码、有效配置、
原计划两种摘要、三个 Beauty 资产均已核对。正式根目录为
`/data/bwwu/lc-rec/g3-formal-20261118/21d495b0be5b`。
[实际请求、资源及原件位置](observation.json) · [查询前账本](state-before-query.json) · [原控制器查询](control-query.json) · [查询命令](query-commands.json)

训练完成 8,214 次更新、4 个 epoch。按验证 loss 选中第 3 轮
`checkpoint-6160`（1.6667720079），最近完整恢复点为第 4 轮 `checkpoint-8214`。
保留清单及在盘成员大小与上一轮相符，大权重身份沿用已验证的生产回执。
三种生成 rank 均绑定同一正式 adapter 和 tokenizer，20 beams。Normal 的同进程 OFF 实测
记录含 22,363 条 all-exact 检查，每个 row_idx 恰好一次。S3/S4 完成回执记录了
`GRID-LETTER-SHUFFLE-V1` 的 n=3/n=4 与 seed 42。
[已核对的选模回执](best-selection.json) · [adapter 生产验证](verify_best_adapter-stage-result.json) · [Normal 生成回执](normal-stage-result.json) · [S3 生成回执](s3-stage-result.json) · [S4 生成回执](s4-stage-result.json)

三份预测各为 22,363 用户和 447,260 个候选，生成与评价回执的预测 SHA-256 一致；
本轮核对预测文件在盘大小、三个 rank 报告 SHA-256、三个评价文件和各自 per-user
CSV/NPZ 的 SHA-256。无效完整 SID 候选按 Normal/S3/S4 分别为 507/32/51，重复候选
和完整 SID 碰撞均为 0。生产数据目录有 12,101 个 item 和 12,101 个唯一完整 SID。
[服务端预测与逐用户文件清单](observation.json) · [Normal 原始离线指标](normal-offline-metrics.json) · [S3 原始离线指标](s3-offline-metrics.json) · [S4 原始离线指标](s4-offline-metrics.json)

三组正式离线评价的全部 16 项与绝对差值见 CSV。指标及差值都用原始比例单位。
这是已完成阶段的生产输出。S4 配对、全流程独立复算、最终非作者审核和研究结论仍待恢复。
G3 的第四位为去重位。观察到的指标差异适用于本次固定权重和 Shuffle 实例。
[16 项指标及绝对差](condition-metrics.csv) · [冻结评价定义](https://github.com/woobowen/recommend/blob/21d495b0be5b606853ec7cec68a3eaa1b58acbd7/lc_rec/a0_beauty/offline_metrics.py)
