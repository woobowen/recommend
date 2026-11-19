# G3 Beauty 正式 Normal/OFF 进度同步

`new-server` 于 **2026-11-19 16:19:30.053996 UTC** 的实测显示，原 Normal/OFF attempt 0
运行正常。rank 记录及日志共同确认 **11,775 / 22,363** 名用户（52.65%），比上次观测的
3,000 名增加 **8,775** 名。当前无需用户处理。Normal 全量生成回执、全量 OFF 验收和
正式离线评价均待阶段收尾，S3、S4、paired、collect 尚待依赖阶段。
[本轮取证](observation.json) · [上轮快照](../20261119T131659Z/README.md)

## 生成进度和身份

| 已确认用户 | 生成循环已用时间（秒） | 约 UTC |
|---:|---:|---|
| 11,675 | 14,593.690 | 16:17:08 |
| 11,700 | 14,624.947 | 16:17:39 |
| 11,725 | 14,656.229 | 16:18:10 |
| 11,750 | 14,688.101 | 16:18:42 |
| 11,775 | 14,718.962 | 16:19:13 |

日志记单调时钟经过时间。表中 UTC 以最新 rank 报告文件的写入时间
`16:19:13.655867Z` 为锚点，按日志时间差换算，精度取秒。当前正式 argv 开启
`--verify-normal-off`。固定源码在每位用户的同进程 Normal/OFF 比较通过后增加计数，
每 25 人落一次进度。当前进度采用日志与 rank 报告的共同值。正式预测文件仍在增长，
末条完整候选记录含 20 个候选，完整文件哈希及全量覆盖留待该阶段终态。
[原生成日志](normal.work.tail.log) · [实际 argv](normal-compute-argv.json) · [生成 rank 摘录](normal-rank0-extract.json)

实际生成使用运行提交 `21d495b0be5b606853ec7cec68a3eaa1b58acbd7` 的 LETTER-LC-Rec，
beam 20，同一 `checkpoint-6160` 导出的 `best_adapter`。运行中记录的 adapter 模型 SHA-256 为
`d3a179b59ba7a7218d6e1767884fd1c5961fffb9b756ea5f8bc5522775db6a28`，
生成模型身份文件 SHA-256 为
`3315622866cd97833741608ca21d82b72a10184ab8bff4dc688892146d0bf915`。
严格加载记录含 448 个 LoRA 张量，完整 embedding 和 head 的形状均为 `[32786, 4096]`。
tokenizer 词表大小 32,786，词表和 G3 Beauty index、交互、商品三份资产身份均匹配原实验。
[生成身份](normal-rank0-extract.json) · [生产 adapter 验证](../20261119T131659Z/verify_best_adapter-stage_result.json)

## 阶段、资源与预算

| 当前账本阶段 | Attempt | Runner 状态 | 退出码 | 已用秒数 |
|---|---:|---|---:|---:|
| train | 0 | SUCCEEDED，已清收 | 0 | 64,854.723 |
| verify_best_adapter | 0 | SUCCEEDED，已清收 | 0 | 160.404 |
| normal | 0 | RUNNING | N/A | 15,067.117 |

训练仍为已完成的 8,214 次更新、4 个 epoch，训练及 adapter 阶段回执与上轮固定版本字节一致。
最优模型仍为 epoch 3 的 `checkpoint-6160`（validation loss 1.6667720079）。
最新完整保存点为 `checkpoint-8214`（epoch 4，validation loss 1.6976389885）。
本轮复核了保留 checkpoint 的在盘成员大小与清单身份，大权重内容身份沿用生产回执。
[已发布的训练终态](../20261119T131659Z/train-rank0-extract.json) · [本轮状态与回执哈希](observation.json)

当前作业为 `letter-a0-beauty-formal-g3-beauty-seed42-normal-a0`，请求 SHA-256 为
`e74b2fbcb1e0061e685d9a01aa90b7d2db65e62cee40968585dc1d9db306df38`。
`WAITING_OR_UNKNOWN` 和阶段字段 `SUBMITTING` 对应原控制器等待当前 RUNNING runner。
控制器 3334030、runner supervisor 3339348、worker 3339349、生成 rank 3339390 的
UID、boot ID、启动 tick 均与上轮匹配。日志末尾及 runner 检查无异常。
[原查询前账本](state-before-query.json) · [控制器 query](control-query.json) · [活跃 runner query](runner-query.json) · [进程启动时间](process-starts.txt)

- GPU1 `GPU-59cac3bd-3645-668b-79c8-839e0d878893` 使用 21,870 / 24,564 MiB，瞬时利用率 88%，
  rank 占用 21,860 MiB。主存可用 232.86 GiB，输出盘可用 2.46 TiB。
- 截至账本时间 `16:19:27.298607Z`，总预算 199,800 秒，已用 80,082.244 秒，
  剩余 119,717.756 秒。train/Normal 分组剩余 71,278.160 秒，Normal 阶段名义剩余
  28,132.883 秒，扣除 120 秒清收预留后距 runner 硬时限 28,012.883 秒。
  [资源及预算明细](observation.json)

## 固定任务与恢复入口

任务 `g3-beauty-seed42`，`SINGLE_GPU`，正式根目录
`/data/bwwu/lc-rec/g3-formal-20261118/21d495b0be5b`。
原计划文件 SHA-256、canonical digest、实际配置 SHA-256 与封存记录一致。
七份相关服务器源码和配置与固定运行提交逐字节匹配。原控制器继续推进现有流水线，
本轮状态同步结束。 [计划封存](plan.identity.json) · [实际查询命令](query-commands.json) · [冻结运行配置](https://github.com/woobowen/recommend/blob/21d495b0be5b606853ec7cec68a3eaa1b58acbd7/lc_rec/a0_beauty/config/formal_g3_beauty_seed42.json)
