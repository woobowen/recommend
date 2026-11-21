# Instruments A0 Shuffle V2 第一组正式结果

原系列于 2026 年 11-21 05:43:14 EST 退出 0。四个新条件的正式生成与 CPU 评价、五项配对统计均已完成。每条件覆盖 24,772 名用户，保留 495,440 条候选。Normal 复用冻结双分片预测。内部终审通过，待网页二审。

[V2 方案、固定资产、有效配置与调用链](../../../../shuffle_v2/README.md) · [启动观察和预审](../launch-observation-20261120/README.md)

## 16 项指标

单位为比例，表内保留 6 位小数。完整精度见 [metrics.csv](metrics.csv)。

| 条件 | hit@1 | hit@5 | hit@10 | hit@20 | ndcg@5 | ndcg@10 | ndcg@20 | mrr@5 | mrr@10 | mrr@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Normal | 0.064508 | 0.090304 | 0.113354 | 0.138584 | 0.077663 | 0.085081 | 0.091466 | 0.073512 | 0.076551 | 0.078305 |
| S2 | 0.055062 | 0.089617 | 0.110326 | 0.135274 | 0.073208 | 0.079832 | 0.086148 | 0.067777 | 0.070468 | 0.072206 |
| S3a | 0.041054 | 0.072865 | 0.090707 | 0.111981 | 0.057816 | 0.063581 | 0.069004 | 0.052830 | 0.055206 | 0.056717 |
| S3b | 0.044607 | 0.081423 | 0.102575 | 0.128048 | 0.063865 | 0.070645 | 0.077118 | 0.058061 | 0.060824 | 0.062617 |
| S4 | 0.032456 | 0.065316 | 0.082432 | 0.103948 | 0.049826 | 0.055328 | 0.060799 | 0.044688 | 0.046938 | 0.048456 |

| 条件 | ItemHit@5 | ItemHit@10 | ItemHit@20 | ItemNDCG@5 | ItemNDCG@10 | ItemNDCG@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Normal | 0.089738 | 0.113031 | 0.138140 | 0.076830 | 0.084318 | 0.090670 |
| S2 | 0.089335 | 0.109761 | 0.134910 | 0.072449 | 0.078980 | 0.085350 |
| S3a | 0.072562 | 0.090344 | 0.111901 | 0.057283 | 0.063023 | 0.068514 |
| S3b | 0.081140 | 0.102172 | 0.127765 | 0.063167 | 0.069905 | 0.076402 |
| S4 | 0.064932 | 0.082311 | 0.103827 | 0.049399 | 0.054988 | 0.060454 |

## 五项配对比较

逐用户分母均为 24,772。五项比较共用 PCG64 有放回用户索引，seed 20261113，5,000 次，线性百分位 95% 区间。全部 80 行见 [paired.csv](paired.csv)，原配对汇总见 [summary.json](summary.json)，逐用户、逐配对及 bootstrap 数值见 [arrays.npz](arrays.npz)。

| 代表指标 | 比较 | 差值（百分点） | 95% CI（百分点） |
| --- | --- | ---: | ---: |
| Hit@5 | shuffle2-normal | -0.069 | [-0.218, +0.073] |
| Hit@5 | shuffle3a-normal | -1.744 | [-1.986, -1.498] |
| Hit@5 | shuffle3b-normal | -0.888 | [-1.090, -0.678] |
| Hit@5 | shuffle4-normal | -2.499 | [-2.761, -2.228] |
| Hit@5 | shuffle3b-shuffle3a | +0.856 | [+0.622, +1.086] |

S2 的 Hit@5 配对区间跨 0。S3a、S3b、S4 的 Hit@5 低于 Normal，S3b 高于 S3a。结论限于这组固定模型、用户轴及 seed=42 的置换实例。区间未做多重比较调整，无法单独识别 SID 层级或预训练的因果作用。

## 原运行、配置与核验

| 条件 | 生成起止（美东） | 单 GPU 墙钟 | 原始预测使用的评价覆盖 |
| --- | --- | ---: | --- |
| S2 | 11-20 11:48:44 EST → 11-20 16:11:31 EST | 4.380 h | [24,772 人，495,440 候选](evaluation/shuffle2/prediction_checks.json) |
| S3a | 11-20 16:12:42 EST → 11-20 20:38:00 EST | 4.422 h | [24,772 人，495,440 候选](evaluation/shuffle3a/prediction_checks.json) |
| S3b | 11-20 20:39:11 EST → 11-21 01:20:05 EST | 4.682 h | [24,772 人，495,440 候选](evaluation/shuffle3b/prediction_checks.json) |
| S4 | 11-21 01:21:14 EST → 11-21 05:39:47 EST | 4.309 h | [24,772 人，495,440 候选](evaluation/shuffle4/prediction_checks.json) |

原正式 GPU 合计 17.792/36 GPU-device-hours，最大单次 4.682/8 h。预跑 10.563/60 device-minutes。原 CPU job 累计 685.478/14,400 s，含清收的 CPU replay r0。新增终审和审核导出的耗时另记于相应证据。 [原终态及预算](terminal-and-budget.json)

四条件均为单张物理 GPU 1，逻辑 CUDA 0，`WORLD_SIZE=1`、rank=0、batch=1。V2 允许恒等。抽样键保留全局用户索引、history occurrence 和种子 42。S3a/S3b 共用相对排列。CPU 全量重放与正式预测对应，恒等用户仍纳入评价。冻结 Normal 沿用原两分片，无新增模型推理。 [CPU 输入基准](../cpu-replay-r1/audit.json)

- 正式 RUN `e2f54ebd2d7709821de1071693f91644f5ed9a30`，[服务器实际导出的文件映射](deployment/e2f54ebd2d7709821de1071693f91644f5ed9a30.json)。
- 串行控制 `4f5758ffb578f3a8d3d06a54262f59d3f63db370`，[控制部署映射](deployment/4f5758ffb578f3a8d3d06a54262f59d3f63db370.json)和[原阶段 ledger](controller-summary.json)。
- 实际 runner 的仓库源码是 [lc_rec/a1_qwen3_06b/scripts/runner.py](../../../../a1_qwen3_06b/scripts/runner.py)，SHA-256 `1fefc09a15c1cb73878417f9eb50d0fe9e8fa1b2f60cc70b27f7cc4ed9cb0947`。对应 `runner-a1-96h-1fefc09a15c1cb73.py` 部署。原客户端的 [gridctl](../../../../evidence/execution_source/gridctl.py)、[transport](../../../../evidence/execution_source/transport.py) 的既有仓库副本与实际导入文件逐字节一致。 [客户端来源与 SHA-256](execution-client/source-identity.json)
- 父作业 `letter-a0-20261112-v2-20261120-group-series-r0`。每个正式生成与评价均为 r0，无 GPU 重试。原 CPU replay r0 退出码 0 后因残存子进程被 runner 标为 `WORKERS_TERMINATED` 并清收，r1 复核通过。原件保留在 [jobs/](jobs/)。

此前 15 分钟为普通监控间隔。用户在同一会话随后明确授权一次 30 分钟观察，原话为“对已经实际启动的正式任务，从本次查询时刻 T0 起，观察30分钟，在以下七个时间点检查”，并要求交付后停主动监视。执行的是 T0 到 T0＋30 分钟的七个时点。 [当时七点原始回执](../launch-observation-20261120/README.md)

## 从保存输出复核

服务器原件位于 `/data/bwwu/lc-rec/letter-shuffle-v2-20261120/`，各条件的预测路径、分片和 SHA-256 在 `evaluation/<mode>/prediction_manifest.json`。外置权重、上游 SID 和 tokenizer 从生效配置及启动回执定位，访问依赖用户已授权的 `new-server` 通道。原评价、统计入口分别为 `lc_rec.shuffle_v2.evaluate` 和 `lc_rec.shuffle_v2.statistics`。完整审核 ZIP 内提供全用户候选与分数、SID 与历史轴、逐行输入与 prefill 证据以及复算入口。整组保持已审生成方法，追加步骤只读取保存预测。

## 全量审核导出与离线复算

单独的 CPU 审核作业 `letter-a0-20261112-v2-20261120-cpu-review-export-r0` 耗时 59.837 秒，退出 0，清收通过。导出代码 commit 为 `a9484a02bf71201bf4b954201c7368a1af361e65`，冻结配置来自正式 RUN。两项非作者服务端复核分别用时 41.79 秒和 33.16 秒。原 CPU 作业、本次复核和导出合计约 820.3 秒，低于 14,400 秒预算。 [导出作业原始回执](jobs/letter-a0-20261112-v2-20261120-cpu-review-export-r0/status.json)

单个审核 ZIP 收录五条件 2,477,200 条候选的 SID、顺序、生成 token 与 binary64 分数，完整用户轴、目标、SID–item 映射、四条件逐行输入与模型边界字段。导出保留每条候选及异常记录，省去重复解码 Prompt 前缀与常量字段，固定到服务器原始分片 SHA-256。独立从导出复算 1,981,760 个逐用户指标及 400,000 个 bootstrap 均值。ZIP 根目录列出离线复核命令。 [提取程序](../../../../shuffle_v2/export_review_data.py) · [全量导出清单](review-export-manifest.json) · [离线输入与指标复核](offline-export-verification.json) · [独立离线配对复核](offline-recompute.json)

## 真实非作者终审

- [运行终审](final-run-review.md)独立读取 99,088 条正式记录及 Normal 两个全量分片，复核实际输入、逐行模型边界回执、候选约束、清收与部署字节映射。S2/S3a/S3b/S4 恒等组数为 86,483/28,935/28,935/7,147，整用户自然恒等计数为 714/6/6/0。每条件的所有用户均纳入最终统计。
- [统计终审](final-stats-review.md)独立复算五条件 1,981,760 个逐用户指标值、400,000 个配对 bootstrap 均值及 160 个区间端点。逐用户及组均值完全一致，配对数值差异在 `1e-13` 核验阈值内。
- 审核以保存的输入、预测和模型边界记录为依据。终审未重新运行 GPU 生成，最终网页二审待进行。
