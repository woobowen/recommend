# 实际启动独立复核

审核者：`/root/fusion_launch_review`。本记录补充 `launch_review.md`，不修改运行中的冻结源码或旧审核身份。

实际 run：`/home/jcy/GRID_wmh_Qwen_TIGER_20261009/codex/title_categories_hierarchy_20261127/runs/20261127_014125`。2026-11-27 北京时间约 01:43，通过独立新 SSH 连接直接回读。

**启动核验通过，尚未完成全量实验。** Supervisor PID 709861，PPID=1；Beauty worker 710015、Instruments worker 710016 均指向该 run 的 frozen source，分别使用 GPU 0/1、2/3，进程存活并已有真实编码进度。观测时 Beauty 386/12311 独特字段文本，Instruments 386/10398；四卡显存分别 8430、7276、8430、7276 MiB，利用率 34–37%。该数字是观测快照，不代表此后实时状态。

实际输入准备检查通过：

| 域 | 共同商品轴 | 双字段主分析 | 单字段回退 | 无有效字段 | 独特字段文本 |
|---|---:|---:|---:|---:|---:|
| Beauty | 12101 | 12094 | 7 | 0 | 12311 |
| Instruments | 9922 | 9661 | 261 | 0 | 10398 |

两份实际 `encoding_preflight.json` 均 passed；重复同文本输出最大绝对差 0；同文本单条与 batch 的余弦分别 0.9999989867、0.9999986887；4096 维、非量化检查通过。worker 实际读取 text_manifest 而非包含 SID hash 的总审计，文本编码阶段没有读取 SID 表。

已定向重读运行中的 `source/encode_fields.py`，确认新增 padding 一致性与 mean/skip_instruction 断言仅加强核验，没有改字段融合/模型。实际 encode SHA 为 `e55819aa1cc97688110c29c8eb993f3ebd5ff2729463c676d83af6bd363c34a5`。本地先前审阅输出与部署当前正文差异来自并行准备期间改动；以本次服务器直接回读的正文与冻结 manifest 为本轮执行身份。

实际 `source/prepare_experiment.py` SHA 为 `61580fbae2f8d1e6ea4f4364e2859b48f92882d28964851a6ad6647fa94aa674`；已定向重读本地相同 SHA 修订，新增 Beauty `sid_column/grid_item_id/p5_item_id` 连续轴和 ASIN 唯一断言，并允许原文本换行的 DOTALL，合理。其余统计实现沿前审范围。

未验证：剩余全量向量、完整五项指标与覆盖率、999 次置换/2000 次 bootstrap、最终退出状态及 GitHub 发布。不能将本次启动通过称为最终实验完成或模型重评完成。
