# SID 顺序扰动：研究导航

这里组织同一个问题：在固定一条路线的模型和输入资产下，改变每个历史物品内部 Semantic ID（SID）码位的顺序，观察推荐输出。先读 [GRID 实现与结果](GRID.md) 或 [LETTER-LC-Rec 实现与结果](LETTER.md)，需要追踪具体运行时读 [21 条条件来源映射](sources.json)。这些链接是阅读入口；历史训练/生成仍由固定 commit 与原 request 定位。

| 实现 | 数据域 | 条件 | 状态 |
|---|---|---|---|
| GRID A、GRID B | Beauty | 各 Normal、S3、S4 | 六组正式测试 |
| GRID B | Toys、Sports | 各 Normal、S3、S4 | 六组正式测试；Toys A 来源未决、Sports A 配套缺失 |
| LETTER A0（LLaMA-7B）、A1（Qwen3-0.6B） | Instruments | 各 Normal、S3、S4 | 六组正式生成与评价 |
| LETTER A0 | Beauty | Normal、S3、S4 | 正式生成；原 paired 超时，S4 独立补充统计完成 |

这里是 21 个逻辑条件，不代表 21 次训练：各 Shuffle 使用对应 Normal 模型。GRID A/B 和 LETTER A0/A1 是独立编号；GRID 上游文本编码器 Qwen 与 LETTER A1 下游推荐器 Qwen3 不是同一模型。同属主题不意味着数据、SID、训练、模型、扰动分布或评价已经公平对齐，也不能据此将结果差异归因于单一因素。

## 仓库角色与其他研究

- `experiments/` 是 GRID 桥接及实验入口；`manifests/` 保存 job、配置、结果与审核索引；`references/wmh-original/GRID/` 为历史作者原件；`GRID baseline/` 是另一条工作基线及其资产入口。
- `lc_rec/letter/`、`lc_rec/a1_qwen3_06b/`、`lc_rec/a0_beauty/` 是 LETTER 路线；`lc_rec/shuffle/` 是 Instruments 干预和共享核心。作者原 LC-Rec 在 `lc_rec/vendor/original/`，LETTER 原件在 `lc_rec/letter/vendor/LETTER/`，两者分别保留身份。
- GRID 的 [三域 MLP 正式结果和固定样本 Attention](https://github.com/woobowen/recommend/blob/9767729a5114e21b9c949892af3d759baca6a98e/manifests/grid-formal-20261015/FINAL_RECEIPT.md) 分别是聚合研究及关联观察；[Parallel 历史资料](https://github.com/woobowen/recommend/blob/9767729a5114e21b9c949892af3d759baca6a98e/docs/sid-extensions/experiments/parallel.md)未恢复真实非自回归正文。
- 原 LC 六任务 [有界技术审核](https://github.com/woobowen/recommend/blob/12eb8b176b156f3be7d7533ee8f443db02441303/lc_rec/six_task_lora/REVIEW_ENTRY.md)、[offload](https://github.com/woobowen/recommend/blob/12eb8b176b156f3be7d7533ee8f443db02441303/lc_rec/offload/REVIEW_ENTRY.md)、[Qwen 单任务预跑](https://github.com/woobowen/recommend/blob/12eb8b176b156f3be7d7533ee8f443db02441303/lc_rec/qwen_preflight_20261110/REVIEW_ENTRY.md)及[六任务 Qwen 预跑](https://github.com/woobowen/recommend/blob/12eb8b176b156f3be7d7533ee8f443db02441303/lc_rec/qwen_six_task_20261111/REVIEW_ENTRY.md)是不同阶段，不能当成正式 LETTER 单任务结果。

Git 中保存源码、轻量配置与结果/回执；大模型、数据、checkpoint 和预测常为服务器引用，Git LFS 指针也不等于实体。本次文档工作没有现场重验这些资产。历史报告的内部审查、原任务状态和之后的外部审核按各自时间保留。文档所示指标来自既有结果，不重跑模型或重算统计。

2026-11-17 针对旧版外审的角色来源修正及检查范围见 [R1—R4 收尾依据](checks/CLOSURE_R1_R4.md)；该记录不代表新版网页外审已通过。
