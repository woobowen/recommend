# 本轮补上传清单与状态

这是2026-11-23对已做实验的一次性整理，结果提交身份由最终交付回执提供；历史运行版本由原source_manifest/export hash定义。此前已有材料继续从旧固定版本和旧Release读取，不重复上传或改写。

| 实验 | 本轮处理 | 发布导航/边界 |
|---|---|---|
| Qwen Beauty/Sports/Toys baseline | 已发布，沿用 | yjc-grid固定6200d587及其历史G3来源限制 |
| Qwen Beauty/Toys五seed、shuffle/MLP/注意力 | 已发布，沿用 | 本分支旧5seed Results及36222920原件 |
| Llama分类、六组LLM2Vec层级分析 | 已发布，沿用 | sid-semantic-hierarchy固定371b4498及旧Release |
| LETTER公开Beauty SID＋自有TIGER | 本轮补齐 | [三域主说明](../LETTER%20SID%20%2B%20TIGER/README.md)，完整三测试；best5200/结束6200 |
| LETTER公开Instruments SID＋自有TIGER | 本轮补齐 | 同上；best5400/结束6400 |
| LETTER公开Yelp SID＋自有TIGER | 盘点发现漏传，本轮补齐 | 同上；best6500/结束7500 |
| Qwen Yelp aligned_v1上游 | 本轮补齐 | [独立处理轴](../Yelp%20Qwen%20SID/README.md)，向量/量化器/SID与数据处理原件 |
| Qwen Yelp patience10 seed43 | 本轮补齐 | 原五seed计划仅这一模型完成三测试；不是五seed完成 |
| Qwen Yelp patience40 seed43 | 本轮补齐诊断 | 单独保留较长patience、固定100用户分析，不替代原结果 |
| Yelp原其他九个计划任务 | 本轮补停止/取消记录和已有产物 | 用户停止4项、取消5项，未生成缺失结果 |
| Llama Instruments seqrec | 本轮补冻结代码/配置与采集时点观察 | [活动实验](../LETTER%20Llama%20Recommendation/README.md)，正式训练/测试未完成 |
| Llama Beauty原三任务 | 本轮补停止记录 | 用户2950步停止；已有失败容量/4用户测试是预跑而非完成 |
| Llama Beauty新seqrec/global64 | 本轮补冻结代码/配置与采集时点观察 | 从基座新训练，正式结果未完成 |

CPU核查15份保存测试与源码、三域SID/分片逐用户对应通过；本轮无模型forward、重评、训练、分类、编码。独立审查round0提出2项修正，任务级round1修复并核3套装配/配置断言、输入hash及指标表；原结果不改。外置载荷、远端ref/文件和正式交付包回读由包外最终回执记录，不预写网页GPT二审通过。

目录盘点边界是本人当前项目14个一级experiments目录与已知codex研究材料，不包括他人机器或账户；同实验多版本分类taxonomy属于尝试历史，不虚增独立实验数。此后上传继续按用户指南和本工作区AGENTS执行。本次不因发现未完成计划而重新启动已停止实验。
