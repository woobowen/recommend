# 非作者审核：复现准备 PARTIAL，原版 GPU 预跑 BLOCKED

审核者：`/root/independent_review`；本轮审查为 round2。此文件是一次性审核回执，不是持续维护的进度看板，不代表网页 GPT 二次审核。

独立核查的证据与状态边界可接受；当前没有发现必须通过更改研究方法、数据或额外模型运行来“修到通过”的理由。原版复现预跑仍未通过。

| 范围 | 结论 |
|---|---|
| 原生分支迁移、LC 创建基点 | PASS；直接审原生 Rename 请求/响应并独立读取真实 refs，三条 GRID SHA 均保持，旧真实 ref 消失。 |
| 官方原件、部署/实际 CPU 源码身份 | PASS；直接重新获取官方固定 Git tree，独立计算全部41份原件 Git blob；运行7项原件 SHA 与原件相符。 |
| 作者资产文件身份/物品轴/SID统计 | 文件身份 PASS；全部6服务器载荷对照原 ZIP 成员一致。9,922item、9,897 SID、23碰撞组、25额外item，930新增token独立复算一致。 |
| 辅助资产与当前 split 配对 | BLOCKED；从原 ZIP 另写只读计算独立重现 train12,233/test7,473错配。3,608个训练辅助目标为当前验证目标、1,623为当前测试目标，均未在对应训练prefix重复；最近20件实际可见历史仍有1,623包含当前验证物品。 |
| CPU 数据/监督检查 | PARTIAL；真实作业 exit2/FAILED，360检查中2项配对失败。六任务各3条真实监督样本通过，32验证仅collate，16用户×17模板只调用约束callback；未执行loss/生成。 |
| 原版双4090全参无offload | BLOCKED；基础Adam FP32两状态分片即25.10GiB/卡，超过实际23.99GiB，尚不含其余状态/激活。 |
| 模型梯度/更新/完整保存重载/beam20指标 | 未执行；不可由CPU或静态检查替代PASS。 |
| 正式候选、审核入口、规则最小更新、ZIP脚本 | 当前静态审查通过；正式配置保留官方条件和重要阻塞，打包源码已绑定HEAD字节，仍须真实发布/包闭环。 |

原始 CPU RUN_COMMIT 为 `53a0f618fc68dc2c936b4e5e10e1b5f0bbf282ad`；只读分类使用 `247adcee4ae881ce6df264b61da66db1d61c65ba`。主 CPU 作业68.374629秒，峰值RSS1,644,838,912 bytes，CUDA未初始化，0次optimizer update/0次生成。

结论只针对本发布交互数组和实际加载路径。没有原始timestamp或历史辅助生成manifest，不能据此声称查明论文历史run的泄漏原因。SID冲突和位置集合约束均保留；不静默补五码、改trie、删任务、改beam或重生成标签。

首选先向作者确认同一split/item轴的完整交互＋辅助资产；再解决无offload算力。单独换大GPU或开启offload不能修复资产配对。当前没有授权实际正式五seed、Qwen或后续研究任务。

最终 GitHub 固定提交/关键文件回读、交付源码与运行快照比较，以及真实ZIP/Windows可访问路径检查，在本回执写入时尚待主控完成。必须以随后的独立收尾核验和外部最终回执为据，不预写PASS；本文件不嵌入其自身最终commit。

详细证据、被审文件SHA与未决项见同目录 `final_review.json`，原始独立计算见 `cpu_runtime_evidence_review_round1.json`、`auxiliary_pairing_independent_round1.json`、`auxiliary_visible_input_independent_round1.json`。
