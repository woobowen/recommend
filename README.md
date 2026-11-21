# recommend

本分支维护 GRID/TIGER 实验及对应历史产物。阅读[实验、执行证据与复现入口](docs/sid-extensions/EXPERIMENTS_AND_REPRODUCTION.md)，或查阅[仓库收尾、审核 ZIP 与复算](docs/delivery/2026-11-20/)与[现行协作规范](AGENTS.md)。


- [GRID baseline](GRID%20baseline/README.md)：本次 Qwen → RQ-KMeans → 学长 TIGER 基线的代码、完整数据、模型权重传输分片、向量、SID、checkpoint 和历史输出。
- [Baseline Results](Baseline%20Results/README.md)：三个数据集的最终测试指标，以及与 TIGER 原论文、GRID 默认配置和 GRID 各项最高指标的数值及相对百分比对比。

大文件通过 Git LFS 保存。使用前请运行 `git lfs pull`。Qwen 原始权重超过 GitHub 的单文件上限，下载后执行 `python "GRID baseline/codex/restore_qwen_weights.py"` 合并并校验。复用已生成 SID 或测试推荐模型时无需重新加载 Qwen。
