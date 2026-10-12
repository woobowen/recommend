# SID Order Invariance

- [All SID Experiments](All%20SID%20Experiments/)：本地「学长基线实验」上传内容，包括基线代码、shuffle、MLP、parallel 文件、配置、Amazon 数据和 Qwen 权重。所有 `codex` 目录均已排除。
- [GRID baseline](GRID%20baseline/)：已完成的 Qwen → RQ-KMeans → TIGER 基线归档，包含向量、SID、checkpoint 和历史输出。
- [Baseline Results](Baseline%20Results/)：三个数据集的测试指标、论文比较和增减百分比。

下载数据和模型请使用 Git LFS，具体步骤及 Qwen 权重合并方法见 [All SID Experiments/README](All%20SID%20Experiments/README.md)。
开始实验前请先进入 `All SID Experiments` 目录。变体实现与启动限制见该目录说明；名为 parallel 的文件仍与 shuffle 相同，不能据此认定并行 SID 解码已实现。
