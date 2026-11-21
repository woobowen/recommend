# SID Order Invariance

- [五种子实验与注意力归档](5seed%20Results/README.md)：Beauty、Toys，seed 42–46，含逐种子测试指标、汇总、原始结果记录及交互注意力压缩包。

- [All SID Experiments](All%20SID%20Experiments/)：本地「学长基线实验」上传内容，保留普通基线、早期 shuffle 快照、配置、数据与Qwen资产。早期 mlp/parallel 目录已从当前树删除；实际五种子 MLP 和 shuffle 实现在下方运行归档中。
- [GRID baseline](GRID%20baseline/)：已完成的 Qwen → RQ-KMeans → TIGER 基线归档，包含向量、SID、checkpoint 和历史输出。
- [Baseline Results](Baseline%20Results/)：三个数据集的测试指标、论文比较和增减百分比。

下载数据和模型请使用 Git LFS，具体步骤及 Qwen 权重合并方法见 [All SID Experiments/README](All%20SID%20Experiments/README.md)。
五种子运行以 [实际源码和运行导航](5seed%20Results/delivery/README.md)为准。`All SID Experiments`是历史工程快照，不是本批五种子完整入口。Parallel仅为历史自回归同源快照，没有新并行实验结果。

维护负责人：Jingchen Yin（Justin-hub-coder）。[2026-11-21 O1–O5补件](5seed%20Results/delivery/README.md)提供20个模型、4000份原张量、原运行源码与过程证据，不重新训练或改变指标。
