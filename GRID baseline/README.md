# Archived GRID baseline

本目录保留已完成的 Qwen → RQ-KMeans → 学长普通 TIGER 基线。源代码、配置、数据、Qwen 权重传输分片、文本向量、SID、checkpoint 和历史输出均保留。
测试指标、论文参照与增减百分比见 [Baseline Results](../Baseline%20Results/)。

在本分支中，所有 `codex` 目录均按要求排除，因此先前放在其中的重测脚本、权重合并脚本和核验清单不在此目录。
这些辅助文件仍可在 [main 分支的原基线目录](https://github.com/woobowen/recommend/tree/main/GRID%20baseline/codex) 查阅。

数据和模型均使用 Git LFS；请参阅 [分支根目录说明](../README.md) 下载并还原 Qwen 权重。此目录中的模型权重分片与根目录的模型内容相同，也可按相同方式合并。
三个最佳 checkpoint 及对应 SID 位于各自的 `experiments/<run>/`，历史配置的绝对路径保留原样，跨机器使用需调整。

| 数据集 | 实验目录 | 最佳 checkpoint |
|---|---|---|
| Beauty | `beauty_rkmeans_qwen_20261010_012253` | `tiger_train_test/checkpoints/checkpoint_epoch=000_step=012500.ckpt` |
| Sports | `sports_rkmeans_qwen_20261010_093035` | `tiger_train_test_val1600_20261010_112259/checkpoints/checkpoint_epoch=000_step=009500.ckpt` |
| Toys | `toys_rkmeans_qwen_20261010_093036` | `tiger_train_test_val1600_20261010_112259/checkpoints/checkpoint_epoch=000_step=006200.ckpt` |

Beauty 的后期结果来自恢复最佳 checkpoint 全训练状态；Sports/Toys 从零重启推荐模型。后期均每 1,600 个训练 batch 验证一次。
