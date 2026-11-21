# Beauty、Toys 五种子实验与注意力归档

本批 GRID/TIGER 实验使用 Qwen 基线的四位 SID，覆盖 Beauty、Toys 及训练种子 42、43、44、45、46。每个数据集和种子独立训练一个普通模型和一个 MLP 模型，共 20 次训练、40 组全量测试。

## 推荐指标

- [逐种子测试指标](per_seed.csv)：Recall@5、Recall@10、NDCG@5、NDCG@10，以及每组测试的 checkpoint SHA-256。
- [五种子汇总](summary.csv)：按数据集和模式报告均值与样本标准差（分母为 n−1）。
- [实验报告](shuffle-mlp-five-seeds.docx)。
- [原始结果与配置](runs/)：包含 40 份测试 JSON、20 份完成记录、启动覆盖参数、输入和源码哈希、模型及固定样本核对记录。模型 checkpoint 文件未纳入本次归档。

同一数据集、同一种子的 `unshuffled`、`shuffle_3`、`shuffle_4` 使用同一最佳 checkpoint。`shuffle_3` 与 `shuffle_4` 仅在推理时重排历史商品内部相应数量的 SID embedding，商品之间的顺序保持不变。MLP 是独立训练的模型。评估随机种子为 2026。全量测试指标与固定用户注意力分析分开计算。

运行 `python collect_results.py` 可以从归档中的原始 JSON 重新生成两个 CSV。实验标识：服务器 `experiments/sid_order_5seed_v1`。

## 注意力分析

- [完整交互包分片](attention/archive-parts/)：合并并解压后打开其中的 `index.html`。包含两个数据集、五个种子、四种模式；每个数据集和种子 100 位固定用户，共 1,000 个用户案例、4,000 份条件数据。
- [静态图](attention/figures/)：数据集总览、配对变化、SID 对齐及代表用户的编码器自注意力和解码器交叉注意力。
- [注意力汇总](attention/attention_comparisons.json)、[数值数组](attention/aggregates.npz)、[来源清单](attention/source_manifest.json)、[数据校验](attention/export_verified.json)、[离线校验](attention/offline_verification.json)。

交互包是经过校验的展示用导出。服务器原始 `.pt.gz` 张量仍保留在原运行目录，来源清单记录了文件路径与 SHA-256。交叉注意力基于各条件生成的 Top1 前缀，前缀不一致时不能视为控制前缀后的因果比较。

ZIP 大小：233,670,923 bytes。SHA-256：`9cdcd9bd4a7193b7e891f402fbe82b3defc5b597d6203be12f502d428f639272`。

分片为普通 Git 文件，不需要 Git LFS。将 [合并脚本](attention/reassemble_archive.py) 和完整的 `archive-parts/` 目录放在一起，或在已检出的本分支直接运行：

```bash
python "5seed Results/attention/reassemble_archive.py"
```

脚本会在 `attention/` 生成 `Beauty_Toys_attention_5seed.zip` 并核对原包 SHA-256。

## 实验实现、权重与原始注意力补件

[O1–O5主说明](delivery/README.md)与[20次运行索引](delivery/run_index.json)给出逐manifest核验的实际源码、启动/失败/进度/终态和模型配对。模型checkpoint与原始.pt.gz已补充到[可下载资产清单](delivery/assets.json)，原40组测试指标保持不变。原文“未纳入checkpoint”是首次归档状态，不再代表本次交付。
