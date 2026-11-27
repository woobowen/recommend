# SID 语义层级实验

完整实验清单、方法、配置、结果与目录结构见[仓库首页](../README.md)。

- [共用 GRID 上游代码、SID、数据与配置路径](docs/shared-code.md)
- [复现命令与输入格式](docs/reproduction.md)
- [额外修改与未执行项目](docs/modifications.md)
- [结果明细和分母索引](docs/results.md)
- [六组 LLM2Vec 指标汇总](results/embedding_summary.csv)
- [自动分类主结果](results/classification/summary.json)

本目录只新增语义实验专用代码及统计，共用上游沿用已有 GRID 基线。归档的历史GPU入口含原服务器路径；可移植CPU入口位于 `scripts/evaluate_embeddings.py`，使用已归一化向量与SID即可重新统计。

- [Title＋Categories：Qwen/LETTER共同商品轴实验](title_categories_20261127/README.md)（2026-11-27，4主＋8补充分析）
