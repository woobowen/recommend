# 结果文件索引

## 六组LLM2Vec结果

- [未舍入汇总JSON](../results/embedding_summary.json)，[CSV](../results/embedding_summary.csv)。
- [Beauty/Toys完整汇总](../results/llm2vec_beauty_toys/combined_results.json)，运行 `20261114_125221`，2026-11-14完成。
- [Instruments完整汇总](../results/llm2vec_instruments/combined_results.json)，运行 `20261115_104723`，2026-11-15完成。

各组路径为 `results/llm2vec_beauty_toys/{beauty,toys}/{title,title_brand}/evaluation/` 和 `results/llm2vec_instruments/instrument/{title,title_brand}/evaluation/`。每组包含：

| 文件 | 内容 |
|---|---|
| `summary.json` | 指标、95%簇bootstrap区间、节点/边/路径分母、随机对照、p值、输入哈希 |
| `observed_summary.json` | 运行置换前保存的观察统计 |
| `prefix_nodes.jsonl` | 所有观察到的前缀、商品数、组内S和跨层统一频率排名；单商品S为null |
| `parent_child_edges.jsonl` | 全部边、商品数、S与Δ、是否真细分、是否父子集合相同 |
| `complete_paths.jsonl` | 全部第三层共享路径、各层商品数和S、是否两次真细分 |
| `top20_three_code_paths.jsonl` | 按第三层商品数选取的20条路径，包含父子集合相同的展示案例 |
| `branch_gaps.jsonl` | 每个有效父节点的分支Gap |
| `permutation_statistics.jsonl` | 两种条件零假设各999条统计，共1998条 |
| `semantic_refinement.png` | 各层S分布、父子增量分布、Top20路径演变 |
| `evaluation_complete.json` | 当时运行完成收据和summary SHA256 |

主实验分母：Beauty 1→2有效边2339、2→3有效边618、有效完整路径618；Toys分别2169、687、687；Instruments分别2033、574、574。完整路径要求两次都真缩小，覆盖率则按所有n≥2节点计算，两个范围不同。Toys第二层可评价节点2170，但有一条父子集合相同的边，所以有效1→2边是2169。

根目录表格的C3是精确库存描述；95%区间针对Δ、比例和Gap，不把置换分布的95%参考区间当作观察效应置信区间。Beauty/Toys两个标题主检验Holm校正记录在 `combined_results.json`；不要对标题＋品牌事后另选主结果。

## 自动分类 C1/C2

- [全部统计视图与随机对照](../results/classification/evaluation.json)。包括大类/细类、不同选组范围及UNKNOWN处理视图，不限README摘要。
- [12行主结果](../results/classification/summary.json)：两数据集×三个前缀长度×粗细两种粒度，所有共享组、known-only。
- [Beauty类别及计数](../results/classification/beauty/taxonomy_and_counts.json)：9大类、41细类；UNKNOWN405件。
- [Toys类别及计数](../results/classification/toys/taxonomy_and_counts.json)：122大类、183细类；UNKNOWN219件。
- 各数据集 `top20_examples.json` 保留高频组类别分布与商品示例。主结果是全体共享组，不用示例替代总体。

这次没有取回逐商品全量最终分类标签；分类表/计数、主指标及高频示例均已保存。重做分类需要按复现指南重新运行原流程，不能从总体计数反推商品标签。

## Beauty前缀树 T1

- [全部17980个前缀节点的类别和品牌分布](../results/beauty_prefix_tree/all_prefix_nodes.json)。
- [215、199、179三棵树及实际商品](../results/beauty_prefix_tree/selected_trees_with_products.json)。

根215体现彩妆内部唇部/面部的局部聚集，但有强品牌聚集；根199包含护肤用途分支；根179多数分支仍同标为Perfume，旧类别无法显出更细差异。这些案例是在扫描标签后选取的探索性例子，不是预先指定的随机样本，也不直接构成三级人类分类树证明。

## 核验与来源

[源码/文件SHA256清单](../provenance/source_manifest.json)、[冻结源码核对](../provenance/frozen_source_verification.json)。运行 `python semantic-hierarchy/scripts/verify_results.py` 可不加载模型，核对全部保存的节点/边/路径/Gap聚合、随机对照p值和README数值。它不替代从完整向量重新计算S；本次不包含约1GiB的LLM2Vec归一化向量。

历史完成核验另见 [Beauty/Toys](../results/llm2vec_beauty_toys/final_report_verification.json) 和 [Instruments](../results/llm2vec_instruments/final_verification.json)。
