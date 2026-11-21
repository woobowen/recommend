# SID Semantic Hierarchy

本分支整理已完成的 **Qwen SID 语义一致性与逐层细化实验**：Llama 自动分类（Beauty、Toys）、Beauty 前缀语义树回查，以及独立 LLM2Vec 编码评估（Beauty、Toys、Musical Instruments，各含标题与标题＋品牌两种输入）。实验代码、冻结配置和原始统计位于 [semantic-hierarchy](semantic-hierarchy)。

**上游沿用 GRID 基线的 Qwen → RQ-KMeans。** 已在其他分支发布的上游代码、数据、Qwen 权重、SID 和 checkpoint 不重复上传；共用代码的固定 GitHub 路径、实际运行配置见 [依赖与路径](semantic-hierarchy/docs/shared-code.md)。此分支继承的 `GRID baseline/` 和 `Baseline Results/` 是原有文件，本次没有重新提交这些文件。

## 1. 已完成实验清单

| 编号 | 实验 | 数据集与范围 | 状态/作用 |
|---|---|---|---|
| C1 | Llama 自动分类 → 前缀类别一致性 | Beauty，12,101 件；9 个大类、41 个细类 | 完成；检查共享前缀是否集中于同类 |
| C2 | Llama 自动分类 → 前缀类别一致性 | Toys，11,924 件；122 个大类、183 个细类 | 完成；类别构建有修订，见下文 |
| T1 | 前缀语义树回查 | Beauty 全部 17,980 个前缀节点；展示根前缀 215、199、179 | 完成；探索性案例，不是新的显著性检验 |
| E1 | LLM2Vec 逐层细化：标题 | Beauty，12,101 件 | 完成；主实验 |
| E2 | LLM2Vec 逐层细化：标题＋品牌 | Beauty，12,101 件 | 完成；补充实验 |
| E3 | LLM2Vec 逐层细化：标题 | Toys，11,924 件 | 完成；主实验 |
| E4 | LLM2Vec 逐层细化：标题＋品牌 | Toys，11,924 件 | 完成；补充实验 |
| E5 | LLM2Vec 逐层细化：标题 | Instruments，10,611 件 | 完成；新增数据集主实验 |
| E6 | LLM2Vec 逐层细化：标题＋品牌 | Instruments，10,611 件 | 完成；补充实验 |
| A1 | 上游向量归一化核查 | Beauty、Toys 的 Qwen 原始向量与量化配置 | 完成；机制核查，不属于语义效果实验 |

按正式“数据集×文本输入”划分，LLM2Vec 共 **6 组**。早期分类器的 v2–v7 调试不能算成额外独立实验。上述实验也不包含推荐 Recall/NDCG、shuffle、LETTER 下游实验。

## 2. 实验怎么做

### 2.1 类别一致性实验 C1/C2

使用 `Meta-Llama-3.1-8B-Instruct`，根据商品文本发现类别、全量草分类、修订并冻结类别、试标注、最终分类。类别数由模型决定，分类进程不读取 SID。**这里输入的是已有完整序列化文本 `Title + Brand + Categories + Price`，不是仅标题，也没有另行加入 description 或用户评论。** 模型看到 Categories，因此它不是排除原始类别信息后的独立语义验证。

随后按 SID 前 1/2/3 位分组，将冻结的大类和细类标签对应回商品，计算加权纯度、商品对同类率及覆盖率。主结果采用所有至少两件商品的共享组；另保存 n≥5、按层 Top-10/20/50/100 等历史补充视图。UNKNOWN 在主指标中排除，并另报排除前后覆盖率。全数据集已知标签置乱 1,000 次，保留 UNKNOWN 位置和各前缀成员。

### 2.2 独立连续语义空间实验 E1–E6

1. 保持 Qwen SID 不变。Beauty/Toys 直接复用 GRID 已有 SID；Instruments 使用同一上游协议新生成 SID。
2. 生成两个文本输入文件：`Title: ...`；`Title: ...\nBrand: ...`（缺失品牌省略）。解码 HTML 实体、合并空白；不额外输入 Categories、Price、SID 或旧分类标签。
3. 用 LLM2Vec **Llama 3 8B** 编码：匹配底座 → 合并 MNTP LoRA → 加载 supervised LoRA。输出 4096 维连续向量，再转 float32 做 L2 归一化。**没有将 LLM2Vec 向量做残差量化或映射到离散语言码本。** 编码进程只读 item_id 和文本，完成后评估进程才关联 SID。
4. 只分析前三位语义码；第四位是去重码。所有节点按不同商品数计频，与评论/交互次数无关。一、二、三位前缀统一排序的全量表见 `prefix_nodes.jsonl`；另按第三层商品数取 Top-20 完整路径作展示，不按相似度挑选。
5. 全部 n≥2 节点参与统计；主证据为第一层→第二层。第三层只评价至少两件商品的节点并报告覆盖率。父子集合相同不计作真细分边。
6. 两种父节点内条件置换各 999 次；第一层子树作为簇，bootstrap 2,000 次。Beauty/Toys 标题 Δ12 两个主检验做 Holm 校正；Instruments 是后来新增的一个主检验，不追溯合并检验族。

标题通常包含商品类型、用途和规格；加入品牌作为补充，检查趋势是否受品牌信息影响。标题本身仍可能含品牌，不能将标题实验称为完全排除品牌后的评估。

## 3. 主要指标定义

令 `z_i=x_i/||x_i||₂`，节点 G 含 n 件不同商品：

`S(G) = 2 Σ(i<j) z_iᵀz_j / [n(n−1)]`

用等价公式 `S(G)=(||Σz_i||₂²−Σ||z_i||₂²)/[n(n−1)]` 精确计算，排除自比较；单商品记 NA。主指标为组内平均两两余弦 S。历史报告要求的“五项展示指标”如下，另附 S 和 Gap：

| 指标 | 计算方式 |
|---|---|
| Δ12 | 对所有真细分 1→2 边，宏平均 `S_child−S_parent` |
| R+12 | Δ12>10⁻⁶ 的边数 / 有效 1→2 边数 |
| Δ23 | 对所有真细分 2→3 边，宏平均 `S_child−S_parent` |
| Rmono | 在 `n1>n2>n3≥2` 的路径中，`S2−S1>10⁻⁶` 且 `S3−S2≥−10⁻⁶` 的比例，即允许第二段数值持平 |
| C3 | 所有 n≥2 第三层节点覆盖的不同商品数 / 总商品数 |
| 分支 Gap | 对父节点，`同一子分支内商品对平均余弦 W − 不同子分支间商品对平均余弦 B`，再按有效父节点等权平均 |

真细分边要求 `2≤n_child<n_parent`。严格双段递增比例另存 `path_strict`。各层 S 是不同合格节点集合的宏平均，其相邻差**不等于**按父子匹配的 Δ。C3 衡量可评价范围，不衡量语义效果。

## 4. 实际配置与 GPU 分配

| 项目 | 类别一致性 C1/C2 | LLM2Vec E1–E6 |
|---|---|---|
| 模型 | Meta-Llama-3.1-8B-Instruct | Meta-Llama-3-8B-Instruct + MNTP + supervised LoRA |
| 模型更新 | 不训练参数，使用冻结模型生成分类 | 不训练参数，使用已公开适配器提取向量 |
| 权重精度 | FP16，无 4/8-bit 压缩 | FP16，无 4/8-bit 压缩 |
| 注意力/池化 | 因果生成；JSON Schema 约束输出 | 双向 SDPA；正文 mean pooling，排除指令 |
| 长度/批量 | 上下文 8192；分类输出≤1024；每请求≤20件，按 token 预算减小 | batch=2；max_length=256；doc_max_length=224 |
| 随机种子 | 20261109 | 20261113 |
| 统计 | 标签置乱 1000 次；24 项主检验 BH 校正 | 条件置换各999次；簇 bootstrap 2000次；epsilon=1e-6 |
| 硬件 | 4×RTX 2080 Ti，单模型用 GPU0/1，串行处理数据集 | 4×RTX 2080 Ti；一个 FP16 模型跨两卡 |

Beauty/Toys 正式运行 `20261114_125221`：GPU0/1 跑 Beauty、GPU2/3 跑 Toys；各自依次编码标题和标题＋品牌，统计在 CPU。Instruments 正式运行 `20261115_104723`：GPU0/1 跑 Qwen、GPU2/3 跑 LLM2Vec；Qwen 完成后 GPU0 跑 RQ-KMeans，最后 CPU 统计。以上是历史分配，不表示本次上传启动了任务。

LLM2Vec 环境：PyTorch 2.6.0+cu124、LLM2Vec 0.2.3、Transformers 4.44.2、PEFT 0.12.0、tokenizers 0.19.1。早期分类环境的 Transformers 为 4.47.0，与 LLM2Vec 环境分开。依赖记录和安装说明见 [复现指南](semantic-hierarchy/docs/reproduction.md)。

## 5. 结果

### 5.1 LLM2Vec 六组结果

| 数据集 | 输入 | Δ12 | R+12 | Δ23 | Rmono | C3 |
|---|---|---:|---:|---:|---:|---:|
| Beauty | 标题（主） | +0.063718 | 79.22% | +0.052267 | 77.02% | 18.99% |
| Beauty | 标题＋品牌 | +0.065174 | 79.48% | +0.052395 | 77.18% | 18.99% |
| Toys | 标题（主） | +0.069007 | 80.82% | +0.047007 | 76.13% | 21.58% |
| Toys | 标题＋品牌 | +0.066546 | 81.33% | +0.043391 | 76.56% | 21.58% |
| Instruments | 标题（主） | +0.069748 | 81.11% | +0.060172 | 82.58% | 21.34% |
| Instruments | 标题＋品牌 | +0.070672 | 81.60% | +0.059555 | 83.62% | 21.34% |

| 数据集 | 输入 | S1 | S2 | S3 | Gap12 | Gap23 |
|---|---|---:|---:|---:|---:|---:|
| Beauty | 标题 | 0.726704 | 0.787629 | 0.863780 | 0.090054 | 0.071789 |
| Beauty | 标题＋品牌 | 0.729673 | 0.791054 | 0.869008 | 0.091665 | 0.073222 |
| Toys | 标题 | 0.708274 | 0.775225 | 0.850173 | 0.092268 | 0.064642 |
| Toys | 标题＋品牌 | 0.723918 | 0.792199 | 0.865476 | 0.091599 | 0.060843 |
| Instruments | 标题 | 0.732279 | 0.796865 | 0.885105 | 0.088523 | 0.084906 |
| Instruments | 标题＋品牌 | 0.739691 | 0.804225 | 0.892910 | 0.089527 | 0.084495 |

上述表格以 [embedding_summary.json](semantic-hierarchy/results/embedding_summary.json) 的未舍入数值为准。全部节点、边、路径、随机分布、置信区间、图及完成核验保存在各运行目录，见 [结果索引](semantic-hierarchy/docs/results.md)。

六组 Δ12 的父节点内条件置换单侧 p 均为 0.001（999 次置换的分辨率下限）。预先指定 Beauty/Toys 标题 Δ12 的 Holm 校正值均为 0.002。Instruments 标题 Δ12 主检验 p=0.001。不能把 p=0.001 写作零概率。

### 5.2 自动分类主结果：全部共享组、排除 UNKNOWN

| 数据集 | 前缀长度 | 细类加权纯度 | 细类商品对同类率 | 大类加权纯度 | 大类商品对同类率 |
|---|---:|---:|---:|---:|---:|
| Beauty | 1 | 59.48% | 47.74% | 76.47% | 67.35% |
| Beauty | 2 | 72.70% | 54.59% | 83.96% | 70.20% |
| Beauty | 3 | 77.00% | 57.51% | 85.29% | 71.00% |
| Toys | 1 | 45.28% | 30.53% | 50.26% | 35.88% |
| Toys | 2 | 63.15% | 40.19% | 67.40% | 47.19% |
| Toys | 3 | 69.36% | 44.37% | 72.99% | 51.19% |

Beauty UNKNOWN=405/12101；Toys UNKNOWN=219/11924。分类结果反映与自动标签的一致性，不能当作人工分类准确率。前缀变长时样本覆盖和组规模同时变化，纯度上升本身不能证明逐层细化。

### 5.3 覆盖与解释范围

| 数据集 | 第一层覆盖率 | 第二层覆盖率 | 第三层覆盖率 | 真细分1→2边 | 真细分2→3边/有效完整路径 |
|---|---:|---:|---:|---:|---:|
| Beauty | 99.88% | 61.39% | 18.99% | 2339 | 618 |
| Toys | 99.88% | 63.78% | 21.58% | 2169 | 687 |
| Instruments | 99.92% | 61.22% | 21.34% | 2033 | 574 |

结果支持“现有 SID 在独立 LLM2Vec 文本空间中存在语义逐层细化与分支区分”，不能直接证明唯一的人类类别树。第三层结论仅适用于可评价范围。Qwen 上游已接触 Categories；独立编码器不等于训练语料完全独立。

## 6. 额外修改与未执行项目

- **沿用的上游**：Qwen2.5-7B-Instruct，3584维；Title/Brand/Categories/Price；右填充、最多128 token、原 WMH MeanAggregation；RQ-KMeans 3层×256、30步、seed=42、`normalize_residuals=true`，第四位去重。上游源码和配置直接链接 GRID 基线，未另造上游实现。
- **分类流程修订**：为避免重复/缺失 JSON 项增加 Schema 约束、结构重试和失败批次拆分；Toys 改为两遍逐项功能类型归一，并用“列名＋行值数组”无损压缩完整类别表以适配8192上下文。Beauty 保留 v1 标签。两数据集类别构建协议不完全相同，详见 [修改记录](semantic-hierarchy/docs/modifications.md)。
- **LLM2Vec 两卡兼容**：显式均分32层到两卡，保留官方包装、正文 mask 与池化，直接 forward；不使用会复制完整模型或移到单卡的 `encode()` 路径；关闭可选 bitsandbytes 派发。
- **Instruments 数据适配**：Amazon Reviews 2018 官方 Musical Instruments 5-core 和元数据；匹配标题/元数据后保留10,611件、231,312条交互；缺元数据9件排除。没有追加迭代5-core过滤，不等同于 LETTER 作者 Instruments 商品范围。
- **未执行**：LLM2Vec 的 n≥3/5/10 门槛敏感性、unsup-SimCSE、同文本去重重跑、Qwen连续向量语义对照。已下载某个适配器不表示跑过该实验。早期分类保存的 n≥5 视图不属于这些未执行的 LLM2Vec 扩展。
- **本次发布**：历史计算源码与 launch 收据哈希逐一核对；新增可移植 CPU 评估入口、结果重汇总与校验工具。没有重训模型或改写已有实验结果。

## 7. 目录结构

```text
README.md                              # 实验清单、方法、配置、主要结果
semantic-hierarchy/
├── archive/                           # 历史运行源码，保留原服务器路径和配置
│   ├── classification_beauty_v1/       # Llama 分类、提示词、统计
│   ├── classification_toys_v7/         # Toys 收敛及上下文适配版本
│   ├── classification_control/         # 串行控制器及 Beauty 树回查脚本
│   ├── llm2vec_beauty_toys/            # Beauty/Toys 冻结源码
│   └── llm2vec_instruments/            # Instruments 数据与编码适配
├── scripts/                           # 可移植 CPU 评估、汇总与校验入口
├── docs/
│   ├── shared-code.md                 # 其他分支的共用代码和配置精确路径
│   ├── reproduction.md                # 输入格式、环境、命令和复现限制
│   ├── modifications.md               # 额外修改及未执行项目
│   └── results.md                     # 完整统计、案例及分母索引
├── results/
│   ├── classification/                # 两数据集分类表、计数、全部统计视图
│   ├── beauty_prefix_tree/            # 全部前缀分布及三个探索性分支
│   ├── llm2vec_beauty_toys/            # 20261114_125221：配置、明细、图和核验
│   ├── llm2vec_instruments/            # 20261115_104723：配置、明细、图和核验
│   └── embedding_summary.{json,csv}    # 六组未舍入指标
├── provenance/                        # 来源/代码SHA256、模型及输入核验
├── audits/qwen_normalization/         # 归一化核查代码与证据
├── requirements-metrics.txt           # CPU统计最小环境
└── requirements-llm2vec.txt            # 独立编码环境
```

首次发布（e8f4603）仅附实验代码、配置与统计，当时未附全量连续向量。现行交付已在[Release资产清单](semantic-hierarchy/delivery/assets.json)发布六组完整LLM2Vec向量、同序ID和norms；SID按item_id关联，见[复现指南第2节](semantic-hierarchy/docs/reproduction.md)。普通Git保存源码、标签、SID/文本输入、收据和统计；LLM2Vec/Llama模型权重仍需按[来源与许可](semantic-hierarchy/docs/shared-code.md)自行获授权取得，不在此次Release中。Beauty/Toys原TFRecord及Qwen资产沿固定GRID入口，Instruments固定SID见delivery/encoding_inputs和results/llm2vec_instruments/inputs。

## 原有基线归档入口

- [GRID baseline](GRID%20baseline/README.md)：本次 Qwen → RQ-KMeans → 学长 TIGER 基线的代码、完整数据、模型权重传输分片、向量、SID、checkpoint 和历史输出。
- [Baseline Results](Baseline%20Results/README.md)：三个数据集的最终测试指标，以及与 TIGER 原论文、GRID 默认配置和 GRID 各项最高指标的数值及相对百分比对比。

大文件通过 Git LFS 保存。使用前请运行 `git lfs pull`。Qwen 原始权重超过 GitHub 的单文件上限，下载后执行 `python "GRID baseline/codex/restore_qwen_weights.py"` 合并并校验。复用已生成 SID 或测试推荐模型时无需重新加载 Qwen。

## 完整向量、标签与过程补件

维护负责人：Jingchen Yin（Justin-hub-coder）。[2026-11-21 H1–H4补件](semantic-hierarchy/delivery/README.md)补齐六组原向量/ID的下载、全量最终标签、readiness与原始请求/重试记录；CPU从原向量重算全部合格节点，与发布统计一致。未重新分类、编码或训练。

2026-11-22收尾：验证器分类/SID分组实算与向量回执检查已明确分开，详见[覆盖说明](semantic-hierarchy/delivery/README.md)。本轮未重新编码或计算六组向量余弦。
