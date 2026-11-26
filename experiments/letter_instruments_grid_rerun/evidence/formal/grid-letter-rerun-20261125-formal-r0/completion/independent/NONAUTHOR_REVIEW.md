# GRID／LETTER Instruments 正式终态独立审核

审核者为 `/root/independent_review`。本轮直接读取固定执行源码、原 job 终态、捕获原件、既有校验过的正式上游资料以及 checkpoint 只读提取，独立编写 `verify_completion.py` 并在 WSL 运行全量复算。未提交或启动计算作业，未改服务器和 Git。结论为 **PASS**，覆盖实际执行身份、终态、训练早停与选模、全量三条件候选、实际 Shuffle 输入、逐用户与聚合指标。

## 终态与内容身份

- `grid-letter-rerun-20261125-formal-r0` 以执行提交 `b08c63940f5351dce63dc4cb8a0f9c6f1d8ac86e` 启动。同一不可变请求的 canonical SHA-256 为 `bf0226675093e04ff777a76077cdd8173b627f977e6153a175b969faafae4eb1`。直接对比终态与上轮实际启动的 request、accepted、attempt、supervisor、worker 和 `started_at`。原 job 记录 `SUCCEEDED`、退出码 0、`NORMAL_EXIT_CHECK` 通过，独立查询的原工作组已清空。启动为 2026-11-25 13:36:38.616311 EST，结束为 2026-11-26 00:04:05.614412 EST。
- 8 个阶段回执齐全，按 `quantizer`、`sid`、`bound`、`training`、`Normal`、`Shuffle3a`、`Shuffle4`、`summary` 顺序实际执行，各为正式模式、退出码 0，均未复用其他 job 的阶段。各阶段产物摘要与终态清单相符，原工作日志保留全过程。
- 终态清单 85 件，读取原始终态导出和既有正式闭合文件重新计算其中 84 件的 SHA-256 及长度。与上轮 45 个共同的闭合文件清单逐项一致，证明正式 SID 和绑定产物由同一作业传到终态。第 85 件为留在服务器的原量化 checkpoint，原值 SHA-256 `864abd1e2028b6144f9300782c3f9bcea522ee106560dba21881cad84c21a6a0` 同终态、fit gate 和既有无损三层状态提取相符。上轮已用该提取及全部共同 embedding 独立复算 29,766 个正确中心指派，本轮继承这一内容核验。

## Normal 训练与最终选模

- 原生终态记录 6,700 次 optimizer update，展开 microbatch 计数直方图合计 107,200 批。67 轮完整验证从 global step 100 到 6,700，每轮 3,097 批，从独立核验过的 24,772 用户验证集读入，每批大小 8、末批 4。独立将每个验证回调的原始 Recall@5 与框架 CSV 行按顺序比对。可见日志和所有数值字段有限。
- 在全部验证回执内查得最大 Recall@5 `0.10083965957164764` 对应 global step 5,700。原日志随后记录连续十轮无改善而触发早停，末轮 global step 6,700 的 Recall@5 `0.10047634690999985`。`best.json`、阶段回执、实际 checkpoint 文件逐字节 SHA-256，以及 CPU 只读提取的回调与优化器内容互相印证。唯一最终 best 的文件 SHA-256 为 `64817119df777563fc8ca84f246099c96d35854d7de0620e38932d5221192087`，状态 SHA-256 为 `6ddd392a517717d51a0a875d2bf35aa2d456553fa9aa32bfff971fc4bce09883`。原提取核查 177 个状态键，127 项 Adam 参数状态的 step 均为 5,700。三个条件的实际 strict reload 回执都绑定此文件、状态摘要和正式 SID。
- 实际最终训练有效 YAML 与启动时已核对的训练 YAML 相同。三个测试的有效 YAML 与训练 YAML 在运行标识、输出路径归一后逐字段一致，均单卡 FP32、normal seed 2026、Beam10、原生配置 legacy shuffle 关停。条件差异由固定的测试入口和 `EncoderV2` 的作用域控制，训练和验证没有启用正式测试干预。

## 完整候选与实际 Shuffle

- 读完全部 74,316 个候选记录和 74,316 个逐用户记录。每条件 24,772 个用户按原绝对编号有序、唯一、齐全，3,096 个 8 用户批及一个 4 用户批。每用户有 10 个四码候选及 10 个原始 FP32 分数，每条件 247,720 个候选。逐个核对 canonical target、截断后有序历史、120 位置数据、mask、原 offset 序列和实际保存的干预序列。目录外候选数 Normal／Shuffle3a／Shuffle4 分别为 34／28／29，原排序保留。三条件均未观测到重复候选 SID、分数并列或 native 与稳定排序差异。
- 每条件 172,929 个实际历史分组。独立按 V2 seed 42、用户绝对行和截断后历史位置复算 SHA-256 抽样，逐组对比预先固定的历史排列参考、实际排列及移动后的完整 offset token。Shuffle3a 的 6 种排列和 Shuffle4 的 24 种排列均出现，含恒等。Shuffle3a 真实改变 143,994 组，第四位全部保持，Shuffle4 改变 165,782 组。两者对照 Normal 的有序候选 ID 行分别改变 23,873 与 24,107 行，原始 FP32 分数字节分别改变 24,766 与 24,772 行。
- 固定源码先核实际 lookup 接收到的原 offset 序列，在测试 encoder 的共享表 prehook 内送入重排 token，并要求同批次的 loss 和 generate 都完成同一计划。正式原件覆盖完整输入与候选；逐元素的 post-lookup embedding、SEP 和更深层激活探针属于此前 preview 的额外证据。此次独立复核依靠正式捕获和固定源码，没有重跑模型推理。

## 指标一致性与范围

从原始 FP32 分数向量、各 native top-k 索引和稳定排序分别独立重算全部 native Recall／NDCG、统一 SID Hit／MRR／NDCG、CCE ItemHit／ItemNDCG。每行的 native、稳定排序和所有逐用户指标均与原始 ledger 精确一致。全量聚合及相对本轮 Normal 的差值、分母与比例均与终态汇总精确一致，原生框架 FP32 聚合与 CPU 重算的最大绝对差为 `4.6290262872838994e-09`。目录的完整 SID 唯一，本轮且未出现重复候选，所以报表内对应 CCE 和 SID 数值一致。

| 条件 | 原始候选复算 Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| Normal | 0.0838850315 | 0.0685926056 | 0.1001937672 | 0.0738723627 |
| Shuffle3a | 0.0819877281 | 0.0668805055 | 0.0984579364 | 0.0722332607 |
| Shuffle4 | 0.0817858873 | 0.0662367819 | 0.0982560956 | 0.0715939237 |

Shuffle3a 的 Recall@5／NDCG@5 相对变化为 −2.261790%／−2.496042%，Shuffle4 为 −2.502406%／−3.434516%。完整未截断数值见 `non-author-completion.json`。原生 `test/loss` 从三个条件各自的 native CSV 和原生回执核对，Beam10 分数不含逐 token 的 teacher-forcing logits，故没有对该字段实施离线重算。结论适用于本轮已固定数据、一个 checkpoint 和一个 V2 seed，未做跨域或种子推断。
