# B：源码与可讲解性初次审查关注点（round 0）

审查者：真实子代理 `/root/audit_code`。本记录作者不是本阶段源码发布或阅读说明的作者。

状态：`SOURCE_READ_PRELIMINARY / DOCUMENT_CANDIDATE_PENDING`。本记录不授予本阶段 PASS；固定 baseline、本次从 new-server 档案提取的发布副本和主控阅读资料尚待正式逐项核对。

## 范围与方法

首条命令为 `ls -la`；`/home/addaswsw/recommend` 实际有 `grid-original`、`wang-grid`、旧接手文本及 PDF/PPTX，未见 `.git`。只读检查了目录链候选 `AGENTS.md`，无实际文件；遵守用户在对话给出的规则及本次明确实施授权。

已阅读旧阶段 `reports/A800_MECHANISMS.md`、`EXPERIMENT_MAP.md`、`BASELINE_RECOVERY.md`、`GAP_AND_RECONSTRUCTION.md`、`DIFF_REPORT.md`。对旧阶段 WSL 保全原件直接读取源码原文、用 `python3 -I -B`/标准库 `ast` 提取符号范围和 `hashlib` 重核文件身份；没有导入来源项目、解析 checkpoint、运行绘图器或 Notebook。该旧副本只作独立对照依据，本阶段原生发布副本必须按主控的 new-server 来源映射再核字节。

旧副本根：`/home/addaswsw/recommend-support/wmh-a800-audit-20261010/preservation/A800-grid-20261010-v1/restored/A800_GRID`。

## 直接原文身份

|路径（相对原件根）|SHA-256|已直接核对的符号/行范围|
|---|---|---|
|`src/models/modules/semantic_id/tiger_generation_model.py`|`c61606b4a03d6fc8cfa6cb3bca093a36bbf9a1846334d39cb8dbcb67b32c296e`|AST：`eval_step` 374–402；`encoder_forward_pass` 584–659；`model_step` 900–953|
|`src/models/modules/semantic_id/tiger_generation_model_shuffle.py`|`e03082ad87bfc842c05467580fc479eff4345c6a0d18c5b2f7faf320d4f96c80`|`_shuffle_hierarchical_embeddings` 83–149；attention保存/eval 446–575；hooks 603–631；构造/encoder/AR generate 642–1061|
|`src/models/modules/semantic_id/tiger_generation_model_parallel.py`|`e03082ad87bfc842c05467580fc479eff4345c6a0d18c5b2f7faf320d4f96c80`|完整哈希与 shuffle 同；AST 同范围。其同字节正文仍为 AR，不是恢复的 NAR|
|`src/models/modules/semantic_id/tiger_generation_model_mlp.py`|`1b75bf94d6bd1187e7c30f996da33ebdc7ea2ffb33d8918a123991abbadafab9`|构造/encoder/decoder/AR 471–898；共享表 934–955；采集 957–1048；loss 1050–1103|
|`src/data/loading/components/pre_processing.py`|`a69c88681760df22e37fb7e66961ad97ed0824af20eae415a543379e881d71c7`|映射/裁切 120–194；整数 row shuffle 321–366|
|`src/data/loading/components/collate_functions.py`|`a4692ce68705ea6a10ed500245f5c5acc99468f7e5d3f2d41e1ff0970fa9a73c`|普通 collate 188–279；整数 input shuffle 282–345|
|`src/models/modules/base_module.py`|`7cfa9da5f73c0e666ab81f22440d04f04d6cc9624803de58cde3256dcae400b0`|构造 16–56；优化器 168–189；test 206–218|
|`src/inference.py`|`f24892ed2d9777c6fd5d3eaf20cc0ea3d180a19a490519baa7ca2390a70c05ba`|完整 1–171 原文|
|`visualize_complete_decoder_attention.py`|`531bbaf5ab118ffedb3b0fef32dd0fe26748298b64e3c7e92a68c911c673e177`|构矩阵 20–149；实际入口 230–319|
|`visualize_encoder_attention.py`|`25622305bddb1d93e93d960ee8263a26e2ca8f601e6ab6be13ad718af0e380c4`|裁切 16–42；取层/head 45–67；入口 200–269|

## 正式文档审核必须覆盖的关注点

1. **SOURCE_FACT / Shuffle 参数结构**：shuffle 构造器创建 encoder/decoder 两张独立 `HV×E` 表；`shuffle_embedding_on_test` 并不是整个文件相对固定 main 的唯一变化。固定 main 共享关系尚待取本次实际 baseline 直接比对。不能把整类替换描述成无参数纯扰动。
2. **SOURCE_FACT / Shuffle 位置与 gate**：先 position offset、SID 表 lookup，再于 encoder 里重排向量（778–818），再 SEP/user。`is_testing` 只由 `on_test_start/on_test_end` 切换（619–627）；`model.eval()` 或 predict hook 不等价。`eval_step` 先 `model_step` 再 `generate`，每次 encoder 都可独立抽排列。不可整除 H 时原样返回；整物品零 mask 不变不保证部分有效 mask 安全。没有排除恒等排列。
3. **SOURCE_FACT / 三类 shuffle**：row 函数打乱原整数、没有自身阶段 gate/padding 过滤；collate 先普通标签构造再打乱输入整数、仅跳过整 padding 组；两者之后按新位置 offset，和查表后向量重排不是同一干预。需分别解释 label、位置码本身份、边界和 mask。
4. **CONFIG_DECLARATION / shuffle YAML**：`configs/experiment/tiger_inference_flat_shuffle.yaml` 明确 `_target_` 仍为普通模型，同时传 shuffle/save_attentions 专用参数；普通模型及其 SemanticIDGenerativeRecommender 构造器不消费这些参数，会转交闭合签名的 `TransformerBaseModule.__init__`（20–34，不含任意 kwargs），构成静态不一致。这里只作静态检查，不声称本阶段已运行触发异常。
5. **SOURCE_FACT / 当前 inference 不是 test**：`src/inference.py` 28–41 实际执行 `trainer.predict`；149–154 的 `trainer.test` 已注释。该入口还包含 `CUDA_VISIBLE_DEVICES='1,3'` 和 RotationSim 后处理。材料不能抄其 test 注释当实际执行，也不能称经路径映射即可运行；本阶段不修改它。
6. **SOURCE_FACT / 原件接口问题**：`BaseModule.test_step` 216–218 传 `batch_idx`/`save_attentions`，普通 `eval_step` 374–402 与 MLP 378–406 不接这两参，属于待适配的静态签名不一致。不能为通过本阶段审核改原件，也不能把旧 eval-v2 工程副本当原件或本阶段验证。
7. **SOURCE_FACT / MLP 不止加一层**：`item_aggregation_mlp` 有序 HE→2E→E，ReLU、Dropout(0.1)；强制 `sep_token=None`。`encoder_forward_pass` reshape 后每历史物品一 token，mask 取末码（不是注释里的 OR/max 备选）；用户 token 仍条件添加。共享 SID 表由 `get_embedding_table` encoder/decoder 均返回 encoder 表（934–955）。
8. **SOURCE_FACT / MLP 仍 AR 与三类 MLP**：`generate` 789–898 依已生成前缀/cache 迭代 H 次；loss 1092–1103 丢末 decoder 输出后各码 CE 求和。聚合 MLP、H 个 decoder 输出头、T5MultiLayerFF 需分开。聚合有参数且对拼接顺序敏感，不保证排列不变；冻结/初始化/公平预算为待讨论。
9. **SOURCE_FACT + UNKNOWN / Parallel**：现文件与 shuffle 完全同字节；已有历史双表/四 BOS 结构是旧证据，不恢复 NAR mask/target/loss/Top-K 组合。必须保留 `NOT_RECOVERED` 与 `RECONSTRUCTION_PENDING_USER_DISCUSSION`，不制造空类补齐。
10. **SOURCE_FACT / attention 选择与图轴**：绘图器取末层、head 均值、当步首 beam、末 query（20–149）；H 行是生成码位/step、横轴是 encoder 位置，非四 attention head。decoder self 与 cross 各自 softmax，拼接不变成联合归一分布。没有最终胜出 beam ancestry。
11. **SOURCE_FACT / 以正文纠正注释**：`build_complete_decoder_attention` docstring 31–33 声称 self 在前、cross 在后，正文 142 实际 `concatenate([cross_matrix, decoder_self_matrix])`。解释须以正文为准。
12. **SOURCE_FACT + HISTORICAL_OBSERVATION / 图与样本**：两个绘图入口独立词典序 glob 排序并用列表序号写图名，不能把 PNG batch2 当 raw batch2。encoder 裁切为行/列和启发式而非真实 mask。528/33 CRC 为前轮结果继承，不在本轮重扫描；raw/图/样本/producer共同链依旧 UNKNOWN。
13. **边界**：旧 v6 CPU 是普通模型两域有界合成接口检查，不覆盖 Shuffle、Trainer/dataloader 或全测试集。本阶段所有模型运行/加载/采集为未执行。旧 T5 历史指标与新 Qwen/RQ-KMeans baseline 分开，历史结果只可记 `HISTORICAL_REPORTED_NOT_RERUN`。

## 待正式审核

- 核本次 server 提取原件与以上原身份相同；静态本地 import 闭包、YAML defaults/target 链直接可读。
- 读固定 baseline `563dc50b1c7e9e2e05cdfddd13e91a3d1888001e` 的普通模型/有效配置和结果，避免将 A800 旧参数或数据流称新 baseline 实测。
- 对主控候选 `CODE_READING_GUIDE` 和四说明逐条原文核对，特别是十项讲解要求、重要陈述性质、准确行号和哈希、PROPOSED 与实现边界。
- 核报告没有把 Git/资产 hash 匹配写成模型运行、物品轴语义重新验证或用户理解确认。

本角色新增依赖/安装：0。未操作 Git index、STATE 或旧审计文件。
