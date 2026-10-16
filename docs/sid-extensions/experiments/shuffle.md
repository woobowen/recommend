# Shuffle：先区分三种重排位置

## 问题与 baseline

`SOURCE_FACT`：保全源码研究了 SID 层级顺序扰动，至少有预处理整数、collate 整数、embedding 查表后向量三条分支。它们不是同一个开关。作者代码事实见本页末表；作者历史实验的真实执行 Python 与全部协议并未唯一绑定，不能把我们的建议冒充作者既定方案。

`SOURCE_FACT`：baseline 入口、保存 command/config 和完整数据链见 [总图](../CODE_READING_GUIDE.md)。普通类位于 `tiger_generation_model.py`，`encoder_forward_pass` 先 offset，再共享表查向量、加 SEP/条件 user token；decoder 的 teacher forcing 与 generate 均仍按原目标码位顺序。固定 Qwen SID/数据/最佳 checkpoint 由 ASSET_MANIFEST 定义，与旧 T5 区分。

## 原件到底改了什么

|位置|作用及保留信息|不能等同的地方|
|---|---|---|
|`pre_processing.shuffle_semantic_id_hierarchies` 321–366|在整数 SID 层重排，进入 collate/标签之前|整数随后会按当前位置重新加 offset；可能连带改变输入和目标语义|
|`collate_fn_train_with_input_shuffle` 282–345|调用普通 collate 后只对模型输入整数按物品 H 分组重排|labels 已产生，输入整数后续在新位置接受 offset；不等于移动已经查好的向量|
|shuffle `_shuffle_hierarchical_embeddings` 83–149 / encoder 778–870|先 offset/查表，再对每物品前 N 个完整 E 维向量重排，随后才加 SEP|原码位表身份已编码在向量内；标签/decoder未同步重排|

`SOURCE_FACT`：原 shuffle 构造器 740–748 新建两张独立 SID embedding 表，getter 1097–1119 分别返回 encoder/decoder 表；普通 main 863–884 两个分支返回同一表。直接整体换成 shuffle 类会改变参数共享、参数量、梯度耦合和 checkpoint 接口，同时还带入 attention/预测等差异，不能称纯输入扰动。

`SOURCE_FACT + STATIC_INFERENCE`：两张表相对原共享结构多 HUE 参数（若 H/U/E 相同）。重排操作本身没有可训练参数；但整类替换的双表让 encoder/decoder 表分别接受梯度。参数影响必须区分这两件事，不凭变量名 `codebooks` 推断量化码本已重排。

## hook、mask 与随机排列

`SOURCE_FACT`：构造时 `is_testing=False`；`on_test_start` 619–622 置 True，`on_test_end` 624–627 置 False。`on_predict_start` 603–605 只调用确定性设置，未置 True。`model.eval()`、`predict()`、手动 `test_step()` 与完整 Trainer test 生命周期不能混为一谈。

`SOURCE_FACT`：输入长度不能被 H 整除时，重排函数直接返回原输入。完全 padding 的物品组跳过；正文用组内 mask 的和大于 0 判有效，所以部分有效组也会重排向量而不重排相应 mask。这不是任意部分 mask 都安全的证明。N 的范围、有效整组和 padded target 需后续明确约束。

`SOURCE_FACT`：每次 encoder 调用都会 `torch.rand(...,N)` 后 argsort 抽排列，恒等排列可能出现。`eval_step` 先 `model_step` 算 loss，再调用 generate，两条路径会分别调用 encoder，因此可能抽到不同排列。本阶段只记录真实协议，没有把它改成固定排列或共享一次抽样。

`CONFIG_DECLARATION + STATIC_INFERENCE`：`tiger_inference_flat_shuffle.yaml` 的 target 指普通模型，但带 `shuffle_embedding_on_test` 等专用参数。普通模型接 **kwargs 后传入没有 **kwargs 的 `TransformerBaseModule.__init__` 20–34；静态可见不兼容，未执行确认报错。原 `src/inference.py` 正文 28–41 调 predict，149–154 的 test 是注释；不能凭文件名把它当完整 test 重排链。

## 输入、输出与教学例子

`TEACHING_EXAMPLE`：H4、N3，offset 查表后一个物品是 `[e0(2),e1(7),e2(1),e3(0)]`。抽到 `[2,0,1]` 时向量序列变成 `[e2(1),e0(2),e1(7),e3(0)]`；其整数/码位表身份没有重新查表。这些是假设数字，不是 checkpoint 实测。

`STATIC_INFERENCE`：向量重排前后 `[B,MH,E]` 形状不变；原 SEP 后 `[B,M(H+1),E]` 形状不变，目标 `[B,H]` 与 generate `[B,K,H]` 仍原码位顺序。训练/推理是否真正触发由 hook/flag 决定。整数重排虽然也可保持形状，信息保留性质却不同；尤其不能把跨物品全序列洗牌当组内洗牌。

## 结果边界、待决定与未来检查

`HISTORICAL_OBSERVATION`：原历史配置选择了整数 N3 和向量 N4 等不同路径，已保留对应 config/overrides；它们说明配置声明，不唯一证明某份当前 Python 产生某行指标。Qwen main 三域普通指标一律历史未重跑，不能当本次 Shuffle 结果。原件不兼容按已知问题保留。

`PROPOSED_NOT_IMPLEMENTED`：后续可保留 main 的全部参数结构，只在 encoder offset/lookup 后、SEP 前加入无参数组内重排。目的为隔离输入顺序影响；预计仅新增受控开关、重排函数及明确生命周期接入，不引入第二张表、不改 SID/量化器/labels/AR/evaluator。何时重排、N=3还是4、整组 mask 约束、随机种子及 loss/generate 共用排列与否，均待用户选择。本阶段没有实现此方案。

`PROPOSED_NOT_IMPLEMENTED`：未来检查目标为关闭开关时源码/参数/输出等价；开启时组边界和向量集合守恒；offset 身份、SEP、padding、目标不变；hook 触发可证；每次排列/seed可追踪；checkpoint 共享表接口与梯度路径明确；公平比较使用同一数据与指标协议。这里没有可自动运行的流水线。

`UNKNOWN`：原作者全部随机化协议与历史运行正文绑定仍不唯一。用户需要先选择是研究“码位标签语义被交换”，还是“保留码位身份的向量顺序被扰动”，再从末表 `_shuffle_hierarchical_embeddings` 开始逐段阅读。


## 冻结源码证据

本表的行号由实际发布字节核对；源码集合身份 `8b3149a98731ba7857e446e0f488489be0715181b841f8c7f9f169d3c4e16ac3`。baseline链接固定完整SHA；原件链接在CONTENT_COMMIT发布后由交付元数据绑定该完整SHA，原始捕获版本和每文件完整SHA始终保留。

|文件/符号/实际行范围|版本|源SHA-256|
|---|---|---|
|[GRID baseline/src/models/modules/semantic_id/tiger_generation_model.py::SemanticIDEncoderDecoder.get_embedding_table L863–884](https://github.com/woobowen/recommend/blob/563dc50b1c7e9e2e05cdfddd13e91a3d1888001e/GRID%20baseline/src/models/modules/semantic_id/tiger_generation_model.py#L863-L884)|`563dc50b1c7e9e2e05cdfddd13e91a3d1888001e`|`c61606b4a03d6fc8cfa6cb3bca093a36bbf9a1846334d39cb8dbcb67b32c296e`|
|[references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py::SemanticIDEncoderDecoder.__init__ L642–776](https://github.com/woobowen/recommend/blob/84d4533fbdd6b010f54d671b433a9952608911ca/references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py#L642-L776)|`A800-grid-20261010-v1`|`e03082ad87bfc842c05467580fc479eff4345c6a0d18c5b2f7faf320d4f96c80`|
|[references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py::SemanticIDEncoderDecoder.get_embedding_table L1097–1119](https://github.com/woobowen/recommend/blob/84d4533fbdd6b010f54d671b433a9952608911ca/references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py#L1097-L1119)|`A800-grid-20261010-v1`|`e03082ad87bfc842c05467580fc479eff4345c6a0d18c5b2f7faf320d4f96c80`|
|[references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py::SemanticIDGenerativeRecommender._shuffle_hierarchical_embeddings L83–149](https://github.com/woobowen/recommend/blob/84d4533fbdd6b010f54d671b433a9952608911ca/references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py#L83-L149)|`A800-grid-20261010-v1`|`e03082ad87bfc842c05467580fc479eff4345c6a0d18c5b2f7faf320d4f96c80`|
|[references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py::SemanticIDEncoderDecoder.encoder_forward_pass L778–870](https://github.com/woobowen/recommend/blob/84d4533fbdd6b010f54d671b433a9952608911ca/references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py#L778-L870)|`A800-grid-20261010-v1`|`e03082ad87bfc842c05467580fc479eff4345c6a0d18c5b2f7faf320d4f96c80`|
|[references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py::SemanticIDGenerativeRecommender.on_test_start L619–622](https://github.com/woobowen/recommend/blob/84d4533fbdd6b010f54d671b433a9952608911ca/references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py#L619-L622)|`A800-grid-20261010-v1`|`e03082ad87bfc842c05467580fc479eff4345c6a0d18c5b2f7faf320d4f96c80`|
|[references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py::SemanticIDGenerativeRecommender.on_test_end L624–627](https://github.com/woobowen/recommend/blob/84d4533fbdd6b010f54d671b433a9952608911ca/references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py#L624-L627)|`A800-grid-20261010-v1`|`e03082ad87bfc842c05467580fc479eff4345c6a0d18c5b2f7faf320d4f96c80`|
|[references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py::SemanticIDGenerativeRecommender.on_predict_start L603–605](https://github.com/woobowen/recommend/blob/84d4533fbdd6b010f54d671b433a9952608911ca/references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py#L603-L605)|`A800-grid-20261010-v1`|`e03082ad87bfc842c05467580fc479eff4345c6a0d18c5b2f7faf320d4f96c80`|
|[references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py::SemanticIDGenerativeRecommender.eval_step L525–575](https://github.com/woobowen/recommend/blob/84d4533fbdd6b010f54d671b433a9952608911ca/references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py#L525-L575)|`A800-grid-20261010-v1`|`e03082ad87bfc842c05467580fc479eff4345c6a0d18c5b2f7faf320d4f96c80`|
|[references/wmh-original/GRID/src/data/loading/components/pre_processing.py::shuffle_semantic_id_hierarchies L321–366](https://github.com/woobowen/recommend/blob/84d4533fbdd6b010f54d671b433a9952608911ca/references/wmh-original/GRID/src/data/loading/components/pre_processing.py#L321-L366)|`A800-grid-20261010-v1`|`a69c88681760df22e37fb7e66961ad97ed0824af20eae415a543379e881d71c7`|
|[references/wmh-original/GRID/src/data/loading/components/collate_functions.py::collate_fn_train_with_input_shuffle L282–345](https://github.com/woobowen/recommend/blob/84d4533fbdd6b010f54d671b433a9952608911ca/references/wmh-original/GRID/src/data/loading/components/collate_functions.py#L282-L345)|`A800-grid-20261010-v1`|`a4692ce68705ea6a10ed500245f5c5acc99468f7e5d3f2d41e1ff0970fa9a73c`|
|[references/wmh-original/GRID/configs/experiment/tiger_inference_flat_shuffle.yaml::(全文) L1–251](https://github.com/woobowen/recommend/blob/84d4533fbdd6b010f54d671b433a9952608911ca/references/wmh-original/GRID/configs/experiment/tiger_inference_flat_shuffle.yaml#L1-L251)|`A800-grid-20261010-v1`|`d60b73c5e502ab1ed105a550340b9e748adbcf181f19992e3a32c55038d768a6`|
|[references/wmh-original/GRID/src/models/modules/huggingface/transformer_base_module.py::TransformerBaseModule.__init__ L20–64](https://github.com/woobowen/recommend/blob/84d4533fbdd6b010f54d671b433a9952608911ca/references/wmh-original/GRID/src/models/modules/huggingface/transformer_base_module.py#L20-L64)|`A800-grid-20261010-v1`|`2135503c66efb71e16a57578b39359dfd668c5c84b57146f68aac5e264d863a5`|
|[references/wmh-original/GRID/src/inference.py::(全文) L1–171](https://github.com/woobowen/recommend/blob/84d4533fbdd6b010f54d671b433a9952608911ca/references/wmh-original/GRID/src/inference.py#L1-L171)|`A800-grid-20261010-v1`|`f24892ed2d9777c6fd5d3eaf20cc0ea3d180a19a490519baa7ca2390a70c05ba`|


## 已保留的历史记录入口

下列全部 `HISTORICAL_OBSERVATION`，仅记录原文件和test行位置，未选择/重算指标，未证明当前Python唯一生成它们；T5历史上下文不替换Qwen工作资产。

- [logs/inference/runs/2025-10-29/14-59-58 原config](https://github.com/woobowen/recommend/blob/84d4533fbdd6b010f54d671b433a9952608911ca/references/wmh-original/GRID/logs/inference/runs/2025-10-29/14-59-58/.hydra/config.yaml)，SHA-256 `2ef1c71862d1da06d3a959c9a0b144d158842486e54b9f5704738c19d133055a`；同目录overrides/hydra原文保留。 [完整CSV](https://github.com/woobowen/recommend/blob/84d4533fbdd6b010f54d671b433a9952608911ca/references/wmh-original/GRID/logs/inference/runs/2025-10-29/14-59-58/csv/version_0/metrics.csv) 的原test行：2，SHA-256 `dde27b856c0345821d64bc2bf9b599b2c03c0359775df8ddb24ea31fa750c70e`。
- [logs/inference/runs/2025-10-29/15-47-20 原config](https://github.com/woobowen/recommend/blob/84d4533fbdd6b010f54d671b433a9952608911ca/references/wmh-original/GRID/logs/inference/runs/2025-10-29/15-47-20/.hydra/config.yaml)，SHA-256 `c7c58df3dd7eaebca64bd316060540344a9cc7b81dd84bd5dbd94147c76f534f`；同目录overrides/hydra原文保留。 [完整CSV](https://github.com/woobowen/recommend/blob/84d4533fbdd6b010f54d671b433a9952608911ca/references/wmh-original/GRID/logs/inference/runs/2025-10-29/15-47-20/csv/version_0/metrics.csv) 的原test行：2，SHA-256 `5dc99b63777675e2a5def3720d99d505e43cf9ba5e173ce3a1ab2fb7db85a222`。


## 2026-10-14 Beauty 执行：PARTIAL_BLOCKED

本节记录本轮一次性授权，旧 P0 正文及禁跑记录保持历史含义。本轮已实际适配/加载/预跑；**六组 A/B Beauty 正式全量评测仍未启动。** A/B 三轮集中修正耗尽后停止追加修补，不能把小样本结果填进正式实验表。用户后来明确批准的独立 MLP 单卡方案已继续完成累积64、完整验证与原callbacks生命周期检查，其当前准入/正式状态单独见 [MLP说明](mlp.md) 和机器回执，不被 A/B 阻塞替代。

|实验/路线|代码与资产|有效配置来源|采用值|原件/适配差异|确认依据|预跑/正式准入|
|---|---|---|---|---|---|---|
|A Normal/Shuffle3/4|author/models.py 继承冻结作者双表；T5/R-Kmeans SID 09-15/20-54-16 + 双表 step4800 11-03/02-47-09|manifests/beauty-extensions/effective-config-sources/a-t5-rkmeans.yaml|H4/E128/U256/K10、seed42、prefix=false、L120、独立 encoder/decoder SID 表|接口桥接、原重排函数加只读 trace/回放；不复制权重、不改正文|用户本轮明确批准，checkpoint 内嵌 SID exact，178 state tensors 键/shape/值 exact|配对及静态/输入通过，3用户真实生成 attention 通过；A 完整 OFF/Shuffle/Trainer 门未完成|
|B Normal/Shuffle3/4|baseline.py 在原共享表 encoder lookup 后复用作者函数；Qwen/R-Kmeans SID + 共享表 step12500|effective-config-sources/b-qwen-rkmeans.yaml|H4/E128/U256/K10、seed42、prefix=false、L120、SEP、无用户表|仅 encoder 作用域 lookup hook；不改参数/decoder/beam/evaluator，OFF 无额外全局随机数|用户本轮明确批准，checkpoint 内嵌 SID exact，177 state tensors exact|同形状 OFF/缓存及三条件各19用户真实 test 完成；尾部验证设备错误使整条 smoke FAILED，正式门 BLOCKED|
|Beauty MLP|author/models.py 继承原逐物品共享 MLP；真正 Beauty T5/RQ-VAE+STE SID 09-21/16-36-33|effective-config-sources/mlp-beauty-original.yaml|用户批准从零全参数、raw32/accum64/val6400、workers8/K10/FP32；原Adam/选模/早停/max320000/10epochs，硬12h|原聚合、共享SID表、无SEP、末码mask/AR loss保留；独立mlp_formal入口、hash绑定非作者门、预算中断保存边界|用户两次明确批准；原配置/源码/历史run记录；2次真实acc64更新及完整val、另1次更新的原callbacks→strict best→16用户test闭环已通过|资源/覆盖/生命周期通过；正式准入与运行结果见独立gate、MLP说明及EXECUTION_SUMMARY，预跑状态不流入新训练|
|Cross-attention|attention/{capture,records,plot}.py，真实生成观测|用户批准的固定32用户、双路径方案|最终Top1祖先主口径 + 每步局部首beam对照，最后层、逐head保存/head均值展示|不重写beam，不换后端，不裁剪/逐行缩放；完整父链/位置/矩阵/图|用户明确批准；A/B/MLP 各自固定32已保存|A/B各3用户Normal工程采集；新MLP正式best2600的固定32用户已采集，须与3-update工程图分开；A/B正式Normal/Shuffle图未运行|

A/B 是不同训练模型和上游条件，各自 Normal 才是本组对照。MLP 采用另行批准的不同 SID 条件，不能声称相对于 A/B 的纯结构对照；无匹配Normal时不补训。具体资产、来源清单、同一 Beauty 数据物品轴继承和只读恢复证据见 [source-review](../../../manifests/beauty-extensions/source-review.json)。

**真实范围。** 两套/三套 SID 均为 `[4,12101]` 整数表；第四位按原 producer 消歧（前三位唯一者0；碰撞组全部成员按索引次序1..组大小），不是第四层量化。完整 testing 元数据为22,363行/22,363不同用户、175文件、0重复，原物品→SID→目标/隐藏组逐条核验。testing 原始长度5..21物品，最长可见20；因此没有超过120-token窗口的真实样例，不声称测过真实截断边界。固定32 evaluation 用户的样本身份、输入/mask/目标、文件/行顺序都已保存，无原始 user_id 公开。

**OFF 与数值边界。** B 同 batch 的原类/适配 hidden、四头logits、loss、SID/score精确相同；连续生成/反序及 B8→B1→B8 回环精确一致。不同 batch shape 的 score 有原模型共同漂移：旧r2首批8.335337e-6；按真实文件assignment的r3首批3.007054328918457e-5，原类与适配B1仍误差0。保留原容差/medium matmul设置与false诊断，不把跨shape偏差伪报为0，也没有据此证明唯一硬件根因。

**修正额度与阻塞。** r0 checkpoint安全加载拒绝保存的partial元数据；r1用hash绑定、静态GLOBALS核查、局部safe_globals解决，随后版本记录误查tensorflow发行名；r2采用真实tensorflow-cpu发行名并核到跨batch原模型数值边界；r3保留该边界并完成真实test，但最后手动val复用CUDA输入时，Lightning Strategy.teardown已把模型移回CPU，发生device mismatch。该脚本设备管理错误未修，r3 job整体FAILED；不继续第四轮，不把已写预测/退出子项当全线PASS。完整日志、request/status与各版快照清单保留。

**A/B未完成。** A路线内验收尚未齐全；B尾部验证/异常清理尚未通过；A正式8-worker生命周期未预跑；全部6组正式Normal/Shuffle NOT_RUN；A/B formal CLI原布尔门尚未升级为报告/代码/配置/资产hash绑定，其summarize子命令未接入（独立verify_saved_predictions与summarize_delivery入口已实测）。A/B不能标READY_FOR_BEAUTY_RUN。MLP的SID/初始化/训练预算已批准，独立正式handler及hash门已提供；其正式训练与32用户有效模型图必须按独立回执判断，不沿用本段早期未批准状态。

**资源与保存。** 仅使用空闲物理GPU1，未抢占GPU0同学作业。必要5实体按归档索引恢复161,660,760 B，每member完整hash通过，未重审78GB原档案。现有GRID环境复用，新增系统/语言包/工具链安装=0，全局配置改动=0，未清理原件。服务器作业均由既有runner脱离SSH运行；完整job终态、用量和产物入口见 [EXECUTION_SUMMARY](../../../manifests/beauty-extensions/EXECUTION_SUMMARY.json)。

恢复入口：`/home/bwwu/grid-audits/sid-extensions-p0/jobs/<job_id>/`；每个request的cwd绑定受控export快照和逐文件SHA，不假称Git checkout。最后交付Git提交只发布实际源码/配置/精简回执；大型checkpoint留服务器，必要外部矩阵与完整回执在轻量ZIP中。A/B需要新的修正额度后才能继续；MLP按已批准条件独立推进，不重复询问已批准的资产、attention或MLP协议。

作者路线按已保全实现适配，非逐字历史运行复现；历史唯一Python绑定缺口继续保留。源码入口：[CLI](../../../experiments/beauty_extensions/cli.py)、[A/MLP派生](../../../experiments/beauty_extensions/author/models.py)、[B接入](../../../experiments/beauty_extensions/baseline.py)、[attention观测](../../../experiments/beauty_extensions/attention/capture.py)。

2026-10-15终验补充：独立MLP正式从零训练已按原patience10在3600次更新结束，验证选中best2600，完整test22363用户/目标及CPU/非作者预测复算通过；其固定32用户真实生成双口径注意力和完整训练曲线已经写出。具体指标、权重、资源与比较限制见 [MLP正式结果](mlp.md#2026-10-15-正式新训练结果)。该完成不补足A/B的缺失门禁：A-Normal/Shuffle3/Shuffle4、B-Normal/Shuffle3/Shuffle4六组正式指标及各自Normal的delta均为空值。没有现象重现失败后的调参，没有第四轮研究代码修补，整体交付仍为PARTIAL_BLOCKED。

## 2026-10-15 接续预跑：工程通过，正式实验未授权

本阶段 `grid-shuffle-mlp-preflight-20261015` 从 Beauty 固定提交 `6b39c4323ab9e7c3a369aab6dab6eb59d6129e2c` 接续。接管时本地 HEAD 为 `01dbaf2e10dfaeb8215048c33493e732e32a25a2`，已有9个未推送提交、3个修改文件及预跑/审核记录。旧 CPU-only 摘要已落后现场；A/B GPU 作业已成功，直接继承，不重新计算。现行总体状态 **PARTIAL_PREFLIGHT**：四条工程线已完成，Toys/Sports 正式研究条件与预算仍未批准。旧 Beauty 的 PARTIAL_BLOCKED、正式 MLP 成果和失败记录保持原样。

|门禁|A：T5/R-Kmeans + 双表 step4800|B：Qwen/R-Kmeans + 共享表 step12500|
|---|---|---|
|严格加载|178 state tensors，13,259,136参数，双表数值各自保留，内嵌SID exact|177 state tensors，13,128,064参数，encoder/decoder共享对象，内嵌SID exact|
|OFF与对应原类|同形状 hidden/logits/loss/SID/beam score exact|同形状 hidden/logits/loss/SID/beam score exact|
|真实 Normal/Shuffle3/Shuffle4|各31条，4批8/8/8/7；原evaluator复算通过|各31条，4批8/8/8/7；原evaluator复算通过|
|生命周期与尾检|普通val11、train OFF、真实Trainer异常后同实例恢复、cache检查通过；另原8-worker datamodule有界4批32用户|普通val11、train OFF、真实Trainer异常后同实例恢复、cache检查通过；原worker0|
|采样|每物品每次出现独立，先offset/lookup再排列；Shuffle3/4各8次调用（loss4/generate4），实际包含6/24种排列及恒等排列|同左；共享表hook仅encoder scope，decoder不触发|
|工程/研究条件/正式审核准备|ENGINEERING_PREFLIGHT_PASS / RESEARCH_CONDITIONS_CONFIRMED / READY_FOR_FORMAL_REVIEW|同左|
|正式执行|FORMAL_TECHNICALLY_READY_PENDING_USER_AUTHORIZATION；未批准、未启动|同左|

**设备错误已通过真实预跑验证修复。** 旧 `beauty-b-smoke-r3` 的三次 Trainer.test 后，Lightning 2.5.0 `Strategy.teardown()` 将模型移至 CPU，尾部输入仍为 CUDA。现有 `cli.smoke` 仅在尾检恢复同一模型 `.to(device).eval()`、普通 validation hook，并从干净 batch 重建输入。两线 `post_test_off.json` 实测模型/输入同为 CUDA、is_testing=false、得分误差0；正常结束及 callback 注入的真实异常都恢复 OFF，普通 train/val 不增加 Shuffle 调用。没有更改原推荐分数、共享关系、生成或评价。

保留数值边界：跨 batch-size 得分漂移 A `7.991213e-6`、B `1.936033e-5`，原模型与适配版相同；同形状 OFF 精确一致，不宣称任意 batch-size 得分完全一致。固定子集最长可见历史20物品，不能冒称覆盖29物品截断上界。A/B 模型、SID和上游不同，只能各自对比 Normal，不以绝对指标主张同模型工程等价。

原 GPU job 为 `gsp-20261015-a-shared-r1`、`gsp-20261015-b-shared-r1`，均 SUCCEEDED/0/cleanup verified；执行 commit `afbd2336fda5d5f87210b342ff3744bee2156c02`、snapshot `bcba4952e2830e13daf00c8001729a231488aad328b1144a82814d6e7350bde4`。非作者核对当前计算源码字节/函数AST，只有 MLP callback JSON报告、部署身份和交付收集代码变化，旧A/B证据继续有效；最终源码绑定另附当前审核，未篡改原部署记录。

**第3轮部署身份修复。** 旧 snapshot `5227c1a77ee4fd85` 只对文件清单取摘要，在 commit变化但文件字节相同时复用了旧 DEPLOYMENT.json。实际旧manifest SHA `a84bafbac6ab3adbc6b5c978bae066ecfd1df8ab9d6a27205b68e638c901338e`，r2新请求期望 `94ab4341a159c3804b02e08c175edff9cecbac16bff8c963e668f0727be121bf`，runner因此正确拒绝，未启动模型。接管继承的补丁将commit+文件记录共同绑定快照身份，固定为 `59c42a213f2c3f4b90d0c069d4d3b98a2d8a398b`；纯文件校验及两个独立恢复job接受/执行均通过。旧r2请求与拒绝原始输出保留，新r3仅验证已完成的round3，不绕过runner、不增加第4轮。

任务累计14个已接收job：12成功、2失败；另r2接受前拒绝。7个GPU请求的walltime（含初始化及失败恢复）合计0.254719264 GPU·h，其中本次新增两恢复0.036208736 GPU·h；不是纯GPU计算时长。原预算2h目标/6h修正上限未重置，修正3/3。原CPU资源拒绝是历史事实；本次GPU1经连续两次轻量观察有充足余量后串行使用，无外部进程终止，无活动本任务job遗留。

四份正式candidate均未签署，formal_approved=false；不含工程row/step overlay。A/B须网页二审和用户另行正式批准后，绑定当前独立审核/计划/代码/资产生成新的授权gate。本轮未调用正式handler、全量测试或注意力采集。逐job版本、产物、来源缺口、恢复边界和非作者审核见 [EXECUTION_SUMMARY.json](../../../manifests/grid-shuffle-mlp-preflight-20261015/EXECUTION_SUMMARY.json)；原日志、预测、采样与部署回执在实际校验的轻量ZIP。

## 2026-10-15 正式阶段接续

用户在预跑网页版二审及临时件清理之后，明确批准 Beauty A/B 各 Normal/Shuffle3/Shuffle4 全量评测，并随后采集主工作baseline B与A/B fixed32 cross-attention。继承93567的真实预跑，不重做31行工程检查。旧MLP正式结果、fixed32及工程注意力记录保留。当前无未提交研究修改/未推送提交/活动旧worker，原服务器工作区仍为export。

新增 `experiments/formal_20261015` 只编排原common.instantiate、原Trainer.test/datamodule、原evaluator与预测记录；模型、Shuffle sampler、beam、评价计算不改。部署复用已验收commit+文件清单身份，正式所有文件须属于固定Git提交；不调用仍有file-only目录问题的旧deploy。每路线先CPU极小身份/gate/完整索引确认，无额外模型/GPU预跑，再分别执行3个正式条件；每条件完整22363目标，保存原预测、完整coverage复算及原始Shuffle事件/总调用数。

A/B分别对自己的Normal作delta，不把不同SID/模型的绝对值当同条件对照。注意力独立job，依赖对应正式结果成功/身份审核；继承同用户/物品的既有固定32 cohort，每路线使用自己的SID/target。显示最后decoder层head均值，保留所有raw heads与actual final-Top1 ancestry/local-first两路径，横轴是encoder位置。观测OFF/ON一致性不等于Shuffle OFF/ON对照。

原预跑3/3修正与0.254719GPUh历史不重置；新增正式入口是本轮授权的初始编排，运行失败不据此自动获得第4轮修正。每正式评测job硬限1800秒；作业由已有runner脱离SSH独立运行，按原ID查询，无成功任务重跑。当前逐job状态与后续结果见 [正式阶段记录](../../../manifests/grid-formal-20261015/EXECUTION_SUMMARY.json)。

## 2026-10-16 Beauty 六组正式评测完成

六组正式全量 `Trainer.test()` 均 SUCCEEDED/0、worker清理确认，并经 `/root/results_publication_review` 直接对原预测/完整目标索引/原evaluator/逐样本指标独立验收。每组22363个用户与目标精确覆盖，全部原候选保留，未按测试调参。运行固定提交 `2cf4c793c2e0b2e9dc7e7a9269f5e54d5c667896`，复用不可变export `3870cb40eb8d040a`。

|路线/条件|Recall@5|Recall@10|NDCG@5|NDCG@10|实际Shuffle调用|
|---|---:|---:|---:|---:|---:|
|A-normal|0.046237089|0.066449046|0.031252529|0.037725352|0|
|A-shuffle3|0.046371236|0.066359609|0.031076979|0.037486158|5592|
|A-shuffle4|0.045968786|0.065733574|0.030933077|0.037285861|5592|
|B-normal|0.043151636|0.065867729|0.029169954|0.036434427|0|
|B-shuffle3|0.043285783|0.065018110|0.029069517|0.036036659|5592|
|B-shuffle4|0.042793900|0.064481512|0.028868502|0.035842434|5592|

Shuffle均为loss2796/generate2796分别采样；实际原生排列记录、全调用数、参数加载和全2796 batch的设备/清理记录保留。A/B分别只与自己的Normal作条件比较，不把两路不同SID/共享结构的绝对指标当同模型对照。单seed结果未提供显著性或普遍顺序不变性结论。

新正式入口的CPU配置保存首次失败保持FAILED：whole-cfg resolve=True误解析未用Hydra now:id。用户明确追加一次严格限定例外后，仅改为与旧成功入口一致的resolve=False；四组真实CPU保存/重载/有效子树解析exact及A/B正式入口identity均实际通过，模型/优化器/训练/评价配置未变。原3/3修正不重置，此例外不授予继续扩大返工。

完整精度/每job输入与结果见 [正式阶段记录](../../../manifests/grid-formal-20261015/EXECUTION_SUMMARY.json)；[六组独立总验收](../../../manifests/grid-formal-20261015/results-review/beauty-six-formal-results-independent-summary.json)。原始全预测与逐样本指标将纳入最终轻量ZIP，正式注意力是后续独立job，不与这六次评测混记。


## Toys/Sports Shuffle 正式补齐（2026-10-17）

Toys/Sports B 的 Normal、Shuffle3、Shuffle4 完整测试、对应六组 fixed32 和2304条注意力描述统计已完成并经真实非作者审核；Beauty仅继承。Toys A选择/上游来源未闭合，Sports A缺完整配套链，总状态 PARTIAL。原始预测、24份NPZ和全部192张PNG保留服务器；代码、配套身份、完整指标/配对、统计、失败记录及审核入口见 [正式报告](../../../manifests/grid-shuffle-20261017/FINAL_REPORT.md)。单seed42，不将不同A/B基础条件或图像相似解释为严格置换不变。
