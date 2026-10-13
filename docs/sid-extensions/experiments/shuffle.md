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
|[references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py::SemanticIDEncoderDecoder.__init__ L642–776](https://github.com/woobowen/recommend/blob/SOURCE_CONTENT_COMMIT/references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py#L642-L776)|`A800-grid-20261010-v1`|`e03082ad87bfc842c05467580fc479eff4345c6a0d18c5b2f7faf320d4f96c80`|
|[references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py::SemanticIDEncoderDecoder.get_embedding_table L1097–1119](https://github.com/woobowen/recommend/blob/SOURCE_CONTENT_COMMIT/references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py#L1097-L1119)|`A800-grid-20261010-v1`|`e03082ad87bfc842c05467580fc479eff4345c6a0d18c5b2f7faf320d4f96c80`|
|[references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py::SemanticIDGenerativeRecommender._shuffle_hierarchical_embeddings L83–149](https://github.com/woobowen/recommend/blob/SOURCE_CONTENT_COMMIT/references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py#L83-L149)|`A800-grid-20261010-v1`|`e03082ad87bfc842c05467580fc479eff4345c6a0d18c5b2f7faf320d4f96c80`|
|[references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py::SemanticIDEncoderDecoder.encoder_forward_pass L778–870](https://github.com/woobowen/recommend/blob/SOURCE_CONTENT_COMMIT/references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py#L778-L870)|`A800-grid-20261010-v1`|`e03082ad87bfc842c05467580fc479eff4345c6a0d18c5b2f7faf320d4f96c80`|
|[references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py::SemanticIDGenerativeRecommender.on_test_start L619–622](https://github.com/woobowen/recommend/blob/SOURCE_CONTENT_COMMIT/references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py#L619-L622)|`A800-grid-20261010-v1`|`e03082ad87bfc842c05467580fc479eff4345c6a0d18c5b2f7faf320d4f96c80`|
|[references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py::SemanticIDGenerativeRecommender.on_test_end L624–627](https://github.com/woobowen/recommend/blob/SOURCE_CONTENT_COMMIT/references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py#L624-L627)|`A800-grid-20261010-v1`|`e03082ad87bfc842c05467580fc479eff4345c6a0d18c5b2f7faf320d4f96c80`|
|[references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py::SemanticIDGenerativeRecommender.on_predict_start L603–605](https://github.com/woobowen/recommend/blob/SOURCE_CONTENT_COMMIT/references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py#L603-L605)|`A800-grid-20261010-v1`|`e03082ad87bfc842c05467580fc479eff4345c6a0d18c5b2f7faf320d4f96c80`|
|[references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py::SemanticIDGenerativeRecommender.eval_step L525–575](https://github.com/woobowen/recommend/blob/SOURCE_CONTENT_COMMIT/references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py#L525-L575)|`A800-grid-20261010-v1`|`e03082ad87bfc842c05467580fc479eff4345c6a0d18c5b2f7faf320d4f96c80`|
|[references/wmh-original/GRID/src/data/loading/components/pre_processing.py::shuffle_semantic_id_hierarchies L321–366](https://github.com/woobowen/recommend/blob/SOURCE_CONTENT_COMMIT/references/wmh-original/GRID/src/data/loading/components/pre_processing.py#L321-L366)|`A800-grid-20261010-v1`|`a69c88681760df22e37fb7e66961ad97ed0824af20eae415a543379e881d71c7`|
|[references/wmh-original/GRID/src/data/loading/components/collate_functions.py::collate_fn_train_with_input_shuffle L282–345](https://github.com/woobowen/recommend/blob/SOURCE_CONTENT_COMMIT/references/wmh-original/GRID/src/data/loading/components/collate_functions.py#L282-L345)|`A800-grid-20261010-v1`|`a4692ce68705ea6a10ed500245f5c5acc99468f7e5d3f2d41e1ff0970fa9a73c`|
|[references/wmh-original/GRID/configs/experiment/tiger_inference_flat_shuffle.yaml::(全文) L1–251](https://github.com/woobowen/recommend/blob/SOURCE_CONTENT_COMMIT/references/wmh-original/GRID/configs/experiment/tiger_inference_flat_shuffle.yaml#L1-L251)|`A800-grid-20261010-v1`|`d60b73c5e502ab1ed105a550340b9e748adbcf181f19992e3a32c55038d768a6`|
|[references/wmh-original/GRID/src/models/modules/huggingface/transformer_base_module.py::TransformerBaseModule.__init__ L20–64](https://github.com/woobowen/recommend/blob/SOURCE_CONTENT_COMMIT/references/wmh-original/GRID/src/models/modules/huggingface/transformer_base_module.py#L20-L64)|`A800-grid-20261010-v1`|`2135503c66efb71e16a57578b39359dfd668c5c84b57146f68aac5e264d863a5`|
|[references/wmh-original/GRID/src/inference.py::(全文) L1–171](https://github.com/woobowen/recommend/blob/SOURCE_CONTENT_COMMIT/references/wmh-original/GRID/src/inference.py#L1-L171)|`A800-grid-20261010-v1`|`f24892ed2d9777c6fd5d3eaf20cc0ea3d180a19a490519baa7ca2390a70c05ba`|


## 已保留的历史记录入口

下列全部 `HISTORICAL_OBSERVATION`，仅记录原文件和test行位置，未选择/重算指标，未证明当前Python唯一生成它们；T5历史上下文不替换Qwen工作资产。

- [logs/inference/runs/2025-10-29/14-59-58 原config](https://github.com/woobowen/recommend/blob/SOURCE_CONTENT_COMMIT/references/wmh-original/GRID/logs/inference/runs/2025-10-29/14-59-58/.hydra/config.yaml)，SHA-256 `2ef1c71862d1da06d3a959c9a0b144d158842486e54b9f5704738c19d133055a`；同目录overrides/hydra原文保留。 [完整CSV](https://github.com/woobowen/recommend/blob/SOURCE_CONTENT_COMMIT/references/wmh-original/GRID/logs/inference/runs/2025-10-29/14-59-58/csv/version_0/metrics.csv) 的原test行：2，SHA-256 `dde27b856c0345821d64bc2bf9b599b2c03c0359775df8ddb24ea31fa750c70e`。
- [logs/inference/runs/2025-10-29/15-47-20 原config](https://github.com/woobowen/recommend/blob/SOURCE_CONTENT_COMMIT/references/wmh-original/GRID/logs/inference/runs/2025-10-29/15-47-20/.hydra/config.yaml)，SHA-256 `c7c58df3dd7eaebca64bd316060540344a9cc7b81dd84bd5dbd94147c76f534f`；同目录overrides/hydra原文保留。 [完整CSV](https://github.com/woobowen/recommend/blob/SOURCE_CONTENT_COMMIT/references/wmh-original/GRID/logs/inference/runs/2025-10-29/15-47-20/csv/version_0/metrics.csv) 的原test行：2，SHA-256 `5dc99b63777675e2a5def3720d99d505e43cf9ba5e173ce3a1ab2fb7db85a222`。
