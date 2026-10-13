# MLP：每个物品的四个码位怎样合成一个 token

## 研究问题与现有实现

`SOURCE_FACT`：现存 `tiger_generation_model_mlp.py` 不是单纯“加一层 MLP”。它在 item 级有序拼接 H 个 SID embedding，聚合到一个 encoder token，并取消 SEP、改变 encoder mask/长度，同时保留自回归 decoder。作者原实现与当前 main 的接口/预测/attention 差异均保留；未来适配建议不是已实现功能。

`SOURCE_FACT`：baseline 的真实命令/配置入口、item→SID→offset→共享表→encoder/decoder→loss/generate→evaluator 见 [总图](../CODE_READING_GUIDE.md)。原件没有可确认的当前即用 MLP experiment 模板；已恢复 `logs/train/runs/2025-10-22/21-25-31/.hydra/` 作为历史配置上下文，不能按 hparams 或路径反推它唯一执行了当前 MLP Python。

## 逐段对照

|正文位置|实际变化|性质|
|---|---|---|
|构造 580–585|`Linear(H*E,2*E) → ReLU → Dropout(0.1) → Linear(2*E,E)`，两个 Linear 有 bias|SOURCE_FACT|
|构造 587–592|`sep_token=None`；不因传 should_add_sep_token 就实际保留 SEP|SOURCE_FACT|
|encoder 614–650|offset 与共享 lookup 后 `[B,MH,E]` view 为 `[B,M,H,E]`，按原顺序拼接 `[B,M,HE]`，聚合 `[B,M,E]`|SOURCE_FACT / STATIC_INFERENCE|
|encoder 649–651|mask view 为 `[B,M,H]`，取 `[:,:, -1]`；注释“可以any/max”并未执行|SOURCE_FACT|
|encoder 665–695|条件 user token 仍存在；当前 main 配置 num_user_bins=null，所以不能无条件画一个 user token|SOURCE_FACT / CONFIG_DECLARATION|
|getter 934–955|encoder/decoder 都返回同一 SID embedding 表|SOURCE_FACT|
|generate 789–898 / loss1050–1103|仍 H 轮 AR、KV cache和beam；teacher forcing 后 H 个输出头各算交叉熵求和|SOURCE_FACT|

`TEACHING_EXAMPLE`：假设一个物品四个 embedding 是 `[a,b,c,d]`，先拼接成 `[a;b;c;d]` 再映射成 m。交换 a/b 会把它们送进第一层的不同权重列，因此普通全连接聚合没有天然排列不变性。例子不是实际训练值；“每个 item 一个 token”也不表示四个目标码并行输出。

## 三种 MLP 不要混淆

`SOURCE_FACT`：`item_aggregation_mlp` 是这里新增的输入物品聚合器。`decoder_mlp` 是 H 个 E→U 的无 bias 输出预测头。`mlp_layers` 则会把 T5 内部 `T5LayerFF` 替换为 `T5MultiLayerFF`。三者作用阶段、形状和参数不同；预测 Head 也不等于 attention head。

`STATIC_INFERENCE`：输入整数和 labels 与 main 相同；encoder 表示由 `[B,M(H+1),E]`（SEP启用）变为 `[B,M,E]`。decoder 输出仍按 H 个目标码位，generate `[B,K,H]`、evaluator 完整 SID 匹配不因此变成非自回归。部分有效码位时“取末位 mask”与 OR 聚合有不同语义；MLP 可把 padded lookup 向量/bias 变为非零表示，但 mask 是否完全屏蔽须留给后续运行检查。

## 参数、梯度和公平比较

`STATIC_INFERENCE`：聚合器参数为 `(HE×2E+2E)+(2E×E+E)=2E²(H+1)+3E`，取消原 SEP 的 E 参数；其余条件相同时净增 `2E²(H+1)+2E`。`TEACHING_EXAMPLE`：H4/E128 时新增164224、净增164096，属于算式教学值，未读取 checkpoint 参数。这个结论不包括启用其他 T5 mlp_layers、用户表或不同初始化所造成的差异。

`SOURCE_FACT + STATIC_INFERENCE`：理论梯度由 loss→decoder cross attention→encoder→aggregation MLP→共享 SID表，另有decoder前缀对同表的梯度路径。ReLU 与 Dropout(0.1) 改变训练映射；eval与自定义 hook 的实际状态仍需核。优化器配置/冻结、是否从普通 checkpoint 初始化、聚合器如何初始化、SEP被移除如何影响旧权重键，以及比较训练步数还是计算预算，均没有由本阶段替用户决定。

## 历史与接口限制

`HISTORICAL_OBSERVATION`：历史配置/权重出现多种 MLP 相关结构，配置相同不保证 checkpoint结构相同；不把当前文件唯一绑定全部历史指标。普通 Qwen 三域结果是 `HISTORICAL_REPORTED_NOT_RERUN`，没有本次 MLP 结果。

`SOURCE_FACT + STATIC_INFERENCE`：原 BaseModule.test_step 206–218 会传 batch_idx/save_attentions，而原 MLP eval_step 378–406 不接它们；属于保留的静态接口不兼容，未执行确认报错。固定 main 已修改普通 BaseModule 调用，不能把原件错误套到 main。原件 predict_step/attention返回协议等也不能凭文件能解析就宣称适配完成。

## 后续建议、检查目标与用户问题

`PROPOSED_NOT_IMPLEMENTED`：可能在 main 共享表之后加入同结构聚合器并取消/明确 SEP，保持数据/SID、decoder AR、loss/evaluator口径。预计改变encoder物品表示与序列长度，可能改变参数/计算量；不改变目标码定义。是否忠实保留末码mask、引入更严格整组校验、如何迁移checkpoint与冻结范围，需要讨论后单独授权。

`PROPOSED_NOT_IMPLEMENTED`：未来验证应逐层确认有序拼接、mask/目标边界、输出shape、梯度流、train/eval Dropout、参数键差异与初始化、AR前缀和beam/evaluator，以及训练预算公平性。只列目标；本阶段没有模型实例化、前向、loss测量、训练或实验脚本。

`UNKNOWN`：原作者历史预算、初始化/冻结细节及唯一 producer Python 尚不能确认。建议用户先读构造580–592，再读 encoder596–707，最后看 generate789–898；能够解释“序列变短为何仍 AR”后再决定适配方案。


## 冻结源码证据

本表的行号由实际发布字节核对；源码集合身份 `8b3149a98731ba7857e446e0f488489be0715181b841f8c7f9f169d3c4e16ac3`。baseline链接固定完整SHA；原件链接在CONTENT_COMMIT发布后由交付元数据绑定该完整SHA，原始捕获版本和每文件完整SHA始终保留。

|文件/符号/实际行范围|版本|源SHA-256|
|---|---|---|
|[references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_mlp.py::SemanticIDEncoderDecoder.__init__ L471–594](https://github.com/woobowen/recommend/blob/SOURCE_CONTENT_COMMIT/references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_mlp.py#L471-L594)|`A800-grid-20261010-v1`|`1b75bf94d6bd1187e7c30f996da33ebdc7ea2ffb33d8918a123991abbadafab9`|
|[references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_mlp.py::SemanticIDEncoderDecoder.encoder_forward_pass L596–707](https://github.com/woobowen/recommend/blob/SOURCE_CONTENT_COMMIT/references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_mlp.py#L596-L707)|`A800-grid-20261010-v1`|`1b75bf94d6bd1187e7c30f996da33ebdc7ea2ffb33d8918a123991abbadafab9`|
|[references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_mlp.py::SemanticIDEncoderDecoder.get_embedding_table L934–955](https://github.com/woobowen/recommend/blob/SOURCE_CONTENT_COMMIT/references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_mlp.py#L934-L955)|`A800-grid-20261010-v1`|`1b75bf94d6bd1187e7c30f996da33ebdc7ea2ffb33d8918a123991abbadafab9`|
|[references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_mlp.py::SemanticIDEncoderDecoder.generate L789–898](https://github.com/woobowen/recommend/blob/SOURCE_CONTENT_COMMIT/references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_mlp.py#L789-L898)|`A800-grid-20261010-v1`|`1b75bf94d6bd1187e7c30f996da33ebdc7ea2ffb33d8918a123991abbadafab9`|
|[references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_mlp.py::SemanticIDEncoderDecoder.model_step L1050–1103](https://github.com/woobowen/recommend/blob/SOURCE_CONTENT_COMMIT/references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_mlp.py#L1050-L1103)|`A800-grid-20261010-v1`|`1b75bf94d6bd1187e7c30f996da33ebdc7ea2ffb33d8918a123991abbadafab9`|
|[references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_mlp.py::SemanticIDDecoderModule.forward L1146–1186](https://github.com/woobowen/recommend/blob/SOURCE_CONTENT_COMMIT/references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_mlp.py#L1146-L1186)|`A800-grid-20261010-v1`|`1b75bf94d6bd1187e7c30f996da33ebdc7ea2ffb33d8918a123991abbadafab9`|
|[references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_mlp.py::T5MultiLayerFF L1241–1273](https://github.com/woobowen/recommend/blob/SOURCE_CONTENT_COMMIT/references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_mlp.py#L1241-L1273)|`A800-grid-20261010-v1`|`1b75bf94d6bd1187e7c30f996da33ebdc7ea2ffb33d8918a123991abbadafab9`|
|[references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_mlp.py::SemanticIDGenerativeRecommender.eval_step L378–406](https://github.com/woobowen/recommend/blob/SOURCE_CONTENT_COMMIT/references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_mlp.py#L378-L406)|`A800-grid-20261010-v1`|`1b75bf94d6bd1187e7c30f996da33ebdc7ea2ffb33d8918a123991abbadafab9`|
|[references/wmh-original/GRID/src/models/modules/base_module.py::BaseModule.test_step L206–218](https://github.com/woobowen/recommend/blob/SOURCE_CONTENT_COMMIT/references/wmh-original/GRID/src/models/modules/base_module.py#L206-L218)|`A800-grid-20261010-v1`|`7cfa9da5f73c0e666ab81f22440d04f04d6cc9624803de58cde3256dcae400b0`|


## 已保留的历史记录入口

下列全部 `HISTORICAL_OBSERVATION`，仅记录原文件和test行位置，未选择/重算指标，未证明当前Python唯一生成它们；T5历史上下文不替换Qwen工作资产。

- [logs/train/runs/2025-10-22/21-25-31 原config](https://github.com/woobowen/recommend/blob/SOURCE_CONTENT_COMMIT/references/wmh-original/GRID/logs/train/runs/2025-10-22/21-25-31/.hydra/config.yaml)，SHA-256 `9050b9de5ae2ed55ab68b6c653f968c04bf3272c17a1129c8f101498747799fd`；同目录overrides/hydra原文保留。 [完整CSV](https://github.com/woobowen/recommend/blob/SOURCE_CONTENT_COMMIT/references/wmh-original/GRID/logs/train/runs/2025-10-22/21-25-31/csv/version_0/metrics.csv) 的原test行：99，SHA-256 `a59494f88dddb851507bb1a56e7cfe4a185199fc47c3e37a469ab047612b1eb2`。
