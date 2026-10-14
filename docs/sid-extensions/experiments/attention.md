# Attention：四行是生成步骤，横轴是历史输入位置

## 研究问题与读图起点

`SOURCE_FACT + STATIC_INFERENCE`：decoder cross-attention 的 query 来自生成目标码位的decoder表示，key/value来自encoder历史表示。H4时四行分别对应预测四个码位的步骤，不是四个attention head。横轴是encoder输入位置（普通模型包含SID码位与SEP，MLP则每物品一token；user分支须看配置），不是四个目标码位。decoder self-attention关注BOS/已生成前缀，必须与cross分开。

`SOURCE_FACT`：固定普通main基线入口/数据/共享表/encoder/AR decoder/loss/evaluator见 [总图](../CODE_READING_GUIDE.md)。普通main没有当前变体那套完整注意力输出协议；不能仅把绘图脚本指向普通checkpoint就宣称能采集。本阶段没有替普通模型加入采集接口。

## 当前原件采集路径

|路径|实际正文|性质与限制|
|---|---|---|
|shuffle test链|原 `BaseModule.test_step`→shuffle `eval_step`525–575→generate(output_attentions=True)→`_save_attention_weights`446–523|需test hook置is_testing，非eval/predict等价|
|shuffle/MLP predict链|各自 `predict_step` 调带attention的generate并保存注意力；正文未保存输入mask|要独立阅读predict正文，不能由test开关推断行为|
|no_pos predict链|`predict_step`1451–1527保存attention；H>6时进入chunk生成|eval_step1584–1650虽接save_attentions但正文未用；chunk返回解包存在静态限制|

`SOURCE_FACT`：shuffle `_save_attention_weights` 的输出目录为相对cwd的 `./attention_weights`，命名含 `encoder_attention_batch_{batch_idx}.pt`、`cross_attention_batch_{batch_idx}.pt`、`cross_attention_full_batch_{batch_idx}.pt`、`decoder_self_attention_batch_{batch_idx}.pt`。当前三个变体的predict正文没有写出 `batch_mask_*.pt` 或稳定样本/输入伴随记录，这正是共同生产链缺口。仅有batch编号不等于跨来源样本相同；没有rank/producer身份的文件也不能自动拼成共同链。

`SOURCE_FACT`：`generate`按层返回attention；保存full cross时取最后一层，并在beam展开后按间隔K选择每样本当步首beam，在query维成功squeeze且各步形状兼容、stack成功时堆为 `[B,H,A,Lenc]`（A=head数）；若stack失败只记warning、不写full文件，原始逐步cross tuple仍另存。这是当步选择，不是最终获胜完整SID路径的祖先回溯。还保存逐步原cross tuple与decoder self。真实值尚未在本阶段读取/采集。

## 实际绘图函数怎样处理

`SOURCE_FACT`：`visualize_complete_decoder_attention.py::build_complete_decoder_attention`20–149 取各步最后一层，mean(dim=1)平均heads，取最后query位置；sample0特殊处理假设原batch32并选首beam。它把self向量填成H×H前缀块，再在142行实际拼接 `[cross_matrix,decoder_self_matrix]`。函数早期注释写self在前，与正文不一致，以正文为准；两个分别归一化的块拼接不构成一个联合attention分布。

`SOURCE_FACT`：`visualize_complete_decoder_attention`152–227完成热图及输入/前缀标签，`main`230–319按文件名词典排序、用位置索引选batch。`visualize_encoder_attention.py::detect_real_sequence_length`16–42、`visualize_encoder_attention`45–156、`visualize_all_layers`159–197处理encoder self-attention并以行列和启发式估计裁剪长度；它没有读取真实输入mask，属于另一种图。层平均/最后层、head平均、样本/query选择应逐函数核对，不能把“平均attention”概括所有路径。

`STATIC_INFERENCE`：普通H4/E128/L120且SEP启用/无user时cross横轴约150个槽；MLP则约30槽。不同来源/聚合长度不能直接按矩阵位置一一比较。它们含padding/被隐藏目标槽，需要输入mask和item/SID/SEP映射共同解释。四行的生成query依次为BOS、此前生成的第1/2/3码表示；teacher forcing的query来源又不同。

## 历史图、容器与 producer 边界

`HISTORICAL_OBSERVATION`：旧审计已检查528个历史容器，其中33个CRC冲突；495个正常只说明其已捕获内容可读，不能证明共同样本/checkpoint/最终beam/绘图producer。这里引用旧绑定证据，不再完整扫描528容器、不生成新tensor、不修图或重算数值相似度。

`HISTORICAL_OBSERVATION`：旧图、健康原矩阵和其档案恢复位置作为有边界的参考，详见旧证据继承摘要。当前源码含beam/sample/mask命名的生产链限制；不把当前脚本唯一绑定历史图。`no_pos`只关闭/清零/冻结decoder relative bias，encoder相对位置、码位offset、SEP和AR前缀仍在，不能称整模型无位置。

`STATIC_INFERENCE`：no_pos H>6 chunk分支临时把beam限为min(3,K)，attention=True时核心返回5项而chunk解包2项，是保留的静态接口冲突，未执行确认报错。原件及历史结果不为“通过静态检查”而改写。

## 参数、建议和未来验收

`STATIC_INFERENCE`：仅返回/保存attention在理想观测接口下不新增可训练参数，但开关、实现后端、cache/beam选择和内存使用可能改变运行行为；本阶段没有以运行证明“零影响”。no_pos冻结decoder relative bias会改变可训练参数集合，与单纯观测不同。采集通常不引入新的训练目标，若引入则须另行讨论。

`PROPOSED_NOT_IMPLEMENTED`：未来可在固定main上添加独立观测接口，记录完整源码/配置/checkpoint/数据hash、稳定样本与item轴、rank、mask、SID/SEP位置、每步query、各层/head、beam parent indices以及最终路径回放。目的为把图绑定真实生产链；预计新增返回/记录点，不提前修旧图算法、不默认改变模型参数或训练目标。本阶段没有实施。

`PROPOSED_NOT_IMPLEMENTED`：未来应核开关关闭等价、attention维度/mask、最终beam祖先、跨run同样本、层/head聚合和颜色/归一化语义，再讨论新采集/回放/数值统计。这里仅列证据目标，不执行。用户需要选择teacher forcing还是生成路径、最后层还是逐层、head平均还是分head、单样本还是统计比较，以及是否接受明确的新采集协议。先从绘图函数142行和保存函数的首beam选择读起。


## 冻结源码证据

本表的行号由实际发布字节核对；源码集合身份 `8b3149a98731ba7857e446e0f488489be0715181b841f8c7f9f169d3c4e16ac3`。baseline链接固定完整SHA；原件链接在CONTENT_COMMIT发布后由交付元数据绑定该完整SHA，原始捕获版本和每文件完整SHA始终保留。

|文件/符号/实际行范围|版本|源SHA-256|
|---|---|---|
|[references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py::SemanticIDGenerativeRecommender._save_attention_weights L446–523](https://github.com/woobowen/recommend/blob/84d4533fbdd6b010f54d671b433a9952608911ca/references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py#L446-L523)|`A800-grid-20261010-v1`|`e03082ad87bfc842c05467580fc479eff4345c6a0d18c5b2f7faf320d4f96c80`|
|[references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py::SemanticIDGenerativeRecommender.eval_step L525–575](https://github.com/woobowen/recommend/blob/84d4533fbdd6b010f54d671b433a9952608911ca/references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py#L525-L575)|`A800-grid-20261010-v1`|`e03082ad87bfc842c05467580fc479eff4345c6a0d18c5b2f7faf320d4f96c80`|
|[references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py::SemanticIDEncoderDecoder.generate L952–1061](https://github.com/woobowen/recommend/blob/84d4533fbdd6b010f54d671b433a9952608911ca/references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py#L952-L1061)|`A800-grid-20261010-v1`|`e03082ad87bfc842c05467580fc479eff4345c6a0d18c5b2f7faf320d4f96c80`|
|[references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py::SemanticIDEncoderDecoder.predict_step L1121–1212](https://github.com/woobowen/recommend/blob/84d4533fbdd6b010f54d671b433a9952608911ca/references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py#L1121-L1212)|`A800-grid-20261010-v1`|`e03082ad87bfc842c05467580fc479eff4345c6a0d18c5b2f7faf320d4f96c80`|
|[references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_mlp.py::SemanticIDEncoderDecoder.predict_step L957–1048](https://github.com/woobowen/recommend/blob/84d4533fbdd6b010f54d671b433a9952608911ca/references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_mlp.py#L957-L1048)|`A800-grid-20261010-v1`|`1b75bf94d6bd1187e7c30f996da33ebdc7ea2ffb33d8918a123991abbadafab9`|
|[references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_no_pos_embed.py::SemanticIDEncoderDecoder.predict_step L1451–1582](https://github.com/woobowen/recommend/blob/84d4533fbdd6b010f54d671b433a9952608911ca/references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_no_pos_embed.py#L1451-L1582)|`A800-grid-20261010-v1`|`7561d2f494764a77efac1d71abff4ea15bc683eea0816c1e8fe067d7d5c78463`|
|[references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_no_pos_embed.py::SemanticIDEncoderDecoder.eval_step L1584–1650](https://github.com/woobowen/recommend/blob/84d4533fbdd6b010f54d671b433a9952608911ca/references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_no_pos_embed.py#L1584-L1650)|`A800-grid-20261010-v1`|`7561d2f494764a77efac1d71abff4ea15bc683eea0816c1e8fe067d7d5c78463`|
|[references/wmh-original/GRID/visualize_complete_decoder_attention.py::build_complete_decoder_attention L20–149](https://github.com/woobowen/recommend/blob/84d4533fbdd6b010f54d671b433a9952608911ca/references/wmh-original/GRID/visualize_complete_decoder_attention.py#L20-L149)|`A800-grid-20261010-v1`|`531bbaf5ab118ffedb3b0fef32dd0fe26748298b64e3c7e92a68c911c673e177`|
|[references/wmh-original/GRID/visualize_complete_decoder_attention.py::visualize_complete_decoder_attention L152–227](https://github.com/woobowen/recommend/blob/84d4533fbdd6b010f54d671b433a9952608911ca/references/wmh-original/GRID/visualize_complete_decoder_attention.py#L152-L227)|`A800-grid-20261010-v1`|`531bbaf5ab118ffedb3b0fef32dd0fe26748298b64e3c7e92a68c911c673e177`|
|[references/wmh-original/GRID/visualize_complete_decoder_attention.py::main L230–319](https://github.com/woobowen/recommend/blob/84d4533fbdd6b010f54d671b433a9952608911ca/references/wmh-original/GRID/visualize_complete_decoder_attention.py#L230-L319)|`A800-grid-20261010-v1`|`531bbaf5ab118ffedb3b0fef32dd0fe26748298b64e3c7e92a68c911c673e177`|
|[references/wmh-original/GRID/visualize_encoder_attention.py::detect_real_sequence_length L16–42](https://github.com/woobowen/recommend/blob/84d4533fbdd6b010f54d671b433a9952608911ca/references/wmh-original/GRID/visualize_encoder_attention.py#L16-L42)|`A800-grid-20261010-v1`|`25622305bddb1d93e93d960ee8263a26e2ca8f601e6ab6be13ad718af0e380c4`|
|[references/wmh-original/GRID/visualize_encoder_attention.py::visualize_encoder_attention L45–156](https://github.com/woobowen/recommend/blob/84d4533fbdd6b010f54d671b433a9952608911ca/references/wmh-original/GRID/visualize_encoder_attention.py#L45-L156)|`A800-grid-20261010-v1`|`25622305bddb1d93e93d960ee8263a26e2ca8f601e6ab6be13ad718af0e380c4`|
|[references/wmh-original/GRID/visualize_encoder_attention.py::visualize_all_layers L159–197](https://github.com/woobowen/recommend/blob/84d4533fbdd6b010f54d671b433a9952608911ca/references/wmh-original/GRID/visualize_encoder_attention.py#L159-L197)|`A800-grid-20261010-v1`|`25622305bddb1d93e93d960ee8263a26e2ca8f601e6ab6be13ad718af0e380c4`|
|[references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_no_pos_embed.py::(全文) L1–1888](https://github.com/woobowen/recommend/blob/84d4533fbdd6b010f54d671b433a9952608911ca/references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_no_pos_embed.py#L1-L1888)|`A800-grid-20261010-v1`|`7561d2f494764a77efac1d71abff4ea15bc683eea0816c1e8fe067d7d5c78463`|


## 已保留的历史记录入口

下列全部 `HISTORICAL_OBSERVATION`，仅记录原文件和test行位置，未选择/重算指标，未证明当前Python唯一生成它们；T5历史上下文不替换Qwen工作资产。

- [logs/inference/runs/2025-11-14/18-26-28 原config](https://github.com/woobowen/recommend/blob/84d4533fbdd6b010f54d671b433a9952608911ca/references/wmh-original/GRID/logs/inference/runs/2025-11-14/18-26-28/.hydra/config.yaml)，SHA-256 `2a9835f85e596f0a8d936ec96d8c782d53b99e5ff50b83f809fd26a6bf7453e2`；同目录overrides/hydra原文保留。 该选中历史上下文没有对应已恢复CSV，不补造结果；attention原容器仍留旧档案。
- [logs/inference/runs/2025-10-29/15-47-20 原config](https://github.com/woobowen/recommend/blob/84d4533fbdd6b010f54d671b433a9952608911ca/references/wmh-original/GRID/logs/inference/runs/2025-10-29/15-47-20/.hydra/config.yaml)，SHA-256 `c7c58df3dd7eaebca64bd316060540344a9cc7b81dd84bd5dbd94147c76f534f`；同目录overrides/hydra原文保留。 [完整CSV](https://github.com/woobowen/recommend/blob/84d4533fbdd6b010f54d671b433a9952608911ca/references/wmh-original/GRID/logs/inference/runs/2025-10-29/15-47-20/csv/version_0/metrics.csv) 的原test行：2，SHA-256 `5dc99b63777675e2a5def3720d99d505e43cf9ba5e173ce3a1ab2fb7db85a222`。


## 2026-10-14 本轮采集范围与边界

批准方案：在查看attention/命中之前固定实际evaluation文件顺序前32不同用户；最终Top1完整祖先为主，局部首beam另标，最后层/head均值，保留逐head必要矩阵/beam父链/位置与mask，不定相似阈值。A/B/MLP各自32用户数值JSON与来源顺序已固定，见本轮机器证据；未执行32用户正式采集。

真实工程采集仅A、B、3-update MLP各3用户，各4码位生成、K10。OFF/ON预测SID与score精确相等、CPU/CUDA RNG相同、参数hash相同、父链回溯等于最终Top1、正常/异常hook恢复、数值NPZ/SHA与PNG重读通过。矩阵分别150（A/B含SEP）和30（MLP逐物品）encoder位置，不拼成同一联合概率；全部位置保留、原值共用色标，不裁剪padding/热区。

B早期工程图用旧首文件3用户；A/MLP工程图用后来核实的DataModule文件assignment首3。不要把这两组图当同用户配对比较。3-update MLP图仅验证链路，不能解释成有效正式MLP的现象。图流程通过不证明四行相似或SID顺序无用；本轮未作科学阈值统计或现象复现声明。完整本轮状态见 [执行记录](shuffle.md#2026-10-14-beauty-执行partial_blocked)。

用户批准MLP独立单卡方案后，`beauty-mlp-fixed8-attention-v1` 又实际检查旧固定32用户的前8位（不是重新选择）：真实四步/K10生成、逐head矩阵、双路径与8张PNG重读通过，OFF/ON SID/score/RNG/参数/清理均精确一致。该3-update工程模型80/80候选不在完整SID表内，父链全部为beam0，因此本批双口径恰好相同，不能声称实测覆盖非零祖先或说明正式模型现象。正式 `mlp_attention.py capture` 只在训练结果status=COMPLETE、formal训练和完整test完成、runner SUCCEEDED/exit0/cleanup、checkpoint/SID/config hash均匹配后开放，保存这些身份；工程8人路径与采集器正文未改。32用户正式图仍等待有效新MLP模型。

## 2026-10-15 新MLP正式32用户采集

上述工程图之后，`beauty-mlp-formal-attention32-v1` 已用本次完成的新MLP best2600严格加载并成功采集固定32用户（四批×8），exit0/worker组退出已核验。训练/完整test完成回执、checkpoint SHA `52f18292490825bc7e172e762600b428a8dcb8c511d2dfc3d8f4fe4621335891`、T5/RQ-VAE+STE SID和原配置逐一绑定，未使用step3工程权重。采集源码提交 `f6e8591f7e6c34ed3de0f58f2d398ac660e07ad4`，快照 `52258f6a7f60c292d290e1a777aee3f42e649623cce6fe711242b4963543dd64`；研究Python与正式训练提交逐字相同，仅增加实际完成结果的采集/曲线输入JSON。

事前固定的evaluation用户文件SHA `dbe80110171ef657bd861fc2aa2d50b657086d0abceb83b640f1079881b48aa8` 未变。全部32顺序、input SID/mask、item/位置和target引用核验一致，没有按attention或命中筛选。保存实际四码位AR生成、beam父索引/前缀/query身份、最后decoder层逐head cross-attention原值、head均值双口径图和完整30个encoder位置；只保存cross-attention，未与decoder self-attention拼接，padding mask单列。主口径为最终Top1完整祖先，历史口径是每步局部首parent beam，不拼接成一条虚构候选轨迹。没有生成后重放或重新随机化历史。

非作者逐原矩阵/前缀/父链审核：21/32样本出现非零Top1祖先，父索引表共796个非零项；若误用选择后的child序号，29个步骤会取到不同的原矩阵，因而这批真实覆盖了工程模型全beam0未覆盖的风险。四批采集OFF/ON回执的SID/score误差均0、CPU/CUDA RNG/参数/hook恢复相同；OFF原矩阵未额外存盘，审计明确区分运行时比较回执与可离线重算的ON原矩阵。32张PNG均核hash并实际重读，共用原始色标、无逐行归一化、裁剪热区或padding插值。

网页精简展示规则固定为第一批原矩阵/记录及sample index0的图，全部32图与矩阵保留在轻量ZIP。图中sample index是批内0..7；全局序号为8×batch+批内index，身份以record.json的sample key为准，四批的index0是四个人。这张固定首样例的四步均主要关注encoder位置0和5，权重大小仍不同，逐步峰值约0.2645/0.3973/0.4436/0.4645。原矩阵的描述性核查中25/32样本四行峰位相同，但峰位相同不等于整行分布相同，也不等于总体遵循某个相似规律。未设置科学相似阈值、未追加六对行间差异统计、不据图宣称SID顺序无用。图生产链成功与现象是否具有普遍性分开报告。

入口：[正式采集结果](../../../manifests/beauty-extensions/runs/beauty-mlp-formal-attention32-v1/RESULT.json)、[第一批位置/身份/父链记录](../../../manifests/beauty-extensions/runs/beauty-mlp-formal-attention32-v1/batch-00/bundle/record.json)、[第一批数值NPZ](../../../manifests/beauty-extensions/runs/beauty-mlp-formal-attention32-v1/batch-00/bundle/arrays.npz)（allow_pickle=False）、[固定首图](../../../manifests/beauty-extensions/runs/beauty-mlp-formal-attention32-v1/batch-00/figures/sample_000_cross_attention.png)、[非作者独立审核](../../../manifests/beauty-extensions/b-attention-review.json)。A/B的正式Normal/Shuffle32人图仍未运行；不同SID/模型/encoder长度的图不作像素级或严格配对比较。整体仍为PARTIAL_BLOCKED。
