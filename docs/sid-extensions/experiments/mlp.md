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
|[references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_mlp.py::SemanticIDEncoderDecoder.__init__ L471–594](https://github.com/woobowen/recommend/blob/84d4533fbdd6b010f54d671b433a9952608911ca/references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_mlp.py#L471-L594)|`A800-grid-20261010-v1`|`1b75bf94d6bd1187e7c30f996da33ebdc7ea2ffb33d8918a123991abbadafab9`|
|[references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_mlp.py::SemanticIDEncoderDecoder.encoder_forward_pass L596–707](https://github.com/woobowen/recommend/blob/84d4533fbdd6b010f54d671b433a9952608911ca/references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_mlp.py#L596-L707)|`A800-grid-20261010-v1`|`1b75bf94d6bd1187e7c30f996da33ebdc7ea2ffb33d8918a123991abbadafab9`|
|[references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_mlp.py::SemanticIDEncoderDecoder.get_embedding_table L934–955](https://github.com/woobowen/recommend/blob/84d4533fbdd6b010f54d671b433a9952608911ca/references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_mlp.py#L934-L955)|`A800-grid-20261010-v1`|`1b75bf94d6bd1187e7c30f996da33ebdc7ea2ffb33d8918a123991abbadafab9`|
|[references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_mlp.py::SemanticIDEncoderDecoder.generate L789–898](https://github.com/woobowen/recommend/blob/84d4533fbdd6b010f54d671b433a9952608911ca/references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_mlp.py#L789-L898)|`A800-grid-20261010-v1`|`1b75bf94d6bd1187e7c30f996da33ebdc7ea2ffb33d8918a123991abbadafab9`|
|[references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_mlp.py::SemanticIDEncoderDecoder.model_step L1050–1103](https://github.com/woobowen/recommend/blob/84d4533fbdd6b010f54d671b433a9952608911ca/references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_mlp.py#L1050-L1103)|`A800-grid-20261010-v1`|`1b75bf94d6bd1187e7c30f996da33ebdc7ea2ffb33d8918a123991abbadafab9`|
|[references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_mlp.py::SemanticIDDecoderModule.forward L1146–1186](https://github.com/woobowen/recommend/blob/84d4533fbdd6b010f54d671b433a9952608911ca/references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_mlp.py#L1146-L1186)|`A800-grid-20261010-v1`|`1b75bf94d6bd1187e7c30f996da33ebdc7ea2ffb33d8918a123991abbadafab9`|
|[references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_mlp.py::T5MultiLayerFF L1241–1273](https://github.com/woobowen/recommend/blob/84d4533fbdd6b010f54d671b433a9952608911ca/references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_mlp.py#L1241-L1273)|`A800-grid-20261010-v1`|`1b75bf94d6bd1187e7c30f996da33ebdc7ea2ffb33d8918a123991abbadafab9`|
|[references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_mlp.py::SemanticIDGenerativeRecommender.eval_step L378–406](https://github.com/woobowen/recommend/blob/84d4533fbdd6b010f54d671b433a9952608911ca/references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_mlp.py#L378-L406)|`A800-grid-20261010-v1`|`1b75bf94d6bd1187e7c30f996da33ebdc7ea2ffb33d8918a123991abbadafab9`|
|[references/wmh-original/GRID/src/models/modules/base_module.py::BaseModule.test_step L206–218](https://github.com/woobowen/recommend/blob/84d4533fbdd6b010f54d671b433a9952608911ca/references/wmh-original/GRID/src/models/modules/base_module.py#L206-L218)|`A800-grid-20261010-v1`|`7cfa9da5f73c0e666ab81f22440d04f04d6cc9624803de58cde3256dcae400b0`|


## 已保留的历史记录入口

下列全部 `HISTORICAL_OBSERVATION`，仅记录原文件和test行位置，未选择/重算指标，未证明当前Python唯一生成它们；T5历史上下文不替换Qwen工作资产。

- [logs/train/runs/2025-10-22/21-25-31 原config](https://github.com/woobowen/recommend/blob/84d4533fbdd6b010f54d671b433a9952608911ca/references/wmh-original/GRID/logs/train/runs/2025-10-22/21-25-31/.hydra/config.yaml)，SHA-256 `9050b9de5ae2ed55ab68b6c653f968c04bf3272c17a1129c8f101498747799fd`；同目录overrides/hydra原文保留。 [完整CSV](https://github.com/woobowen/recommend/blob/84d4533fbdd6b010f54d671b433a9952608911ca/references/wmh-original/GRID/logs/train/runs/2025-10-22/21-25-31/csv/version_0/metrics.csv) 的原test行：99，SHA-256 `a59494f88dddb851507bb1a56e7cfe4a185199fc47c3e37a469ab047612b1eb2`。


## 2026-10-14 本轮真实工程检查

本轮状态为 PARTIAL_BLOCKED，完整条件/修正边界见 [Shuffle执行记录](shuffle.md#2026-10-14-beauty-执行partial_blocked)。

使用真正Beauty原有效配置及其 T5/RQ-VAE+STE SID，不复用普通模型ckpt、不重评旧MLP当新训练。原逐物品聚合 `[B,M,4,128] → [B,M,512] → [B,M,128]`、Linear512→256/ReLU/Dropout0.1/Linear256→128（均有bias）、共享聚合、无SEP/末码mask/AR decoder均保留。实际首训练batch扩展405窗口，末microbatch493等；raw microbatch32不是固定512个训练窗口。

真实工程训练2 optimizer update（32 microbatch）→validation→保存161,299,736 B checkpoint；独立进程strict恢复后再1 update，global_step2→3，非零有限MLP梯度/更新、优化器覆盖通过。原类/派生同shape hidden/logits/loss/生成SID/score相同，重复/换序/回batch通过，19用户真实test完成。恢复改变工程输出目录，Lightning未恢复部分checkpoint callback路径状态；数据游标和RNG轨迹不宣称与不中断完全相同。全部预跑状态禁止流入正式训练。

短训含验证/保存约9.32秒，恢复含验证/保存约5.57秒，峰值显存分别1,965,480,448/2,023,921,664 B。这些是observed smoke成本，不是完整训练预算或公平时延结论。3-update工程cp SHA `49eb045c6346c32d87be3605f59a3bcf9504d7da87495e659ddd92b106958290`，只用于接口/attention预跑。

最新用户批准已取代早期的 NEEDS_USER_DECISION：采用上述真正Beauty T5/RQ-VAE+STE SID、从零初始化、全参数训练、Adam1e-3/weight_decay1e-4/无scheduler/seed42；单4090原始microbatch32、accum64、每6400 microbatch验证（名义每100次更新），原val/recall@5最大值选模/patience10/max_steps320000/max_epochs10。12小时是硬资源上限，不能冒称原训练终止条件。用户额外要求正式前实测累积64的更新与完整验证成本；明显无法在预算内完成时暂停正式。MLP与A、B的SID条件不同，无匹配Normal时只披露比较限制，不补训。

保存的独立入口 `experiments/beauty_extensions/mlp_execution.py` 及 `specs/mlp-approved-resource.json` 只负责这一新获批准的资源检查：独立CPU进程按实际evaluation分配建立完整目标multiset索引，另一进程从零做2次真实accum64更新，再执行一次无limit的完整Trainer.validate。保留workers8、K10、原模型大小/精度/损失/evaluator；记录实际collate窗口、优化器覆盖/梯度/更新、全量覆盖和成本。全部工程状态仍禁止流入正式新训练。这不是对已三轮后B故障的第四次修正，B研究源码未改。

历史估时参照已独立核验：真Beauty MLP run `2025-10-24/15-10-17` CSV含35次验证（step99..3499）；最佳checkpoint global_step2500，最后测试step3500。最佳后10次验证未改善与patience10吻合，但日志没有显式停止原因。原四卡fit含验证7079.261秒，不能迁移成4090吞吐。新估时场景为 `3500×实测每64microbatch更新成本 + 35×完整验证成本 + 测试/保存/启动开销`，历史终点不保证新run同处早停；无限训练IterableDataset也不能把max_epochs10解释成有限十轮。

`beauty-mlp-evaluation-index-v1` 与 `beauty-mlp-accum64-fullval-v1` 已实际成功且非作者复核。后者从零执行128 microbatch=2 optimizer update，原始名义4096行、实际53,210 collate窗口（每microbatch265..512，均值415.703125）；优化器覆盖13,292,160个可训练参数，有限梯度，聚合MLP四个参数均更新，保存的Adam step均为2。完整验证2,796 batch（2,795×8+3）覆盖22,363个evaluation目标/用户，key-target multiset与独立CPU索引精确一致，未减少样本。峰值allocated 2,093,154,304 B/reserved 2,891,972,608 B。两次更新分别5.5124/4.3641秒；完整验证外层136.6970秒、hook内135.6603秒。工程checkpoint禁止流入正式。

采用实测第二次更新与外层验证成本，历史3500/35场景约5.572小时（尚未包含完整测试、额外保存和attention）；没有改善时最早patience10场景约1.751小时；配置320000上界约509.425小时，不能承诺12小时内触发科学终止。正式硬超时43200秒，提前180秒预留状态保存/worker退出；写入挂载点安全余量20GiB，预计新增结果/两份checkpoint/有限attention小于1GiB。该预留触发或runner超时均标未完成，不改max_steps/epochs/earlystop，不自动续跑。GPU1一张，GPU0同学任务未触碰。

`beauty-mlp-lifecycle-v1` 首跑已成功：独立从零1次acc64更新、64 microbatch/25,490展开窗口，完整val同样22,363目标；实际原ModelCheckpoint、EarlyStopping(patience10/min_delta0)、RichModelSummary、CSV logger全部启用。原val/recall@5最大值规则实际保存工程best，随后weights_only+局部safe_globals strict加载，键/shape/值、内嵌SID、共享表和optimizer/callback存在性均核验；同一Trainer真实test16用户/2 batch，预测与原evaluator CPU复算精确一致。工程四项指标均0，160/160候选SID无效（原should_check_prefix=False，未过滤或补齐），不是新正式模型结论。工程best/global_step1、r2/r3及resource状态均禁止流入正式。

这次含原回调/完整覆盖检查的完整validation hook实测190.601秒，fit含一次更新/验证/保存为199.075秒，单次更新4.769秒。比最初136.697秒资源验证多出的观测/检查成本保留；以4.36–4.77秒/update及约191秒/validation估计，历史3500/35场景约6.1–6.5小时，再留测试/保存余量，预计约6.2–7小时。仍需真实早停，不保证完成于12小时。正式入口为 `mlp_formal.py formal --spec specs/mlp-formal.json`，实际必须经deploy保存runner request调用；非作者gate绑定canonical计划和所有部署研究Python hash，正式部署要求全部源码/配置来自已提交HEAD字节。

## 2026-10-15 正式新训练结果

`beauty-mlp-formal-v1` 已 SUCCEEDED/exit0，worker组退出并核验。运行源码提交 `89b6d961c81333b0a44c4eb17527ced11897849f`，服务器为无.git的固定export快照 `12e70793731a8329504a50e854e467be64af230a4ce7f9d46014b17b1bc16299`。准入gate绑定实际MLP路径的95份Python；部署清单另包含工作baseline等共167文件/155份Python，两者范围不同。后续提交只增加报告、结果和采集输入，研究源码及正式训练spec保持逐字不变。

从零、全参数、单4090/raw32/accum64/val6400及其余已批准原配置实际执行。第3600次optimizer update按原patience10早停，230400 microbatch、95287516个collate窗口、36次完整验证；每次2796 batch精确覆盖22363个目标。验证Recall@5唯一最佳为step2600的0.05871305242180824，之后10次均未改善；未使用test选模、未延长预算。fit实测20815.297秒，完整runner时长20993.907秒（约5小时50分），在12小时硬上限内自然结束。allocated峰值2092865536 B，reserved峰值3351248896 B。

最佳checkpoint为 `checkpoint_epoch=000_step=002600.ckpt`，161258134 B，SHA-256 `52f18292490825bc7e172e762600b428a8dcb8c511d2dfc3d8f4fe4621335891`。weights_only/局部safe_globals严格加载180个state tensor，键/shape/值、内嵌SID和共享表核验通过。另存训练终点step3600的完整optimizer/callback/RNG恢复状态SHA `60823ff3aabf363d843f7679d719541cc1558a3eba60492616e8ac45c66df403`，未将best权重和终点优化器混称一个可精确续跑状态；IterableDataset worker游标/预取队列未保存，不宣称恢复轨迹等同不中断运行。

完整真实test为22363用户/目标、每人K10、四码位AR生成。指标为比例值：

|本次新MLP独立条件|Recall@5|NDCG@5|Recall@10|NDCG@10|
|---|---:|---:|---:|---:|
|T5 / RQ-VAE+STE|0.04333050176501274|0.030580179765820503|0.061709072440862656|0.036571044474840164|

完整预测与目标multiset精确一致，969个Top5命中、1380个Top10命中。原evaluator CPU复算与Trainer四项差为0；非作者独立stdlib公式复算最大差8.73e-10，逐样本最大差2.14e-8（浮点累加差）。223630候选中24个SID不在完整物品SID集合，重复候选0；保留原should_check_prefix=False及评价行为，没有过滤/补齐/重排。纯数字SID审核导出支持独立合法性复核，没有执行pickle或重新生成SID。

本结果是本次新训练，工程step3模型和旧历史MLP重评均不替代它。与Shuffle A的T5/R-Kmeans、B的Qwen/R-Kmeans条件不同；本轮没有匹配Normal，不报告跨SID严格结构delta，也不额外训练baseline。A/B六个正式评测仍NOT_RUN，故整个一次性任务为PARTIAL_BLOCKED。

核验入口：[训练结果](../../../manifests/beauty-extensions/runs/beauty-mlp-formal-v1/training_result.json)、[严格best加载](../../../manifests/beauty-extensions/runs/beauty-mlp-formal-v1/best_checkpoint_load.json)、[原始CSV](../../../manifests/beauty-extensions/runs/beauty-mlp-formal-v1/csv/version_0/metrics.csv)、[完整验证记录](../../../manifests/beauty-extensions/runs/beauty-mlp-formal-v1/validations.jsonl)、[正式曲线](../../../manifests/beauty-extensions/runs/beauty-mlp-training-curves-v2/training_validation_curves.png)、[非作者审核](../../../manifests/beauty-extensions/final-review.json)。全量预测/逐样本指标/原日志/终态回执在最终轻量ZIP；大型权重保留new-server原job输出目录。

## 2026-10-15 Toys/Sports：2+1工程预跑已完成，正式条件待确认

两域均 **ENGINEERING_PREFLIGHT_PASS**：各自从零2次optimizer update，保存后在独立进程严格恢复，再1次update，val11/test19及预测复算闭合。总体 **PARTIAL_PREFLIGHT**，因为两域 `RESEARCH_CONDITIONS_CONFIRMED=false`、`READY_FOR_FORMAL_REVIEW=false`；不将工程候选冒称已批准正式方案。Beauty已完成正式MLP、最佳checkpoint及原预测均未重跑/修改。

|域|真实来源和工程采用值|结果与正式边界|
|---|---|---|
|Toys|作者run `2025-10-22/21-25-31` config SHA `9050b9de…`；RQ-VAE+STE SID SHA `26bae0c9…`，shape[4,11924]、U255；H4/E128、原raw32/collate cap512/accum16/Adam.001/wd.0001/no scheduler/seed42，scratch全参数|13,291,136参数，4个MLP参数均有限梯度且实际更新；原档案另有真正Toys R-Kmeans候选。上游T5 factory与Qwen路径冲突、唯一producer Python缺口继续披露；未来需批准整套条件与单卡协议/预算|
|Sports|本域既有Qwen/R-Kmeans SID SHA `89b2c2bb…`，shape[4,18357]、U256；原普通工作config SHA `59a6a756…`，raw16/cap128/accum16/Adam.001/wd.0001/no scheduler/seed42，scratch工程模型|13,292,160参数，4个MLP参数均有限梯度且实际更新；HISTORICAL_CONDITION_UNCONFIRMED。普通配置不是作者Sports MLP历史设置；未来须明确批准本域SID+作者MLP的新结构实验、初始化和预算|

结构直接继承保全作者MLP及已验证Beauty adapter：逐物品 `[B,M,4,128]→[B,M,512]→Linear512,256→ReLU→Dropout0.1→Linear256,128`，MLP跨全部历史物品共享，无SEP；本次有效config无user table，末码物品mask，decoder仍四码位自回归。不是整段用户历史聚合，也未重写loss/生成/evaluator。三split×8条item→SID/label/mask、部分有效组拒绝、真实尾批test8/8/3及val8/3均通过；原TFRecord/SID/item映射未改。

Dropout按原结构和Lightning fit/validation/test的train/eval生命周期处理；独立性probe显式eval后恢复状态，代码与真实训练/评价链已经审核。没有逐阶段Dropout.training字段采样日志，不能伪称记录了该字段。原件与适配版空长度输入同样抛IndexError，全padding行为同样返回有限张量；这是保留的边界，不宣称空历史科学输入已被完整支持。

|域|原scratch job /本次独立恢复job|实际更新、窗口和时间|峰值GPU显存|
|---|---|---|---|
|Toys|`gsp-20261015-toys-short-r1` / `gsp-20261015-toys-resume-r3`|2+1更新，32+16 microbatch，13,463+6,726 collate windows；训练/val/保存/test段10.376+7.043秒；worker总118.185+96.250秒|scratch allocated1,996,942,336 B/reserved3,806,330,880 B；恢复reserved3,026,190,336 B|
|Sports|`gsp-20261015-sports-short-r1` / `gsp-20261015-sports-resume-r1`|2+1更新，32+16 microbatch，3,744+1,879 collate windows；对应段9.150+6.210秒；worker总95.093+29.255秒|scratch allocated693,331,456 B/reserved750,780,416 B；恢复reserved742,391,808 B|

全部上述job均SUCCEEDED/0且cleanup verified。独立恢复模型所有键/shape/value、optimizer/global_step严格一致，恢复前19条SID/target/order及得分误差0；EarlyStopping exact。ModelCheckpoint换输出目录后按现装Lightning仅恢复best_model_path，其目录绑定排序状态重置；原状态/当前状态逐字段记录。没有RNG/数据游标连续恢复保证，不宣称不中断训练等价。工程checkpoint及optimizer状态禁止成为未来正式训练起点。

**不利结果照实保留。** 每域scratch及恢复后test各190/190候选不在合法SID集合，原prefix约束false，原evaluator未过滤、补全或重排，Recall/NDCG均0。四码位输出/有限score/严格恢复与评价接口通过，不等于几次更新具有推荐效果；不得把它们用作正式指标。

原Toys恢复r1在on_train_start记录callback的-inf sentinel时JSON失败，尚未新增update；已有round2递归编码修正保留有限loss/梯度参数检查。r2为manifest身份拒绝，无worker/GPU。接管只验证已有round3commit+files部署补丁，在新的不可变快照c42c3fcc9bba292f上完成两域恢复，没有第4轮研究修复。所有旧失败保留。

资源预算估计仅以以上真实窗口、峰值及训练/验证/保存/test成本为依据；两个更新不足以给出可靠全量ETA，不用CPU导入耗时或Beauty步数冒充本域正式预算。预跑原硬限每job1200秒、总目标2GPUh/修正上限6GPUh，累计保守请求wall0.254719264GPUh，全部本任务job已终止。两个工程spec仍2+1更新、val11/test19，正式candidate不继承这些限制且始终未授权。

仅剩未来正式决策：Toys首选沿本次已配对RQ-VAE+STE整套及原优化器/选模规则，用户批准单卡累积、验证节奏与停止/资源预算；原3卡名义raw32×acc16推导单卡acc48/val4800只是一项候选，未批准且不保证历史轨迹等价。Sports首选明确批准“本域Qwen/RK+作者逐物品MLP”的新结构实验及独立初始化/预算；若要求历史复现，需可信Sports作者bundle。不套用Beauty acc64/val6400/12h，不将历史best3800/终测4800变成新停止步数。

真实代码/配置、来源审核及逐job身份见 [EXECUTION_SUMMARY.json](../../../manifests/grid-shuffle-mlp-preflight-20261015/EXECUTION_SUMMARY.json) 和同目录独立审核；原request/status/log、19条预测、callback状态与必要来源原文在校验后的轻量ZIP。当前停止，待GPT网页版二审与用户批准，不启动正式训练或注意力采集。

CPU曲线v1的请求因plot/capture阶段生成不同manifest身份串而被runner在接受前拒绝，未启动worker。保留原拒绝请求/回执，v2只绑定已核不可变快照的真实manifest SHA，调用同一冻结绘图脚本和完整CSV成功，PNG实际重读。此为提交身份处理，未修改研究源码、训练/评价设置或重跑模型；[完整依据](../../../manifests/beauty-extensions/curves-submission-binding.json)随交付保留。

## 2026-10-15 Toys/Sports 正式条件批准

用户本轮明确批准：Toys沿已配对作者RQ-VAE+STE整套，从零全参数训练，单4090 raw32/acc48/val4800（每100更新完整验证）；Sports沿本域Qwen/RK SID+原逐物品MLP，从零全参数，raw16/acc16/val1600，明确新结构实验。两域各12小时总硬额度包含导入、训练、完整验证、strict best、完整test及收尾，不是12+1小时；预计明显超额先暂停该域，不自动延长或改科学停止协议。旧预跑当时未批准记录保留历史含义。

固定：seed42、原FP32配置及src/train.py matmul处理、Adam .001/wd .0001、无scheduler、H4/E128/L120、无user table、逐物品共享MLP/Dropout.1/noSEP、四步AR K10/prefixfalse；val/recall@5 max选best，patience10/min_delta0，max_steps320000/max_epochs10。Toys acc48/val4800是原3卡名义batch/验证更新节奏换算，不保证DDP随机轨迹等价；Sports原工作协议不是历史Sports作者MLP复现。

新train入口复用旧Beauty已验收的完整val/test多重集检查、有限梯度/参数更新、保存RNG/optimizer/callback、strict best含SID与共享表、原evaluator复算；仅将Beauty固有raw/acc/cap/workers守卫改为本域批准cfg。完整域CPU metadata/index与成本场景估算须正式启动前完成；2+1工程checkpoint禁止作为起点。12h耗尽保存并记INCOMPLETE，不当早停或完整结果。见 [正式阶段记录](../../../manifests/grid-formal-20261015/EXECUTION_SUMMARY.json)。
