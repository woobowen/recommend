# A800 不变性、历史结构及图表溯源

任务：`wmh-a800-audit-20261010`。本报告为已取回 A800 原件的增量分析收口，作者 `/root/a800_mechanisms_final`，继承真实作者 `/root/a800_mechanisms`、`/root/mechanisms` 的已保存工作；最终非作者审核在独立记录中另列。本报告没有给全任务授予 READY，也不将已解释的历史未知改写成已确认机制。历史时间戳原时区未知。

## 1. 已处理范围与证据身份

来源为 `A800-grid-20261010-v1`，原根 `/data_a/mhwang/Rec/GRID`；完整源清单 SHA-256 `c18dc041088fa069958554144be9bb7f5ed9846ccaf0e97b0aa62edadb9907f6`。本报告使用 A800 外的 `preservation/A800-grid-20261010-v1/restored/A800_GRID/`，不连接或改动来源库。完整留存及最终逐字节复核由主控独立验收，不从本分析的结构通过状态推断。

- 机制责任的11项原源码/Notebook全部核 SHA，对九份源码及一份旧 Notebook 按完整相同字节复用已完成的原文及全部差分阅读；新 `logs/test.ipynb` 两 cell、全部输出/metadata 已完整读取。`A800_CODE_READING_CATALOG.jsonl` 保留每路径上下文、源 SHA、优先捕获与主捕获两种身份、原阅读依据及非作者复核。
- 原库140份 PNG 全部实际重哈希、完整解码；80张新图按20张 contact sheets 全图逐幅阅读，另5张原分辨率检查；60张相同字节复用旧逐图解释。`IMAGE_VISUAL_READING_CATALOG.jsonl` 是逐图责任与可见观察，未声称恢复所有浮点 attention 值。
- 528份 raw attention PT（cross/self/encoder 各176，batch0..175）均已恢复。严格CRC校验下惰性结构分类509份，19份原 `data.pkl` CRC 冲突被严格reader拒绝；后续有界原storage bytes探查19/19均可惰性语法解析，明确标 `CRC_CONFLICT_BYTES_INERT_PARSED_UNTRUSTED`，不将损坏描述当可信历史张量结构；无仍待读的结构项。另真正全字节、全ZIP成员 CRC 作业核528原 SHA 均匹配：495份全部 CRC 一致，33份存在原字节与 central-directory CRC 冲突，其中14份 `data.pkl` 可读但某些 storage CRC 冲突。结构可读与数值载荷健全分别登记。
- 全64 checkpoint 已读所有嵌套 config/scalar/参数键与形状，21种完整 key/shape/dtype 分组。机制再按45份推荐 checkpoint 的参数约束分类、读取各自768B decoder relative-bias 表，共34,560B，15份全零/30份非零。17个 MLP/双表/四BOS关键 checkpoint 又直接从原 ZIP 重解析 `data.pkl`、读取选定完整 numeric storages，并全读68个相邻小 config/overrides/train.log/CSV；12个字节独有有效配置按一份完整原文与全部原文差异阅读。没有调用 pickle.loads、torch.load、来源构造器或训练。
- 345保存运行、199个字节独有配置、102个非身份行为组、789日志全流式解析与全部 CSV 的既有分析按 A800 当前文件完整 SHA 复用，保留旧接收来源，新增 A800 身份关联。旧 `ALL_RUN_CHAIN_INDEX` 的早期 targeted 字段不替代后续 `LOCAL_COVERAGE_NOTE.md`、`RUN_READING_DEPTH.jsonl` 的完整分层深度。8个完整 prototype、全部原文差分和189种字段差异解释是配置深读依据，自动机器索引不是该依据。
- 机制历史177个 file/class occurrence来自115个不同路径，352个关键原字节窗口在 A800 恢复副本重新核对（269,375B）；24个重点机制 run、30个有 attention 保存声明的 Hydra run、6个 RotationSim run 进入 `A800_EXPERIMENT_RUN_INDEX.jsonl` 全345运行的多标签索引。类别重叠、外层重复与不同 rank 不当作独立实验数。

`analysis/a800-mechanisms/ATTENTION_STRUCTURAL_SUMMARY.json`、`CHECKPOINT_MECHANISM_STRUCTURE_SUMMARY.json`、`historical-parameter-context-v1/SUMMARY.json`、`A800_EXPERIMENT_RUN_SUMMARY.json` 给出机器集合与哈希；原源码完整行读说明见 `reports/MECHANISMS_LOCAL.md`，该旧报告保留其当时本地输入范围，不再用于表示当前 A800 是否取得。

## 2. 当前真实实现与历史运行边界

| 机制 | 当前实现的确定行为与原文定位 | 历史材料支持什么、不能支持什么 |
|---|---|---|
| 普通 AR | `tiger_generation_model.py` SHA `c61606b4a03d6fc8cfa6cb3bca093a36bbf9a1846334d39cb8dbcb67b32c296e`；584–659层级offset/shared查表、SEP/user；661–736与900–953 teacher forcing和各码CE相加；253–372/738–827逐H beam及cache | 现存源码行为已确认。历史相同target/path/参数形状不能绑定此Python版本；`codebooks`在该消费者是SID映射，不是量化centroid表 |
| test embedding shuffle | `tiger_generation_model_shuffle.py` SHA `e03082ad87bfc842c05467580fc479eff4345c6a0d18c5b2f7faf320d4f96c80`；83–149/619–627 test hook；778–870先offset和lookup后逐(batch,item)抽前N向量排列，再加SEP/user | 当前训练不因该test gate改变。码本lookup身份随向量保留、物品边界保留；N3留末码、N4全四码；恒等可出现、每次encoder新抽，loss/generate可各抽一次；mask不随动。16历史配置含shuffle/N字段，有实际test记录者的历史具体实现仍未绑定 |
| row整数shuffle | `pre_processing.py` SHA `a69c88681760…`，321–366逐物品前N整数randperm | 函数自身无test gate/padding过滤；本次345完整保存配置中没有显式引用该row函数。存在不等于启用，不能外推未保存运行 |
| collate整数shuffle | `collate_functions.py` SHA `a4692ce68705ea6a10ed500245f5c5acc99468f7e5d3f2d41e1ff0970fa9a73c`，282–345先形成label再重排input整数，整padding物品跳过 | 后续依新位置offset重新解释码本，与向量shuffle不同。10-29/14-59-58明确N3/testing配置和test事件，历史源码仍未唯一绑定 |
| 当前MLP | `tiger_generation_model_mlp.py` SHA `1b75bf94d6bd1187e7c30f996da33ebdc7ea2ffb33d8918a123991abbadafab9`；580–651有序H×E拼接→Linear2E/ReLU/Dropout.1/LinearE，末码mask，每item一个encoder token、SEP强制None；710–898/1050–1103仍AR与teacher forcing | 架构不保证置换不变性。3历史checkpoint实际含对应四个聚合参数，历史完整forward/mask与当前版仍UNKNOWN；`mlp_layers=2`是T5 FF替换参数，不等于item聚合 |
| 当前parallel文件 | SHA完整等于shuffle；952–1059仍依赖已生成前缀与cache的H步AR | 当前文件事实不能否定真实历史并行尝试。12月初始化日志和四BOS checkpoint真实存在；当时mask/目标/loss/generate代码没有由当前文件恢复 |
| 当前no_pos | `tiger_generation_model_no_pos_embed.py` SHA `7561d2f494764a77efac1d71abff4ea15bc683eea0816c1e8fe067d7d5c78463`；1747–1759只关闭/清零/冻结decoder相对bias；encoder、层级offset、SEP、因果性保留 | 不能称全模型无位置线索。45推荐checkpoint中15 bias表全零仅为参数兼容约束，非历史非参数flag/source身份 |

所有短 SHA 的完整身份在 `A800_CODE_READING_CATALOG.jsonl`，对应原件在审核包的源码或明确外部原件映射。当前训练扰动不能从函数自身没有gate反推发生过；用户明确确认的测试期干预与各历史配置仍分别保留。

## 3. Shuffle 的真实历史结果与对照限制

重点配置都保留原 seed42、H4、beam10等声明，不将其作为其他项目默认协议。整数collate run `logs/inference/runs/2025-10-29/14-59-58/.hydra/config.yaml` SHA `2ef1c71862d1da06d3a959c9a0b144d158842486e54b9f5704738c19d133055a` 指定 `collate_fn_train_with_input_shuffle`、N3、testing；CSV SHA `dde27b856c0345821d64bc2bf9b599b2c03c0359775df8ddb24ea31fa750c70e` 第2行同一次test R5/N5/R10/N10=`.0072635482/.0049820463/.0121574281/.0065505267`，loss13.469239。低结果仍保全，未删除或当作无效研究。

16个model shuffle/N配置实际列表在 `A800_MECHANISM_HISTORY_RECONCILIATION.json.mechanism_run_candidates`，包含N1恒等、N3、N4、只存N而未存开关的11-09/00-19-11，以及没有test CSV的失败/未完成尝试。它们的target均仍是普通模块路径；当前普通源码不接受shuffle专属kwargs，所以不能用同target把历史运行绑定当前源码，也不能自动用shuffle文件替换后称恢复。

可核对的同一test事件示例：Toys N4 `10-29/15-47-20`=`.0387389250/.0266478788/.0541417673/.0315964669`；Beauty N4 `10-29/19-42-55`=`.0371148773/.0251362361/.0556723140/.0311247967`；Beauty N4 `20-54-14`=`.0417654179/.0273563042/.0611724742/.0335662700`。这些不是自选最高值；每行单次CSV事件与对应config/SID/checkpoint声明都保留。先前指标不同步的最佳val没有拼成一次test。

## 4. 恢复的MLP、双表、四BOS历史结构

64 checkpoint分为19个量化/其他、28个单表无聚合单BOS、3个单表有聚合单BOS、13个双表无聚合单BOS、1个双表四BOS。参数结构是事实，历史算法身份是另一层。

3个真实MLP checkpoint对应：Toys `10-22/21-25-31` step3800（SHA `fb215c22cdc413fc81233354439801a93eaa2d5b01ca328fdfc71dadb2a9d028`）；Toys `21-28-23` step3600（`baf73e828cbee108864e24933abfc76b8475b427f4743defd1ddce62e41df6f2`）；Beauty `10-24/15-10-17` step2500（`095f97fb5c6add6c023dc0c4b746def9725757027dec95397d0d4de44135ec9a`）。原参数有 `item_aggregation_mlp.0.weight[256,512]`、bias[256]、`.3.weight[128,256]`、bias[128]；四完整storage实际CRC通过、数值有限。各自test同事件CSV99/95/73分别为：

| 运行 | R5 | N5 | R10 | N10 |
|---|---:|---:|---:|---:|
| Toys21-25-31 | .0333814137 | .0238610897 | .0466206484 | .0281280763 |
| Toys21-28-23 | .0397177003 | .0271299444 | .0586750470 | .0332578421 |
| Beauty15-10-17 | .0435540862 | .0299934261 | .0623351075 | .0360782146 |

这些CSV记账step是4800/4600/3500，并不等于已选checkpoint的3800/3600/2500；报告没有把二者偷换。外层原日志的MLP实例化和结束/测试支持实际尝试，仍不能证明历史每个forward/mask细节等于当前有序聚合实现。checkpoint hparams仍保留 `should_add_sep_token=True`，当前MLP却强制None，是不应单凭保存flag判定实际token组织的例子。

新增强反例：Toys MLP `10-22/21-25-31` 与双表无聚合 `10-29/23-38-53` 的有效config逐字SHA均为 `9050b9de5ae2ed55ab68b6c653f968c04bf3272c17a1129c8f101498747799fd`；各自checkpoint却有不同键结构。历史运行可以在同配置/同target下导入不同内容的模块，配置同字节不足以绑定历史Python。

14份双表（含四BOS）checkpoint的encoder/decoder SID表完整选定storage均实际读取，14对数值bytes均不相同；不能把保存的 `weight_tying=True` 理解成实际两表相同。`SMALL_ORIGINAL_PARAMETER_WITNESSES.json` 另提供14对原storage前128个float32中各8个实际不等坐标及原4字节hex，并完整保留四BOS的512个float32与2048字节hex，便于外审直接复核数值证据。13单BOS分支包含10-29四run、11-03九run；其中23-38-53、02-05-24、02-44-02实际hparams为V255、表[1020,128]，其他十份为V256、表[1024,128]。单BOS和完整参数组不能证明它们与当前shuffle forward/generate相同，也不能按外层`embed2_yiqichu`名称认定非AR。

`12-02/00-47-41` 的原train.log SHA `b9d8193d721c9ce9cc6eb7970984d51da6938750e4fa876f4793663ac7e7cfd2` L6明确“初始化并行生成模型”，L25最佳路径step24375。实际checkpoint SHA `6a6387233e67b7a30907c4e95bba7c2bb48bd1e677a47d741c31213db7ad72fa` 含 `decoder.bos_tokens[4,128]`，四行完整float32 bytes各不相同，双SID表[1024,128]且内容不同；hparams H4、beam5。CSV第64行test R5=R10=.010150695219635963、N5=N10=.00809971522539854、loss17.524559，CSVstep30625；低指标与topK相等不能独立决定生成方式。

12-03/00-59-29又有初始化日志，加载旧B1 step1700出现缺 `decoder.bos_tokens`/decoder SID表、多出单 `decoder.bos_token` 的原错误；这是消费者与权重结构不同的具体证据。完整17上下文、所有68小原文、所有选定storage SHA与CRC见 `historical-parameter-context-v1/`。本轮能确认这些权重/结果确实来自A800捕获，不能恢复尚无原文绑定的历史 mask、目标组织、训练loss、生成/beam规则。未写新parallel方法冒充原件。

## 5. Attention 原件、生产链与反例

全528原件大小15,970,626,848B；cross176、decoder_self176、encoder176，命名各0..175。descriptor正常者：cross168、self170、encoder171。cross是8个step×4layer float32，常见 `[32,6,1,135]` 首步、`[320,6,2,135]` 次步和后续 `[320,6,1,135]`；self对应长度1..8；encoder是四层 `[32,6,135,135]`。原 device分布cuda0..3，文件名不能确定rank或样本。

33原CRC冲突为cross17/self9/encoder7；其中19份data.pkl CRC冲突（8/6/5）被严格ZIP读取拒绝。后续真实固定作业 `wmh-a800-20261010-WSL-attention-corrupt-descriptor-probe-v1` 对明确stored payload offset的小原字节作有界惰性解析，19/19语法均能读到描述；没有执行pickle构造器、修原件或去掉CRC冲突标记。这些语法可读结果单列为UNTRUSTED，不当作CRC健全或历史shape/device已确认。全原SHA匹配源捕获说明这是已捕获原件内的CRC不一致，而非本轮只根据一个本地坏副本猜测源损坏；不能证明CRC字段或payload哪一侧先损坏、何时或为何损坏。33份完整清单/坏member及actualCRC在真实终态作业 `wmh-a800-20261010-WSL-attention-crc-v1/output/`，原件全部保留，不修正CRC或生成假历史浮点值。

只对按词典序将被现入口选中的25份raw做过有限numeric读取：20份选定layer/sample/query/head/行片段CRC与数值通过，1份选定storage CRC失败、4份metadata失败。`raw-attention-selected-v2/SELECTED_NUMERIC_INDEX.jsonl` 明确采样范围；没有声称15.97GB浮点数逐元素语义核验。保存25索引与成功numeric JSON可供外审重算选定135列和/像素候选。

同batch设备反例经过6个原 `data.pkl` 直接惰性重读：batch24三文件全部storage CRC通过，cross/self存cuda0、encoder存cuda2；batch65也全部CRC通过，cross/self存cuda1、encoder存cuda2。全descriptor能比较的三文件组中157组同device，5组不一致，14组因原data.pkl失败无法比较；同device不等于同样本，不一致也不能单凭metadata确定原因。两组三文件已证明“同basename batch号”不足以绑定共同writer/device；它们与单纯CRC损坏相区分。

30个Hydra run保存过attention（另3个外层汇总复制上下文），H4为27、H6为2、H8为1。唯一当前清单内H8保存声明是 `11-14/18-26-28`，config SHA `2a9835f85e596f0a8d936ec96d8c782d53b99e5ff50b83f809fd26a6bf7453e2`（完整身份以 `ATTENTION_RUN_CONTEXT_INDEX` 为准），rank0保存175组0..174；raw却每类176份0..175。log L8033/8072/8074/8076表明rank1/0/3/2分别写5504/5595/5632/5632预测，L8079合22363。当前no_pos `predict_step`1451–1527向共享 `./attention_weights` 写同batch路径且无rank guard；RankedLogger仅过滤日志，不限制torch.save。多rank覆盖或跨run残留是可兼容解释，尚非已确认原因，更不能用于解释CRC发生时间。

现no_pos H>6代码1170–1277采用chunk、beam临时min(3,k)，attention返回5项但chunk调用按2项解包；当前这个入口不能直接生产已存 `[32/320,...]` 的8步raw结构。不能据H8/135维度兼容将当前源码绑定该历史run。原log L20的stack形状32×6×135与32×6×2×135冲突是兼容线索，但不是唯一producer证明。原始完整窗口和来源身份见 `ATTENTION_ORIGIN_COUNTEREXAMPLES.json`、`ATTENTION_AND_PARALLEL_DECISIVE_RAW_LINES.jsonl`。

### 绘图版本、样本与beam

`visualize_complete_decoder_attention.py` SHA `531bbaf5ab118ffedb3b0fef32dd0fe26748298b64e3c7e92a68c911c673e177` 20–149按 `H=len(cross)` 构造末层、head均值、sample0/当步首beam、最后query；self和cross来自独立softmax，拼接并非联合归一化。未沿最终胜出beam祖先回溯，跨生成步的行可以来自不同最终路径。非0 sample索引处理也不完整。`visualize_encoder_attention.py` SHA `25622305bddb1d93e93d960ee8263a26e2ca8f601e6ab6be13ad718af0e380c4` 16–42用行/列和启发裁切，不读取真实mask。

两个入口按词典序独立排序文件，再将列表序号作为输出batch标签；全组0..175的前十是0/1/10/100/101/102/103/104/105/106，encoder前五0/1/10/100/101。图batch2不能直接等同raw batch2。当前完整绘图器对这些raw会构造8×135 cross与8×8 self，不会原样生成保留PNG的4×150或MLP4×30。尚缺样本ID、输入SID/mask、checkpoint、采集源码版本、最终beam路径、绘图版本的共同manifest，故不能从图片外观或目录名确认机制/对照/“不使用结构”。

## 6. 140图和用户PPT对应

140图逐路径都保留观察与SHA，没有RGBA全图同像素重复组。MLP35图的decoder cross4×30，encoder30位置；shuffle3/4各35图及顶层35图的decoder cross4×150，self4×4，但轴标布局有分歧。MLP encoder有效长度标题10/5/21/5/5；shuffle3标题45/20/35/90/20；旧另外一组标题70/40/20/35/30。标题来自启发裁切、独立色标和同batch都不能建立真实同样本控制。

用户输入实际为 `王旻镐-语义ID顺序不变性研究.pptx`，828,147B，SHA `135529b32fdd1f2bbeabd4958b2af53d23057bc9b35137b2c94f1020e11a8f06`；没有取得同名PDF，不编造PDF来源。完整7页/111 ZIP成员保全，研究分析覆盖1–5，第6–7按用户排除不恢复研究内容。

第5页 `ppt/media/image9.png` SHA `c66dda8614bcdc00ebbeabfad99b4a0094c827ba7d86418965ebf41252f99a0d` 与A800 `attention_shuffle_4/.../complete_decoder_attention_batch0.png` SHA `a6c911aceaa602cf4a893e32abb2eb4738e8725e374971101e3dbe8724c8af58`；`image10.png` SHA `1c439e0933bfc2113951ac2138ad7aa827777de17a189065320fde78a9fbde1f` 与新恢复 `attention_shuffle_3/.../complete_decoder_attention_batch0.png` SHA `711365c80907b56e69c68e58ac53cab9212a9f59bb22ecbe569ac2d96dfac990`，各前50×4格200个RGB中心/全部150×4格600中心均一致。在30个兼容4×150候选内各零距离唯一，10个MLP图因4×30明确不兼容。它确认强渲染内容对应候选，不是整图像素同一，更非raw/checkpoint/样本/实现身份。材料独立复核改用每格3×3内部点，支持同样的有界结论；image9边框抽点有12/5400不同，禁止写成全像素身份。

第4页18条原生表行在1651个单次完整test/val事件里保留相同显示规则的四值候选，8行有一致候选、10行没有；保留舍入/截断/多候选/最近val和标签冲突，不跨step拼最高值。增量 `PPT_RESULT_CANDIDATES_A800.json` 实际重哈希37份 A800原CSV/37 config，回读87个不同原CSV事件的四值，全与旧候选一致；旧接收来源仍保留。

两项具体反例：Beauty表“MLP/R-Kmeans”数值候选10-24/15-10-17实际声明SID09-21/16-36-33，其上游配置明确RQ-VAE+STE/Beta；不能因表头改写成已确认R-Kmeans。Toys表R-Kmeans无扰动对应三个数字相同事件，且声明上游3584维与Qwen7B路径却保留T5EncoderModel target，与T5表头冲突。Toys/Beauty MLP行最近的11-03/02-44-02、02-45-07实体还分别恢复为双表、无 `item_aggregation_mlp` checkpoint；它们既非完整四值显示匹配，更不能由“最近数值”绑定聚合算法。18行全部候选与原始字符串在机器表保留，不强制选一。

## 7. 其他实验、失败及研究价值

全345 run的8种保存experiment：语义embedding24，R-Kmeans训练33/推理19，RQ-VAE训练12/推理7，RVQ训练2，推荐训练146/推理102。都是原声明与实际日志/产物索引，不按名称自动认定具体历史Python。R-Kmeans当前为layer-wise归一化residual+MiniBatchKMeans；RQ-VAE有encoder/decoder、STE/Beta、joint与不归一化分支；RVQ保持原维STE。量化H1/3/5/7/9，推荐H2/4/6/8/10及不同beam/数据/评估声明均保留。Beauty/Toys主要域，Sports只有实际数据与映射范围，没有在345保存Hydra run里发现Sports运行；不外推历史从未做过。

no_pos当前265–393计算generated/input物品内H×H normalized Gram，按batch均值再平均；802–1004按训练loader上下文tuple频率分Head/Body/Tail，不自动等于原始交互总体流行度。11-10/15-11-30还打印变长“序列Item Embedding”相似度矩阵，实际有5/6/7/8/9/21等尺寸，不能等同当前每item内部H×H实现。13,612个历史矩阵marker均完整解析首个数值矩阵，含重复日志，不等于13,612独立实验。

RotationSim当前 `src/inference.py:36–147` 在predict后按同条top1物品相邻向量拟合 `T=Y@pinv(X)`，又在同对评价，未约束正交、未跨样本训练公共旋转；高重建cosine不是泛化证据。12-03六条run的原汇报从仅total11227，到dataset_avg=-.0005，再T_dataset_avg=1.0000、T_pairwise_avg=.0067/.0085；字段/版本分歧真实保留。它与量化梯度的RotationTrick是不同实验。

Toys H6四个相同保存config/SID/checkpoint声明的推理却有R5=.120812744对其他三份=.043529775的冲突；前三份同次整体四指标超出Head/Body/Tail三组范围，第四份在范围内。只有同全集、同定义、同状态分组时整体应为加权均值；这是条件矛盾，不确定原因，不选最高作正确baseline，也不把第四份认作已确认评估版本。

历史失败包括MLP/DDP unused parameter、shuffle kwargs、eval_step额外kwargs、decoder heads/表形状、single/fourBOS、relative bias额外key、attention stack、dataloader、SID越界等。原错误窗口、源字节定位及73个有错误的保存run保留；缺历史退出码者不伪造FAILED/SUCCEEDED。当前BaseModule普通/MLP eval_step接口、普通shuffletarget/kwargs、getz、null loss等静态缺口已登记工程后续，不能修好当前入口便声称恢复历史指标。主控独立CPU工程副本与历史原件分离，本机制没有启动训练或自行改变研究算法。

Notebook `logs/test.ipynb` SHA `cdace6d97fa1cc62513cd2e2e7af094564468f7e9787e9715446a6b9af841118` cell0未执行输出，cell1保存19412条预测和4条user10×4SID打印。独立惰性全PKL解析与四条原stdout全部相同，但同SHA预测群在5个run路径存在，不能唯一绑定Notebook声明producer。Beauty Notebook保存TFRecord API AttributeError；根SVD scratch不是隐藏MLP/parallel实现，失败和诊断仍登记。

## 8. 身份缺边、后续可做工作与审核

已知各当前源码行为、真实保存配置/结果、参数结构、图像对应和原CRC例外都有可复核原文/张量结构证据。尚无法唯一绑定历史MLP/parallel/attention producer的Python、mask/目标/loss/generate、样本及beam ancestry；这类缺边应以 UNKNOWN 保留，不以当前文件、参数量、多个BOS、图片相近或接近论文值填补。源码/历史对象/缓存/全部checkpoint内文本候选和归档成员的反例审核由材料/基线非作者最终记录限定查找范围；没有确认不存在于历史所有时点。

下一阶段可从保全原件建立当前有序MLP/AR/shuffle的明确可写版本，以及针对已证实参数/输入链做受控评价；要复现历史并行/图因果解释，先取得额外历史生成源码/采集manifest或明确按证据驱动重建，不能将新设计称原版。本轮未正式复训/复评论文指标。

关键非作者证据：`review/A800_MECHANISM_PRIORITY_SOURCE_REVIEW.json`、`review/A800_ATTENTION_RENDERING_NONAUTHOR_REVIEW.json`、旧 `review/MECHANISMS_INDEPENDENT_REVIEW.md` 及pyc纯字节独立复核、历史标签/H6原文反例；新增最终报告/33CRC/17结构/混device反例的独立复核见随后 FINAL 记录。作者承担的基线非作者证据包括 `B1_DIRECT_ORIGINAL_ARTIFACT_REVIEW.v2.json`、`RQ_FORMULA_SAMPLED_NONAUTHOR_REVIEW.json`、`BASELINE_SAFETENSORS_NONAUTHOR_READBACK.json` 与固定粗筛双向差异复核，以及 `review/A800_ARTIFACT_CATALOG_NONAUTHOR_FINAL_REVIEW.json`：746精确产物路径、668个原ZIP中央目录/38,881成员、248推荐上下文、55完整原test行及19损坏descriptor原字节独立复解析。抽样公式64/24025项不冒称第二遍全量公式验证，源代码兼容也不冒称历史源码身份。

本角色0安装、0GPU/训练、0A800修改、0学弟baseline研究；新增脚本只处理明确已核验副本和小证据。大实体由主控完整留存，轻证据包包含源全文、原CSV/log、原Notebook/PPT关键member、完整结构/CRC索引与差异。最终机制清单/轻证据manifest必须与最终审核状态一起发布。
