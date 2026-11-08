# LC-Rec Instruments＋原版 LLaMA：官方源码定向审核

审核者：真实子代理 `/root/official_audit`。审核日期：2026-11-08。只读官方源码、论文与锁定候选依赖源码；没有操作 Git refs、服务器、安装环境、数据下载或模型运行。此报告是本轮静态源码审核材料，不是网页 GPT 二次审核，不证明六任务真实数据、训练更新、保存重载或论文复现已通过。

## 1. 原件身份与检查范围

- 官方仓库：[RUCAIBox/LC-Rec](https://github.com/RUCAIBox/LC-Rec)，固定提交 `e6f1b3b3f8b4fee55a49a7b3dfeba29dbfcf0669`。
- 先用 GitHub 插件读取该提交 `README.md`，再通过 GitHub codeload 取小型源码归档；GitHub 插件没有源码二进制归档工具，使用公共 codeload 是能力补足。归档与展开均在 `source_audit/`，无 `.git`，不称 checkout。
- `official.tar.gz` 为 901,637 字节；展开原件 41 个文件，1,339,146 字节。全部 41 个 Git blob SHA-1 均与 GitHub 该提交的完整递归 tree 回读相等。依据见 `github_tree.json`、`source_identity_verification.json`；逐文件 SHA-256/字节数见 `official_manifest.json`。
- 原件目录：`LC-Rec-e6f1b3b3f8b4fee55a49a7b3dfeba29dbfcf0669/`。未修改任何原文件。
- 实际读取 README、run.sh、finetune.py、utils.py、data.py、prompt.py、collator.py、test.py、test_ddp.py、evaluate.py、config/ds_z3_bf16.json，以及本文涉及的数据处理、文本编码、RQ-VAE/索引、转换代码。
- 用 `ast.parse` 检查 28 个 Python 文件；所有 JSON 可解析；5 个 shell 文件 `bash -n` 均为 0。`prompt.py` 直接加载仅构造常量，实际模板数为 seqrec=17、item2index=19、index2item=17、fusionseqrec=12、itemsearch=11、preferenceobtain=11。结果见 `static_source_checks.json`。这不是 import/GPU/数据运行验证。
- 该固定源码 tree 没有 LICENSE/COPYING 文件；保留 README 的论文/数据来源和致谢，不自造许可证。第三方依赖文件保留原始 copyright/license header。实际资产许可仍须跟随模型/数据来源核对。

## 2. 生效配置链：脚本覆盖与默认值

以下路径/行号均相对官方原件目录。

| 设置 | 真实来源及含义 |
|---|---|
| 主模型/路径 | `run.sh:71–93` Instruments 段意图使用 `huggyllama/llama-7b`。第72行有多余空格（见问题表），不能逐字执行整份三域脚本。模型与 tokenizer 必须单独固定 HF revision/hash；官方脚本未固定 revision。 |
| 训练路径 | `finetune.py:51–56,64–106` 为 `LlamaForCausalLM`＋普通 Trainer 全参路径。没有应用 LoRA/冻结/量化。`utils.py:70–76` 虽定义 LoRA 参数，主路径未使用；`lora_finetune.py` 不是本轮原版主路径。 |
| 六任务 | `run.sh:90–92`、`utils.py:140–172` 顺序为 seqrec,item2index,index2item,fusionseqrec,itemsearch,preferenceobtain，concat 后统一训练。prompt sample 各1，data sample `0,0,0,20000,0,0`。 |
| 正式超参 | `run.sh:81–89` 覆盖 micro8、accum2、lr5e-5、epochs4、wd.01、epoch eval/save、BF16、response-only；官方 launch 为8进程，有效batch128。 |
| 未被该段覆盖的默认值 | `utils.py:37,61–86` max_his_len20件、model_max_length2048 token、adamw_torch、warmup_ratio.01、cosine、seed42。lr parser 默认2e-5，但本段实际是5e-5，不能抄默认覆盖真实脚本。 |
| 验证集 | `utils.py:172` 仅 SeqRecDataset valid，默认每用户无放回抽2个模板并在构造时固定（`utils.py:50–55`，`data.py:220–237`）。不是六任务验证总 loss，也不是全模板推荐指标。 |
| 模板抽样 | 训练 `__getitem__` 每次随机取一模板；`prompt_sample_num=1` 时每底层样本每epoch出现一次。Fusion 20k 子集在构造时抽一次，不是每epoch重抽20k。 |
| 正式 eval/save | `finetune.py:83–94` 同为epoch，save_total_limit5，load_best_model_at_end=True，epoch eval_delay1。4.31 callback使用eval_delay<=state.epoch，故完整第1个epoch末会验证；不是跳过首epoch。 |
| 梯度检查点/缓存 | `finetune.py:37,82,98` gradient_checkpointing=True，训练前设置model.config.use_cache=False；保存后的生成可能继承该值，不能假定cache自动恢复为True。 |
| 精度/加载 | `finetune.py:51–55` 未显式 torch_dtype，torch.float16 那行被注释；其 BF16 发生于后续 Trainer/DS 路径，初始整模加载仍须按实际依赖/上下文核对，不能用BF16最终权重估计加载峰值。 |

### Transformers 4.31.0 的候选依赖行为

官方 README 仅明确 `torch==1.13.1+cu117`；accelerate、bitsandbytes、deepspeed、evaluate、peft、sentencepiece、tqdm、transformers 无精确版本锁。没有找到 requirements/lock 文件。不能把本次选定版本声称为作者历史环境。

主控给定本轮 CPU 数据/监督环境使用 Transformers 4.31.0；为核默认行为读取了 [v4.31.0 原始源码](https://github.com/huggingface/transformers/tree/v4.31.0/src/transformers)，快照为本目录 `transformers_v4.31.0_*.py`。此环境复用torch2.6.0只读组件并覆盖旧tokenizer依赖，仅用于CPU数据/监督；DeepSpeed/Accelerate候选未安装、未执行。以下是该版本源码静态结论，不能替代 DS 实际 import/展开回执。

- `training_args.py:692–695`：Adam β=(.9,.999)、eps1e-8、max_grad_norm1。`training_args.py:1306–1311`：load_best=True 且未指定 metric 时 metric=`loss`、greater_is_better=False；`trainer.py:2352–2367` 将键转换为 `eval_loss` 并取较小者。最终选模依据是固定验证loss，不是测试HR。
- `trainer.py:1563–1577`：令 `L=len(actual_train_dataloader)`、`A=accum`，每epoch优化步 `U=max(L//A,1)`，正式总步 `T=ceil(4*U)`；不能简单无条件把 `ceil(N/128)*4` 当真实 T。DDP shard补齐、batch取整与accelerate版本决定实际 L。
- `trainer_callback.py:449–459`：epoch末在`eval_delay<=state.epoch`时eval；epoch save没有该delay。因此即使callback在第10次update令should_training_stop=True，Trainer退出该epoch仍可能调用on_epoch_end并保存巨大checkpoint；有界预跑必须额外约束eval/save回调与最终保存，不能只添加停止标记。
- `training_args.py:1991–1996`：未显式warmup_steps时 `W=ceil(T*.01)`；`optimization.py:131–169`：warmup线性、随后默认半余弦（num_cycles=.5）。本轮若有训练预跑，优先保留正式总步构造再用callback停到最多10 optimizer updates；max_steps=10 会改T/W，必须另标局部检查。
- 原 `ds_z3_bf16.json` 没有 optimizer/scheduler 节。`deepspeed.py:281–311` 在这种配置下调用 Trainer optimizer/scheduler，不是擅自选择 DeepSpeed FusedAdam 或另一个DS scheduler。
- `deepspeed.py:125–176`：auto → train_micro=实参micro，accum=实参accum，train_batch=world×micro×accum，gradient_clipping=1.0，bf16.enabled=True。按授权两卡候选 micro1/accum64 就是有效batch128，不能执行8进程。
- `deepspeed.py:188–216`：hidden_size4096时 reduce_bucket=16,777,216元素；prefetch=.9×4096²=15,099,494.4，实际DeepSpeed整型schema解析后的值须另核；persistence_threshold=40,960。其余原值 sub_group/max_live/max_reuse 为1e9。这里是元素数，不是字节数；BF16 bucket乘2字节，通信与重叠有额外峰值。
- DS stage3，无CPU/NVMe offload；stage3_gather_16bit_weights_on_model_save=False。原 `flops_profiler.enabled=True, profile_step=10`，正好可能命中本轮update上限附近，应把其开销纳入预算而不是暗中忽略。
- `modeling_utils.py:1395–1419,1464–1481,1534–1573` resize输入embedding与未绑定输出head，调用 `_init_weights` 后拷贝旧行；`models/llama/modeling_llama.py:463–472` 为Normal(mean0,std=config.initializer_range)，不是新版本mean_resizing。新SID行有意初始化，旧主干不得靠strict=False漏载。是否绑定以真实config.tie_word_embeddings与实测pointer为准；原LLaMA通常为False，两端需分别检查。

## 3. 六任务必需文件与语义配对

运行最低集合为 `<data_path>/Instruments/Instruments.{inter,index,item,user}.json`。文件名严格以 `.index.json` 默认后缀生成；并非任意同名旧GRID数据。`Instruments.user2id`、`.item2id` 不被下游loader直接读取，但用于追踪Amazon ASIN/原用户与数字轴，缺少时应标来源链限制。

| 任务 | 消费文件 | 样本/目标/历史（官方data.py） | 模板 |
|---|---|---|---:|
| seqrec | inter＋index | `:146–161` 先取交互[:-2]，遍历i=1..len−1，用[:i]历史预测[i]；最近20物品。`:165–197` valid目标[-2]/历史[:-2]；test目标[-1]/历史[:-1]。 | 17 |
| item2index | item＋index | `:495–519` 对全部item元数据构造样本，以title/description/两者映射canonical SID。title清理首尾标点。 | 19 |
| index2item | item＋index | 同ItemFeatDataset；SID映射title/description/两者。保留brand/categories等原字段，但当前模板未把它们作为额外输入。 | 17 |
| fusionseqrec | inter＋item＋index | `:315–344` 同训练prefix范围；同时构造SID历史、引号包围title历史、目标title/description/SID；无放回抽20,000条prefix。 | 12 |
| itemsearch | user＋index | `:568–618` 只用user_vague_intention.train及user_explicit_preference；每uid含item、inters、querys两条。`:657–659` 随机挑显式偏好与两种query。目标和历史来自user包，必须与inter[-3]、inter[:-3]逐uid交叉核对。 | 11 |
| preferenceobtain | user＋inter＋index | `:685–720` 读取user_explicit_preference，历史固定inter[:-3]并截最近20件；每次随机选一条preference作回答。 | 11 |

`data_sample_num=0` 不是零样本：Fusion、ItemFeat、ItemSearch、Preference仅在sample_num>0时采样，否则全量；SeqRec训练甚至不使用sample_num，因此本轮0保留全部训练prefix。若底层Fusion不足20,000条，np.random.choice(replace=False)会失败，不能补采样或静默改数。

以实际资产统计为准。仅作条件推导：若交互确与论文24,773用户/206,153次一致且每用户≥5，seqrec prefix数为206,153−3×24,773=131,834。若user辅助全覆盖、item轴9,923、Fusion20k，则concat数为221,226。这是静态条件推导，不是已经读到Google Drive资产的数量。

### 上游链与原始恢复边界

- `data_process/amazon18_data_process.py:14–25,28–49,153–195,201–228`：Amazon2018 ratings CSV＋metadata，物品保留title/description/brand/categories；按meta存在过滤、去重复user-item、迭代5-core、时间排序、映射数字轴、leave-one-out。不要用旧GRID Amazon2014/Qwen索引替代。
- 该脚本从set收集inter并只以timestamp排序，同timestamp的稳定次序/数字轴可能依赖输入/集合迭代；未与作者实包核对前，重新处理出相近数量不等于作者轴。
- `amazon18_data_process.py:289,295` 的review读取/写出被注释；`get_llm_output.py:334–339` 却依赖review JSON。不能因为公开脚本存在就声称辅助资产可无代价复建。
- `amazon_text_emb.py:44,53–58,87–105`：分字段编码title和description；按int(item)重排为连续数字轴；对每字段的有效token隐藏态mean pooling，再将字段向量mean。`data_process/utils.py:82–89` 用AutoModel并未固定revision。需绑定真实LLaMA权重/tokenizer与embedding provenance，不能只查npy形状。
- `index/run.sh:3–17` 有效脚本覆盖RQ-VAE四层256、e_dim32、末层Sinkhorn epsilon.003、batch1024、lr.001。`index/main.py` parser默认三层，不能抄默认称原版四层；`--bn False`配type=bool存在CLI解析陷阱，不能将原始checkpoint的bn事实由该文本推出。
- `index/models/rqvae.py`＋`rq.py`＋`vq.py` 为残差量化、直通梯度和重建/码本损失；`layers.py:85–108` 是Sinkhorn。`generate_indices.py:87–128` 先无SK生成，然后对同SID冲突组重分配，只有末层启用SK，最多20轮；注释声明重复物品不继续区分，故资产可能残留冲突。没有自行添加第五消歧位的授权；前缀数组里虽列到e层，不意味着实际用了五层。
- `get_llm_output.py:32–39,123–132,222–231` 分别以训练末项[-3] review和历史[:-3]、测试末项[-1] review和历史[:-1]、偏好历史[:-3]构造标签；辅助train不能误读test分区。`data_process/utils.py:16–65` 用旧openai.Completion API且无限重试；本轮不执行。

## 4. token、监督与截断需实测的断言

- `data.py:43–54` 遍历实际index值收集token，按字符串字典序sorted新增，不是按a_0,a_1数值排序，也不能假定1024个都实际出现。须保存token→token_id全表和index中每组长度/词表/冲突。
- `finetune.py:31–40` slow LlamaTokenizer，right padding、pad_id0，新增token来自第一个SeqRec dataset，全部六任务共享相同index。
- `collator.py:25–44`：full_text是完整SFT串（含输入）再拼EOS；tokenizer同时把full_text作为text，把无回答prompt作为text_target；labels复制full input_ids，pad_id0位置−100，再把target侧非0坐标遮掉。不是仅tokenize回答，也不是全序列监督。
- `prompt.py:3–4` response紧接 `### Response:`，无另加空格。BOS/EOS、AddedToken与句界分词必须按真实tokenizer检查；单独prompt token序列应与full前缀一致，不得凭长度假定。
- pad_id=unk_id=0意味着原逻辑也屏蔽真实UNK0，而不是仅按attention_mask屏蔽。必须记录原行为；不能静默用另一mask定义替换。
- tokenizer按总token2048右侧截断，与20件history是两类限制。长item描述/20title历史可使回答全被截掉；应统计token长度、有效监督数、EOS是否截断。全−100标签不能当可训练样本成功，修改截断/数据属于重要设置。
- test collator左padding；训练右padding。不能沿用训练collator做生成。新token是否逐个单token、四位串是否exact四token，要按真实AddedToken行为验证。

## 5. 生成与评价的关键冲突

`test.py:53–71,92–112,138–160`：test_prompt_ids=all时遍历17个SeqRec模板；每模板对同一个SeqRecTest数据集（实际采用SeqRecDataset test）生成；num_beams=20、num_return_sequences=20、max_new_tokens10、early_stoppingTrue、output_scoresTrue。按sequence score排序，先每模板按用户均值算hit/ndcg，再对17模板等权平均并报告min/max。作者参考值不是五seed平均。

但原实现不满足“每一步只允许完整目录中合法SID前缀”的严格定义：

1. `data.py:67–89` 仅建立第0/1/2/3位各自允许的token集合，随后EOS；并不根据已经生成的SID token前缀过滤下一位。它可以组合出目录里不存在的四元组。方法上的“full ranking entire catalog”和“position vocab约束”必须分开。
2. 分组token_id取 `tokenizer(token)[input_ids][1]`；假设只有一个BOS且其后就是SID。靠逆扫 `Response:` delimiter决定当前生成位；真实分词/截断/特殊标记不匹配会失败，EOS后越界也要定向查。
3. `utils.py:96` filter_items默认False；原run_test脚本未显式开。即使开，`evaluate.py:9–12` 只是把非法完整串score改−1000，不丢弃、不补足20个合法候选。
4. `evaluate.py:6–28` 去Response前缀、strip并删普通空格，以SID串相等判断命中；未做canonical SID→item映射，未去重。`ndcg_k` 对每个命中位置累加；若重复命中可偏高。需对真实输出检查碰撞/重复，不能直接自选去重规则改变原指标。
5. `test_ddp.py:72` DistributedSampler(drop_last=True)会丢失不能整除world的尾用户；`:121–124` OOM后自动减beam且无本任务式重试界。这两点与本轮固定≤16用户/beam20要求有冲突。单卡test.py没有降beam代码；未来派生DDP必须fail closed且验证用户集合，但这是候选适配，不是已跑结果。
6. 训练设置use_cache=False可能被保存；test未显式覆盖do_sample、length_penalty、temperature、use_cache等。因此实际generation_config必须从最终权重/config及4.31默认回读，不能只列显式kwargs冒充全有效配置。

严格trie约束、canonical item解码或候选去重若改变实际候选/评分，都属于重要未决偏离。首选是原件保留，单列官方position约束结果与本次用户要求冲突，由用户/网页GPT确认下一阶段是否采用合法目录trie；本轮不静默把改过的算法称逐字原版。若仅增加审计并发现真实输出均合法，结论也只适用于已核样本。

## 6. 原始缺陷与允许适配边界

| 位置 | 确认的问题/风险 | 本轮处理边界 |
|---|---|---|
| run.sh:5/39/72 | `BASE_MODEL= huggyllama/llama-7b` 把模型名当命令；bash -n仍会通过。 | 派生Instruments独立入口修空格/路径属拼写适配；不执行整份三域脚本。 |
| finetune.py:51–55 before:68 | 整模加载在TrainingArguments/DS上下文前；未显式dtype，可能在单张24GiB上FP32加载即失败。 | 当前已有训练状态资源不足证据，不以反复OOM验证；未来移动DS初始化/安全CPU加载须保留旧初始化行为并留diff。 |
| get_llm_output.py:312 vs论文 | 源码默认text-davinci-003与论文GPT-3.5描述不一致；无历史运行绑定。 | 资产优先；不自换模型、付费重生成或填空。 |
| get_llm_output.py:358 | 测试intention写入条件检查`not in intention_train`，按全用户train键可能导致test空。 | 核真实包，不改原件、不猜作者已经修正；本轮训练只消费train，但来源链需记录。 |
| index/main.py:31＋run.sh:9 | argparse type=bool对字符串False会解析成True。 | 不重训上游；读取实际checkpoint args/作者资产，不能推造历史bn。 |
| index/generate_indices.py:109–128 | 冲突20轮上限且允许重复item残留，未必无冲突。 | 统计实包，不加第五位、不重建映射。 |
| convert/convert.sh:5–15 | 遍历全部checkpoint并保存FP32中间态和最终态，可重复占用大量空间；主run把相对OUTPUT_DIR带入cd convert，路径也可能失配。 | 本轮不照抄后台全量转换；未来只对一个必要保存结果显式转换并验证完整载荷，不能只认config.json。 |
| SeqRecTestDataset:839–843 | 该备选类__getitem__缺return。 | 当前utils:179实际用SeqRecDataset，故不是本路径阻塞；不要把注释类重新启用。 |

## 7. 资源判断与正式候选准备边界

官方全参BF16＋ZeRO3＋Adam需要参数、梯度、FP32主副本、两个Adam矩、激活、通信/临时buffer和加载峰值。以约6.7B参数计算，仅常见16bytes/parameter模型状态/2卡已约50GiB/卡；即使乐观忽略若干状态仍远超24GiB。两个24GiB不能作为单块48GiB使用，microbatch/accum仅降低激活与保留有效batch，不会消除此模型状态下界。此处只作静态推导，现场GPU/内存/磁盘以主控回执为准。

已有充分下界证据时，不需要加载13GB权重或制造OOM；继续CPU六任务真实样本/tokenizer检查、源码/配置发布与审核。原版训练、梯度更新、保存重载及生成检查仍应分别标BLOCKED/NOT_EXECUTED，不能以CPU数据通过冒充原配方预跑PASS。

下一阶段首选保持无offload原方法，使用GPU状态容量足够的资源，例如两张80GiB级卡，先复核激活/通信和保存预算；这是新增设备/资源授权待确认，不能本轮执行。若只能保留双4090，候选为ZeRO3 CPU optimizer/parameter offload，保留全参/BF16/有效batch128，但会改变训练数据移动和优化器执行路径，CPU RAM和PCIe时间显著增加，需现场CPU内存与预算单独确认；本轮没有授权启用。

正式5seed候选只准备配置：seed42–46，固定同一数据/SID/辅助资产，4epoch，lr5e-5/wd.01/cosine/warmup.01，六任务原采样，epoch验证/保存，按eval_loss取best，SeqRec17模板先汇总每seed再求seed均值与样本标准差。有效batch128、原BF16/ZeRO3/checkpointing条件不变。micro/accum按已确认新资源计算，未获资源前不可排队自动启动。时间预算不能由零训练实测推成确定值；当前没有本轮update/sec或完整保存耗时。

## 8. 审核结论

- **PASS（限静态身份/源码理解）**：41份原件与固定Git树逐blob对应，必读代码链和论文定向内容已读；28份Python AST与5份shell语法通过；模板/任务采样/监督/DS/选模来源明确。
- **PARTIAL（工程准备）**：真实数据、tokenizer、环境、分支、服务器与交付由主控证据补齐；本报告未自行验证其结果。
- **重要未决**：严格全目录合法SID与原position约束冲突；实际资产残留冲突/辅助来源链；训练所需资源超双4090；原官方依赖无锁。不能用轻量模型、LoRA或任意新依赖掩盖。
- **NOT_EXECUTED**：本代理没有模型加载/训练/生成，没有改原件，没有安装依赖；无新增系统/语言包/工具链/配置文件。
- 初次源码检查为round0；本代理未进行方法修正或模型重跑，不占用额外计算预算。任务级修正轮次以主控统一记录为准。

论文定向交叉核对：[arXiv v4](https://arxiv.org/html/2311.09049v4)，III-B、III-C、III-D与IV-A/Table III：LLaMA标题/描述编码、四级RQ索引/末级USM，六类对齐，leave-one-out、全目录/beam20；Instruments参考HR@1/@5/@10=.0706/.1002/.1220、NDCG@5/@10=.0856/.0926，表注为多模板平均，不能当五seed均值或本轮短跑阈值。论文仅用于交叉核对，源码冲突如上保留。
