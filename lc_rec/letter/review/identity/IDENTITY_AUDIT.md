# LETTER-LC-Rec 研究身份与原始源码独立核查

核查者：真实子代理 `/root/identity_audit`，仅执行公开原始材料读取、固定源码静态核查、CPU 数据计数；未运行 GPU，未安装依赖，未操作 Git 状态，未编辑 vendor 原件。日期：2026-11-10。项目规范从 `/home/addaswsw/recommend/` 显式读取；实际源码工作树为 `/home/addaswsw/recommend-worktrees/lc-rec`。首命令 `ls -la` 观测到现有 `lc_rec/`、`docs/`、`experiments/` 等，未发现该树根的 AGENTS，不把外部规则称作自动发现。

## 1. 可成立的身份结论

**A 有可确认的已发表方法和作者补充配置依据，但没有闭合到历史 checkpoint、准确命令、环境和 Table 1 指标的逐次运行身份。** 论文 v2 §4.1.3 明确 LC-Rec 实例采用 LLaMA-7B 和 LoRA；Table 1 的 Instruments LETTER-LC-Rec 行为 R@5 0.0913、R@10 0.1115、N@5 0.0789、N@10 0.0854。作者在 Issue #19 直接链接补充参数文档；本轮以文档 Instruments 记录为候选，而不是把公开脚本默认值当成历史生效值。[论文](https://arxiv.org/html/2405.07314v2#S4.SS1.SSS3)、[作者补充配置回复](https://github.com/HonghuiBao2000/LETTER/issues/19#issuecomment-2798745616)

**B 只能标为“原 LC-Rec SID + LETTER 的 SeqRec LoRA 派生候选”。** Table 1 普通 LC-Rec Instruments 行确有 R@10 0.1006、N@10 0.0772，但没有找到将该行与“原 SID + 单任务 + 某套 LoRA + 某个 temperature + 确切评价 prompt”绑定的独立运行证据。Issue #33 只说其他 baseline 基本沿用各自原文设置；Issue #12 提到两方法均能 LoRA 微调，不足以补齐该行的历史配置。[论文 Table 1](https://arxiv.org/html/2405.07314v2#S4.T1)、[作者 baseline 设置答复](https://github.com/HonghuiBao2000/LETTER/issues/33#issuecomment-5066718250)

A 的新完整结果可以与 LETTER-LC-Rec 行作有条件比较；B 可与普通 LC-Rec 行作参考比较，同时明确训练身份未确认。两者均不能仅凭训练可执行或接近某一数字宣称“精确复现论文行”。

## 2. 固定证据与对应关系

- 固定 LETTER 源：`8d0154e28de37dbb6e24871c508ad8ddb1921cda`，作者提交时间 2025-01-01T16:43:17Z，提交仅修正上游文本 embedding 的 item 索引位置；这不是 Table 1 的历史训练回执。
- 已独立通过 GitHub connector 读取完整固定源树及提交元数据。对主控物化的 19 个必要 py/sh/config/data 文件逐一重算 Git blob SHA-1，全部与固定源树条目相等。8 个 Python 原件全部 AST 解析通过。此结论不包含依赖导入或模型可运行。
- 作者补充文档保存于 `../../evidence/author_hyperparameters.txt`，其精确字节 SHA-256 在 `static_verification.json.doc_sha256`。文档包含 Python args 形式的参数记录，**不是实际 argv、checkpoint、wandb 运行 ID 或完整训练日志**。作者链接证明其来源；未找到它与固定源码 commit 的显式历史绑定。
- 文档中的基础模型是作者本地路径 `/storage_fast/rhshui/llm/llama_hf/7B`，没有 Hugging Face revision/权重 hash。现有 huggyllama LLaMA-7B 资产可符合模型家族，但“与作者逐字节同 checkpoint”仍未知，须由主控资产证据限定表述。
- [固定源 README](https://github.com/HonghuiBao2000/LETTER/blob/8d0154e28de37dbb6e24871c508ad8ddb1921cda/README.md) 只固定 `torch==1.13.1+cu117`，未固定 PEFT/Transformers/DeepSpeed/bitsandbytes/Accelerate 版本；FastChat 和 flash-attn 不在 requirements 列表里。不能以 README 证明完整历史环境。

可复跑独立核查：从仓库根执行 `python3 lc_rec/letter/review/identity/check_static_identity.py`。结果文件 `static_verification.json` 含每个源码/数据 blob、文档 hash、公开脚本参数解析、SID 见证和计数。此核查不加载任何模型。

## 3. 公开脚本与作者 Instruments 记录的实质差异

有效值按 `run_train.sh` 显式参数覆盖 `utils.py` 默认值解释；脚本的空 `BASE_MODEL` 必须先填写，否则原样不能运行。

| 项目 | 公开 run_train.sh + utils.py | 作者补充 Instruments 记录 | 影响 |
|---|---|---|---|
| 每卡 batch | 16 | 8 | 4 卡原脚本 global batch = 128；补充记录 global batch = 64 |
| accumulation | 2（默认） | 2 | 双 4090 经用户批准采用 8×4×2 = 64；必须记录设备布局变化 |
| 学习率 | 1e-4 | 2e-4 | 更新强度实质不同 |
| LoRA alpha | 32（默认） | 16 | r=8 时 alpha/r 从 4 变 2 |
| temperature | 1.0 | 0.8 | 改变训练交叉熵定义 |
| only_train_response | False（默认，脚本未传标志） | True | 原脚本监督整段提示词与响应，且 padding 也未在该分支 mask；作者记录只监督响应 |
| save_and_eval_steps | 1000（默认） | 1309 | 两边均 strategy=epoch，普通 Trainer 语义下该步数不触发按步评价/保存；不可把它解释成历史 step 数 |
| 路径 | BASE_MODEL 空；data=../data；ckpt/Instruments | 作者本地模型路径；data=./data；ckpt/Instruments-diversity | 路径适配，不能据路径证明资产身份 |

一致部分：4 epochs、AdamW、wd=0.01、r=8、dropout=0.05、全部七类 q/v/k/o/gate/down/up 投影、`modules_to_save=embed_tokens,lm_head`、BF16=True、8-bit 基础模型加载（来自代码）、ZeRO-2、max length 2048、history 20、cosine、warmup 0.01、seed42、`tasks=seqrec`、train prompt sample=1、train data sample=0、sample_valid=True、valid prompt sample=2。

源码定位：[run_train.sh L10–23](https://github.com/HonghuiBao2000/LETTER/blob/8d0154e28de37dbb6e24871c508ad8ddb1921cda/LETTER-LC-Rec/run_train.sh#L10)、[utils.py L28–88](https://github.com/HonghuiBao2000/LETTER/blob/8d0154e28de37dbb6e24871c508ad8ddb1921cda/LETTER-LC-Rec/utils.py#L28)、[collator.py L23–44](https://github.com/HonghuiBao2000/LETTER/blob/8d0154e28de37dbb6e24871c508ad8ddb1921cda/LETTER-LC-Rec/collator.py#L23)。

## 4. 原始 LoRA / loss / 保存路径

1. `lora_finetune.py L63–83`：`LETTER.from_pretrained(load_in_8bit=True)` → 设训练 temperature → 扩 embedding → `prepare_model_for_int8_training` → 七类 LoRA + 两个 modules_to_save → `get_peft_model`。BF16 是 TrainingArguments 设置，不能概括为“所有参数/运算均 BF16”；int8 存储、激活计算、float32 loss、LoRA 与包装 embedding 的 dtype 需实测分别登记。
2. `L100–102` 仅将名字含 `original_module` 的旧副本冻结。它**没有直接冻结** `modules_to_save.default.weight` 活跃副本。Issue #12 的口头“论文 embeddings 未训练”结论与这段条件不能直接等同；必须按最终精确 PEFT 版本检查模块 forward 使用的对象、requires_grad、grad、optimizer 参数和保存后的权重。不能为了贴合口述而另加冻结。
3. `modeling_letter.py L33–39, L117–122` 实现 `CrossEntropy(shift_logits / temperature, shift_labels)`，默认 ignore_index=-100；temperature=.8 将 logits 乘 1.25，既改变 softmax 也改变梯度尺度，不能通过修改生成 temperature 代替。返回给生成的 logits 没有除 temperature（L114、L128–130）；测试脚本用原 HF LlamaForCausalLM 加 adapter 与此设计可相容。
4. token 来源是 index 每条 SID 的集合，经 `data.py L43–54` **字符串排序**后 add_tokens。训练/验证/测试均由同一个 index remap（L130–143），但恢复时必须用保存 tokenizer；token 集合相同不足以证明 token ID 顺序相同。
5. response-only mask 对 padding 和 input prefix 一并置 -100；须用实际 tokenizer 验证边界、EOS、至少一个监督 SID token。长样本截断造成全 -100 应失败而不能记成功。
6. `L85–98` 的 resume 只尝试 `adapter_model.bin`，并把 Trainer resume 改为 False；它不是 optimizer/scheduler/RNG 的完整恢复。不存在文件时仅打印后继续；现代 safetensors 保存也不会由这条旧路径自动识别。正式续训应独立确认真实恢复语义，本轮只可测试小 adapter 保存/重载。
7. `L131, L147` 开启 gradient checkpointing，训练禁用 cache；生成时应按原始 test 路径使用 cache，防止拿禁用 cache 的技术模型推估正式测试。
8. `L7` 硬编码四卡可见性、`L9–13` 无条件 FastChat monkey patch、旧 PEFT int8 API、W&B report_to、`torch.compile` 在 Trainer 构造之后赋给局部 model 均需工程核查。后者不能单凭源码宣称 Trainer 真正编译过。移除硬编码设备/补齐路径和禁用遥测属于运行适配；替换 attention 计算或更换 API 需证明数值语义，没有证明时保留风险，不冒充原版行为。

## 5. SeqRec 数量、切分与 prompt 协议

由固定三 JSON 和原始 `data.py` 直接核对：

- 24,772 用户、206,153 次交互。
- 训练对每用户先去末两次，再构造前缀预测；合计 131,837 条，而不是照搬旧任务数字。代码 L146–163。
- 每用户倒数第二个标签为验证，最后一个标签为测试；history 截到 20。代码 L165–204。验证文本 49,544 条（2 个 sampled prompt/用户），测试 24,772 条/prompt。
- 训练 `prompt_sample_num=1` 表示每条训练记录每次 `__getitem__` 从 **17 个模板随机挑 1 个**，并不表示只用 prompt0。L252–268。
- 验证在构造数据集时每用户无放回抽两个模板，此后保存为固定 valid_text_data；`sample_valid=True` 时 `valid_prompt_id=0` 不把验证限制到模板0。L220–237。
- 公开 `run_test_ddp.sh L18–20` 明确 beam20、`test_prompt_ids=0`；测试 defaults 同样为0，`all` 才循环17模板并取均值（test_ddp.py L66–74、L99、L184–207）。
- 论文 §4.1.2/4.1.3 没有给出 Table 1 具体测试 prompt 或 beam 数，补充训练 args 也没有测试参数；**最接近公开入口的候选是单 prompt0，但无法证明 Table 1 历史结果就是 prompt0**。不得套用17模板均值作为已确认论文协议。预算应明确 prompt0 实测外推及17模板约17倍生成量的可选上界，而不能把二者混为同一协议。

验证为 teacher-forced eval_loss，原 Trainer `load_best_model_at_end=True` 默认按验证 loss 选 checkpoint；不是每 epoch 完整 beam20 推荐测试。原码 eval_delay=1，实际首次 eval 时点须按精确 Transformers 版本确认。

## 6. 新发现：论文推理描述与代码约束不一致

论文 §3.2.2 描述 Trie 约束得到真实 item identifier，但 `data.py BaseDataset.get_prefix_allowed_tokens_fn L67–89` 实际只按**输出位置**集合返回该层所有 token，不检查已经生成的 SID 前缀。固定 index 的层 token 数为 150/256/256/256，允许笛卡尔积 2,516,582,400 组合；真实唯一 SID 仅 9,897。

独立 CPU 见证 `<a_0><b_0><c_0><d_0>` 每个 token 均在对应位置允许集内，但完整组合不在 index。该见证证明约束不是 Trie；以上组合比例不是实测生成非法率，不用它预测质量。公开默认 `filter_items=False`；可选过滤仅将非法项 score 置 -1000，既不重新生成足量合法候选，也不等价于 Trie。请保留此公开代码事实；换 Trie、额外过滤、或补足候选均会改变评价定义，需另行研究决策。

另有两项必须防止静默改变评价规模：

- `test_ddp.py L127–130` OOM 会将 beam 从20逐次递减，本任务要求 beam20，预跑必须记录精确 beam，OOM 应阻塞对应验收，不能默默降 beam。
- L77 的 DistributedSampler(drop_last=True) 会丢弃不能整除卡数的尾样本；本次24,772可整除2与4，因此该固定全量数据在2/4卡不因这条规则丢尾。其他子集不能沿用此结论。

## 7. 作者 SID 及碰撞

固定 Instruments index 共 9,922 item，9,897 唯一 SID，23 个碰撞组涉及48 item（额外碰撞25）。例：item112和175均为 `<a_118><b_238><c_151><d_0>`。评价按 SID 字符串比较，不按原始 item ID 消歧；碰撞影响不同 item 的可辨识性。不得擅自追加 token 或重训 tokenizer“修复”作者已发布资产。

Issue #35 作者确认 data/index 是此前训练得到的 SID；这支持复用作者发布 SID，不证明某个历史 checkpoint 的 SID hash。Issue #16 说 Instruments 来自 LC-Rec 的公开数据，#30 进一步说 Instruments=2018、Beauty=2014；是否与服务器现有文件逐字节/轴/切分一致，必须看主控实际对比报告，不能由口述替代。[SID 来源答复](https://github.com/HonghuiBao2000/LETTER/issues/35#issuecomment-6125365300)、[数据来源答复](https://github.com/HonghuiBao2000/LETTER/issues/16#issuecomment-2635730681)、[年份澄清](https://github.com/HonghuiBao2000/LETTER/issues/30#issuecomment-3297243045)。

## 8. B 的边界

固定 LETTER 源树没有单独普通 LC-Rec baseline 目录、完整基线训练配置、LoRA checkpoint 或 Table 1 测试回执。项目保留的原 LC-Rec `e6f1b3b3f8b4fee55a49a7b3dfeba29dbfcf0669` 原件确有 lora_finetune.py，因此不能说“原作者没有 LoRA 代码”；但其 Instruments run.sh L71–93 调用的是 finetune.py，8卡×8×accum2，六任务，lr5e-5、ZeRO-3，与 B 不是同一入口。其 LoRA 原件也不启用 LETTER 的8-bit加载/temperature loss。

直接把 A 的 index 换成原 LC-Rec index 至少改变：SID assignment（token语义与前缀）、token集合/词表大小/新增权重初始化；如果 SID 碰撞处理/级数不同，还改变输出空间和生成成本。保留 LETTER temperature=.8 则仍含 LETTER ranking-guided loss；改为1.0又是额外方法决策。SeqRec 单任务也不能冒充原多任务完整 LC-Rec。Issue #25 仅证实多任务经 ConcatDataset 混合，不证明普通baseline行用了哪些任务。

因此 B 只做静态/资产分析，名称和比较必须保留其派生身份；不能通过“替换 index.json”宣布复现普通 LC-Rec 行。

## 9. 四项指定复现风险的状态

| 风险 | 本次状态 | 仍缺什么 |
|---|---|---|
| 公开脚本与作者补充配置不一致 | **差异已定位；历史身份未闭合** | 原始命令、环境锁、checkpoint与指标绑定 |
| 复现者报告显著低于论文 | **事实已确认；原因未证实已解决** | 使用补充配置的可核验完整复现，不可用技术预跑代替 |
| 作者 SID 与原 LC-Rec SID 不同 | **预期方法差异，须靠主控实比量化** | A/B 各自 hash、轴和tokenizer配对；不能混用 |
| 普通 LC-Rec 行缺完整独立训练身份 | **仍未解决** | baseline该行的准确配置、任务、SID、LoRA/全参及评价回执 |

Issue #19 的复现者明确报告用作者 Instruments 文件和公开代码得到 R@10≈.07771、N@10≈.05964；#15 也有约.05的N@10报告，其中部分使用LLaMA2-7B。不能把LLaMA2差异解释为全部失败的唯一原因，因为#19同时有LLaMA-7B用户报告。[原始复现报告](https://github.com/HonghuiBao2000/LETTER/issues/19)

首选方案仍是 A：按作者补充 Instruments args、固定作者 SID、现有 LLaMA-7B资产与用户允许的双卡global64，完成有界技术验证。历史运行身份、SID碰撞、非Trie约束和论文prompt未知应作为独立限制保留；任何修正这些研究语义的方案需在正式采用前单独确认。

## 10. 原始回执与核查范围

本目录 `issue_{12,15,16,19,25,30,33,35}.json` 来自 GitHub connector，包含 issue 与全部评论及精确链接；`fixed_source_tree.json`、`fixed_source_commit.json`、`upstream_readme.json` 也是该 connector 取得。论文通过 web 工具读取指定 v2 原文，对应§3.2.1/3.2.2/4.1及Table1；本报告未复制整篇论文。

此代理直接复核了全部十个 LETTER-LC-Rec py/sh入口及配置、三个数据JSON的必要统计，重算19个blob；没有读取服务器GPU日志，没有运行PEFT参数对象，没有测试adapter加载，没有验收96小时估算或主控未来适配实现。相关项必须由主控实测证据和后续非作者复核收口，不能把本报告当成它们已PASS。

未新增系统/语言包或工具链，未修改环境/配置。Git提交、服务器源码映射、最终审核ZIP与 `GPT_UPLOAD_REQUIRED` 由主控在实际发布后统一给出。

