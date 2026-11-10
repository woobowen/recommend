# LETTER / LC-Rec Instruments：身份核验与双 4090 有界预检审核入口

2026-11-10；分支 `woobowen/recommend:wbw-lc-rec`；一次性任务。**最终状态：PARTIAL_TECHNICAL_BLOCKED。A 的论文方法与作者补充参数有依据；CPU 数据身份核查通过；GPU 在首次 backward 失败，完整 optimizer update 为 0；一个 seed 的 96 小时可行性未知。** 已达到项目允许的三轮任务级集中修正上限，停止新增 GPU 尝试。没有正式训练、完整 epoch、推荐指标测试、五 seed、B GPU、Qwen 或 Shuffle。

此文件是唯一入口。先读以下九项结论，再按相邻链接核查原件和机器回执。`PASS_PRELAUNCH`、CPU PASS、初始化成功及有限 loss 均不代表 GPU 技术验收或论文复现成功。

## 1. A 是否有可确认的已发表方法／配置依据？

**有方法和作者参数记录依据，尚不能逐次绑定论文历史运行。** [论文 v2 §4.1.3 / Table 1](https://arxiv.org/html/2405.07314v2) 的 LETTER-LC-Rec 使用 LLaMA-7B + LoRA；Instruments 为 Recall@5 **0.0913**、Recall@10 **0.1115**、NDCG@5 **0.0789**、NDCG@10 **0.0854**。[Issue #19 作者回复](https://github.com/HonghuiBao2000/LETTER/issues/19#issuecomment-2798745616) 指向[补充参数文档](https://docs.google.com/document/d/1rVeu3JqGXZNjapiMi1ioXIlXdXSt8zl94FmaAFYwX4g/edit)。本次保存了文档原文、来源元数据、八个 Issues 全部评论及固定代码，未将公开 run_train.sh 默认值当成作者实际参数。

固定上游 **`8d0154e28de37dbb6e24871c508ad8ddb1921cda`**。19 个必要源码／配置／数据文件的字节均与该提交 Git blob 一致，作者原件保存在 [vendor/LETTER](vendor/LETTER)。文档中的 args 字段可一一对应 `utils.py`、`lora_finetune.py`、`modeling_letter.py`、`data.py` 和 `collator.py`；它是参数记录，缺少对应完整 argv、依赖锁、checkpoint、SID hash、选模和测试回执，且没有明确绑定此上游提交。

作者模型路径为 `/storage_fast/rhshui/llm/llama_hf/7B`，没有 revision/hash。本次复用已核验的 `huggyllama/llama-7b@4782ad278652c7c71b72204d462d6d01eaaf7549`，未重下权重；能确认模型家族和本次资产身份，不能确认与作者历史权重逐字节相同。上游 README 只固定 torch1.13.1+cu117，未固定 PEFT / DeepSpeed / bnb / FastChat 完整环境。本次实际环境不是已证实的作者历史环境。

证据：[作者参数原文](evidence/author_hyperparameters.txt)、[解析后的原参数](config/author_instruments.json)、[来源元数据](evidence/author_document_metadata.json)、[19 文件物化验证](evidence/upstream_materialization.json)、[非作者原件核查](review/identity/IDENTITY_AUDIT.md)。

## 2. 双 4090 是否能正确执行 A 的 LoRA 技术预跑？

**本轮未通过。** 实际两张 RTX4090 均完成 int8 模型、LoRA、ZeRO-2 初始化，并各完成一个真实 SeqRec microbatch 的 forward 和有限 loss；首次 backward 在 checkpoint 重算中失败。没有任何完整 optimizer update，尚无完整梯度、参数变化、adapter 保存／重载、beam20、稳态 samples/s 或有效 tokens/s 验收结果。

| 实际项目 | 结果与边界 |
|---|---|
| 双卡布局 | 每卡 micro8、accum4，配置 global64，seed42；不是完整训练 |
| 数据／mask | 原作者 Dataset + collator；真实训练期记录；prompt/pad=-100；每条 response 为四个 SID token + EOS |
| 基础权重 | bnb Int8Params、torch.int8；量化 scale 为 FP32；没有改为4-bit/QLoRA/全参 offload |
| temperature | 原 `CE(logits.float() / 0.8, labels)`；rank0 原 loss=13.9681138992，独立 CE0.8 完全一致，CE1.0=11.8582515717；rank1 为13.4201202393 vs11.4293384552 |
| 训练对象 | 450 tensors / 289,652,736 参数：448 LoRA tensors + embed_tokens/lm_head 两个活跃 modules_to_save 权重；741 个具名参数均记录 |
| 旧 embed/head | 两个 original_module 副本 requires_grad=False，不在 optimizer；活跃副本 requires_grad=True 且在 AdamW；不能据此宣称其梯度和更新已经验收 |
| BF16 适配 | 先实测原始 Float/BFloat16 错误，再仅在必要的224个 LoRA A nn.Linear 调用内部启用 autocast；FP32 输入、BF16 权重／输出，调用后 autocast 关闭；base、LoRA B、dropout、CE 未被全局 autocast 包裹 |
| 有限性 | 每 rank 仅1个完整 microbatch loss 有限；后续 full-gradient / full-parameter finite check 没有到达。raw finite_checks 中 scope 是计划描述，不是已通过结果 |
| 最终错误 | PEFT0.5 `result += output` 修改 bnb custom Function 返回的 view，被 torch2.6 autograd 拒绝；发生于 backward → checkpoint 重算；不是 OOM |

用户批准的 BF16 autocast **只适用于本轮有界技术预检**；属于数值精度适配，不能宣称历史1:1等价，不能自动用于正式 baseline。原始 dtype 异常、RNG 恢复、无 step、采样参数未变、每个 LoRA A 的实际 dtype 和作用域都保存在 rank JSON。原诊断可能填充 bnb cache，cache 保留，不能称为模型所有字节完全还原。

| GPU job_id | 固定实际运行源码 | 实际结果 | guard wall time |
|---|---|---|---:|
| letter-20261110-gpu-r0 | `66c903a70ea709f7fc07c7ef211c7a35cbc3394a` | 原 dtype 已观测并适配；FastChat unpad 返回值4/5不兼容；FAILED/exit1 | 147.278430 s |
| letter-20261110-gpu-r1 | `6f59f44f02e50b49c4b6f2fec83fc8110dcbeb2f` | forward/CE0.8通过；首次 backward view/inplace失败；FAILED/exit1 | 86.384781 s |

两次 GPU 作业 wall 合计233.663211秒；按全时两卡保守计费 **7.788774 GPU-device 分钟 / 上限120分钟**，每 job 均远低于60分钟。两 job 均 cleanup_verified，worker/group 已结束；最终两卡无 GPU 任务。没有失败后偷偷重跑或降低 microbatch。

r1 失败前缀的 CUDA peak allocated 为10.6414/10.6459 GiB，reserved 为11.4512/11.3359 GiB；这不包括完整 optimizer update 的峰值，不能作为正式训练显存通过证明。每 rank ru_maxrss 为10.6654/10.6801 GiB；guard 5秒采样的 task RSS 合计峰值只有4.9314 GiB，存在漏采，不能当真实 aggregate peak。r1 采样最低主机 MemAvailable235.255 GiB，保留40GiB要求未触发。没有观测到本任务 swap 增长。CPU 数据核查峰值1.8626GiB。

证据：[r1 原始请求／argv／日志／状态](evidence/jobs/letter-20261110-gpu-r1)、[rank0](evidence/jobs/letter-20261110-gpu-r1/output/probe/rank0.json)、[rank1](evidence/jobs/letter-20261110-gpu-r1/output/probe/rank1.json)、[r0](evidence/jobs/letter-20261110-gpu-r0)、[预算 ledger](evidence/gpu_budget_ledger.json)、[实际参数摘要](evidence/parameter_initialization_summary.json)、[授权边界](evidence/precision_decision.json)、[根因独立复核](review/final_compatibility_blocker.json)、[最终机器状态](evidence/final_status.json)。

## 3. A 一个 seed 能否在96小时内完成？

**UNKNOWN；目前不能稳妥承诺，也没有证据证明一定超时。** 缺少完整 backward/update 和稳定吞吐；adapter、验证和 beam20 都没有实测，不能从失败作业总时长、旧全参预跑或合成短样本推导小时区间。

实际重查的 LETTER 数据为24,772用户、9,922物品、206,153交互，训练前缀 **131,837/epoch**，并非直接沿用旧数字。四轮基础训练527,348样本呈现；global64下预计每轮2,060、共 **8,240 optimizer updates**，cosine warmup_ratio.01对应83步。DDP sampler 的尾部补齐／最后部分 batch 与精确 Trainer 语义须在将来完整入口中核对；以上是名义记录计数与调度预算，不是本轮已执行步数。

| 分项 | 已核实工作量 | 本轮时间结论 |
|---|---|---|
| 4轮 SeqRec 训练 | 527,348基础样本呈现；预计8,240更新 | warmup后 samples/s、有效 tokens/s 未取得，时间区间未知 |
| 每 epoch 验证 | 24,772用户×2抽样prompt=49,544条；按4次预算198,176条 | teacher-forced eval_loss；未实测，保守预算4次。原 eval_delay=1 的精确首次触发时点未执行 |
| 选模及保存 | epoch checkpoint，load_best_model_at_end，验证 loss 选模 | 选择、保存和I/O未实测 |
| adapter 保存／重载 | LoRA + 完整活跃 embed/head + tokenizer | 未到达；不能仅按LoRA几十MB估计，两个大权重也被保存 |
| 完整 beam20 prompt0 测试 | 24,772用户 | 未实测；不得降 beam 后冒称beam20 |
| 17 prompt 成本情景 | 421,124用户-prompt呈现 | 约17倍生成量仅是数量情景，未确认论文使用该协议，时间不一定严格17倍 |

设 r_train、r_valid、r_gen 为**双卡总体**真实吞吐（分别为训练样本/秒、验证文本/秒、beam20用户-prompt/秒），则小时预算为：

`T = 527348/r_train/3600 + 198176/r_valid/3600 + 24772*P/r_gen/3600 + T_init + T_select_save + T_adapter_reload + T_margin`。

P=1 是公开 prompt0 情景；P=17 是另列的全部模板情景。若生成实测按单卡计，则需先换算双卡吞吐，不能重复除2。即使假定其他开销为0/6/12/24小时，训练吞吐仍分别至少需1.526/1.628/1.744/2.035 samples/s，才能数学上落在96小时内；这些阈值不是已测性能。安全余量尚无法计算。

真实长度已完成 CPU 核查：所有训练行×17模板的均匀模板分布推导 min62、p10=68、median83、p90=136、p95=159、p99=172、max175、mean92.5389 tokens；验证实际两模板 mean94.7836；prompt0测试输入 min79、median84、p90=119、max159、mean93.9041。每条训练/验证响应含5个受监督 token。全部181,381实际行都做归一化匹配和prompt0实际tokenize，另核15,760个非零模板真实文本，其他模板用经核实的有限history-length文本类别推导；不是把全部行17遍完整tokenize，也不是未来随机epoch的精确模板频次。无2048截断。

短／中／长／极长68/83/136/175真实样本已保存并 CPU mask 验证，但 GPU 未到这些分组的稳态计时，**不同长度的训练耗时变化未测得**。完整训练、验证、生成必须分别补齐有界计时，之后才可给小时区间。

证据：[预算机器推导](evidence/budget_estimate.json)、[推导脚本](scripts/estimate_budget.py)、[真实 CPU audit](evidence/jobs/letter-20261110-data-r1/output/audit/audit.json)、[CPU 非作者验收](review/identity/runtime_cpu_review.json)。

## 4. 作者代码需要哪些最小工程修改？

作者原件未改，适配集中在独立 [gpu_probe.py](scripts/gpu_probe.py)、[compat.py](scripts/compat.py) 和尚未执行的 [inference_probe.py](scripts/inference_probe.py)。[原件到技术 harness 的 diff](review/official_to_bounded_probe.diff) 展示入口替代，不能把整个 harness 宣称为一行纯接口修复。

| 处理 | 本轮状态／实际影响 |
|---|---|
| 去除作者入口硬编码4卡、补齐模型／数据路径、隔离输出、关闭W&B | 在派生入口处理；不修改模型原件 |
| 作者 Trainer 换成有界9更新 DS循环 | 仅技术验收设计，实际0更新；保留原模型／数据／collator、Trainer.create_optimizer参数分组、author loss及调度定义；不能代表完整 Trainer 选模已验收 |
| 8-bit旧 API | 仍调用 `prepare_model_for_int8_training`；对 PEFT访问的 bnb 已移除 False字段及构造 kwarg做签名限定适配；没有改量化算子定义 |
| FastChat／flash-attn | 仅在 FastChat模块alias处返回新unpad_input原始前4项；禁止unused_mask，实际核第5项等于mask sum/cu差；attention kernel本体未替换 |
| BF16 dtype | 用户批准后，实际错误触发的LoRA A线性调用内autocast；重要数值适配，历史等价未知，正式训练未批准 |
| DS配置 | ZeRO-2/BF16原bucket保留；auto值显式解析；双卡8×4替代四卡8×2保持global64；仅计时关闭flops profiler |
| torch.compile | 未执行作者在Trainer构造之后的局部赋值；原代码不证明Trainer实际使用编译模型。本轮不据此宣称与历史compile等价 |
| 最终view/inplace阻塞 | **未修复**。独立审核首选未来在派生旧PEFT forward中将 `result += output` 改为out-of-place加法，保留梯度，禁止detach；此提议未实施／未测试，需新一轮授权后再验收 |

PEFT0.10 小型候选wheel已下载但**未安装／未使用**；静态审核指出其初始化RNG和依赖差异，不能作为本轮已成功方案。没有换 QLoRA、4bit、任务、epoch、temperature、训练范围、SID、标签或生成协议。

保存／恢复额外限制：原 `resume_from_checkpoint` 只尝试 adapter_model.bin 并关闭 Trainer完整resume，不恢复optimizer/scheduler/RNG；现代safetensors也不被这段旧逻辑自动识别。本轮adapter保存／重载未执行，不能声称已解决续训。

## 5. 公开脚本与作者补充配置有哪些实质差异？

| 项目 | 公开 run_train.sh + utils默认 | 作者补充 Instruments |
|---|---|---|
| 每卡batch / 四卡global | 16 / 128 | 8 / 64；accum均2 |
| learning_rate | 1e-4 | 2e-4 |
| LoRA alpha（r=8） | 32 | 16 |
| temperature | 1.0 | 0.8 |
| only_train_response | False | True |
| save_and_eval_steps | 1000 | 1309；均epoch策略，不能把此值当实际更新总数 |

response-only差异直接改变监督：False分支还没有把pad labels改成-100，不能视为无关默认。主要共同设置为4epochs、AdamW/weight_decay.01、r8/dropout.05、q/k/v/o/gate/down/up、modules_to_save embed/head、BF16、8bit加载、ZeRO2、max_length2048、history20、cosine/warmup.01、seqrec、seed42、valid_prompt_sample_num2。

训练 `train_prompt_sample_num=1` 实际为每次从17模板随机选1，不是单prompt0；验证每用户抽两个模板且固定下来。公开 [run_test_ddp.sh](vendor/LETTER/LETTER-LC-Rec/run_test_ddp.sh) 是prompt0/beam20，但论文与补充文档没有绑定历史测试prompt；不得套用原LC-Rec17模板均值。原公开测试还会OOM后自动减beam，不能将此行为用于本任务beam20验收。

## 6. 原 LC-Rec SID 的 B 会偏离什么？

**B 为派生候选，历史配置未知。** LETTER固定树没有普通LC-Rec行的独立运行脚本、任务、LoRA参数、temperature、checkpoint及评价回执。Issue #33“沿用各baseline设置”和#12“可以LoRA”不足以证明Table1普通行=单任务LoRA+原SID。保留的原LC-Rec上游 `e6f1b3b3f8b4fee55a49a7b3dfeba29dbfcf0669` 有LoRA代码，但run.sh调用全参finetune.py、六任务、8卡/global128、lr5e-5、ZeRO3；不能挪用其名义证明B。

| 已实比资产／语义 | A LETTER已发布SID | B 服务器原LC-Rec SID |
|---|---|---|
| inter / item | 与B逐字节相同 | 与A逐字节相同 |
| 用户／item轴、顺序、交互、切分、历史／标签item | 全部逐行相同 | 同A；可证明现有numeric轴，不补造缺失原始ASIN/user历史 |
| index SHA-256 | `33c87e28c52ae367985bdf494cb49d676d181e13a39497b4ad035f46111546dd` | `b94d16398a178220a117abda5effd5508069beaccc2b8110eab25ae50f2af600` |
| index bytes | 518,041 | 516,781 |
| 同item SID | 9,922项全部与B不同 | 全部与A不同 |
| SID层数／唯一值 | 4 / 9,897 | 4 / 9,897 |
| 碰撞 | 23组48items、额外25；与B相同碰撞item组 | 同A，但SID字符串不同 |
| 新增token／总词表 | 918 / 32,918 | 930 / 32,930 |
| SID token排序／ID | 字符串sorted后从32000起；A ids已保存 | B单独排序，不能套A tokenizer |

inter hash=`0a11d47791d90483005da93c36d826679e20e4592d1d2b86518d5f56f218e743`；item hash=`1d9938e576a8d964e6b977a2b63aa56402840a788eaa6685f26f0d910a7133f2`。两侧所有9,922拼接SID均验证成4个原子token；BOS1/EOS2/pad=unk0。train/valid/test碰撞目标分别941/165/122条。评价按SID字符串，无法消歧碰撞item；原资产保留未修复。A输入、监督和候选约束都绑定同一A index；B也独立核查，未混用。A tokenizer保存再读一致，但A模型adapter重载未完成。

更换index改变每个item的SID语义、前缀、新token集合、token ID、词表及新增权重初始化。若保留temperature.8仍包含LETTER ranking-guided loss；改为1.0则增加另一项方法决策。单任务也不等同原多任务LC-Rec。不能直接换index后宣布复现普通LC-Rec行。

未读取、修补或重生成辅助user intention标签；SeqRec不使用它们。没有训练LETTER tokenizer。B原件未覆盖；包内仅保留独立[小型B index证据](evidence/assets/B/Instruments.index.json)。

## 7. A/B 可与哪一行比较？

A的将来完整结果可与LETTER-LC-Rec Instruments行（R@10=.1115、N@10=.0854；R@5=.0913、N@5=.0789）作**注明历史缺口与工程偏离的比较**。本轮没有推荐指标，不能声称已接近或复现这行。

B可参考普通LC-Rec行（R@10=.1006、N@10=.0772），但准确名称须保留“原SID + LETTER SeqRec LoRA派生候选”；该行历史训练身份未知，不能声称精确复现，也不能将B结果写为A。loss有限或下降都不能预测论文Recall/NDCG。

## 8. 当前阻塞、四项复现风险与首选方案

当前技术阻塞是旧PEFT与bnb／torch自定义反向view的原地加法冲突，三轮修正已用尽；不是显存或96小时已证实失败。仍缺：完整反向、LoRA A/B及活跃SID权重梯度和参数变化、NaN/Inf全参数验收、稳态短中长吞吐、adapter保存重载、beam20、每epoch验证、选模与I/O时间。问题详细证据和未实施最小提案见[独立根因报告](review/final_compatibility_blocker.json)。

| 用户指定复现风险 | 结果 |
|---|---|
| 作者补充配置与发布代码不一致 | 差异已定位，历史实际运行绑定仍缺 |
| 公开脚本复现指标明显低于论文 | #15/#19报告事实已确认，原因未证实已解决；#19有LLaMA-7B报告R@10约.07771/N@10约.05964，不能全归咎LLaMA2 |
| LETTER SID与原LC-Rec SID不同 | 已完整量化、隔离和tokenizer核查；这是方法差异，不可互换 |
| 普通LC-Rec行缺完整独立训练身份 | 仍未解决，B不能升级为已公开完整配方 |

另有独立评价风险：论文描述Trie，但发布 [data.py](vendor/LETTER/LETTER-LC-Rec/data.py) 仅按位置返回各层token集合。A允许150×256³=2,516,582,400组合，实际唯一SID9,897；`<a_0><b_0><c_0><d_0>`为经验证的可允许但不存在完整SID。默认filter_items=False；该数量不是实测非法率。本次没有换Trie或自行过滤，不能将公开代码等同论文Trie描述。24,772测试用户可整除2，当前drop_last不丢尾；其他小子集需独立处理。

首选仍是 **A + 固定作者SID + 作者补充Instruments参数**。网页GPT二次审核后若继续，优先单独评议最小out-of-place兼容提案，再授权新的有界验收；无需先改训练方法或降研究条件。尚无实测超96小时的证据，因此本轮不提出“必须减epoch/任务”的虚假时间取舍；若后续真实预算超限，可再由用户决定换更小已发表方法／不同研究对照，不能自动删任务、改temperature、减epoch或启动正式训练。本次到此停止。

## 9. 非作者审核、固定源码映射与交付

真实非作者 `/root/identity_audit` 直接复读论文／Issues／固定源码、19个blob、CPU原始数据和服务器B资产；核对全部CPU输出成员hash、真实行／标签／mask／tokenizer及runner状态，并直接复核两次GPU的rank JSON、请求、源码、错误和资源；没有用主控总结替代原件。`/root/data_audit` 审核GPU派生实现／FlashAttn接口，并独立核对精确PEFT/bnb wheel源码和torch2.6 guard，保存未解决根因。GPU代码作者为 `/root/gpu_probe_author`，不是这两位最终审核者。主控负责串行Git整合和交付。审核PASS均限定范围，**没有独立GPU技术PASS**。

审核入口：[研究身份报告](review/identity/IDENTITY_AUDIT.md)、[CPU运行审核](review/identity/runtime_cpu_review.json)、[GPU负结果审核](review/identity/runtime_gpu_review.json)、[GPU导出成员复核](review/identity/gpu_negative_artifact_verification.json)、[兼容根因审核](review/final_compatibility_blocker.json)、[启动前代码审核](review/gpu_prelaunch_review.json)。CPU审核中“精度批准待定”属于当时状态，后续批准以[precision_decision](evidence/precision_decision.json)为准；历史审核不事后改写成新结论。

| 身份 | 固定提交／服务器位置 |
|---|---|
| 原LC-Rec固定交付基点 | `6b9f336d0252b44a31a48fbcc59817c9af1197e2`；旧结果／全参预跑未重复 |
| CPU源码 | `7a7365c6033864c232a4c2b042bcc9c494c938b3` |
| GPU r0源码 | `66c903a70ea709f7fc07c7ef211c7a35cbc3394a` |
| GPU r1源码 | `6f59f44f02e50b49c4b6f2fec83fc8110dcbeb2f` |
| 服务器固定export | `/home/bwwu/lc-rec/letter-20261110/releases/<上述完整SHA>/`；无.git，按RUN_VERSION.json逐文件映射，不声称checkout |
| 作业／argv／日志／退出状态 | `/home/bwwu/grid-audits/sid-extensions-p0/jobs/<job_id>/`；重要轻量原件已在[evidence/jobs](evidence/jobs) |
| 复用模型 | `/data/bwwu/lc-rec/assets/llama-7b/4782ad278652c7c71b72204d462d6d01eaaf7549/` |
| 隔离依赖 | `/data/bwwu/envs/letter-20261110/`；继承既有只读site包，不修改旧环境 |

CPU payload exit0且数据验收PASS，但其runner最终为WORKERS_TERMINATED/LEADER_EXITED_WITH_MEMBERS：runner只清理捕获的本任务残余helper后验证无人存活，不能改写为SUCCEEDED。CPU r0目录接口失败与environment r0强制CUDA的CPU导入失败原始记录均保留。第一次打包曾用本地格式化等价request覆盖仓库的原始request副本；已恢复原始导出字节，另存local-request，全部成员重新独立核hash通过，没有改服务器请求或重跑数据。

最终GitHub发布完整SHA、分支ref回读、关键文件内容回读和最终服务器export回执放在最终交付ZIP的 `final_receipts/`，并随交付回复提供；实际GPU运行仍严格绑定上表r0/r1 SHA，不能因报告发布新提交而替换运行身份。ZIP只含本次小型源码、数据证据、配置、日志、审核和外部回执，不含模型权重／adapter／wheel／凭据；成员manifest、真实解压、逐成员hash及秘密检查由[打包脚本](scripts/build_review_zip.py)完成，ZIP自身SHA放包外。

新增安装仅限隔离环境：**peft0.5.0、bitsandbytes0.45.5、fschat0.2.31、flash-attn2.7.4.post1**；einops已存在未重装，PEFT0.10/hub候选未装。无apt／驱动／VPN／SSH／共享GRID环境／shell配置改动，仅新venv及其任务内.pth。实际torch2.6.0+cu124、Transformers4.31.0、DeepSpeed0.16.9、Accelerate0.21.0。安装命令、下载hash、版本和stdout见[install_log](evidence/install_log.json)、[dependency_downloads](evidence/dependency_downloads.json)、[environment jobs](evidence/jobs/letter-20261110-environment-r1)。轮次与普通传输重试分开见[correction_rounds](evidence/correction_rounds.json)。

终止后资源快照：[final_resources](evidence/final_resources.json)。本任务两个新服务器目录合计约1.37GiB分配空间（含wheel与固定export；不含原有共享依赖／模型），加runner小型job记录仍远低于30GiB；后续最终报告export增加量单独记录在交付回执。/data可用约3.3TB，两卡均1MiB且无compute进程，主机可用约239GiB。没有清理未知文件、旧成果或未提交的两个既有Qwen tokenizer；仅stage/push本次 `lc_rec/letter/`。

**正式采用尚未批准；本轮精度适配批准不延伸到正式训练。GPT_UPLOAD_REQUIRED: YES**（最终轻量包的真实WSL/Windows路径、大小、SHA-256和解压验证回执随最终交付给出）。
