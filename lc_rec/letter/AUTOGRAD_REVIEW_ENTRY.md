**GPU TECHNICAL PASS — LETTER-LC-Rec Instruments 有界技术链路。** 本轮真实完成双 RTX 4090、LLaMA-7B、Int8 LoRA、BF16、ZeRO-2 的 9 次 global64 更新，并完成 validation loss 计时、adapter 严格保存重载和原协议 beam20。正式四轮训练、完整验证/测试、Recall/NDCG、五 seed、Qwen、Shuffle 均未执行。

**96 小时结论：条件性支持可行，正式验收仍为 `INCONCLUSIVE_FORMAL_CHECKPOINT_AND_SUSTAINED_RUNTIME`。** 条件中心估计 14.446985 小时，保守估计 17.417500 小时，对 96 小时分别剩余 81.553015／78.582500 小时。这里没有证明完整 seed 已验收，也没有把 adapter 写入计时冒充正式 Trainer/DeepSpeed checkpoint 全部成本。

这是用户新授权的一次性接续任务。历史 [61854bb 交付](https://github.com/woobowen/recommend/blob/61854bb1df0f26a71e68d95ab8eb564f7c762f76/lc_rec/letter/REVIEW_ENTRY.md) 仍保持 `PARTIAL_TECHNICAL_BLOCKED`、0 更新和三轮修正耗尽；旧失败记录未改。本轮 GPU 补丁/训练/推理均在 initial round0 通过；随后以 round1 校正交付草稿中的 sampler 事实描述，没有新 GPU 运行或数值变化。本轮使用修正轮次 **1/3**，没有更换任务名无限重试。

先读 [本轮机器摘要](evidence/autograd_20261110/summary.json)、[真实预算及分项](evidence/autograd_20261110/budget_estimate.json)，再按下文定向读取源码和原件。此前 CPU 数据/资产审计和 19 个作者源码冻结结论沿用原交付，没有重复下载权重、全量审计或调查其他 baseline。

**1. 修复及适用身份**

实际修复在 [compat.py](scripts/compat.py)：`peft05_out_of_place_forward` 与 `install_peft05_out_of_place`。不修改 vendor、site-packages 或 bnb/Torch wheel。只替换已核 PEFT 原类的运行时 forward，参数对象、shape、dtype、requires_grad 和 state_dict keys 在安装前后逐项不变，安装不消耗初始化 RNG。

原错误发生于 `engine.backward` 中 gradient checkpoint 重算：bnb `MatMul8bitLt` 返回 view，旧 PEFT `result += output` 试图原地修改 custom Function 输出。派生计算使用已核 MRO 的 `bnb.nn.Linear8bitLt.forward(self, x)`，解决复制函数的无参数 `super()` 类闭包问题；残差改为 `result = result + output`，保留 base、A/B、dropout、scaling、dtype、active/disable/r0 分支及输入梯度。

| Guard | 固定实际身份 |
|---|---|
| PEFT / bnb | 0.5.0 / 0.45.5 |
| 原 PEFT 文件 SHA-256 | `cbcab4d64b5254d0e1a0018f5e870f7bd93f0178ed1b14d2b750e033f871aa01` |
| 原 forward SHA-256 | `df705367a1512d2f8bc3ff067a2617218a0a69ca5f151f2d2d1e233462d06088` |
| 原签名 | `(self, x: torch.Tensor)` |
| bnb 父 forward SHA-256 | `d3f4146d34e7100027694d30fa36bac6c8691c244f865a15ff55e3c462b12665` |
| 派生 forward SHA-256 | `976afde20397526a0be194d4629f34edc1b945d099299322cf61ac09b5c0bf33` |
| 实际目标 | 224 个 `peft.tuners.lora.Linear8bitLt`，全部真实命中及 checkpoint 重算 |

MRO、完整方法源码、版本和初始 CPU-hidden 导入身份见 [installed_peft_identity](evidence/autograd_20261110/installed_peft_identity.json)。父 forward 版本/hash、PEFT 文件/方法 hash、类、签名、MRO 任一不匹配即拒绝安装。新方法只计数观测，不对计算路径 detach/no_grad，不关闭 checkpointing 或跳过 backward。

[最小计算 diff](review/autograd_20261110/peft05_forward_semantic.diff) 明确只展示两处计算变化；[独立 AST 检查](review/autograd_20261110/derived_forward_ast_result.json) 验证去除明示观测计数后的计算树仅有这两处差异。最终完整 `compat.py` diff 另在 [全部实际适配 diff](review/autograd_20261110/actual_continuation.diff) 中，可据此检查计数、guard 和原有适配，而非把整个 harness 称为只改一行。

**2. 小型正确性与实际环境**

[test_autograd_compat.py](scripts/test_autograd_compat.py) 在真实 PEFT 类上使用 CPU 自定义 Function view 作为临时测试 base；它不是 bnb CUDA 验收。原函数真实复现 view/inplace 错误；新函数在非零 A/B、dropout=.05、scale=2 的输出和 input/A/B 梯度与参考差异均为 0。reentrant checkpoint、B 零初始化导致合法 A 零梯度、disable/missing active/r0 分支、base 输入梯度均通过。CPU FP32 容差 `atol=rtol=1e-6`，未初始化 CUDA。原始 [CPU job](evidence/autograd_20261110/jobs/letter-20261110-autograd-cpu-r0) 与 [非作者 CPU 验收](review/autograd_20261110/cpu_result_review.json) 完整保留。

真实模型作业版本为 Torch 2.6.0、Transformers 4.31.0、PEFT 0.5.0、bitsandbytes 0.45.5、DeepSpeed 0.16.9、Accelerate 0.21.0、FastChat 0.2.31、flash-attn 2.7.4.post1。使用既存 `/data/bwwu/envs/letter-20261110`，本轮新增系统包、语言包、工具链及环境配置改动均为 **0**。

保留此前批准的 LoRA A 线性调用内必要 BF16 autocast，以及 FastChat 模块 alias 的 unpad 5→4 返回值适配。实际所有 224 个 A 模块均记录了 FP32 输入/BF16 权重/输出，outer 和返回后的 autocast 为 False；base、B、dropout、CE 不在该 autocast 内。unpad 第五项与 mask sum、cu_seqlens 差实际相等，attention kernel 和 mask 未换。

非原地加法与 unpad arity 属工程兼容；A-only BF16 涉及数值精度，不能声称历史逐位等价。BF16 与新 Autograd 补丁都仅批准用于本次技术预检，**尚未批准成为正式训练配置**。没有安装 PEFT 0.10、换 QLoRA、量化方案或关闭 checkpointing。

**3. 双卡训练闭环**

[gpu_probe.py](scripts/gpu_probe.py) 与 [固定配置](config/probe_instruments.json) 保持用户确认条件：LETTER 作者 Instruments SID、SeqRec、r8/alpha16/dropout.05、q/k/v/o/gate/down/up、active embed/head、lr2e-4、weight_decay.01、temperature.8、seed42、history20、max_length2048、response-only、两卡每卡8×累积4=global64、ZeRO-2。原4 epochs仅作预算，未执行 epoch。

| 检查 | 真实结果 |
|---|---|
| Trainable | 450 张量，289,652,736 元素；448 LoRA + 2 active embedding/head；新增 SID token 918 |
| Optimizer | Trainer 与 ZeRO-2 原参数组均与 trainable 集合严格相同，无重复、意外冻结/解冻 |
| Forward/Backward | 每 rank 36 个完整 micro-step 均通过；temperature.8 CE 独立核对完全一致 |
| 更新边界 | 每第四 micro-step 触发一次，共9次；Adam状态1…9且跨卡同步；首步LR0、其余8步非零 |
| 梯度 | 全450张量各36次观测均有限，最终无全零张量；输入梯度及实际使用/监督SID行梯度通过 |
| 参数变化 | 每rank224个A及224个B均实际变化；active SID embed/head变化，所有trainable元素有限 |
| 冻结原件 | 两个完整 original embedding/head 的逐字节hash不变，未进入optimizer、无梯度 |
| 双卡参数 | 全部LoRA及active所有新增SID行hash跨rank相同 |

原件：[训练 r0 请求、日志、终态和资源](evidence/autograd_20261110/jobs/letter-20261110-autograd-gpu-r0)，[rank0](evidence/autograd_20261110/jobs/letter-20261110-autograd-gpu-r0/output/probe/rank0.json)、[rank1](evidence/autograd_20261110/jobs/letter-20261110-autograd-gpu-r0/output/probe/rank1.json)。[真实非作者训练验收](review/autograd_20261110/gpu_runtime_review.json) 为 `PASS_CORE_GPU_TRAINING_AND_ADAPTER_SAVE`。

训练9个更新秒数依次为 **5.6992、4.7216、4.3908、4.3887、4.3445、4.3655、4.0032、4.0235、4.0969**。前三次仅为测量warmup；稳态3次总192呈现、17,983有效token、27,096含padding位置，其中padding9,113；全局 **14.657983 samples/s、1372.888026有效tokens/s**。稳态包含数据构造/mask检查时每global64为4.47286、4.42435、4.45122秒；前向、反向、step、transfer、GPU利用率、进程RSS及swap原始记录均在rank JSON和resource-samples中。

计时使用 CUDA synchronize、MAX rank耗时和SUM token数。GPU phase计时包含全梯度观测、同步及guard开销，CPU collator/mask另记；它不是无监测正式训练吞吐。六次mixed使用同一个128真实row/prompt池，稳态3次只有128独立行而非192新行；短/中/长是重复真实样本的长度边界测试，不用最快一轮外推。稳态长度均值93.6615，已审全量均匀17模板均值92.5389，范围均覆盖至175；这不是长期或完整epoch测量。

**4. Validation、保存与原协议生成**

使用原 SeqRec validation 和 collator，warmup后mixed/short/middle/long每双卡16样本，Forward分别 **0.26077／0.24727／0.23652／0.24696 秒**；mixed含CPU准备为0.28183秒。记录真实row、长度、有效/监督/padding token及loss，未扫描完整验证集或选模。

adapter保存为原生 **adapter_model.bin，579,376,762 bytes**，SHA-256：

`1069b6df364d21ac2ecdab25b97f30b1ee03d5182203d5fe909bdfe2d5c0624a`

包含全部448 LoRA和完整active embed/head共450张量，直接与模型训练集合逐key/shape/dtype/值核对；tokenizer完整vocab与SID ID顺序保存后重读一致。写入1.01095秒，含本轮严格验证4.52194秒。未写完整模型/optimizer训练checkpoint。载荷仅保留服务器：

`/home/bwwu/grid-audits/sid-extensions-p0/jobs/letter-20261110-autograd-gpu-r0/output/probe/adapter/`

[inference_probe.py](scripts/inference_probe.py) 实际让PEFT读取该.bin，再对全部450key、289,652,736元素、包括全部918 SID行的重载值逐项核对；保存dtype转换到实际推理dtype后严格相等，差异全部记录。没有用strict=False忽略缺失训练参数。.bin实际open路径已记录，歧义格式拒绝；**本轮未声称safetensors adapter重载通过**。

保持作者 test_ddp.py 的LoRA推理分支：重新加载完整BF16 LLaMA与vanilla HF attention，训练分支才是Int8/FastChat。两卡基础模型重载15.759／16.064秒、adapter重载94.336／94.446秒，成本均纳入估计。

| prompt0层 | 输入tokens | 双卡2个不同用户秒数 | users/s |
|---|---:|---:|---:|
| q10 | 79 | .519121 | 3.852667 |
| q50 | 84 | .529194 | 3.779334 |
| q90 | 119 | .643960 | 3.105783 |
| max | 159 | .774102 | 2.583640 |

另有q10 warmup 1.89317秒，排除吞吐。共8个不同timed用户；每用户原beam20/return20/max_new_tokens10，全部生成token满足原逐位置allowed callback，均第5个token EOS；未换trie、未OOM降beam。非作者还直接用冻结SID index和tokenmap重建allowed集合核验全部200 beams（含warmup40）/1000 token。未计算这些样本的推荐命中率。

原件：[推理r0](evidence/autograd_20261110/jobs/letter-20261110-autograd-inference-r0)，[rank0](evidence/autograd_20261110/jobs/letter-20261110-autograd-inference-r0/output/inference/inference_rank0.json)、[rank1](evidence/autograd_20261110/jobs/letter-20261110-autograd-inference-r0/output/inference/inference_rank1.json)，[独立重载/生成/预算验收](review/autograd_20261110/inference_budget_runtime_review.json)。

**5. 一个seed的96小时条件估计**

保留实际131,837训练行/epoch、4epoch=527,348基础呈现。实际构造而未遍历的正式Trainer loader是 Accelerate `DataLoaderShard`，其 `BatchSamplerShard` 为 `even_batches=True, split_batches=False, num_processes=2`，内部 `BatchSampler/RandomSampler`；真实记录的每rank loader长度 **8,240 batches**，micro8/累积4对应每epoch2,060更新，四轮 **8,240更新**。预算JSON中的65,919行/rank与131,838总行只是DistributedSampler算术对照，不能当作这次实际loader的呈现/尾部事实。9个完整global64没有执行真实epoch尾，不能断言实际尾部是部分microbatch还是even-batches补齐。保守工作核算按8,240个global64等价单位；原始结构见rank JSON的 `formal_train_dataloader_inspection`。

验证按每epoch两模板49,544条、四次198,176条；测试按prompt0/beam20全部24,772用户。中心训练使用3次稳态mixed均值加CPU准备；保守训练取post-warmup中最慢测量；生成中心按已审24,772用户长度histogram分层映射，保守取最慢代表用户。裕量10%/25%是明确假设，不是置信区间。

| 分项（小时） | 中心 | 保守 |
|---|---:|---:|
| 训练 | 10.184357 | 10.237876 |
| 两模板验证×4（含准备） | .969642 | .969642 |
| adapter保存×4代理 | .001123 | .005024 |
| adapter重新加载 | .026235 | .026235 |
| prompt0 beam20完整测试 | 1.925451 | 2.668408 |
| 初始化（训练+推理base） | .026815 | .026815 |
| 环境/I/O裕量 | 1.313362 | 3.483500 |
| **条件合计** | **14.446985** | **17.417500** |
| **至96h剩余** | **81.553015** | **78.582500** |

**仍未量化**：正式Trainer/DeepSpeed optimizer、scheduler、RNG和其他训练状态checkpoint；best选择/restore及最终Trainer保存；正式验证worker/aggregation额外开销和完整CPU过滤/metric聚合。没有越过禁止写完整训练checkpoint或完整测试指标的边界。长期热状态、共享I/O及内存碎片也未测。故完整seed验收为INCONCLUSIVE；条件余量可容纳这些未测开销的总和，不能称它们为0。

在已测其他成本及10%裕量下，训练数学门槛约 **1.737186 samples/s**，当前14.657983 samples/s；有证据支持候选路径可行，没有证据证明已完成正式seed。原始计时、真实分母、全部分项、缺测与风险在[预算JSON](evidence/autograd_20261110/budget_estimate.json)，可用[脚本](scripts/estimate_budget.py)重算；非作者未导入该脚本，独立从raw秒数及全量histogram重算全部分项，结果一致。

**6. 资源、预算、修正及停止**

| 指标 | rank0 / rank1 |
|---|---|
| 训练CUDA allocated峰GiB | 12.033432 / 12.038423 |
| 训练CUDA reserved峰GiB | 14.269531 / 14.275391 |
| 推理CUDA allocated峰GiB | 18.314717 / 18.314717 |
| 推理CUDA reserved峰GiB | 21.912109 / 21.912109 |
| 训练进程ru_maxrss GiB | 10.669685 / 10.682297 |
| 推理进程ru_maxrss GiB | 10.488029 / 10.499821 |

主机采样最低可用234.077946GiB，始终高于40GiB；task swap、swap增长及swap I/O未触发。5秒采样的任务树RSS会漏峰，不把其max当精确并发峰，也不直接相加两个rank历史ru_maxrss。[资源原始记录](evidence/autograd_20261110/jobs/letter-20261110-autograd-gpu-r0/output/resources/resource-samples.jsonl) 与[最终资源](evidence/autograd_20261110/final_resources.json)保留。

原任务7.788774、本轮9.158351，累计 **16.947124 / 120 GPU-device分钟**，剩余103.052876。新训练/推理guard wall分别137.376207／137.374308秒，每次均按两倍全进程时间计费，包含加载/保存。新CPU小测试未使用CUDA，不计GPU算时。历史ledger缺失或使用量低于冻结467.326422秒即拒绝新GPU作业，不会清零。[累计ledger](evidence/autograd_20261110/gpu_budget_ledger.json)、[guard](config/guard_autograd.json)、[本任务修正记录](evidence/autograd_20261110/correction_rounds.json) 与旧记录分开保留。

guard统计相关保留目录和新job的分配峰2,092,994,560bytes，包含继承环境/资产，远低于30GiB；adapter不进入GitHub或ZIP。没有删除旧证据、模型、数据、环境、CUDA开发组件或未跟踪Qwen文件。

本轮三个job均SUCCEEDED/exit0/cleanup_verified，worker/group已结束，最终两卡均1MiB无compute进程。**没有活动任务，不自动重启或继续训练。** 全部9次技术更新额度已使用，剩余GPU分钟不授权更多更新。是否正式采用baseline及精度适配仍待用户/网页GPT/学长决定。

**7. 审核、版本及交付入口**

非作者 `/root/independent_review` 未参与修复/探针实现；实际直接读取旧错误、冻结PEFT原件、新源码、request/gitblob、两rank结果、原日志、资源和export成员hash。审核分阶段为CPU、完整训练及保存、实际重载/beam20和预算算术PASS，均有明确范围；未冒称独立运行模型或正式seed PASS。

| 身份 | 固定值 |
|---|---|
| 历史原交付 | `61854bb1df0f26a71e68d95ab8eb564f7c762f76` |
| CPU小测试RUN_COMMIT | `059f9feb55fde067187afbf3b80f285a00a63ec3` |
| GPU训练/推理/估计器RUN_COMMIT | `8e6399c2a1269dfde58d705eef25a499400dc59b` |
| 服务器固定源码export | `/home/bwwu/lc-rec/letter-20261110/releases/8e6399c2a1269dfde58d705eef25a499400dc59b/` |
| 服务器job原件 | `/home/bwwu/grid-audits/sid-extensions-p0/jobs/letter-20261110-autograd-{cpu,gpu,inference}-r0/` |
| LETTER / LLaMA固定来源 | `8d0154e28de37dbb6e24871c508ad8ddb1921cda` / `4782ad278652c7c71b72204d462d6d01eaaf7549` |

服务器是无.git的固定export，部署RUN_VERSION逐文件验证；不假称checkout。最终交付提交与RUN提交不同，实际运行及估计器源码/配置字节未变，见[summary.source](evidence/autograd_20261110/summary.json)。最终完整DELIVERY_COMMIT、branch HEAD/关键文件真实回读和最终服务器export对应放在ZIP `final_receipts/publication.json`及同目录回执，并随最终回复给出，避免把自己的最终SHA嵌入自身。

GitHub统一入口即本文件。轻量ZIP包含本轮与必要旧失败证据、实际源码/配置、原始日志/各rank/资源/预算、非作者审核、外部发布和服务器回执、成员SHA-256；不含权重、adapter或wheel。实际ZIP CRC、解包逐成员hash、重复/危险路径、秘密扫描与Windows可访问性在包外回执验证。

**GPT_UPLOAD_REQUIRED: YES**。请将最终回复提供的本轮轻量ZIP整体交网页GPT复核，不必逐个寻找服务器回执。
