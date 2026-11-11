# A0 Instruments seed42：独立方法与恢复验收终审

**结论：NO-GO。完整 checkpoint 跨进程续训未通过本轮固定验收条件，不得启动正式四轮 A0 Normal。** 本轮只有有界验收更新，没有正式训练、正式 validation 选模或全量推荐测试结果。进程退出 0 与恢复正确性通过是不同结论。

审核者：`/root/a0_method_review`，真实 Codex 子代理；2026-11-11。本审核者未参与训练实现，未安装依赖、加载模型、运行 GPU 或提交远端作业。唯一写入文件是本审核说明。已直接读取代码、原始双 rank 日志、request/status、checkpoint 清单及 CPU 比较报告，并使用本地 Python 标准库独立复算日志中的批次身份、学习率、loss 与 grad norm 差异。

## 1. 固定身份、材料与审核边界

本轮最终验收源码提交为 `87ba6556ce7237f8ef32c690165449d0d6a28294`，分支 `wbw-lc-rec`。直接比较本地文件、该提交 Git blob 与真实 request 的输入 SHA，以下六项全部一致：

| 文件 | SHA-256 |
|---|---|
| `scripts/formal_train.py` | `e6616dbb8ece1b66db3a8500b3a84870b4cbbbf4b7d25bdd85e0f9f9d173076c` |
| `scripts/formal_pipeline.py` | `9d714977a2826fec26000859db8ca16229a567971ce19cb159dfadb318db6ca0` |
| `scripts/formal_evaluate.py` | `e5c6cbb4b10cace5b1f97097d4157862b45f193d023a6f5ac4e1878ee1a5f2e9` |
| `scripts/formal_compare.py` | `98e3c142503569ade29153afec29a69494cac3309d46c983704b3ee9bc69fbee` |
| `scripts/compat.py` | `c8135e5eeda3c1f528065cbe717a31582cbb4b5043ea13bfa76f01b370604a20` |
| `config/formal_instruments_seed42.json` | `9e058cecb7151ebee3ab714ab8328af0008537156cd96684c988d89e7ee10600` |

上述路径相对 `lc_rec/letter/`。服务器实际源码为无 Git 的 release export：`/home/bwwu/lc-rec/letter-a0-normal-20261111/releases/87ba6556ce7237f8ef32c690165449d0d6a28294/`，不能称服务器 Git checkout。

审核已显式读取 `/home/addaswsw/recommend/{AGENTS,PROJECT_RULES,REMOTE_EXECUTION}.md`。原始资料在本代理同一会话中实际读取：

- [LETTER 论文 v2](https://arxiv.org/html/2405.07314v2)：loss、LoRA 实例化、推理叙述及 Instruments 表格参照。
- [作者固定源码](https://github.com/HonghuiBao2000/LETTER/tree/8d0154e28de37dbb6e24871c508ad8ddb1921cda)：通过 GitHub 连接器逐文件回读。data/collator/modeling/utils/evaluate/test/prompt/ZeRO-2 配置八文件的 Git blob SHA 与本地 vendor 一致；另直接读取固定 `lora_finetune.py`。
- [作者后补 Google Doc](https://docs.google.com/document/d/1rVeu3JqGXZNjapiMi1ioXIlXdXSt8zl94FmaAFYwX4g/edit)：通过 Drive 连接器读取，Instruments 配方与本地保存内容一致。它决定本轮配置，不唯一证明作者历史 wheel、设备数、argv 或 checkpoint。
- 实际导出的 Transformers 4.31、Accelerate 0.21、DeepSpeed 0.16.9、PyTorch 2.6 源码；并回读官方 [Accelerate 0.21 DS wrapper](https://github.com/huggingface/accelerate/blob/v0.21.0/src/accelerate/utils/deepspeed.py)、[DataLoaderShard](https://github.com/huggingface/accelerate/blob/v0.21.0/src/accelerate/data_loader.py)、[PyTorch RandomSampler](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/utils/data/sampler.py)。当前官方 Accelerate 的 DS stage≥2 `no_sync` 处理也实际读取，回读 blob 为 `e666b8d7a07c789245bed88c9e68f220a81eb3e7`。

本审核者没有亲自连接服务器或重新读取约 2.3 GB/份的四个原始 acceptance state payload。张量级结论来自已保存的服务器 CPU 全状态比较报告；本审核者检查了比较程序、报告与内容哈希，并另用原始 JSONL 复算关键标量和离散序列。没有把主控执行的远端检查描述为审核者现场重验。

## 2. 可确认的方法实施范围

### 输入与方法身份

独立解析本地固定作者资产，确认：24,772 用户、9,922 item；item 轴与 SID 轴一致，所有交互 item 有对应 SID；每项四位置，918 个新增 token、9,897 个唯一完整 SID、23 个碰撞组（涉及 48 个 item）；训练 131,837 条、验证两模板构造 49,544 条、测试 24,772 用户。三项 SHA 与任务固定值一致：

- SID：`33c87e28c52ae367985bdf494cb49d676d181e13a39497b4ad035f46111546dd`
- 交互：`0a11d47791d90483005da93c36d826679e20e4592d1d2b86518d5f56f218e743`
- item：`1d9938e576a8d964e6b977a2b63aa56402840a788eaa6685f26f0d910a7133f2`

真实双 rank 报告再次记录相同输入哈希。模型为固定 `huggyllama/llama-7b` revision `4782ad278652c7c71b72204d462d6d01eaaf7549`；复用了既有官方 LFS 校验回执，本轮记录配置哈希、权重大小与文件身份。审核者没有重复哈希约 13 GB 权重，因此模型大载荷身份结论依赖该回执及本轮轻量复核，不能说审核者重新完成全权重官方校验。

本轮使用现成作者上游 SID，没有复现上游 embedding/SASRec/RQ-VAE/SID 导出。配置保留 Instruments SeqRec 单任务、seed42、r8/alpha16/dropout0.05、七类投影、active embedding/head、AdamW 2e-4/weight_decay0.01、cosine/warmup0.01、历史20、最大2048、temperature0.8、四 epoch 计划。双卡 micro8×accum4 为本轮明确批准布局，不宣称与作者未知历史四卡路径数值逐位同轨。

### 监督、参数与兼容

vendor `collator.py` 保留原始 instruction/response，response 后附有效 EOS，prompt/padding 屏蔽；`modeling_letter.py` 用 FP32 logits 除以 temperature 计算 causal CE。相关 vendor 文件未改，本次短路径消费原始 collator/model。旧完整数据与 temperature/gradient 审计只作为历史技术证据复用，不冒充本轮完整实验。

本次入口与真实 rank 报告确认预期 450 个 trainable 张量、289,652,736 元素，224 个 LoRA 模块及两项 active embedding/head；冻结 original_module，优化器成员与 trainable 集合有明确断言。

用户已明确批准既有 A-only BF16 正式沿用，见 [precision_approval.json](../../evidence/formal_20261111/precision_approval.json)。当前 `compat.py` 字节与既有已审版本相同。实际两个路径各 rank 都命中恰好 224 个 LoRA A，组合为 FP32 input/BF16 weight、外层 autocast=False、局部 autocast=True、退出局部上下文后=False。out-of-place residual、bnb 旧字段与 FastChat unpad arity guard 保留。此处可确认局部适配生效与范围受限，不能证明作者历史环境数值等价，也不能由此解释恢复不一致的唯一根因。

实例级 ZeRO-2 `no_sync` 适配保留 DS 自身 backward/loss scaling/partition/step，其他模型路径使用原上下文。实际连续路径每 rank 12 次 bypass、恢复路径 6 次，对应每个 global update 的前三个 microbatch；恢复区间实际 DS 边界为 `False,False,False,True` 重复两次。四个 microbatch 才发生一次 global update，未把 LR=0 的首更新当作有效参数训练收益。

### 验证、选模与测试

静态入口保留 epoch validation/save、`eval_delay=1`、最低 `eval_loss` 选 best、`load_best_model_at_end=True` 及作者 `save_total_limit=1` 语义。实现增加完整 DS、外部 HF scheduler、loader/RNG 状态保存；不是使用原作者 adapter-only resume 冒充完整恢复。

测试源码保留原 LoRA 推理分支的 BF16 非量化 base、prompt0、beam20、逐位置 allowed-token callback、`filter_items=False`、完整 SID 字符串匹配和原 evaluator。无严格 Trie 替换、无碰撞消歧、无候选 dedup；无效/重复/碰撞另计。完整用户覆盖和指标复算在代码中有检查。**本轮没有执行该全量测试；源码静态一致不等于评价已通过。** 本轮也没有实际 epoch validation/best checkpoint，短验收 `best_checkpoint=None` 是预期边界，不是已选出最佳模型。

## 3. 真实跨进程恢复结果：部分结构成功，数值验收失败

实际两个作业均绑定同一最终源码：

- 连续路径：`letter-a0-20261111-acceptance-cont-r3`，每 rank 从 global step0 到4，共16个训练 microbatch。
- 新进程路径：`letter-a0-20261111-acceptance-resume-r3`，从上述 `checkpoint-2` 恢复到4，每 rank8个训练 microbatch。

原训练进程已退出后才启动新恢复进程。真实 request/status、两个 rank 的 `strict_ds_restored`、`loader_replayed` 和 optimizer 事件支持“确有跨进程加载并继续两次更新”，不是只做 adapter 推理或重新从 step0 训练。collective checkpoint 清单包含双方 ZeRO optimizer 分片、完整 model state、各 rank RNG/loader state、外部 scheduler、Trainer state 与 adapter。清单本身不证明所有大 payload 数值连续正确。

本轮固定比较标准为 `abs(resumed-reference) <= 1e-6 + 1e-3*abs(reference)`，离散状态/RNG/scheduler/step 要求 exact；没有观察失败后放宽阈值。[acceptance_comparison.json](../../evidence/formal_20261111/acceptance_comparison.json) 的真实 SHA-256 为 `1819c4b076e4e88414534fa32f450370ceb6f04aa3f17c876613ee4df785b4bf`，大小 1,014,752 bytes，已与外部取回回执核对。

独立审核所得：

| 检查 | 结果与边界 |
|---|---|
| 恢复批身份 | 每 rank 全部8批的 step、loader_yielded、input/label SHA、shape 与连续路径相同 |
| step/LR/边界 | 恢复后更新3、4及LR一致；FFFT/FFFT，含非零LR真实更新 |
| RNG | 报告中的 CPU、独立loader generator、两 CUDA RNG tensor exact；NumPy数组 exact。已识别的 Python/离散/scheduler 状态无相应不一致条目 |
| 有限值 | 所有被比较模型/optimizer tensor 为 finite，不代表数值等价通过 |
| 模型状态 | 两 rank 各450个 trainable tensor 均有元素超过固定容差，包括 active embedding/head |
| optimizer 状态 | 两 rank 的 FP32 master、Adam `exp_avg`、`exp_avg_sq` 均有超容差元素 |
| 比较器覆盖缺口 | 各1项 `zero_stage.__dict__.__objclass__` 的 EnumMeta 类型不支持；不能说该字段完整通过 |

因此，每 rank 的454条失败应准确拆为 **453条浮点失败（450模型＋3 optimizer）和1条 schema/类型覆盖缺口**，不能把454全部称为数值错误，也不能把失败全部归因于 enum 比较器。

本审核者直接从双 rank JSONL 独立复算：恢复后第一个 update 的四个 micro-loss 与连续路径完全相同；第二个 update 的 loss 开始有差异。rank0 有3个 loss 超限，最大差 `0.004361867904663086`；rank1 有1个超限，最大差 `0.004642486572265625`。这些是 `Trainer.training_step` 已除以4的日志值。

第3次 update 的 grad norm 为连续 `62.56051254272461`、恢复 `62.56745529174805`，在此容差内；第4次为连续 `67.86307525634766`、恢复 `64.69631958007812`，差 `3.1667556762695312`，相对连续值约 **4.66639%**，超限。原比较程序在 loss 失败时已抛错，未把后续 grad norm 比较结果完整写进 events 子结构；这里的 grad norm 结论来自审核者独立读原始事件复算。

上述结果排除了“仅凭同step或同输入即可判恢复通过”的说法。现有材料不能唯一定位是 optimizer/quantization/runtime 状态、非确定性数值路径或其他因素；不宣称漏存 optimizer 为已证实根因，不宣称已证明所有差异只是 BF16 非确定性。`rtol=1e-3` 比常见 BF16 单 ULP 更严格这一事实已在比较器预声明，但不能事后据此自动改判 PASS。

## 4. 修正预算、资源与停止决定

[corrections.json](../../evidence/formal_20261111/corrections.json) 与保留的原始失败日志显示：round0 在更新前发现 HF 外部 scheduler 假设错误；round1 在首训练 microbatch 前遇到 DS ZeRO-2/no_sync 不兼容；round2 的源码 guard 误读 contextmanager wrapper 并在训练前拒绝；round3 修正该 guard 后完成连续与恢复路径，但最终状态比较 FAIL。初次 round0 加三个集中修正已经用尽，不允许第四轮、阈值放宽或在此缺口下启动正式任务。

CPU 实际 DataLoaderShard 合成 fixture 的中途/下一 epoch replay PASS，及上述真实 batch 哈希相同，只证明输入/RNG replay 的相应覆盖范围；不能替代完整模型与优化状态续训验收。

[final_resource_and_ledger.json](../../evidence/formal_20261111/final_resource_and_ledger.json) 记录五项验收尝试累计 `1212.328182` GPU-device-seconds，即 `20.2054697` device-min，未耗尽初始60 device-min预算。该数值按双卡占用时间记账，不是GPU利用率积分。此次停止是必要正确性门槛与三轮修正上限触发，不是96小时正式墙钟用尽。

截至该服务器回执 `2026-11-11T18:15:02.460641+00:00`，正式 ledger/正式目录不存在，两张 GPU 均约1 MiB且无compute进程；各作业终态及 cleanup receipt 均保留。此为有时间点的服务器回执，本审核者没有现场实时查询。runner 持久化、会话断开、pidfd清理和部署发布的完整验收由独立执行审核及最终交付回执覆盖，本说明不替其签发完整 PASS。

## 5. 最终可发表结论

- **方法与资产：** 固定作者资产、确认配方和关键源码约束有证据支持；只能对已读代码和短路径覆盖作局部实施结论。
- **恢复前置条件：FAIL/NO-GO。** 跨进程加载、输入replay和继续更新确实发生，但完整数值状态连续性未在固定标准下通过，另有一个状态类型比较缺口。
- **正式运行完成性：未完成且未启动。** 四轮正式训练、49,544条正式validation收口、best选择和24,772用户正式test均没有完成。
- **论文数值重现：不可评价。** 没有真实正式Recall/HR/NDCG，不能与0.1115/0.0854作实验差值，也不能把短验收loss当推荐指标。
- **研究结论：无新增结论。** 没有多seed稳定性或Shuffle顺序不变性证据。未启动 A1、其他模型、额外seed或其他数据集。

应交付失败证据、固定源码与未完成边界后停止。后续若要继续定位或修改恢复实现，需要用户重新授权；不得用本审核替代正式运行许可或宣称 A0 Normal PASS。
