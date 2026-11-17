# LETTER-LC-Rec：SID 顺序扰动

[返回主题](README.md) · [逐条件来源和完整精度](sources.json)。作者 LETTER 原件固定 `HonghuiBao2000/LETTER@8d0154e28de37dbb6e24871c508ad8ddb1921cda`，与 `RUCAIBox/LC-Rec@e6f1b3b3f8b4fee55a49a7b3dfeba29dbfcf0669` 的原六任务 vendor 不同。本页的 A0/A1 是 LETTER 的模型变体编号，不是 GRID A/B。

## 数据到生成

作者 `LETTER-LC-Rec/data.py::SeqRecDataset` 根据 split 构造序列推荐样本：测试保留最后物品为目标、历史至多最近20件；`prompt.py` 提供模板，正式测试固定 Prompt0。`collator.py::Collator` 构造训练 input/response、EOS、padding 和 response-only labels；`TestCollator` 生成测试 input IDs 与 attention mask。SID 是每件物品四个注册 token，原层级 token 身份保留；真实 tokenizer、资产及交互划分以各模型 RUN 为准。模型使用自回归语言模型，训练为 response-only CE，LoRA 训练并保存指定 embedding/head；epoch 验证按较低 eval_loss 选最佳 checkpoint。正式测试每用户 beam20、return20、max_new_tokens10，沿用每位置允许 token 回调及 EOS，`filter_items=False`，不是完整目录 trie，也没有以 OOM 减小 beam。

| 变体 | 实际配置和选模 | 资产身份 |
|---|---|---|
| Instruments A0 | seed42、4 epoch、8240 updates、best 8240；micro8 × accum4 × 2 = 64，AdamW 2e-4、wd .01、cosine warmup .01；Int8 主干/BF16、LoRA r8 alpha16 dropout .05，FA2 deterministic backward 适配。 | LLaMA-7B `4782ad278652c7c71b72204d462d6d01eaaf7549`；918 SID token；[正式回执](https://github.com/woobowen/recommend/blob/3396d45a041424fe4af625bfc0b047467150c5c1/lc_rec/letter/A0_NORMAL_FINAL_REVIEW_ENTRY.md)。 |
| Instruments A1 | seed42、4 epoch、best 8240；global64、LoRA r8、tied embedding/head 的严格保存和加载；训练配置 `load_in_8bit` 与正式 Shuffle 推理 BF16 unquantized Base 分阶段记录。 | Qwen3-0.6B-Base `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`；[正式回执](https://github.com/woobowen/recommend/blob/3396d45a041424fe4af625bfc0b047467150c5c1/lc_rec/a1_qwen3_06b/A1_NORMAL_FINAL_REVIEW_ENTRY.md)。 |
| Beauty A0 | seed42、4 epoch、8214 updates、best 6160；global64、AdamW 2e-4、LoRA r8、BF16；同一个 best adapter 用于三组。 | LLaMA-7B 同 revision；965 SID token；数据身份及 S4 补算见 [最终审计](https://github.com/woobowen/recommend/blob/12eb8b176b156f3be7d7533ee8f443db02441303/lc_rec/a0_beauty/evidence/paired-s4-supplement-20261117/FINAL_AUDIT.md)。 |

Instruments 作者数据 index/inter/item SHA 分别为 `33c87e28c52ae367985bdf494cb49d676d181e13a39497b4ad035f46111546dd`、`0a11d47791d90483005da93c36d826679e20e4592d1d2b86518d5f56f218e743`、`1d9938e576a8d964e6b977a2b63aa56402840a788eaa6685f26f0d910a7133f2`；Beauty 数据身份见其最终审计。二者不是因为同名而共用映射。本次只读历史清单，没有重取大数据或权重。

## 干预和专用入口

Instruments 的 [entry.py](https://github.com/woobowen/recommend/blob/f0242b9f90ae23a4979d688a287734ca9f88cea1/lc_rec/shuffle/entry.py) 校验各自冻结 Normal 推理入口 SHA，生成每 rank 派生源码和 diff；`runtime.py::intervene` 在原完整 prompt tokenization 与 collate 后、`generate` 前处理 input IDs。[core.py::transform](https://github.com/woobowen/recommend/blob/f0242b9f90ae23a4979d688a287734ca9f88cea1/lc_rec/shuffle/core.py) 校验 SID 组及 mask，仅在各历史物品四码组内部置换，保持物品顺序、目标、组外 token、mask 和 token 身份。S3 从 3!−1 个非恒等排列选取，第四位固定；S4 从 4!−1 选取。以协议、seed42、用户行和历史物品位置 SHA256 拒绝采样，不依赖 rank/batch；一次生成不重抽。生成沿原 position/RoPE、causal mask 与空初始 cache，不输入旧位置或旧 cache；A1 的适配回调先验原 prompt，再绑定新前缀。层级 token 身份仍存在，不能说隐去了所有位置信息。

Beauty 固定 RUN `6e1b211d9c37f358c43b1540b40b17f6703150d2` 有自己的 [shuffle_entry.py](https://github.com/woobowen/recommend/blob/6e1b211d9c37f358c43b1540b40b17f6703150d2/lc_rec/a0_beauty/shuffle_entry.py)、[shuffle_runtime.py](https://github.com/woobowen/recommend/blob/6e1b211d9c37f358c43b1540b40b17f6703150d2/lc_rec/a0_beauty/shuffle_runtime.py) 和 [formal_evaluate.py](https://github.com/woobowen/recommend/blob/6e1b211d9c37f358c43b1540b40b17f6703150d2/lc_rec/a0_beauty/formal_evaluate.py)。前者校验 Beauty 正常入口 SHA `998c5b3201954d9c72fa71aeb5ba6e7473dc6ce6e5fcce342889f6fde4daf20f`，保存派生文件及 diff；Beauty runtime 显式导入共用 `lc_rec/shuffle/core.py` 并核 965 个 SID token。`formal_evaluate.py` 实际装载作者 `SeqRecDataset`、`TestCollator`、Prompt0、固定用户轴及双 rank 不等长 sampler，再调用原 positional callback 和 generate。它不是 Instruments 入口的简单同名引用。

额外修改分别是：FA2 backward deterministic 及双 rank 工程边界（训练行为须按原 RUN 阅读）；Beauty 域数据、965 token 与不等长分片（数据域适配）；Shuffle 的 token 组内置换（研究干预）；CPU CCE、分组与 Bootstrap（评价扩展）。这些改变的来源及对应阶段在映射中，不把所有差异称为环境适配。

## Beauty 阶段来源与训练适配

训练、生成、评价是不同 Runner 阶段：[训练 request](https://github.com/woobowen/recommend/blob/12eb8b176b156f3be7d7533ee8f443db02441303/lc_rec/a0_beauty/evidence/formal-launch-20261116/request.json) 的 `beauty_stage=train`，仅证明训练调用，不能充当生成 request。[归档 state](https://github.com/woobowen/recommend/blob/12eb8b176b156f3be7d7533ee8f443db02441303/lc_rec/a0_beauty/evidence/paired-s4-supplement-20261117/evidence/original/control/state.json) 的 `attempts` 分别记录 `normal`/`s3`/`s4` 与 `normal_eval`/`s3_eval`/`s4_eval` 的 job、attempt、请求路径、摘要和实际 worker argv；[sources.json](sources.json) 对每组给出对应 selector、预测 rank 文件身份与 primary metrics。六个生成/评价 request 的原始字节未随该证据归档；记录的请求摘要不是本轮对原文件字节重算的结果。训练 request 原件的原始文件 SHA 为 `f930c136e44fce00cad5bf05573779144e43aa108184c1dfc603762be20002b8`；对其排序键、紧凑 JSON 的规范化 SHA 为 state 中 `abd09c34bc141f0f97772a54ebd85d467c48d3f2d00252b0c8e5de5e98ac1e86`，两种身份不混用。

[固定训练源码 formal_train.py](https://github.com/woobowen/recommend/blob/6e1b211d9c37f358c43b1540b40b17f6703150d2/lc_rec/a0_beauty/formal_train.py)、[trainer_loop_patch.py](https://github.com/woobowen/recommend/blob/6e1b211d9c37f358c43b1540b40b17f6703150d2/lc_rec/a0_beauty/trainer_loop_patch.py)、[zero2_partial_state.py](https://github.com/woobowen/recommend/blob/6e1b211d9c37f358c43b1540b40b17f6703150d2/lc_rec/a0_beauty/zero2_partial_state.py) 与 [既有工程审核](https://github.com/woobowen/recommend/blob/12eb8b176b156f3be7d7533ee8f443db02441303/lc_rec/a0_beauty/FORMAL_ENGINEERING_REVIEW_ENTRY.md) 是下表依据。早期补丁文件的 DRAFT 注释属于其写作阶段，后来授权和正式运行以固定 Plan、训练 request 与最终审计为准。

| 实际改动 | 原因与保持项 | 源码及既有核验、边界 |
|---|---|---|
| `AuditTrainer`、`build_trainer_loop` 和 `sync_with_dataloader=False` 跨 epoch 保留未满的梯度累积 | 消费完整四轮微批后再按 GAS4 更新；micro-batch 与 optimizer update 不等价。epoch 验证发生在边界，8214 是已批准协议中的更新数，不是追加重训；原 sampler 补齐保留，每轮审计 11 个重复样本。 | `formal_train.py::Callback.on_epoch_end`、`trainer_loop_patch.py::patch_text`；[短程验收材料](https://github.com/woobowen/recommend/blob/12eb8b176b156f3be7d7533ee8f443db02441303/lc_rec/a0_beauty/evidence/e2e-20261116-r0/manifest.json)只覆盖其试验范围。 |
| `IndexedDataset` / `IndexedCollator` 包装训练索引，验证时恢复作者 collator | 训练索引用于核对实际消费顺序，验证数据是原字典结构，不能用训练索引包装的 collator 直接处理；模型监督及验证目的保留。 | `training_rows.py`、`AuditTrainer.get_eval_dataloader`；[第一轮输入及损失比较](https://github.com/woobowen/recommend/blob/12eb8b176b156f3be7d7533ee8f443db02441303/lc_rec/a0_beauty/evidence/e2e-20261116-r0/early-comparison.json)，只证明其中所列容差与样本范围。 |
| `ReplayLoader`、`saved_partial`、`partial_snapshot` 和 restore | 记录数据顺序、未完成累积的梯度及 micro-step、RNG 和 scheduler；恢复前先重放 CPU 数据构造并核顺序，DS 严格恢复 optimizer 等状态。 | `formal_train.py::ReplayLoader`、`zero2_partial_state.py::snapshot/restore`；Continuous/Early/Resumed 是有界预验收，不承诺任意环境或中断均逐位恢复。正式训练实际是否恢复不能从恢复能力推定；本次不声称发生过正式 resume。 |
| ZeRO-2 `no_sync` 兼容限定当前实例 | Accelerate 0.21.0 与 DeepSpeed 0.16.9 组合下，仅对指定 stage-2 engine 使用局部上下文，让 DS 保留自身累积和同步行为；非全局关闭同步。 | `formal_train.py::zero2_native_accumulation` 中版本、engine、GAS 和 batch 断言；不得外推其他版本。 |
| checkpoint 完成标记与 receipt 后轮转，存储上限2 | 集体保存模型/optimizer、loader 和 partial state 后写 `complete.json` 与 receipt，再轮转；保留 best 与最新恢复点。按最低 validation loss 选 checkpoint-6160 并核导出 adapter，非选最后 8214。 | `AuditTrainer._save_checkpoint`、`checkpoint_receipt.py`；[最终审计](https://github.com/woobowen/recommend/blob/12eb8b176b156f3be7d7533ee8f443db02441303/lc_rec/a0_beauty/evidence/paired-s4-supplement-20261117/FINAL_AUDIT.md)绑定 adapter。 |

FA2 deterministic backward 是已有共享适配；训练实际以 Int8 主干加 LoRA/BF16 配置加载，生成回执另记录 BF16 unquantized Base。模型、loss 和已确认核心超参数的沿用不意味着跨 epoch 调度、保存及恢复行为无需单独说明，也不宣称任何运行数学轨迹普遍等价。S3 复用原配对产物，S4 采用独立补算；Normal 是对照，没有单独 Shuffle 配对作业。

## 结果与状态

表为原 SID 字段，0–1 比例；完整 16 指标、碰撞修正 ItemHit/ItemNDCG（CCE）、覆盖率、分组和 5000 次用户 Bootstrap 见 [Instruments 结果](https://github.com/woobowen/recommend/blob/3396d45a041424fe4af625bfc0b047467150c5c1/lc_rec/shuffle/results/RESULTS.md) 和 [Beauty 最终审计](https://github.com/woobowen/recommend/blob/12eb8b176b156f3be7d7533ee8f443db02441303/lc_rec/a0_beauty/evidence/paired-s4-supplement-20261117/FINAL_AUDIT.md)。用户 Bootstrap 是冻结模型和一次置换下的不确定性，不是训练 seed 方差；没有 S4−S3 专项配对检验。分组数字是描述统计，不借总体 CI 说明分组显著。

| 模型 | 域 | 条件 | Hit@5 | NDCG@5 | Hit@10 | NDCG@10 |
|---|---|---|---:|---:|---:|---:|
| LETTER A0 | Instruments | normal | 0.090303569 | 0.077663104 | 0.113353787 | 0.085081132 |
| LETTER A0 | Instruments | S3 | 0.067374455 | 0.052773019 | 0.083763927 | 0.058035628 |
| LETTER A0 | Instruments | S4 | 0.064629420 | 0.049711914 | 0.081543678 | 0.055129293 |
| LETTER A1 | Instruments | normal | 0.087195220 | 0.074377047 | 0.106935249 | 0.080685962 |
| LETTER A1 | Instruments | S3 | 0.068262554 | 0.052706185 | 0.085863071 | 0.058358425 |
| LETTER A1 | Instruments | S4 | 0.064306475 | 0.047821343 | 0.082270305 | 0.053586629 |
| LETTER A0 | Beauty | normal | 0.049277825 | 0.033363477 | 0.071189018 | 0.040369525 |
| LETTER A0 | Beauty | s3 | 0.020524974 | 0.012761355 | 0.035326208 | 0.017507923 |
| LETTER A0 | Beauty | s4 | 0.017886688 | 0.010823148 | 0.032151321 | 0.015404917 |

Beauty 原 GPU 训练和生成完成；原 paired job `TIMED_OUT/-15`、原 collect 未运行。S4 由 `1934c01eee767b665ef1043995ffc741958e30e0` 独立 CPU job 补算，补充汇总完成；原失败未被回填。原审计写明真正非作者科研审核及网页最终二审待完成，用户转交的本轮方案外审不替代它。
