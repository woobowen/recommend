# C2/C3/C4 独立静态复核：checkpoint、正式预算与数据依赖

本报告由真实子代理 `/root/checkpoint_budget` 于 2026-11-09 直接读取固定官方原件、已保存依赖源码、当前候选配置和既有轻量执行回执后形成。该代理不是本轮 offload 实现作者，也不是既有 checkpoint 报告作者。**结论是静态审核通过并发现正式运行门槛；保存、恢复、导出和正式训练仍为 NOT TESTED。** 本报告不改变主控 GPU 作业的实际结果。

已从外部显式读取 `/home/addaswsw/recommend/AGENTS.md`、`PROJECT_RULES.md`、`REMOTE_EXECUTION.md`，不是声称这些规则自动跨同级工作树发现。首个 shell 命令为入口目录的 `ls -la`，结果含并列三份规则、两处原件目录和研究 PDF；该入口与本次只读工作树 `/home/addaswsw/recommend-worktrees/lc-rec` 不同。未操作 Git/SSH、安装包、启动 GPU/模型/数据全审计或写 checkpoint；只写本目录的有限交付材料。

源码和回执的实际大小/SHA-256、算术及验证记录在 [findings.json](findings.json)。已重新计算既有 `checkpoint_source_identity.json` 中本地依赖源码的 SHA-256，全部匹配。源码身份包括 DeepSpeed 0.16.9 固定提交 `bdba8231bc8fc17980a5941437e6363dac69418d`、Transformers 4.31.0、Accelerate 0.21.0、Torch 2.6.0；这不是对新版本行为的推测。作者原件绑定 `e6f1b3b3f8b4fee55a49a7b3dfeba29dbfcf0669`。

## 1. 原生 checkpoint：master 不会在 model/optimizer 里双写一份完整参数

令 `N=6,746,034,176`（930 新 SID 行后的可训练参数），`B=13,476,876,661 bytes`（已有基础 safetensors 两片合计）。两种批准 offload 候选均 `stage3_gather_16bit_weights_on_model_save=false`。以下为两 DP rank 合计的**静态主 tensor 载荷**，所有场景还要加非负 `H`（buffers、元数据、padding/storage 余量、RNG/tokenizer/config、序列化及文件系统开销）；不是精确文件大小或峰值上界。

| 源码与行 | 直接读取的保存语义 |
|---|---|
| `ds_stage3.py:2635–2648` | `_rigid_state_dict` 写 optimizer state 和 `FP32_FLAT_GROUPS`。全参 CPUAdam step 后主要是 FP32 master `4N`、FP32 m `4N`、FP32 v `4N`，没有梯度或 BF16 分片字段。 |
| `ds_v0.16.9_cpu_adam.py:138–165` | `exp_avg/exp_avg_sq` 在 CPU 创建；默认 FP32；step 不再把 p 本体作为一个额外 state tensor。 |
| `torch_v2.6.0_optimizer.py:700–723` | `param_groups.params` 序列化为参数 ID；并非再序列化一次 master tensor。 |
| `ds_engine.py:3290–3292`，`ds_stage3.py:2929–2930` | 保存前调用 prologue，释放/partition 全部参数。 |
| `ds_parameter_offload.py:188–195`，`ds_partition_parameters.py:282–292` | partition 后 `param.data` 是 empty(0)，实际分片 `ds_tensor` 不作为普通 module 参数 state 自动另存。 |
| `ds_engine.py:3513–3545,3693–3696` | model-state 的 optimizer 字段在 ZeRO 路径为 None；model 文件主要是参数 placeholder、buffer、形状/共享关系、scheduler 与训练元数据。optimizer 文件保存 master/m/v。 |
| `ds_torch_checkpoint_engine.py:18–23` | 直接 `torch.save`，没有另存 BF16 分片的步骤。 |
| `ds_stage3.py:2736–2778` | 恢复 master/optimizer 后，由 master 复制回 16bit 分片。 |

因此正常全参、无 PP/TP/frozen/NVMe 副本的原生主载荷是 `12N≈75.392807 GiB`，每 rank 约 `6N≈37.696404 GiB`。不是 DP 每 rank 各一套完整 BF16 模型加全优化器，也不是固定 `14N`。Parameter Offload 改变驻留位置，通常不改变这套 native 持久化字段。

`14N` 仅表示 native `12N` 加额外一份 BF16 完整导出 `2N`；若保留 FP32 合并中间 `4N` 和 BF16 最终 `2N`，则 `18N`。既有 `CHECKPOINT_STORAGE_BUDGET_ERRATUM.json` 已将先前误沿用的 200 GiB 改成当前 100 GiB；不能再引用旧资源估算的 16N checkpoint 字段当当前格式实测。

## 2. 四轮、保留 5 与 root 结尾保存必须分开

作者 `finetune.py:83–94` 是按 epoch 验证/保存、`save_total_limit=5`、`load_best_model_at_end=True`；`105–106` 训练结束另调用 `save_state()` 和 root `save_model(output_dir=args.output_dir)`。四轮正常只有 4 个 `checkpoint-*`，不是自动生成 5 个。当前 HF4.31/gather=false root 保存又走 native DS checkpoint，处于 `checkpoint-*` 轮转之外，故按原结尾有第 5 套独立存量的风险。若未来真有 5 个 checkpoint-* 加 root，则是 6 套；本轮没有修改保留或选模策略。

**额外 I/O 发现：** HF `Trainer._save_checkpoint:2294–2298` 先调 `save_model`；后者在 `2753–2766` 因 gather=false 已 fallback 为 `engine.save_checkpoint`，然后 `_save_checkpoint` 又直接调一次。两次同一目录/默认 global-step tag，通常覆盖同名文件，不能把存量翻倍，但写入耗时和总写入量可能显著增加。未实际保存，故不报写入耗时。

所有表均只加**共享 base B 一次**；五 seed 的同一基础资产不必复制 5 份。各 seed 的训练 checkpoint 独立；下列不是当前 100 GiB 探针的授权范围。

| 情景 | 主载荷 bytes | GiB + H |
|---|---:|---:|
| 单 native 12N | 80,952,410,112 | 75.392807 |
| 单 native + BF16 14N | 94,444,478,464 | 87.958275 |
| 单 native + FP32 中间 + BF16 18N | 121,428,615,168 | 113.089211 |
| 单 native + 作者旧 tmp/final 各一份 FP32 20N | 134,920,683,520 | 125.654678 |
| 一 seed：4 个 epoch native + base | 337,286,517,109 | 314.122547 |
| 一 seed：上述 + root native | 418,238,927,221 | 389.515355 |
| 一 seed：5 retained checkpoint-* + root + base（额外情景） | 499,191,337,333 | 464.908162 |
| 一 seed：4 epoch + root + 所选一份 FP32/BF16 + base | 458,715,132,277 | 427.211758 |
| 五 seed：各 4 epoch native + 共享 base | 1,632,525,078,901 | 1520.407460 |
| 五 seed：各 4 epoch + root native + 共享 base | 2,037,287,129,461 | 1897.371495 |
| 五 seed：各 5 retained checkpoint-* + root + 共享 base（额外情景） | 2,442,049,180,021 | 2274.335530 |
| 五 seed：各 4 epoch/root + 所选一份 BF16 + 共享 base | 2,104,747,471,221 | 1960.198834 |
| 五 seed：各 4 epoch/root + 所选 FP32/BF16 + 共享 base | 2,239,668,154,741 | 2085.853512 |
| 一 seed：4 epoch 全按原旧转换保留 tmp/final FP32，另 root/base | 634,112,020,853 | 590.562840 |
| 五 seed：上述各留完整转换产物，base 共享 | 3,116,652,597,621 | 2902.608921 |

作者 `convert/convert.sh:5–18` 遍历 checkpoint，每套先写 `tmp-checkpoint-*/pytorch_model.bin`；`convert.py:15–16` 没有 `torch_dtype`，默认加载/另存仍为 FP32，不是 BF16 导出。清理行被注释，不能自行删掉产物。文件系统还需始终留 `max(30 GiB, 10% filesystem total)` 安全余量，并考虑创建新 checkpoint 后才轮转的临时保留量。上述正式总量与本轮新增量是不同口径。

## 3. RAM、导出与本轮不写大 checkpoint 的准确理由

原生保存的 optimizer.state_dict/master 通常引用现存 CPU tensor，源码没有 deepcopy 全套；不能直接把磁盘 12N 又计成额外 CPU RSS 12N。page cache、序列化缓冲、buffer 搬运仍会增加内存，未实测峰值。16bit consolidation（`ds_engine.py:3764–3809`）按 module 在 GPU gather，并在 rank0 保留 `.detach().cpu()`：约额外 `2N≈12.565468 GiB` CPU 权重持有量，另有逐层 GPU 瞬时聚合。CPU 参数 offload 还可能使 CPU 分片与完整导出并存。

**对旧静态报告一处描述的纠正（原件保留）：** `CHECKPOINT_STORAGE_STATIC_AUDIT.md` 第 2 节称 vendor 旧 `zero_to_fp32`“先收集所有 optimizer 文件/约 12N，然后才释放 m/v”。实际 `vendor/original/convert/zero_to_fp32.py:150–193` 在每个 rank 文件 `torch.load` 后，就在同一循环里删除非 master 字段并 `torch.cat` 该 rank 的 master。不能据此断言全部 rank 12N 同时驻留。两 rank、同形分片下，加载第 2 片时是先前约 2N master 加当前约 6N 文件主载荷；后续 `407–449` 又逐参数构造独立完整 FP32 state，4N master 分片与约 4N 重构权重并存。该约 8N 级主 tensor 活跃量只是源码推导，Python allocator、storage、缓存、buffers、序列化及加载顺序使它不是 RSS 上界；未测 RAM。此纠正不改变 native 12N 磁盘结论，不改变任何运行代码/研究设置。

DS0.16.9 新 `zero_to_fp32:148–155` 使用 mmap 并立即丢弃 m/v；`600–678` 支持 lazy 与分片输出，但其默认 sharding 需要 `huggingface_hub.split_torch_state_dict_into_shards`，旧依赖环境不能默认已支持。不能未经核对把新工具特性套在 vendor 旧工具上。

**单 native 的裸 75.393 GiB 并未自动违反 100 GiB**，不能写成“任何保存都必超限”。当前探针显式禁止 save/eval（`offload_probe.py:182–206,540–547`），目的是将有界时间/内存留给核心加载与更新。完整保存/恢复需先核本轮已有新增文件后的剩余总量、40 GiB CPU 安全余量、文件系统阈值和时限，并审查下节实际恢复入口；本代理未执行这一资源核准或保存。两 native 已 150.786 GiB、单 native+FP32+BF16 已 113.089 GiB、原旧转换单套已 125.655 GiB，均超本轮 cap，不能临时当附加测试运行。

## 4. 保存内容不等于完整 resume 或最佳 checkpoint 重载已通过

Epoch 保存链能静态覆盖：DS master/m/v、engine step/config、DS scheduler（若由 engine 管理）；HF `Trainer._save_checkpoint:2366–2393` 另保存 trainer_state 和每 rank Python/NumPy/CPU/CUDA RNG。当前 probe 的 scheduler 采用派生 clock 适配，未来正式入口需核 scheduler 对象/状态由哪一保存方管理，不能只说“DS 自带就全部恢复”。HF 训练 resume 委托 `tf_v4.31.0_deepspeed.py:371–387`，显式 `load_optimizer_states=True,load_lr_scheduler_states=True`，并有 `Trainer:2240–2277` 恢复 RNG。root 的 `save_model` 本身不走 epoch 的 RNG 保存段，不能默认等同完整 epoch resume 包。

本次独立确认的正式门槛：

1. **best-model discovery 兼容缺口。** HF4.31 `Trainer._load_best_model:2092–2107` 先要求独立 `pytorch_model.bin/model.safetensors/adapter` 存在才调用 DS reload；`2167–2177` 另只处理普通 `WEIGHTS_INDEX_NAME`，否则 warning。gather=false native-only 目录通常不含这些文件，不能将 `load_best_model_at_end=True` 声称为实际按最低 eval_loss 重载成功。最终 root 可能仍是末轮参数，不能假设已是最优轮。未来首选是保持原 eval_loss 选模定义，在单独授权的保存/恢复验收中实现明确 native DS 路径，并证明选择与参数身份，不改选模准则。
2. DS `engine.load_checkpoint:2953–2956` 警告 ZeRO3 save 后原 partitioned engine 不适合直接 load，需 pristine/reinitialized engine；`3169–3174` 普通 optimizer 恢复还要求相同 DP world size。不能承诺两卡 checkpoint 直接换 GPU 数无损恢复。
3. 保存/恢复、native→独立模型导出、RNG/optimizer/scheduler/数据跳过恢复轨迹、完整最佳轮 reload 的状态均为 **NOT TESTED**。静态已有字段和接口不等于完整兼容通过。没有写 checkpoint，也没有伪造“保存成功”退出码。

## 5. 正式 6912 updates / warmup 70 的条件性推导

复用既有真实 CPU loader 回执：SeqRec 131,837；Item2Index/Index2Item 各 9,922；Fusion 20,000；ItemSearch/Preference 各 24,772，总计 221,225。`utils.py:140–174` 将六 Dataset 合到一个 `ConcatDataset`、一个模型/Trainer 联合训练；不是六模型或六阶段训练。`full_validation_text_rows_constructed=49,544` 仅构造文本，未运行 validation loss。

在 unchanged corpus、micro1、world2、accum64、不 drop_last、普通 even shard/pad 条件下：

- 每 rank loader 长度 `ceil(221225/2)=110613`。
- HF4.31 `Trainer:1562–1577` 先 floor：`110613//64=1728` updates/epoch。
- 四轮 scheduler step 数 `ceil(4×1728)=6912`。
- `TrainingArguments.get_warmup_steps:1991–1996` 为 `ceil(.01×6912)=70`。
- 五 seed 共 `34560` optimizer updates；不是本轮执行授权。

本地纯算术已断言验证。仍是**条件性源码推导**，完整正式 loader、尾段/跨 epoch 累积、实际 scheduler tick、每样本曝光次数没有在本次正式四轮实测。原 8×micro8×accum2 与两卡候选的 batch 内归组/RNG/topology 不同；这里不声称原论文的唯一历史 step 数。

## 6. 正式时间预算只能给测量条件和公式

本代理读取的执行截点没有代表完整六任务、全上下文范围的真实 update 测量，因此不虚构小时数。令 `τ128,six` 是六任务长度分布代表性、同 GPU/CPU/NUMA/PCIe/软件、同 micro1×accum64 的稳态实际秒/update。一个 seed 的条件式为：

`T_seed = T_load + 6912 × τ128,six + Σ_epoch(V_epoch + C_epoch) + T_best_reload + C_root + T_export + T_test17`。

五 seed 顺序执行为每 seed 上式之和；使用共同速率的近似才是 `5×(...)`。需要分列验证、可能重复 native 写入、最终 root 保存、export 和 17 模板 beam test，不能将全部作业时长除以两次 update 再乘6912冒充训练吞吐。

若主控后来取得 SeqRec 技术 probe 的时间，只能代入 `τ128,seqrec` 作为同长度路径的**条件性替代估算**；response 长度、辅助任务长输出、最长 2048 输入、CPUAdam 工作量、PCIe 通信和 NUMA 的额外代价均未代表。技术 accum1 的一小步不能当 effective128 更新。首次加载、编译、证据采样开销应与稳定训练段分别记录；没有可靠组件分解时只报实际总用时，不减出虚构“纯训练”值。

## 7. 数据错配是正式复现门槛，不阻断训练期 SeqRec 技术链

没有重跑全量旧审计；直接读原审计、分类回执和原下游代码。

| 既有实测事实 | 本次依赖判断 |
|---|---|
| 24,772 用户中 train pair mismatch 12,233；test pair mismatch 7,473 | 不能称为已清理的 1:1 原历史资产，不能因 tokenize/mask 通过消除来源缺口。 |
| train mismatch 中 target 在当前 validation 3,608，test 1,623，training 7,002；1,623 的辅助历史含 validation item | 旗标可能重叠，不能乱相加。它们是与发布 split 的 ID/位置关系，不证明生成 API prompt 历史或论文实际处理原因。 |
| test mismatch 中 target 在 training 4,037，validation 3,436 | 该 test auxiliary 不是本轮六任务训练/SeqRec test 的直接输入，不能把 test 文件 mismatch 等同当前 GPU probe 读入测试监督。 |
| `data.py:596–608` ItemSearch 直接消费辅助 item/history，截最近20 | train mismatch 可直接改变该任务的输入/监督；即使截断输入偶有一致，也不能证明辅助文字生成来源一致。 |
| `data.py:707–718` Preference 用当前 inter[:-3] 配同 uid preference 文本 | 文本实际来源仍需作者同版材料，不可只凭当前训练期输入判纯净。 |
| 9,922 item / 9,897 unique SID，23碰撞组、25额外item；官方 callback position-only | 保留原 SID/生成协议；不改 trie、补 SID 位或任选一 item 伪造一一映射。 |

真实训练期 SeqRec 构造 `data.py:146–161` 先 `items[:-2]`，再取内部前缀目标，因此可不用 auxiliary 完成有界技术更新。当前 probe 有 prefix 源位置映射与 canonical SID 断言；其具体运行证据由主控出具，不由本报告静态推断 PASS。GPU 技术成功不能自动解封含 auxiliary 的正式六任务或五 seed。

首选待确认材料是作者绑定同版 `inter/user/SID` 的完整身份、处理脚本/有效参数与 split 说明，以及原辅助生成输入/修订来源。现有公开生成链与实包非空 test 还存在已记录的装配来源冲突；不能自行归因为“作者本来随机挑 target”。本轮没有重排交互、改写辅助目标、删除用户/任务或重生成标签。

## 8. 六任务 token-mean 的严格 1:1 差异保留

原 HF4.31 LLaMA causal CE 将 shift 后非 `-100` token 扁平计算 `mean`，collator 保留 response-only mask。令样本 `i` 的监督 token 数为 `n_i`、CE和为 `s_i`；布局 `(W,A,m)` 的有效梯度对应：

`L = (1/(W×A)) × Σ_(rank,accum) [(Σ_(i in microbatch) s_i) / (Σ_(i in microbatch) n_i)]`。

原 `8 ranks×micro8×accum2` 是 16 个八样本 token-mean 的均值；当前 `2×micro1×accum64` 是 128 个每样本 token-mean 的均值。回答长度可变时系数通常不同，可能改变六任务相对权重，**不只是浮点舍入差异**。global batch 同为128、相同 CE/mask，不足以宣称严格同一监督加权。相同 `n_i` 的特殊 batch 可代数等价，仍需真实记录 shift 后 count；不由“四 token SID+EOS”口头假设取代实测。

本次只复核现有推导，不改 normalization/loss。正式运行前应集中提交这一重要未决项：首选按原 micro8 参考分组保持 token-mean 语义，或另经明确授权定义归一化及梯度一致性验证；不能自行改成全局 token mean、任务等权或自选样本权重。六任务仍应是同模型联合训练，而非六段顺序微调。

## 9. 可交付结论与边界

C2 静态保存/恢复/导出审查已完成，发现并保留 native best-model discovery、same-engine reload、scheduler/RNG 和资源核准门槛；实际 save/resume/export 均 NOT TESTED。C3 的空间算术和 6912/70 条件推导已验证，正式耗时仍待代表性测量。C4 已以既有真实回执完成因果/依赖复核，正式 clean 1:1 结论继续 BLOCKED，真实训练期 SeqRec 技术探针独立可推进。所有原件与已有历史结论保持原字节，没有把本报告的静态推导当作新 GPU 通过证据。

本代理没有新增安装或环境配置变化。GitHub 发布、固定 RUN/DELIVERY_COMMIT、服务器映射、GPU 终态和 ZIP 身份由主控串行收口；本文件是待纳入该固定交付的独立证据，不自行宣称已发布。`GPT_UPLOAD_REQUIRED` 由主控依据最终 GitHub/外部回执覆盖程度统一判断。
