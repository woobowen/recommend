# LETTER Beauty 历史 SID 双向注意力

本实验沿用 LETTER Beauty SID 和原 LLaMA-7B revision，检验正常推荐与历史物品内 Shuffle V2 的敏感性。旧 A 为已结束的双卡 causal 训练与五条件结果。B 使用同一预训练基座，单卡执行，在每层 causal self-attention 上增加任意两个有效历史 SID token 之间的边。匹配 causal 配置 A′ 使用同一套单卡代码与 CUDA efficient SDPA。B 与 A′ 共享词表、原模型结构、训练目标和评价协议。

## 实施与依据

- 原训练 commit `6e1b211d9c37f358c43b1540b40b17f6703150d2`，原五条件 Shuffle V2 commit `1182dfd914c20bd3dc11786587eae6879fba6b9d`。训练入口以当前分支已验收的单卡 Trainer/ZeRO-2 执行骨架为工程依据，LETTER vendor 与原训练对应。新配置使用历史 seqrec 数据和初始化条件。
- 数据连接使用冻结的 Beauty index、item 和交互文件。基础模型为 `huggyllama/llama-7b@4782ad278652c7c71b72204d462d6d01eaaf7549`。校验和及运行路径见 `identity.py` 和历史 V2 配置。
- `inputs.py` 先调用原 Collator，随后从 17 个原模板中的精确历史插槽取出物品 SID，核对实际保留的 ID 顺序，再生成与原 input IDs 对齐的 H。目标通过 response 边界单独解析。实际训练为右 padding，测试为左 padding。
- `attention.py` 保留原 RoPE、位置生成、Q/K/V/O 和参数模块。保留 key 的有效性，按物理序列索引联合 causal 三角与 H 外积。各层与 checkpoint 重算接收同一个显式加性 mask。全屏蔽的无效 query 暂开一个数值哨兵并将 attention 输出归零。生成的原二值 mask 用于原生 position IDs。H 始于同一次 prefill，随后扩展新增位置为 false。cache 标记所属模型与 prefill mask，随 beam 重排。
- `evaluate.py` 使用原 Prompt 0、batch 1、20 beam、20 返回、限长 10、code-position callback、原分数与稳定排序。保存五条件独立记录，使用原完整 SID 和 CCE 指标以及 V2 双向条件相对 Normal 的配对统计。正式任务如需对两模型的降幅再做差，应作为独立分析，保留一致的用户轴。
- 历史训练 8-bit 基座、BF16 ZeRO-2 与原兼容补丁，原训练 FlashAttention 2 使用 deterministic backward 适配。新训练保留 dropout、完整活跃 embedding/head 和 224 个 LoRA 注入，attention 计算使用同一 CUDA efficient SDPA 后端处理两个模式。原生成使用 eager，加载基座与 embedding/head 为 BF16，LoRA 为 FP32，新生成沿用加载 dtype。
- 按纯 mask 效应提出研究主张时，应加入 A′。新旧运行设备拓扑、浮点累加及 CUDA dropout 轨迹有差异，且历史训练与推理 attention kernel 不同。旧 A 可作历史锚点，B/A′ 用相同单卡运行、样本及提示计划和匹配后端估计 mask 效应。A′ 预算为与 B 相同的 8,214 updates 与一次五条件评估。

## 本轮预跑预算

通过已有 Beauty runner 在一张 RTX 4090 上串行执行。先核 UUID、实时计算进程、可取得的排队信息、磁盘及可用显存，固定部署和专用输出。入口是 `preflight.py`，保存每阶段实际命令和 runner 终态。脚本仅包含技术预跑。

| 阶段 | 覆盖 | 墙钟上限 |
| --- | --- | ---: |
| CPU 和 CUDA 小模型检查 | 17 训练模板、32 验证 prompt、4 测试行、截断、左右 padding、三层模型的可见性、全屏蔽、跨物品与标签泄漏、checkpoint 重算、cache 及双样本 20 beam | 各 180 秒 |
| 真实 7B 训练 | 4 个短 epoch，每个 20 micro-batches ×8。全局 ×64，共 10 更新，边界 2/5/7/10。每次验证 32 输入。若分段，第一 epoch 停在 2 更新、4 micro-batches 待续，从原 checkpoint 恢复。检查真实参数、梯度、checkpoint、验证与选优 | 1,800 秒总预算 |
| 真实 7B 回归 | 旧 A 权重，一条 20 物品真 prompt，原 eager／新 causal 的 logits、旧温度和位移 CE、两端 LoRA 梯度与 20 beam。双向 prefill/cache、未来目标变化、kernel 采样 | 720 秒 |
| 真实 7B 五条件 | 短/长、整条 S2 和 S3a/S3b 恒等行，固定 4 用户 ×5 条件 ×20 beam。随后复算 16 项指标，范围只覆盖预跑 | GPU 720 秒，CPU 300 秒 |

另用真实 8-bit/BF16 LETTER 历史首批对比旧 FastChat FA2 deterministic backward 与新 causal 的训练 logits、loss 和梯度，独立阶段上限 720 秒。单卡 GPU 总墙钟上限为 4,140 秒，单设备 24 GiB。训练按实际 observed phase 估计正式成本。任何 OOM、NaN、身份冲突、设备占用或超时由该阶段返回失败，保留原日志和状态。正式 4 epoch、全量五条件以及正式 A′ 在本轮均不启动。

## 后续正式入口

训练使用 `python -m lc_rec.beauty_history_attention.train`，world size 为 1，输入当前快照的 LETTER vendor、固定 Beauty 数据和基座，选择 `config_bidirectional.json` 或 `config_causal.json`。正式训练需要固定代码/config/mode 的用户授权与网页二次审核材料，详见入口参数。八个连续的八样本 micro-batch 对应旧基线四个双 rank 更新片段。单卡保留各虚拟 rank Python 提示流、独立采样 RNG、原 NumPy 验证抽取以及跨 epoch 累积。最低验证 loss 选择模型，同分保留较早者。

用 `evaluate.py generate` 对新 adapter 执行五条件，`evaluate.py metrics` 从相同预测回读计算。模型和数据条件及分行范围由校验清单和快照保存。全量分析需后续明确启动。

正式预算按 131,413 不同训练样本、每 epoch 11 条历史采样器补齐位置计算。65,712 个八样本 micro-batches 分配到四个 epoch，累积组跨 epoch 延续，optimizer/scheduler 边界为 2,053、4,107、6,160、8,214。正式 cosine warmup 为 83 updates。原两个 rank 各有独立 Python 提示 RNG，当前输入序列按 rank0 的一组八条、rank1 的一组八条交错执行。独立 CPU 回放逐条核对全量 525,696 个样本位置，并复现历史日志保留的 312 条带标签 token 指纹。验证输入的最后一个八样本 loss 使用原六条真实样本和两条补齐样本计算，平均时计入六条真实样本，延续历史跨 rank gather 的权重。历史全部 CUDA 状态与 dropout 数值轨迹仍无法由这些记录还原。

后续启动时将证据、源文件、配置和模式绑定到同一正式授权记录。训练入口校验的字段为 `formal_run_authorized`、`external_review_status`、`run_commit`、`config_sha256`、`train_sha256`、`mode`。用 `--formal-authorization` 和 `--formal-authorization-sha256` 传入。预跑 checkpoint 拥有单独 scope，正式评价只接受正式 adapter。生成成功后将 `--predictions` 指向全五条件输出，使用现有无 GPU 配置运行 `evaluate metrics --purpose formal`。配对统计使用同一用户顺序、包含恒等的 V2 置换和原 5,000 次共享 PCG64 用户 bootstrap。

GPU 预跑使用物理 1 的 UUID 映射到逻辑 CUDA 0。已发现的 CPU、GPU 子进程和资源轨迹保存在实际作业回执。启动正式任务前需复核实验室排期和设备占用，以实际同类运行的显存、速度及 checkpoint 文件体积留出空间。

## 本轮 7B 训练证据

训练执行提交为 `b9dfa8f8adad6ee89c15bda9d978c5465ac7e772`。两个独立作业 `letter-beauty-v2-history-attn-train-r1` 与 `letter-beauty-v2-history-attn-train-resume-r1` 均退出 0，完成 80 个真实八样本 micro-batches 和 10 次 optimizer update。断点在 micro 20、global step 2，待累积 4 个 micro-batches，重载后部分梯度、scheduler 和 loader 状态通过精确验收。预跑验证范围固定为 32 条真实输入，四个边界的 loss 为 12.78091049194336、8.512457847595215、8.239721298217773、8.086517333984375。step 10 以实际最低 loss 选中，严格重载后导出的 adapter 与该 checkpoint 逐 tensor 一致。

原 450 个训练张量、290,037,760 个参数均进入 optimizer。最终 LoRA A、LoRA B、完整 embedding 及完整 head 都出现有效更新，冻结张量字节不变。训练实际 Q/K/V 为 BF16，使用 CUDA efficient SDPA，checkpoint 重算传入每个 batch 自己的 mask。两段整体 allocator 峰值为 14,777,927,168 字节 allocated 和 20,206,059,520 字节 reserved。实际单段训练完整保存文件约 18.7–19.3 GB，保存及 SHA-256 逐次校验占用显著时间。

58 个邻接、无 optimizer 边界的 micro-batch 计时的中位数为 0.900 秒。按 65,712 个正式 micro-batches、四次全量验证和 checkpoint 成本外推，训练原始估算约 18.4 单卡小时，仍应给恢复、优化器更新和共享设备额外余量。正式训练输出空间应覆盖约 57.9 GB 的三个短时共存 checkpoint 及 adapter、日志、候选输出所需空间。预算依据是已测技术运行，实际资源安排在正式启动前复核。

## 五条件生成与因果对照

生成、评价和推理回归的执行提交为 `a26a9e9698fb5c66f04e93dd062f8beef8bade8b`。正式数据的 CPU 输入重放覆盖 22,363 行和 159,966 个历史组，四种扰动的整条输入流 SHA-256 与原 V2 一致。GPU 预跑用户行是 0、8、47、734。选择依据是 prompt 长短及整条恒等条件，选择过程无分数输入。真实 B adapter 严格重载，加载后 448 个 LoRA 张量为 FP32，活跃 embedding 和 head 为 BF16。20 次 prefill 合计运行 3,200 个层调用，含 80 次有 cache 的单 query decode。三条整输入恒等样例的候选、排序和原分数逐项一致。S4 在 22,363 行的历史输入中没有整条恒等样例。CPU 使用原 CCE 脚本验证五组 16 项指标。

GPU 真实生成峰值 allocated 为 19,663,988,224 字节，reserved 为 21,437,087,744 字节。20 个用户–条件的实测生成时间外推约 19.4 单卡小时，样本有意覆盖长短和恒等边界，正式耗时还受分布及 I/O 影响。B 或 A′ 每个约 38 单卡小时的训练加生成计算预算需要再留运行余量，两者的顺序按网页审核后的授权与 GPU 排期确定。

旧 A 的同一权重对照使用原 BF16 eager 推理与新 causal efficient SDPA。在一条 20 物品样本上，全 logits 最大绝对差 0.875、RMS 0.09478。受监督位置最大绝对差 0.46875，shift 后带 0.8 温度的 CE 为 1.5834109783172607 与 1.584939956665039。首尾两层 q_proj LoRA-B 梯度 RMS 差为 8.54e-6、1.67e-5。20 个候选有 19 个重合，顺序有变化。原生 eager 的多阶段 BF16 舍入与 fused reduce 路径的差异可传递至近邻 beam 分数，因此严格 mask-only 比较使用同后端的 A′。

B 的双向诊断使用真实输入，以第一层和末层 rotated Q/K/V 计算显式 FP32 参考，fused context 相对 RMS 分别为 0.148% 和 0.110%，预跑判据为 1.5%。相同下一 token 的 cached/uncached logits 相对 RMS 为 0.321%，判据为 3%，argmax 一致。改写未来目标 token，预测该 token 的前一位置及所有更早 logits 完全一致，各层历史 hidden state 完全一致。GPU profiler 记录 `aten::_scaled_dot_product_efficient_attention` 及 `fmha_cutlassF_bf16_aligned_64x128_rf_sm80`，两模式固定同一 SDPA backend。

补充证据见 [机器汇总](evidence/summary.json)、[实际作业回执](evidence/runner/)、[原始预跑记录](evidence/stage/)、[非作者审核](evidence/nonauthor/) 与 [旧 LETTER Beauty 五条件历史结果](../shuffle/results/v2-a0-beauty-letter/final-r0/README.md)。本轮输出均为实验前技术证据，网页 GPT 二次审核及实验启动授权仍待完成。

## 修正与留档

1. 初次 CPU 截断断言对左截断直接截取完整结果尾部，遗漏原 tokenizer 截断后再添加的 BOS。更正检查索引后重新运行，用原 collator 字节一致性核对左右 padding 与截断。
2. 小型 CUDA 数值输出通过，首次 runner 收尾发现 TorchInductor 工作进程。后续请求固定 `TORCHINDUCTOR_COMPILE_THREADS=1`，小型 CUDA 复验 exit 0，清收完成。真实 context 诊断中的 hook 调用也按固定 HF 4.31 的关键字参数修复并在实际 7B 上复验。
3. 旧训练路径诊断首次把 FastChat 类补丁装在量化模型加载之后，Accelerate 已把 native eager forward 捕获到实例。诊断按历史加载时序先装 FastChat，完成旧路径后恢复类属性，再在同一模型对象上装新 causal。该修正只涉及诊断，实际 B 训练和五条件生成维持各自固定提交。首次失败保留原请求、日志、状态和报错。

历史双 rank 舍入和运行时随机状态只可按保留的样本和提示证据追溯。A′ 正常执行时和 B 的配置只差 `mode`。复用相同框架版本、同一单卡、相同 batch/随机输入与调度、相同权重初始化、同一 efficient SDPA 调用和 mask 表示。相同新初始化由 450 个逐张量 SHA-256 的本轮实测证据核对。需要严格归因时，先获得 B 和 A′ 的正式授权，并预先确定对比口径。

## 旧训练后端的同权重诊断

修正后的训练诊断执行提交为 `3ce7c70264454d16f4e9ffc8174ffac3b5a888e0`。同一 7B 量化模型、同一组已初始化权重、同一组历史 rank0 首批训练输入和相同 dropout 起始 RNG，依次运行两种 causal 路径。诊断保留 8-bit 基座、BF16 active 参数和原 LETTER loss。梯度在 ZeRO 归约前读取，端到端 ZeRO-2 更新另由真实 B 预跑验证。

旧 FastChat 记录 64 次 FA2 varlen forward，32 次实际 deterministic backward。新 causal 记录 64 次 efficient SDPA forward，32 次 efficient backward。所有有效位置 logits 最大差 6.734375、RMS 0.31655。40 个监督位置有 31 个 top1 相同，原／新 CE 为 12.88660717010498／12.908281326293945。Embedding、LoRA-B 和 head 梯度的相对 L2 差分别约 17.55%、24.35%、6.81%。初始化 LoRA-B 为零，两条路径的 LoRA-A 梯度也为零。完整 450 张量逐项数值保留在诊断记录。

差异发生在两套不同的实际 causal 数值路径。旧 FA2 先变长去 padding，使用原 FA2 deterministic backward。新路径按同一可见性在带显式 padding mask 的 BF16 张量上做 efficient SDPA。相同源权重、字节一致的样本、相同 loss 和 RNG 起点排除了这些设置发生变化。诊断未逐层拆分 fused 归约舍入、8-bit 投影对数值的传播及执行顺序效应，也未证明两个后端逐点数值等价。这些实测差异与旧推理候选差异共同决定 A′ 对严格归因的必要性。
