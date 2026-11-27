# LETTER Beauty 历史 SID 双向注意力

## 最后一轮正式前准备

正式计划只运行 B。旧 A 的已发布结果作为历史参考。A′ 配置留作以后独立讨论，正式链没有 A′ 阶段，也不以 A′ 作为 B 的执行条件。B 的 Normal 和四种 Shuffle 使用同一个按完整验证 loss 选中的 B adapter。研究目标是测量历史 SID 双向可见性对推荐和扰动敏感性的影响。Normal 与旧 A 的距离只作为结果和诊断线索。

### 最终计算路径

训练沿用已经验证的 BF16 efficient SDPA 选择性 mask。旧训练调用的 FastChat FA2 接受因果开关和 padding 边界，本任务需要表达仅历史 SID 之间的附加双向边，单卡部署与这个表达需求是两回事。新一轮的 `train.py` 修改限于正式入口门禁。共享 `attention.py` 增加 inference-only 分支，默认仍进入相同的 SDPA 实现。在固定上一轮训练快照与当前默认实现之间，CPU 小模型使用相同参数及 gradient checkpointing，logits、loss 和 21 个梯度张量逐位一致。训练的真实 7B、80 个 micro-batch、10 次更新、部分梯度恢复、原样本和提示流、四次验证及严格重载，继承上一轮原始实测。

推理采用首选方案 **原生 eager**。继续使用固定 Transformers 4.31 的 `LlamaAttention.forward`。`inference_attention.py` 调原生 mask builder，针对有效历史 SID 的交集将相应 bias 置零。普通提示、标点、目标和有效 padding 屏蔽沿用原计算。原生 eager 的无效 query 行处理保留，生成和有效位置均受有效 key 屏蔽。原生 Q/K/V、RoPE、FP32 softmax、context matmul、位置处理、20 beam 扩展与重排都沿用。公共包装只携带 history bits 并校验 KV cache 归属和前缀。attention 源文件 SHA-256 在安装时核对，GPU profile 在实际 B 上捕获 `aten::bmm` 和 `aten::_softmax`。

旧 A 同权重对照用真实样本 0、8、734。原始加载、BF16 基座与 embedding/head、FP32 LoRA、tokenizer、候选约束、原分数和稳定排序相同。带左右 padding 的 logits 逐位相同，60 个候选及分数、稳定顺序逐项相同。真实 B 样本 8 的 3,160 条附加逆序边仅连接历史 SID，32 层都观察到跨物品新边上的正权重，有效 query 的禁止边权重为零。改写未来目标后，各层历史状态和此前 logits 逐位相同。与显式 FP32 参考比较，首末层 context 相对 RMS 约 0.374%／0.263%。B cached 与 uncached 下一 token logits 相对 RMS 约 0.702%，top-1 相同。

推理统一回到旧 A 的原生 eager 数值算子，因此消除了上一轮 SDPA 推理造成的额外对照差异。原训练 FlashAttention 与 B 训练的 SDPA 舍入和归约差异、双卡转单卡的梯度累加顺序以及 CUDA dropout 轨迹仍存在，不能单独归因于 mask。CPU 全量 V2 输入流的冻结哈希和旧 A 的单条件历史结果继续沿用。预跑输出仅用于技术验收和成本估算。

### 正式链与授权

正式技术快照为 `81d4dc00f84426c77b48b7beb57b57903e97c20d`，路径在 `new-server:/data/bwwu/lc-rec/beauty-history-attention-final-20261127/releases/` 下。实际旧 A 与 B mask GPU 诊断快照是 `5a1a7297aa4078b46aa800f70ab15a649304cd67`，五条件预跑快照是 `d43787a43a64c175ff6edbf3d1820e5fd0c2baa2`。后两者对应的 attention 与测量实现已和正式候选逐文件核对，正式链代码由正式候选独立验证。

`formal.py` 使用既有 Beauty runner，请求绑定冻结源码 manifest、有效配置、固定物理 RTX 4090 UUID 和单独的授权证据。授权门禁要求两份真实文件，网页二审和用户启动授权。二审记录包含 `review_kind=WEB_SECOND_REVIEW`、实际 `status`、`plan_sha256`、`run_commit`、`config_sha256`、`train_sha256`、`mode`、`backend`、`snapshot_manifest_sha256`。单独的用户授权记录包括相同绑定字段、`formal_run_authorized`、`external_review_status` 和实际二审文件的 `web_review_path`、`web_review_sha256`。工具校验记录和文件身份，记录内容须由操作方按照真实审批留档。缺少授权时，正式提交、直接训练及直接全量生成均拒绝。

B 作业由同一服务器 runner 监督。训练 65,712 个八样本 micro-batches、四次正式验证、8,214 次 optimizer update，保存并核验原选模最优点，严格重载后发布 adapter。失败后先用原 ID 查询。只能在先前 attempt 已确认终止并完成进程清收后递增 attempt。训练从同一 lineage 内最近的可验证 epoch 完整 checkpoint 恢复 optimizer、scheduler、RNG、样本与提示流及部分梯度。没有完整 checkpoint 时使用独立训练目录，从初始状态重来。若 step 8214 已完整落盘但尚未导出，核对四次完整验证与最佳点后严格重载对应 PEFT adapter 并恢复输出。这条兜底路径只恢复导出。

五条件预测以 32 用户为正式提交单位，每个单位同时保存五种条件。重开后验证计划、后端、权重、输入和每块身份、完整覆盖及哈希，只复用连续的完整前缀。失败 attempt 的私有块保留，半行或部分条件不纳入聚合。最终单独原子发布五条件完整结果，每条件覆盖同轴 22,363 用户。完成入口重开时只校验已提交结果，无重复 Normal 生成。随后在禁用 CUDA 的阶段执行原 16 项指标和同用户 5,000 次配对 bootstrap，保留结果及完工清单。

获授权后从上文正式快照目录调用 `python -B -m lc_rec.beauty_history_attention.formal submit --plan ABS_PLAN --formal-authorization ABS_AUTH --formal-authorization-sha256 SHA256`。启动后和约 30 分钟时调用 `formal observe --plan ABS_PLAN --attempt 1 --output ABS_OBSERVATION`。观察实际 worker、step、loss、梯度 norm、LR、更新与 micro-batch 速率、GPU/空间、阶段、runner 尾日志和产物。评价尚未产出时标记“尚未产生”。已完成作业可用 `observe --verify-completion` 复核产物身份。一个尚未授权的固定计划以及实际否定门禁检查进入本轮证明材料。



### 真实五条件接续与性能

本轮 GPU 占用墙钟合计 641.86 秒，即 10.70 单卡分钟，含严格加载、诊断和重复请求的校验。GPU 使用之前重新核对了实际空闲设备，物理卡 1、逻辑 CUDA 0。未发现本机可直接查询的队列工具。冻结计划仍需按实验室排期取得使用授权。

20 名技术预跑用户由极短、极长、整输入恒等案例与原输入长度分层选出，分值未参与选择。B 使用同一个已严格加载的旧预跑 adapter，先提交含 8 名用户的五条件完整块并退出，重连后只计算剩余 12 名。首块六个文件的 SHA-256 在接续和重复提交后全部一致。最终 8／8／4 三块共 2,000 个候选，各条件用户轴无重复无漏行。重复运行实测只返回 `verified_complete`，未再次加载或生成。5 个整输入恒等案例的候选、分数、顺序逐项一致，S2、S3a、S3b、S4 分别为 3、1、1、0。实际 CPU 阶段输出了 1,600 个指标值以及 80 个配对差值和置信区间。独立非作者复算的最大绝对差为 0。另一项真实 CPU 子进程强制退出检查覆盖尚未提交的半行，确保读取只接受完整块。

采用冻结 22,363 个测试 prompt 长度分布重加权。技术样本覆盖 79–159 token，与冻结测试集最短、最长相同。将两个进程各自第一个用户作为预热，约 118 秒／次的严格加载分开计时。正式估算来源见 [资源原始计算](evidence/final-preparation/verification/resources.json)。

| 阶段 | 单卡或 CPU 耗时粗估 |
| --- | ---: |
| 65,712 个正式训练 micro-batch | 16.42 小时 |
| 四次全量验证及 checkpoint 写入 | 1.86 小时＋0.11 小时 |
| B Normal | 3.67 小时 |
| Shuffle2／3a／3b／4 | 3.58／3.58／3.58／3.60 小时 |
| CPU 指标、配对、I/O 与收尾 | 约 10–25 分钟，参考历史完整 CPU 运行和本轮有界流程 |

合计约 36.4 单卡小时的训练、验证、保存及五条件生成。计划按约 42–46 小时预留连续资源，仍保留每阶段及整链上限。checkpoint 单次最大实测 19.26 GB，三个可短时共存。预测块与最终输出可能暂时双存，按带详尽诊断字段的预跑粗估合计约 4.50 GB，另留日志、adapter、CPU 产物和恢复空间。正式计划启动时需 72 GiB 可用工作盘。

真实五条件 eager 的峰值 allocated 为 19,664,529,408 字节，reserved 为 24,744,296,448 字节，CUDA 可见总显存为 25,252,724,736 字节。分配量低于预留量，预留峰值距总额约 0.47 GiB，本轮样本全部完成。正式运行需要独占预留设备并观察显存余量。若发生显存中断，保留日志，只从校验通过的完整用户块恢复，必要时先排查资源。已提交的块不会因恢复重复生成。

### 本轮证据和启动状态

门禁在正式冻结快照、CUDA 隐藏的条件下做了 9 个真实负向检查，涵盖正式提交、worker、直接训练、试图用短程写正式输出和直接全量生成。正式 runner 目录、训练输出目录均未创建。第一次门禁收集器的预期报错文本过窄，修改匹配后在独立路径复验，前后记录均保存。25 个本地 CPU 测试通过，非作者也使用固定服务器解释器复核了核心 controller 和预测接续测试。正式全长作业及真实 step 8214 后故障恢复未执行。后者由独立控制流测试、真实严格 adapter 加载及已验证训练 checkpoint 规则覆盖。

实际未授权计划的 SHA-256 为 `8cf577a1d7d11fa1228996f634395544a24e3848117a482b1b48de5b85418e20`，服务器路径为 `/data/bwwu/lc-rec/beauty-history-attention-final-20261127/inputs/formal-gate-check-81d4dc00f844-verified/b-formal-plan.unsigned.json`。实际解释器是 `/data/bwwu/envs/letter-20261110/bin/python`。本轮未创建有效正式授权记录，也未安装依赖或改变持久环境配置。

[未授权计划与门禁证据](evidence/final-preparation/formal/) · [本轮真实原始结果](evidence/final-preparation/actual/) · [独立非作者签署](evidence/final-preparation/nonauthor/FINAL.md)

## 上一轮受审技术预跑记录

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
