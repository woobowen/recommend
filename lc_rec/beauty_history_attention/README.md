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
| 真实 7B 训练 | 4 个短 epoch，每个 20 micro-batches ×8；全局 ×64，共 10 更新，边界 2/5/7/10；每次验证 32 输入。若分段，第一 epoch 停在 2 更新、4 micro-batches 待续，从原 checkpoint 恢复。检查真实参数、梯度、checkpoint、验证与选优 | 1,800 秒总预算 |
| 真实 7B 回归 | 旧 A 权重，一条 20 物品真 prompt，原 eager／新 causal 的 logits、旧温度和位移 CE、两端 LoRA 梯度与 20 beam；双向 prefill/cache、未来目标变化、kernel 采样 | 720 秒 |
| 真实 7B 五条件 | 短/长、整条 S2 和 S3a/S3b 恒等行，固定 4 用户 ×5 条件 ×20 beam。随后复算 16 项指标，范围只覆盖预跑 | GPU 720 秒，CPU 300 秒 |

单卡 GPU 总墙钟上限为 3,420 秒，单设备 24 GiB。训练按实际 observed phase 估计正式成本。任何 OOM、NaN、身份冲突、设备占用或超时由该阶段返回失败，保留原日志和状态。正式 4 epoch、全量五条件以及正式 A′ 在本轮均不启动。

## 后续正式入口

训练使用 `python -m lc_rec.beauty_history_attention.train`，world size 为 1，输入当前快照的 LETTER vendor、固定 Beauty 数据和基座，选择 `config_bidirectional.json` 或 `config_causal.json`。正式训练需要固定代码/config/mode 的用户授权与网页二次审核材料，详见入口参数。八个连续的八样本 micro-batch 对应旧基线四个双 rank 更新片段。单卡保留各虚拟 rank Python 提示流、独立采样 RNG、原 NumPy 验证抽取以及跨 epoch 累积；最低验证 loss 选择模型，同分保留较早者。

用 `evaluate.py generate` 对新 adapter 执行五条件，`evaluate.py metrics` 从相同预测回读计算。模型和数据条件及分行范围由校验清单和快照保存。全量分析需后续明确启动。 
