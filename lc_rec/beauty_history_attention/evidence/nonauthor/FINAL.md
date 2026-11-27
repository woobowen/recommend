# LETTER Beauty 历史 SID 双向注意力非作者审核

审核者为子代理 `/root/review`。审核者未编辑训练、注意力或评价实现，独立 CPU 检查使用禁用 CUDA 的固定 LETTER 解释器。直接读取原 Git 对象、服务器冻结资料、实际部署文件和 runner 记录。下列已检查范围按证据分别判定。

**非作者审核结论为技术预跑 PASS，待网页 GPT 审核和后续正式授权。** 本轮运行终态、最早旧 baseline 和新受审源码已交叉核验。正式训练、全量评价、跨设备数值等价及研究效应需要后续证据。旧 A 作为历史对照。训练和推理 causal 回归均显示后端产生了可测差异，归因于 mask 的主张需要匹配 A′。

## 源码和固定材料

- 原训练为 `6e1b211d9c37f358c43b1540b40b17f6703150d2`。直接核对原 LETTER 数据、17 个模板、collator、LETTER 温度与位移损失、FastChat FA2 训练适配、BF16 ZeRO-2、训练日志及 best selection。旧 A 选 step 6160，最低验证 loss 1.6796015501022339。历史推理实际使用 BF16 LLaMA 基座、活跃 embedding/head 和 FP32 LoRA，原推理 attention 为 HF eager。
- 原 V2 运行 `1182dfd914c20bd3dc11786587eae6879fba6b9d`。独立按 Prompt 0 重新分词全体 22,363 人，复核 159,966 历史组。重新实现排列 SHA-256 随机映射，逐行核对原哈希、mask、所有模式排列编号、变更哈希、canonical target 及完整输入流 SHA-256。整条恒等行各为 S2 640、S3a 6、S3b 6、S4 0。见 `frozen_v2_input_review.py` 及原始输出。
- 代码 `b9dfa8f8adad6ee89c15bda9d978c5465ac7e772` 的 13 个实验文件逐个对远端实际部署核哈希，全部一致。Python AST、JSON 解析通过。见 `b9dfa8f8-snapshot-check.json`。
- 已读该版本注意力、训练、评价、身份绑定、独立阶段入口与有效配置。新 B 在每个实际 attention 层显式应用包含有效 key 和历史双向边的 mask。位置与 RoPE 仍由原 LLaMA 处理。Cache 带模型归属和 mask 前缀，beam 重排时同步重排。A′ 同样使用 CUDA efficient SDPA，按开关省略新增历史边。训练、推理路径的历史 kernel 分别为 FastChat FA2 和 HF eager，模型区分需要在结论中保留。

## 执行证据

- 分别独立核对四个原训练 epoch 的双 rank 数据行。实测新串行 sampler 4 × 131,424 个位置全部等于交错排列的旧 archive，原子批长 8。每 epoch 131,413 个 distinct 样本、11 个各重复一次。完整计划 525,696 个位置，边界更新 2,053、4,107、6,160、8,214，奇数 epoch 边界保留 32 样本梯度。见 `historical_sampler_replay.py` 及原始输出。
- 每个历史 rank 单独从 Python seed 42 原生动态 randint 抽模板，整个轨迹中有原始哈希证据的 312 个 batch 均完全复现 input_ids 和 labels 的哈希。无记录位置的模板顺序据实际源码和已核对采样推导。见 `historical_prompt_replay.py` 及原始输出。
- 真实 7B 的 `train-r1` 中首 20 个训练 micro-batch 与旧 A 对应的交错 rank 批逐项匹配，input_ids、labels 哈希和 shape 相等，H 长度 4–80。见 `real_train_prefix_review.py` 及原始输出。
- 真实训练 r1 与 resume-r1 终态均 `SUCCEEDED`、退出 0、清收 verified。直接遍历完整 80 条 micro loss、四个短 epoch 和 10 次更新。每条 DeepSpeed 更新边界与 scheduler 推进符合 8-batch 累积，10 个梯度 norm 有限。停点依据同一 checkpoint 显式恢复 micro20、step2 的未提交累积梯度和 optimizer/scheduler，实际 exact 回执在 `events.rank0.jsonl`。验证点为 2、5、7、10，按四个原始 eval loss 重算最低点为 step10、8.086517333984375。导出 adapter 与选中 checkpoint 的活动 450 个权重一致，SHA-256 为 `a9ce5337e30d0ffe6917755455c18f6544ad9a73c4a29cf6e656e6872ccb3665`。全部 224 个 LoRA A、224 个 LoRA B、整表 embedding 和整表 head 均实际变化，冻结张量及梯度保护通过。resume 峰值 allocated 14,777,927,168 字节、reserved 20,206,059,520 字节。见 `independent_training_terminal.json`。
- 独立三层 tiny LLaMA CPU probe 在 pinned Torch 2.6.0+cu124、HF 4.31.0、PEFT 0.5.0 上执行。两个模式逐对 mask 和单 query slice 均等于直接定义。改未来目标时此前 logits 和每一层历史 hidden states 逐位相同，改跨物品较晚 H 会改变较早 H 输出。含左/右 padding 和全屏蔽时输出有限。旧 causal eager vs 新 causal 有效 logits 最大绝对误差 0。cache/uncached 最大绝对误差 1.4901161193847656e-7，两种模式均为同值。两样本各 20 beam 的生成序列一致，分数最大绝对误差 1.1920928955078125e-7。并存两份不同 mask 的 checkpoint 重算梯度与 eager 逐位相同。外来 causal cache 正确拒绝。见 `attention_cpu_probe.py`、启动脚本和 stdout。
- 直接审读 r1 CPU/CUDA 小模型 JSON。实测 CUDA BF16 causal 与 native eager 有效 logits 最大绝对误差 0.00341796875，cache 最大 0.00250244140625，重算梯度差 0。r1 CUDA 检查进程退出 0 且数值文件完整，runner 发现额外子进程并终止，标记 `WORKERS_TERMINATED`，最终清收 verified。已从 pinned Torch 源码验证 `TORCHINDUCTOR_COMPILE_THREADS=1` 会绕过编译进程池预热。

## 真实 7B 生成和评价

- 对 `a26a9e9698fb5c66f04e93dd062f8beef8bade8b` 读取原始四个样本和五个条件的全部 400 个候选。输入 ID、H、canonical target 和 V2 hash 与原冻表一致。逐条件验证实际首次 prefill 的 H、位置、mask、20 beam 及空 cache，并检查每条记录四次单-query decode 的位置与 cache 长度。逐候选复核所有位置的允许码集合、EOS、解析与稳定排序。三个整条恒等 pair 分别属于 Shuffle2、Shuffle3a、Shuffle3b，全部候选、分数与顺序均等于 B Normal。没有 S4 整条恒等输入。
- 从原预测和 SID 目录独立重算 320 个逐用户指标，含 16 种 SID/CCE 定义，绝对误差上限 1e-14。读取严格加载的完整 450 权重，其中 448 个 LoRA 为 FP32，embedding/head 为 BF16。实际 B 生成和指标两作业均 `SUCCEEDED`、退出 0、清收 verified。GPU 峰值 allocated 19,663,988,224 字节、reserved 21,437,087,744 字节。见 `independent_generation_terminal.json` 与 `generation_timing_review.json`。
- 从 76 个实际 epoch 内 micro-batch 时间差所得中位 0.9174 秒，4 × 32 个有界验证输入分别达到 24.114–30.135 rows/s。按 65,712 个正式训练 micro-batch、四次完整验证、四次保存的线性外推约 18.7 小时。四条长度及扰动身份选定的测试样本外推五条件生成约 19.4 小时。二者均为短样本推算，受分布和长时资源共享影响。B 或匹配的 A′ 粗估 38 GPU 小时，另留加载、回读和换卡余量。

## 真实 7B causal 回归

- 原 A adapter 单行实际推理，旧 eager 与同权重新 causal 在 20 个完整后缀中共用 19 个，只有 7 个处于相同序位，共用候选最大移位 4。比较的是实际生成 token 和原始 sequence score。有效 logits 最大绝对差 .875、RMS .09477616。同样五个监督预测位最大差 .46875、RMS .08105836，top-1 相同。复算温度 .8 的 CE 为 1.5834109783 与 1.5849399567。首末 LoRA q_proj B 梯度相对 RMS 约 5.40% 与 3.16%。保留原始误差与候选记录，报告明确允许 BF16 kernel 漂移。
- 切到双向模式诊断，变动未来目标后的此前 logits 和各层 H hidden states 逐位相同。首末层真实 Q/K/V 的 fused 与显式 FP32 参考相对 RMS 约 .148%/.110%，最大绝对差 .001953125/.015625。B 单 query cache 与重新 prefill 的 logits 最大差 .09375、相对 RMS .321%，top-1 相同。CUDA profiler 直接捕获 `aten::_scaled_dot_product_efficient_attention` 和 `fmha_cutlassF_bf16_aligned_64x128_rf_sm80`。见 `inference_regression_independent.json` 和当次原始 `regression.json`。
- 独立比对首次 8-bit 训练回归尝试与 B 正式配置预跑首次 fresh load 后的 BF16 活跃权重哈希，450 个全部相同。真实第一批 input_ids、labels 和 H 的哈希也相同。已有证据支持新实现可复现初始化与采样前缀。旧 A 历史初始化没有对应逐张量记录。见 `initialization_hash_review.py` 及原始输出。

## 审核发现

- 真实 7B 训练 causal 回归使用真实首批 8 × 165 序列、941 个有效 token、40 个监督位置。同批 input IDs、labels、H 的哈希与主预跑第一批完全相同。原 FastChat FA2 确认 64 次 varlen 前向或重算、32 次配对的 native/extension deterministic backward。新 causal 的真实 Torch profiler 有 64 次 efficient 前向或重算、32 次 efficient backward。实际 Q/K/V 两路都是 BF16，同一参数对象、相同随机状态，复用原 LETTER 温度和 label shift。两次诊断首次加载 450 项活跃权重哈希都与真实 B 初始哈希逐项相等。
- 旧 FA2 与新 SDPA 的 CE 为 12.88660717010498 与 12.908281326293945，绝对差 0.021674156188964844。有效 logits 相对 L2 5.52%、cosine .99848，监督预测位相对 L2 5.21%，top-1 31/40 个相同。完整参数梯度相对 L2 为 embedding 17.55%、LoRA B 24.35%、lm_head 6.81%，相应 cosine 为 .98629、.97122、.99768。fresh LoRA B 为 0，两路 LoRA A 梯度精确为 0。实验因此能确认因果可见性和训练通路，浮点结果及离散结果存在实测差异。诊断未作 optimizer update 或 ZeRO-2 跨 rank 归约，两后端差异在量化模型和混合精度条件下观测，现有测试没有单独辨识误差如何被各层运算放大。见 `independent_regressions_terminal.json`、`terminal_initialization_replay.json` 及原始 `training-regression.json`。
- 独立发现 `d40da2438f4cd9ae6892e833ee5fffd5191d4efd` 中 context pre-hook 用了位置参数，而固定 HF4.31 的实际 decoder 按关键字传入。修复为 `kwargs["hidden_states"]`，纳入 `a26a9e9698fb5c66f04e93dd062f8beef8bade8b`。两版各 14 份实验文件独立对远端部署逐个核哈希。
- 在实际 a26 服务器源码上 CPU 执行修复版 context hook，首末 tiny LLaMA 层与显式 FP32 计算最大差 0。正式 b9 起参考路径对全屏蔽 query 在 softmax 前写有限值，之后将概率清零。独立 backward 证实 Q/K/V 梯度有限，所有无效行和未用 key 梯度为 0。见 `reference_hook_review.py` 及原始输出。
- 训练回归 r1 原始失败为二维 FastChat mask 进入原 HF eager 模块。已从固定 Accelerate 源确认加载设备映射时其工具在实例 forward 中捕获旧绑定方法。修正版 `3ce7c70264454d16f4e9ffc8174ffac3b5a888e0` 在模型构造前安装旧补丁，与历史入口一致。旧路径计算后恢复类，再以相同参数对象和 RNG 安装新实例方法。改动只影响诊断。实际 r2 退出 0、标记 `SUCCEEDED` 且清收完整。r1 退出 1 和真实堆栈原样保留。3ce 文件 14/14 与远端部署一致，见 `3ce7c702-snapshot-check.json`。推理诊断也退出 0 且清收完整。见 `independent_regression_jobs.json`。

## 固定原件位置和签署

- 真实训练原件包含 `train-r1/training/events.rank0.jsonl`、`rank0.json` 和 `best_selection.json`，位于 `/data/bwwu/lc-rec/beauty-history-attention-20261127/`。对应 b9 实际执行。
- 同一根目录下 `generate-r1/predictions/`、`metrics-r1/metrics/` 和 `regression-r1/regression/regression.json` 对应 a26。训练回归失败和修正原件分别为 `training-regression-r1/training-regression.json` 对应 a26、`training-regression-r2/training-regression.json` 对应 3ce。runner 原状态、清收、请求及输出在 `/home/bwwu/grid-audits/sid-extensions-p0/jobs/letter-beauty-v2-history-attn-*` 中，按同名阶段对应。
- 判断范围限于实际被审源码、历史原件、CPU 复算及已终态的有界真实模型预跑。正式 8214 更新的采样与边界通过全量 CPU 重放验证，正式训练和全用户 GPU 评价尚待另行授权。记录到的 short preflight 恢复 exact 来自实际 step2 状态恢复和随后完整轨迹，跨机故障或其他任意断点重现尚无本轮实测。

签署为非作者审核者 `/root/review`，2026-11-27（America/New_York）。
