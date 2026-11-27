# LETTER Beauty 历史 SID 双向注意力 B 独立非作者审核

**PASS。通过固定源码、真实有界预跑、恢复与未授权启动保护审核。** 审核者 `/root/review` 独立读取原件、Git 对象、服务器源码和原始作业产物，并执行独立 CPU 检查。审核期间未修改受审实现，未提交 GPU 作业。

## 源码与实际运行绑定

- 审核了正式候选 `81d4dc00f84426c77b48b7beb57b57903e97c20d`。服务器固定快照位于 `/data/bwwu/lc-rec/beauty-history-attention-final-20261127/releases/81d4dc00f84426c77b48b7beb57b57903e97c20d`。复核清单 176 个文件与 Git 内容逐项相等，清单 SHA-256 为 `e0c88d8225f35786e93774a6af92769919a4d4a73dadf8827601969270b1432b`。实测生成使用 `d43787a43a64c175ff6edbf3d1820e5fd0c2baa2`，原生 eager mask、SDPA 共用适配、生成入口及预测事务代码与正式候选字节一致。两个真实 7B 对照探针使用 `5a1a7297aa4078b46aa800f70ab15a649304cd67`，原生 eager mask、共用适配及探针源码与正式候选字节一致。固定历史训练和 V2 源码分别为 `6e1b211d9c37f358c43b1540b40b17f6703150d2` 与 `1182dfd914c20bd3dc11786587eae6879fba6b9d`。见 `fixed-source-audit.json`。
- 检查两个注意力路径。训练默认仍为此前实测的 BF16 8-bit efficient SDPA。在固定依赖上比较改动前后 tiny 模型重算的 21 个参数梯度、logits 与 loss，作者原始配套结果均逐值相等。训练和恢复的真实短跑依据来自此前 B 预跑。本轮新增推理保留 Transformers 4.31 原生 `LlamaAttention.forward`，只在原 decoder additive mask 中将两个位置均属于 H 的已禁止项设为可见。原始 LLaMA 实现 SHA-256 为 `2abf84a5dd3644330f93c88e036831578e407de7a9812b5454ef1bb8a18c3422`。核对 RoPE、原生 softmax 浮点转换、缓存首轮与单 query decode、beam 重排、有效 padding 键、目标边界和原位置，均符合所述语义。独立 CPU FP32/BF16 检查覆盖不同左／右 padding、未来目标变化、跨物品边、跨 batch cache/beam 和 PEFT，所有有效位置符合独立参考。全 padding query 按原生 eager 有限值处理，其被丢弃的输出未用作监督或合法键。见 `native_eager_probe-fixed-5a1a7297.log`。
- 审核旧原件及实际资产的运行依赖，补齐旧包遗漏的 `uneven_sampler.py`、`preflight_users.py`、`preflight_users_seed42.json`、原 Normal `evaluation_config.json`、原 Normal `rank0.json` 和 V2 `evaluate.py`，各有固定 SHA-256。原生 HF attention、部署 runner 和版本身份也须随包可读。见 `dependency_inventory_review.json`。

## 两个真实 7B 对照及五条件预跑

- 真实旧 A 同权重，行 `0/8/734`。原生 causal 与新增适配的 causal 有效 logits 最大绝对误差为 0，60 个 beam 后缀、分数及排序逐项精确相等。观察到 15 次 cache 重排。该诊断说明实测输入上的推理可比性。历史训练采用 FA2，B 采用 SDPA，单卡轨迹也有差异，因此旧 A 的正式结果可作为历史参照。严格 mask-only 因果归因需要日后另行授权的匹配 A′，本次 B 的启动门不依赖 A′。见原始 `old-causal-r0/result.json` 与 `real_eager_independent.json`。
- 真实 B 诊断使用此前真实预跑 adapter。第 8 行有 80 个 H，独立枚举新增 3,160 个反向边，实际 32 层均观察到正的跨物品注意力，禁止键概率为 0。改变未来目标时，先前 logits 和 H 层隐状态逐值相等。缓存与全量计算最大绝对误差 0.1875，relative RMS 约 0.702%，argmax 一致。独立 FP32 context 参考的两层 relative RMS 为 0.374% 和 0.263%。观察到 `aten::bmm` 和 CUDA `softmax_warp_forward`，没有走 fused SDPA。见原始 `b-mask-r0/result.json` 与 `real_eager_independent.json`。
- 实际 20 用户 × 5 条件 × 20 beam，共检查 2,000 个候选的 token 位置约束、终止、canonical target、配对稳定顺序及各行原始输入绑定。五个整输入 identity 观察（Shuffle2 三个、Shuffle3a 一个、Shuffle3b 一个）与 B Normal 的候选和排序分数逐值相等。独立重算全部 1,600 个逐用户 SID/CCE 数值，以及 80 个配对统计量、160 个区间端点，绝对误差为 0。验证对应真实 CPU metrics 与 paired 完成回执及原始预测文件。见 `real_resumable_generation_audit.json`、`pilot/final/` 与 `metrics-r0/metrics/`。
- 实际 GPU 在首个 8 用户事务提交后按预设干净退出，第二次复用 8 人并补做 12 人，最终事务大小为 8/8/4。首次观察的 6 个事务成员在恢复完成后 SHA-256 逐项相等。再次从原根调用仅核验已完成 20 人并正常退出，原始 `work.log` 为 `verified_complete`，尝试序列仍只有两次。另用独立 CPU 子进程真实退出测试覆盖 spec 写到一半、私有事务写到一半、锁和重复输入。见 `real_pilot_first_chunk_equal.json`、`prediction_store_review-fixed.log` 和重复尝试 runner 日志。实际 GPU 干净停止验证了已提交数据的恢复，进程故障与半写事务由 CPU 检查覆盖。
- 新 GPU 有界 runner 占用合计 641.864601 秒（10.70 分钟，保守计入重复核验），运行在同一 RTX 4090，物理 GPU 1 → 逻辑 CUDA 0，UUID `GPU-59cac3bd-3645-668b-79c8-839e0d878893`。所有实测作业回执有真实退出码与清理状态。生成阶段峰值 19,664,529,408 B 已分配、24,744,296,448 B 已保留，接近该卡容量。按旧训练短测及本轮分长度 eager 短测估计正式训练 18.39 小时，五条件生成 18.01 小时，合计约 36.41 单卡小时。长跑资源和耗时仍需用正式运行的监控数据核实。见 `real_runner_receipts_audit.json` 和 `resource-estimate.stdout.json`。

## 正式 B 编排和保护

审核 runner → B 训练 → 最低验证 loss（同分较早）→ 严格检查并导出一个 B adapter → 同一个 adapter 的 B Normal 和四种 V2 → 原口径 CPU 16 指标及用户配对链。固定计划含 8,214 更新、22,363 测试用户、唯一物理 GPU、固定配置和源码，预算上限 64 小时。记录了 30 分钟初次轻量观察和 72 GiB 起跑空间门槛。独立重跑 18 项 controller CPU 检查，覆盖相同 job ID 仅查询、终态与清理重启门、跨尝试预算、精确运行身份和断点保护。直接 train 与正式 generate 的缺授权实测均提前拒绝且未生成目标目录。核对实际未签计划 SHA-256 `8cf577a1d7d11fa1228996f634395544a24e3848117a482b1b48de5b85418e20`，及该计划运行环境的 9 项负例回执，目标目录和 runner job 均不存在。正式执行需要匹配固定计划的网页审核和另行收到的真实用户授权。见 `controller_tests-fixed.log`、`formal_gate_negative-fixed.log` 和 `evidence/formal/` 原始材料。

独立审核指出的 spec 半写障碍、正式身份发布非原子性、CPU 产物提前存在误判和源文件闭包问题均在最终源码修正，并用对应回归检查确认。当前证据覆盖准备阶段。四轮正式训练、全轴实际模型生成和其间物理故障后的端到端恢复需待授权后观测；20 人预跑的效果值无正式研究含义。

`FORMAL_RUN_STARTED: NO`

非作者审核签署 `/root/review`，2026-11-27，America/New_York。
