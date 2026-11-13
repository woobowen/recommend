# A1 预验收实测成本与正式运行规划

状态：**仅有界实测＋条件规划，不是正式全量成本**。本文件由训练实现作者从主控桥接的原始证据做算术核算；不构成非作者科研审核，未运行新 GPU/SSH 作业。证据时间为 `2026-11-13T11:48:24.121228+00:00`。完整来源 SHA、间隔记录和公式见同名 JSON。

训练入口固定提交：`d9de77adfc92328f2bfd5e64ba40b652946f096f`；推理入口提交：`ea922c12d5872a95180c84d5660ec853138c10a4`。两者保持当前源码冻结。

## 已实测

- 主控已 reconcile 的五次 GPU 预验收尝试累计 **590.881410 GPU-device-seconds = 9.848 GPU-device-minutes**，包含 r0/r1/continuous-r2 失败终态与 resume-r2/inference-r3 成功终态；60 device-minutes 上限还剩 50.152。此处引用账本回执，不另建或重置预算。
- continuous-r2 连续 12 updates、resume-r2 从 CP2 fresh 恢复后继续 10 updates。排除各进程首 update、以及含 checkpoint 的间隔后，取每个 update 两 rank 较慢间隔，合计 **19** 个观测：mean **5.807s**、median **5.795s**、range **5.130–6.607s/update**。这些仍含 393 对象梯度审计 hook、全 tied 梯度 finite/nonzero 检查、GPU/CPU 同步及每 micro 审计；正式路径去掉这些高开销观察，但没有测得的 hook-free 加速系数，不能据此扣减某个百分比。
- 两次 32 行 Validation 的 rank-max 分别 **1.383s / 1.249s**。只是原始真实样本的有界接口验证，不能声称完整 Validation 已运行。
- 三份完整 CP 的 collective 保存耗时 **6.227–6.598s**；每份 23 成员，payload 约 **4.42 GB**。CP2 为 **4,417,916,589 bytes**，CP12 为 **4,417,916,640 bytes**，不强行把二者记成相同字节数。严格冷恢复约 **3.764s**；两次热 best 接口严格恢复约 **3.882–4.060s**。
- inference-r3 实际 **16 用户、320 候选**，每人 Beam20；输入 71–167 tokens，包含全测试轴最长真实历史包络 167。两 rank 纯生成耗时 **3.704s / 3.654s**；含首次缓存/kernel warmup，双卡合计约 **4.32 users/s**。各进程总耗时约24.21s，其中严格 Base＋Adapter 加载核验约10.2s；不要把纯生成耗时与进程墙钟混用。
- 16 个用户级生成时延 P50 **0.350s**、P95 **1.122s**；P95 明显受两个首用户 warmup 影响。剔除每 rank 首用户后，剩余14人的 mean **0.352s**、P95 **0.375s**。这是定向预验收样本统计，不是全量用户分布的已测结论。
- 训练探针跨四个 rank 报告的最高 allocated **5.96 GiB**、reserved **13.91 GiB**；推理每卡 allocated **2.00 GiB**、reserved **2.32 GiB**。推理进程各自峰值 RSS约1.90–1.92GiB。离散资源样本最低 host available约 **232.8 GiB**、最高系统 Swap used **0.142 GiB**、最低输出盘可用约 **2754.4 GiB**；系统 Swap 已有用量不能自动归因为本作业新增。资源采样不是无间断峰值测量。

**不要采用 callback 截停探针的 Trainer `train_steps_per_second` / `train_samples_per_second` 作为吞吐。** 其速度分母仍可能使用未缩短的8240步／4轮 horizon；本核算使用真实 optimizer event 间隔。

## 正式规划：条件外推与工程余量分开

| 环节 | 直接线性 proxy | 工程规划 |
|---|---:|---:|
| 8240 updates | 13.3h（当前 hook-heavy mean） | 12–22h |
| 49,544 Validation ×4 | 2.15–2.38h（32行样本率） | 2–4h |
| 24,772用户 Beam20、双卡 | 1.59h（含首用户成本摊销）；稳态proxy约1.21h | 1.25–2.5h |
| 四次完整保存＋best恢复、阶段加载 | 当前四次保存合计proxy约26.4s，另有加载／恢复 | 0.05–0.5h |

前三环节按较慢有界 Validation 率直接相加约 **17.3h**；它不是精确预计完成时间。建议继续使用**18–30h**工程规划窗口、**18–20h**中心工作预期。范围由真实短跑率与对完整数据／prompt长度、长时间共享资源、全量Validation/IO的余量构成，不是统计置信区间，也不要求一定运行到18h。

正常作业超过30h先核实际吞吐和资源，不能据此提前停训。唯一正式seed仍受累计**96h**硬上限及原有安全／正确性停止条件约束；四轮与全量Beam20正常完成后立即结束。正式运行中的首段稳定事件及第一次全量Validation会提供更强的耗时依据，届时只更新估计，不改变配方。

这里没有测量完整 CPU Bootstrap、非作者终审和GitHub/ZIP交付耗时，18–30h不可当成最终交付的SLA。A1正式成本必须以实际run ledger和完整阶段记录为准，不能将上述预验收proxy写成全量实测。
