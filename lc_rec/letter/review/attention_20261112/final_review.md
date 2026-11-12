# A0 Attention backward 专项非作者审核

审核者：Codex 子代理 `/root/final_independent_review`
方式：**非作者通过桥接审核**
结论：**支持有界诊断交付，不构成正式训练放行。**

## 1. 审核范围

本审核者没有文件系统、命令执行工具，未直接读盘、修改文件或启动作业。主控桥接了 Boundary、真实 case 重放、deterministic 控制、预算提交、FastChat/RoPE/unpad 的源码核心摘录，以及 FA 调用、原始事件、比较报告和服务器回执；未声称逐行审核完整源码或整库。

此前另一非作者独立编写的 CPU 脚本实际读取原始 Tensor，复算哈希、逐元素差异、边界映射及状态/预算；本审核者审查其实现摘录和实际输出，并独立复核资源算术。完整大型 acceptance_state 由未修改的固定比较器处理，本审核者没有独立 GPU 重跑，也没有再次全量逐元素复算这些大型状态。

## 2. 总体判定

```text
A0 ATTENTION DIAGNOSIS: ROOT_CAUSE_NARROWED
ORIGINAL RESUME GATE: FAIL
DIAGNOSTIC VARIANT: PASS — FORMAL ADOPTION NOT AUTHORIZED
A0 FORMAL NORMAL: NOT_STARTED
```

组件层面可以确认真实第28层 case 的 FA 默认 backward 存在数值非确定性；完整历史恢复故障的唯一首次分歧及传播链尚未确认。

## 3. 实际边界与局部分歧

实际路径为：

`q_proj [B,S,4096] → view [B,S,32,128] → RoPE → unpad → FA varlen KV-packed → pad → o_proj`。

FA 原始 dQ 需经过 unpad scatter、RoPE backward 和 reshape 才到达 q_proj 输出梯度，二者不能混称。包装绑定唯一第28层、rank、step、micro 和 primary/recompute；可见 hook 返回原梯度，detach 仅用于观测副本，未见替换梯度或训练参数的行为。

默认配对 rank0 的必要 FA 输入及 dO exact，原始 dQ 有1元素不同，最大差 `4.547473508864641e-13`，超原容差数为0，符合 C2。但差异经 RoPE backward 后消失，最终 q_proj 梯度 exact；rank1 的88个边界均 exact。因此本 case 没有重现历史 q_proj 分歧的完整传播链，不能把该微小差异直接认定为全部 gate 失败的唯一原因。

CPU 同步与存盘会改变时序。最后完整对照关闭本轮 Boundary，但保留旧 Probe，不能称“完全无 hook”。

## 4. FA deterministic 证据

安装版为 `flash-attn==2.7.4.post1`，Python interface 与 CUDA extension 有固定 SHA。安装源码参数顺序及两rank原始事件证明 deterministic=True 进入实际 `varlen_bwd` extension，并非只存在于配置。

Profiler 与固定官方源码支持两模式采用不同 dQ 累加/汇总路径；未反汇编，不应声称锁定唯一机器指令。

每rank使用3个新进程、各模式每进程3次重复。rank0 默认 dQ 出现6个 hash，deterministic 为1个；forward/dK/dV 均为1个，rank1 两模式均稳定。默认最大 dQ 差 `3.725290298461914e-9`，仍无原容差超限。重放保持数值、shape、dtype、stride及调用参数，但重新分配地址和 allocator 上下文，不能推广为完整训练上下文相同。

隔离耗时与峰值显存只是该 case 的测量；固定模式顺序和 profiling 限制了正式成本外推。

## 5. 完整恢复与源码控制

默认完整恢复配对仍 FAIL。诊断 deterministic 两次独立恢复配对，以及最后 continuous/checkpoint-resume 配对，均由原固定比较器判为 PASS，两rank每对各458项 Tensor 全 exact。

最后配对从 step2 恢复至 step4，覆盖8个 microbatch、两次非零 LR optimizer updates；loss、grad norm、输入标签、loader、step/LR序列均 exact。实际 Adam 入口 LR 分别为 `4.819277108433735e-6`、`7.228915662650602e-6`，未混用 post-update scheduler LR。原容差保持 `1e-6 + 1e-3×abs(reference)`。

已记录六类状态字段 exact；224个量化模块 canonical CB/SCB exact。continuous/resume 原始 bnb 记录存在冷/热位置差异，不能称全部运行时记录逐字一致。

正式六个关键文件的共同 SHA 支持训练配方未被替换。不过默认与 deterministic 完整配对使用不同 RUN_COMMIT，含观测工程修正，不能声称完整源码只差一个布尔值。同 case native 重放是更严格的单变量证据。未发现已确认、需要修补 checkpoint 恢复算法的工程缺陷。

## 6. 预算、终态与交付边界

独立求和得到 `3790.4792780689896 device-seconds`，即 **63.17465463448316 GPU-device-minutes**。7个GPU作业低于8个上限，最长384.862764秒，低于30分钟。

最终桥接回执时间为 `2026-11-12T09:47:09.142001+00:00`：17/17作业（7GPU、10CPU）均 SUCCEEDED、exit0、cleanup verified；两GPU各1MiB、0%，无 compute 进程。新增服务器逻辑量独立核算为 `52,418,635,207 bytes = 48.81865829881281 GiB`，低于80 GiB。历史 checkpoint 全20文件重新核 SHA 一致，旧失败记录保留。

本轮未启动正式 A0、A1、Shuffle、额外 seed 或其他数据集。建议将正式采用 FA deterministic 作为首选待批准方案，披露数值路径变化及正式成本尚未充分测量。原方案不能追溯改判 PASS，亦不得自动启动正式训练。

本报告仅支持带上述限制的有界诊断交付。最终 GitHub 发布回读与 ZIP 完整性由主控执行并附真实回执，**尚未纳入本次非作者审核**。
