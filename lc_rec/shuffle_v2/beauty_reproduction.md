# LETTER Beauty Shuffle V2

本轮用冻结的原 LETTER Beauty 数据和 LLaMA-7B 测试 SID 顺序干预，用户轴为原交互 JSON 的 22,363 行。每条可见历史由 `seq[:-1][-20:]` 得到。先构造 Prompt0 并调用原 `TestCollator`，再按历史商品出现位置移动四位 SID 内的完整 token ID。模型按移动后的位置使用原生 RoPE 和因果上下文。目标、固定位、组外 token、mask 和原逐码位生成约束沿用冻结实现。

| 条件 | 活动码位（从 1 开始） | 全排列数 | 单组恒等概率 |
| --- | --- | ---: | ---: |
| S2 | 3、4 | 2 | 1/2 |
| S3a | 1、2、3 | 6 | 1/6 |
| S3b | 2、3、4 | 6 | 1/6 |
| S4 | 1、2、3、4 | 24 | 1/24 |

抽样键为紧凑 UTF-8 JSON `[protocol,42,m,row_idx,visible_item_idx,counter]`。完整 SHA-256 经无符号大端解释与拒绝采样映射到字典序排列，S3a 与 S3b 共享三槽相对排列。恒等出现时按实际抽样执行。源索引方向是 `new[A[j]] = old[A[perm[j]]]`。

## 原版与单卡派生

原 Beauty 运行提交为 `6e1b211d9c37f358c43b1540b40b17f6703150d2`。历史 Normal 由其两个不等长分片生成。原选模按验证 loss 选择 step 6160，对应 adapter SHA-256 `dd9f3d2983af7ce85ace99534bd23df9362d9cd91ac39d0b356b219cb790b1c1`，基座 revision 为 `4782ad278652c7c71b72204d462d6d01eaaf7549`。基座、embedding 和 lm_head 以 BF16 装载，冻结 PEFT 0.5.0 实际以 FP32 装载 LoRA 张量。身份、生效路径和历史来源见配置。 [原正式入口](https://github.com/woobowen/recommend/blob/6e1b211d9c37f358c43b1540b40b17f6703150d2/lc_rec/a0_beauty/formal_evaluate.py) · [资产与预算配置](a0_beauty_letter.json)

`beauty.py` 对冻结 Beauty 脚本及原 V1 接入做哈希匹配后，导出派生源码与 diff。原 `uneven_sampler` 在同一 seed 0、epoch 0 排列上取单 rank 全轴，其偶、奇位置分别对应历史 11,182 和 11,181 行。生成用 batch 1、beam/return 各 20、最多 10 个新 token、early stopping 和原 EOS/pad/cache。运行时严格核对每个输入与 CPU 清单、真实模型边界处的 20 个 beam、原生 position IDs 和空初始 cache。GPU 预跑另核对逐 token embedding 查表结果。 [生成派生](beauty.py) · [模型守卫](runtime.py)

`beauty_evaluate.py` 将冻结 16 项指标的读取合同改成 Normal 两片、新条件一片，保留其稳定排序、原精度分数、重复候选及 CCE 实现。无效完整 SID 消耗一个位置，碰撞组按 item 展开，在首次目标 SID 的组内均匀分配 credit。四种干预分别与同一 Normal 配对，另比较 S3b − S3a。五项比较使用共享 PCG64 用户 Bootstrap，5,000 次有放回抽样，seed 20261113，线性百分位 95% 区间未经多重比较调整。 [评价](beauty_evaluate.py) · [配对统计](statistics.py) · [独立复算](independent_beauty_review.py)

## 执行与复算

固定 checkpoint 重评使用配置声明的服务器资产、依赖环境与受审导出。使用已有 runner 依次执行 S2、S3a、S3b、S4，每次生成完整退出并通过单分片全量评价之后进入下一条件。首次提交与恢复按已持久化请求和原 job ID 核对；独立工作证据见 `beauty_control` 和 `beauty_observe`。正式 GPU 单次尝试最多 8 小时，累计 36 GPU-device-hours，四条件预跑累计 60 GPU-device-minutes。CPU 限额按 runner 作业 wall time 单列。 [提交和恢复](beauty_control.py) · [现场观察](beauty_observe.py)

在解压审核包后运行根目录的 `offline_review.sh`，给出独立审核 JSON 的输出路径。脚本用包内冻结原始候选、原精度分数、用户及目标轴、SID 目录、逐用户矩阵和统计数组复算，依赖 Python 和 NumPy。该入口与 checkpoint 重评是两个复现起点。

结论只覆盖此固定权重、冻结 Beauty 用户轴及 seed 42 的置换实例。
