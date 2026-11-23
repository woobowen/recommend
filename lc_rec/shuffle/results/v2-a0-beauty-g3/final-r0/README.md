# 第三组 GRID G3 Beauty Shuffle V2 正式结果

`THIRD_GROUP_COMPLETE_PENDING_WEB_REVIEW`。原正式父任务 `g3-beauty-v2-group-formal-r0` 于 2026-11-23 02:46:48 EST（07:46:48 UTC）达到 `PASS_G3_BEAUTY_V2_FORMAL`。S2、S3a、S3b、S4 四个 GPU 生成作业、四个对应 CPU 全量评价以及最终配对均为 `SUCCEEDED`、退出码 0。原父任务和九个子阶段的进程清收均已验证。正式链无重试。本阶段内部终审通过，网页终审待进行。 [终态、预算、固定身份与原件哈希](terminal.json) · [已审启动观察](https://github.com/woobowen/recommend/blob/6ae4e0d1f53c11d6062944db720ba560291794c0/lc_rec/shuffle/results/v2-a0-beauty-g3/launch-observation-r0/README.md)

## 冻结方案与执行

使用 G3 GRID 四位 SID、已核验 Beauty 商品桥接、原 SeqRec Prompt0、LLaMA-7B 和 checkpoint-6160。冻结 G3 Normal 是原单 rank 全量预测。测试轴为 22,363 人，12,101 件商品有 12,101 个唯一完整 SID，碰撞组为 0。原 tokenizer 含 32,786 个 token，其中 786 个为新增 SID token。随后只在完整历史 SID 组内置换 token，位置从 0 起，S2 活动槽 (2,3)，S3a (0,1,2)，S3b (1,2,3)，S4 (0,1,2,3)。采用正式前已审的 SHA-256 拒绝采样、全局行号、可见出现位置、seed 42 与允许恒等的完整字典序目录。S3a、S3b 共用相对排列。保持固定槽、组外输入、mask、目标、组间顺序和原逐码位生成约束。 [冻结方案](https://github.com/woobowen/recommend/blob/910034add71fe9f8f988c052b52cb484c47f0bf5/lc_rec/shuffle/results/v2-a0-beauty-g3/README.md)

GPU 生成固定 `86660be5a62c8325caf74cf2810581850ca12803`，CPU 评价和正式控制器固定 `cb3b1f36d69ef0c1a7f554ef5455026523635f78`，模型训练来源 `21d495b0be5b606853ec7cec68a3eaa1b58acbd7`。实际使用一张 RTX 4090，UUID `GPU-59cac3bd-3645-668b-79c8-839e0d878893`，物理 1、逻辑 0，`WORLD_SIZE=1`、`RANK=0`。batch 1、beam 和 return 均为 20、最多 10 个新 token、确定性 beam、原 EOS 及稳定排序。正式模型范围为 FORMAL，使用原 BF16 基座、embedding 与头、FP32 LoRA 和严格 450 项 tensor 装载。原 Normal 预测直接复用。 [GPU 实际入口](https://github.com/woobowen/recommend/blob/86660be5a62c8325caf74cf2810581850ca12803/lc_rec/shuffle_v2/g3_generate.py) · [CPU 实际入口](https://github.com/woobowen/recommend/blob/cb3b1f36d69ef0c1a7f554ef5455026523635f78/lc_rec/shuffle_v2/g3_evaluate.py)

服务器根目录为 `/data/bwwu/lc-rec/g3-beauty-v2/`，原版产物保存在 `gpu-formal-<mode>-r0`、`cpu-formal-eval-<mode>-r0`、`cpu-formal-paired-r0`、`group-formal-r0`。原计划与授权留在既有 requests 路径。实际请求、接受回执及归档输出由同一 runner 的查询按哈希绑定。正式 GPU 用量 58,174.669349 device-seconds，父任务墙钟 58,611.587900 秒。本轮接续运行了两个独立 CPU 审核和一个证据导出 worker。预算账本含历史预跑、CPU 准备和失败尝试。 [原父任务只读查询](https://github.com/woobowen/recommend/blob/b9bc7b4496f39908e432662ec6137215fed655d1/lc_rec/shuffle/results/v2-a0-beauty-g3/launch-observation-r0/README.md) · [完整累计与剩余额度](terminal.json)

## 指标与配对

[五条件 16 个指标](metrics.csv) 使用比例，保留写出的完整精度。[四模式相对 Normal 的变化](changes_from_normal.csv) 含带符号比例差、百分点差和相对变化百分比，基线为零时相对变化留空。[80 条配对](paired.csv) 按 `comparison` 中第一个条件减第二个条件，区间以比例表示。`relative_change` 是相对第二项的比例差。五项对照共用同一用户轴及 5,000 次 PCG64 `20261113` 有放回抽样。95% 区间用线性百分位，逐项计算且未经多重比较调整。索引流 SHA-256 `2c3905c7666c8639a02c2836cf34217c6d49d0753de1a25f81cec5c1bd2b031d`。审核 ZIP 含全量逐用户 CSV/NPZ、每项差值和 5,000 次统计均值。 [表格生成脚本](results_tables.py)

16 个指标的点估计均为 Normal > S2 > S3b > S3a > S4。相对 Normal，S2 有 13 个区间低于零，SID Hit@1、SID Hit@10、ItemHit@10 的区间含零。S3a、S3b、S4 各自的 16 个区间均低于零。S3b−S3a 的 16 个区间均高于零。解释仅覆盖本次固定权重、Beauty 测试轴及 seed 42。相同数值的 SID 和 Item 指标仍分别按冻结规则计算。

## 两个非作者终审

- 方向 A `review_generation` 对服务器原始 JSONL 逐行独立检查了 89,452 条正式预测、1,789,040 个候选及 639,864 次历史商品出现。所有正式输入均对应冻结 CPU 证据，四模式保存的 embedding、20-beam prefill、position IDs 和初始 KV 回执通过。两种 S3 的相对排列流一致。A 审核覆盖离线复算输入与运行时回执，模型数值直接沿用正式作业的真实守卫和保存证据。 [方向 A 审核意见](review-a.md) · [方向 A 结果](review-a.json) · [独立脚本](audit_formal.py)
- 方向 B `review_launch` 从五条件 2,236,300 个原始候选重算 SID 与冻结 CCE 指标，逐用户 CSV/NPZ、均值、80 条配对点估计和原 Normal 存档最大差均为 0。独立 bootstrap 均值数组的最大绝差为 `1.1102230246251565e-16`，95% 区间端点为 `4.85722573273506e-17`。Normal、S2、S3a、S3b、S4 的目录外完整 SID 分别为 507、317、90、210、58，均保留原排名，重复候选数均为 0。 [方向 B 审核意见](review-b.md) · [方向 B 结果](review-b.json) · [独立脚本](verify_full_metrics_b.py) · [审核包调用入口](verify_bundle_metrics_b.py)

自然整条输入恒等为 S2 640，S3a 6，S3b 6，S4 0。652 条条件–用户记录的输入、候选内容、原生成及排序次序和原精度分数均与对应冻结 Normal 精确一致。S4 在此置换实例下无自然整条恒等可比较。完整逐行比较保存在审核 ZIP 的 `evidence/review/natural-identity.jsonl`。

## 离线复核

在审核 ZIP 根目录按根 README 执行两段独立 CPU 复算。需要 Python 3.10+、B 方向需要 NumPy。五份预测保留服务器原始 JSONL 字节，逐条含原精度分数、输入和可用的模型边界回执。包内有实际训练与执行身份、冻结配置和请求、运行派生产物及差异。固定 checkpoint 与 LLaMA-7B 大权重存放服务器，需有读取权限，按预检中所列绝对路径与 SHA 核验。 [前置资产、离线依赖与权重路径](https://github.com/woobowen/recommend/blob/910034add71fe9f8f988c052b52cb484c47f0bf5/lc_rec/shuffle/results/v2-a0-beauty-g3/preflight/README.md)
