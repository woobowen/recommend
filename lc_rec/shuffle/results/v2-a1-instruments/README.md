# 第四组 Instruments A1 Qwen Shuffle V2

**正式前验收通过，等待网页二审和用户单独的正式启动授权。** 2026-11-23，美国东部时间。正式父任务和正式子任务均未提交，正式 GPU 用量为 0。32 人指标只用于核验工程链。

## 研究方案

采用 LETTER 发布的 Musical Instruments 四位 SID、原 LETTER-LC-Rec 推荐任务与冻结 A1 Qwen3-0.6B-Base。比较 S2、S3a、S3b、S4 分别减冻结 Normal，以及 S3b 减 S3a。历史 V1 采用独立的非恒等协议，由固定交付入口保留。 [V1 与原 Instruments A0/A1](https://github.com/woobowen/recommend/blob/3396d45a041424fe4af625bfc0b047467150c5c1/lc_rec/shuffle/FINAL_REVIEW_ENTRY.md)

| V2 模式 | 活动槽，从 0 起 | 固定槽 | 字典序目录大小 | 理论恒等概率 |
|---|---|---|---:|---:|
| S2 | 2, 3 | 0, 1 | 2 | 1/2 |
| S3a | 0, 1, 2 | 3 | 6 | 1/6 |
| S3b | 1, 2, 3 | 0 | 6 | 1/6 |
| S4 | 0, 1, 2, 3 | 无 | 24 | 1/24 |

协议 `GRID-LETTER-SHUFFLE-V2`，`allow_identity`，Shuffle seed 42。紧凑 UTF-8 JSON 键 `[protocol,seed,m,row_idx,visible_item_idx,counter]` 经完整 SHA-256，按无符号大端整数拒绝超出均匀取模上界的值。索引映射为 `new[A[j]]=old[A[perm[j]]]`。三位模式共享取样键与相对排列。row_idx 来自全局冻结轴，visible_item_idx 是历史出现位置。排列与 rank、batch、进程和 GPU 无关。四种模式各自作用于同一冻结输入。

在冻结 A1 的 SeqRecDataset Prompt0、AutoTokenizer 和显式 mask collate 完成后，模型 generate 前置换 `seq[:-1][-20:]` 每件商品的四个完整 token ID，包括层级前缀和 embedding 查表身份。商品时间顺序、边界、固定槽、组外 Prompt、原 mask、目标和生成端码位顺序保持原语义。新 token 在 Qwen 的当前位置使用原生 RoPE 和因果上下文。每次生成由长度为零的 DynamicCache 开始。

## 冻结资产

- LETTER 原数据有 24,772 名测试用户、9,922 个商品、9,897 个不同 SID、23 个碰撞组和 172,929 个可见历史商品组。index/inter/item 逐文件身份见有效配置。
- 基座 `Qwen/Qwen3-0.6B-Base@da87bfb608c14b7cf20ba1ce41287e8de496c0cd`，权重 SHA-256 `cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba`。选择回执绑定 step 8240 的原 A1 best adapter，SHA-256 `fe96ea74fc3c7253b0c3687239266b0f073e1a51ef9158e7d97bf30b4527bd03`。
- 原 GPU Normal RUN `0e3e2c6f5c1a76152ea0228390d94ffc1ff75ba5`，原 CPU 评价 `445f1475cd1dd70d19881dbc1639bb311f7cccf7`。两个冻结分片各 12,386 人，共 495,440 个原始候选，见路由和原始 SHA。 [冻结 Normal 终审](https://github.com/woobowen/recommend/blob/3396d45a041424fe4af625bfc0b047467150c5c1/lc_rec/a1_qwen3_06b/A1_NORMAL_FINAL_REVIEW_ENTRY.md)
- 词表 152,587，新增 SID token 918。BOS 无且编码不自动添加 BOS，EOS=PAD=151643，左 padding。tokenizer 与原 A1 保存件、重新构造的基座词表按完整映射比较。历史词表摘要使用无排序字典序列化，本轮改用排序键摘要以避免 Rust HashMap 遍历顺序波动。原历史摘要及本轮摘要均在有效配置。
- 使用 `/data/bwwu/envs/qwen-preflight-20261110-qwen3/bin/python`，Torch 2.6.0、Transformers 4.57.6、PEFT 0.21.2、Accelerate 1.12.0。GPU 沿用历史 CUDA 环境变量，CPU 屏蔽 CUDA 并使用 CPU accelerator。命令与路径见 [有效配置](../../../shuffle_v2/a1_instruments.json)。在同一服务器通过授权账户读取这些固定位置，模型可按 revision 从 [Qwen 基座](https://huggingface.co/Qwen/Qwen3-0.6B-Base/tree/da87bfb608c14b7cf20ba1ce41287e8de496c0cd) 取回并核对哈希。Adapter、Normal 和正式资产使用服务器访问权限，已提供可离线审核的 tokenizer、数据和预测原字节。

## 实际调用链

[原 evaluate.py](https://github.com/woobowen/recommend/blob/0e3e2c6f5c1a76152ea0228390d94ffc1ff75ba5/lc_rec/a1_qwen3_06b/scripts/evaluate.py) 由 [A1 派生入口](../../../shuffle_v2/a1_generate.py) 做固定锚点适配，每次保存实际运行源码和统一 diff。冻结 `strict_reload` 原样调用，未量化 BF16 基座、392 项 BF16 LoRA、`autocast_adapter_dtype=False`、394 个保存键和活跃 embedding/head 同 Parameter 同 storage。模型级审计验证所有预跑参数前后 digest 一致。原 [Allowed](https://github.com/woobowen/recommend/blob/0e3e2c6f5c1a76152ea0228390d94ffc1ff75ba5/lc_rec/qwen_preflight_20261110/scripts/common.py) 先用原文本、原 IDs 和 offset 检验实际 Response 分隔符，随后绑定已置换前缀，逐码位合法集与 EOS 保持其原生行为。

运行一张 GPU、一个 rank、batch 1。CPU 证明 world=1 的全量 sampler 与原双 rank 分片逐位交织一致，维持 `shuffle=True,seed=0,epoch=0,drop_last=True`。预跑复用真实 V1 A1 rank0 的 16 人和 rank1 的 16 人，按两份冻结名单顺序连接。生成使用 beam=return=20、max_new_tokens=10、deterministic beam search、early stopping、cache、原解码和 FP32 分数稳定降序。真实 GPU prefill 验证全部 20 beam 的 IDs、mask 与位置，核验 cache_position、28 层空 cache、BF16 embedding 的置换及模型 eval/no-grad 状态。

## 实测前置证据

| 模式 | 恒等历史组 / 172,929 | 恒等全用户输入 / 24,772 | 实际变化 token | 32 人自然恒等 |
|---|---:|---:|---:|---:|
| S2 | 86,483 | 714 | 172,892 | 2 |
| S3a | 28,935 | 6 | 345,429 | 0 |
| S3b | 28,935 | 6 | 345,429 | 0 |
| S4 | 7,147 | 0 | 519,226 | 0 |

- 生产 CPU 及独立审核 A 从 A1 tokenizer 重放 24,772 人、99,088 条模式–用户记录、691,716 次组核验。显式 mask、组内完整 token、组外与冻结原输入及 OFF 不变性通过。A0 只用于共享出现轴和相对排列的交叉核验。 [CPU 全量结果](cpu-replay-audit.json)
- 两个冻结 Normal 原分片完整复算，24,772×16 矩阵与全部 16 个均值均精确一致。原序、候选和碰撞绑定完整校验。 [Normal 复算](normal-recomputation.json)
- S2 → S3a → S3b → S4 真实串行预跑，共 128 条模型边界记录、2,560 个候选。每候选四个合法码位 token 后真实 EOS，原精度和次序保留。S2 行 12005、17716 与各自冻结 Normal 输入、候选、原序、稳定排序和 40 个 FP32 分数逐项精确一致。预跑未执行 Normal 或强制恒等，四模式零正式样本。 [GPU 回执](gpu-preflight-summary.json)
- 评价保留原候选、目录外占位与重复处理。ItemHit 和 ItemNDCG 由冻结 CCE 按碰撞组展开、对组内 item 给 uniform credit。SID Hit@1/5/10/20、NDCG@5/10/20、MRR@5/10/20、ItemHit 和 ItemNDCG 各 @5/10/20。五组比较共享同一用户轴及 5,000 次 PCG64 有放回抽样，统计 seed 20261113，95% 线性百分位，无多重比较调整。80 条工程结果及全部数组由独立 B 从原候选验证。比例差按右减左，百分点为差乘 100，相对改变为差除以基线，基线为零时不定义。 [工程评价](cpu-assess-summary.json) · [80 条工程记录](paired-engineering.csv)
- 审核 A 直接核对实际源码和全部边界、独立 Prompt0、采样和模型证据。审核 B 直接从原候选重算 Normal 和预跑四条件、共享 bootstrap、全部区间，核对控制计划和服务器预算。独立统计求和顺序的最大 draw 偏差为 1.39e−17，80 个上下界均精确一致。 [审核 A](A_REVIEW.md) · [审核 B](B_REVIEW.md) · [审核 A 原始回执](A_recomputation.json) · [审核 B 原始回执](B_recomputation.json)

## 未提交的正式计划

未来九个阶段依次为 S2 GPU → S2 CPU → S3a GPU → S3a CPU → S3b GPU → S3b CPU → S4 GPU → S4 CPU → paired。每个 GPU 条件覆盖 24,772 人、495,440 个候选。依赖阶段在上一步产生验证产物并实际清收后推进。运行失败保留原 ID 与输出，按原 ID 查询并审核后处理接续。CPU 读取已清收的单分片预测。正式父监督程序对网页二审与独立用户授权有硬门槛。

- 预算与实际用量。GPU 预跑 167.244 / 10,800 device-seconds；CPU 144.011 / 28,800 worker wall-seconds（含监督清收的保守上界）。六项实际任务退出 0，另一个生命周期夹具按设计超时退出 −15，证明清收归属内的独立 session，七项都已验证清收。实验 r0 首试通过，修正轮次 0。 [预算与终态](budget.json)
- 正式估时。 32 人预跑按全量历史长度重加权的四模式点估计约 2.57、3.33、2.61、2.61 小时，合计 11.13 小时，规划波动约 10.41–12.80 小时。单模式上限 8 小时，整组计划上限 36 GPU-device-hours。严格装载 15.79–15.93 秒/次，峰值 allocated 约 2.00 GiB，reserved 约 2.32 GiB。估计保留了首次生成的计时影响和较长历史过采样的限制。
- 全量 CPU 每模式约 20–90 秒，最终 paired 约 60–360 秒，CPU 九阶段中的五项合计约 140–720 秒。四个原预测预计约 1.62 GB，正式抽样索引约 0.99 GB，审核 ZIP 预计 0.4–0.9 GB、解包约 3–4.5 GB。导出暂存预留 10 GB，最后审核整理和交付估计 30–90 分钟。 [估算明细](resource-estimate.json) · [估算程序](estimate.py)
- [未提交请求计划](formal-plan.json) · [dry-run](formal-dry-run.json) · [正式零提交服务器回查](formal-zero-final.json) · [非作者计划核验](B_formal_plan_review.json)

## 版本、服务器和复现

| 用途 | 固定提交与部署 |
|---|---|
| 正式前 CPU replay、Normal、GPU | [c487f7e7176bc223445d3deecfab87e0bcc30d1a](c487f7e7176bc223445d3deecfab87e0bcc30d1a.deployment.json) |
| 预跑评价统计及未来 parent/CPU | [3cc1110097c4764ebfd93bb38ee2913c8afb6958](3cc1110097c4764ebfd93bb38ee2913c8afb6958.deployment.json) |
| 终态证据导出 | [9245a4ad9845ed3fc4fde7ee3f5e6c2a7f654fb8](9245a4ad9845ed3fc4fde7ee3f5e6c2a7f654fb8.deployment.json) |

服务器结果根 `/data/bwwu/lc-rec/a1-instruments-v2`，runner 作业根 `/home/bwwu/grid-audits/sid-extensions-p0/jobs`，前缀 `a1-instruments-v2-`。部署回执按每个 Git blob 的字节数及 SHA-256 核对服务器 export，export 无 Git 元数据。发布版本包含审阅材料。完整审核 ZIP 的根 README 提供离线精确复算，包含冻结 Normal 原预测、实际 GPU 四分片、模型边界、源码/diff、hash、评价数组、作业日志及清收。

```bash
python3 -B -m lc_rec.shuffle_v2.a1_control status
python3 -B -m lc_rec.shuffle_v2.a1_control query gpu-preflight-r0
python3 -B -m lc_rec.shuffle_v2.a1_control query cpu-assess-r0
python3 -B -m lc_rec.shuffle_v2.a1_formal dry-run \
  --plan lc_rec/shuffle/results/v2-a1-instruments/formal-plan.json
```

本组已验证的是固定资产上的工程行为及指定 32 人的生成。自然恒等证据只覆盖两名 S2 用户。正式推荐差异、全量单卡与历史双卡的数值轨迹及跨训练 seed 的变动仍待相应实验或证据。下一步为网页二审和用户单独的正式启动授权。涉及研究的三个既有组及 V1 历史冻结材料均保留。
