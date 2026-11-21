# LETTER Beauty Shuffle V2 第二组

**状态 `PARTIAL_BLOCKED`。** Beauty 单卡适配、22,363 个用户的四模式 CPU 输入重放和历史 Normal 复算已核验。2026-11-21 11:31（America/New_York），真实预跑启动时，S2 的单 rank 在导入 LLaMA 的依赖链发现 GPU 请求没有传入原 Beauty 的 `CUDA_HOME`。该尝试在装载权重前退出，记录为 0 用户、0 候选，全部任务进程已清收。四模式 GPU 预跑、正式四条件、配对统计及终审仍缺。 [GPU 原日志](evidence/failed-preflight-shuffle2.log) · [rank 回执](evidence/failed-preflight-rank0.json) · [runner 作业记录](evidence/jobs.json)

## 方案与资产

本轮固定 `GRID-LETTER-SHUFFLE-V2`、`allow_identity`、Shuffle seed 42，沿用 LETTER Beauty 的 Prompt0、`seq[:-1][-20:]`、四位 SID、未量化 BF16 LLaMA-7B、原 step 6160 best adapter，以及 batch 1、20 beam、20 return、最多 10 个新 token 和原 canonical 逐码位回调。原权重 SHA-256 为 `dd9f3d2983af7ce85ace99534bd23df9362d9cd91ac39d0b356b219cb790b1c1`。运行入口从 `6e1b211d9c37f358c43b1540b40b17f6703150d2` 的原 LETTER Beauty 版本派生。新入口核对单 rank 完整 assignment，写出原始候选以及每条真实 prefill 的输入、mask、位置和 cache 回执。模型边界在本次 GPU 失败中未执行。 [完整方法、配置和复现入口](../../../shuffle_v2/beauty_reproduction.md) · [实际派生源码](evidence/generation-derived.py) · [生成 diff](evidence/generation.diff) · [评价 diff](evidence/evaluation.diff)

## 已验证结果

原双片分别包含 11,182 和 11,181 条，按 seed 0、epoch 0 的原排列奇偶位置恢复完整单卡 assignment。全量 CPU 对照验证了 159,966 个可见历史组，原始编码逐行与归档 Normal 一致。固定槽、组外 token、长度和 mask 检查无违规。两个非作者独立核对了全量排列及归档状态。 [重放证据](evidence/cpu-replay.json) · [分片证据](evidence/single-shard.json) · [非作者核验](evidence/replay-independent.json)

| 模式 | 恒等历史组 | 整条输入恒等用户 | 实际变化 token |
| --- | ---: | ---: | ---: |
| S2 | 79,937 | 640 | 160,058 |
| S3a | 26,744 | 6 | 319,567 |
| S3b | 26,744 | 6 | 319,567 |
| S4 | 6,630 | 0 | 479,920 |

Normal 继续使用原双分片的 447,260 条候选。复评的 16 项指标、逐用户矩阵和逐用户 CSV 与历史归档一致。独立实现从原候选复算后，逐用户值及均值最大偏差均为 0。目录共 12,101 件商品、12,088 个完整 SID，其中 11 个 SID 碰撞组。S2、S3a、S3b、S4 的正式推荐指标、相对变化和 80 项配对比较尚无结果。 [Normal 全精度 16 项](evidence/normal-summary.json) · [Normal 非作者逐用户复核](evidence/normal-independent.json)

## 修正与接续

三次集中修正对应缺少 `json` 导入、样本回执容器类型与预算前缀、以及 TorchInductor 退出时的残留进程。第三次重放用第一组已验证的 `TORCHINDUCTOR_COMPILE_THREADS=1`，正常退出。累计 runner CPU 作业时间为 490.863291 秒。GPU 预跑失败请求的 wall time 为 8.454977 秒，按一张卡全额计入预跑预算。正式 GPU 用量为 0。未做新 Normal／OFF GPU 推理。

待重新授权时，首选从原不可变 Normal 请求恢复 GPU 作业的 `CUDA_HOME` 与 `LD_LIBRARY_PATH`，先在新身份四模式预跑中验证模型、候选和 GPU 用时，再由非作者审查并判定正式启动。现场确认的 CUDA 工具链是原实验使用的 12.4.131。本次 GPU 尝试入口 `/home/bwwu/grid-audits/sid-extensions-p0/jobs/letter-beauty-v2-gpu-preflight-r0`，累计跑数及输出入口 `/data/bwwu/lc-rec/letter-shuffle-v2-beauty/`。实际运行和控制源码快照 `d9fbfadd92864e5432c55c998cf23ffbd24a68b3`。历史 Instruments 第一组已通过网页二审。 [V2 总入口与第一组](../../../shuffle_v2/README.md)
