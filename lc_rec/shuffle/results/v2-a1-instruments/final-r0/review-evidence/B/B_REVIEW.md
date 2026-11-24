# 第四组正式终态独立审核 B

**PASS_NON_AUTHOR_B_FORMAL_FINAL**。本审核直接读取冻结 Normal 与四个正式条件的原始预测、实际目录及 tokenizer，独立复算全部指标和配对统计，并核验正式执行链和发布表。审核者未实现正式控制器、生成、CPU evaluator、paired 或发布表程序。

## 原始候选与评价

- Normal 两分片各 12,386 人。S2、S3a、S3b、S4 每条件 24,772 人、495,440 条原始候选。合计核对 2,477,200 条候选，含 target、每行 20 beam、原生成序、原精度有限 FP32 分数、稳定降序、SID 码位、EOS 和重复记录。正式单 rank 行序逐条对应两份冻结 Normal 的交织用户轴。目录核对 9,922 商品、9,897 个 SID 与 23 个碰撞组。
- 重新实现候选 SID 排名和显式 item 位置展开。保留目录外候选的占位和冻结 CCE 均匀 credit；所有五个条件的原始候选计数、碰撞诊断和每项均值与正式记录相等。发现目录外完整 SID 的个数依次为 239、159、70、58、41。五条件的重复候选计数均为 0。
- 五个逐用户 NPZ 矩阵各有 396,352 个值，合计 1,981,760 个，与独立计算逐值相等。五份逐用户 CSV 同样逐值核对。冻结 Normal 的历史 16 项均值及矩阵完整复算，也核对两分片原始 SHA-256。 [全量计算回执](full-metrics.json)

## 全量配对

5,000 个 PCG64 seed 20261113 的共享用户重采样，共 123,860,000 个 int64 索引，与恢复的原始 NPY 逐值相等。索引数据 SHA-256 为 `db52e9d47cea6b73d2334551586477ad141a716e87f2bc65c541f042efc170a9`。原始 NPY 文件 SHA-256 为 `906ca6be4d68bf94d4b005886af2210eba3b8d73f43071f0b211463a0b378792`。

从独立得到的逐用户指标复算 1,981,760 个 paired 用户差值，全部相等。用用户重数做加权归约，独立重建 400,000 个 bootstrap 均值。与原程序按抽样顺序求均值的结果最大绝对差为 `1.8735013540549517e-16`。80 条 paired 记录的均值、原比例变化、百分点及相对变化逐值相等。两端点均对 2.5% 和 97.5% 线性插值作独立复算，160 个端点最大绝对差为 `1.0408340855860843e-16`。直接使用已保存的抽样均值重算百分位时，160 个端点逐值相等。 [全量计算回执](full-metrics.json)

| 配对 | 95% 区间严格低于 0 | 区间含 0 | 区间严格高于 0 |
|---|---:|---:|---:|
| S2 − Normal | 14 | 2 | 0 |
| S3a − Normal | 16 | 0 | 0 |
| S3b − Normal | 16 | 0 | 0 |
| S4 − Normal | 16 | 0 | 0 |
| S3b − S3a | 0 | 8 | 8 |

16 项点估计均呈 Normal > S2 > S3b > S3a > S4。S2 中区间含零的两项是 SID Hit@20 和 ItemHit@20。配对区间基于同一冻结用户轴、模型与一次 Shuffle seed 42 实例，采用未作多重比较校正的逐指标区间。 [发布表复核](publication-tables.json)

## 终态、证据与预算

- 逐条核对正式父与九个子任务的原始 request、accepted、attempt、status、cleanup、stdout/stderr 和新连接终态查询。请求与固定计划、用户授权摘要、两个服务器源码快照及实际前序产物 SHA-256 相符。每个后继任务在前序真实退出并清收后接受。正式任务均退出 0，清收通过，终态查询无运行中的原 job worker 或 supervisor。
- GPU 使用同一审核 UUID，服务器上物理 GPU 1、任务内逻辑 GPU 0。每阶段检查 990 条顺序进度检查点及实际 GPU/内存/空间记录。启动日志中的 torch_dtype 弃用提示和 NCCL 单 rank 映射提示均有后续完整生成和明确退出记录。检查范围包含加载、CUDA/OOM、验证断言、超时、I/O、重试和清收。
- 正式 GPU 的 runner 占用上界按 S2、S3a、S3b、S4 分别为 9,607.291075、10,140.820307、10,154.132332、10,155.429938 device-seconds。累计 40,057.673652 秒，约 11.127 GPU-device-hours，低于每项 8 小时和整组 36 小时授权。正式 CPU 的四次逐条件评价与 paired 共计 123.678643 worker wall-seconds。父任务 wall 40,296.946756 秒，UTC 2026-11-24 04:26:12.091519 正常退出，对应纽约时间 2026-11-23 23:26:12.091519。准备阶段 GPU 167.243993 device-seconds、CPU 144.010992 worker wall-seconds，分别记账。 [终态与资源复核](execution-trace.json)

发布候选还通过独立复核。对比正式 paired 的全部 80 条记录、五条件全部 80 个均值、64 个相对 Normal 的变化及方向分类。输出比例和原值保持全精度。相对变化百分数采用 `(100 × 差值) ÷ Normal`；换序计算只产生至多 `7.105427357601002e-15` 的机器舍入差。 [发布表复核](publication-tables.json)

本次审核调整发生在本地复核脚本。首次复算的四个正式条件和冻结 Normal 已核对，进入统计时原始 int64 NPY 恢复尚未到达，因而复算入口增加先检查文件再处理的前置条件。发布表复核使用与输出一致的百分数算式验证全精度。这些调整未产生正式作业重试或 GPU 用量。

## 离线复核

以 `raw`、`tables` 和 `review-B` 的共同父目录为当前目录。先按包内说明恢复原始 int64 NPY。使用 Python 3.11 或更新版本与 NumPy。

```bash
python3 -B review-B/verify_trace.py --raw raw --output execution-trace.recomputed.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B review-B/verify_results.py \
  --raw raw --indices bootstrap_indices.restored.npy --output full-metrics.recomputed.json
python3 -B review-B/verify_tables.py --raw raw --tables tables \
  --metrics-audit full-metrics.recomputed.json --output publication-tables.recomputed.json
```

复算时间因机器而异。bootstrap 独立归约的末位舍入可随 BLAS 变化，原索引、指标值、点估计与已存抽样的百分位仍按程序中的逐值检查验收。实验解释以固定模型、固定 Instruments 测试轴和 seed 42 为限。
