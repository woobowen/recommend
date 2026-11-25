# GRID Instruments 正式启动与 30 分钟观察

正式作业 `grid-letter-rerun-20261125-formal-r0` 已通过上游量化、SID 准入和数据绑定，Normal 训练在观察截止时持续推进。截止时完成 2 轮全量 Normal 验证。交付前独立查询得到 `RUNNING`。同一 supervisor 和 worker 继续运行，Codex 的 30 分钟主动观察已结束。三条件测试和离线复算等待训练正常完成并生成最终 best。

## 执行身份

- 接受时刻 2026-11-25 13:36:38 EST，服务器 started_at 为 2026-11-25 13:36:38 EST。正式执行提交 `b08c63940f5351dce63dc4cb8a0f9c6f1d8ac86e`，请求 canonical SHA-256 `bf0226675093e04ff777a76077cdd8173b627f977e6153a175b969faafae4eb1`。
- 服务器 `bwwu@zwlab-01`，runner profile `new-server`。supervisor PID 3526583、PPID 1，worker PID 3526585。两者分别拥有独立 session 和进程组，均无 controlling TTY，stdin 为 `/dev/null`。在提交连接结束后以独立 SSH 查到真实进程、固定 argv、日志和 GPU 上的本作业进程。
- 物理 GPU 1 为 NVIDIA GeForce RTX 4090，UUID `GPU-59cac3bd-3645-668b-79c8-839e0d878893`，作业内映射逻辑 CUDA 0。请求和实际设备一致。启动前该卡 1 MiB，资产与工作目录均在服务器。
- 首次查询原 ID 返回 request 文件不存在，现场目录独立核对仅有已完成的 `preview-r0`。本轮使用受审正式模板提交一次。正式计算无重试。采用相同源码、输入、设备及 86,400 秒 runner timeout。
- 原执行位置 `/data/bwwu/grid-letter-rerun-20261125/source-b08c63940f5351dce63dc4cb8a0f9c6f1d8ac86e`。作业输出 `/home/bwwu/grid-audits/sid-extensions-p0/jobs/grid-letter-rerun-20261125-formal-r0/output`。发布提交提供本次观测证据，运行继续绑定上面的执行提交。

## 实际阶段

| 阶段 | 回执秒数 | 核查结果 |
| --- | ---: | --- |
| 正式量化 | 131.895 | 30 次 Lightning 更新，三层各 10 次且每次对应活跃中心变化。原生 36 批，含每层两批缓冲与初始化，累计 73,728 次商品访问，覆盖全部 9,922 件。 |
| 正式 SID | 31.077 | 9,922 个唯一 SID、8,625 个前三码前缀、784 个碰撞组、第四码最大 16。全量准入及原生 lookup 通过，SID SHA-256 `e1cc4d2b0fddb7a48d9a26f5e99d338965590657ddaa18ec90fa3814806e948e`。 |
| 数据绑定 | 47.274 | 新 SID 绑定到共同目录、用户轴和有序交互，训练、验证和测试 TFRecord 分别读回 24,772 用户。验证历史 148,927 个位置，测试历史 172,929 个位置，每个用户各保留一个目标。 |
| Normal 正式训练 | 进行中 | 受审配置从随机初始化开始，单卡 FP32、train seed 42、源 batch 16、梯度累积 16、Adam，每 1,600 个训练 microbatch 进入一次完整验证。 |

## 观察

观察从 2026-11-25 13:37:12 EST 至 2026-11-25 14:07:15 EST，服务器 UTC 记录精确覆盖 1803.641 秒。起点是独立连接查到 formal worker、进程组和正式拟合日志的时刻。T00 为起点后的补采样。GPU 列为单次瞬时读数。每个采样的原文含 job 进程、资源、时间、日志边界和最新写入时间。交付前查询独立保留。

| 观察点 | 美东时间 | 阶段 | CSV step | train/loss_step | 完整验证数及最近 global_step | val/loss | Recall@5 / NDCG@5 | GPU 显存，利用率 |
| --- | --- | --- | ---: | ---: | --- | ---: | --- | --- |
| 正式确认 | 2026-11-25 13:37:12 EST | quantizer | 尚未产生 | 尚未产生 | 0 | 尚未产生 | 尚未产生 | 4,268 MiB，0% |
| 启动后补查 | 2026-11-25 13:38:56 EST | sid | 尚未产生 | 尚未产生 | 0，最近 global_step 尚未产生 | 尚未产生 | 尚未产生 / 尚未产生 | 1 MiB，0 % |
| 上游及绑定回执齐备 | 2026-11-25 13:40:10 EST | training | 尚未产生 | 尚未产生 | 0，最近 global_step 尚未产生 | 尚未产生 | 尚未产生 / 尚未产生 | 1 MiB，0 % |
| T10 | 2026-11-25 13:47:16 EST | training | 99 | 14.45938491821289 | 0，最近 global_step 尚未产生 | 尚未产生 | 尚未产生 / 尚未产生 | 4006 MiB，7 % |
| T20 | 2026-11-25 13:57:14 EST | training | 199 | 11.876701354980469 | 1，最近 global_step 100 | 13.089644432067871 | 0.060108184814453125 / 0.03632291406393051 | 4008 MiB，7 % |
| T30 | 2026-11-25 14:07:15 EST | training | 299 | 9.923819541931152 | 2，最近 global_step 200 | 11.213273048400879 | 0.05869530141353607 / 0.04362117499113083 | 4008 MiB，9 % |
| delivery | 2026-11-25 14:08:32 EST | training | 299 | 9.923819541931152 | 3，最近 global_step 300 | 10.332762718200684 | 0.066324882209301 / 0.049337804317474365 | 4008 MiB，16 % |

CSV step 来自 logger 行索引，callback step 和 checkpoint global_step 来自 Lightning。两种值直接引用各自来源。每次完整验证 3,097 批，共 24,772 用户。最近完整验证的其他 Recall 和 NDCG 指标可直接从 CSV 的原始指标行取得。

截至观察时的 top-1 checkpoint 对应 global_step=100，监控项 `val/recall@5`，分数 0.060108184814453125。同一 checkpoint 中 Adam 计数为 [100]，覆盖 127 个有状态参数。只读提取核对了观察时 stat 身份、原 SHA-256、checkpoint 回调和优化器状态。过程指标只覆盖已经完成的验证。最终 best、三条件指标及完整流水线终态待服务器产出。

截止前可见训练 loss 及验证指标有限，训练进展、实际验证覆盖与 top-1 选模相互对应。正常配置警告保留于原日志。观察原始文件以各自打开时的文件长度为取证边界。读取顺序及边界详见采集清单，期间训练继续写出。完整审核 ZIP 包含闭合阶段文件、全量 item/SID/lookup、精确量化状态提取、共同输入、三份派生 TFRecord、训练/验证原始前缀、复核源码和取证脚本。

## 复核和接续

- 任务级修正 1 轮，范围是只读 checkpoint 取证。首次提取在反序列化已保存的原生类时遇到 import 循环。随后按正式 fit 的原生模块导入顺序初始化，仅重试一次。同一 inode、设备、大小和 mtime 的 checkpoint 提取得到上述选模及 Adam 记录。原诊断在 `best-checkpoint-collection-r0-failed/`，正式 worker 继续运行。
- 独立非作者直接检查 formal request、固定源码、已关闭上游产物、数据和观测证据。复核范围与原始复算记录见 [NONAUTHOR_REVIEW.md](NONAUTHOR_REVIEW.md)。
- 观察包没有改动实验源码或配置。发布的新增脚本负责只读采集、提取、整理及校验。未新增工具链或环境依赖。
- 继续时先查询原 job ID。已结束的阶段只在模式、执行版本和输入一致、旧 attempt 终态、阶段回执成功且产物哈希匹配时可由既有 `--reuse-job` 接续。当前 fit 入口从空 Adam 和 step 0 开始，中断后的训练需要先确认可用恢复路径和权限。
- 原作业查询命令如下。

```bash
python3 /home/addaswsw/recommend-support/sid-extensions-p0/scripts/gridctl.py --target new-server query grid-letter-rerun-20261125-formal-r0
```
