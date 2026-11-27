# LETTER Beauty / LLaMA B 正式启动及初始观察

**正式 B 已从固定基础模型和原初始化启动。** 2026-11-27 美东 16:33:23 至 17:03:25，观察 30 分 02 秒，同一 attempt 在第 1 个 epoch 内持续更新。末次观测为 RUNNING，runner 持续独立运行。主动观测在该点结束。以下均为本轮正式值或明确标为历史估计的依据。

## 作业和有效条件

| 项目 | 本轮实际记录 |
| --- | --- |
| job ID | `letter-beauty-v2-history-attn-formal-b-letter-beauty-final-20261127-r1` |
| attempt 与接受时间 | 1，2026-11-27 16:31:53.344 EST |
| 独立 SSH 重连 | 16:32:05.555 EST，已为 RUNNING，模型加载中 |
| 真实训练 | 16:33:12.943 从 step 0 开始。16:33:14.453 首个真实 forward/backward，16:33:21.145 首个 optimizer update |
| 进程 | supervisor 3586579，PPID 1。worker 3586580，训练子进程 3586586。三者无 TTY，stdin 为 `/dev/null`，stdout/stderr 落服务器日志 |
| 正式阶段 | 4 epoch、8,214 updates、65,712 micro-batches，验证点 2,053、4,107、6,160、8,214。BF16 efficient SDPA、8×8=64，seed 42 |
| 启动方式 | 固定基座 revision `4782ad278652c7c71b72204d462d6d01eaaf7549`，首次训练 `resume_from=null`，实际 argv 无 `--resume` |
| 接受的规范化 request SHA-256 | `d497fdb5733a5ff672385516f433d5a3964f8cbb3e4137a56971df66b48d3177` |
| 网页已审发布版本 | `bd6f7e16e6ca44d0055b28b46bc1080daed39848` |
| 执行冻结版本及计划 | `f35a06132c9a969b04b782f8d98f32de966da947`，计划 SHA-256 `664694b681b92c9c46c90aa85c4123467233e86c31f087fa74d2b349e9ee9091` |

启动前同一 lineage 的 attempt、作业和输出均未出现。现有门禁验证了两份新的独立 JSON，网页 PASS 来源于用户传达的实际审核结论，正式授权来自用户本轮启动要求。计划、授权、网页记录及冻结依赖的实际请求绑定见完整 ZIP。上一轮未授权记录保持原样。

现场验证了 176 项冻结源码、固定数据和 SID 哈希、replay 哈希、已审核依赖版本、模型及 tokenizer。固定模型两分片沿用已核验 SHA，此次按固定路径、文件大小及纳秒 mtime 复核。冻结计划的物理 GPU 1、逻辑 CUDA 0 对应 RTX 4090 UUID `GPU-59cac3bd-3645-668b-79c8-839e0d878893`，现场首启 1 MiB，无计算进程。GPU 0 有其他计算进程。主机没有可用的 `squeue`、`qstat` 或 `bjobs` 查询入口，现场依据是本轮指定 GPU 的使用授权和实时进程，runner 再次核验空闲。工作盘首启可用 2,654,721,499,136 B，超过 72 GiB。已有最大 checkpoint 实测约 19.26 GB，三个短时共存约 57.8 GB。预测块及最终导出短时双存约 4.5 GB，余量亦覆盖 adapter、日志与 CPU 产物。

## 健康观察

步数采用每 10 步的 optimizer 日志。已完成 micro-batch 等于该记录步的累计 updates ×8，表示采样时已证实的完成下界。epoch 是 Trainer 的累计 epoch 进度。区间速率从前一个表中确认的 optimizer 事件算至本点事件。

| 美东时间 | optimizer step / 8,214 | 已完成 micro-batch | epoch | train loss | 全局 grad norm | LR | updates/小时 | GPU 已用／利用率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16:33:23 | 1 | 8 | 第 1 个 epoch | — | — | — | — | 17,754 MiB／76% |
| 16:43:26 | 80 | 640 | 0.0390 | 4.7696 | 6.111521 | 1.927711e-4 | 524.15 | 18,862 MiB／54% |
| 16:53:25 | 170 | 1,360 | 0.0828 | 4.0444 | 2.345801 | 1.999435e-4 | 526.96 | 18,862 MiB／29% |
| 17:03:25 | 260 | 2,080 | 0.1266 | 3.8221 | 2.801572 | 1.997662e-4 | 525.68 | 18,862 MiB／28% |

step 10 首条训练汇报为 loss 12.9115、grad norm 62.638107、LR 2.409639e-5。到 step 260 有 26 组相邻每 10 步的有限 loss、梯度和学习率。学习率按 83 步 warmup 后进入 cosine。首个更新到最后记录的 259 次更新用时 1,773.795 秒，约 525.65 updates/小时和 1.168 micro-batch/秒。训练事件和工作日志持续增加，同一训练 PID 是冻结 GPU 上唯一计算进程。最后可用盘 2,654,720,208,896 B。

采集前缀中没有 traceback、非有限 loss/grad、OOM、重启或持续停滞。启动日志有已知兼容提示，涉及 NCCL 初次 barrier 的逻辑设备、tokenizer legacy、PEFT 弃用、gradient checkpointing `use_reentrant` 和 bitsandbytes 内部量化转换。已完成的 BF16 forward/backward 和更新均来自本轮训练。历史配置里的 DeepSpeed 打印间隔哨兵 `Infinity` 属于日志配置。

本轮没有服务器训练失败、attempt 恢复或正式源码修正。WSL 本地观测采样器第一次调用将基线文件名误写为 `startup-213323.json`，在首个间隔采样前改用实际的 `startup-213322.json`，真实基线及后续点未挪动。观测后的本地汇总首次索引了首启缺省字段 `checkpoint_resume`，随后以真实 `resume_from=null` 和实际 argv 检查并复算通过。两段本地诊断保留在 ZIP 的 `transport/`，原始采集未改变。本轮未安装包、工具链，也未改持久环境配置。

已写出正式 runner 请求与接收状态、训练阶段和有效配置、tokenizer 派生文件、事件与工作日志。末次采样时未到 step 2,053，因此尚未产生正式验证 loss、epoch checkpoint 或选定 adapter。五条件全量生成与正式推荐指标、配对统计尚未产生。当前结论限于正式训练启动和观察窗口内推进。A′ 未启动。

## 时间估计、后续查询和审核

按真实初段速率估计末次观测后剩余约 15.13 小时更新计算，再加已审短测估计的验证／保存约 1.97 小时，训练粗估美东 11 月 28 日 10:09 完成。全量生成仍按短测约 18.01 小时、CPU/I/O 收尾约 10–25 分钟估算，整链粗估 11 月 29 日 04:20–04:35 美东。后一区间仅反映 CPU 预留区间，真实耗时还受全量验证、保存、生成、I/O 与运行条件影响。既有约 36.4 单卡小时是短测外推，42–46 小时是资源建议预留。冻结阶段上限仍为 28/30/2 小时，同系列累计硬上限 64 小时，计入失败 attempts。

后续接手时，在 `new-server` 查询当前原 job，先核对终态与同一进程／产物谱系再判断恢复：

```bash
bash /data/bwwu/lc-rec/beauty-history-attention-final-20261127/inputs/formal-b-launch-20261127/formal-entry.sh observe 1
```

输出根为 `/data/bwwu/lc-rec/beauty-history-attention-20261127/formal-b-letter-beauty-final-20261127`。runner 状态和工作日志位于 `/home/bwwu/grid-audits/sid-extensions-p0/jobs/letter-beauty-v2-history-attn-formal-b-letter-beauty-final-20261127-r1`。若终态确需恢复，核实本 attempt 已终止并清收，按固定计划的同谱系 checkpoint 或完整预测前缀，采用已审 `formal.py` 入口和独立下一 attempt。窗口结束时无需恢复。

发布证据包含阶段、接收、两份门禁记录、各点观测、带时间边界的训练事件、非作者报告和同步工具。完整 ZIP 还含实际请求和 argv、进程和文件边界、原工作日志、执行源码、必要依赖及远端固定提交回读。工作日志和训练事件从字节 0 读取至各文件打开时的可见大小，T30 的两份原文及提取范围见 ZIP。 [机器摘要](window-summary.json) · [T30 原始观测](T30.json) · [非作者复核](NON_AUTHOR_REVIEW.md) · [辅助工具审核](PUBLICATION_HELPERS.md)
