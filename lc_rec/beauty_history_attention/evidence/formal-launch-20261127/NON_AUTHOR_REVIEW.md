# LETTER Beauty / LLaMA 正式 B 启动及首次观察非作者复核

审核者 `/root/independent_observation`。我在 WSL 独立复算实际 request、授权绑定、进程和有限原始日志。21:32:41.690027 UTC 另开 Windows OpenSSH 连接对服务器作只读核验。前一轮正式版实现审核复用原 `independent/FINAL.md` 及 `ADDENDUM.md`。本记录的网页二审事实来自用户在本轮授权消息中转达。

## 计划、授权和资源

- 复算实际固定计划 SHA-256 `664694b681b92c9c46c90aa85c4123467233e86c31f087fa74d2b349e9ee9091`。对照受审发布 `bd6f7e16e6ca44d0055b28b46bc1080daed39848` 和服务器冻结执行快照 `f35a06132c9a969b04b782f8d98f32de966da947`。服务器源码清单 SHA-256 为 `34d1cb7445abb670f4e92efc127f09f585e0d67f8841e09d232e3c07855d3cdd`。预提交门禁校验 176 项冻结源码成员及所需解释器版本、原 Beauty 数据、固定基模与生成 replay。
- 从 `authorization/approval-basis.md` 核对用户原文摘录。复算两份独立门禁记录，网页 PASS 文件 SHA-256 为 `8b5f980f1f732ca5b4d7c31fae808ffb9987edddf56c9c0246314523cf66fb94`，用户授权为 `0837f5640937f34626a6d29e8850f30bcac3dfb7dddc6149c7e442d6a93ecd4c`。两份记录均匹配计划、配置、训练文件、模式、后端及源码清单，且授权文件指向实际网页记录字节。实际授权范围为 B，A′ 未获启动授权。
- 预提交独立服务器读回时，目标 RTX 4090 物理卡 1 的 UUID 为 `GPU-59cac3bd-3645-668b-79c8-839e0d878893`。设备显存总量 24,564 MiB，使用 1 MiB，计划卡无计算进程。正式 worker 掩码映射该卡至逻辑 CUDA 0。启动前工作盘余量 2,654,721,499,136 B，满足 72 GiB。门禁回执中无既有正式 lineage。主机 PATH 未发现 `squeue`、`qstat`、`bjobs`，站外实验室安排无法从这些证据独立核验。

## 接收、独立启动与生效配置

- 唯一正式 job 为 `letter-beauty-v2-history-attn-formal-b-letter-beauty-final-20261127-r1`，第 1 次 attempt。runner 在 21:31:53.341770 UTC 接收，21:31:53.439659 UTC 写入 durable claim，21:31:53.441629 UTC 进入 `RUNNING`。两份原始 request 的 JSON 完全相等，规范化 SHA-256 `d497fdb5733a5ff672385516f433d5a3964f8cbb3e4137a56971df66b48d3177` 与接受、claim 和后续 live query 一致。180 个唯一输入包含两份授权、固定计划、源码清单及 176 项源码。worker 的 `cwd`、argv、离线环境及 GPU UUID 与冻结计划相符。训练阶段上限 100,800 秒，生成 108,000 秒，CPU 7,200 秒，整个 lineage 累计 230,400 秒。
- 21:32:41.690027 UTC 直接比较服务器 `/proc`、runner 记录和训练调用。supervisor 3586579（父进程 1）、worker 3586580、训练 3586586 的启动 ticks 分别为 642045024、642045033、642045068，均无 TTY，stdin 为 `/dev/null`。supervisor 和 worker 已分别创建独立会话。训练属于 worker 会话和进程组，GPU1 的计算 PID 为该训练子进程。该时刻训练仍在初始化，计时以首次 optimizer update 的现场确认开始。
- 训练 argv 从冻结基模 `4782ad278652c7c71b72204d462d6d01eaaf7549` 开始，`resume_from=null`，没有 `--resume`，rank0 标记正式范围、seed 42、LoRA-B 初始为零、初始 step 0、实际训练参数 290,037,760。首个 optimizer update 后报告记录了 512 次 BF16 attention 层调用，使用 `EFFICIENT_ATTENTION only`。训练实参和 DeepSpeed 有效配置核对为 BF16、单卡、micro-batch 8、累积 8、有效 batch 64，4 epochs、8,214 updates。计划首个完整验证边界在 step 2053，验证轴为 44,726 条。
- 独立比较 `run/training-start/training/events.rank0.jsonl` 与上一轮原始 `inherited/actual-run/train-r1/training/events.rank0.jsonl`。前 20 个八行 micro-batch 的 step、epoch、loader 位置、ID/label/历史 SID mask 的 SHA-256、tensor shape 和每行历史 token 数均逐项一致。覆盖范围为观测到的起始前缀。

## 首次正式观察

**PASS。** 首个实际 optimizer update 的事件时间为 21:33:21.144596 UTC。窗口从独立 observe 首次读到该事件的 21:33:23.566942 UTC 开始，到 T30 查询 22:03:25.835914 UTC，共 1,802.268972 秒（30.0378 分钟）。模型加载和 optimizer 初始化发生于窗口开始之前。T10 和 T20 正式查询分别距基线 602.560001 秒、1,202.185494 秒。

| 观察 | UTC | 最近记录 step / 8,214 | 已完成 micro-batches，按 step × 8 | epoch | 训练 loss | global grad norm | LR | 卡上显存 MiB / 瞬时利用率 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| T10 | 21:43:26.126943 | 80 | 640 | 0.038958 | 4.7696 | 6.111521244 | 0.0001927710843373 | 18862 / 54% |
| T20 | 21:53:25.752436 | 170 | 1360 | 0.082785 | 4.0444 | 2.345800877 | 0.0001999435089771 | 18862 / 29% |
| T30 | 22:03:25.835914 | 260 | 2080 | 0.126613 | 3.8221 | 2.801572084 | 0.0001997662459786 | 18862 / 28% |

- 逐份核验 `run/training-start`、`run/T10`、`run/T20`、`run/T30` 的 `capture.json` 中记录的 17 个成员边界及 SHA-256。比较原始 work.log、supervisor.log 和事件文件的附加前缀。T30 事件限于 `[0, 39446)` 字节，SHA-256 `f07b18120e8f707b2e551debefe857cecd17445ffc74802271c1a36c1c1e9dca`，92 条完整 JSONL 事件中的 60 条非 batch 事件与同一边界的 work.log 结构化原文逐项一致。T30 最新 loss 事件为 22:02:54.940908 UTC。观察间隔内的同一 Linux boot、每个 PID 的启动 ticks、无 TTY、stdin、cwd、argv、进程组及 GPU 归属仍与启动记录相符。
- T30 累计有 step 10–260 的 26 组每十步 optimizer 和 loss 观测。逐组核对有限训练 loss 与 global grad norm，以及 83 更新 warmup 后的 cosine LR。此前后日志中的 loss 范围 3.8166–12.9115，global grad norm 范围约 2.206104–62.638107，相邻十步记录间的最长耗时约 72.25 秒。计算 T10 与 T30 对应真实事件间的 180 次更新除以 1,231.201019 秒，得到约 526.315 updates/h、1.170 micro-batches/s。按该窗口速度外推，剩余 update 计算约 15.11 小时。四次正式验证、保存、严格选模与导出开销仍缺本次正式实测。
- 观察中正式训练子进程在计划卡上占用 18,852 MiB，三个定时采样的卡上总占用均为 18,862 MiB。瞬时利用率为 54%、29%、28%，同时连续的 optimizer 记录给出了实际推进。T30 盘余 2,654,720,208,896 B。T30 的 job 仍为同一 attempt 的 `RUNNING`，runner 及工作进程存活，GPU1 上仅该训练 PID。
- work.log 出现既有 PEFT API deprecation、PyTorch checkpoint `use_reentrant` 和 bitsandbytes 内部量化转换提示。supervisor.log 为空。原始日志内无 traceback、非有限训练数值、OOM 或重启迹象。DeepSpeed rank 报告中的 `steps_per_print=Infinity` 为打印间隔配置。
- 本地观察采样器最初使用错误的 baseline 文件名，原始 `FileNotFoundError` 保留在 `transport/sampler-initial-invocation.stderr`。在 T10 前改用实际 `observations/startup-213322.json`，仍由同一条首次有效观察时间安排 T10、T20、T30。同一 job 的 runner、训练和持续记录无中断。窗口后本地汇总器读取全新训练 rank0 的顶层 `checkpoint_resume` 时发生 `KeyError`，见 `transport/summary-initial-invocation.stderr`。修正汇总器后直接核验训练选择文件的 `resume_from=null` 和实际 argv 无 `--resume`，原始捕获字节和时间戳保留。两次修正均位于本地记录脚本的调用或解析，正式执行快照内容保持原版本。
- 截至 T30，初次完整验证边界 step 2053 尚未到达，训练目录已有 rank、tokenizer 和事件记录，尚未产生正式验证 loss、epoch checkpoint、选中 adapter、正式生成输出及推荐指标。此次观察结论限于正式 B 成功进入训练，并在上述窗口持续推进。服务器状态结论截至 T30 取证时刻，主动观察在该边界结束。

复核于 2026-11-27 完成（America/New_York）。复核使用已有固定证据与轻量只读查询，无新增 GPU 计算。
