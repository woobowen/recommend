# 第三组 G3 Beauty Shuffle V2 正式启动与 30 分钟观察

**THIRD_GROUP_FORMAL_RUNNING_30MIN_OBSERVATION_PASS**。已审正式父控制器在服务器接管四项串行生成、逐项 CPU 全量核验与评价及最终五项配对。七点观察结束。正式组最终指标与结果终审留待全量任务完成。

## 冻结方案与启动链

G3 Beauty 22,363 用户、12,101 件商品、12,101 个唯一完整 SID、零碰撞。G3 tokenizer 为 32,786 总词表、786 个新增 SID token。沿用原 FORMAL LLaMA-7B best adapter，验证 loss 选中 checkpoint-6160，以冻结 G3 单 rank Normal 及既有 CPU 评价为对照。正式 S2、S3a、S3b、S4 的活动槽从 0 编号为 (2,3)、(0,1,2)、(1,2,3)、(0,1,2,3)，均允许恒等，Shuffle seed 42。完整 SHA-256、紧凑 JSON、拒绝采样、字典序目录与全局用户／历史出现轴沿用已审 V2。

原 SeqRec Prompt0、最近至多 20 件历史、完整 SID token ID 组内置换和逐行 CPUOracle 绑定生效。模型在当前位置保持原始 attention mask 和空初始 cache。单 GPU，WORLD_SIZE=1，RANK=LOCAL_RANK=0，batch 1，beam/return 20，最多 10 个新 token，确定性 beam，early stopping 和 cache 生效。沿用冻结 BF16 base/embedding/lm_head 与 FP32 LoRA。候选保留原逐码位约束、EOS、原精度分数、排序及目录外候选处理。评价沿用 16 项指标与 CCE。四项减冻结 Normal 及 S3b−S3a 共用 5,000 次 PCG64 抽样、seed 20261113 和线性 95% 区间。

原训练 21d495b0be5b606853ec7cec68a3eaa1b58acbd7，GPU 86660be5a62c8325caf74cf2810581850ca12803，CPU 与正式控制器 cb3b1f36d69ef0c1a7f554ef5455026523635f78。启动前验证两个部署、必要 CPU 输入与模型身份，沿用原计划 GPU UUID 和物理卡 1。冻结计划保持原字节和占位符。实际父请求恢复分离的部署输入，另绑定计划和独立授权记录。首个 S2 子请求由父控制器创建。 [现场核对](evidence/prelaunch.json) · [实际计划](evidence/formal-plan.json) · [本次授权](evidence/authorization.json) · [实际父请求](evidence/parent-request.json) · [服务器原件回读](evidence/server-startup/readback.json)

父作业为 g3-beauty-v2-group-formal-r0，首子为 g3-beauty-v2-gpu-formal-shuffle2-r0。T0 与新 SSH 会话都查到同一组进程。父监督／控制器 PID 为 3427705／3427707，S2 监督／torchrun／rank 0 为 3427748／3427749／3427779。服务器监督进程独立 session、stdin 为 /dev/null，进程由服务器接管，日志持续写入服务器。本轮只有 rank 0 持有本任务 CUDA context。 [独立启动核验](evidence/review-launch/startup-review.json)

## 七点观察

America/New_York 当地时间为 UTC−05:00。完整用户以采样打开文件时固定字节边界内的完整换行 JSONL 为准。增量逐行与正式 sampler 对照，末尾半行若出现留待后续。Rank 报告每 25 条更新。表内 GPU 利用率是瞬时值。每点原始 JSON 另存父子状态、进程身份、UTC、文件 mtime、日志范围、可用空间和分项账本。 T+25 的 worker 日志在读取文件边界之后更新到 2,125 条，该点完整文件边界仍为 2,124 条。

| 点 | 纽约时间 | 阶段 | 完整用户 | 新增 | 剩余 | 预测文件字节 | 预测新增字节 | S2 日志新增字节 | MiB / 瞬时利用率 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| [T00](evidence/observations/T00.json) | 10:30:03 | 启动／加载 | 0 | 0 | 22,363 | 0 | 0 | 0 | 1 / 0% |
| [T05](evidence/observations/T05.json) | 10:35:22 | 生成 | 238 | 238 | 22,125 | 5,763,095 | 5,763,095 | 2,975 | 21,848 / 46% |
| [T10](evidence/observations/T10.json) | 10:40:42 | 生成 | 741 | 503 | 21,622 | 18,330,907 | 12,567,812 | 3,394 | 21,848 / 28% |
| [T15](evidence/observations/T15.json) | 10:45:36 | 生成 | 1,202 | 461 | 21,161 | 29,881,017 | 11,550,110 | 3,221 | 21,848 / 41% |
| [T20](evidence/observations/T20.json) | 10:50:30 | 生成 | 1,672 | 470 | 20,691 | 41,332,550 | 11,451,533 | 3,063 | 21,848 / 41% |
| [T25](evidence/observations/T25.json) | 10:55:16 | 生成 | 2,124 | 452 | 20,239 | 52,539,549 | 11,206,999 | 3,245 | 21,848 / 7% |
| [T30](evidence/observations/T30.json) | 11:00:09 | 生成 | 2,587 | 463 | 19,776 | 64,074,085 | 11,534,536 | 3,076 | 21,848 / 41% |


## 预算、估算和交付前复核

七点实际跨度 1806.087 秒。各点同一作业身份稳定，未发现异常退出、OOM、超时或输入守卫失败。T+30 的 rank 报告为 2,575 人。最低可用输出空间 2.699 TB。

- 正式 GPU device-seconds。T+30 已用 1810.661295，余 127789.338705。
- 预跑 GPU device-seconds。T+30 已用 812.994045，余 9987.005955。
- CPU worker 秒。T+30 已用 260.875971，余 28539.124029。
- 父监督墙钟秒。T+30 已用 1812.565310，余 124187.434690。

严格装载实测 157.90 秒。按 T+5 至 T+30 记录增量计得 1.580 用户/秒，避开模型加载。T+30 时 S2 预计还需 3.48 小时。按准备阶段各模式耗时比例、实测 S2 速率折算，S3a/S3b/S4 约 3.57/3.94/3.61 小时。整组预计还需 14.60 GPU 小时。单卡争用、输入长度及写盘会改变耗时。每项 CPU 评价另计几十秒至数分钟，配对另计数分钟，最终审核和交付另估 20–45 分钟。

另在 2026-11-22T11:00:12.164263-05:00 完成交付前只读复核，S2 已有 2,591 条完整预测，父控制器仍正常推进。 [交付前复核](evidence/delivery-check.json) · [机器可读汇总](evidence/stage-summary.json) · [七点验收](evidence/window-complete.json)

## 首六条与非作者核验

按实际输出前六条固定取样，row_idx 为 8755、21472、6726、17283、12152、4046。57 个历史组、120 个原精度候选通过主控与非作者检查。核对 CPU 输入、固定槽、组外 token、mask、原生位置、空 cache、20-beam prefill、实际 BF16 embedding 回执、FP32 有限分数、EOS、码位和稳定排序。六条均为正式 S2，rank 的完整分配轴为 22,363 人。正式首批记录的证据范围为六条。 [原始首六条](evidence/first-six.jsonl) · [真实 rank 和截取边界](evidence/first-six-capture.json) · [首六条非作者核验](evidence/review-launch/first-six-review.json) · [主控验证](evidence/verified-stage.json)

日志保留 tokenizer legacy 和 NCCL barrier 设备映射提示。CUDA 进程实测匹配原计划卡，完整记录持续增长。T0 后本地采集器改用 torchrun 的直接子进程判据，区分继承 rank 环境的 CPU DataLoader helper。T0 原快照和原采集器保留，服务器计算进程持续沿用同一身份。 [采集器接续](evidence/observer-continuation.json)

T+5 与 T+10 的服务器间隔比本地名义 300 秒长约 19 秒，随后用本地 UTC 绝对截止和短等待校正取样。保留前三点原时间，整组模型进程持续运行。 [时间校正回执](evidence/observer-clock-correction.json) · [独立七点审核](evidence/review-launch/observation-review.json) · [原预测字节范围复核](evidence/review-generation/live-byte-range-review.json)

## 接续与停止边界

输出位于 /data/bwwu/lc-rec/g3-beauty-v2/。优先使用以下已验证查询入口核对原父作业，并读同一 job 的原日志和控制器 summary。控制器在阶段失败或清收未通过时停止后继，保留失败原件。Runner 查询不会提交作业，检测到控制丢失时可能保存观察回执。

```bash
/mnt/c/Windows/System32/OpenSSH/ssh.exe -o BatchMode=yes new-server \
  "/usr/bin/python3 -I -S -B /home/bwwu/grid-audits/sid-extensions-p0/scripts/runner-g3-beauty-v2-107b340e5ccfdc8a.py query /home/bwwu/grid-audits/sid-extensions-p0/jobs/g3-beauty-v2-group-formal-r0"
```

到本阶段终点后，服务器计算按原授权继续，Codex 的主动定时观察停止。等待完整四条件、CPU 评价、统计、非作者结果终审、GitHub 结果发布以及网页终审。
