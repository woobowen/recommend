# LETTER Instruments → GRID 原生 Normal 正式启动与 30 分钟观察

**正式启动通过，七点观察通过。** 原 `grid-letter-instruments-normal-20261124-formal-r0` 的单卡训练与完整验证已见实测推进。T+30 后已停止定时观察，服务器正式作业继续运行。完整正式结果待产出。

worker 启动 2026-11-24 13:36:30.954 EST / 2026-11-24 18:36:30.954 UTC。T+30 实测 2026-11-24 14:06:32.284 EST / 2026-11-24 19:06:32.284 UTC，采样时为验证。额外一次只读交付查询为 2026-11-24 14:06:47.114 EST / 2026-11-24 19:06:47.114 UTC，同一作业 `RUNNING`，为验证，最近训练事件 step 400 / 6400 microbatch，最近完成验证为 step 300。完整测试、候选导出和离线评价待执行。 [七点表](timeline.md) · [原始快照](snapshots/) · [交付查询](delivery-status.json)

## 全量验证与 best

| 训练更新 / CSV step | 验证结束 EST / UTC | 批数 | 验证秒数 | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| 100 / 99 | 13:45:29.136 / 18:45:29.136 | 3097 | 318.300 | 0.06297432631254196 | 0.08017116039991379 | 0.04039168730378151 | 0.04598501697182655 |
| 200 / 199 | 13:54:05.748 / 18:54:05.748 | 3097 | 313.641 | 0.07532697916030884 | 0.0892539992928505 | 0.05763069540262222 | 0.06215467303991318 |
| 300 / 299 | 14:02:37.963 / 19:02:37.963 | 3097 | 313.739 | 0.07916195690631866 | 0.09518811851739883 | 0.06466443836688995 | 0.0698271319270134 |

T+30 确认 best 为 step 300，`val/recall@5=0.07916195690631866`，文件 `checkpoint_epoch=000_step=000300.ckpt`，159,007,470 字节。观察中已写出的训练 loss 和验证指标均为有限值。工作日志包含既有 dataloader worker 数提示。七点、验证继续推进与当前产物支持本阶段运行通过。正式效果和训练收敛留待后续结果审核。 [原始日志](captured/work.log) · [原始 CSV](captured/output/run/csv/version_0/metrics.csv) · [原始事件](captured/output/run/events.jsonl)

## 资源与估时

T+30 按一张分配设备的 worker 墙钟累计 1801.330 GPU-device-seconds，剩余 84598.670 秒。一次交付查询记录 1816.161 秒，剩余 84583.839 秒。正式阶段没有失败 GPU attempt 或新增 CPU runner 作业。准备阶段 273.744618 GPU-device-seconds 继续按独立账本保存。七点和交付查询的服务器、本机暂存以及 Windows Downloads 均保留至少 2 GiB 可用空间。 [预算与估时](budget.json)

初始化到 `train_start` 约 10.233 秒。现场初始周期的训练中位数约 204.4 秒／100 更新，全量验证中位数约 313.7 秒，合计约 518.1 秒／周期。以 4,000／6,400／9,000 updates 为估时情景，加沿用先前外推的测试、写出和离线耗时，对应启动后约 6.09／9.54／13.28 小时。实际停止由已审早停和框架上限决定，阶段速度和测试／离线耗时尚有不确定性。runner 的 86,400 秒时限到达后按超时中止与清收记录。 [估时复算脚本](budget.py)

## 授权与实际方案

本次一次性授权包含一张 RTX 4090、一个正式训练作业、30 分钟在线观察和阶段证据交付。采用网页二审发布 `098677ab130eae4dfb14bae5f2c56922817d8183`，运行代码为 `390df9f508b7dc11eaa57b79a5a91fb5a7f99385`。提交载荷与固定模板相同，未变更 GPU UUID、运行参数或输出路径。服务器入场重新核对 104 项运行文件和九项数据资产，实际 effective config、overrides 与冻结 dry-run 同字节。 [实际请求](request.json) · [服务器请求](server-request.json) · [现场准入](admission.json) · [实际有效配置](captured/output/run/effective_config.yaml) · [已审入口](https://github.com/woobowen/recommend/tree/098677ab130eae4dfb14bae5f2c56922817d8183/experiments/letter_instruments_grid_normal)

LETTER Instruments 使用四位 SID、原划分的 24,772 人与 9,922 件商品。GRID 原生模型实测可训练参数 13,128,064，保留共享 SID embedding、BOS/SEP、四个预测头。训练从 seed 42 的随机初始化和空 Adam state 开始，`ckpt_path=null`。单卡训练用户 batch 16、原 causal duplicate collator、每 16 microbatch 更新、FP32、matmul medium、确定性算法、`CUBLAS_WORKSPACE_CONFIG=:4096:8`。Adam 学习率 0.001、weight decay 0.0001、betas 0.9/0.999、eps 1e-8，沿用原无 scheduler 行为。 [模型启动回执](captured/output/run/model.json) · [依赖与数据回执](captured/output/run/startup.json) · [原始事件](captured/output/run/events.jsonl)

每 1,600 个训练 microbatch，即每 100 次 optimizer update，完整验证 3,097 批。按 `val/recall@5` 最大值留一个完整 best，patience 10，框架上限 320,000 updates 和 10 trainer epochs。训练源使用循环 iterable，trainer epoch 与 TFRecord 全量遍历分别记账。训练按原条件结束后，入口严格重载 best，以 eval seed 2026 进行一次全体 24,772 人 Normal 测试。四步 beam 10 保留原概率乘积、cache、前缀过滤关闭和尾批 4 人，预期 247,720 个候选。从同一次候选计算原生 Recall/NDCG@5/10、统一 SID Hit@1/5/10 与 NDCG/MRR@5/10，以及统一 Item／CCE ItemHit/ItemNDCG@5/10。 [固定运行入口](https://github.com/woobowen/recommend/blob/390df9f508b7dc11eaa57b79a5a91fb5a7f99385/experiments/letter_instruments_grid_normal/run_normal.py)

## 独立接管与接续

`new-server` 实际主机 `zwlab-01`，账号 `bwwu`，UID 1005。原 job 保存于 `/home/bwwu/grid-audits/sid-extensions-p0/jobs/grid-letter-instruments-normal-20261124-formal-r0`，正式输出在其 `output/run`。supervisor PID 3467516，worker PID 3467518，worker PPID 3467516，supervisor PPID 1。两进程各自的 SID、PGID 与自身 PID 相同，TTY 号均为 0，stdin 指向 `/dev/null`，stdout/stderr 指向各自的服务器日志。新建 SSH 连接时原进程仍活跃，PID、boot ID、start ticks 和 request digest 匹配。实际 RTX 4090 为物理 1，唯一可见 CUDA 设备为逻辑 `cuda:0`，UUID `GPU-59cac3bd-3645-668b-79c8-839e0d878893`。其余 GPU 用于上下文占用观察，未归属本任务的计算进程 PID 在发布快照中隐去。 [接受回执](submission.json) · [T0](snapshots/T00.json) · [启动后重连](snapshots/start-reconnect.json) · [部署标识](runner-deployment.json)

启动命令由服务器 runner、固定快照和当地解释器执行。源码、资产、环境和输出均在服务器，后续训练、验证、严格重载、一次 Normal、候选导出和离线计算可以在本地 Codex、SSH、WSL 和 VPN 关闭后继续。以后查询或诊断需要重新打开 Codex 并显式请求。请求使用既有 `gridctl.py` 和 Windows OpenSSH，`query` 和 `logs` 为现场已核对应的只读入口。活动阶段证据使用只读采集器，现有 runner 的 `fetch` 用于终态清收后的产物导出。

```bash
python3 -B /home/addaswsw/recommend-support/sid-extensions-p0/scripts/gridctl.py \
  --target new-server query grid-letter-instruments-normal-20261124-formal-r0
```

## 记录口径与修正

观察以真实 worker 内核启动时刻为基准。每个采样 JSON 记录服务端开始与结束时间、UTC 和 New York 时间、目标时间、偏差和 WSL 查询边界。各文件分别读取，整点快照跨越一个短采集区间。事件的 optimizer step 和 microbatch 来自最近一条 `train_batch`，其时间单独记录。CSV 单行有 step/epoch，采样用文件 mtime 辅助判断新鲜度。原生 `train/loss_step` 是 Lightning 对 `train/loss` 步级记录生成的列。正式模式没有预跑专用的 loss 和梯度探针字段。原实现无逐验证 batch 的持久计数，验证阶段参考开始、首批、结束事件与 GPU 活动。

启动前的第 1 轮工程修正处理了读取 CLI 帮助产生的本机 `__pycache__` 和准入脚本中的 YAML 批量字段名称。生成的字节码移到本轮暂存区，准入改用原 `batch_size_per_device` 字段后通过。服务器只读探查确认设备/路径时，初始文本命令发现远端缺少 `rg`，后续用既有 Python 标准库核对。该轮只影响准入和采集。正式作业为 r0，一次接受、一次启动，训练配置及运行身份沿用固定版本。未新增系统包、语言包、工具链或环境配置。

采集器以打开文件时的 inode、mtime 和大小确定字节上界，采集始末见 [捕获清单](captured/capture-manifest.json)。checkpoint 阶段证据来自保存日志与文件状态，权重留在服务器。观察截止后取消本地定时采样，保留服务器 supervisor、worker 和限时清收机制。额外一条交付核查单独记录。 [观察脚本](observe.py) · [只读采集器](capture.py) · [阶段校验器](verify_phase.py) · [非作者审核](NON_AUTHOR_REVIEW.md)
