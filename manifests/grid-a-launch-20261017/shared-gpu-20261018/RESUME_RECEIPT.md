# 共享 GPU 调度接续与终态回执

结论：**PARTIAL_RESUME**。共享调度已真实生效，A1全依赖链完成；A2遇到独立DataLoader文件描述符错误，已保护中断并阻断下游。本回执不是科研结果验收。

最后一次服务器原件查询：2026-10-18T01:35:40.201318+08:00。34阶段：23 SUCCEEDED、1 BLOCKED_NO_RETRY、10 BLOCKED_DEPENDENCY。实际提交24个研究阶段job：23 SUCCEEDED/0，A2量化INTERRUPTED/-9；10个下游均NOT_SUBMITTED，无真实job或exit code。全部24个job及两个controller已核验没有存活worker、supervisor或工作组，cleanup_verified均为true；没有仍在写出本项目结果的worker。

## 四条工作流

| 工作流 | 实际原job/入口 | 当前状态 |
|---|---|---|
| Toys A-5300 | ga-20261017-toys5300-stats-v1（全部7个原子job见terminal-jobs-summary.json） | 三评测、三注意力、统计均SUCCEEDED/0；COMPUTATION_COMPLETE_REVIEW_PENDING |
| Toys A-5000 | ga-20261017-toys5000-stats-v1（全部7个原子job见清单） | 同上；未重跑 |
| Sports A1 | ga-20261017-sportsa1-train-v1；eval-0/3/4、attention-0/3/4、stats原ID见清单 | 训练step9600正常patience10早停、best step8600；三评测、三注意力、统计均SUCCEEDED/0；COMPUTATION_COMPLETE_REVIEW_PENDING |
| Sports A2 | ga-20261017-sportsa2-encode-v1、ga-20261017-sportsa2-quantize-v1 | 编码SUCCEEDED/0；量化INTERRUPTED/-9，cleanup已验证；未生成有效quantizer/SID，10个依赖未提交并阻断 |

原始终态与退出码逐个保存在 [runtime/terminal-jobs-summary.json](runtime/terminal-jobs-summary.json)，完整查询原件、实际request/accepted/status/cleanup对象及输出内容身份见 [runtime/terminal-inventory.json](runtime/terminal-inventory.json)。CPU统计RESULT较大不在本轮重新解析，其路径/大小/SHA在真实终态artifacts中保留；本轮不审核指标或注意力科研结论。

## 原配置错误与新准入

原allow_shared未设置时，resource_ok以其他compute process存在直接否决；旧共享分支还以peer显存<=2048MiB、util<=30%作额外限制。它没有落实用户允许按真实资源共享的要求。

新源码：`experiments/launch_20261017/queue.py` 的gpu_snapshot/resource_assessment/prelaunch_samples/allocation_stage，以及 `experiments/launch_20261018_shared_gpu/build.py` / `resume.py`。对所有尚未提交的GPU阶段启用需求共享：

| 阶段 | 预计峰值MiB | 余量MiB | 基础真实free要求MiB |
|---|---:|---:|---:|
| 评测/注意力 | 2048 | 1024 | 3072 |
| 双卡RQ，每卡 | 6144 | 2048 | 8192 |
| Sports训练/准入 | 20480 | 2048 | 22528 |

逐卡3次、间隔2秒真实采样；每次free达标且启动前util<=50%，并把观测显存波动增加到需求上。没有peer进程存在性拒绝或固定peer显存上限。训练峰值以A1实测reserved19692MiB加CUDA上下文保守取整；RQ是基于原distance分块/张量规模的静态保守包络，未冒称实测峰值。单卡可选原两UUID之一；RQ仍固定原两卡、BF16、raw2048/card、30步、3x256及第四位消歧，不改单卡/降batch/换方法。运行时按本项目进程组扣除已占显存，检查剩余峰值+余量（最少512MiB），保留锁、预算、空间、超时和失败阻断。

三次启动前实测为GPU0 free23564MiB/util20%，GPU1 free23398MiB/util15–17%，两卡均有yychen进程。RQ stage-entry还保存另一组真实采样：GPU1的yychen PID1965921/676MiB存在时仍获准启动。不能把存在同学进程当作拒绝理由。

A1 Shuffle3原job于01:26:14接收，wrapper PID1973602、compute PID1973629、GPU1；Shuffle4于01:26:19接收，wrapper1973628、compute1973669、GPU0。01:26:41分别1544/520样本；01:28:47分别21000/19464；01:31:31均真实SUCCEEDED/0。后来自动接续三组fixed32注意力和CPU统计。有效共享观察区间 2026-10-18T01:26:41.579401+08:00 至 2026-10-18T01:35:40.201318+08:00（约8分59秒），包含正常完成的后继阶段，不要求每个job持续运行。没有观测到A1 OOM/NaN或模型错误；pmon保留同学进程继续计算的证据，不据此宣称其吞吐零下降。

## 安全接续与A2保护

- 原CPU控制器ga-20261017-queue-v1在01:17:36.931仅对精确worker1762403发pidfd SIGTERM；01:18:00完成STOPPED_NO_NEW_SUBMISSIONS/outer FAILED2/清理与锁释放。这是受控替换退出。旧源码、配置、已接受request均未改。
- 停稳最终清单17个已接收原job：16成功、A1 Normal独立运行。旧队列可能在SIGTERM时完成当前循环，故重新读完全部34个原ID并证明没有SUBMITTING或不明目录后才接续；Normal原job随后自然SUCCEEDED/0，没有重提。
- 新CPU控制器ga-20261018-shared-queue-v1于01:23:17接收，supervisor1968507、worker1968508；独立session/DEVNULL/服务器runner及周期状态，不依赖WSL/SSH/Codex。最后SUCCEEDED/0且清理完成，逻辑状态TERMINAL_PENDING_RESULT_REVIEW仅表示编排终态，不表示研究全成功。
- A2原量化job首次实际接收于01:23:26，wrapper1968688、科研主进程1968703、第二rank1968802，原双UUID真实可见；没有重跑embedding。
- 原始日志在保护前已出现24次`OSError: [Errno 24] Too many open files`；多个DataLoader worker FD=1023/soft1024，另一些进程soft524288。没有progress.json，没有有效量化步进或SID/quantizer.ckpt。不能把GPU上的RUNNING当作30步成功。
- 01:25:55.217仅向精确A2 supervisor1968687发pidfd SIGTERM，既有runner清理其UID1005/pgid与session1968688自有组，最终INTERRUPTED/-9、cleanup true、group空。没有向新controller、A1或yychen发停止信号。29条runner清理事件与原始before/after/limits/log均已保留；未作文件描述符修复，未自动重试。
- A2进入BLOCKED_NO_RETRY，其10个下游NOT_SUBMITTED/BLOCKED_DEPENDENCY。A1两评测及后继独立完成。当前所有34阶段已终态，停止本地监视；不继续15分钟轮询，也不再提交。

## 资产与不可变版本

- 科研原提交 `b44399926c98369191c2ff5c2425377c8c07d3c8`，旧快照 `snapshots/b6203414ee58a400`；旧交接 `9f7330eceb448bf3a639e8fc42b73b90829183de`。
- 调度源码提交 `6dafbc7215b58a2dbb47905a687c930caecf7b44`；快照 `/home/bwwu/grid-audits/sid-extensions-p0/grid-a-launch-20261017/snapshots/cdfd283823bd1928`，完整snapshot SHA `cdfd283823bd19281ce086c7bb3f594455047092d2b6d7693d534f6cd88b46c6`，DEPLOYMENT SHA `7a5a6e49ec61502cdbdb10d361f77c30e7bca456bdeb37101e8b7562188a4a93`，255文件已实际核验。
- 实际绑定发布提交 `0043bad9c8a0cbd6e1d5b71de163a4ef8e745185`，16文件GitHub完整blob回读已通过。该绑定在新controller提交之前已发布。
- 接续配置SHA `d5528b2c185395bd5c77803189f97a756891dd2f5db4d798fbb5f2e1deb04df3`；远端 `/home/bwwu/grid-audits/sid-extensions-p0/grid-a-launch-20261017/shared-20261018/compiled/queue-config-d5528b2c185395bd.json`。
- 研究命令/cwd/PYTHONPATH/capture/数据/输入/seed/优化器/批次/precision/选模/Shuffle/注意力实现逐项与C2一致；S1相比旧244文件仅queue.py、queue_runner.py、deploy.py三个工程文件变化，241个原文件不变。
- A1 best checkpoint完整SHA `0f2ce9ab6155eb9864363c32972d9f9ab775291e58f392b56e1db9901f3cb6f1`；A1 SID `89b2c2bbb94eda26ef0e147404fbff29e6ec8246fbdc754c5d885f313f2de293`。
- A2 embedding仍为18357x512 FP32，原路径 `/home/bwwu/grid-audits/sid-extensions-p0/jobs/ga-20261017-sportsa2-encode-v1/output/item_embeddings.pt`，SHA `b137cb8a006a8b950396524ed5be3bbc91d711b36068d671c01da3742d7cda6f`。A2 quantizer/SID尚未产生，不编造身份。
- Toys两候选checkpoint/SID及来源缺口保持原启动材料：历史上游Qwen/3584声明与512维producer未唯一对应；本轮不重训、不重新编码或重测Toys。

## 资源与审核

累计GPU请求wall含先前工程用量：11.637918738小时；39.75小时实际余额28.112081262小时；因先前扣整秒，控制器保守可用余额28.111984199小时。本次恢复阶段含旧队列启动A1 Normal共0.349137216小时；A2双卡分别累计，未只算单卡wall。原训练每套12小时、评测每条件1800秒、注意力1200秒、量化双卡3600秒均不变。原绝对期限仍为2026-10-19 13:40:30.012007北京时间，未重置；当前已提前终态。

最后GPU0 UUID `GPU-dfff0e95-1db6-e2f4-3431-55bb852fa0ae`，真实free24082MiB/util0%；GPU1 UUID `GPU-59cac3bd-3645-668b-79c8-839e0d878893`，真实free21816MiB/util29%，只有yychen PID1965921/2258MiB。本任务GPU进程均已退出。/home free619741188096B，/data free4339264352256B；最后受控存储逻辑约3.20GB/50GiB，未删除历史结果或修改环境。

非作者：`/root/resource_audit`（源码、13项独立CPU断言、实际资源配置）与`/root/migration_audit`（源码/迁移12项CPU测试、真实stop/binding以及保护作用范围）。主控合计19项unittest通过；具体SHA与范围见两份源码审、两份实际绑定审及runtime/postlaunch_protection_review.json。审核只签启动/调度/保护机械准入，不签SID科学结果或最终推荐/注意力结论。

任务级修正已用**3/3**，剩余0。本轮新增系统包、语言包、全局配置改动均0。研究方法修改0、原成功阶段重跑0；仅新的CPU接续控制器和原计划此前未提交阶段首次提交。A2后续FD适配/重试须用户另行决定及授权，不因新会话/新队列或预算有余额而自动获得修正额度。

## 后续只读入口与停止

```bash
python3 /home/addaswsw/recommend-support/sid-extensions-p0/scripts/gridctl.py --target new-server query ga-20261018-shared-queue-v1
python3 /home/addaswsw/recommend-support/sid-extensions-p0/scripts/gridctl.py --target new-server query ga-20261017-sportsa2-quantize-v1
```

新队列状态目录为 `/home/bwwu/grid-audits/sid-extensions-p0/grid-a-launch-20261017/shared-20261018/compiled/queue-state/`；研究job和日志仍在 `/home/bwwu/grid-audits/sid-extensions-p0/jobs/<原job_id>/`。逐个原ID见terminal-jobs-summary；禁止重提旧ID、重启旧/接续控制器或重跑已完成Toys/A1/encoding。未来先读本回执、原status及日志，再由用户决定A2工程修复/有界接续和正式非作者结果验收。

最终发布的证据只保存一次，本地周期监视已停止；没有部署常驻AI。全部必要调度源码、配置、准入、原件查询、保护日志和终态内容身份随本回执发布wbw，未来大模型、数据和全量结果继续留服务器。

`GPT_UPLOAD_REQUIRED: NO`
