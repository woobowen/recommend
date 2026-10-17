Toys A 双候选／Sports A1-A2：启动及20分钟健康观察交接

状态：**LAUNCH_HANDOFF_READY**。本轮只完成启动与健康观察；任何未来训练、测试指标、attention及科学结论均未预签。

北京时间2026-10-17，首次真实optimizer更新完成回执为2026-10-17T05:42:20.997100+00:00（UTC）。观察区间2026-10-17T13:42:51.918139+08:00—2026-10-17T14:03:07.966055+08:00，实际1216.048秒。T0/T10/T20新连接查询及服务器41个机器健康样本验证通过，最大间隔33.492102秒，无OOM、NaN、设备错误、重复worker或异常终态。机器原记录见`compiled/handoff-report.json`、`health/health-T0.json`、`health/health-T10.json`、`health/health-T20.json`与`health/health-continuity.json`。

|工作流|观察结束状态|实际最近job_id|所列作业worker wall|后续|
|---|---|---|---:|---|
|Toys A-5300|三条件测试、fixed32注意力及CPU统计均机械SUCCEEDED；结果待独立审核|ga-20261017-toys5300-stats-v1|1.633秒（CPU统计）|未来结果审核|
|Toys A-5000|Normal/Shuffle3成功；Shuffle4运行至3592样本|ga-20261017-toys5000-eval-4-v1|35.154秒|三条件注意力→CPU统计→未来审核|
|Sports A1|RUNNING；step300第三次完整validation已处理34824目标|ga-20261017-sportsa1-train-v1|1257.625秒|训练终止及best身份门→三条件测试→注意力→统计|
|Sports A2|编码SUCCEEDED；双卡量化WAITING_GPU、NOT_SUBMITTED|ga-20261017-sportsa2-encode-v1|178.436秒|量化→SID身份门→metadata/GPU门→训练→best门→测试/注意力/统计|

排队不是已运行。A2量化预定ID`ga-20261017-sportsa2-quantize-v1`尚未创建/接收。完整34阶段请求及planned/actual区别见`compiled/queue-config.json`和T20快照。四个workflow各自保留，不能互相替代。

最新真实进度：A1于2026-10-17 14:03:02.899 CST记录step300、validation batch4352、34824目标；此前已完成2次完整35598目标验证，key-target multiset精确，最近完整val/loss=13.246960639953613。A-5000的Shuffle4进度batch448、samples3592。A2最终结果为18357×512 FP32，最后周期性progress的RUNNING不覆盖真实SUCCEEDED终态。

服务器GPU0 UUID`GPU-dfff0e95-1db6-e2f4-3431-55bb852fa0ae`：RTX4090，显存18968/24564MiB、空闲5116MiB，瞬时利用率13%；GPU1 UUID`GPU-59cac3bd-3645-668b-79c8-839e0d878893`：740/24564MiB、空闲23344MiB，利用率16%。实际compute进程与owner见T20的compute_apps/process_owners。queue supervisor/worker为1762401/1762403，A1 supervisor/wrapper为1762409/1762410；各PID均结合boot_id/start_ticks/UID/argv核对。stdin为/dev/null、tty0，输出在服务器work.log/supervisor.log；独立子作业已剥离outer标记，通过新SSH连接按原ID查询，继续运行不依赖WSL、Codex、SSH或VPN。

运行源码：`b44399926c98369191c2ff5c2425377c8c07d3c8`。提交前发布回读：`98579e415dc48bfa3a6c5dc50effe70cf25670fd`。服务器244文件只读export：`/home/bwwu/grid-audits/sid-extensions-p0/grid-a-launch-20261017/snapshots/b6203414ee58a400`。最终证据提交与运行源码分别表达，后续证据更新不改变活动源码。

资产：

- Toys5300 checkpoint SHA256：`5e88fac352f603cb0babad1935f7044072b1e75975bf9b41c30fda11f70435c5`；Toys5000：`ed627a41aae73d941fe354e2c29d99d1e9d768e407c9d83eb2c401ca2f562090`。
- 两Toys共同SID：`a487a4bdadef63f3a87386ddce52ec03f4771bedbaf554f63272c45fe373e7d1`，item轴0..11923。原件恢复路径在`reviews-and-identities/asset-restoration-receipt.json`。
- Sports A1 Qwen SID：`89b2c2bbb94eda26ef0e147404fbff29e6ec8246fbdc754c5d885f313f2de293`，原路径`/data/bwwu/sid-extensions-assets/563dc50b1c7e/experiments/sports_rkmeans_qwen_20261010_093035/semantic_ids.pt`。
- A2原T5-small权重：`bd944e5f1b3ad9b70dd9d00010a517059e19265671076b8b0a4a58d9491842bc`，恢复目录`/home/bwwu/grid-audits/sid-extensions-p0/grid-a-launch-20261017/assets/A800_T5_SMALL`。
- A2新embedding：`b137cb8a006a8b950396524ed5be3bbc91d711b36068d671c01da3742d7cda6f`，`/home/bwwu/grid-audits/sid-extensions-p0/jobs/ga-20261017-sportsa2-encode-v1/output/item_embeddings.pt`。R-Kmeans checkpoint与最终SID尚未生成，不能预填哈希。

A2保留原text字段、max128、原Tokenizer/MeanAggregation/Identity、冻结T5、FP32；RQ保留原world2、BF16、raw2048/卡、3×256×512、30steps、原初始化及层级残差归一化。第四位为确定性碰撞消歧。20%编码duty cycle为纯调度节流，空闲时间也计GPU worker wall。Toys历史Qwen/3584声明与512维上游producer绑定未唯一确认；当前checkpoint/SID/item轴及模型语义真实配对，作为用户批准的新适配，不冒称唯一历史执行。

Sports两路线同一从零完整双表协议：seed42、32-true、Adam lr0.001/wd0.0001、无scheduler、raw16/cap512、累积16、每1600 loader batches完整验证、val/recall@5选max best、patience10、max_steps320000/max_epochs10、K10、prefix检查False。原Beauty A确证world_size1。没有加载推荐生成器权重、共享/复制/冻结双表或缩小batch；测试不参与选模。各模型三条件使用同checkpoint/SID与预先固定32用户。Shuffle仍在offset和lookup后按每个history occurrence内部重排；beam/evaluator保持原件。

确定性CPU queue固定ID`ga-20261017-queue-v1`，config SHA256`0de997aad986903da46bb3865c5d26507d45b0944d13b46c48da9c01c272997a`。所有后续阶段已纳入服务器34阶段受控依赖图；每次只检查前置SUCCEEDED/exit0/verified cleanup、固定capture/workflow、内容hash、SID覆盖、完整validation与严格best。失败、身份冲突、空间不足或超时只停止相关链，不改方法、不无限重跑。GPU锁和in-flight预留避免重复并发；原ID重复请求不重启。48小时仅扩展纯CPU队列监督，所有计算阶段仍使用原runner与原限额；期限到达停止新增提交，已接收子任务继续，未接收尾链需用户未来开启Codex后判断接续。未来结果的非作者验收仍须手动安排，机器队列不预签研究结论。

T20资源：已用新增GPU请求2446.690999秒（0.679636小时），39.75小时预算余39.070364小时；包含先期58.650573秒。两卡作业各自累计worker wall，不把并行时间合并计费。单eval≤1800秒、单attention≤1200秒、每模型eval/attention合计≤9000秒、每Sports训练≤43200秒、T5≤10800秒、双卡RQ≤3600秒worker wall/7200 GPU秒；工程池总≤2700秒，未来A2 GPU门预留900秒。A2未来CPU数据/统计阶段预留2400秒，加已执行资产准备仍远低于6小时。未分配1741.349427秒不用于新增研究。

/data空闲4374179307520字节，/home空闲620053528576字节；本轮目录逻辑大小2893799018字节（约2.695GiB），50GiB保护覆盖资产及11个先期/34个未来job实体。MemAvailable246793464kB，loadavg3.954/3.588/2.759。无新增系统包、语言包、工具链或全局配置改动。任务级集中修正2/3：短queue fixture预留及CPU工程标签整组切片；旧失败保留，正式训练/GPU分支未因CPU修正改变。

实际非作者：`/root/sports_training`审核主控桥接、配置与A2方法；`/root/assets_a2`审核训练与队列；`/root/runner_queue`审核训练、A2、最终请求及真实启动/健康。对应回执在`reviews-and-identities/`，均限定源码、准入或启动健康，未来结果未预签。Git仅wbw更新，其他分支及历史成果保持；没有未提交正式研究源码，bytecode缓存未清理。

接续仅先只读原任务，禁止重复提交：

```bash
python3 /home/addaswsw/recommend-support/sid-extensions-p0/scripts/gridctl.py --target new-server query ga-20261017-queue-v1
python3 -B /home/addaswsw/recommend-support/grid-a-launch-20261017/health_query.py --queue-deployment /home/addaswsw/recommend-support/grid-a-launch-20261017/compiled-b6203414ee58a400/queue-deployment.json --label resume
```

服务器每个实际job的原始request/status/work.log/supervisor.log/output在`/home/bwwu/grid-audits/sid-extensions-p0/jobs/<job_id>/`；queue-state入口在`compiled/queue-deployment.json`。材料只含轻量证据，不含大型checkpoint、全量数据或模型权重。

GPT_UPLOAD_REQUIRED: NO。必要启动证据均发布在固定GitHub证据提交，服务器原记录具有明确身份与恢复入口。

本轮仅完成启动及20分钟健康观察，远端合法作业保持运行；当前 Codex 执行已停止，等待用户后续监视指令。
