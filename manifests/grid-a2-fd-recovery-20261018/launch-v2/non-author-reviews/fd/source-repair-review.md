# A2 FD 工程修复独立审核

审核者：`/root/fd_repair_reviewer`。未编写、编辑主修复、研究源码、测试源码、controller、builder或请求；仅编写本目录的独立检查和审核回执。

**FD_ENGINEERING_EQUIVALENCE_PASS**

**SCIENTIFIC_RESULT_NOT_YET_AVAILABLE**

此处只通过工程等价和当前源码审核。正式提交需要固定部署后对具体11份spec、plan、launch-review、queue-config、rendered requests、冻结lineage和live验证另行准入；本回执没有预先批准正式科研结果，也不能单独作为提交许可。

## 直接审阅范围

已读原失败 `work.log`、中断前prefix及其hash、旧request/status/cleanup、原进程FD捕获、原有效配置、原collate/ItemData/DataModule/dataset/shuffle/预处理、原ResidualQuantization与writer、已保存PyTorch/Lightning/multiprocessing安装源码。原件与关键本地源比对证据为 `baseline-inspection.json`。

本轮已逐段读新 `adapter.py`、`check_equivalence.py`、`controller.py`、`build.py` 及 `a2_upstream.py`、`run.py` 全部实际diff。`inspect_repair_static.py` 移除明确列出的工程改动后，两个legacy文件与保全C2源码全文相等。保全学长源码未变。

已直接读取服务器CPU独立作业抓取的 `output/equivalence.json`、`config-equivalence.json`、27个FD回执、`work.log`、`status.json`、`cleanup.json`。独立脚本逐成员复核status记录的大小与SHA，日志内JSON与产物JSON完全相等；不是只接受主控摘要。

## 等价和FD证据

CPU job：`ga-20261018-a2-fd-cpu-equivalence-v1`，真实 `SUCCEEDED/0`，54秒左右，cleanup verified且没有owned_live。CUDA_VISIBLE_DEVICES为空，GPU请求为0。

- 原T5 embedding真实身份为18357×512 FP32，SHA-256 `b137cb8a006a8b950396524ed5be3bbc91d711b36068d671c01da3742d7cda6f`。测试实际遍历全部18357物品，dense ID覆盖、无遗漏/重复及每条embedding配对经过assert。
- 固定2048真实样本的ID保持int32、顺序和值相同，原单项形状[1]变成统一容器[2048,1]；embedding [2048,512]、FP32、逐值一致，附加feature分支也一致。ID独立storage **2048→1**，全batch独立storage **2049→2**；带附加feature为2050→3。
- 原始model_step、predict_step与工程PredictionWriter在确定性CPU质心fixture上输出逐值一致，keys与预测配对相同。fixture明确是未训练测试模型，不是已完成量化结果；不把接口验证写成科研复现。
- Python、NumPy、Torch RNG状态相同。adapter不覆盖worker_init、不改sharing strategy、不额外调用随机数。cfg恢复两个collate target后与旧有效DataModule完整相同。
- 512×2 worker的小IPC四批，原ID storage各512，新各1；原主进程FD **527/527/1040/527**，新 **16/18/16/16**。原、新逐批2048个原始ID及embedding hash对应一致；独立用struct重建int32字节，SHA与direct fixture记录一致。
- 原native2048×12 worker配置，用同一main PID顺序模拟rank0/rank1，各真实返回3批。主FD68–75；24个不同worker在采样时FD20–96，PPID与CPU main对应，soft/hard均524288。两个模拟rank各12份worker_id回执完整。
- 入口实测soft/hard由[1024,524288]变为[524288,524288]；只调用本进程resource.setrlimit，未改系统配置。CPU测试只有一份main入口limit记录；正式DDP两rank的实际继承仍必须现场核验。

## 新增只读checkpoint门

原30步fit完成并save_checkpoint后，新增仅FD模式执行的权重门：逐key和逐tensor确认保存state与当前训练模型精确相等，再strict=True加载这些相同tensor到同一模型，写出回执。没有新建随机模型、额外初始化或修改训练算法。该源码尚未产生正式checkpoint，最终应核实际global_step、payload.current_layer/layers_initialized、全部state、SID导出和训练日志，不能只凭门文件存在判通过。

## A2-only控制器源码

新builder从原已审核11个A2阶段派生新job ID、SID/训练/评估/注意力的动态依赖，未生成encode请求。controller只接受这11阶段，继承原runner、GPU锁、storage门、两卡UUID及共享资源策略。旧两个controller和34阶段仅只读核验，旧量化保留INTERRUPTED/-9；存在SUBMITTING/不明目录、旧任务未清空或新ID占用时拒绝准入。

历史stage GPU请求秒重新扣除；原prior工程58.650573秒另外保留。新总cap采用原143041秒保守预算减历史stage41837.856885秒，即101203.143115秒，较精确总原预算剩余101203.492542秒少0.349427秒，没有重置额度。量化外层3420秒×2=6840 GPU秒，较该组剩6897.950524秒留57.950524 GPU秒；内部callback3300秒。训练仍12小时，eval/attention组仍9000GPU秒，GPU admission沿用原工程组，不新增GPU额度。

原绝对期限 `2026-10-19T05:40:30.012007+00:00`（北京时间13:40:30）直接继承，提交每阶段前要求完整原wall加60秒可容纳。at-most-once逻辑沿原queue：终态/不明接收不自动重试。上述仅是源码审阅，编译后的实际请求、预算数值、冻结证据成员、部署源和live验证仍需单独审核。

## 正式启动与结果边界

正式任务启动后必须直接确认真实DDP两rank、每rank12worker及实际limit、first batch、精确配对、step增量、有限loss、初始化和FD趋势。限额保护为每进程FD达到min(8192,soft×0.5)时停止该rank并由本任务runner精确清理；监控失败也停止本任务，不能以进程存在代替健康计算。

原失败根因证据强烈支持小Tensor IPC放大FD，但没有旧FD target分类/时间序列，不能声称唯一根因。原rank0 worker候选的soft是524288，与任务提示表1024冲突，已按原件明确保留。旧worker归属缺PPID/worker_id原件，仍只称argv推导候选。

历史3/3修正耗尽继续保留，本次是另行授权的一套工程修复与一次新正式量化。若新量化重大失败，停止依赖并保留原新失败，不能自动再试。

未安装依赖，未修改全局环境配置。

GPT_UPLOAD_REQUIRED: YES（纳入最终统一ZIP；本回执不是最终科研交付。）
