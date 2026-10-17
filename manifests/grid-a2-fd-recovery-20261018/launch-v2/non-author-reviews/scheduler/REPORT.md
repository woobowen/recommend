# A2-only 接续调度真实非作者审核

`A2_ONLY_SCHEDULER_SOURCE_CPU_REVIEW_PASS_CONDITIONAL_DEPLOYMENT`

审核者：`/root/sports_a1_result_audit`。本代理不是本次 FD 适配或新 build/controller 的作者；本审核独立于 `/root/fd_repair_reviewer`。本报告仅对调度源码、CPU 编译与当前已给出的真实部署/spec 绑定作出结论。正式 queue-config、条件 launch-review、frozen 证据与服务器运行准入尚待最终绑定，因此此标记不等于正式量化已获完整内部准入，更不等于科研结果 PASS。

## 实际阅读与验证

直接阅读了新 `experiments/a2_fd_recovery_20261018/{build.py,controller.py}`、原 `launch_20261017/{queue.py,run.py}`、共享调度 `launch_20261018_shared_gpu/{build.py,resume.py}`、`initial-scene.json`、原11份 A2 stage/spec 和原队列状态/预算。没有改动主源码、Git、旧34阶段、原失败或 A1 最终审核报告。

`check_scheduler.py` 实际执行了22项相称 CPU 正/负例。全部操作限制在本目录隔离 fixture 中，使用假 deployment 标记和假成功/失败产物，**真实 runner submit=0、GPU/model calls=0**。fixture 明确不是科研结果或真实作业回执。

- 真实调用新 `specs()` 和 `compile_queue()`，构建11个唯一新 job（quantize＋原10个下游）。没有 encode stage，也没有第二套34阶段流水线。
- 原模板逐字段归一化比较，只允许新快照/capture/job路径、去掉无关旧 prereq、加入仅量化的FD工程字段；所有 seed42、K10、研究身份、原config/training_protocol hash保持不变。
- 动态调用真实 `run.resolve()`：原encode只作为已成功依赖，只有量化模板允许它的旧capture；metadata/train/eval/attention/stats全部使用新量化SID，三条件eval与attention全部使用同一新train best，三注意力与stats身份配套。
- 原历史 accounting 独立从原 shared-state 的逐阶段真实 started/ended/GPU数复算，与 initial-scene 精确一致；原绝对期限从原controller started_at＋原 max_queue_seconds 复算相符。
- group预算重置、总预算重置、deadline延长、旧job复用、改GPU policy、少/多阶段、runner改变、遗漏旧storage、占用新ID、历史SUBMITTING、旧失败量化链接和best/SID冲突都实际被拒绝。
- 在真实 `queue.supervise()` 的隔离fixture中给量化设置已接收FAILED：只查询1次、0次submit，量化 `BLOCKED_NO_RETRY`，10个依赖全部 `BLOCKED_DEPENDENCY`，controller进入机械终态；再次启动相同controller被拒绝。它没有以新ID偷偷再跑一次量化。
- 两卡资源fixture允许每卡已有16000MiB peer，只按原6144MiB峰值＋2048MiB余量准入；任一卡8191MiB free、utilization51%或UUID缺失都拒绝。未改原共享策略、三次预采样、锁、runtime/disk guard。
- 已测试原绝对deadline无法容纳完整stage wall＋60秒时拒绝提交，以及原deadline前60秒停止新增工作。

## 预算与依赖预留

| 项目 | GPU请求秒 |
|---|---:|
| 原阶段实耗 | 41837.856885 |
| 原prior工程实耗 | 58.650573 |
| 合计真实历史实耗 | 41896.507458 |
| 原143100秒总额精确剩余 | 101203.492542 |
| shared保守整数扣减后新queue cap | 101203.143115 |
| 旧量化实耗 | 302.049476 |
| 原量化组7200秒剩余 | 6897.950524 |
| 新量化wrapper 3420秒×2卡 | 6840 |
| 量化组额外余量 | 57.950524 |
| GPU admission | 900 |
| 新train上限 | 43200 |
| 三条件eval＋三attention | 9000 |
| 新11阶段GPU请求总预留 | 59940 |

量化计算入口内部 hard wall3300秒，外层3420秒为启动/收尾保留120秒；不能因此重新获得7200秒。新CPU metadata＋stats预留2400秒。工程900秒位于原剩余工程组2641秒内，CPU本审核不消耗GPU额度。

原期限保持 `2026-10-19T05:40:30.012007+00:00`，即北京时间2026-10-19 13:40:30.012007。没有重新计时到两天后。运行前和worker入口均要求原期限能容纳本stage原完整预算。

## 当前真实发布字节补核

已对用户主控提供的真实发布做第二轮独立绑定：

- commit：`0d18820025e7a28675ed7112be0b1b4ff0110025`。
- snapshot：`c31d2d3eaaafbd1899bffc5f4226ffe6ffdd51d88b4d1f014289d3e17ead44ad`。
- server root：`/home/bwwu/grid-audits/sid-extensions-p0/grid-a-launch-20261017/snapshots/c31d2d3eaaafbd18`。
- DEPLOYMENT SHA-256：`6acbdcf12b7e46e85888c8e01d29a89a3825dfc7077f10052b493c2569013504`。
- 对262份manifest成员逐份 `git show <完整SHA>:<path>` 验证字节SHA一致；CPU检查所审6份源码hash全部未变。
- 通过既有 Transport 在真实服务器只读取得DEPLOYMENT，并直接核9份相关新/原queue源码字节SHA，与本地审阅和固定提交相符，回执 `actual_server_source_binding.json`。
- 对真实 `compiled-c31d2d3eaaafbd18/specs/` 11模板与plan独立重新编译：逐字段一致、plan文件字节SHA一致、每template identity均位于plan，Python源码哈希映射完整精确。回执 `actual_specs_review.json`。

这一步未调用真实模型、未提交实验。它不替代随后正式queue-config与review文件的不可变SHA绑定。

## 待主控提供后必须补核

1. 正式11份spec已绑定的 `launch_review` path/hash、真实review.plan/source-map/FD-equivalence结论。
2. 正式queue-config、frozen evidence完整成员及SHA、实际requests/render、服务器对应字节。
3. 在服务器原runner下真实 `controller validate` 的旧controller/旧34阶段终态、无SUBMITTING/新ID冲突和预算检查回执。
4. controller被原独立runner接收并脱离SSH继续；量化启动后的实际两rank/24workers/FD和有效step健康由主控与FD审核者核验。

最终实际配置若与本报告受审源码/模板/资源条件不一致，本条件PASS不自动转成正式准入。A2科学结果目前仍未产生，必须保留 `SCIENTIFIC_RESULT_NOT_YET_AVAILABLE`。

本审核未安装包、未改环境配置、未做Git写操作。`scheduler_cpu_review.json`与stdout包含22项具体结果和假fixture路径；原A1报告保持不变。

GPT_UPLOAD_REQUIRED: YES（由主控合并本目录必要脚本及回执到最终单一审核ZIP。）
