# GRID 原任务共享GPU调度接续（修正 round3/3）

本目录是原34阶段任务的调度接续证据，不是实验结果验收。

- 科研实现固定为 `b44399926c98369191c2ff5c2425377c8c07d3c8`；新调度源码为 `6dafbc7215b58a2dbb47905a687c930caecf7b44`。
- 原队列 `ga-20261017-queue-v1` 已精确停止未来提交，`STOPPED_NO_NEW_SUBMISSIONS`。外层 `FAILED / exit2` 为受控替换退出；研究worker停止0。
- 最终停稳清单为17个已接收阶段：16个成功终态，Sports A1 Normal原job `ga-20261017-sportsa1-eval-0-v1` 在停止旧CPU控制器前已独立启动，保留原request和运行。
- 待启动控制器 `ga-20261018-shared-queue-v1` 继承全部已接收原身份，未提交阶段仍属于原计划；不重跑Toys、A1训练或A2编码。
- 科研snapshot仍为 `snapshots/b6203414ee58a400`；调度snapshot为 `snapshots/cdfd283823bd1928`。新配置SHA见清单及不可变远端配置文件名。
- 每个待提交GPU阶段依据真实memory.free、预计峰值、余量、3次观测显存波动及负载准入。没有peer进程存在性或固定peer显存上限；双卡RQ逐卡满足条件。单卡可选择原批准两UUID之一。运行时只保护本任务进程。
- 原GPU预算143041秒（已保守扣除先前工程用量）、分组及单阶段上限保持；原绝对监督截止 `2026-10-19T05:40:30.012007+00:00`（北京时间19日13:40:30），不重置。
- `stop-receipt-server-original.json` 是SCP校验的服务器原始字节，保留全部原job请求和终态证据；不将格式化副本sha当作服务器文件sha。
- 非作者源码审核、CPU回归及实际绑定审核见 `experiments/launch_20261018_shared_gpu/`；不预签未来SID、训练、测试和注意力结果。

当前这些文件记录的是提交前准入。实际新controller及子job启动状态须读取后续真实运行回执，不把预定ID写成已运行。
