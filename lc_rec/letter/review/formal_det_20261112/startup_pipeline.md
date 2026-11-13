# 启动身份与服务器独立运行非作者审核

STARTUP_REVIEW: PASS

真实审核者：`/root/preaudit_pipeline`。主控桥接原件范围；审核者未直接读盘、SSH或执行GPU。

RUN_COMMIT `4c6a83d9ba42dc2fe737792759e410ef1874da76`；核心文件与预验收相同，GitHub回读及2160文件export映射支持部署身份。唯一job `letter-a0-20261112-normal-det-seed42-r0`，提交前正式目录/ledger/job均不存在、两卡空闲；request未启用probe/diagnosis/preflight/resume。Request canonical JSON SHA-256（不含换行）`a6b6c6f6518d389a7c777c5f44552bdcd1e180b6a40ebb100f547f4f3499ffd3`。

runner于2026-11-12 18:41:18 UTC启动，18:44:09 UTC新SSH查询确认supervisor、pipeline、torchrun及两个rank存活，进程身份匹配，unsupported/unconfirmed为空。两rank step10 loss/LR/grad norm一致有限，真实native/extension backward均记录deterministic=True。

当前只通过启动审核；正式状态RUNNING/train。四轮训练、Validation/best、全量测试及最终指标尚未验收，不能据此标记完整实验PASS。
