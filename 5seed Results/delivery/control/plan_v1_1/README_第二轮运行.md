# 第二轮运行记录

2026年11月4日用户授权检查第一轮、在指定Word中登记数据并启动下一轮。已在原路径更新桌面 `shuffle与MLP五seed实验执行计划_v1.1.docx`，项目副本同步；第八节是完成登记，第九节是第一轮实测数据，第十节是第二轮与一致性核验。

北京时间2026-11-04 13:17:17快照：T03 Toys shuffle seed42已完成7400优化步训练、3次完整测试、3组各100人注意力，全部输出哈希与张量读回验收通过。最佳checkpoint来自step6400。T01 Beauty shuffle、T02 Beauty MLP、T04 Toys MLP仍在训练或验证，尚无最终测试结果。

第二轮严格按计划指定物理卡：T05 GPU0 Toys MLP seed43，T06 GPU1 Beauty shuffle seed43，T07 GPU2 Beauty MLP seed43，T08 GPU3 Toys shuffle seed43。T07已在空闲GPU2从零启动，断开SSH后重新连接，原PID133395由28步推进至100步；T05/T06/T08已在后台独立session中等待第一轮对应GPU完成并释放锁。没有中断任何第一轮任务。

服务器项目根目录为 `/home/jcy/GRID_wmh_Qwen_TIGER_20261009`。第二轮状态和日志在 `codex/second_round_20261104/`。启动脚本是 `codex/plan_v1_1/launch_round2.py`；训练仍调用同一 `run_task.py`、`protocol.py`、`attention.py`，核心文件与原模型、配置哈希逐项核验一致。T07对T02实际配置逐字段比对，仅训练seed、输出目录和运行标识不同，输入文件哈希和固定名单完全一致。

任务完成全部测试、注意力后才释放GPU。前项失败则下一项停止并记录错误，不自动换seed、不重试、不覆盖输出。本次只授权第二轮，未启动第三轮。

证据在本地 `status/execution_report_latest.json`（文档使用的快照）、`status/execution_report_after_reconnect.json`、`status/round2_disconnect_verification.json`；服务器 `codex/second_round_20261104/configuration_comparison.json`记录T07/T02配置比对。

后续检查应运行服务器 `collect_execution_report.py`并下载最新快照，不能把本文的时间点状态当作实时状态。原 `sync_status.py`仅处理第一轮，勿用它覆盖总进度；原 `build_experiment_plan.py`为初版制订脚本，勿重跑覆盖已登记结果。`update_execution_doc.py`属于本次更新脚本，使用本次保存的原文档备份，后续编辑应以现有最新Word为基础保留已追加记录。
