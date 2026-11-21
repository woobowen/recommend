# SID 前缀语义实验队列核验

核验时间：北京时间 2026-11-10 01:10。用户要求“排好训练队列”，按本会话正在进行的 SID 前缀语义实验核验后续队列。本实验调用 Llama 分类，不涉及模型参数训练。

当前服务器控制器 PID 419697 正常运行，父进程 PID 1，标准输入 /dev/null，日志写入 full_pipeline.log；SSH 断开不影响运行。复用现有控制器，无需重复启动。

执行顺序：

1. Beauty 最终全量分类（当前正在运行，已完成约 10,800 / 12,101 件）。
2. Toys 类别发现（1,500 件）。
3. Toys 全量草分类（11,924 件）。
4. Toys 根据文本和草分类分布修订并冻结类别表。
5. Toys 试标注（250 件），输出质量摘要后自动继续。
6. Toys 最终全量分类（11,924 件）。
7. Beauty、Toys SID 前缀语义评估，包含 1,000 次随机置乱对照。

GPU 0、1 运行单个 FP16 模型副本，各阶段成功且通过完整性检查后自动执行下一项。GPU 2 上原有 PID 394258 保持运行，GPU 3 当前空闲。本轮无需再确认。失败会保留日志并停止，不跳过失败阶段、不自动重启。

服务器实时状态：
`/home/jcy/GRID_wmh_Qwen_TIGER_20261009/experiments/sid_prefix_semantics_v1/single_replica_control/status.json`

本地“运行状态快照.json”仅为同步时状态；“authorization_and_plan.json”是启动时计划，不能当作实时进度。队列由现有 run_full_pipeline.py 执行，不依赖本地电脑在线。
