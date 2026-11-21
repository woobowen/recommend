# 当前后台队列（2026-11-10 收敛修正后）

当前源码：`sid_prefix_semantics_v7_compact`。
当前控制器：`run_full_pipeline_compact.py`，启动 PID 453294。
当前日志：`full_pipeline_compact.log`。
当前状态：`experiments/sid_prefix_semantics_v1/single_replica_control/status.json`。

北京时间 2026-11-10 22:21：Beauty v1 最终分类保持已完成；Toys 已完成修正后的类别归一、通过上下文检查并生成首批 20 件草分类标签。所有候选名称有完整映射链，当前类别为待修订草案，人工质量验证未完成。

后续自动队列：Toys 全量草分类 → 基于文本和全量分布修订并冻结类别 → 250 件试标注 → 最终全量分类 → Beauty/Toys 正式 SID 前缀语义评估。使用 GPU 0/1，一个模型副本，SSH 断开不影响运行。错误会保留日志并停止，不自动重启整个任务。

修正以本会话用户“解决收敛问题”为授权，保留原始发现批次、Beauty 结果和所有失败记录。未向主会话发送消息。详细方法、协议变化、核验和局限见 `../sid_prefix_semantics_v7_compact/README_当前版本.md`。旧控制器和旧快照仅供历史追溯，不要重复启动。
