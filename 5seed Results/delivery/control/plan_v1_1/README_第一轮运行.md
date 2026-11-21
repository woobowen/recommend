# 第一轮运行记录

本轮按 v1.1 计划只启动 T01 至 T04，训练 seed 均为 42。GPU0 为 Beauty shuffle，GPU1 为 Beauty MLP，GPU2 为 Toys shuffle，GPU3 为 Toys MLP。四项均从零训练，使用共享 SID embedding 表和既有 Qwen/RQ-KMeans/SID 输入。

服务器项目根目录为 `/home/jcy/GRID_wmh_Qwen_TIGER_20261009`。运行入口为 `codex/plan_v1_1/run_task.py`，监督启动入口为 `codex/plan_v1_1/launch_round.py`。

正式训练输出为 `experiments/sid_order_5seed_v1/<dataset>/<variant>/seed_42/`，进度见各任务 `progress.json`，事件见 `events.jsonl`。训练结束后选择验证 Recall@5 最高的本任务 checkpoint，shuffle 依次完整测试 unshuffled、shuffle_3、shuffle_4；MLP 完整测试一次。完整指标测试关闭绘图张量导出，再对各数据集固定 100 名用户独立补算注意力，所有层和 head 原张量按最终 Top1 前缀保存。

四项任务的进程、日志和状态在服务器 `codex/first_round_20261104/`：`launch_manifest.json`，`T01_status.json` 至 `T04_status.json`，`T01.log` 至 `T04.log`。每项使用独立的 Linux session、`/dev/null` 标准输入和文件日志；监督进程脱离 SSH 后由 PID 1 接管，不依赖本地窗口、终端或 SSH 持续连接。

只启动第一轮，不自动启动后续 seed，不自动重试失败任务。退出本地窗口或断开 SSH 后任务继续；服务器关机或重启不属于本次保持运行的范围。

固定用户名单与样本在服务器 `experiments/sid_order_5seed_v1/fixed_test_users/`。Beauty 和 Toys 各 100 人，五个 seed 与各条件共用，生成于查看正式预测结果之前。

预检记录在服务器 `codex/first_round_20261104_preflight_v2/`，仅为 2 个优化步、32 个训练 batch 的诊断，不计入正式结果；源 batch=16、扩增上限128、累计16、验证/测试 batch=8 与计划一致。四项均通过训练、短验证/测试、注意力开关一致性、最终 Top1 前缀导出及读回检查。`protocol_verification.json` 记录批量8人与逐人预测一致性、原模型参数兼容及正常预测完全一致性。

本地运行 `sync_status.py` 可通过新的 SSH 连接只读刷新状态，并更新项目 `codex/experiment_plan_review/实验计划_v1.1.json`。最新状态保存在同目录 `status/latest_snapshot.json`；断连验证另存 `status/disconnect_verification.json`。这些均为时间点快照，后续查看应重新刷新。

不要重新运行启动脚本以查询状态，也不要重复提交 T01 至 T04。训练是否完成以任务的 `stage_result.json`、进程状态及输出验收为准；启动成功不等于完成。
