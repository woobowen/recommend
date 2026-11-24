# 本轮验收状态

**PARTIAL / BLOCKED**。CPU 数据和模型结构核验通过，单卡预跑在首个 microbatch 后退出。正式训练和全量 Normal 测试尚未提交。正式 dry-run 已生成配置，用于复核。本轮未达到正式启动门槛。

## 直接证据

| 项目 | 本轮实测 |
|---|---|
| 原数据与 SID | [全量读回](full-axis-readback.json)覆盖 3 × 24,772 个用户行。三个 TFRecord 和 SID tensor 与历史 SHA-256 一致。既有 LC-Rec 目录 index 覆盖相同 9,922 件商品，四码记录逐商品有差异，采用从固定 LETTER 提交取得的原 index。见[逐商品对比](index-comparison.json)。 |
| Collator、mask、BOS/SEP | [CPU Collator](collator-check.json)验证固定 20 人、4 至 20 件历史、8/8/4 batch、训练长序列及 30-item 窗口。 [实例探针](model-cpu-check.json)验证 1,024 槽共享表、四层 encoder 和 decoder、四预测头、BOS、SEP、位移和 mask。CPU 与 GPU 实例初始化参数摘要逐字段一致。 |
| 执行 | [GPU 实例和环境](preflight/)记录 13,128,064 个参数、FP32、单张 RTX 4090、固定初始化 SHA-256、step 0、空 Adam state 及实际参数组。首个 microbatch 的 loss 通过回调有限性检查，已到 Lightning backward 后的 `on_train_batch_end`。该回调读取批次失败，未取得梯度记录或 optimizer update。 |
| 生成与评价 | 原生与统一评价实现、原精度导出接口和[合成边界](evaluation-fixtures.json)已核。GPU 未到验证、checkpoint 保存和严格重载、Normal 生成或候选导出。两套指标的真实候选复算、导出开启／关闭等价、真实末批、实际验证周期与选模尚缺。 |
| 作业 | [单次状态和计量](budget-and-attempts.json)涵盖两个 CPU、三个单卡请求。CPU 成功一次。GPU 最后一次失败退出 1，所有任务均已验证清收。三个单卡请求从 worker 启动到结束累计 37.265162 秒，含启动、非计算段和失败退出。 |
| 正式未提交 | [服务器现场](server-inventory.json)按任务前缀核对 job、请求、产物及进程。正式增量 0 GPU-device-seconds。服务器配置生成路径不产生 runner 请求。 |

## 修正记录及阻塞

1. 第一轮处理 CPU 数据准备窗口。原转换入口自己追加 `inputs`，适配器初次给出的根多了一级。新 attempt 采用正确父目录，原失败目录和日志保留。独立 collator 夹具按原入口包导入顺序启动，避免直接从 interfaces 触发原循环导入。全量数据和夹具复核通过。
2. 第二轮修正 GPU r0 的 OmegaConf 结构化预跑字段写入，仅开放 trainer 节点装配已有预跑配置。GPU r1 通过配置解析，实例审计读取层数时在 T5EncoderModel 外壳取 `block` 报错。
3. 第三轮实例审计读取实际内部 T5Stack，CPU 前向测得 BOS/SEP 和掩码。GPU r2 中模型实例、初始化和 Adam 起点通过。首个训练 microbatch 后 `Recorder.on_train_batch_end` 读取外层 `batch[0].mask`，但 `batch[0]` 是内层列表。

原 `SequenceDataModule.get_dataloader` 返回包含 DataLoader 的 tuple，Lightning fit 阶段的 `TransformerBaseModule.training_step` 解包以后才取得模型输入。建议下一轮只对齐新增训练回调的解包方式，先用真实 Lightning 训练／验证回调边界做定向检查，再重跑原 100-update 上限的独立预跑，延续同一研究设置。测试回调应按评估路径实际输入形状核验。当前阻塞发生在预跑专用观察逻辑中，正式链所需闭环证据仍未取得。

## 正式执行方案草案

[正式 dry-run](formal-plan/)保存实际 Hydra 配置、overrides、配置生成回执和未提交的 runner 计划。现有训练和验证数据流保持原配置，正式测试指向完整 testing 分片。预跑上限为 100 update、1600 个 microbatch、24 个 validation batches、固定 20 名测试用户及相同输入的导出关闭对照。当前预跑停于 step 0。两份配置的训练 seed、单 GPU、验证间隔、模型、优化器和精度一致。

正式链按顺序进行身份和单卡检查、新初始化并 fit、按验证指标取 best、严格重载、单次全量 Normal、保存同次候选、离线复算及覆盖核对。runner 以固定 job ID、提交锁和持久 attempt 记录防重复计算。在预跑闭环、网页二审与用户授权完成后提交。新 SSH 会话按原 ID 查询、读取持久日志和终态，失去回执时先查同一 ID。超时和非零退出均保留记录并停止链路，训练恢复策略由用户单独授权。预跑状态不作为正式训练起点。

本机无稳态 GPU 训练、完整生成与写盘实测，故正式资源预算未冻结。历史证据可用于量级推算。每 100 optimizer update 的训练阶段首三段约 247–250 秒，随后完整原生验证各约 350 秒。历史训练至 step 6400 约 10.57 小时，Normal 全测约 329 秒。当新运行需要 4,000–9,000 updates 时，按该历史速率约 6.7–15 小时，其中验证约占 58%。新运行的终步、吞吐、候选写盘、离线复核和传输尚需测量。计划文件中的 86,400 秒是 runner 可接受的墙钟上限占位，等待实测和用户确认。原 `max_steps=320000` 与 patience 保持生效，资源超时视为执行中断。

每个全量 Adam checkpoint 以权重和两份 moment 的 FP32 数据计算约 150 MiB，尚有状态和序列化元数据。临时重叠 checkpoint、候选 JSONL、运行记录和证据导出都要预留空间。计划启动检查至少保留 2 GiB，现场见服务器容量记录。训练、验证、导出与离线复算完成时间需在补充预跑中更新。

沿用现有 GRID 环境和任务 runner，未安装依赖或修改用户环境配置。
