# 正式前验收与资源方案

**正式前预跑及内部技术审核均通过，正式计划等待网页二审与用户授权。** 实际执行版本为 `390df9f508b7dc11eaa57b79a5a91fb5a7f99385`，服务器快照为 `/data/bwwu/grid-letter-instruments-normal-20261124/runtime-390df9f508b7`。2026-11-24 **12:27:57.403555 ET** 的[服务器盘点](server-inventory.json)确认正式父／子作业、已接受请求、正式模型或预测产物及本实验活动进程均为空。

上轮三次集中修正及失败由[原 PARTIAL 发布](https://github.com/woobowen/recommend/blob/c9ede1914c2e03e3ca0e31e58652c2f2d58baa9e/experiments/letter_instruments_grid_normal/evidence/STATUS.md)和已交付 PARTIAL ZIP 保留。旧包当时没有真实 GPU checkpoint 或候选。补充阶段使用 1 轮授权修正，采用新的 CPU `boundary-s1` 和 GPU `preflight-s1`。旧 GPU `r2` 仍见 [preflight](preflight/)，原未提交方案仍见 [prior-formal-plan](prior-formal-plan/)。

## 补充修正

- 实装 Lightning 2.5 的训练 CombinedLoader 保留一个外层，原 `training_step` 先解包一次。新增训练观察按同一契约处理，验证与测试观察接收直接 inputs／labels。保留每批 loss、代表梯度、参数哈希、Adam 状态、验证事件及尾批检查。CPU 真实框架边界验证 4 个 microbatch、两次更新、两批验证、严格重载与两遍 `8/8/4`。
- `evaluate.native` 核对选中分数与真实 top-K 分数的完整降序多重集。负例 `[0,2,3,4,5]` 遗漏 0.8，予以拒绝；合法的 `[0,1,2,3,5]` 和并列内部重排均通过。原生 `torch.topk`、统一稳定排序及指标公式按既有实现执行。
- 协议、原生生成、原生 evaluator 及冻结 SID/CCE 文件与上轮执行版本逐字相同。单卡 GPU 预跑和正式有效配置与原方案一致。上轮 DATA 全轴验证按本轮请求中的同一组资产哈希复用。另核对原 A0 双 rank 和 A1 正式 Normal 确实使用 `33c87e28…` 的同一 LETTER index，配套交互和商品表也逐项匹配。上轮差异比较的是通用 LC-Rec 目录中 `b94d1639…` 的另一副本。[历史实际资产提取](historical_asset_binding.json)

## 实测闭环

| 项目 | 实际结果 |
|---|---|
| 训练 | step 0、空 Adam state、seed 42 从头训练。1,600 个 microbatch，累积 16，共 100 次更新。实际扩展 batch 55–128，均值 105.853125。第 16/1,600 个 microbatch 后的第 1/100 次更新保存九个代表参数的有限非零梯度、更新前后哈希与 Adam step。 |
| 验证与 checkpoint | 第 1,600 个 microbatch 后真实验证 24 批，`val/recall@5=0.0364583320915699`。ModelCheckpoint 保存 step 100 的完整 best，159,007,470 字节，SHA-256 `08aac6a10e0ec8972338923a8d2615ee398bacc11cea117730cf9ee7f07de3ed`。 |
| 严格重载与早停 | 两遍 GPU 测试实际严格加载 177 个模型状态键，零缺漏或多余键，张量状态哈希一致。方向 A 重新实例化模型独立反序列化并逐张量复核，127 个 Adam 状态均为 step 100。一次真实验证与 patience 10 逻辑夹具通过。自然停止点留待正式运行。 |
| 真实 Normal | 固定 20 人、8/8/4、200 个候选。每遍 3 次生成、12 次 decoder 路径调用，每批首步 cache 为空，输入 mask 不含目标。开启／关闭观察的完整 trace 和四种 RNG（Torch CPU、CUDA、Python、NumPy）字节一致。两遍原生指标一致，Shuffle 调用为 0。 |
| 评价 | 原生 `Recall@10=0.05000000074505806`、`NDCG@10=0.01666666753590107`，两项 @5 为零。统一 `Hit@10=ItemHit@10=0.05`、`NDCG@10=ItemNDCG@10=1/60`、`MRR@10=1/140`，其余六项为零。方向 B 从真实候选逐用户重算全部 15 项，原生 FP32 最大绝差约 `8.69e-10`，低于预先设定的 `24 × 2^-24≈1.43e-6`。有 6 个目录外候选，0 个重复、0 个并列，1 个碰撞目标。 |

[实际输出](success/)含有效配置、事件、原生聚合、逐用户候选和指标、严格加载回执。两份观测原件的 SHA 在 runner 导出清单内，包内只存一份相同字节。[方向 A](REVIEW-A.md)和[方向 B](REVIEW-B.md)均通过。20 人分数用于工程复核。

## 资源、正式计划与恢复

新增单卡预跑 worker 236.479456 秒，累计单卡预跑 273.744618 GPU-device-seconds，含上轮失败及启动／清收。3 小时预算余 10,526.255382 秒。CPU runner 累计 69.758122 秒，8 小时余 28,730.241878 秒。直接 dry-run、非 runner 独立 CPU 复核、传输和本机打包另列。两次新作业均退出 0、清收验证通过。[累计回执](budget-and-attempts.json)

[资源模型](resource-estimate.json)用本机实际 100 次更新推算。初始化约 9.61 秒，稳定训练中位数 2.09 秒／update。24 批验证用时 4.92 秒，按首批、稳态与固定结束开销分项外推 3,097 批约 521 秒，约占每 100-update 周期的 71%。20 人 Normal 含重载约 1.37 秒，候选写出约 1.19 毫秒，离线复算约 0.60 秒。完整测试按稳态批估算约 428 秒，候选序列化约 1.48 秒；含固定解析成本的离线线性估算为 748 秒。显存峰值约 2.58 GiB allocated、3.99 GiB reserved。正式候选和逐用户指标估计分别约 29.1 MB 和 8.64 MB。服务器与本地暂存各预留至少 2 GiB，覆盖 checkpoint、同时存在的临时权重和归档。

4,000／6,400／9,000 次更新情景约 8.43／13.29／18.56 小时。**正式建议单卡、单次墙钟上限 24 小时、累计 24 GPU-device-hours。** 相对 9,000-update 情景留约 5.44 小时。24 小时同时为现行 runner 的单次上限，这次基于实测模型将同一数值由上轮占位转为预算建议，仍需用户批准。全量验证速度、主机争用和新训练自然停止点会改变实际耗时。

[正式 dry-run 与未提交模板](formal-plan/)已绑定执行版本和 9 项正式资产，以 seed 42 与 `ckpt_path=null` 从头训练一次，按原监控条件选择 best，严格重载后按 seed 2026 对 24,772 人做一次 Normal，预期 247,720 候选，保留 4 人尾批。无限训练数据流保留 320,000-step、10-epoch 框架上限与 patience 10，epoch 并非 TFRecord 全量遍历次数。资源超时是执行中止，测试指标不参与选择。批准后用原 `gridctl.py --target new-server submit` 指向模板，提交前复核时间上限和物理卡 UUID。断线先按原 ID 查询，不自动重试或恢复训练。非零退出和超时保留原记录，续跑策略另行批准。

正式未提交，正式 GPU 增量为 0。环境沿用既有 GRID Python、Torch 2.6.0+cu124、Lightning 2.5.0。原 TFRecord 随机化未显式固定 TensorFlow seed，正式结果无需与短预跑或历史逐位一致。
