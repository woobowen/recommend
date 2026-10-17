# A2 EMFILE 独立准入审核：原件核查与待验收条件

审核者：`/root/fd_repair_reviewer`。本审核者不编写主修复、不操作 Git、不提交或启动 GPU 正式实验。显式读取 `/home/addaswsw/recommend/{AGENTS.md,PROJECT_RULES.md,REMOTE_EXECUTION.md}`；首命令 `ls -la` 确认规范入口和 Git 工作树是不同目录。只读范围为指定原故障包、当前待审核工作树、保存的安装源码。所有审核输出仅写本目录。

**当前结论：BASELINE_EVIDENCE_VERIFIED_NEW_REPAIR_ADMISSION_PENDING。** 尚未取得修复 diff、原始测试输出与新请求，不能准入正式重试。科研结果为 `SCIENTIFIC_RESULT_NOT_YET_AVAILABLE`。

## 已直接核实的原始证据

- `a2-failure/work.log` 有 24 次 `Too many open files`；首 traceback 经 `multiprocessing.queues._feed`、`torch.multiprocessing.reductions.reduce_storage`、`multiprocessing.reduction.DupFd`、`multiprocessing.resource_sharer.DupFd.__init__`。日志本体及中断前前缀 SHA 与保存清单一致。
- 原 `ga-20261017-sportsa2-quantize-v1` 保存终态为 `INTERRUPTED/-9`；cleanup verified，最后记录没有 owned_live，`worker_may_be_live=false`。`-9` 来自受控终止，不据此认定 GPU OOM。本审核未用历史记录代替本轮真实进程查询。
- 原 FD 捕获为 rank0 main 51354/524288；rank1 main 119/1024。rank1 的 12 个 worker 候选各 1023/1024。**原文件中 rank0 的 12 个 worker 候选 FD89–133，soft/hard 均为524288，与用户提示表中的 soft1024 不一致。应保留此原件差异。** worker 归属仅按 argv 推导，缺 PPID/worker_id 原件；错误行无 PID/rank 前缀，不能指派每条错误的具体来源。
- 保全 collate 在 ID 分支执行 `list(field_value)`，其他 feature 按 batch `torch.stack`。`ItemData.item_ids` 已允许 Tensor。`ResidualQuantization.model_step` 读取 `transformed_features['input_embedding']`；`predict_step` 遍历 item_ids 后逐项 `.item()`；工程 PredictionWriter 将 key/tensor 成对保存，再按 item_id 合并并检全部18357物品覆盖与无重复。
- 原 `ItemDataModule.get_dataloader` 保留 rank 文件映射、dataset `set_distributed_params`，返回单元素 tuple 内 `DataloaderWithIterationRetry`。训练每卡batch2048、workers12、shuffle=true、assign_all_files_per_worker=true、pin_memory=true、persistent_workers=true、timeout60，默认 prefetch_factor2 来自已保存安装源码。数据 shuffle seed 根据原 rank/worker_id 计算，另有 Lightning `seed_everything(42, workers=True)` 自动 worker 初始化。
- 原有效量化模型为 3 层、每层256、512维、init_buffer3072、层级训练、残差归一化、MiniBatchKMeans、SquaredEuclideanDistance、KMeansPlusPlusInitInitializer、initialize_on_cpu=false、SGD lr0.5、原 world-size loss scaling。trainer DDP2、bf16-mixed、accumulate1、max_steps30、max_epochs10。不能只检查提示词列出的几个参数。
- 原件代码与待修复前工作树逐字比较见 `baseline-inspection.json`，脚本只作 stdlib 只读核查；未安装依赖。

FD 的数量级与独立小 Tensor IPC/预取放大一致，但原捕获没有 FD target 分类或时间序列，不足以认定唯一根因。不能把名义 `12×2×2048` storage 数写成实际已分类 FD 计数。

## 正式准入前必读与必核

1. 原件冻结路径零改动。新适配只在本轮 A2 量化 DataModule 使用，不能影响T5编码、Toys或A1。比较旧新有效配置，允许差异仅为本次有明示依据的 collate target / DataModule工程封装 / 输出与身份路径 / 任务级资源保护，原模型、优化器、trainer及loader科学条件需逐项一致。
2. 同一固定真实样本与CPU fixture：数量、ID值/顺序/dtype、embedding形状/dtype/逐值、全部其他feature、ID与embedding逐行配对、ItemData接口。不能以排序后相同代替原序一致。
3. 原 `model_step` 的真实模型输入一致；真实 `predict_step` 的 keys/预测与 PredictionWriter 的 ID/值配对一致。合成/未训练模型可用于接口工程测试，但不得称真实量化科研结果。
4. 同batch独立storage数量需实际枚举并显示从碎片化ID到连续ID的下降；有界真实 DataLoader multiprocessing 测试需核出真实传输后数量与FD、没有缺失/重复和轴变化。不能只报告 `torch.stack` 形状。
5. 如加 worker_init_fn，必须保留 Lightning 实际 worker seeding。已保存环境源码说明：用户已有 worker_init_fn 会使 `workers=True` 不再自动注入。禁止单纯用FD记录函数覆盖它。记录代码不额外消费 Python/NumPy/Torch RNG，不改变原dataset的worker分配或shuffle种子。
6. 每rank在创建worker前执行本进程 soft-limit保护，值不超过原hard；保留原值/新值和实际rank、worker_id、PPID、FD数、sharing strategy、prefetch、worker数。需要具体FD增长/耗尽停止门及任务隔离证明，不改全局配置。
7. 新 A2-only 请求和依赖图：新job ID不占用、不覆盖原失败；原T5成功和embedding身份复用；旧两个controller终态，无SUBMITTING/孤儿冲突；SID门、metadata门、训练准入、训练、三eval、三attention、stats读新SID与新best身份，原34阶段历史保持原件。
8. 预算重算应包含原已消耗和本轮工程预跑。量化新正式worker wall不得超过 `6897.950524/2=3448.975262`秒且预留安全余量；总39.75 GPU小时、训练12小时、评价/attention2.5 GPU小时、工程组原额度及北京时间2026-10-19 13:40:30原绝对期限不得重新起算。
9. 新源码/配置固定发布与服务器不可变字节映射、job请求绑定、实际argv/cwd/解释器/GPU UUID/输入身份、原runner清理/锁/资源门完整，不能只测一个本地函数后启动完整量化。

本轮是用户另行明确授权的一次专项工程恢复，历史3/3轮已经耗尽的记录仍保留；不能因审核者发回发现而自行授权多次正式量化。

`FD_ENGINEERING_EQUIVALENCE_PASS` 只在上述具体代码、原始输出与请求审核通过后给出，不替代启动后DDP2、24worker、真实batch与step、FD实测，也不替代最后30step/SID/科研结果审核。

GPT_UPLOAD_REQUIRED: YES（审核回执应并入主控最终单一ZIP；本阶段没有制作最终ZIP，也不宣布完整交付。）
