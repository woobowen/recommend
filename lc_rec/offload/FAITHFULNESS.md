# 原版与 CPU Offload 派生版本的边界

本文件是静态源码结论；实际运行是否通过见 REVIEW_ENTRY 和原始作业证据。
原版来自 `../vendor/original`，固定官方提交 `e6f1b3b3f8b4fee55a49a7b3dfeba29dbfcf0669`，不修改原件。
适配复用原版 `data.py`、`utils.py`、`collator.py`、Transformers 的 LLaMA 和 Trainer，不另造替代模型。

| 项目 | 作者原版 | 本轮 CPU Offload 候选 | 忠实性结论 |
|---|---|---|---|
| 模型及初始化 | huggyllama/llama-7b；基础词表32000，normal初始化新增SID行 | 同一模型 revision `4782ad278652c7c71b72204d462d6d01eaaf7549`；真实930个SID token；先启用ZeRO3/BF16再resize | 模型身份和初始化分布不变；新增行的BF16初始化、随机数消费和舍入与原版晚转换的FP32初始化不同，不能称逐位或严格初始化轨迹一致 |
| 全参／冻结 | 全模型参数训练，输入embedding与lm_head不共享 | 所有参数requires_grad=True，保持untied；跨rank和新增行须实际检查 | 目标范围不变；是否真实更新必须看模型作业证据 |
| 六任务及采样 | 一个ConcatDataset、一个模型、一个Trainer/optimizer；样本数与随机模板来自原版loader | 正式候选保留六任务和采样；技术探针仅固定256条真实训练SeqRec prefix | 不将探针冒充六任务训练；不采用六次顺序微调。零采样值表示不限制，不是零条 |
| 优化器和更新规则 | Trainer AdamW/adamw_torch；lr5e-5、wd.01、betas(.9,.999)、eps1e-8、clip1.0；cosine/warmup.01 | Accelerate0.21映射为DeepSpeedCPUAdam；保留原Trainer衰减/免衰减参数组；AdamW模式、bias correction、FP32 master/m/v在CPU | 算法对应；AVX512/FMA与torch AdamW实现舍入不同。小张量实测只支持窄范围验证，不能代替7B模型更新 |
| 数值精度 | BF16、gradient checkpointing、ZeRO3 | BF16计算/分片不改；CPUAdam FP32主参数/状态；单独记录BF16梯度累积buffer | 不量化、不改LoRA/冻结。状态放置改变不自动证明数值同轨迹 |
| 有效batch与loss聚合 | 8rank×micro8×acc2=128；每micro对有效回答token求均值 | 2rank×micro1×acc64=128；同一CE/mask | 每个SeqRec回答均4SID+EOS时可对齐该长度因素；六任务回答长短不同时，micro1会改变样本/任务隐式权重，严格联合loss等价未解决。未经确认不重写loss或宣称1:1 |
| 损失及监督 | response-only；20件历史；总token上限2048；原模板和EOS处理 | 复用原Collator/数据类；真实20件长历史覆盖；另有明确标记的人工2048tokens无update压力段 | 同一实现；人工重复token不等于真实最长多任务路径，也不产生推荐性能结论 |
| 权重加载 | 原finetune先from_pretrained再TrainingArguments，存在先完整实例化风险 | 在from_pretrained前构造并保留TrainingArguments的ZeRO3配置；不混device_map或low_cpu_mem_usage=True；基础权重严格载入后resize | 这是分片加载工程适配；不以strict=False/忽略不匹配隐藏缺权重。实际旧行tiny slice及新行训练检查由探针记录 |
| 调度与分布式兼容 | 官方cosine，4epoch总更新步数6912，warmup70 | 局部max_steps2，但create_scheduler保持总6912；禁用Accelerate额外world-size倍增调度；只对DeepSpeedEngine采用窄no_sync兼容 | 选定依赖的必要修复；每次optimizer update只推进一次scheduler。首update lr=0、第二update lr=5e-5/70，不能拿首步无参数变化断言冻结 |
| 保存恢复 | 按epoch验证/保存，validation loss选best，最终保存 | 技术探针禁用eval/save/load_best；正式候选保留原方式且标为不可运行 | 本轮无正式checkpoint，不声称保存重载或完整optimizer恢复通过；未来必须独立验证聚合瞬时内存与磁盘预算 |
| 推理与评价 | 官方17模板、beam20、逐位置token集合约束 | 本轮不运行生成/正式指标；不换trie、不增消歧token | 旧SID碰撞/非严格全目录约束门槛保留，不由offload解决 |

DS JSON不添加optimizer/scheduler节点，保留HF创建原AdamW参数组与cosine的路径。固定整数bucket为16777216/15099494/40960，避免HF4.31的auto prefetch浮点数与DS0.16.9/Pydantic2整型字段冲突。Optimizer offload的pin_memory=False；现场memlock为8MiB，小1MiB CUDA锁页成功不能推断大规模锁页可用。第二候选仅增加CPU parameter offload，不使用NVMe。

跨双路Xeon的GPU拓扑为SYS。CPU线程、PCIe/UPI传输、主机内存余量以及保存瞬时占用仍是风险。候选脚本不自动绑定/抢占别的任务；启动前需确认双卡空闲，运行资源守卫在MemAvailable低于40GiB、swap/时间/磁盘阈值触发时仅停止本任务。

可以保持联合定义的数据流优化包括共享只读解析、惰性prefix索引或仅缓存确定性SID转换；它们主要节约CPU内存/加载时间，不能释放7B参数和Adam状态。本轮未实现这类额外数据重写。原数据类有可变字典和随机模板，不能盲共享或缓存第一次随机输出。

原始证据入口：`review/JOINT_TRAINING_SOURCE_EVIDENCE.md`、`evidence/source_audit/OFFLOAD_SOURCE_AUDIT.md`及固定版本真实源码、`review/loss_reduction_review.json`、`evidence/preparation/static_memory_budget.json`。静态预算不是显存实测；技术probe通过也不等于作者数据可用于正式复现。
