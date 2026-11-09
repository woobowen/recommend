# CPU Offload 新轮非作者风险检查清单

审核者：`/root/independent_review`。这是本轮一次性技术审查清单，不是持续维护的看板。旧审核包、旧数据错配和 SID 冲突结论继承，不重新运行或修补旧数据。首次命令已在新证据根执行 `ls -la`；新规则相关条款和固定 2e0e266 源码已实际重读。

本清单基于主控转交的新授权；本轮完整授权将保存在 `lc_rec/offload/README.md`，落盘后须再核文字。当前尚无新 probe 源码/运行结果，不能预写可行性 PASS。

## 本轮接受范围

- 固定 `huggyllama/llama-7b@4782ad278652c7c71b72204d462d6d01eaaf7549`、两张现场确认的 RTX4090、seed42、全参数 BF16。
- 首选 optimizer-only CPU offload；仅首选出现显存不足时才进入 parameter CPU offload 第二候选。每候选一次主尝试、最多2次真实 optimizer update；故障修正与所有候选共享统一预算。
- 可用真实 SeqRec 训练前缀完成最低更新检查；六任务定义不变，存在历史错配的辅助样本只作技术 forward/mask 探针，不据此放行正式数据复现。
- 正式 world2 × micro1 × accumulation64 = batch128 布局必须单独取得实际证据。最低 SeqRec 小 batch 成功不能替代该证据；也不能替代六任务混合/最长路径可行性。
- 每 GPU 候选作业墙钟不超过90分钟，所有尝试累计不超过120分钟；CPU MemAvailable 至少保留40GiB，不借助失控 swap。
- 模型新增下载最多20GiB；本轮新增磁盘峰值最多100GiB。禁重复下载 safetensors/bin 两套、完整 checkpoint/optimizer state 导出。
- 不运行4epoch/five-seed，不生成正式评价；本轮不要求保存重载 PASS，必须明确记录将来的完整保存/重载风险。
- 不改数据、任务正式定义、精度、冻结范围、LoRA/量化、评价/trie；不用 NVMe offload、额外GPU、系统 swap/驱动/网络设置变更。

## 启动前必须能检查的内容

1. 固定官方/依赖源码来源。DeepSpeed tag/commit、CPUAdam与ZeRO3代码、transformers4.31行为、accelerate/torch版本及真实 import 路径、编译命令/返回码/产物身份必须保存；独立环境，不改旧 GRID/旧 LC 审核环境。当前机器没有 /usr/local/cuda，只有gcc15.2和已有torch CUDA12.4运行组件，不能凭包安装成功宣称CPUAdam/NCCL可用。
2. 使用实际下载权重而非随机构造/旧 delta。两片载荷对固定HF LFS SHA逐一核对；tokenizer沿用已核固定SP载荷，930 SID token映射不变。严查 missing/unexpected 参数；不能以 strict=False 掩盖未载权重。
3. 先创建并保活 HfDeepSpeedConfig/正确zero.Init上下文，再按该版本支持路径加载/扩词表；移除旧 finetune 在ZeRO建立前将默认FP32整模 device_map 上卡的加载路径。审真实调用顺序，不能仅凭配置写stage3认定分片初始化生效。
4. 所有主干与独立 embedding/head 都应 trainable；新增930行是有意初始化，其余权重必须恢复。禁止在检查中复制整模型；必要 collective必须各rank同序调用，避免只rank0调用ZeRO完整梯度读取造成死锁。
5. 追踪实际 optimizer 类/AdamW模式、betas/eps/LR/decay、分组、梯度dtype与CPU状态位置。若绕过Trainer，需保留4.31源码的norm/bias不decay分组；单组 model.parameters(wd=.01)不能冒称同一正式配方。
6. 保存候选有效DS展开配置，明确哪些是optimizer offload、哪些parameter offload；pin_memory、buffer、overlap、persistence、gradient accumulation与clip的最终值必须能从engine回读。第二候选不能被依赖/网络/CPU内存失败误触发。
7. 固定源snapshot/完整RUN_COMMIT、两rank物理GPU UUID映射、argv/cwd、真实数据样本/模板与hash；现有独立runner需保留稳定job_id和断连后可查询状态，不盲重交。
8. 模型下载/环境构建时间单列；GPU候选预算从实际加载/计算开始到终态统一计时，包含失败、OOM、卸载与收尾，不按重开会话/换attempt重置。

## 运行与预算审查

- 外部看护须在模型加载及first optimizer state allocation前已生效；持续检查全任务两rank的CPU/Swap、MemAvailable、GPU显存和磁盘。Python训练callback尚未到达时也不能丢失CPU40GiB保护。
- MemAvailable是全机真实阈值；全局现有swap占用不是本任务新增swap，须记录基线及任务VmSwap/pswpout趋势，不能把“没有OOM”当作无swap失控。只停止确认属于本任务的进程组，不修改全局swap或终止他人任务。
- 当前stageA基线可用约239GiB，两卡空闲；这些只是该采样时刻事实，启动前仍须刷新。NUMA双路/80逻辑CPU、memlock8192KiB和numactl缺失须如实记录；是否实际pin成功应通过实际分配/engine与资源证据判断，不能单靠配置或ulimit推定。
- micro-step与optimizer update分别计数。每候选的最低batch和global128实测若都执行更新，须共享2次额度；优先用所需布局的2updates一次取得证据，不额外添加未经授权训练。
- first optimizer step可能初始化大批FP32 master/m/v；必须覆盖这个阶段的真实峰值，不能只报告模型BF16载入后显存。
- 保存全机MemAvailable最小值、各rank torch allocated/reserved与设备采样峰值、任务RSS/Swap、每阶段墙钟与CPUAdam时间；标清采样峰值和精确框架峰值的不同范围。
- 不意外触发Trainer epoch验证/保存/load_best、DeepSpeed全checkpoint、FL0PS profiler第10步、全用户test或后台下一任务。
- 最多2updates不是loss收敛/论文效果验收。需有限loss/梯度与真实更新证据：实际LR、engine/optimizer step、CPU FP32 master或optimizer状态抽样，及出现SID行和代表主干参数的梯度/更新。正式warmup可能首步LR=0，BF16舍入也会让可见权重短程不变；不能仅凭计数器或单个BF16值下结论。

## 结果和交付必须分开的结论

1. 依赖安装/CPUAdam编译/NCCL初始化成功。
2. 固定真实权重严格载入且全参范围正确。
3. 最小SeqRec路径、所测shape与所测batch的2updates成功。
4. global128布局实际成功，以及六任务forward覆盖的独立范围。
5. 六任务正式训练/最长2048路径/长期稳定与正式4epoch性能：没有对应实测就继续未知。
6. 原始资产错配/历史生成原因仍未解决；技术probe成功不会覆盖旧PARTIAL。
7. 完整checkpoint保存/转换/重载及正式生成仍未测试；新增100GiB约束不能用偷偷导出完整状态来证明。
8. 候选失败、真实退出码、每阶段预算和清理状态保留；发布固定ref/关键源码回读与实际部署hash闭环，正式ZIP独立解压CRC/成员hash/秘密/Windows路径检查。旧ZIP保留，新包不倒填旧run身份。

本审核者只读取材料并写独占审核产物，不修改共享实现、Git状态、服务器进程，不下载安装。后续审核须直接读取新代码和实际机器证据，不能只复述主控总结。
