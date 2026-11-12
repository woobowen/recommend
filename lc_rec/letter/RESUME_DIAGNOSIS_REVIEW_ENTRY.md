# A0 checkpoint 恢复专项审核入口（2026-11-12）

**A0 RESUME DIAGNOSIS: ROOT_CAUSE_NARROWED**  
**A0 RESUME GATE: RESUME_GATE_FAIL**  
**A0 FORMAL NORMAL: NOT_STARTED**

本轮有界调查已收口，恢复阻塞未解除。没有发现足以支持修改保存/加载算法的已证实缺陷，没有把问题归为单一 GPU/FlashAttention 原因。没有启动正式四轮 A0、A1、Shuffle、额外 seed/数据集或正式评价。下一阶段须另获用户授权。

## 1. 先读哪些证据

- [机器汇总与五组完整比较导航](evidence/resume_20261112/final_summary.json)：固定阈值、两 rank 数值、最终参数/master/moments 失败分类、原始文件 SHA。
- [诊断矩阵及每次扩展理由](evidence/resume_20261112/diagnostic_matrix.md)、[实际调用链与观测时点](evidence/resume_20261112/source_call_chain.md)。
- [最早差异原始事件](evidence/resume_20261112/earliest_trace_raw_events.json)、[完整首 microbatch 语义对齐](evidence/resume_20261112/full_trace_comparison.json)。
- [非作者最终审核](review/resume_20261112/final_review.md)，附三份独立检查源码、结果及执行桥接限制。
- [修正轮次与所有异常](evidence/resume_20261112/corrections.json)、[最小正式文件 diff](evidence/resume_20261112/minimal_formal_diff.patch)、[正式路径字节身份](evidence/resume_20261112/formal_path_identity.json)。
- [资源/原件保护/服务器源码回执](evidence/resume_20261112/jobs/letter-resume-20261112-cpu-final-receipt/run/final_resources.json)、[源码部署回执集合](evidence/resume_20261112/deployments/)。

## 2. 固定身份与方法边界

上一任务 DELIVERY `2c13d30c61337be961dca3d3d4cc05a63a463117` / RUN `87ba6556ce7237f8ef32c690165449d0d6a28294`，原 [BLOCKED 入口](https://github.com/woobowen/recommend/blob/2c13d30c61337be961dca3d3d4cc05a63a463117/lc_rec/letter/NORMAL_REVIEW_ENTRY.md) 及全部旧 round0–3 原件保留。原 LoRA 适配依据是 [Autograd 固定审核](https://github.com/woobowen/recommend/blob/903b2971a78bcf140491669ca59004beaa628815/lc_rec/letter/AUTOGRAD_REVIEW_ENTRY.md)。

本轮实际 source commits：

| commit | 实际用途 |
|---|---|
| `822983bd50d3307cfddad6b50544d01f59e5be52` | CPU 原 checkpoint 全量检查，初始探针/Enum 支持；load-r0 启动前锁拒绝 |
| `d321dc5976d80c7b31a15814ccb2a1f4b9d56ad0` | 身份/待结束作业 guard；load-r1 |
| `af00e365eb150e3b9a4ee09fbe93a4a76f9a8a38` | repeat、default 配对、attention 隔离 |
| `be15ad0a7c8b3cb8f1adac2c4668e7e6973b7164` | **最终 GPU RUN_COMMIT**：trace 配对与 native deterministic 配对 |
| `d84088222df1f00dffef02f3c97c43efb200ccdc` | CPU inventory/state joins/原比较器/replay；其中训练、探针、配置字节沿用最终 GPU RUN |

`73e75f3138376f559142e49a0d702422590fa842` 是中间证据/CPU 工具提交，未用于新的 GPU 配方。最终 GitHub delivery ref 和固定文件回读见本次交付回执；运行代码身份由上述 RUN 及其 server export 字节映射确定，不依赖分支当前内容。

服务器固定 exports：`/home/bwwu/lc-rec/letter-resume-20261112/releases/<完整SHA>/`，均为无 `.git` 的 export。新结果：`/data/bwwu/lc-rec/resume-20261112/`。接受并独立执行的 runner SHA 为 `d5a324e16f4ec36b3e9351bd275eb70ad301fdb6d4cb4e859994d3b0041471d7`；未安装远端 Codex。

真实 Instruments train131837/valid49544、原 SID、原 LLaMA7B Int8、r8 LoRA、active embedding/head、micro8/GAS4/world2/ZeRO2、BF16、AdamW 与四轮8240-step scheduler 均沿用。这里只在 global step4 停止，continuous 保存 step2，恢复执行2→4。450 个 trainable tensor /289652736 elements；每 rank FP32 master144826368 elements。正式 JSON SHA256 始终为 `9e058cecb7151ebee3ab714ab8328af0008537156cd96684c988d89e7ee10600`。

已批准的224个 LoRA A-only BF16 autocast、PEFT out-of-place、FastChat/unpad/no_sync 原适配边界保留。正式 pipeline、runner、control、ledger、compat 与原 RUN 字节一致。没有扩大 autocast、改变 trainable 范围、Loss、优化器定义或正式精度，没有升级任何依赖。

Native deterministic 是单独的 **DIAGNOSTIC_ONLY**：`torch.use_deterministic_algorithms(True)`，进程启动前 `CUBLAS_WORKSPACE_CONFIG=:4096:8`；FlashAttention deterministic 选项仍 False。该组合未通过，不被采纳为正式设置。FA-only deterministic 配置文件虽已准备，未执行对应完整模型配对。

## 3. 最早分歧与状态连续性

| 点 | 真正时点 | 结果与覆盖 |
|---|---|---|
| S0 | 连续 step2 更新完成、native save 前 | 完整 module/optimizer/scheduler/RNG 指纹 |
| S1 | 实际 checkpoint 文件 | 原 CP20文件全量 SHA；新 default/native CP 也 CPU inventory 全量 hash |
| S2 | 新进程 strict load、zero_grad 后，loader replay 前 | S1→S2 module1221项及完整 optimizer 两 rank exact；含 masters、moments、step、映射元数据 |
| pre-forward | replay 后，下一批已 yield9，实际 forward 前 | 配对 model/optimizer/scheduler/所有已采 RNG/参数描述/残余梯度、effective CB/SCB exact；输入、标签、shape/位置 exact |
| S3 | 第一次恢复 microbatch 的 primary forward 与 recompute | trace 每 rank449项 projection input/output全部 exact；两条路径首更新4个 micro-loss exact |
| 局部 backward | 同一首次 microbatch，rank0 layer28 q projection 收到梯度 | 两份 `events.rank0.jsonl` **行351**首个已采 output-gradient hash不同；行352 q LoRA B梯度 hash不同 |
| S4 | ZeRO optimizer.step 入口，当前更新归约后、clip/Adam前 | 首更新 reduced gradients 已不同；此前 masters/moments仍 exact |
| S5 | native ZeRO optimizer.step 返回、engine增加global_steps前 | 参数、master/moments出现不同；下一个更新的 forward/loss随后不同 |

新 default 的 [S0/S1/S2/RNG joins](evidence/resume_20261112/jobs/letter-resume-20261112-cpu-suite/run/verification/state-default-cont-r2.json) 与 [native joins](evidence/resume_20261112/jobs/letter-resume-20261112-cpu-suite/run/verification/state-torchdet-cont-r3.json) 均无对应差异。Loader保存step2/yield8；pre-forward比较在yield9，与S2的早期初始化 RNG不能混比。`engine.micro_steps` 连续8/恢复0，但 GAS4取模边界相同；没有为表面计数相同而修改它。

q输出梯度形状 `[8,168,4096]` BF16；两次 norm都为 `0.009680621325969696`，完整原始字节 hash 不同。**这是首次已观测的逐位差异，未保存该梯度完整数值，不报告其单独逐元素越阈幅度。** rank0 output-gradient 28/225 exact，Parameter gradient59/450 exact；rank1首 microbatch225个output-gradient与450个Parameter gradient全部exact。高层及同层所采MLP/o/v/k分支一致。

应解释为“差异已存在于该 projection 的本地反向输入，发生在该参数梯度归约和 Adam 更新之前”，不能说 q_proj 自己就是原因，也不能说整个 backward 尚未发生任何通信：ZeRO overlap 可能同时处理其他参数。不能从最终参数差异倒推出 optimizer 保存错误。

## 4. 原阈值下的真实结果

始终使用 `abs(candidate-reference) <= 1e-6 + 1e-3*abs(reference)`；scheduler/step/RNG/离散状态 exact，未放宽容差、减少450张量或跳过失败。

| 对照 | Step3 grad norm 参考/候选 | Step4 grad norm 参考/候选 | Step4相对差 | 完整比较 |
|---|---:|---:|---:|---|
| 历史 continuous/resume |62.56051254 /62.56745529|67.86307526 /64.69631958|4.66639%|FAIL|
| 新 default continuous/resume |63.14250183 /63.11900330|67.31925201 /66.85213470|0.693884%|FAIL|
| 同旧CP两次独立resume |62.56509781 /62.58864975|63.53452682 /64.50354767|1.525188%|FAIL|
| 同旧CP两次full trace resume |62.55195236 /62.57701111|64.84837341 /64.60298157|0.378409%|FAIL|
| native deterministic continuous/resume |62.27927780 /62.24782562|70.18115997 /75.11285400|7.027091%|FAIL|

各组Step3 norm虽非逐位相同但在原阈值内；Step4都越阈。各rank后8批输入/label/位置与LR序列一致，首4个micro-loss exact，后4个出现数值差异。日志 micro-loss 是HF `training_step`返回值，**已除GAS4**。真正Adam前S4记录Step3/4有效LR分别 `4.819277108433735e-6`、`7.228915662650602e-6`；step-end日志LR是scheduler更新后的下一值，不能混称本次已用LR。

五组最终完整比较覆盖所有450 trainable、FP32 master、两moments和RNG；float tensor全部finite，要求exact的tensor/NumPy状态未报告差异。历史每rank453项浮点失败：450 trainable+master+exp_avg+exp_avg_sq。default/repeat/trace的rank0同为453，rank1为452（exp_avg_sq未越阈，不代表exact）；native两rank453。**这是失败的浮点tensor对象计数，非数百个独立训练缺陷。** EnumMeta覆盖已修复，不再混入453计数。

## 5. 被排除与仍未排除的原因

- 在已覆盖持久状态内，完整保存/strict加载的module、FP32 masters、moments、scheduler损坏或漏载，不支持作为本次首差原因。没有据此修改保存/加载算法。
- RNG/replay的实际前向前状态和后续输入一致，不支持样本错位或已采随机流不同作为解释。合成Loader regression通过，不冒称真实GPU epoch-boundary通过。
- bnb0.45.5实际保存CB/SCB；冷启动首forward将其从weight迁入MatmulLtState。已比canonical有效值一致，不能将正常迁移/空缓存直接定为缺陷。
- [真实attention隔离rank0](evidence/resume_20261112/jobs/letter-resume-20261112-attention-r2/run/isolate/isolate.rank0.json) / [rank1](evidence/resume_20261112/jobs/letter-resume-20261112-attention-r2/run/isolate/isolate.rank1.json)：同真实q/kv/dout，False/True各5次，组内/组间forward/dq/dkv hash全部exact。这是该case的阴性结果；case未显式记录layer，不能推广到layer28或确认/排除所有Flash调用。
- Native strict控制不足以消除故障，不能据此唯一归因cuBLAS，也不能保证自定义extension服从PyTorch开关。
- **UNVERIFIED_STATE**：RoPE非持久cos_cached/sin_cached/max_seq_len_cached没有实测hash；layer28内部实际q/kv/dout/dq及unpad/rotary边界未直接采集；GPU workspace/allocator/stream上下文不在序列化状态比较范围。`state_dict` exact不等于所有运行时状态exact。
- `parameters`摘要只含名字/dtype/shape/requires_grad，不单独证明共享别名。原on_train_begin断言核450 trainable与optimizer bit16 groups集合一致；该证据与别名、partition offset顺序分开。完整DS状态元数据一致也不扩大成所有地址别名已核。
- CPU复制/hash同步可能改变执行时序。两条同探针对照仍复现差异，但不能把受探针约束的结果推广为所有运行的数值分布。

最值得下一次做的唯一边界检查：绑定layer28，保存其实际attention输入/上游dout/原始dq，再逐段检查unpad/rotary及nonpersistent caches；在相同输入和跨进程条件下分别考察kernel干预。不能直接更换正式precision/optimizer/attention/dropout/容差。新观测要新增训练探针，而本任务已用尽三组修正，故本轮停止；尚余GPU预算并非必须消耗完。

## 6. 最小修复、失败与审核边界

实际必要工程修复是Enum类/成员比较支持和提交身份/待结束任务guard；正式训练只增opt-in有界探针。没有实施未经证实的“恢复算法修复”。

原 CPU-suite 自检错误地期待负例仅2条失败；Enum name/value和float可产生3条，assert失败，整作业保持FAILED。随后**直接调用同一未改动fixed formal_compare**得到五组真实FAIL，不将suite改成PASS。独立replay第一次因为输出父目录不存在而写盘FAILED；同一源码、环境、输入改用runner已存在output目录后PASS，原件保留。详见[corrections](evidence/resume_20261112/corrections.json)。

非作者通过主控执行桥接读取原始文件：审核者独立写stdlib检查、复算标量、对齐事件、核字节/预算；主控原样执行并返回原始JSON。审核者未直接SSH，不能声称现场独立GPU重跑或全仓PASS。审核结论仅支持NARROWED/FAIL及证据完整性，不支持正式恢复放行。

## 7. 全部作业、预算、原件与接续

完整job_id见[20个作业终态表](evidence/resume_20261112/job_table.md)；status/argv/退出码见[结束观测与作业清单](evidence/resume_20261112/closing_snapshot.json)。原始job request/status/work.log/cleanup在`evidence/resume_20261112/jobs/<job_id>/job/`，其run目录保存小型JSON/JSONL证据；大型原始张量只留服务器。

9个实际双卡GPU作业成功退出，8个模型作业各2或4更新，另1个attention隔离0更新；另有load-r0启动前锁拒绝。保守账目取`max(外层runner墙钟*2,pipeline含20秒清收预留的device-seconds)`，总 **92.957769/120 GPU-device-min**。没有重置累计预算或修正轮次。新增持久空间在结束观测给实际字节，始终低于80GiB；6:17回执为75,122,994,328 bytes。原CP20文件共18,673,208,961 bytes两次全量hash一致，历史源码export348个文件初查映射一致。没有删除旧资产、日志或失败记录；无新增包、工具链、环境配置。

旧CP原址：`/data/bwwu/lc-rec/a0-normal-20261111/letter-a0-20261111-acceptance-cont-r3/training/checkpoint-2`。新default/native CP位于同名新continuous job的`training/checkpoint-2`，从未覆盖旧CP。原历史四轮任务没有启动。

只恢复查询，不盲重提：在实际工作树运行 `python3 lc_rec/letter/scripts/resume_control.py query --commit d84088222df1f00dffef02f3c97c43efb200ccdc --job <原job_id>`。原runner/jobs命名空间和服务器输出在每个request/status中；任务结束不表示本地AI离线后持续分析。

轻量ZIP按[保存的包装脚本](scripts/resume_package.py)从固定DELIVERY制作，包含成员SHA清单、必要原始小证据和包外发布回执，实际CRC/完整解压/逐成员hash/危险路径/秘密扫描/Windows可访问性验证见最终包外回执。权重/大型checkpoint/原始数据不进GitHub/ZIP。
