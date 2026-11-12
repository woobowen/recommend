# A0 第28层 Attention backward：最终审核入口

本轮专项已完成真实双卡诊断、官方 FA deterministic 验证和受控恢复复核。**真实 case 的默认 FA backward 非确定性已确认；完整历史故障的唯一传播链尚未确认。诊断模式通过恢复验收，原配方仍失败。**

```text
A0 ATTENTION DIAGNOSIS: ROOT_CAUSE_NARROWED
COMPONENT FA BACKWARD NONDETERMINISM: CONFIRMED
ORIGINAL RESUME GATE: FAIL
DIAGNOSTIC VARIANT: PASS — FORMAL ADOPTION NOT AUTHORIZED
FORMAL DECISION: NEEDS_USER_DECISION
A0 FORMAL NORMAL: NOT_STARTED
GPT_UPLOAD_REQUIRED: NO
```

## 1. 核心原始证据

- [最终机器汇总](evidence/attention_20261112/final_summary.json)，由[汇总程序](scripts/attention_summary.py)读取实际回执生成。
- [默认完整恢复比较](evidence/attention_20261112/jobs/letter-attention-20261112-cpu-default-pair/run/numeric_comparison.json)：FAIL，两rank各453项浮点状态超原门槛。
- [FA-det同原checkpoint两次恢复](evidence/attention_20261112/jobs/letter-attention-20261112-cpu-det-pair/run/numeric_comparison.json)：PASS，两rank各458项Tensor全部exact。
- [FA-det continuous/resume](evidence/attention_20261112/jobs/letter-attention-20261112-cpu-cont-resume/run/numeric_comparison.json)：PASS，两rank各458项Tensor全部exact，关闭本轮Boundary、保留旧Probe。
- [默认边界逐元素比较](evidence/attention_20261112/jobs/letter-attention-20261112-cpu-default-pair/run/boundary_comparison.json)、[det边界比较](evidence/attention_20261112/jobs/letter-attention-20261112-cpu-det-pair/run/boundary_comparison.json)、[真实case隔离重复](evidence/attention_20261112/jobs/letter-attention-20261112-cpu-isolate/run/isolate_comparison.json)。
- 非作者脚本从原Tensor独立复算：[默认边界](evidence/attention_20261112/jobs/letter-attention-20261112-cpu-independent-default/run/independent_result.json)、[det边界](evidence/attention_20261112/jobs/letter-attention-20261112-cpu-independent-det/run/independent_result.json)、[隔离case](evidence/attention_20261112/jobs/letter-attention-20261112-cpu-independent-isolate/run/independent_isolate_result.json)。
- [非作者最终审核](review/attention_20261112/final_review.md)、[桥接方式](review/attention_20261112/bridge_method.md)、[独立状态及预算核算](review/attention_20261112/independent_state_budget_result.json)。没有冒称非作者直接读盘完整源码、独立GPU重跑或全量重算大型acceptance_state。
- [原checkpoint、固定源码及资源最终核验](evidence/attention_20261112/jobs/letter-attention-20261112-cpu-final-receipt/run/final_resources.json)、[17作业最终清收](evidence/attention_20261112/final_closure.json)。

所有 `jobs/<job_id>/job/` 留request/status/日志/cleanup；`run/` 留有效配置、原始事件及报告。执行成功不等于数值gate通过：默认比较CPU job exit0，但报告FAIL。

## 2. 配方、资产与固定身份

沿用历史DELIVERY `6dad49e92ab3099d8e6f441a84f843d3e7666409`、GPU RUN `be15ad0a7c8b3cb8f1adac2c4668e7e6973b7164`；更早验收DELIVERY `2c13d30c61337be961dca3d3d4cc05a63a463117`、RUN `87ba6556ce7237f8ef32c690165449d0d6a28294`全部保留。

原checkpoint只读引用：

```text
/data/bwwu/lc-rec/a0-normal-20261111/letter-a0-20261111-acceptance-cont-r3/training/checkpoint-2
```

末次对20个文件逐项流式重哈希，与历史完整清单一致。没有修改、删除或替换原件。新增continuous的CP2位于独立目录：

```text
/data/bwwu/lc-rec/attention-20261112/letter-attention-20261112-det-cont-r1/training/checkpoint-2
```

该新CP2为20文件、18,673,208,961bytes，见[输入清单](evidence/attention_20261112/resume_input_checkpoint.json)，不是历史原CP2。

固定LETTER-LC-Rec / Amazon Instruments / 作者LETTER SID，LLaMA7B Int8+BF16、LoRA r8七投影及active embedding/head（450可训练Tensor、289,652,736元素）、ZeRO-2、GC、seed42、micro8×accum4×world2=global64、四轮8240-step scheduler。训练诊断最多4updates，恢复从CP2到step4，包含2次非零LR更新。没有正式四轮训练。

原比较标准始终为 `abs(candidate-reference)<=1e-6+1e-3*abs(reference)`；scheduler/step/必要RNG/离散状态exact。未放宽阈值、删除失败检查或使用strict=False。

## 3. 第28层真实链与最早局部差异

实际[FastChat冻结源码](evidence/attention_20261112/dependency_sources/fastchat_train_llama2_flash_attn_monkey_patch.py.txt)，观测[attention_boundary.py](scripts/attention_boundary.py)：

```text
q_proj [B,S,4096] → view BSHD [B,S,32,128] → Q RoPE → unpad Q
k_proj → view BSHD → K RoPE ┐
v_proj → view BSHD ────────┴→ stack(K,V,dim=2) → unpad KV
 → flash_attn_varlen_kvpacked_func → reshape / pad → o_proj

反向：o_proj.input.grad → pad backward → FA dO
 → 原始FA dQ/dK/dV → unpad scatter → RoPE backward → view
 → q_proj.output.grad
```

不是BHSD链。padding mask使本case进入varlen KV-packed。rank0有效token719、max_seq168；rank1有效token669、max_seq113。head_dim128、BF16，q stride `[4096,128,1]`、k/v `[8192,128,1]`；dropout0、causal=True、scale0.08838834764831845、window[-1,-1]、softcap0、alibi=None。cu_seqlens、mask/索引、shape/dtype/stride/layout/offset、SHA/norm/max_abs均在原始事件和描述中；索引唯一、scatter/packing/view对应经独立CPU复算。

绑定rank/layer28/step2/micro9/phase/call1，phase各一次断言。HF4.31与torch2.6现有reentrant GC：primary no_grad，反向recompute grad-enabled。非持久RoPE cache留长度、device/dtype及内容指纹，实际cos/sin/position也有张量记录。

默认配对：q/k/v、out、LSE、cu、RNG、选项及dO exact，rank0原始dQ的2,945,024元素中只有1个差异：

```text
flat_index 2820290
reference 1.000444171950221e-10
candidate 9.958966984413564e-11
abs_diff 4.547473508864641e-13
fixed_limit 1.0000001000444172e-6
outside_tolerance 0
```

dK/dV exact。该差异沿unpad及rope.q.post.grad保留，经过RoPE backward舍入消失：rope.q.pre.grad及q_proj.output.grad exact。rank1的88项边界全部exact。符合C2，确认FA组件非确定性；**没有重现历史row351的q_proj完整分歧链**。raw dQ与q_proj最终梯度不能混称，也不能把这1个未超容差且随后消失的差异直接解释为完整453项状态失败的唯一原因。

本case未发现RoPE/unpad新增最终q_proj差异，不能排除其他调用。CPU观测同步、存盘会改变调度，阴性结果不推广到完全无hook情况。

## 4. FA专用deterministic的实际生效与隔离重放

实际 `flash-attn==2.7.4.post1`；[Python interface身份](evidence/attention_20261112/dependency_identity.json) SHA `d8eaf989a5401fedd23d11a336a7f38e67596d0a4d5a279df14cbe2d7de2ed75`；[CUDA extension身份](evidence/attention_20261112/flash_extension_identity.json) SHA `49bd9d3367051b74dd86039532a3786cdf324fd88befcd0b7b3f3aeaae1abbd2`。

复用install_diagnostic_attention，仅显式诊断开关向FastChat实际引用函数传True。未开启torch全局deterministic，CUBLAS_WORKSPACE_CONFIG为空。native wrapper位置20（第21参）、CUDA extension varlen_bwd位置21（第22参），两rank真实fa_native_dispatch/fa_extension_dispatch事件均确认True；不是只看配置。末维stride均1，无末维复制/head padding。

Profiler中默认flash_bwd_dot_do_o_kernel<true>，det为<false>并有填充路径。主kernel名称可相同；固定官方[launch源码](evidence/attention_20261112/official_flash_bwd_launch_template.h.txt)及[backward源码](evidence/attention_20261112/official_flash_bwd_kernel.h.txt)将det开关绑定grid、dQ累加缓冲及归约路径。没有反汇编，不声称定位唯一机器指令。

[隔离重放程序](scripts/attention_case_replay.py)复用真实q/k/v/out/LSE/dO/RNG/cu与调用选项，原stride重建。每rank3个新Python进程，每进程每模式3次：

| 观测 | rank0 Default | rank0 Det | rank1 两模式 |
|---|---:|---:|---:|
| 每模式重复 | 9 | 9 | 各9 |
| forward/dK/dV hash数 | 各1 | 各1 | 各1 |
| dQ hash数 | 6 | 1 | 各1 |
| dQ最大差 | 3.725290298461914e-9 | 0 | 0 |
| 原容差超限数 | 0 | 0 | 0 |

这确认真实case默认backward非确定，det在此次重复消除字节差异。新地址、offset、allocator上下文不保证与训练相同；不能扩大成所有未来调用证明。

6个非profiler样本backward中位数：rank0默认0.352816ms、det0.400480ms；rank1默认0.322160ms、det0.503968ms。两模式allocated峰值相同（rank0 114,546,176、rank1 104,752,640bytes），reserved均132,120,576。完整同CP配对rank0 allocated默认12,740,192,768、det12,740,242,432（+49,664bytes），rank1相同。模式顺序固定，有首次profiling；这些有界测量不能外推正式四轮吞吐/显存。

## 5. 完整恢复验收与因果边界

| 配对 | checkpoint | 本轮Boundary | 完整状态结果（每rank） |
|---|---|---|---|
| Default1/Default2 | 同历史CP2 | 开 | FAIL，453项超容差 |
| Det1/Det2 | 同历史CP2 | 开 | PASS，458项Tensor exact |
| Det-cont/Det-resume | 新continuous CP2 | 关，旧Probe保留 | PASS，458项Tensor exact |

两det配对各有1114个exact scalar leaves、NumPy RNG624个uint32 exact，覆盖模型/LoRA、FP32 master、Adam moments、scheduler和必要RNG。每rank master、exp_avg、exp_avg_sq各144,826,368 FP32元素。恢复后2非零LRupdates、8microbatch的输入/标签/loader顺序/loss/grad norm exact。

最后配对grad norm `63.30452346801758 / 61.98331069946289`；真实Adam入口LR `4.819277108433735e-6 / 7.228915662650602e-6`。日志post-update LR是后继值，两者不混淆；loss是Trainer返回、已除GAS4的值。完整比较报告SHA `4d51ab869eba1d64a9300a6e2e71e789cd97960c1cd0b15fcd9041e52e61464d`。

独立审核核三配对起始六类状态字段、224个量化模块canonical CB/SCB exact。continuous/resume原始bnb记录有冷/热缓存存放位置差异，不称全部非持久运行时元数据逐字一致；该差异未阻碍这组完整轨迹exact。

Default与Det完整配对分别使用9bb与0dd RUN，含公开的[观测工程修正diff](evidence/attention_20261112/observer_revision_diff.patch)，**不能声称两组完整源码仅差det布尔值**。同case native重放提供更严格单变量证据；同RUN关闭Boundary的continuous/resume降低观测同步混杂，仍保留旧Probe。整体根因NARROWED。

## 6. 最小改动、修正轮次与服务器源码

[最小运行diff](evidence/attention_20261112/minimal_runtime_diff.patch)含诊断入口、Boundary、重放/分析/回执工具；resume_probe.install只有4行条件安装，未默认改变正式训练。[修正记录](evidence/attention_20261112/corrections.json)：round1修复Bool统计量兼容缺陷（真实CPU复现），只将统计量转float，保存dtype/字节不变；真实GPU mask为Int64，默认配对不受影响。另完善extension实参记录、可选关闭Boundary及回执。见[Bool回归](evidence/attention_20261112/statistic_cpu_validation.json)和[cpu-statistic日志](evidence/attention_20261112/jobs/letter-attention-20261112-cpu-statistic/job/work.log)。

[正式文件共同身份](evidence/attention_20261112/formal_identity.json)：formal_train、formal_compare、compat、formal_runner、formal_pipeline、formal_instruments_seed42与历史DELIVERY字节一致。没有确认普通checkpoint保存/加载缺陷，**未修改恢复算法**，没有用诊断PASS冒称原配方修复。

- 默认GPU RUN：`9bb7207e4a84ec3eca02d505da53ad2a771c2ee5`。
- 后续GPU RUN：**`0dd56bc4c44d8a0cda60d7dabb7e949b611fffd5`**。
- 独立边界CPU：`cbe248b4c654a7a2f2805ad6edabcdd00adc6540`。
- 隔离审核/回执工具：`b7dca996e6e0c9e26d95429b1f9d67efe5daef4c`。
- 最终CPU回执：`ca302c109e9921f07d025ebed00517ffa59544bc`。

两GPU RUN的[关键源码快照与清单](evidence/attention_20261112/source_snapshots/manifest.json)便于ZIP独立审核。服务器是无.git的固定export，按[部署映射](evidence/attention_20261112/deployments/)逐文件SHA核验，不称checkout：

```text
/home/bwwu/lc-rec/letter-attention-20261112/releases/<完整RUN_COMMIT>/
/data/bwwu/lc-rec/attention-20261112/<job_id>/
/home/bwwu/grid-audits/sid-extensions-p0/jobs/<job_id>/
```

交付后文档/汇总变化没有用于GPU，实际GPU计算仍绑定上述RUN。

## 7. 所有GPU作业、预算和最后终态

均经既有独立runner与Windows SSH/SCP。解释器/cwd/argv/输入/GPU UUID/预算/日志/cleanup在job回执。用户关闭Codex时continuous按原ID独立继续；重开后查询收集，再只提交原定第7项，未盲重跑。7项均SUCCEEDED、exit0、cleanup verified。

| 完整job_id | 外层runner墙钟秒 | 保守device-min |
|---|---:|---:|
| letter-attention-20261112-default1-r0 | 284.332136 | 10.134370 |
| letter-attention-20261112-default2-r0 | 278.971948 | 9.955231 |
| letter-attention-20261112-isolate-r1 | 20.730044 | 1.344585 |
| letter-attention-20261112-det1-r1 | 259.842895 | 9.317820 |
| letter-attention-20261112-det2-r1 | 267.922449 | 9.589144 |
| letter-attention-20261112-det-cont-r1 | 384.862764 | 13.487237 |
| letter-attention-20261112-det-resume-r1 | 260.686511 | 9.346269 |

实际监督双卡折算58.57829157device-min；保守pipeline含每GPU20秒预留，与外层逐项取较大值，封账**63.17465463/120 device-min**。7/8 GPU jobs，最长384.862764秒<1800。round0后仅round1集中工程修正，没有重置历史次数。

最后服务器观测**2026-11-12T09:47:09.142001+00:00**（纽约04:47:09）：17项远端任务（7GPU、10CPU）全部SUCCEEDED/exit0/清收通过；两GPU各1MiB、0%，无compute进程。data51,809,720,856 + exports539,930,679 + 传输tar/job目录68,983,672 = **52,418,635,207bytes =48.81865830GiB/80GiB**。

原checkpoint、旧失败、旧exports与无关工作区文件保留；没有新增系统包、语言包、工具链或系统配置变化。

## 8. 发布、官方参考与停止点

只发布wbw-lc-rec。固定GitHub材料包含源码/有效配置、关键原始事件与张量描述/差异数值、完整比较、负面结果、独立审核及作业/部署回执；大型Tensor/模型/checkpoint留服务器，以路径/SHA引用。网页GPT二次审核所需轻量证据均在固定提交，GPT_UPLOAD_REQUIRED: NO。

内容发布回读见 `evidence/attention_20261112/publication_content_readback.json`。最终DELIVERY的SHA和回读放ZIP的FIXED_IDENTITY.json / PUBLICATION_READBACK.json及包外回执，避免Git提交自指。ZIP从最终固定Git版本构造，有逐文件SHA清单；实际CRC/完整解压/成员大小与哈希/重复与危险路径/敏感信息/整体SHA/Windows副本校验结果及真实路径见最终回复和包外validation.json。

官方参考：[FA issue414](https://github.com/Dao-AILab/flash-attention/issues/414)、[固定v2.7.4.post1接口](https://github.com/Dao-AILab/flash-attention/blob/v2.7.4.post1/flash_attn/flash_attn_interface.py)、[固定launch实现](https://github.com/Dao-AILab/flash-attention/blob/v2.7.4.post1/csrc/flash_attn/src/flash_bwd_launch_template.h)、[PyTorch2.6 reproducibility](https://docs.pytorch.org/docs/2.6/notes/randomness.html)、[Transformers案例31787](https://github.com/huggingface/transformers/issues/31787)、[DeepSpeed checkpoint](https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html)。Issue是候选假设，结论来自本轮实测。

首选待批准：正式配方显式采用FA专用deterministic backward，保留原Int8/BF16、LoRA、GC、batch和ZeRO-2。它改变数值backward路径，**未获正式采用授权**，完整四轮时间/显存成本未测。若要求唯一根因闭环，应另行授权对其他调用/微批次采样首次分歧传播链，不能扩大tiny dQ证据。

本轮诊断交付后停止。没有运行正式四轮A0、A1、Shuffle、额外seed或其他数据集；等待网页GPT二次审核及用户批准正式FA-det采用和正式A0。
