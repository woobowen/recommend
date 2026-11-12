# A0 layer28 Attention backward 专项：用户暂停快照

**本次是暂停保存，不是最终交付或正式训练放行。** 用户要求关闭 Codex，暂停新增工作；已经接受的服务器作业继续独立运行，未执行取消/重启。

当前已验证：

- 原 FA-default 两次同 checkpoint 恢复：原门槛 **FAIL**，两 rank 各453项浮点状态失败。
- layer28 的必要 FA 输入 q/k/v/dO/out/LSE/cu/RNG exact，rank0 native dQ有1元素差异，最大4.547473508864641e-13、原容差超限0；dK/dV exact；RoPE后q_proj梯度在本次配对exact，不能把这一处tiny差异独自等同全部轨迹FAIL。
- 绑定case的默认/确定性：每模式3个新进程×每进程3次，rank0默认dQ共6种hash、确定性仅1种，rank1两组均稳定；forward/dK/dV均稳定。非作者独立脚本直接从原Tensor复算了该结论。
- FA-det完整同CP两次恢复：**原始比较器PASS，两rank全部458项Tensor exact，参数/master/moments/scheduler/RNG及两个非零LR更新通过**；176个边界（两rank各88）exact。真实扩展入口确认True。
- 普通工程修正仅诊断Bool统计量与观察/回执完善；正式训练入口、比较器、compat、runner、正式配置均与历史DELIVERY字节一致；没有恢复算法修改。

## 继续时的唯一入口

首先重新读取本文件、`evidence/attention_20261112/pause_snapshot.json` 和规范，核对Git工作区；按原ID查询 **`letter-attention-20261112-det-cont-r1`**，不要重新提交同一任务。保存时它正在服务器独立执行，是本轮第6个GPU作业；0dd RUN，FA-det、关闭本轮Boundary、保留旧Probe，continuous stop4/save2。模型、输入及运行全部在服务器，不依赖本地Codex/SSH生命周期。

```bash
python3 lc_rec/letter/scripts/attention_control.py query --commit 0dd56bc4c44d8a0cda60d7dabb7e949b611fffd5 --job letter-attention-20261112-det-cont-r1
python3 lc_rec/letter/scripts/attention_control.py collect --commit 0dd56bc4c44d8a0cda60d7dabb7e949b611fffd5 --job letter-attention-20261112-det-cont-r1
```

若终态和完整CP2通过检查，再按原授权提交**尚未提交的最后第7个GPU作业** `letter-attention-20261112-det-resume-r1`，同0dd RUN，参数 `--kind resume --diagnostic-attention-deterministic --boundary-disabled --resume /data/bwwu/lc-rec/attention-20261112/letter-attention-20261112-det-cont-r1/training/checkpoint-2`。必须先刷新GPU/空间/剩余总预算和旧进程，成功后CPU运行原formal_compare核continuous/resume；不可盲目重跑。它是本轮最后有信息价值的对照，不启动正式四轮A0。

还需完成：continuous/resume验收；原checkpoint最后全量哈希保护回执与总GPU/磁盘封账；非作者JSON状态/预算脚本（已保存未运行）；非作者最终结论；最终审核入口重写、源码映射、GitHub最终固定发布回读和完整审核ZIP。`attention_receipt.py`和`attention_delivery.py`已保存；最终ZIP尚未生成。没有声称全部任务完成。

固定身份：

- 初始默认GPU RUN `9bb7207e4a84ec3eca02d505da53ad2a771c2ee5`。
- 后续isolate、det恢复与当前continuous GPU RUN **`0dd56bc4c44d8a0cda60d7dabb7e949b611fffd5`**。
- 非作者boundary CPU脚本提交 `cbe248b4c654a7a2f2805ad6edabcdd00adc6540`，脚本SHA `baf6d0fe48db5a3d5edf7c9eab6871ba35bd9f5dd1ebae316224e4702641202b`。
- 非作者isolate CPU与最终回执工具提交 `b7dca996e6e0c9e26d95429b1f9d67efe5daef4c`。新state/budget非作者脚本待部署/运行。
- 输出 `/data/bwwu/lc-rec/attention-20261112/`；固定export `/home/bwwu/lc-rec/letter-attention-20261112/releases/<SHA>/`；本地support `/home/addaswsw/recommend-support/letter-attention-20261112/`；job根 `/home/bwwu/grid-audits/sid-extensions-p0/jobs/`。
- 预算仍最多120 device-min、8 GPU jobs、80GiB；round0后仅使用round1工程修正，不重置预算。最新已封账部分和仍运行条目以pause_snapshot/原ledger为准，不能把运行中0秒当已免费完成。
- 新增依赖/系统配置：无。原checkpoint、历史失败、原source export全部保留；两份无关Qwen tokenizer未跟踪文件保留。
- **A0 FORMAL NORMAL: NOT_STARTED。** FA-det属于DIAGNOSTIC_ONLY，正式采用及正式A0仍须用户批准。

以下保留本轮启动说明与实验依据；接续后据最终证据重写最终入口。

---

# A0 layer28 Attention backward 专项（2026-11-12）

本轮新授权：layer28 真实边界定位、FA2 专用 deterministic 诊断、必要普通工程修复与受控恢复复核；正式 A0 NOT_STARTED。沿用历史 DELIVERY `6dad49e92ab3099d8e6f441a84f843d3e7666409` 和 GPU RUN `be15ad0a7c8b3cb8f1adac2c4668e7e6973b7164`，原 checkpoint 与旧失败全保留。

实验卡：同一原 checkpoint-2、Instruments seed42、LETTER SID/LLaMA7B、Int8/BF16、LoRA r8 七投影与 active embedding/head，micro8×GAS4×world2，ZeRO2、GC 与四轮8240-step scheduler 固定；恢复仅执行 step2→4，两次非零 LR 更新。默认方案与 FA deterministic 仅诊断比较，不正式采纳。阈值保持 abs(diff)<=1e-6+1e-3*abs(reference)，离散/RNG/scheduler exact。

预算：新任务独立最多120 GPU-device-min、8 GPU jobs、每job≤1800s、持久增量≤80GiB。复用已审核 formal_runner + Windows SSH/SCP；新根 `/data/bwwu/lc-rec/attention-20261112` 和 `/home/bwwu/lc-rec/letter-attention-20261112/releases/<RUN_COMMIT>`。预计先默认两次恢复，再绑定case的默认/确定性同进程3次×新进程3次；有信息价值才做完整FA-det配对，保留余量用于最小有证据补测。任务round0之后最多3组集中修正。

真实调用链：FastChat pinned forward 将q/k/v投影 `[8,s,4096]` view为 `[8,s,32,128]`；原apply_rotary_pos_emb gather cos/sin，对q/k执行q*cos+rotate_half(q)*sin。padding路径分别unpad q以及stack(k,v,dim2)，调用flash_attn_varlen_kvpacked_func，输出reshape→pad→o_proj。FA ctx的q/k/v/out/lse/cu/rng与真实dO在backward记录；原native dispatch output buffers dq/dk/dv原样记录，再经过unpad scatter、stack拆分、RoPE两分支累加及view回到q_proj。raw FA dQ不等于q_proj最终梯度。unpad indices唯一性实际断言。

HF4.31 LlamaModel源码调用torch.utils.checkpoint.checkpoint且不传use_reentrant，当前torch2.6默认reentrant：primary no_grad、backward重算grad enabled；绑定rank/layer/step2/yield9/phase/call，阶段计数不被重算推进。非持久rotary cache逐phase采样。CPU观测会同步CUDA，可能改变调度，局部阴性不推广为无hook的全局结论。

改动：仅opt-in resume_probe安装新Boundary；formal_train、formal_compare、compat、正式配置均保持上轮字节。新attention_control/pipeline复用既有transport、runner、锁、request/status/cleanup机制，使用独立预算根；det开关复用原install_diagnostic_attention。共享依赖包不改。所有关键张量保存原dtype连续CPU副本并记录原shape/stride/layout/offset与字节SHA、norm/max_abs；重放按原stride新建，offset/alignment/allocator上下文无法跨进程相同，明确限制。

结果、修正及非作者桥接审核将在本入口补充。当前仅启动前范围声明，未宣称定位或验收通过。

## 新证据驱动的最后对照（启动前记录）

Default pair两rank原始输入/必要FA输入exact，rank0 native dQ出现1元素4.547473508864641e-13差异且未超容差，经过RoPE后q_proj梯度exact；完整两次恢复原门槛FAIL，每rank453项浮点state失败。真实layer28隔离rank0默认9次出现6种dQ字节，det9次仅1种，forward/dK/dV均1种，rank1均稳定。这证明局部FA backward非确定性，不能将这一处微小误差直接称全部轨迹FAIL的唯一原因。

FA-det完整两次恢复的176个边界均exact，S5第二次更新后的参数与optimizer descriptor exact；原始张量完整比较器并行复核。下一步有信息价值：在同一FA-det条件下用既有probe（关闭本轮新增Boundary），continuous stop4/save2，与该完整CP2的新进程resume2→4配对，保持model/data/seed/BF16/Int8/GC/GAS/ZeRO2/四轮scheduler不变。其目的检验新hook同步之外的真实恢复轨迹，不能叫完全无hook。只使用最后两个GPU作业，预计累计<80 device-min，总jobs7/8，保留80GiB上限；不正式采纳det模式。
