# A0 layer28 Attention backward 专项（2026-11-12）

本轮新授权：layer28 真实边界定位、FA2 专用 deterministic 诊断、必要普通工程修复与受控恢复复核；正式 A0 NOT_STARTED。沿用历史 DELIVERY `6dad49e92ab3099d8e6f441a84f843d3e7666409` 和 GPU RUN `be15ad0a7c8b3cb8f1adac2c4668e7e6973b7164`，原 checkpoint 与旧失败全保留。

实验卡：同一原 checkpoint-2、Instruments seed42、LETTER SID/LLaMA7B、Int8/BF16、LoRA r8 七投影与 active embedding/head，micro8×GAS4×world2，ZeRO2、GC 与四轮8240-step scheduler 固定；恢复仅执行 step2→4，两次非零 LR 更新。默认方案与 FA deterministic 仅诊断比较，不正式采纳。阈值保持 abs(diff)<=1e-6+1e-3*abs(reference)，离散/RNG/scheduler exact。

预算：新任务独立最多120 GPU-device-min、8 GPU jobs、每job≤1800s、持久增量≤80GiB。复用已审核 formal_runner + Windows SSH/SCP；新根 `/data/bwwu/lc-rec/attention-20261112` 和 `/home/bwwu/lc-rec/letter-attention-20261112/releases/<RUN_COMMIT>`。预计先默认两次恢复，再绑定case的默认/确定性同进程3次×新进程3次；有信息价值才做完整FA-det配对，保留余量用于最小有证据补测。任务round0之后最多3组集中修正。

真实调用链：FastChat pinned forward 将q/k/v投影 `[8,s,4096]` view为 `[8,s,32,128]`；原apply_rotary_pos_emb gather cos/sin，对q/k执行q*cos+rotate_half(q)*sin。padding路径分别unpad q以及stack(k,v,dim2)，调用flash_attn_varlen_kvpacked_func，输出reshape→pad→o_proj。FA ctx的q/k/v/out/lse/cu/rng与真实dO在backward记录；原native dispatch output buffers dq/dk/dv原样记录，再经过unpad scatter、stack拆分、RoPE两分支累加及view回到q_proj。raw FA dQ不等于q_proj最终梯度。unpad indices唯一性实际断言。

HF4.31 LlamaModel源码调用torch.utils.checkpoint.checkpoint且不传use_reentrant，当前torch2.6默认reentrant：primary no_grad、backward重算grad enabled；绑定rank/layer/step2/yield9/phase/call，阶段计数不被重算推进。非持久rotary cache逐phase采样。CPU观测会同步CUDA，可能改变调度，局部阴性不推广为无hook的全局结论。

改动：仅opt-in resume_probe安装新Boundary；formal_train、formal_compare、compat、正式配置均保持上轮字节。新attention_control/pipeline复用既有transport、runner、锁、request/status/cleanup机制，使用独立预算根；det开关复用原install_diagnostic_attention。共享依赖包不改。所有关键张量保存原dtype连续CPU副本并记录原shape/stride/layout/offset与字节SHA、norm/max_abs；重放按原stride新建，offset/alignment/allocator上下文无法跨进程相同，明确限制。

结果、修正及非作者桥接审核将在本入口补充。当前仅启动前范围声明，未宣称定位或验收通过。
