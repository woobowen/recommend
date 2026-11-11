# A0 Instruments seed42 Normal：BLOCKED 审核入口

本轮没有正式四轮训练结果。正式运行的硬性前置条件「完整 checkpoint 跨进程恢复一致性」未通过；初次执行后的三轮任务级修正已耗尽，因此没有提交正式 seed，没有全量测试、best checkpoint 或推荐指标。两个技术探针退出 0 不能代替恢复验收通过。

本文为本轮唯一入口。历史 [AUTOGRAD_REVIEW_ENTRY.md](AUTOGRAD_REVIEW_ENTRY.md) 仅代表有限技术验证。本轮机器终态见 [final_summary.json](evidence/formal_20261111/final_summary.json)，原始失败报告见 [acceptance_comparison.json](evidence/formal_20261111/acceptance_comparison.json)。

## 1. 固定身份与范围

- 仓库/分支：`woobowen/recommend` / `wbw-lc-rec`；未建立其他分支，未同步 main 或 GRID 分支。
- 本轮最后实际执行源码 `RUN_COMMIT=87ba6556ce7237f8ef32c690165449d0d6a28294`，仅用于有界 acceptance。**不存在正式训练 job_id 或正式 RUN_COMMIT。** 本入口所在 GitHub 固定提交为 DELIVERY_COMMIT；发布回读回执另随审核 ZIP 保存，避免提交自引用。
- 服务器：`bwwu@zwlab-01`，SSH alias `new-server`。固定 export：`/home/bwwu/lc-rec/letter-a0-normal-20261111/releases/87ba6556ce7237f8ef32c690165449d0d6a28294/`，没有 `.git`，不是 checkout。
- 资产/输出：`/data/bwwu/lc-rec/a0-normal-20261111/`。基础模型 `huggyllama/llama-7b` revision `4782ad278652c7c71b72204d462d6d01eaaf7549`，存储于 `/data/bwwu/lc-rec/assets/llama-7b/4782ad278652c7c71b72204d462d6d01eaaf7549/`。
- [服务器部署重新回读](evidence/formal_20261111/server_fixed_deployment_readback.json)：348 个 export 文件及 runner SHA 重新核对，并逐一与 `git show RUN_COMMIT:path` 比较通过。[RUN GitHub 源码回读](evidence/formal_20261111/run_github_readback.json) 记录三个核心文件实际字节比较。
- 唯一尝试范围是 LETTER 作者已发布 SID + LLaMA-7B LoRA，Instruments，seed42，Normal，seqrec。未重训上游、未重导 SID，未运行 A1/Qwen、A2、Shuffle、额外 seed 或其他数据集。

## 2. 方法来源、输入和配置

实际读取的依据为 [LETTER 论文 v2](https://arxiv.org/html/2405.07314v2)、[作者固定源码 8d0154e28de37dbb6e24871c508ad8ddb1921cda](https://github.com/HonghuiBao2000/LETTER/tree/8d0154e28de37dbb6e24871c508ad8ddb1921cda)、[作者 Instruments 后补参数](https://docs.google.com/document/d/1rVeu3JqGXZNjapiMi1ioXIlXdXSt8zl94FmaAFYwX4g/edit) 及 [DeepSpeed 完整 checkpoint 文档](https://deepspeed.readthedocs.io/en/stable/model-checkpointing.html)。原件保留于 [vendor](vendor/LETTER/LETTER-LC-Rec/)；不使用原 LC-Rec 六任务或 930-token SID。

有效配置：[formal_instruments_seed42.json](config/formal_instruments_seed42.json)，SHA-256 `9e058cecb7151ebee3ab714ab8328af0008537156cd96684c988d89e7ee10600`；[作者参数抄录](config/author_instruments.json)；[原 ZeRO-2 BF16 配置](vendor/LETTER/LETTER-LC-Rec/config/ds_z2_bf16.json)。目标为 4 epochs、AdamW、lr 2e-4、weight decay .01、cosine、warmup .01、history20、max2048、response-only CE/temperature .8、LoRA r8/alpha16/dropout .05、七种 projection 与 active embedding/head。双卡 micro8 × accum4 × world2 = global64。实际 loader 每 epoch 每 rank 8240 microbatch，计划 2060 updates/epoch、总8240 updates、warmup83；仅完成探针的前4步，不能据此声称尾批次或完整 epoch 轨迹已经实测通过。

作者设置原为四卡布局；当前双卡累积布局不能仅凭 global64 宣称与历史数值严格等价。本轮保留原 Trainer、数据、prompt、response mask、CE 与 min-eval_loss 选模，未进行 test 选模或调参。原生 epoch save/eval、eval_delay1、load_best_model_at_end 的正式路径尚未完整执行。

数据身份由 [初始预检](evidence/formal_20261111/initial_preflight.json) 和双 rank 实际 `rank*.json` 交叉确认：24772 users、9922 items、四位置918 SID tokens、9897 unique full SID、23 碰撞组/48 items、131837 train rows、49544 validation rows、24772 目标测试用户。

| 资产 | SHA-256 |
|---|---|
| LETTER SID index | `33c87e28c52ae367985bdf494cb49d676d181e13a39497b4ad035f46111546dd` |
| interactions | `0a11d47791d90483005da93c36d826679e20e4592d1d2b86518d5f56f218e743` |
| items | `1d9938e576a8d964e6b977a2b63aa56402840a788eaa6685f26f0d910a7133f2` |

[模型身份回执](evidence/formal_20261111/model_identity_preflight.json) 复用此前官方 LFS 校验；本轮核对 revision、配置 SHA、权重大小/mtime，未重新下载或重哈希 13GB 权重。来源边界是消费作者已发布上游资产，未复现其上游生成过程。

## 3. 代码与工程适配

- [formal_train.py](scripts/formal_train.py)：固定作者数据/模型/Trainer，严格检查450 trainable tensors、289652736 elements（448 LoRA tensors + active embed/head）；真实集体 DS save/load，外置 HF scheduler 状态单独持久化；RNG 与 sampler generator/prefetch replay。
- [compat.py](scripts/compat.py)：复用旧审核的 PEFT out-of-place、FastChat unpad 及 bitsandbytes 字段兼容。**用户已明确批准正式 A-only BF16 适配**，见 [precision_approval.json](evidence/formal_20261111/precision_approval.json)；原始错误与此前补丁审核固定在 [903b2971a78bcf140491669ca59004beaa628815](https://github.com/woobowen/recommend/blob/903b2971a78bcf140491669ca59004beaa628815/lc_rec/letter/AUTOGRAD_REVIEW_ENTRY.md)。本次实际版本、源码签名及224个 A Linear 范围通过 guard；base、B、dropout、CE 上下文未扩大。这是当前环境适配，不证明作者历史逐位等价。
- 当前进程的 Accelerator.no_sync 仅对被核实的 ZeRO-2/GAS4/world2 engine 返回 nullcontext；DS 原生 backward、loss scaling、accumulation 与 step 未改。源码 SHA 与实际计数保存于 rank 报告。此兼容不能抵消本轮恢复失败。
- [formal_pipeline.py](scripts/formal_pipeline.py)、[formal_control.py](scripts/formal_control.py)、[formal_runner.py](scripts/formal_runner.py)、[formal_ledger.py](scripts/formal_ledger.py)：固定服务端 export、稳定 job_id、防重复、身份核验、监督、阶段收口及不重置预算。终端只提交/查询；服务端没有本地文件服务依赖。
- [formal_evaluate.py](scripts/formal_evaluate.py)：保留 prompt0/beam20、逐位置 token 约束、原字符串 SID evaluator、`filter_items=False`、原重复/碰撞行为；没有改为 Trie。全量 per-user 预测与异常计数路径已实现，**本轮没有执行全量生成**。
- [formal_compare.py](scripts/formal_compare.py)：在 CPU 对所有 trainable、完整 ZeRO FP32 master/Adam moments、scheduler、step 与 RNG 比较；容差事前固定 `abs(diff) <= 1e-6 + 1e-3*abs(reference)`，未事后放宽。

真实依赖为 torch2.6.0、transformers4.31.0、accelerate0.21.0、deepspeed0.16.9、peft0.5.0、bitsandbytes0.45.5、fschat0.2.31、flash-attn2.7.4.post1。见 [实际依赖源码清单](evidence/formal_20261111/dependency_sources.json) 与同名 [原始副本目录](evidence/formal_20261111/dependency_sources/)。未安装新包、未修改共享环境/系统/SSH/VPN/驱动。原始 rank JSON 中 DS `steps_per_print` 写作 Python `Infinity`；这是原始序列化事实，严格 JSON parser 需要识别该非标准常量，不能据此认为 loss 非有限。

## 4. 实际执行、恢复失败与修正额度

完整历史原件在 [jobs](evidence/formal_20261111/jobs/)，每 job 包含 request、accepted、status、cleanup、work/supervisor log 及双 rank 证据。[corrections.json](evidence/formal_20261111/corrections.json) 保存任务级 round0—round3；没有通过重新命名重置次数。

| job_id 后缀（统一前缀 `letter-a0-20261111-`） | 固定源码前缀 | 实际结果 |
|---|---|---|
| acceptance-cont-r0 | 2843f7de98bd | FAILED，0 updates；HF scheduler 不在 DS engine 内 |
| acceptance-cont-r1 | 2ad3f2bc9290 | FAILED，0 updates；旧 Accelerate.no_sync 不兼容 DS ZeRO2 |
| acceptance-cont-r2 | 5b166e3bbfe3 | FAILED，0 updates；decorated source guard 读到 contextlib wrapper |
| acceptance-cont-r3 | 87ba6556ce72 | SUCCEEDED/exit0，连续4 optimizer steps、每 rank16 microbatch |
| acceptance-resume-r3 | 87ba6556ce72 | SUCCEEDED/exit0，原进程结束后另进程严格加载 CP2，继续2 optimizer steps至4 |
| runner-timeout-r1 | 2ad3f2bc9290 | 预期 TIMED_OUT；无GPU、TERM-resistant 独立 session 子进程身份核验后被清收 |

最后两条 GPU 作业是同一项有界恢复对照，不是两个正式 seed。首次 optimizer step 使用 LR0，仅初始化状态，不冒充有效模型更新。probe 的 callback stop 会产生 `epoch_end` 日志，但最终 epoch 仅 `0.001941747572815534`，不是完成一轮。

恢复原件路径：`/data/bwwu/lc-rec/a0-normal-20261111/letter-a0-20261111-acceptance-cont-r3/training/checkpoint-2`。完整分布式 CP 含双 rank optimizer、module、外置 scheduler、Trainer、RNG、loader 状态；[complete.json](evidence/formal_20261111/jobs/letter-a0-20261111-acceptance-cont-r3/run/training/checkpoint-2/complete.json) 记录18673205984 bytes、两rank。大 tensor 未逐一哈希，manifest 中明确为 null；不能称已完成全部大文件 SHA 校验。

另进程确实严格恢复 global_step2 和 scheduler.last_epoch2，LR/输入/标签/批次顺序/DS边界一致；8个恢复 microbatch 的前4个 loss 完全相同。后4个出现偏离，rank0有3个、rank1有1个 loss 超限；最后 grad norm 为连续67.86307525634766、恢复64.69631958007812，相差4.66639%。

全状态报告每 rank454条失败：**450个 trainable + FP32 master + Adam exp_avg/exp_avg_sq 共453个浮点比较超限；另1项 ZeRO enum 的 EnumMeta 未支持**。全部 tensor finite，记录的 RNG tensor 精确一致。schema 缺口不能解释已独立成立的 loss/gradient/参数差异；不能把454条全部称作数值错误。根因未确认，不擅自归因 optimizer 丢失或 GPU nondeterminism。未放宽容差、未修第四轮、未启动正式训练。

[CPU replay round3](evidence/formal_20261111/cpu_replay_round3.json) 的 synthetic 两rank mid/next-epoch 检查通过，但不等于真实 GPU epoch 边界恢复。恢复成功加载并继续更新与可信优化轨迹恢复是不同结论。

## 5. 持久化、预算和终态

服务端 runner 在提交 SSH 结束后独立完成探针；request/status 可由后续独立连接查询。真实 timeout fixture 验证了两条 TERM-resistant、不同 session 的本任务身份与清收。**没有正式长任务，不能声称已经完成96小时正式流程或真实训练断线后的计算恢复。**

[最终实时资源与 ledger](evidence/formal_20261111/final_resource_and_ledger.json)：2026-11-11T18:15:02Z 两卡各1MiB/0%，无计算进程；所有本轮 job 终态并有 verified cleanup；不存在 formal_seed42 目录/ledger。GPU UUID 分别为 `GPU-dfff0e95-1db6-e2f4-3431-55bb852fa0ae`、`GPU-59cac3bd-3645-668b-79c8-839e0d878893`。

预验收按 runner 起止（含加载/保存/清收）保守计费1212.328182 GPU-device-sec =20.2054697 GPU-device-min，低于60预算；正式96小时预算消耗0。单卡 observed peak reserved15040774144 bytes；每完整CP约18.67GB，原正式两份保留/三份保存峰值预算约37.35/56.02GB。最终 `/data` 可用3231172415488 bytes，`/home` 可用592210427904 bytes。未清理任何旧资产、checkpoint 或他人文件，全部失败与探针 payload 保留服务器。

## 6. 真实非作者审核与结果边界

- [方法最终审核](review/formal_20261111/method_final_review.md)，真实代理 `/root/a0_method_review`，直接读取原件、配置、数据和恢复证据，独立复算事件。
- [执行最终审核](review/formal_20261111/execution_final_review.md)，真实代理 `/root/a0_execution_review`，直接检查部署/runner/双rank日志及固定比较报告。
- [比较器作者说明](review/formal_20261111/comparator_author_note.md)，`/root/a0_evaluation` 是 evaluator/comparator 作者，不能充当这两份实现的非作者 PASS。
- [初始条件审核](review/formal_20261111/preliminary_review.json) 为历史 pending，不代表最终放行；最终均以 NO-GO/BLOCKED 结论及原始事实为准。

| 维度 | 本轮结论 |
|---|---|
| 方法实施 | 静态与局部运行证据具备；完整恢复硬门槛失败，不能验收整体正确 |
| 正式四轮、验证、best选模 | 未执行；仅前述有界探针 |
| 全量测试/异常SID统计 | 未执行；实际测试用户0，所有指标 N/A |
| 论文复现程度 | 无正式指标，不能计算与论文 HR@10=.1115/NDCG@10=.0854 的绝对/相对差异 |
| 研究结论 | 不支持单seed正式推荐结果、多seed稳定性或Shuffle结论 |

## 7. 审核备份与停止点

本固定提交公开实际代码、有效配置、原始关键日志、完整轻量比较报告、部署/资产回执及真实审核；checkpoint、模型、全量数据不重复上传。审核 ZIP 由 [formal_package.py](scripts/formal_package.py) 从 DELIVERY_COMMIT 及最终发布回执生成，包含成员 SHA-256 清单，实际检查 CRC、完整解包、成员大小/hash、重复/危险路径、敏感模式及 Windows 可访问性；真实路径/大小/整体SHA在交付回复与包外 verification JSON 中提供。

`GPT_UPLOAD_REQUIRED: NO`：网页审核所需轻量证据在本固定 GitHub 提交中；ZIP 是备份。完整张量 checkpoint 留服务器，如以后要继续数值根因调查，需新的修正/GPU授权，不能将当前报告当作已通过恢复。

本轮只尝试 A0 Instruments seed42 Normal，已在明确正确性停止条件收口。没有启动正式 A0、A1、Shuffle、额外 seed 或其他数据集；等待用户查看本次 BLOCKED 证据后重新授权。
