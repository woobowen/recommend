# A1 Qwen3-0.6B-Base × LETTER Instruments Normal

本轮为用户2026-11-13明确批准的一次性完整seed42实验。A1 PRELAUNCH: PASS。唯一正式seed42已启动且两rank真实更新；A1 FORMAL TRAINING仍为RUNNING，完整实验尚未完成。以下早期阶段叙述是保留的历史记录，最新状态以本文件末尾为准。不能将预验收PASS称为完整baseline通过。

实验卡：仅替换A0下游LLaMA-7B为官方Qwen3-0.6B-Base；固定作者918 SID/交互、17训练prompt、历史20、response-only CE(logits/0.8)、Int8/BF16、r8/alpha16/dropout.05、七投影、完整active tied Embedding/Head、AdamW2e-4/wd.01/cosine/warmup.01、micro8×acc4×2=global64、4epoch/8240实际更新核验、Validation loss选best、Prompt0 Beam20全24772用户。无early stopping、无测试选模。新增预验收≤60 GPU-device-min，唯一正式seed累计≤96小时；任务级round0＋最多3轮集中修正。

固定依据：A0 DELIVERY `55a1371da28a45a2ec81ceee7bd82fd42b7192d5`；A0 GPU RUN `4c6a83d9ba42dc2fe737792759e410ef1874da76`；CPU评价RUN `3d11258daa6aa4fb2bed0cd013bca50536fe351d`。Q06-L技术GPU RUN `f29527de3b3f94ae5640d2e98faa1a7f74d2e0e8`。原件保持不变，A1源码/config/结果位于本目录；不混入六任务/global128实验条件。

- [A0冻结协议及实际CPU核验](review/a0_protocol_basis.md)
- [Qwen历史原始证据复核及恢复边界](review/qwen_historical_basis.md)
- [当前服务器/旧作业终态](evidence/initial_remote.json)
- [实际安装源码身份](evidence/runtime_source_manifest.json)
- [服务器独立pipeline](scripts/pipeline.py)、[既有transport客户端](scripts/control.py)、[Qwen推理](scripts/evaluate.py)

实际规则显式读取自 `/home/addaswsw/recommend/{AGENTS.md,PROJECT_RULES.md,REMOTE_EXECUTION.md}`；实际Git工作树为 `/home/addaswsw/recommend-worktrees/lc-rec`，分支仅 `wbw-lc-rec`。首次目录扫描已实际执行`ls -la`。原6项未跟踪文件保留不stage。

模型固定 `Qwen/Qwen3-0.6B-Base@da87bfb608c14b7cf20ba1ce41287e8de496c0cd`，模型实载荷1,192,135,096字节，SHA `cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba`。将复用现有官方原件与Qwen3隔离环境，不重新下载或安装。CPU gate将再核当前实体、环境版本与全部tokenizer文本边界。

Qwen使用原生SDPA/GC，无LLaMA FastChat、FA2 deterministic或LoRA A-only补丁。现代Trainer/PEFT默认不严格加载和排除冻结参数不能满足本轮恢复标准，正式路径显式完整保存并严格恢复。bnb热SCB元数据兼容只按当前实际版本/source核对，不改变forward、kernel、精度或研究配方。上述实现与真实跨进程恢复必须另行验收，当前未放行。

16项排名/CCE、SID诊断、固定分组、5000次PCG64用户Bootstrap及A0/A1配对分析均按A0冻结协议；尚无A1正式预测或指标。不得把A0实测的非法/重复/碰撞数硬编码为A1门槛。

GPT_UPLOAD_REQUIRED: NO（当前已发布材料只用于准备阶段核查；最终结果、审核、固定交付和ZIP仍未完成）。

## 本轮CPU门槛与round1

CPU job `letter-a1-q06-20261113-cpu-r0`，源码 `a9d36910e20b95ab7df6622f29479b5d542f0b68`，真实SUCCEEDED/exit0/cleanup verified。完整权重SHA、当前依赖、tokenizer全部9922物品及保存重载、全部长度分母、真实EOS/mask、CPU Int8 cold/hot序列化契约、A0配对CSV/NPZ当前SHA均通过；CUDA未初始化。训练/验证最大184 tokens，Prompt0测试最大167。

runner timeout fixture `letter-a1-q06-20261113-runner-timeout-r0`按预期TIMED_OUT/-9，真实清收并无live；不是实验失败。第一次GPU预验收`continuous-r0`则真实FAILED：Trainer导入可选TF分支碰到Keras3/tf_keras缺失，发生在GPU/model初始化前，无更新。失败原件保留，计入同一预验收ledger。任务级round1将USE_TF/USE_FLAX设0并排除强制TF覆盖，真实CPU全Trainer/PEFT/Accelerate/DS/bnb导入通过；无安装或训练算术修改。后续GPU复核仍待完成。[修正记录](evidence/corrections.json)。

Round1 GPU复核真实失败于on_train_begin审计JSON：已确认393独立BF16 trainable、196投影、loader8240/microbatches及global64，但尚无第一microbatch/update。实际Accelerate源码把steps_per_print置正inf作为stdout日志哨兵；不是Loss/Grad Norm非有限。任务级round2只对该DS配置报告字段作显式字符串编码，保持真实engine配置、kernel和strict JSON/训练finite判定；未知非有限值继续拒绝。原失败及预算保留，下一次真实GPU复核仍未完成。

## 已验证的完整训练恢复门槛（仍未正式开训）

`continuous-r2`真实两rank完成12updates、CP2/12各23成员约4.418GB、32条Validation及196 Int8模块热strict恢复；整体runner仍保留FAILED，因为后续推理审计断言失败。`resume-r2`新进程从同CP2真实恢复至12并SUCCEEDED/cleanup verified；`compare-r2`独立CPU全状态对照PASS：每rank1363张量全部exact/零差，包括393trainable、FP32master/moments、196Int8动态；40个恢复microbatch输入/labels/loader、全部RNG、step/LR/loss/gradnorm精确相同。原容差没有放宽，也没有切换SDPA/kernel/精度。[原始完整对照](evidence/jobs/letter-a1-q06-20261113-compare-r2/run/comparison.json)。

任务级最后round3修正推理审计器：已核实native PEFT把lm_head移至modules_to_tie，并将modules_to_save重写为embed_tokens/model.embed_tokens，当前adapter与历史Q06-L整份配置规范化后同型。改为检查真实双wrapper、同BF16 active Parameter/storage、392LoRA及394完整键/shape/dtype/值；不改训练或adapter。最后推理预验收将复用已通过的CP12及adapter，不重训reference。当前正式seed仍未启动；剩余GPU推理与非作者预审门槛尚待通过。


## 最新预启动状态：真实门槛通过，正式作业仍待固定发布

`inference-r3`源码`ea922c12d5872a95180c84d5660ec853138c10a4`真实SUCCEEDED/exit0/cleanup verified；两个rank共16名不同用户，每人20条完整Beam20候选，包含真实最大167token输入。CP12→preflight_adapter全部394个tensor的键、shape、BF16、值CPU exact，文件SHA同为`095d7f3b7ed843c06a8470030d4048d1f14a42929d8a5f179f2ebdbd300a660f`。实际native PEFT冷加载的全部394键/shape/dtype/值、393独立active对象和tied双wrapper/同Parameter/storage通过。[绑定原件](evidence/jobs/letter-a1-q06-20261113-inference-r3/run/adapter_binding.json)、[rank0原始预测](evidence/jobs/letter-a1-q06-20261113-inference-r3/run/test_preflight/predictions.rank0.jsonl)、[rank1原始预测](evidence/jobs/letter-a1-q06-20261113-inference-r3/run/test_preflight/predictions.rank1.jsonl)。位置约束150/256/256/256/EOS151643正确；12step接口预测的质量不代表正式结果。

预验收live账本从5个terminal runner/cleanup回执reconcile后共590.881410 GPU-device-seconds＝9.848024device-min，包含失败、加载、保存与清收，低于60device-min。全部清收verified，formal seed42账本为空；两GPU各1MiB/无compute进程，可用RAM约256.5GB、磁盘约2.957TB。[原始账本与资源](evidence/prelaunch_budget_resources.json)。任务级三轮修正已全部使用，不重置计数；所有失败原件和终态保留。[修正记录](evidence/corrections.json)。

[真实Qwen非作者预审](review/prelaunch_qwen_runtime.md)、[服务器pipeline及CPU衔接非作者预审](review/prelaunch_pipeline.md)、[成本条件估计](review/prelaunch_cost_estimate.md)。审核者实际读取主控桥接的原始源码/运行记录/预测，没有冒称独立SSH或GPU重跑。正式GPU RUN冻结后仍须绑定E/R固定字节映射、部署回执和最终PASS gate，之后才启动唯一正式seed。

完整服务器流水线已接入：4epoch/8240真实updates→每轮49544条Validation→best strict恢复→BF16 Base+tied adapter→两rank全24772用户Prompt0 Beam20→A1独立预测SHA manifest→CPU原论文4项独立复算→16指标/CCE/SID诊断/分组/5000用户Bootstrap→独立扩展/统计核验→A0/A1配对及独立核验。CPU评价RUN先冻结，GPU RUN绑定其固定export；DELIVERY另外发布。任何A0身份常数均不冒充A1预测身份。

训练、验证、best、全量测试和CPU统计由服务器自主衔接，不依赖Codex重新打开。关闭Codex不取消任务，也不表示离线AI仍在监控；重连先查原job_id，不重复新seed。完成全部实际结果、非作者终审、GitHub固定回读及轻量ZIP验证后才达到授权终点。尚无正式A1指标，未重训A0、未运行其他seed/模型/数据域/Shuffle，未同步main，未安装包或改全局配置。


CPU评价RUN已冻结并实际GitHub回读/服务器逐文件校验：`445f1475cd1dd70d19881dbc1639bb311f7cccf7`。正式GPU RUN采用下一固定提交，训练/推理/评价核心与本版逐字节相同；两份非作者报告与实际预跑源码身份已绑定。[评价版本身份](evidence/evaluation_run_identity.json)、[GitHub回读](evidence/evaluation_github_readback.json)。唯一拟正式job_id为`letter-a1-q06-20261113-normal-seed42-r0`，尚未提交。


## 正式唯一seed42已独立启动，尚未完成

GPU RUN `0e3e2c6f5c1a76152ea0228390d94ffc1ff75ba5`；CPU评价RUN `445f1475cd1dd70d19881dbc1639bb311f7cccf7`。job_id `letter-a1-q06-20261113-normal-seed42-r0`，启动`2026-11-13T11:57:04.931972+00:00`。断开提交SSH后真实supervisor/worker/两rank持续存活，两rank真实optimizer step10→20、Loss/LR/Grad Norm有限。[启动回执](evidence/formal_start.json)、[正式PASS gate](evidence/prelaunch_review.json)、[固定GPU源码回读](evidence/formal_gpu_github_readback.json)、[正式部署映射](evidence/deployments/0e3e2c6f5c1a76152ea0228390d94ffc1ff75ba5.json)。

训练路径`/data/bwwu/lc-rec/a1-qwen3-06b-instruments-20261113/formal_seed42/training`；正式测试及CPU评价位于同根`letter-a1-q06-20261113-normal-seed42-r0`。本地可用`python3 lc_rec/a1_qwen3_06b/scripts/control.py observe --commit 0e3e2c6f5c1a76152ea0228390d94ffc1ff75ba5 --job letter-a1-q06-20261113-normal-seed42-r0`只读接续。关闭Codex不取消服务器作业；后续先查原ID，禁止重提seed。当前没有完整训练/测试指标或最终DELIVERY/ZIP，不声明最终通过。
