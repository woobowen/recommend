# A1 Qwen3-0.6B-Base × LETTER Instruments Normal

本轮为用户2026-11-13明确批准的一次性完整seed42实验。当前阶段：正式路径适配，尚未GPU预验收，正式训练未启动；不能将历史PASS_TECHNICAL称为完整baseline通过。

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
