# 网页GPT单一审核入口：双4090 LLaMA全参CPU Offload（2026-11-09）

**真实模型更新 PASS；观测SeqRec batch128 PASS；严格1:1正式六任务复现仍BLOCKED。** 本次只做技术预跑。先读[完整结果与验收](RESULTS_20261109.md)，再按下面顺序读原始材料。最终非作者结论与发布/ZIP回执必须一并核；不能从本入口的技术PASS推断全流程无偏离。

1. [机器摘要](results_summary_20261109.json)和[原始GPU job文件](evidence/gpu_20261109/)：request/status/cleanup/chain、两个rank完整记录、resource samples/summary、账本。核真实before/after、CPUAdam m/v/step、SID/backbone梯度及delta，区分首lr0、第二非零lr与人工2048零update段。
2. 实际运行源码：[probe](scripts/offload_probe.py)、[资源保护](scripts/resource_guard.py)、[独立候选包装](scripts/run_bounded_candidates.py)；实际派生diff在[review](review/official_to_probe.diff)。RUN_COMMIT `c8fbe3fc8c627935f22483935ec59f0422b03602`；[服务器固定文件映射](evidence/preparation/runtime_source_mapping_20261109.json)和[完整部署清单](evidence/deploy/deploy-c8fbe3fc8c627935f22483935ec59f0422b03602.json)。实际snapshot不是Git checkout。
3. 两候选[optimizer-only](config/ds_optimizer_only.json)、[optimizer+parameter](config/ds_optimizer_parameter.json)；[正式候选](config/formal_candidate.json)仅静态不可运行；[环境／前期安装](ENVIRONMENT.md)。最终生效数值以raw report为准。
4. [原版对照](FAITHFULNESS.md)、[联合六任务源码证据](review/JOINT_TRAINING_SOURCE_EVIDENCE.md)、[micro regrouping loss差异](review/loss_reduction_review.json)。不批准新Loss/LoRA/量化，不能用相同global batch替代严格等价证明。
5. [权重完整校验继承与轻量回读](evidence/preparation/model_identity_readback_20261109.json)、[小输入身份](evidence/preparation/small_input_identity.json)、[原件未修改](evidence/preparation/official_preservation_20261109.json)。旧CPU/数据审计沿用上一轮[原审核入口](../REVIEW_ENTRY.md)，该历史入口不被本轮改写。
6. [checkpoint/预算/数据依赖独立复核](analysis_20261109/report.md)与[源码身份/算术](analysis_20261109/findings.json)。保存/恢复/导出/生成未实测；新的报告明确纠正旧报告的转换RAM描述，不能任选旧结论。
7. [Qwen资产调查](qwen_20261109/QWEN_ASSET_REVIEW.md)、[API候选身份](qwen_20261109/candidate_summary.json)、[小文件hash](qwen_20261109/asset_checks.json)、[真实失败/中断与预算偏离](qwen_20261109/collection_attempts.json)。没有Qwen权重/训练；Base只是待批准推荐；大tokenizer cache排除发布。
8. [真实非作者最终审核](review/independent_task_review_20261109.json)。旧pre_run_closeout、旧PARTIAL资产报告是其写作截点历史，不能用来覆盖新的真实job，也不能反向伪造旧PASS。
9. [RUN GitHub回读](evidence/publication/RUN_READBACK.json)。最终DELIVERY_COMMIT等于本入口固定GitHub URL中的提交，并由ZIP `publication/DELIVERY_RECEIPT.json`及最终回复给出；避免SHA自引用。ZIP含全部必要轻量源码/证据及外部回执，`MEMBER_SHA256.json`与包外SHA/解包/Windows验证需核对。

本轮是一次性交付，不是持续维护看板。技术seed42不等于正式seed42；未跑四轮、五seed、Qwen或新研究实验。原job查询入口在完整结果中；收尾后不重启。

GPT_UPLOAD_REQUIRED: YES。请将已交付ZIP交给网页GPT核对外部回执；无需上传模型、原始数据或checkpoint。
