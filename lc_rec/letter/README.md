# LETTER Instruments 身份与有界预检

本目录属于 `wbw-lc-rec`。固定上游 `HonghuiBao2000/LETTER@8d0154e28de37dbb6e24871c508ad8ddb1921cda`；原件位于 `vendor/LETTER`，逐文件 Git blob 已验证。

本轮仅核查 A（作者 SID + 作者补充 Instruments SeqRec LoRA 配置）的技术可行性；B（原 LC-Rec SID + 单任务 LoRA）仅静态和数据核查。没有正式训练或完整指标测试授权。

**单一审核入口：[REVIEW_ENTRY.md](REVIEW_ENTRY.md)。最终状态 PARTIAL_TECHNICAL_BLOCKED：CPU 数据核查通过；GPU 首次 backward 失败，0 次完整 optimizer update；96 小时可行性 UNKNOWN。三轮集中修正用尽，已停止新增 GPU 尝试。**

- `config/author_instruments.json`：作者补充记录。
- `config/probe_instruments.json`：双卡 global64 技术候选；正式训练未授权。
- `scripts/audit_data.py`：真实 A/B 资产、切分、tokenizer、mask 与长度核查。
- `scripts/gpu_probe.py`：设计上限 9 次有界技术更新，实际 0 次；原始真实 dtype 错误触发后，仅对必要 LoRA A 线性路径启用用户批准的 BF16 autocast。
- `scripts/inference_probe.py`：预备的 adapter 重载和固定少量用户 beam20 计时入口，本轮未执行。
- `scripts/resource_guard.py`：双卡设备秒累计预算、单作业时限、主机内存及新存储约束。
- `review/identity/IDENTITY_AUDIT.md`：论文/配置/Issues/源码研究身份与四类复现风险。
- `review/gpu_prelaunch_review.json`：非作者启动前审阅；不替代实际运行验收。
- `evidence/jobs/`：实际服务器请求、日志、状态和 manifest。

作者历史 PEFT/DeepSpeed 版本、模型字节身份、训练命令、checkpoint 和测试 prompt 尚未闭合。本轮工程或技术成功不代表 Table 1 指标复现。显式 BF16 适配仅获本轮预检授权，不自动成为正式 baseline 配方。
