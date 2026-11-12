# A0 Instruments Normal：正式执行与审核入口

当前状态：PREPARING，尚未启动正式四轮训练。本文随真实证据更新，不将技术预跑记为正式结果。

## 本轮实验卡与批准差异

用户于2026-11-12明确批准一次性完整复现：Instruments / seed42 / LLaMA-7B LoRA / Int8主干 BF16计算 / 2×4090 / micro8×GAS4=global64 / 4 Epoch / 原response-only CE、temperature0.8 / AdamW 2e-4、wd0.01、warmup0.01、cosine / ZeRO2、GC / 原两模板Validation loss选模 / Prompt0 Beam20全24,772用户原evaluator。

配方身份：`LETTER-LC-Rec A0 / LLaMA-7B LoRA / FA2 Deterministic Backward Adaptation`。

本轮批准的FA原生`deterministic=True`旨在提高训练与完整checkpoint恢复的数值可复现性，改变Attention backward的实际数值执行路径，不声称与作者未知历史CUDA执行逐位相同；不改变结构、数据、Loss、优化器、训练任务或评价协议。正式配置显式启用，独立于诊断入口；不安装Boundary/Probe梯度张量观测，不改变全局PyTorch deterministic或cuBLAS。

保留获批LoRA A-only BF16、PEFT out-of-place residual、FastChat unpad与ZeRO2/no_sync适配，源签名和版本继续严格检查。作者原件不改。

新增存储适配：作者`save_total_limit=1`继续作为来源值记录，实际Trainer设2，保留best与最新完整恢复点；完整两rank的DS model/optimizer、RNG、loader、外部scheduler及Trainer/adapter/tokenizer文件齐备并原子写成`complete.json`后才轮换。安装版Transformers4.31结束时会删除非best（limit1），因此实际limit2是恢复安全所需。训练轨迹与eval_loss选模标准不变。大文件完成性仅校验大小，小文件还核SHA，不能冒称每次checkpoint全量内容重哈希。

历史恢复原件复用：[FA专项入口](ATTENTION_BACKWARD_DIAGNOSIS_REVIEW_ENTRY.md)，同CP独立恢复及continuous/resume两rank各458tensor exact，原容差未放宽；历史有界实测不能替代完整四轮。历史默认FA失败保留。

## 定向预验收与预算

新GPU预验收独立ledger，最多60 GPU-device-minutes，包括加载、失败、保存、推理与清收；不重置旧任务记录。一次12updates的真实正式训练实现（四轮scheduler），轻量FA实参断言、两次完整checkpoint、32条真实Validation接口、完整DS恢复及严格adapter reload、16用户原Beam20接口；明确不是完整实验。Epoch边界ReplayLoader用现有实际类AST、真实CPU DataLoaderShard定向回归，不重新调查全部旧根因。

正式唯一seed沿用持久`/data/bwwu/lc-rec/a0-normal-20261111/formal_seed42_ledger.json`，累计96h硬上限，预留清收时间；15–24h只是规划参考。无early stopping，不用测试选模，不扩域/seed/A1/Shuffle。服务器accepted runner独立拥有所有输入与完整train→best→test链路，原job_id先查再接续。

规范实际来源：`/home/addaswsw/recommend/{AGENTS.md,PROJECT_RULES.md,REMOTE_EXECUTION.md}`，不假称从同级工作树自动发现。Git根`/home/addaswsw/recommend-worktrees/lc-rec`，只发布`wbw-lc-rec`。

## 原始证据入口

- [本轮启动前服务器实测](evidence/formal_det_20261112/initial_remote.json)
- [实际环境与安装源码身份](evidence/formal_det_20261112/environment.json)
- [作者后补超参数本轮Docs只读回读](evidence/formal_det_20261112/author_supplement_readback.txt)
- [有效配置](config/formal_instruments_seed42.json)、[正式FA实现](scripts/formal_attention.py)、[训练入口](scripts/formal_train.py)、[服务器pipeline](scripts/formal_pipeline.py)、[原协议评价入口](scripts/formal_evaluate.py)。

作者固定源码：`8d0154e28de37dbb6e24871c508ad8ddb1921cda`；模型`huggyllama/llama-7b` revision `4782ad278652c7c71b72204d462d6d01eaaf7549`；SID SHA `33c87e28c52ae367985bdf494cb49d676d181e13a39497b4ad035f46111546dd`。

论文最终参照HR@10=0.1115、NDCG@10=0.0854，只有完整测试后才比较。不以指标相近证明正确，不追数值改协议。单seed、双卡累积与作者未知历史执行的差异，以及使用公开SID仅复现下游的范围最终均披露。

GPT_UPLOAD_REQUIRED: YES（当前尚无本轮固定最终交付；不是已完成声明）。
