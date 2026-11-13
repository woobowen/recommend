# A0 Instruments Normal：正式执行与审核入口

当前状态：RUNNING（2026-11-12 18:41 UTC启动），正式四轮与全量测试尚未完成。本文随真实证据更新，不将技术预跑记为正式结果。

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

GPT_UPLOAD_REQUIRED: NO（本次启动与预审证据已在本入口固定提交；最终运行结果仍待交付）。

## 最后预验收实际结论（2026-11-12）

- GPU源码 `a1b1618e04c8be9bb62918c90b59418c8de34f52`；CPU收口源码 `d9d0a5337e23ce0ceb8666ebdbd7ea209403f9fc`；[核心10文件字节身份](evidence/formal_det_20261112/prelaunch_source_identity.json)相同。最终RUN以本入口所在固定提交为准，提交前再核同一核心身份。
- `letter-a0-20261112-det-preflight-r2`：SUCCEEDED / exit0 / cleanup verified；两rank 12updates、真实native/extension deterministic计数一致、CP2与CP12完整保存、32条真实Validation、224模块严格full DS hot恢复、450 tensor选定checkpoint/adapter精确相同、fresh BF16严格450 key加载及16用户Beam20接口通过。
- [原始CPU收口回执](evidence/formal_det_20261112/jobs/letter-a0-20261112-det-receipt-cpu-r2/job/output/preflight_receipt.json)：累计22.0989842/60 GPU-device-minutes；稳态update中位4.769s，训练峰值allocated11.87GiB/reserved15.01GiB，推理峰值allocated16.74GiB/reserved18.46GiB；历史同协议最大长度包络reserved21.91GiB。保守条件预算22.6996小时，非保证，96h为累计硬上限。
- [训练/FA/checkpoint非作者预审PASS](review/formal_det_20261112/prelaunch_training.md)；[pipeline/runner/evaluation非作者预审PASS](review/formal_det_20261112/prelaunch_pipeline.md)。两名真实子代理通过主控桥接源码与原始证据审核，未独立SSH/GPU执行；最终全量结果尚未审核。
- 已用任务级修正2/3轮：[原因、最小修正和回执](evidence/formal_det_20261112/corrections.json)。r0预验收真实失败保留：bnb原生hot加载SCB元数据缺口；获批工程范围内版本/source-pinned临时alias修复，不改权重/量化/优化器。另有runner新日期前缀身份识别最小修正及真实超时清收fixture。
- [模型现存身份复核](evidence/formal_det_20261112/model_identity.json)、[作者后补38字段一致性](evidence/formal_det_20261112/author_supplement_check.json)、[实际Windows transport身份](evidence/formal_det_20261112/transport_identity.json)均已保留。大权重复用历史官方LFS身份，本轮核尺寸及小文件哈希，没有再次下载或无必要全量重哈希。

尚待正式运行验证：8240更新/4 Epoch、各实际Validation时点与loss、真正best选择及较早best恢复、完整24,772用户Beam20、最终原evaluator/独立复算与论文差异。本轮没有新增依赖安装或系统配置修改。

## 唯一正式作业与断线接续

RUN_COMMIT：`4c6a83d9ba42dc2fe737792759e410ef1874da76`。

job_id：`letter-a0-20261112-normal-det-seed42-r0`；accepted runner真实启动时间 `2026-11-12T18:41:18.357863+00:00`。独立supervisor与两rank工作进程已在提交SSH退出后经新连接验证存活；正式FA native/extension实参True；初始step10 loss13.1409、LR2.4096385542168677e-5、双rank grad norm64.38441467285156。

[不可变request及真实状态](evidence/formal_det_20261112/jobs/letter-a0-20261112-normal-det-seed42-r0/job/request.json)、[正式启动身份](evidence/formal_det_20261112/formal_launch.json)、[已发布RUN回读](evidence/formal_det_20261112/run_github_readback.json)、[本次逐文件固定export映射](evidence/formal_det_20261112/deployments/4c6a83d9ba42dc2fe737792759e410ef1874da76.json)。服务器目录没有.git，不称Git checkout。

服务器训练目录：`/data/bwwu/lc-rec/a0-normal-20261111/formal_seed42/training`；测试目录：`/data/bwwu/lc-rec/a0-normal-20261111/letter-a0-20261112-normal-det-seed42-r0/test`；runner日志：`/home/bwwu/grid-audits/sid-extensions-p0/jobs/letter-a0-20261112-normal-det-seed42-r0/work.log`。

在实际Git根 `/home/addaswsw/recommend-worktrees/lc-rec` 查询；以下参数已由实际--help与调用验证：

```bash
python3 lc_rec/letter/scripts/formal_control.py query --commit 4c6a83d9ba42dc2fe737792759e410ef1874da76 --job letter-a0-20261112-normal-det-seed42-r0
python3 lc_rec/letter/scripts/formal_control.py logs --commit 4c6a83d9ba42dc2fe737792759e410ef1874da76 --job letter-a0-20261112-normal-det-seed42-r0
python3 lc_rec/letter/scripts/formal_observe.py --commit 4c6a83d9ba42dc2fe737792759e410ef1874da76 --job letter-a0-20261112-normal-det-seed42-r0
python3 lc_rec/letter/scripts/formal_control.py collect --commit 4c6a83d9ba42dc2fe737792759e410ef1874da76 --job letter-a0-20261112-normal-det-seed42-r0
```

正常在线约每15分钟轻量查询；Codex离线后不声称AI继续监控，服务器runner与资源守护继续记账并自动训练→best→全量测试。重连先查原ID，运行中只恢复监控，不重新提交；完成后收集结果与独立审核。当前没有正式checkpoint/best或正式指标，不能将RUNNING写成完成。

## 运行中首轮实际验收（非最终结果）

2026-11-12 21:03:58 UTC，两rank真实完成epoch1 / step2060。全49,544条Validation于21:17:54 UTC完成，eval_loss `1.840831995010376`，eval_runtime `836.2467s`。21:18:29 UTC，checkpoint-2060完整保存，两rank保存耗时约34.945s，当前best=checkpoint-2060。

[首轮checkpoint只读核验与两rank原始里程碑](evidence/formal_det_20261112/epoch1_checkpoint_audit.json)：19成员，必需成员齐备，大小及小文件SHA匹配；没有重新加载模型或GPU重跑。21:33 UTC核对GPU进程仅为本任务rank PID3227289/3227290，两卡56–57°C。21:32 UTC训练已自动进入epoch1.102 / step2270，最近loss1.7429，资源正常。完整四轮、最终best及测试仍待完成。

## 运行中第二轮实际验收（非最终结果）

2026-11-12 23:34:25 UTC，两rank真实完成epoch2 / step4120。第二次全量Validation于23:48:11 UTC完成，eval_loss `1.6257708072662354`，eval_runtime约 `826.172s`。23:48:42 UTC，checkpoint-4120完整保存，耗时约30.804s；19成员，总18,673,256,113 bytes，经必需成员、大小与小文件SHA只读核验。

[第二轮checkpoint及原始两rank里程碑](evidence/formal_det_20261112/epoch2_checkpoint_audit.json)。阶段性best更新为checkpoint-4120；剩余两轮及最终best恢复、完整测试仍待执行。没有Early Stopping，没有运行测试选模，没有中断/恢复。Codex已在线监控超过5小时，服务端任务独立继续；用户关闭本地后重连查询原job，不提交新seed。

## 六小时独立运行复核（仍为RUNNING）

2026-11-13 00:49 UTC，原job已连续运行约6h08m，step5000 / epoch2.427，最近loss1.5035，双rank一致。当前best与最新完整恢复点为checkpoint-4120；尚无全量测试指标。

[六小时身份与独占GPU核验](evidence/formal_det_20261112/runtime_six_hours.json)：10个核心文件仍与预审绑定SHA相同，GPU compute仅为原rank PID3227289/3227290，两张完整checkpoint标记均在，温度61/58°C。request文件字节SHA `af12fffefdefb88e96e4905ab0b84c359a6bfd035b97e93e4d41c2b90999e7b2`；runner使用不含换行的规范化JSON指纹 `a6b6c6f6518d389a7c777c5f44552bdcd1e180b6a40ebb100f547f4f3499ffd3`，各自均与启动时一致。

[一次只读查询断言口径错误及核实](evidence/formal_det_20261112/six_hour_identity_query_note.json)：主控最初错误比较上述两种SHA，查询退出1；未改request、代码、配置或训练进程。随后按runner.canonical/digest与原始请求逐项比对通过，并明确标签；这不是实验重试或实质实现修正。任务级实质修正仍为2/3，原失败记录保留。
