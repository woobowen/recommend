# A0 Instruments Normal：最终科研审核与离线扩展评价

**A0 FINAL AUDIT: PASS**（2026-11-13）。正式四轮训练及完整测试已结束，原四指标、扩展指标、CCE和用户统计均完成真实独立审核。科研结论限于本次 seed42、固定模型与公开SID下游复现；不是A1/Shuffle授权。

| 分项 | 状态 |
| --- | --- |
| A0_FORMAL_TRAINING | PASS |
| A0_ORIGINAL_FULL_TEST | PASS |
| ORIGINAL_METRICS_INDEPENDENT_AUDIT | PASS |
| EXTENDED_METRICS | PASS |
| CCE_EVALUATION | PASS |
| FINAL_NONAUTHOR_REVIEW | PASS |
| A0_FINAL_AUDIT | PASS |

## 固定身份与执行证据

- 原GPU RUN：`4c6a83d9ba42dc2fe737792759e410ef1874da76`；原job：`letter-a0-20261112-normal-det-seed42-r0`；4/4 Epoch、8240/8240 updates、四次全量Validation；best=`checkpoint-8240`，eval_loss=1.5409067869186401；原runner SUCCEEDED/exit0/cleanup verified，单attempt，无训练/测试重跑。
- CPU评价RUN：`3d11258daa6aa4fb2bed0cd013bca50536fe351d`；job：`letter-a0-20261112-final-audit-20261113-offline-r1`；[真实request/status/log/pipeline与结果](evidence/final_audit_20261113/jobs/letter-a0-20261112-final-audit-20261113-offline-r1/)。服务器固定CPU export `/home/bwwu/lc-rec/letter-a0-final-audit-20261113/releases/3d11258daa6aa4fb2bed0cd013bca50536fe351d`，[七文件逐字节映射](evidence/final_audit_20261113/deployment-3d11258daa6aa4fb2bed0cd013bca50536fe351d.json)，目录没有.git。
- 模型：huggyllama/llama-7b revision `4782ad278652c7c71b72204d462d6d01eaaf7549`；[模型既有身份](evidence/formal_det_20261112/model_identity.json)。原adapter SHA `f4651c5c6e6ed841cd3384c0c8ebfab30880e94597e515ffa21916b876001ed5`，位于 `/data/bwwu/lc-rec/a0-normal-20261111/formal_seed42/training/best_adapter`。
- 2×RTX4090 / Int8主干 BF16 / LoRA r8 alpha16 dropout.05 / 七投影模块+embed/head / AdamW 2e-4 wd.01 cosine warmup.01 / micro8×GAS4×2=64 / ZeRO2 / GC / response-only CE temperature.8。两rank真实TA/DS/optimizer 450 trainable tensors/289652736 elements及224 A-only模块已[独立核验](review/final_audit_20261113/terminal_effective_check.json)。save_total_limit作者1/实际存储2，保留best及最新完整恢复点，不改选模。
- FA2 2.7.4.post1 deterministic backward正式适配，两rank各1,054,720次native_varlen与CUDA extension deterministic=True；未扩大torch/cuBLAS deterministic。原源码/生成/配置/作者文件与GPU RUN完全一致。compat元数据有旧“probe-only”文字，真实正式审批/源码/调用已核，保留原文字不改写历史。
- 原预测仍在 `/data/bwwu/lc-rec/a0-normal-20261111/letter-a0-20261112-normal-det-seed42-r0/test/`。rank0 190712530 bytes，SHA `bdcbdde15d3ac6850708888e055aa67ae61212dbb675d9f915b9947b2a208bde`；rank1 190154918 bytes，SHA `f5d2ce538bfe86b024c38070f74cce3411a457554ae08ec610ec26987570c629`。本轮StageA完整流式SHA一次；主实现及另一非作者实现各一次必要评价读取，不重复大型哈希。
- SID `33c87e28c52ae367985bdf494cb49d676d181e13a39497b4ad035f46111546dd`；inter `0a11d47791d90483005da93c36d826679e20e4592d1d2b86518d5f56f218e743`；item `1d9938e576a8d964e6b977a2b63aa56402840a788eaa6685f26f0d910a7133f2`。

四次正式Validation记录（均COMPLETE，仅按eval_loss选模）：

| Epoch | Step | UTC完成 | eval_loss | runtime(s) |
| --- | --- | --- | --- | --- |
| 1.0 | 2060 | 2026-11-12T21:17:54.485243+00:00 | 1.840831995010376 | 836.2467 |
| 2.0 | 4120 | 2026-11-12T23:48:11.679620+00:00 | 1.6257708072662354 | 826.172 |
| 3.0 | 6180 | 2026-11-13T02:24:55.773597+00:00 | 1.5417826175689697 | 821.4312 |
| 4.0 | 8240 | 2026-11-13T04:59:42.320874+00:00 | 1.5409067869186401 | 822.1264 |

## 第一层：原论文指标及独立门槛

[原始metrics.json只读副本](evidence/final_audit_20261113/metrics.json)，[完整独立复算回执](evidence/final_audit_20261113/independent_original.json)。24772用户轴逐行对应，零失败/跳过，每人20候选，原relevance/target/原始解析/稳定排序一致。HR完全相同；NDCG均值差@5=4.3021142204224816e-16、@10=3.469446951953614e-16，源于求和次序，沿用原1e-13均值/1e-9总和容差，**不声称浮点逐位一致**；未改变原metrics。

[LETTER v2 Table1](https://arxiv.org/html/2405.07314v2)的Instruments参照：

| 指标 | 原正式结果 | 论文 | 绝对差 | 相对差 |
| --- | --- | --- | --- | --- |
| hit@5 | 0.090303568545 | 0.0913 | -0.000996431455 | -1.0914% |
| hit@10 | 0.113353786533 | 0.1115 | +0.001853786533 | +1.6626% |
| ndcg@5 | 0.077663103912 | 0.0789 | -0.001236896088 | -1.5677% |
| ndcg@10 | 0.085081132008 | 0.0854 | -0.000318867992 | -0.3734% |

接近论文不单独证明实现正确；单seed不能代表多seed稳定性。本轮不通过测试选模或追数值。

## 第二层：扩展排名与生成诊断

原SID匹配口径，HR=Recall；NDCG保留所有匹配折扣相加，MRR只取首次命中。新增指标 `NO_DIRECT_PAPER_REFERENCE`。

| K | HR/Recall | NDCG | MRR |
| --- | --- | --- | --- |
| 1 | 0.064508315840 | — | — |
| 5 | 0.090303568545 | 0.077663103912 | 0.073511760590 |
| 10 | 0.113353786533 | 0.085081132008 | 0.076551290628 |
| 20 | 0.138583885031 | 0.091465549106 | 0.078305397613 |

| K | 有效/候选 | 有效率 | 非法数/含非法用户 | 重复数/用户 | 碰撞候选/用户 | 有效SID Coverage |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | 123684/123860 | 99.857904085% | 176/150 | 0/0 | 2453/2337 | 1739/9897=17.570981105% |
| 10 | 247406/247720 | 99.873243985% | 314/241 | 0/0 | 3435/3177 | 2495/9897=25.209659493% |
| 20 | 494462/495440 | 99.802599709% | 978/724 | 0/0 | 4384/3660 | 3305/9897=33.393957765% |

所有候选保留，未过滤、去重或改顺序。各rate及其分母均在[完整指标JSON](evidence/final_audit_20261113/jobs/letter-a0-20261112-final-audit-20261113-offline-r1/results/primary/metrics.json)。Coverage是SID覆盖，不是真实物品覆盖。目录9922 items/9897 SIDs，23碰撞组（21组×2、2组×3），48items，最大组3；122名用户的目标在碰撞组。

首次目标排名1–20人数：`1598, 224, 154, 150, 111, 123, 136, 120, 105, 87, 78, 76, 73, 75, 66, 72, 59, 49, 43, 34`；miss=21339，总和24772。

## 第三层：独立CCE

| K | ItemHit | 相对SID-HR绝对变化 | ItemNDCG | 相对SID-NDCG绝对变化 |
| --- | --- | --- | --- | --- |
| 5 | 0.089738414339 | -0.000565154206 | 0.076830328389 | -0.000832775523 |
| 10 | 0.113030841272 | -0.000322945261 | 0.084317508779 | -0.000763623229 |
| 20 | 0.138139835298 | -0.000444049734 | 0.090669896888 | -0.000795652218 |

定义依据：[CCE v2 §3.1, Eq4–7/Fig2](https://arxiv.org/html/2605.25330v2)，[官方固定源码6dd5d314f62d3379e664a52c9007e7645556970c](https://github.com/Nishikata97/CollisionGenRec/blob/6dd5d314f62d3379e664a52c9007e7645556970c/model/cce.py)，4544bytes/SHA `173c257a8711a885e8f6d5abcdb0ad2510ad4fab01c1700631b5ec261f50cd2d`，MIT原件保留。P为首次目标前各SID展开大小之和，g为目标组大小，m=min(g,max(0,K-P))；ItemHit=m/g，ItemNDCG=sum(i=1..m,1/log2(P+i+1))/g。非法SID占1位置，前置重复各自展开、目标只取首次。每名正式用户与官方函数逐项比对一致；独立实现显式展开位置，266合成/随机cases含论文Fig2通过。CCE是平均身份不确定性credit，不是模型恢复出了唯一item；本轮未执行ZCR或修改原SID。

## 补充分析与不确定性

`SUPPLEMENTARY_ANALYSIS`。history=`items[:-1][-20:]`，padding不计数；热度只数原训练可见`items[:-2]`每次交互，156609条，不数展开训练记录的重复前缀，排除heldout验证/测试。正频catalog item nearest-rank tertiles得q1=5/q2=11，ties不拆；UNSEEN=0、LONG_TAIL1–5、MID6–11、POPULAR>=12，先固定规则再看指标。


| history_group | 用户数 | HR@10 | NDCG@10 | ItemHit@10 | ItemNDCG@10 |
| --- | --- | --- | --- | --- | --- |
| 0-5 | 12556 | 0.133243071 | 0.100600698 | 0.132844855 | 0.099572714 |
| 6-10 | 8652 | 0.102635229 | 0.077729041 | 0.102288488 | 0.077138866 |
| 11-20 | 3564 | 0.069304153 | 0.048253583 | 0.069304153 | 0.048000243 |

| popularity_group | 用户数 | HR@10 | NDCG@10 | ItemHit@10 | ItemNDCG@10 |
| --- | --- | --- | --- | --- | --- |
| UNSEEN | 35 | 0.000000000 | 0.000000000 | 0.000000000 | 0.000000000 |
| LONG_TAIL | 3990 | 0.001754386 | 0.001115224 | 0.001754386 | 0.001025654 |
| MID | 4132 | 0.003146176 | 0.001563437 | 0.003146176 | 0.001547593 |
| POPULAR | 16615 | 0.167800181 | 0.126194399 | 0.167318688 | 0.125081331 |

UNSEEN仅35用户（catalog9items），小样本不稳定；分组仅描述，不据此改组/选模。总体Bootstrap不能冒充各组CI。

整体用户级95% percentile CI：5000次有放回、seed20261113、PCG64、linear分位，仅反映固定模型下测试用户不确定性，不是跨训练seed方差。

| 指标 | 点估计 | 95% CI |
| --- | --- | --- |
| hit@1 | 0.064508316 | [0.061359600, 0.067616664] |
| hit@5 | 0.090303569 | [0.086670434, 0.093855966] |
| hit@10 | 0.113353787 | [0.109316971, 0.117189771] |
| hit@20 | 0.138583885 | [0.134224124, 0.142741805] |
| ndcg@5 | 0.077663104 | [0.074372336, 0.080841151] |
| ndcg@10 | 0.085081132 | [0.081778792, 0.088214156] |
| ndcg@20 | 0.091465549 | [0.088113556, 0.094576179] |
| mrr@5 | 0.073511761 | [0.070288212, 0.076599269] |
| mrr@10 | 0.076551291 | [0.073306923, 0.079632409] |
| mrr@20 | 0.078305398 | [0.075069451, 0.081341593] |
| ItemHit@5 | 0.089738414 | [0.086145648, 0.093290812] |
| ItemHit@10 | 0.113030841 | [0.109034394, 0.116865816] |
| ItemHit@20 | 0.138139835 | [0.133780074, 0.142297756] |
| ItemNDCG@5 | 0.076830328 | [0.073548922, 0.079940592] |
| ItemNDCG@10 | 0.084317509 | [0.081006333, 0.087423991] |
| ItemNDCG@20 | 0.090669897 | [0.087331382, 0.093742720] |

稳定用户轴/CSV/float64 NPZ留服务器：`/data/bwwu/lc-rec/a0-final-audit-20261113/letter-a0-20261112-final-audit-20261113-offline-r1/primary/`。CSV 6186926bytes/SHA `423309c6765395e42fc6a18a43706a4653d71692b184c65fe5755f78a34df6fb`；NPZ 67315bytes/SHA `d46590f04dcd0f8438b67ccea636b3310a61ff081e146305032c614f444a1689`。不含候选重复副本。用户轴SHA `1daf63ce4c77b0265777a46b61939006aafab0276b9106c742de602c8c7e867a`表示ordered user_id列表，与原测试完整user/target records SHA的序列化范围不同，并非轴冲突。[全结果manifest](evidence/final_audit_20261113/derived_manifest.json)。未来同用户A1/Shuffle可配对，但本轮无比较数据。

## 时间、资源与审核范围

原train stage=37172.006s（含Validation/checkpoint/best恢复），Trainer runtime=37076.5684s；Validation合计=3305.9763s；四次checkpoint保存=130.119s；最终CP完成到strict恢复的包含开销区间=25.017s，不能冒称纯恢复独占时长。训练runtime扣Validation约33770.592s，仍含存储/恢复开销。

完整test stage=8148.061s；原总runner wall=45322.209s（12h35m22s）；charged GPU-device-time=90643.889s（25.179h双卡预留计费，非kernel活跃时长）。训练peak allocated最大11.867GiB/reserved15.014GiB；推理allocated18.313GiB/reserved22.922GiB。原RAM最低可用202.36GiB，swap恒定约145.14MiB，盘最低可用2.702TiB，无资源保护触发。

单用户generate P50=0.542769s/P95=0.733554s/mean=0.576667s；双卡全test吞吐=3.0402users/s。不能把两rank用户时延相加当端到端墙钟。

本轮CPU runner累计44.563513s（含一次失败）；主评价15.029s、独立扩展7.326s、独立统计14.065s，总pipeline36.893s；主峰RSS165.62MiB；持久派生结果6997369bytes，新增GPU=0、无依赖安装/系统环境修改。原指标/SID/权重/预测/checkpoint未改变，受保护原元数据前后指纹一致。

真实非作者报告：[训练及原四指标](review/final_audit_20261113/training_review.md)、[指标/CCE](review/final_audit_20261113/evaluation_review.md)、[分组/Bootstrap](review/final_audit_20261113/statistics_review.md)。训练审核者独立读原件并补一次只读SSH有效字段核验；另外两名通过主控桥接服务器原始文件、独立编写程序执行，没有声称独立SSH/GPU复现。24772×16=396352逐用户值差0；独立5000×16=80000 Bootstrap均值核对，CI最大差4.996e-16；原均值容差不放宽。

任务级修正2/3轮：[保留的失败、原因与修正](evidence/final_audit_20261113/corrections.json)：一次CPU数据路径错误（扫描预测前失败）；一次deploy违反既有transport写路径边界（上传前失败）。未发生模型/评价方法修正，原实验一次自然完成。[机器最终摘要](evidence/final_audit_20261113/final_summary.json)。

## 固定交付与停止

[复用评价定义](EVALUATION_PROTOCOL_V1.md)及机器配置已冻结。最终DELIVERY取本文件固定GitHub链接中的完整SHA；与CPU RUN、原GPU RUN明确分离。GitHub发布逐文件回读及ZIP CRC/完整解压/每成员大小SHA/安全路径/秘密模式/Windows实测由外部交付回执绑定最终提交，避免自引用。审核ZIP不含模型/adapter/checkpoint/完整预测/原始全量数据；用户Downloads保留正式备份。

GPT_UPLOAD_REQUIRED: NO（必要源码、原始轻量证据及终审均发布至本固定GitHub提交）。本轮授权终点完成后停止；未重训/重生成/改SID/改原论文指标/启动GPU/A1/Shuffle/额外seed/其他数据集。

<details><summary>以下为启动及运行中历史入口原文，仅保留当时事实，不代表当前状态</summary>

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

</details>
