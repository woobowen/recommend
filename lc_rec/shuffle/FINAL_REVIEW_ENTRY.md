# A0/A1 LETTER-LC-Rec × test-time SID Shuffle3/Shuffle4 最终审核入口

**INTERNAL FINAL AUDIT: PASS**

**IMPLEMENTATION PASS / RUN COMPLETE / METRICS PASS / NONAUTHOR REVIEW PASS**。四组全量正式测试已成功，内部科研验收通过。本文冻结的是实际运行、评价和非作者终审。对本固定提交随后完成的GitHub回读及由它构建的ZIP校验，作为外部交付回执随最终交付提供，避免提交/ZIP哈希自引用；本文不伪造尚未发生的发布后证明。最终完整交付状态以实际回执及用户收到的最终回复为准。

## 1. 结论与读取顺序

四组各24,772固定用户、双rank各12,386用户、每人20候选，共每组495,440、总计1,981,760候选，无缺失/重复/失败/跳过。每组正式GPU生成均首试成功，CPU独立评价通过，真实退出0/清收通过，严格按A0-S3→A0-S4→A1-S3→A1-S4串行。最终series亦SUCCEEDED，双卡无compute进程。

在这套固定模型、Instruments、Prompt0、seed42非恒等置换下，四组全部16项指标低于各自Normal，64个配对用户Bootstrap区间上界均低于0。S4点估计低于S3，但没有做S4对S3的专门配对检验。不把A0/A1绝对差异解释为Shuffle因果效应。

建议依次读取：

1. [完整结果与全部区间](results/RESULTS.md)、[原精度汇总JSON](results/SUMMARY.json)。SUMMARY的PENDING标签是它在非作者终审前形成的机械统计阶段身份，不是本轮最终状态；本入口及下述终审报告给最终内部判定。
2. [非作者方法/运行终审](review/METHOD_RUN_FINAL.md)、[非作者统计/科研范围终审](review/STATISTICS_FINAL.md)。
3. [独立四组一致性程序](review/final_results_verify.py)及[真实四组执行PASS](evidence/independent_result_consistency_final.json)。
4. [实际终态、串行与部署完整回读](evidence/final_control/terminal_and_deployment.json)、[35份在线监控回执](evidence/final_control/observations)。
5. [正式启动前真实GitHub回读](evidence/control/receipts/github_prelaunch_readback.json)、[96文件部署映射](evidence/control/receipts/deployment-f0242b9f90ae23a4979d688a287734ca9f88cea1.json)、[固定请求/门槛](evidence/control/requests)。

## 2. 固定版本与对照身份

| 身份 | 完整commit |
|---|---|
| A0 GPU Normal RUN | `4c6a83d9ba42dc2fe737792759e410ef1874da76` |
| A0 CPU评价RUN | `3d11258daa6aa4fb2bed0cd013bca50536fe351d` |
| A0 Normal最终交付 | `55a1371da28a45a2ec81ceee7bd82fd42b7192d5` |
| A1 GPU Normal RUN | `0e3e2c6f5c1a76152ea0228390d94ffc1ff75ba5` |
| A1 CPU评价RUN | `445f1475cd1dd70d19881dbc1639bb311f7cccf7` |
| A1原完成状态检查 | `8957e6bbb7f99935c6b53a30e8caab4c05486d5a` |
| A1补充非作者最终审核发布 | `f0242b9f90ae23a4979d688a287734ca9f88cea1` |
| 四组SHUFFLE_RUN_COMMIT | `f0242b9f90ae23a4979d688a287734ca9f88cea1` |
| 四组EVALUATION_RUN_COMMIT | `f0242b9f90ae23a4979d688a287734ca9f88cea1` |

最终DELIVERY_COMMIT为包含本入口及最终证据的Git提交，由发布后外部回执给完整SHA；不在文件内部伪造自引用SHA。后续只有交付工具、结果证据及说明新增；服务器运行export没有改动、没有.git，不冒称checkout。最终96文件重新哈希与RUN部署一一相同。

A0原始推理入口SHA `47aa87e2d4b1073e076a3d76ef8af07682cb45fc9dd8c3fb09113e1bb4d000d7`，A1原始推理入口SHA `c08f6cceb982147009e6d602a6b693065830e33a5b24e7d40b1d79b7156981f1`，都与对应真实GPU RUN原件字节相同。每rank实际派生执行源码及diff保存于各结果的`test/derived.rank*.py/.diff`；`execution_sources.json`单独哈希实际派生字节。rank中的entry_sha字段明确指原始入口，不冒充派生源码身份。

A1此前COMPLETE_PENDING_AUDIT已用现存原始结果完成真实非作者终审：新增独立程序核24772×16逐用户结果、全部分组，差0；训练8240步/四轮Validation/选择8240/生成协议/严格tied加载及16指标/CCE/Bootstrap既有证据核验通过。[A1最终入口](../a1_qwen3_06b/A1_NORMAL_FINAL_REVIEW_ENTRY.md)、[终审原报告](../a1_qwen3_06b/review/final_20261114/FINAL_REVIEW.md)。A1 Shuffle在此前置门实际通过并发布后执行；未重训或重新全量生成Normal。

## 3. 模型、数据、Tokenizer与Adapter

两模型都使用其Normal选定step8240。A0为原BF16 unquantized Base＋LoRA分支；A1为原BF16 Base＋active tied embedding/head Adapter，严格加载、参数共享和词表断言保留。

| 资产 | 固定身份 |
|---|---|
| A0模型 | `huggyllama/llama-7b@4782ad278652c7c71b72204d462d6d01eaaf7549` |
| A1模型 | `Qwen/Qwen3-0.6B-Base@da87bfb608c14b7cf20ba1ce41287e8de496c0cd` |
| A1原始权重SHA | `cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba` |
| A0最佳adapter SHA | `f4651c5c6e6ed841cd3384c0c8ebfab30880e94597e515ffa21916b876001ed5` |
| A1最佳adapter SHA | `fe96ea74fc3c7253b0c3687239266b0f073e1a51ef9158e7d97bf30b4527bd03` |
| A0实际tokenizer vocab语义SHA | `7a2f1620a30b53a21addbcf9a0d8827c66fc91e4820cd9679ccc387ea106f22b` |
| A1实际tokenizer vocab语义SHA | `4d0e7267626dde1ab31259f328a81a4be2a923a824e2ae62c2bd74d4d01a4ccc` |
| 作者LETTER源码 | `HonghuiBao2000/LETTER@8d0154e28de37dbb6e24871c508ad8ddb1921cda` |
| SID index SHA | `33c87e28c52ae367985bdf494cb49d676d181e13a39497b4ad035f46111546dd` |
| 交互SHA | `0a11d47791d90483005da93c36d826679e20e4592d1d2b86518d5f56f218e743` |
| 物品SHA | `1d9938e576a8d964e6b977a2b63aa56402840a788eaa6685f26f0d910a7133f2` |
| 固定用户轴SHA | `1daf63ce4c77b0265777a46b61939006aafab0276b9106c742de602c8c7e867a` |
| 全用户共享置换manifest | 4,952,762 bytes；`f936c87e07a48413fb3012d0afd1c05fad8f50823b0c0bd5ea6048bb6bb37be1` |

9922物品、9897不同SID、918新增SID符号；172929个可见历史组，历史`items[:-1][-20:]`，真实历史长度4–20。CPU定向用例另覆盖0/1组、重复物品、padding、不同tokenizer边界及非法额外SID等。[A0完整CPU/Tokenizer证据](evidence/cpu-bound-r0/a0/audit.json)、[A1完整CPU/Tokenizer证据](evidence/cpu-bound-r0/a1/audit.json)含所有保存Tokenizer文件SHA及全轴长度、输入摘要、样本和manifest定位。

完整真实base/adapter等13文件共15,884,238,960 bytes的后续CPU-only哈希核验通过，耗时46.968937秒，无模型加载或GPU使用：[实际hash报告](evidence/asset_hashes.json)、[来源/请求/终态](evidence/assets)。两个A0权重分片SHA为`d43476fdd2fca0c44d55ee930039dd5dafb6331764dc0b5e5f89c60b551fcc12`及`94712c4b457815e0f8aaa4f78170aebba2b9ee3339ff2b40fceb983338fbafee`。原件读取前后inode/size/mtime/ctime稳定。它与A1原终审当时的元数据/加载核验是不同时间的证据，未回写历史声明。

## 4. 实施正确性与OFF/ON门

[entry.py](entry.py)校验固定原入口，最小派生补丁；[runtime.py](runtime.py)把完整Prompt原tokenization得到的`input_ids`在传入原生`generate`前交给[core.py](core.py)。按冻结SID词表枚举输入中全部SID token，再按实际历史物品顺序逐组严格匹配四ID；兼容两Tokenizer各自非SID边界。仅组内ID位置变化，原SID Token身份不改、不重新加offset，不跨物品或目标洗牌。

S3前三位从5种非恒等排列均匀选，第四位固定；S4四位从23种非恒等排列选，不强迫所有位置每次都移动。协议`GRID-LETTER-SHUFFLE-V1`，shuffle_seed42；compact JSON UTF-8 `[protocol,42,n,row_idx,visible_item_idx,counter]`→SHA256 unsigned big-endian→拒绝采样排除模偏→字典序非恒等目录；`new[j]=old[perm[j]]`。不依赖rank、全局RNG、worker或batch次序；同用户同出现位置跨模型共用，生成期间不重抽。

实际CPU全用户门通过；每模型32真实分层用户、双rank的GPU门通过：

- 原冻结入口的原collate张量、原回调和AST提取的原生generate表达式，旁路本次transform/重建/ShuffleAllowed，与适配OFF对比。不是另起一个完整原CLI运行。
- 输入ID、mask、原生embedding、20条SID、score和排序exact；与历史保存预测最大score差0，预声明容差1e-5未放宽；target及16指标核对通过。
- ON真实S3/S4输入、组外不变、多重集合/目标守恒；`Embedding(Shuffle(IDs)) == Shuffle(Embedding(IDs))`实际查表exact。没有另外维护正式Embedding干预流水线。
- 模型prefill hook核实际送入新IDs、原生新序列position、初始cache长度0；参数状态不变。正式运行不传旧position_ids、不传旧KV，不改原生RoPE、causal mask或Attention类。
- A1保留原Allowed对原Prompt/Response边界的验证，再验证授权差异并绑定实际新前缀；逐码位合法约束/EOS保留。A0按自身回调分别适配。

[GPU验收真实回执](evidence/preflight_final_receipt.json)、[实际分rank验收](evidence/preflight-r1)、[技术预审](review/PRELAUNCH_REVIEW.md)。正式四组的[独立输入程序](review/independent_inputs.py)又从原始数据/Normal/Shuffle原预测逐行重放全部用户及候选前缀：每组172929组、非恒等率100%、输入改变率100%，无Prompt/目标污染。全量组外不变与守恒以逐行检查通过为依据，不以指标变化代替实施证据。

## 5. 生成、评价与统计保持项

Prompt0、最近20历史、对应最佳Adapter、原有效batch1/rank、WORLD_SIZE2、BF16、beam20/return20、max_new_tokens10、原EOS、原逐码位约束、sequence score稳定降序、filter_items=False、无新增去重，全部沿用Normal。两rank实际原DistributedSampler设置亦记录，无单卡重写或并行争卡。

[统一协议](../letter/EVALUATION_PROTOCOL_V1.md)、[机器定义](../letter/config/a0_evaluation_v1.json)数学未变。CCE仅评价层，官方`Nishikata97/CollisionGenRec@6dd5d314f62d3379e664a52c9007e7645556970c`，原函数SHA `173c257a8711a885e8f6d5abcdb0ad2510ad4fab01c1700631b5ec261f50cd2d`；前置碰撞展开、首次目标SID、均匀credit、非法占位、重复前置组及K截断保持。每组官方CCE全用户比较通过，无ZCR/生成过滤。

每组原始预测独立复算396352个逐用户metric值，全差0；作者原四指标差0。历史组0–5/6–10/11–20人数12556/8652/3564；热度UNSEEN/LONG_TAIL/MID/POPULAR人数35/3990/4132/16615，训练热度仅`items[:-2]`、156609可见交互，阈值5/11，全16分组不变。目标碰撞用户122、23个碰撞SID组/48物品；Coverage分母9897。

用户Bootstrap5000次、PCG64、seed20261113、95% linear percentile；Normal和相应Shuffle同用户配对。独立每组80000均值复核，draw最大差7.771561172376096e-16，CI最大差5.412337245047638e-16；配对draw最大差2.706168622523819e-16、CI最大差1.5265566588595902e-16，原tol1e-13；point/groups差0。没有分组CI，也不把用户抽样方差称模型训练或Shuffle实现方差。

## 6. 服务器身份、原始结果与恢复入口

实际服务器`zwlab-01`，既有Windows SSH/SCP通道`new-server`，用户bwwu/uid1005，原runner SHA `1fefc09a15c1cb73878417f9eb50d0fe9e8fa1b2f60cc70b27f7cc4ed9cb0947`。[实际transport版本绑定](evidence/transport_identity.json)；没有改系统SSH/代理/驱动/框架。GPU UUID：

- `GPU-dfff0e95-1db6-e2f4-3431-55bb852fa0ae`
- `GPU-59cac3bd-3645-668b-79c8-839e0d878893`

series job：`letter-shuffle-instruments-seed42-20261114-series-r0`。作业目录根`/home/bwwu/grid-audits/sid-extensions-p0/jobs/`；结果根`/data/bwwu/lc-rec/letter-shuffle-20261114/`；源码export`/home/bwwu/lc-rec/letter-shuffle-20261114/releases/f0242b9f90ae23a4979d688a287734ca9f88cea1`。

四个正式job_id（前置日期是原runner支持namespace，实际执行UTC是2026-11-14）：

- `letter-a0-20261112-shuffle3-instruments-seed42-20261114-r0`
- `letter-a0-20261112-shuffle4-instruments-seed42-20261114-r0`
- `letter-a1-q06-20261113-shuffle3-instruments-seed42-20261114-r0`
- `letter-a1-q06-20261113-shuffle4-instruments-seed42-20261114-r0`

每组原始预测在结果根`<job_id>/test/predictions.rank{0,1}.jsonl`；所有原score、原顺序、实际输入/排列摘要均保留。原始大预测/模型/Adapter不进入GitHub或ZIP。每组[results目录](results)保存实际request/status/cleanup、派生源码/diff、rank报告、预测大小/SHA、全部评价、独立复算和资源；JSONL大文件通过下表完整身份定位。

恢复仅查询原job：`python3 lc_rec/shuffle/control.py query --job <job_id>`。本轮所有任务已终结，不需要重启、重投或继续训练。

## 7. 成本、失败及范围

正式GPU阶段含加载/生成/退出合计25222.07205544319秒=7.006131小时；GPU-device50444.14411088638秒=14.012262小时；四runner实际墙钟25638.797335秒，series总墙钟25706.934179秒，均低于18小时。单组A0<4h、A1<3h。CPU正式评价另计414.618596秒，预跑含失败/清收471.270312device秒=7.8545052device分钟<60。主控终审及发布/ZIP发生在正式结束之后，实际交付UTC随外部回执记录，不冒充GPU时间。

A0最高allocated18.313GiB、reserved22.921875GiB；A1最高allocated约2.003GiB、reserved2.322266GiB。最低MemAvailable251857805312 bytes，Swap_used最大152190976 bytes，正式资源采样最低磁盘2945681059840 bytes；最终磁盘快照另存。逐组load/gen/CPU/runner、P50/P95/吞吐、每卡峰值、RAM/Swap/磁盘见完整结果，模型大小差异不构成单独架构或Shuffle速度因果解释。

任务round0初查后3轮额度全部使用：[完整失败与修正记录](review/CORRECTIONS.md)。R1参数缩写与真实native OFF补证，R2已有runner支持namespace及有界CPU清收probe，R3新增交付collector本地argv过长，最小路径/摘要适配后立即通过。R3没有远端调用/GPU或研究变化。[R3真实回执](evidence/delivery_capture_round3.json)。四组正式GPU均一次成功；没有以新job_id重置修正或预算。

非作者技术审核者`/root/shuffle_non_author_review`，统计审核者`/root/a1_non_author_audit`，均为真实独立子代理，未参与被审的主实现/运行控制。它们工具不可用，依赖主控桥接真实源码/回执，独立程序由它们编写、主控落盘并服务器执行；不冒称独立SSH、完整原CLI/GPU重跑或本地扫描服务器原始大文件。

原Token身份保留，Embedding仍可能隐式透露层级。本实验仅排除额外显式旧位置/cache旁路；不取消层级身份、不证明严格置换不变、也不能用下降单独证明语义。用户Bootstrap条件于冻结模型及单次固定置换实现，不估计实现之间的随机方差。

## 8. 授权停止与交付说明

未重新训练A0/A1、未重训/重建SID、未扩模型/数据域、未加训练seed、未增加Shuffle随机实现或挑种子、未改写Normal预测/指标、未同步main/yinjingchen、未进入新研究阶段。没有安装系统/语言包、工具链或字体；没有改全局配置。大资产/完整预测留new-server，Windows/WSL只收轻量审计材料。原有无关未提交内容保持。

完整内部终审之后发布本固定提交；经GitHub重新读ref/入口/实际源码/配置/结果/manifest/非作者报告并比较固定blob字节，再生成轻量ZIP。ZIP包含必要外部回读，成员SHA清单不自哈希；ZIP总SHA及验证回执放包外。实际CRC/全解压/逐字节/危险重复路径/秘密扫描、Windows native哈希和用户Downloads保留备份通过后，最终回复给路径/大小/SHA/完整DELIVERY_COMMIT与最终完整状态。若外部发布回执只随包提供，明确`GPT_UPLOAD_REQUIRED: YES`。

## 9. 四组原始预测大小与SHA-256

| 条件 | rank | bytes | SHA-256 |
|---|---:|---:|---|
| a0-S3 | 0 | 233861723 | `299ca814ec8b60a823e7033b3d858a238632312815b318631f15bb3de5c27aff` |
| a0-S3 | 1 | 233050781 | `5fa13b0dfa30e968d353623bdafcca4cbb1464d9f0ef53142fb72ebff94c5e38` |
| a0-S4 | 0 | 233912996 | `7cca6bbb87be051cbc1c3b9891d45d8360cff52e213288ab9762ec3aeccf5b13` |
| a0-S4 | 1 | 233104661 | `d7d2439831bfb92617d13602c5e61695e1b88bb53e77a2eeaa9002b1671e8871` |
| a1-S3 | 0 | 228029208 | `9fbc24356382b1c1f9ae80a6143c07cb559ed3eb731240ffb4e886dcfb18fb90` |
| a1-S3 | 1 | 227264894 | `5c27ac63bdd9cd6cd63462994cfcebe8d67d780417fa5d80feb833f58b5cef2c` |
| a1-S4 | 0 | 228067756 | `5b4c5de6684f1ff59deeaf7a783a3af7e90582bdab782eed30cfffe03d45f77a` |
| a1-S4 | 1 | 227303791 | `da8053461db000e11562dfdbf94d1522f247c9341b6988e0a9fe8532110b2056` |

Normal预测仍只读保留：A0 rank0 `bdcbdde15d3ac6850708888e055aa67ae61212dbb675d9f915b9947b2a208bde`、rank1 `f5d2ce538bfe86b024c38070f74cce3411a457554ae08ec610ec26987570c629`；A1 rank0 `790506cf834bca27f4bb01d35d7d4048d494872f74ff4d9a8bcdac516c475b4d`、rank1 `028a74dbab9ba2afaa018994b1518c465d20d58ce05218720634e5c42a4a2a45`。每组独立输入核验均绑定这些冻结SHA与新预测SHA。
