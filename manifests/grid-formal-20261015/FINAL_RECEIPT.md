# GRID 正式实验统一结果与最终交付回执

## A. 总体状态

**READY_FOR_REVIEW（最终交付成立时状态，条件见I）**：六组Beauty正式全量评测、三域MLP正式成果、八组新增fixed32注意力及3072条CPU统计、非作者统计与整合审核、C1内容提交实际GitHub回读和交付候选包独立验包均已完成。C2元数据提交的实际远端回读及最终封装的独立验包不在本文预签；其真实SHA/回读/验包记录在包外回执与最终回复中绑定。仅在这两项实际PASS后最终交付状态才生效；最终网页GPT二次审核仍待进行。

本轮没有训练、全量测试、注意力重采或GPU模型调用。GPT已审阶段快照仅覆盖旧成果；新统计和最终发布仍等待网页二次审核。原进行中报告已在支持目录 finalization-input-history-20261017 保留原文。

## B. Beauty Shuffle

每组22363用户；A、B分别比较各自Normal。原should_check_prefix=False及候选评价不变，不过滤/补齐/重排无效SID。

| 条件 | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
|---|---:|---:|---:|---:|
| A-Normal | 0.046237089 | 0.031252529 | 0.066449046 | 0.037725352 |
| A-Shuffle3 | 0.046371236 | 0.031076979 | 0.066359609 | 0.037486158 |
| A-Shuffle4 | 0.045968786 | 0.030933077 | 0.065733574 | 0.037285861 |
| B-Normal | 0.043151636 | 0.029169954 | 0.065867729 | 0.036434427 |
| B-Shuffle3 | 0.043285783 | 0.029069517 | 0.065018110 | 0.036036659 |
| B-Shuffle4 | 0.042793900 | 0.028868502 | 0.064481512 | 0.035842434 |

同路线Normal的绝对差及相对百分比（比例值差，不是百分点）：

| 条件 | ΔR@5（相对） | ΔN@5（相对） | ΔR@10（相对） | ΔN@10（相对） |
|---|---:|---:|---:|---:|
| A-Shuffle3 | +0.000134148 (+0.2901%) | -0.000175551 (-0.5617%) | -0.000089437 (-0.1346%) | -0.000239193 (-0.6340%) |
| A-Shuffle4 | -0.000268303 (-0.5803%) | -0.000319453 (-1.0222%) | -0.000715472 (-1.0767%) | -0.000439491 (-1.1650%) |
| B-Shuffle3 | +0.000134148 (+0.3109%) | -0.000100438 (-0.3443%) | -0.000849620 (-1.2899%) | -0.000397768 (-1.0917%) |
| B-Shuffle4 | -0.000357736 (-0.8290%) | -0.000301452 (-1.0334%) | -0.001386218 (-2.1045%) | -0.000591993 (-1.6248%) |

四项相对变化处于约−2.105%至+0.311%，但这不表示用户推荐列表不变：

| 条件 | Top1改变用户 | 有序Top10改变用户 |
|---|---:|---:|
| A-Shuffle3 | 3307 | 21137 |
| A-Shuffle4 | 4031 | 21733 |
| B-Shuffle3 | 3325 | 21575 |
| B-Shuffle4 | 4115 | 21927 |

来源、预测hash、完整候选合法性及逐用户文件见 [既有配对分析](results-review/beauty-paired-shuffle-analysis.json)、[六条件验收](results-review/beauty-six-formal-results-independent-summary.json) 和 [本次事实摘录](final-integration-source-facts-20261017.json)。

## C. 三域MLP

三域均scratch、全参数训练；模型为共享逐物品MLP聚合，decoder仍四码位自回归。原验证Recall@5最大值选best、patience10早停；不按test选模。

| 域 | raw/accum/val间隔 | 可训练参数 | final/best step | 完整验证次数 | test条数 | worker小时 |
|---|---|---:|---|---:|---:|---:|
| beauty | 32/64/6400 | 13292160 | 3600/2600 | 36 | 22363 | 5.831641 |
| toys | 32/48/4800 | 13291136 | 5400/4400 | 54 | 19412 | 6.452553 |
| sports | 16/16/1600 | 13292160 | 5900/4900 | 59 | 35598 | 6.915800 |

| 域 | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
|---|---:|---:|---:|---:|
| beauty | 0.043330502 | 0.030580180 | 0.061709072 | 0.036571044 |
| toys | 0.034823820 | 0.025292708 | 0.049814548 | 0.030142274 |
| sports | 0.018090904 | 0.011313325 | 0.026405977 | 0.014029855 |

Beauty为已批准T5/RQ-VAE+STE相关条件。Toys沿用作者固定RQ-VAE+STE bundle，上游T5 factory与Qwen路径冲突及唯一历史producer未绑定的限制仍保留，不能写成已证明某个唯一上游编码器。Sports为用户批准的本域Qwen/R-Kmeans＋学长逐物品MLP结构实验，不是已恢复学长原始Sports完整历史实验。各域SID、模型及训练条件不同，不作严格横向结构对照。单卡累积协议经批准，但不等同多卡DDP完整随机轨迹。

best checkpoint真实路径、大小、SHA、配置/SID及原训练source均见 [事实摘录](final-integration-source-facts-20261017.json)。三域训练/测试均低于各自12h；Toys/Sports加正式attention累计约6.466h/6.929h。已有源码/选模/严格加载与完整预测的真实非作者审核继续继承；未重载大checkpoint。

## D. 正式注意力与描述统计

八组新增采集均32用户、4批，每批8用户，32份原始NPZ和256张原PNG保留；B-Normal就是当前普通工作baseline，不新增重复baseline。Beauty MLP既有fixed32独立保留：旧formal RESULT.status=PASS，不满足未修改统计程序的COMPLETE入口契约，故不纳入本轮八组数值统计，不改写原件。

主路径为最终Top1完整祖先链；辅助路径为每步局部首beam。统计最后一层六heads平均的四个SID生成步骤，对actual encoder_mask==1位置计算，每路径32×6=192条，每job384条，共3072条。cosine/Pearson使用原可见权重，常数Pearson为null；TV/JS仅在运算中按可见行和概率化，JS用自然对数；不改原矩阵/热力图，不跨Shuffle原始列做未对齐相减。

| 条件 | Top1 cosine mean / median / min | 局部首beam cosine mean / min | Top1 TV mean | Top1 JS mean(nats) |
|---|---|---|---:|---:|
| a-normal | 0.887526 / 0.930186 / 0.507270 | 0.887874 / 0.507270 | 0.213815 | 0.068554 |
| a-shuffle3 | 0.884839 / 0.926655 / 0.509317 | 0.884990 / 0.509317 | 0.217076 | 0.070150 |
| a-shuffle4 | 0.875993 / 0.923632 / 0.509056 | 0.876830 / 0.509056 | 0.223655 | 0.072309 |
| b-normal | 0.982282 / 0.992096 / 0.853979 | 0.983317 / 0.873379 | 0.133652 | 0.034792 |
| b-shuffle3 | 0.982482 / 0.992736 / 0.856199 | 0.983528 / 0.876866 | 0.131906 | 0.034377 |
| b-shuffle4 | 0.981168 / 0.991705 / 0.851714 | 0.982074 / 0.851714 | 0.134803 | 0.035341 |
| toys-mlp | 0.992510 / 0.996080 / 0.944671 | 0.992798 / 0.944671 | 0.114701 | 0.022089 |
| sports-mlp | 0.850915 / 0.910593 / 0.251510 | 0.851438 / 0.226848 | 0.234727 | 0.060870 |

各组padding注意力质量均为0；raw行和误差最大值见原统计JSON（约1e−7量级）。所有min/median/mean/max、逐样本六对记录、Pearson与两路径least-three均完整保留于 [实际统计](attention-statistics/attention-descriptive-statistics.json)。least-three依既定每用户六对平均cosine升序、sample key平局排序选取，未挑图。

观察：B路线及Toys的cosine摘要较高，但B-Normal/Toys的TV均值仍约0.134/0.115，不能称四行分布完全相同。A三条件的主路径cosine均值约0.876–0.888，并存在负Pearson配对。Sports主路径单对cosine最低约0.252；用户 c6001178f9652bfb338d86de 的六对平均约0.629（两者是不同统计量），局部首beam单对最低约0.227。两种路径均不能抹去这些不相似情况。

固定32用户、原seed及各自条件下的行内描述支持有限的注意力相近/差异观察；与Shuffle汇总指标较稳且用户列表明显变化可并列讨论，尚不能证明模型忽略SID结构、数学上的置换不变性或四码位可无损并行生成。不同条件的摘要比较是描述性的，不是跨模型的因果结构效应。

## E. 作业、源码和恢复入口

18个核心原job均SUCCEEDED/exit0，本轮只读确认无存活worker/supervisor、task group为空；6份不可变部署manifest与原request身份吻合。完整job_id、commit、snapshot、资产及结果/审核关联见 [版本对应](FINAL_VERSION_MAP.json)，注意各类运行不是全用当前HEAD。

服务器原job：`/home/bwwu/grid-audits/sid-extensions-p0/jobs/<job_id>/`；新增attention WSL原件：`/home/addaswsw/recommend-support/sid-extensions-p0/jobs/new-server/<job_id>/retrieved/`。原始record/NPZ、256图全清单及有限公开代表/最弱样本图见 [产物身份](ATTENTION_ARTIFACT_INDEX.json)。大型权重/数据/全预测保留原位置，不重复训练或生成。

## F. 非作者审核及修正边界

新统计实际由 `/root/implementation_review` 使用原独立程序从原始NPZ逐条复算（1e−12是计算一致性容差，不是相似性阈值），并核对fixed32、source/export、6-head均值和原件hash；详见 [runtime](attention-results-review/attention-statistics-runtime-review.json) 与 [来源复核](attention-results-review/attention-statistics-source-provenance-review-20261017.json)。旧技术审核不是本轮新采集。最终整合非作者审核已通过，见 [整合审核](final-integration-review-20261017.json)；已发布内容候选包也通过真实非作者验包，见 [交付候选审核](final-delivery-independent-review-20261017.json)。最终元数据封装的包外独立回执随最终交付提供；不把旧审核冒充新执行。

原三轮3/3及一次限定resolve=False配置例外不重置；本轮新研究修正0轮、GPU预算0、新安装0，无新增研究方案。失败原件与既有CPU回归保持历史身份。

## G. 发布与交付状态

`wbw`是唯一发布目标。本文内容提交后的实际完整SHA、远端关键文件回读以及最终ZIP大小/成员/SHA在支持目录最终发布/交付回执中记录，避免自身SHA自引用。已完成内容提交 `574baa497f2eec59c77311621695c9d52fd6a2f5` 的正常发布与六个关键文件完整正文回读，见 [实际回读](publication-readback-content-20261017.json)。本次最终元数据HEAD和最终封装的实测身份在包外回执给出；不改变已审研究数字/源码、统计输出或原件。

## H. GPT审核包

GPT_UPLOAD_REQUIRED: YES。旧阶段ZIP b3627f…保留；新最终ZIP须包含本轮实际统计、原始矩阵、真实新审核和最终GitHub发布证据。路径、大小、完整SHA以本轮真实校验后回执为准。

## I. 当前剩余必做项

研究计算与内容审核已闭合。本次READY的最终交付成立条件为包外C2远端回读及最终ZIP独立验包均实际PASS；条件满足后本次授权范围无剩余必做项。研究结论的外推限制仍保留，等待网页GPT对新统计与最终交付二次审核；不自动开展Parallel、扩域或新增训练。
