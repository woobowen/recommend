# Toys / Sports Shuffle 正式结果与注意力

`PARTIAL`：两域 B 六条件及对应 fixed32、配对分析和2304条描述统计完成；Toys A未决、Sports A配套缺失。Beauty仅继承，未重跑。

正式评测源码：`c8acac42d5e13ca80af75faac5d1f7bc33fb589f`。逐作业快照/实际 argv/输入 hash/原预测/矩阵/审核链见 EXECUTION_SUMMARY.json。

| 域 | 路线 | 条件 | 用户 | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 | 非法候选 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Beauty | A | Normal | 22363 | 0.04623709 | 0.03125253 | 0.06644905 | 0.03772535 | 106 |
| Beauty | A | Shuffle3 | 22363 | 0.04637124 | 0.03107698 | 0.06635961 | 0.03748616 | 100 |
| Beauty | A | Shuffle4 | 22363 | 0.04596879 | 0.03093308 | 0.06573357 | 0.03728586 | 119 |
| Beauty | B | Normal | 22363 | 0.04315164 | 0.02916995 | 0.06586773 | 0.03643443 | 75 |
| Beauty | B | Shuffle3 | 22363 | 0.04328578 | 0.02906952 | 0.06501811 | 0.03603666 | 72 |
| Beauty | B | Shuffle4 | 22363 | 0.04279390 | 0.02886850 | 0.06448151 | 0.03584243 | 87 |
| Toys | B | normal | 19412 | 0.04208737 | 0.02698686 | 0.06583557 | 0.03460294 | 61 |
| Toys | B | shuffle3 | 19412 | 0.04049042 | 0.02599545 | 0.06542345 | 0.03403118 | 63 |
| Toys | B | shuffle4 | 19412 | 0.03982073 | 0.02569822 | 0.06516588 | 0.03385324 | 66 |
| Sports | B | normal | 35598 | 0.02418675 | 0.01585043 | 0.03592898 | 0.01962937 | 3 |
| Sports | B | shuffle3 | 35598 | 0.02227653 | 0.01461020 | 0.03250182 | 0.01791045 | 5 |
| Sports | B | shuffle4 | 35598 | 0.02272600 | 0.01473860 | 0.03438395 | 0.01850235 | 3 |

三条件各域采用同 config/SID/checkpoint、seed42、原 batch8、32-true、120 SID输入位、TopK10及原生成器/evaluator；prefix约束保持False，非法候选不补齐、不过滤、不重排。A/B不是严格同模型对照。

| 域 | 条件 | ΔRecall@5 (相对%) | ΔNDCG@5 (相对%) | ΔRecall@10 (相对%) | ΔNDCG@10 (相对%) |
|---|---|---:|---:|---:|---:|
| Toys | shuffle3 | -0.00159695 (-3.794%) | -0.00099141 (-3.674%) | -0.00041211 (-0.626%) | -0.00057175 (-1.652%) |
| Toys | shuffle4 | -0.00226664 (-5.386%) | -0.00128864 (-4.775%) | -0.00066969 (-1.017%) | -0.00074969 (-2.167%) |
| Sports | shuffle3 | -0.00191022 (-7.898%) | -0.00124023 (-7.825%) | -0.00342716 (-9.539%) | -0.00171892 (-8.757%) |
| Sports | shuffle4 | -0.00146076 (-6.039%) | -0.00111182 (-7.014%) | -0.00154503 (-4.300%) | -0.00112703 (-5.742%) |


| 域 | 对比 | Top1改变 | 有序Top10改变 | Top10集合改变 |
|---|---|---:|---:|---:|
| Toys | Normal→Shuffle3 | 4270 | 19016 | 15884 |
| Toys | Normal→Shuffle4 | 4726 | 19185 | 16499 |
| Sports | Normal→Shuffle3 | 14889 | 35381 | 33431 |
| Sports | Normal→Shuffle4 | 12282 | 35390 | 32744 |

三域 B 的 Shuffle 都改变大量逐用户推荐。Beauty B 的 Recall@10 相对变化为 Shuffle3约−1.29%、Shuffle4约−2.10%；Toys约−0.63%/−1.02%，Sports约−9.54%/−4.30%。下降幅度有域差异，Sports中Shuffle4下降小于Shuffle3，不能推出扰动数量与损失单调对应。Toys/Sports本轮四指标各条件均下降；Beauty Shuffle3的Recall@5略升，其余变化按原表保留。A只可继承Beauty现有结果，不能声称A的三域现象已覆盖。

四指标绝对/相对变化、命中迁移、逐用户分布/原始配对记录和预声明异常样本见 results-audit/toys-sports-paired-shuffle-analysis.json。

每条件32用户，共4×8，三条件共同cohort且预测前确定。原始最后层6heads、4个autoregressive query、真实最终Top1完整父链与局部首父链均保留。全部24 NPZ/192 PNG在服务器正式job及已验证本地export；轻量包图像按事先固定规则展示。
Decoder query身份另以CPU从实际保存的beam前缀/父链和固定源码映射派生：首步为learned BOS（无SID整数ID），后续为对应parent前缀末码及offset lookup索引。每组256身份记录，六组1536条；这是可审查的源码推导，不宣称直接捕获未保存的T5Stack input_ids，也不是2304条注意力行对统计。

| 正式注意力条件 | 路径 | Cosine median | Pearson median | TV median | JS median (nats) |
|---|---|---:|---:|---:|---:|
| toys-b-normal | final_top1_cross | 0.930084 | 0.857540 | 0.215096 | 0.049034 |
| toys-b-normal | local_first_cross | 0.927405 | 0.847824 | 0.218129 | 0.056751 |
| toys-b-shuffle3 | final_top1_cross | 0.929388 | 0.884870 | 0.223247 | 0.052919 |
| toys-b-shuffle3 | local_first_cross | 0.936219 | 0.880810 | 0.216459 | 0.050384 |
| toys-b-shuffle4 | final_top1_cross | 0.925049 | 0.878831 | 0.222895 | 0.049984 |
| toys-b-shuffle4 | local_first_cross | 0.929670 | 0.858752 | 0.219159 | 0.052320 |
| sports-b-normal | final_top1_cross | 0.876541 | 0.355859 | 0.190005 | 0.052117 |
| sports-b-normal | local_first_cross | 0.914058 | 0.688053 | 0.166086 | 0.038670 |
| sports-b-shuffle3 | final_top1_cross | 0.771875 | 0.218469 | 0.263009 | 0.069458 |
| sports-b-shuffle3 | local_first_cross | 0.805828 | 0.480971 | 0.240859 | 0.061751 |
| sports-b-shuffle4 | final_top1_cross | 0.820554 | 0.400018 | 0.234092 | 0.057661 |
| sports-b-shuffle4 | local_first_cross | 0.905347 | 0.645392 | 0.196359 | 0.042852 |

每路径192条，共每条件384条；各指标min/median/mean/max、全部逐pair记录、最不相似三个样本、padding质量与行和误差见 attention-statistics/attention-descriptive-statistics.json。无相似度验收阈值；不能据此证明decoder不利用目标前缀或SID严格置换不变。

资产与A缺口见 asset-source-review.md、asset-audit/runtime-asset-admission.json。Toys A需要学长指定唯一完整run及真实T5/R-Kmeans来源；Sports A需要完整匹配链。本轮不训练、不混拼、不重建SID。

实际GPU请求worker wall总计 0.511934 小时，含CPU准备；预算5.5小时。修正3/3（新入口bootstrap、未用Hydra元数据回读、最终审核索引读取实际六条件query回执），旧Beauty额度不变。第三轮仅修复元数据路径，不改预测、注意力或统计。历史准入门保留当时2/3记录。可选callback CPU v1两失败原件保留；CSV/stage_result选模和B严格语义门已通过。

所有本任务作业终态、进程组清理已核，无活动job。服务器恢复入口 `/home/bwwu/grid-audits/sid-extensions-p0/jobs/gs-20261017-*`；只按原job查询，不自动重启。

`GPT_UPLOAD_REQUIRED: YES`。固定GitHub发布、实际回读、最终ZIP大小/成员/SHA及真实非作者容器审核见最终交付回执。

可执行条件完成；缺失路线未执行。执行停止，等待GPT网页版二次审核。
