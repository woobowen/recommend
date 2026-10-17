# 跨域固定证据索引与原指标来源

本文件是最终交付用的固定来源说明，不是新实验验收、进度看板或新的研究结论。`cross-domain-evidence-index.json`可供主控合并；`cross-domain-original-metrics.json/csv`保留原始比例指标。8条具体路线、24个条件中21个有真实已审结果；Sports A2三个条件在本索引中显式PENDING/null，不预签最终PASS。

## 实际读取与继承范围

- 旧证据统一从仓库 `woobowen/recommend` 的固定完整提交 `9ab3d0ea5d32db00e278f74c599a1943885453db` 使用只读 `git show <SHA>:<path>`取得，53份、1,289,538字节。每份含完整提交、repo路径、Git blob OID、字节数、SHA-256、固定GitHub链接和包内副本路径，见`fixed-source-manifest.json`。没有执行旧审核脚本、重新计算旧预测或加载旧模型。
- Beauty A/B六条件、Toys B与Sports B六条件，继承原真实非作者验收及其明确范围；没有把“继承”写成当前重新验收。原正式RESULT.metrics与原已签审核中的字典精确一致，本索引只读取这些原值。
- Toys A双候选：本轮`/root/toys_result_audit`直接审阅原预测/NPZ/源码后的已完成报告，hash绑定在`current_review_sources/toys/`。
- Sports A1：本轮`/root/sports_a1_result_audit`真实非作者审核，包含CPU严格加载和原TFRecord固定32条核验，hash绑定在`current_review_sources/sports_a1/`。本索引仅引用其已签结论，不声称亲自重新执行Sports A1审核。
- Sports A2：只记录用户已批准的研究身份`Sports-A2-T5RKMeans-DualTable-NewTraining`及待补结果；checkpoint/SID/指标/审核者均不猜测。主控须在真实终态与非作者验收后，从实际结果加入新证据。

## 原指标表

数值是各自原evaluator已接受输出的原始比例值，不是重新评价结果；每条Shuffle只与同一具体模型的Normal比较。PENDING不等于零。

| 路线 | 条件 | testing用户 | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 | 来源类别 |
|---|---|---:|---:|---:|---:|---:|---|
| Toys-A-5300 | Normal | 19412 | 0.040078300982714 | 0.027255861088634 | 0.061302285641432 | 0.034059926867485 | 本轮真实非作者审核 |
| Toys-A-5300 | Shuffle3 | 19412 | 0.040232844650745 | 0.026425955817103 | 0.060271997004747 | 0.032848287373781 | 本轮真实非作者审核 |
| Toys-A-5300 | Shuffle4 | 19412 | 0.040129818022251 | 0.026403736323118 | 0.059859879314899 | 0.032738946378231 | 本轮真实非作者审核 |
| Toys-A-5000 | Normal | 19412 | 0.040644962340593 | 0.026936978101730 | 0.059859879314899 | 0.033158622682095 | 本轮真实非作者审核 |
| Toys-A-5000 | Shuffle3 | 19412 | 0.039820730686188 | 0.026538845151663 | 0.059396248310804 | 0.032830830663443 | 本轮真实非作者审核 |
| Toys-A-5000 | Shuffle4 | 19412 | 0.038841951638460 | 0.026054665446281 | 0.058056872338057 | 0.032222375273705 | 本轮真实非作者审核 |
| Sports-A1 | Normal | 35598 | 0.025816056877375 | 0.017168486490846 | 0.037586379796267 | 0.020962946116924 | 本轮真实非作者审核 |
| Sports-A1 | Shuffle3 | 35598 | 0.019748300313950 | 0.013000275008380 | 0.029496038332582 | 0.016144171357155 | 本轮真实非作者审核 |
| Sports-A1 | Shuffle4 | 35598 | 0.019860666245222 | 0.012999451719224 | 0.029299398884177 | 0.016071824356914 | 本轮真实非作者审核 |
| Sports-A2 | Normal | PENDING | — | — | — | — | 待真实A2审核 |
| Sports-A2 | Shuffle3 | PENDING | — | — | — | — | 待真实A2审核 |
| Sports-A2 | Shuffle4 | PENDING | — | — | — | — | 待真实A2审核 |
| Beauty-A | Normal | 22363 | 0.046237088739872 | 0.031252529472113 | 0.066449046134949 | 0.037725351750851 | 固定旧验收继承 |
| Beauty-A | Shuffle3 | 22363 | 0.046371236443520 | 0.031076978892088 | 0.066359609365463 | 0.037486158311367 | 固定旧验收继承 |
| Beauty-A | Shuffle4 | 22363 | 0.045968785881996 | 0.030933076515794 | 0.065733574330807 | 0.037285860627890 | 固定旧验收继承 |
| Beauty-B | Normal | 22363 | 0.043151635676622 | 0.029169954359531 | 0.065867729485035 | 0.036434426903725 | 固定旧验收继承 |
| Beauty-B | Shuffle3 | 22363 | 0.043285783380270 | 0.029069516807795 | 0.065018109977245 | 0.036036659032106 | 固定旧验收继承 |
| Beauty-B | Shuffle4 | 22363 | 0.042793899774551 | 0.028868502005935 | 0.064481511712074 | 0.035842433571815 | 固定旧验收继承 |
| Toys-B | Normal | 19412 | 0.042087368667126 | 0.026986861601472 | 0.065835565328598 | 0.034602936357260 | 固定旧验收继承 |
| Toys-B | Shuffle3 | 19412 | 0.040490418672562 | 0.025995448231697 | 0.065423451364040 | 0.034031182527542 | 固定旧验收继承 |
| Toys-B | Shuffle4 | 19412 | 0.039820730686188 | 0.025698222219944 | 0.065165877342224 | 0.033853244036436 | 固定旧验收继承 |
| Sports-B | Normal | 35598 | 0.024186752736568 | 0.015850426629186 | 0.035928983241320 | 0.019629370421171 | 固定旧验收继承 |
| Sports-B | Shuffle3 | 35598 | 0.022276531904936 | 0.014610195532441 | 0.032501824200153 | 0.017910448834300 | 固定旧验收继承 |
| Sports-B | Shuffle4 | 35598 | 0.022725995630026 | 0.014738601632416 | 0.034383952617645 | 0.018502345308661 | 固定旧验收继承 |

## 源码身份不得混同

固定证据提交9ab3是可读取的发布版本，不代表全部实验都在该源码commit上执行。索引逐条件另存实际eval与attention的source commit及snapshot：

| 固定继承路线/阶段 | 实际科研commit | snapshot SHA-256 |
|---|---|---|
| Beauty A/B evaluation | `2cf4c793c2e0b2e9dc7e7a9269f5e54d5c667896` | `3870cb40eb8d040a468ff3509e91703da7e833f90f8c279772b35d296b60d04a` |
| Beauty A/B attention | `beb5079177fad847196b975537a82c3c4e160403` | `a76001dce00e5053212501cfdf925ed1bad3924622fbfbfe5b20aa1b7eda684c` |
| Toys/Sports B evaluation | `c8acac42d5e13ca80af75faac5d1f7bc33fb589f` | `f892ae7892f8c981ffe3494368fbaa6a88f693605c2d0c902581cfeea0d64c50` |
| Toys/Sports B attention | `d575532084ed5fe14bb78e3f51b77a4cd38c253a` | `2f36013cffe5aebf696e2fb4773a65543dc6a9e67f5773eb2df6360e8d27a214` |
| Toys/Sports B attention | `c7c63b9869975b29abfc189aa5e716d0a798e8d0` | `cdad8a0a69b23ebb424fb157283df1798175d7277352f7c469575697176186fd` |

本轮Toys A/Sports A1原计算身份是`b44399926c98369191c2ff5c2425377c8c07d3c8`及b620部署；共享调度的`6dafbc7215b58a2dbb47905a687c930caecf7b44`不能代替科研源码。本轮A2修复后的工作树及新任务另有固定身份，不能改写旧Toys/A1执行字节。

## 原验收结论与旧范围限制

- Beauty六评测：`BEAUTY_SIX_FORMAL_EVALUATIONS_INDEPENDENTLY_ACCEPTED`；旧最终内容整合：`INDEPENDENT_FINAL_RESULTS_INTEGRATION_PASS`。旧交付candidate包的PASS只按其原范围继承，不将候选包验收当作本轮最终ZIP验收。
- Toys/Sports B：`QUALIFIED_B_RESULTS_ATTENTION_AUDIT_COMPLETE`，六全量条件、六fixed32、两条beam路径、原配对及2304条query-pair统计均有旧真实非作者原件；本索引不重新复算。
- 旧`grid-shuffle-20261017/FINAL_REPORT.md`的PARTIAL及“A缺失”记录完整保留，反映当时任务范围；本轮Toys双候选和Sports A1新审核在此新增引用，不回写旧报告。A2结果仍由本轮真实后续产物决定。
- Toys A上游历史producer仍不唯一：不能从同名Qwen/3584配置与512维原producer材料冲突中捏造唯一历史编码器。两候选均保留，不按test选择删除。
- Sports A1/A2是经过本轮授权的SID条件下从零训练方法复现，不是找回学长原始Sports推荐权重。A1/A2、A/B及不同域之间同时存在SID/模型/训练或checkpoint差异，不能把数值差异归因为单一因素。
- 每路线三条件内部共用本模型Normal对照、SID、checkpoint、testing目标与原evaluator；非法候选原样保留。完整资产SHA和预测SHA逐条件可查。
- fixed32来自输出检查前按实际文件/行顺序固定的evaluation/validation cohort；全量指标来自testing。相同user key不意味着两split历史/target相同。
- 原统计是各条件内部可见位置上四query的描述：Cosine/Pearson使用raw权重，TV/JS仅为计算概率分布归一化；不直接对未对齐的Shuffle列做跨条件相减。没有跨seed显著性或严格置换不变性证明，也不能推出decoder不利用目标prefix。

## 合并和查证入口

- `cross-domain-evidence-index.json`：逐路线、逐条件原指标、checkpoint/SID/config、prediction hash、真实执行commit/snapshot、原审核状态/来源、attention cohort/审核来源、已审配对分析及全部边界。
- `cross-domain-original-metrics.json/csv`：24个规范化条件行；A2三行指标为null。
- `fixed_sources/`：53份从固定Git版本读出的旧原件；每个JSON来源ref同时给固定GitHub URL。
- `current_review_sources/`：本轮Toys/A1已签报告副本与原文件SHA绑定。
- 正式终态后主控可按`route_id=Sports-A2`补入真实三条件结果与审核；不得将本索引中的PENDING直接改写为无证据PASS，也不重置旧预算或实验修正次数。

本整理新增GPU调用0、旧模型加载0、旧结果复算0、Git写入0、新依赖0。原始大模型、旧预测/attention全部保留原处，没有重复收集完整档案。
