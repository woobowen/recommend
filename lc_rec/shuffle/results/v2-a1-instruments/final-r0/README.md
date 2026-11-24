# 第四组 Instruments A1 Qwen Shuffle V2 正式结果

**FOURTH_GROUP_COMPLETE_WEB_REVIEW_PASS**

原父作业 a1-instruments-v2-formal-parent-r0 于 2026-11-23 23:26:12 EST（2026-11-24 04:26:12 UTC）结束。四个 GPU、四个 CPU 评价和最终 paired 均以原 r0 串行执行、退出 0，并经原 runner 清收。两方向非作者终审和网页 GPT 最终二审均已通过。 [九阶段终态](terminal-chain.json)

## 网页最终二审

2026-11-24，网页 GPT 核验了最终 ZIP 的 423 个逻辑文件、全量正式输入、2,477,200 个原始候选、全部逐用户指标、123,860,000 个共享抽样索引、80 条配对记录和 37 项 GitHub 固定文件。四份正式结果文件离线再生后逐字节一致。至此，Shuffle V2 四组正式实验的结果均已通过网页审核。 [二审前固定交付版本](https://github.com/woobowen/recommend/blob/335d860f2a0c072c46058e6625164bf2babd0e67/lc_rec/shuffle/results/v2-a1-instruments/final-r0/README.md)

离线复核工具需要 Python 3.11 或更高版本。包内 `verify-offline.sh` 最后的完整统计审核 JSON 相等断言对不同 NumPy／BLAS 环境过严。网页二审在 NumPy 2.3.5 下独立复算，draws 最大绝对差为 1.596e−16，CI 端点最大绝对差为 7.980e−17，均满足原 1e−13 容差。全部确定性字段精确一致，正式发布结果逐字节一致。此兼容性问题不影响终审结论。

## 固定方案与来源

LETTER Musical Instruments 四位 SID，原 LETTER-LC-Rec Prompt0、切分与冻结 Normal 双分片。Qwen/Qwen3-0.6B-Base revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd，原 checkpoint-8240 best adapter，未量化 BF16、BF16 LoRA、原生 SDPA。严格加载校验 394 个保存键、392 个 LoRA 键与 tied embedding/head。数据包含 24,772 人、9,922 商品、9,897 不同 SID、23 碰撞组和 918 个新增 SID token。

单 GPU、rank 0、batch 1、beam／return 20、max_new_tokens 10、确定性生成、early stopping、cache、EOS/PAD 151643、原逐槽 Allowed、稳定分数顺序、filter_items=False。Shuffle V2 使用 seed 42 和 allow_identity。S2、S3a、S3b、S4 的活动槽分别为 (2,3)、(0,1,2)、(1,2,3)、(0,1,2,3)。两个三槽模式共享相对排列。依全局 row_idx、可见出现顺序与活动槽数取样，在 seq[:-1][-20:] 中逐组置换完整 token ID。

- 计划 SHA-256 f37fbe7ac295480a181daf2fed098810bd5b2d3c9499e19a7eec67d160603c7b。 [固定计划](../formal-plan.json)
- GPU 固定源 c487f7e7176bc223445d3deecfab87e0bcc30d1a，CPU／控制器 3cc1110097c4764ebfd93bb38ee2913c8afb6958，两个服务器 export 共 52 个文件逐一匹配。 [GPU 部署](../c487f7e7176bc223445d3deecfab87e0bcc30d1a.deployment.json) · [CPU 部署](../3cc1110097c4764ebfd93bb38ee2913c8afb6958.deployment.json)
- 冻结 Normal GPU 0e3e2c6f5c1a76152ea0228390d94ffc1ff75ba5，CPU 445f1475cd1dd70d19881dbc1639bb311f7cccf7。 [预测、逐用户及统计原件位置和 SHA](artifact-identities.json)

## 五条件指标

所有 16 项均为 Normal > S2 > S3b > S3a > S4。下表为比例尺度。 [指标 CSV](metrics.csv) · [64 项相对 Normal 变化](changes_from_normal.csv) · [完整精度 JSON](results.json)

| 指标 | Normal | S2 | S3a | S3b | S4 |
|---|---:|---:|---:|---:|---:|
| hit@1 | 0.06135960 | 0.05364928 | 0.04000484 | 0.04040853 | 0.03132569 |
| hit@5 | 0.08719522 | 0.08606491 | 0.07189569 | 0.07342968 | 0.06357985 |
| hit@10 | 0.10693525 | 0.10556273 | 0.08989989 | 0.09510738 | 0.08315840 |
| hit@20 | 0.13091394 | 0.13083320 | 0.11299047 | 0.12158889 | 0.10689488 |
| ndcg@5 | 0.07437705 | 0.07040098 | 0.05668498 | 0.05745766 | 0.04809532 |
| ndcg@10 | 0.08068596 | 0.07664717 | 0.06249779 | 0.06441814 | 0.05441259 |
| ndcg@20 | 0.08679036 | 0.08309141 | 0.06837751 | 0.07114043 | 0.06044166 |
| mrr@5 | 0.07018004 | 0.06524436 | 0.05164904 | 0.05219132 | 0.04297729 |
| mrr@10 | 0.07274082 | 0.06778762 | 0.05404221 | 0.05503521 | 0.04557648 |
| mrr@20 | 0.07443720 | 0.06958395 | 0.05567711 | 0.05689395 | 0.04724558 |
| ItemHit@5 | 0.08671080 | 0.08543921 | 0.07137090 | 0.07304618 | 0.06321653 |
| ItemHit@10 | 0.10643738 | 0.10495721 | 0.08951639 | 0.09470370 | 0.08295656 |
| ItemHit@20 | 0.13067173 | 0.13055062 | 0.11286937 | 0.12130631 | 0.10671322 |
| ItemNDCG@5 | 0.07355197 | 0.06953336 | 0.05606049 | 0.05682348 | 0.04767843 |
| ItemNDCG@10 | 0.07986342 | 0.07579535 | 0.06192533 | 0.06376982 | 0.05404487 |
| ItemNDCG@20 | 0.08602704 | 0.08232002 | 0.06786632 | 0.07051665 | 0.06007506 |

## 配对统计

PCG64 seed 20261113，5,000 次共享用户有放回抽样，95% 线性百分位区间，未作多重比较调整。每个差值为 right − left。 [80 条配对记录](paired.csv)

| 比较 | 区间低于零 | 含零 | 高于零 |
|---|---:|---:|---:|
| S2 − Normal | 14 | 2 | 0 |
| S3a − Normal | 16 | 0 | 0 |
| S3b − Normal | 16 | 0 | 0 |
| S4 − Normal | 16 | 0 | 0 |
| S3b − S3a | 0 | 8 | 8 |

S2 的 Hit@20 与 ItemHit@20 区间含零。S3b 在全部 16 项的点估计上高于 S3a，其中 8 个区间含零。区间适用于这一冻结权重及 seed 42 实例的用户重采样。独立归因某个 SID 层级的语义作用需要额外证据。

## 全量边界与两方向终审

[非作者审核 A 原文](review-evidence/A_REVIEW.md) · [非作者审核 B 原文](review-evidence/B/B_REVIEW.md)

审核 A 独立重建正式输入和四种置换，并核对 prefill 回执和候选约束。覆盖 99,088 条模式–用户记录、691,716 个历史组出现和 1,981,760 个候选。每模式均 24,772 人、172,929 历史组、495,440 候选。

| 模式 | 恒等历史组 | 整输入恒等 | 改变 token | 与 Normal 的候选及分数完全相等 |
|---|---:|---:|---:|---:|
| S2 | 86,483 | 714 | 172,892 | 714 |
| S3a | 28,935 | 6 | 345,429 | 6 |
| S3b | 28,935 | 6 | 345,429 | 6 |
| S4 | 7,147 | 0 | 519,226 | 0 |

726 条自然恒等模式–用户记录的输入、候选 token 和 SID、EOS、原始解码与顺序、稳定排序及 FP32 分数完全一致，实测最大分差为 0。正式终态通过生成调用次数、参数版本连续性和 tied embedding/head 核验。逐 beam embedding 前向证据和全参数内容摘要来自既有 GPU 预跑。正式留证采用实际 prefill 摘要和守卫。

审核 B 从五条件 2,477,200 个候选重算了 1,981,760 个逐用户指标值、同量逐用户差值、80 个指标均值和 80 条配对记录，与正式值一致。全部 123,860,000 个共享索引核对一致。对 400,000 个均值按用户 multiplicity 独立归约，最大差 1.874e-16。160 个已保存区间端点从原 draws 重算精确一致，独立归约端点最大差 1.041e-16。 [双向审核汇总](independent-review-summary.json) · [复算源码](review-code/)

冻结 CCE 复算覆盖 23 个碰撞组和 122 个目标碰撞用户。五条件目录外合法码位候选依次为 239、159、70、58、41，重复候选为零。保留碰撞展开、uniform credit 与目录外占位。

## 资源与原件

- S2 GPU a1-instruments-v2-formal-gpu-shuffle2-r0，9607.291 秒，退出 0 并清收。
- S2 CPU a1-instruments-v2-formal-cpu-shuffle2-r0，17.103 秒，退出 0 并清收。
- S3a GPU a1-instruments-v2-formal-gpu-shuffle3a-r0，10140.820 秒，退出 0 并清收。
- S3a CPU a1-instruments-v2-formal-cpu-shuffle3a-r0，16.718 秒，退出 0 并清收。
- S3b GPU a1-instruments-v2-formal-gpu-shuffle3b-r0，10154.132 秒，退出 0 并清收。
- S3b CPU a1-instruments-v2-formal-cpu-shuffle3b-r0，16.886 秒，退出 0 并清收。
- S4 GPU a1-instruments-v2-formal-gpu-shuffle4-r0，10155.430 秒，退出 0 并清收。
- S4 CPU a1-instruments-v2-formal-cpu-shuffle4-r0，16.995 秒，退出 0 并清收。
- paired PAIRED a1-instruments-v2-formal-paired-paired-r0，55.977 秒，退出 0 并清收。

正式 GPU 合计 40,057.674 device-s（11.1271 h），剩余 24.8729 h。正式 CPU worker 123.679 s，准备 CPU 144.011 s。预跑 GPU 167.244 device-s。父监督墙钟 40,296.947 s。正式端首试通过。

最终审核 ZIP 收录六份全量预测、完整逐用户表、统计数组、实际源码和终审入口。共享索引采用无损 uint16，附原 NPY header。恢复后的 int64 NPY 原件为 990,880,128 字节，原 SHA-256 和全部数值均已核验。 [复原回执](bootstrap-restore-proof.json)

服务器原件位于 /data/bwwu/lc-rec/a1-instruments-v2/。作业根为 /home/bwwu/grid-audits/sid-extensions-p0/jobs/。
