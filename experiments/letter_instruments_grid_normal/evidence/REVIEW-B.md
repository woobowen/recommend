# 非作者审核 B · 生成、评价与执行

本补充阶段对运行提交 `390df9f508b7dc11eaa57b79a5a91fb5a7f99385` 和 `grid-letter-instruments-normal-20261124-preflight-s1` 的生成、评价与执行证据给出 **PASS**。旧 PARTIAL 交付和三个旧 GPU 失败保留原状态。本轮范围为有界 GPU 预跑，正式研究结果待后续正式运行。

## 独立核验的范围

- 通过独立 SSH 回读服务器 8 份关键源码的 SHA-256，与固定运行提交的 Git blob 哈希逐项核对，并直接审阅固定 Git 源码。原生协议 `protocol.py`、四步生成 `sid_order_shuffle_model.py`、原生评价 `eval_metrics.py`、冻结 SID 与 CCE 两份公式均与上一 GPU 执行版 `b41e5d860e11d57c6c37c019c4e969933d82b7fb` 逐字节相同。生成后将同一候选和 FP32 概率交给原生指标及观察器。原生指标使用实际 `torch.topk` 的索引和顺序。
- 用实际修正后的 `evaluate.native` 检查用户指定的错选 top-5，并接受选择另一并列位置的合法 top-5。独立脚本用分数多重集与单调降序复核，生产代码逐项核对所选分数与最高 K 项。统一 SID 稳定排序及 CCE 碰撞 credit 保持原公式。另在合成边界上直接运行原独立夹具，覆盖 1,024 个原生标签掩码、1,000 组 SID／CCE 与 10,000 个 FP32 位模式。
- 用本审核独立编写的 `review_b_real.py` 在服务器只读执行并从取回的原件再次复算，两个回执逐字节相同。复算读取三份正式 LETTER 原始 JSON、20 个用户的全候选、逐用户指标、原生聚合、两遍观察原件与事件。固定轴为用户 0–18 和 24771，通过原始序列独立重建目标、历史、120 维输入与 mask。每用户的十个候选及四个原生 top-k 分别核对。有效候选保留原顺序，目录外候选和碰撞标签按冻结公式处理。
- 比较开启与关闭观察器的两份实际原件，日志记录的两个评估各为 `8 / 8 / 4`，四个原生指标分母均为 20。观察原件逐字节相同，SHA-256 为 `a35d763549b8bfce7a2b5ef85489af07c5576b7f6c3f76d40d8a6ee632e4d0f5`。全部输入、mask、候选、FP32 位模式、指标、Python／NumPy／Torch CPU／CUDA RNG 状态对上。每遍三次生成、12 次 decoder 调用，每个 batch 四步的 prefix width 为 0、1、2、3，首步 self/cross cache 均为空。自由生成的实际输入只含历史和 padding。两个测试终态的 Shuffle 调用均为零。
- 流式读取实际 best checkpoint 并核对字节 SHA-256 `08aac6a10e0ec8972338923a8d2615ee398bacc11cea117730cf9ee7f07de3ed`。两次真实 `load_state_dict` 严格加载回执都关联该文件，missing 与 unexpected 均为空，加载后的模型状态摘要相同。方向 A 另核 checkpoint 内容、optimizer 与 callback 状态。本审核未另启动 GPU 或做生成重放。

## 真实复算

独立复算覆盖 20 人和 200 个候选。统计为 6 个目录外候选，0 个重复候选，0 行分数并列，1 个碰撞目标。逐用户离线值最大绝差为 `5.551115123125783e-17`。原生 FP32 聚合与独立 FP64 公式最大绝差为 `8.692343997362961e-10`，小于既定容差 `24 × 2^-24 = 1.430511474609375e-6`。该界覆盖四舍五入、最多十项折扣与归约、分子分母比值及最终 FP32 汇总。

| 指标体系 | 非零预跑结果 |
|---|---:|
| 原生 | Recall@10 = 0.05，NDCG@10 = 1/60 |
| 统一 SID | Hit@10 = 0.05，NDCG@10 = 1/60，MRR@10 = 1/140 |
| 统一 CCE | ItemHit@10 = 0.05，ItemNDCG@10 = 1/60 |

其余批准指标均为零。这是固定 20 人工程预跑的结果。

## 作业与历史资产

独立 SSH 核验时间为 **2026-11-24 12:27:15.475697 ET**。任务 namespace 有 7 个作业和 7 个请求，全部终态、无活 worker 或 workgroup、均有清收确认。新补充 GPU 作业计费式 worker 墙钟为 236.479456 秒，继承三个旧 GPU 作业 37.265162 秒，累计 273.744618 device-seconds。CPU runner worker 累计 69.758122 秒，含新 CPU boundary 39.181771 秒；监督、手工离线复算和传输另属不同口径。两项计量均在批准累计上限内。GPU1 归还后 1 MiB、0%，无计算进程。正式父或子作业、请求及候选／训练产物为空，仍无正式 GPU 消耗。

原 PARTIAL 中比较的 `/data/bwwu/lc-rec/assets/data/Instruments/Instruments.index.json` 是 SHA-256 `b94d16398a178220a117abda5effd5508069beaccc2b8110eab25ae50f2af600` 的通用目录副本。历史 A0 正式终态双 rank 与 A1 正式测试分别记录自己的固定 release 中的 LETTER 原件，index SHA-256 同为 `33c87e28c52ae367985bdf494cb49d676d181e13a39497b4ad035f46111546dd`，也与本轮 GRID 的 index 字节一致。三者的 inter、item SHA-256 也逐项相同。历史比较证据已按固定 Git blob 提取并保留原文标识。[通用目录回执](https://github.com/woobowen/recommend/blob/af360a158280165f07249b80d2cfa0e871902498/lc_rec/evidence/cpu_input_asset_hashes.json#L14-L17)、[A0 终态 rank 0](https://github.com/woobowen/recommend/blob/af360a158280165f07249b80d2cfa0e871902498/lc_rec/letter/review/final_audit_20261113/terminal_effective_raw.json#L33-L58)、[A0 终态 rank 1](https://github.com/woobowen/recommend/blob/af360a158280165f07249b80d2cfa0e871902498/lc_rec/letter/review/final_audit_20261113/terminal_effective_raw.json#L9982-L10007)、[A1 原正式测试](https://github.com/woobowen/recommend/blob/af360a158280165f07249b80d2cfa0e871902498/lc_rec/a1_qwen3_06b/evidence/status_checks/20261114T034030Z/existing_results/test/metrics.json#L61-L78)

复核命令使用 `python3 -B review_b_real.py --experiment <实际代码根目录> --assets <assets> --run <已取回的成功 output/run>`。若包中以 runner 导出 manifest 去重两份 observer，另传 `--export-manifest <成功作业的 export-manifest.json>`，两条原始记录均须绑定到包内同一份原件的完整哈希和长度。
