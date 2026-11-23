# 已做实验与上传缺口（只读核查）

核查范围：当前 ECNU 项目的 `experiments/` 全部14个一级运行目录、指定 `codex/` 研究目录及 Justin-hub-coder 负责的三个 GitHub 分支完整递归树。核查时间及原始状态见 `inventory_server_records.json`；树未截断，固定 HEAD 见 `inventory_github_matches.json`。本文件为发布前清单，不能当成已上传回执。

| 实验族 | 实际完成情况 | 本轮开始时上传状态 | 本轮处理建议 |
|---|---|---|---|
| Qwen→RQ-KMeans→自有 TIGER：Beauty/Sports/Toys baseline | 已完成；旧来源缺口沿前轮说明 | 已在 `yjc-grid@6200d5875e264b5e6ec9841cf2cdb3b4a12d71a4` | 复用固定入口；不重训、不重复大载荷 |
| Qwen SID：Beauty/Toys 五种子 shuffle、MLP、注意力 | 20次训练、三测试条件及对应分析已完成 | 已在 `SID-Order-Invariance@362229201f0cb1eab4f154e5be30c74ef137dfee` | 复用固定入口 |
| Llama 分类：Beauty/Toys SID语义一致性 | 已完成；taxonomy v2–v7 属同实验尝试历史 | 已在 `sid-semantic-hierarchy@371b4498281b39ea87d0d065ff200e24ff4912b0` | 复用已上传完整标签、统计、请求记录 |
| LLM2Vec：Beauty/Toys/Instruments 标题、标题＋品牌六组层级分析 | 已完成 | 同上，六组向量已发布 Release | 复用固定入口 |
| LETTER公开 Beauty SID＋自有 TIGER | 正式完整训练＋unshuffle/shuffle_3/shuffle_4；best5200，结束6200 | **未上传** | 补上传源码快照、配置、SID身份与碰撞说明、三个原始结果、checkpoint、失败预跑 |
| LETTER公开 Instruments SID＋自有 TIGER | 正式完整训练＋三测试；best5400，结束6400 | **未上传** | 同上；24772测试用户，122目标具有官方SID歧义 |
| LETTER公开 Yelp SID＋自有 TIGER | 正式完整训练＋三测试；best6500，结束7500 | **未上传** | 同上；30431测试用户，1306目标具有官方SID歧义 |
| 自有 Qwen→RQ-KMeans：Yelp aligned_v1 上游 | 18775商品向量、量化器、SID和数据处理完成 | **未上传** | 补上传处理链、有效配置、向量/SID/量化器和来源 |
| 自有 Qwen Yelp aligned_v1：原 patience10 seed43 | Y03完整三测试；best200，结束1200，28330测试用户 | **未上传** | 完成结果正常归档；明确这是原五种子计划中的单种子结果 |
| 自有 Qwen Yelp aligned_v1：patience40 seed43诊断 | 完整三测试；best17100，结束21100；另有100用户注意力/收敛诊断 | **未上传** | 单独标注诊断，不替代原 patience10 条件 |
| 自有 Qwen Yelp 五种子计划剩余任务 | Y01/Y02/Y04/Y07用户停止；Y05/Y06/Y08/Y09/Y10用户取消 | **未上传运行历史** | 随同 Yelp 归档原状态、停止回执和已有配置，不称五种子完成 |
| LETTER-LC-Rec Instruments LLaMA推荐 | 当前训练6065/8240（状态快照时）；测试仅4用户预跑完成 | **未上传，仍运行** | 只归档冻结源码/配置与真实状态；不称正式测试已完成、不干预 |
| LETTER-LC-Rec Beauty 原三任务 | 用户于2950/19448停止 | **未上传停止历史** | 保存原尝试/停止回执；不能称训练完成 |
| LETTER-LC-Rec Beauty seqrec batch64 | 当前训练1300/8212（状态快照时）；2步容量预跑通过 | **未上传，仍运行** | 同 Instruments 运行中处理 |

## 实际原始入口

- LETTER Beauty：`experiments/letter_official_beauty_20261115_125008/formal/`；失败预跑 `letter_official_beauty_20261115_124859/`。
- LETTER Instruments/Yelp：`experiments/letter_official_instruments_yelp_20261116_142754/{instruments,yelp}/formal/`。
- Qwen Yelp上游：`experiments/yelp_rkmeans_qwen_aligned_v1/`，`codex/yelp_aligned_v1/`。
- Yelp原计划：`experiments/sid_order_yelp_5seed_v1/`，`codex/yelp_aligned_v1/training_queue_v1/user_stop_manifest.json`。
- Yelp长耐心诊断：`experiments/yelp_diagnostic_patience40_seed43/`，`codex/yelp_diagnostics/`。
- 活动 LLaMA：`experiments/letter_lc_rec_formal_20261121_123952/Instruments/`；新Beauty `experiments/letter_beauty_seqrec_batch64_20261122_155523/Beauty/`。

服务器共同根为 `/home/jcy/GRID_wmh_Qwen_TIGER_20261009/`。

## 统计口径与边界

1. GitHub 三完整树中均没有任何包含 `letter` 或 `yelp` 的路径；已有实验族由前轮公开发布回执及本次树内原结果入口对应确认。没有把其他负责人 `wbw-lc-rec` 的结果当作本人上传。
2. 只读扫描按已知项目边界进行，未扫描其他账户或无关文件；每个运行读取深度至5、单JSON至1.5MB的状态/结果材料。该有界策略的原程序为 `inventory_readonly.py`，原始证据保留。
3. Yelp aligned_v1 是28330用户、18775商品，LETTER官方Yelp是30431用户、20033商品。这两套上游/处理协议不同，不能当同一实验直接合并。
4. 原 `progress.json` 的 `failed` 不能覆盖明确 `user_stop_manifest.json`；本地历史 `current_state_verified.json` 记录原队列用户停止/取消。发布时应取服务器停止原件。
5. 100用户 `full_test_reference_*.pt`、100用户注意力和4用户LLaMA预跑均不是全测试逐用户原表；不据此宣称全量候选有效率或完成正式测试。
6. 活动作业的进度是读取时快照，没有冻结训练；后续发布应写明观察时点，不能仅凭旧 `running` 状态声明进程实时存活。
