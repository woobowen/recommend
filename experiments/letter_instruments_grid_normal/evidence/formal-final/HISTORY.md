# 历史 GRID Instruments Normal 对照

直接核对的历史记录是 `letter_official_instruments_yelp_20261116_142754/instruments/formal` 的 `unshuffle`，由事后发布提交 `74911e3ce668e265a8c3e7bc9a3929ae9247f11a` 的 `run_index.json` 关联到运行文件和源码快照 `b04385629968f59accfb83607a7b481913704e9cca254a8deeca7c92a1cc97e8`。历史 `test_unshuffle.json`、有效配置、阶段回执、模型验证、输入来源及映射审计与本轮运行源码 `390df9f508b7dc11eaa57b79a5a91fb5a7f99385` 中相应历史文件逐字节一致。历史 Normal 回执确认 24,772 人全部评价、`shuffle_calls=0`，并绑定 step 5,400 checkpoint 的 SHA-256 `ae3d9c69e9b0a0205a021a4afa02d041487491e8e88c94dc597a243d0a91d3e9`。本轮正式 checkpoint 由独立的新训练选出。[历史运行索引](https://github.com/woobowen/recommend/blob/74911e3ce668e265a8c3e7bc9a3929ae9247f11a/LETTER%20SID%20%2B%20TIGER/run_index.json) · [历史 Normal 原始结果](https://github.com/woobowen/recommend/blob/74911e3ce668e265a8c3e7bc9a3929ae9247f11a/LETTER%20SID%20%2B%20TIGER/runs/instruments/test_unshuffle.json)

| 原生测试口径 | 历史 Normal | 本轮 Normal | 本轮 − 历史 |
|---|---:|---:|---:|
| Recall@5 | 0.0849749743938446 | 0.08307766914367676 | −0.00189730525016784 |
| Recall@10 | 0.10277733206748962 | 0.10047634690999985 | −0.00230098515748977 |
| NDCG@5 | 0.0726066380739212 | 0.07116406410932541 | −0.00144257396459579 |
| NDCG@10 | 0.0783795714378357 | 0.07676350325345993 | −0.00161606818437577 |
| 测试 loss | 8.461666107177734 | 8.439092636108398 | −0.022573471069336 |

四项原生测试排名指标下降约 **0.144–0.230 个百分点**，测试 loss 则下降约 **0.02257**。选模验证 Recall@5 从历史 **0.10241401940584183** 变为本轮 **0.10257548838853836**（+0.00016146898269653）。本轮 best 和训练终步各提前 200 次更新。历史 64 次完整验证与 CSV 逐条对应，step 5,400 后连续十次低于 best，step 6,400 正常结束训练并进入 Normal。这些历史记录与 patience 10 早停吻合，固定发布范围没有 Trainer 的原始历史早停 stdout。本轮日志直接记录 patience 10 触发，62 次完整验证中 step 5,200 产生 best，step 6,200 结束训练。[历史生效配置](https://github.com/woobowen/recommend/blob/74911e3ce668e265a8c3e7bc9a3929ae9247f11a/LETTER%20SID%20%2B%20TIGER/runs/instruments/.hydra/config.yaml) · [历史完整事件](https://github.com/woobowen/recommend/blob/74911e3ce668e265a8c3e7bc9a3929ae9247f11a/LETTER%20SID%20%2B%20TIGER/runs/instruments/events.jsonl)

两次运行的三份 LETTER JSON、三个 TFRecord 分片及四位 SID tensor 的哈希相同，因此现有证据支持同一数据与划分。历史与本轮均从头训练、seed 42、eval seed 2026、13,128,064 参数、单卡 FP32、测试 batch 8、原生四步 Beam10。历史 evaluator 与本轮的对应代码差异只是在相同 `torch.topk` 后增加排序观察，`protocol.py` 增加原生候选保存入口。已核的原生推理工具和数据流实现逐字节相同。两次各自训练得到不同 best 权重，原始 TFRecord 读取保留 `shuffle(buffer_size=128)` 且未显式给 TensorFlow 行 shuffle 设置 seed。历史运行记录物理 GPU 0，本轮实际为 RTX 4090 物理 GPU 1。历史原件缺少已核实的设备型号和完整硬件条件。单 seed 的两次结果支持这里的同口径描述，现有证据不足以归因分数变化或评估多 seed 稳健性。[历史映射和划分审计](https://github.com/woobowen/recommend/blob/74911e3ce668e265a8c3e7bc9a3929ae9247f11a/LETTER%20SID%20%2B%20TIGER/runs/instruments/mapping_audit.json) · [候选和排序观察 diff](https://github.com/woobowen/recommend/blob/390df9f508b7dc11eaa57b79a5a91fb5a7f99385/experiments/letter_instruments_grid_normal/native_observation.diff)

可对比范围为历史实际保存的四项原生指标和测试 loss。历史发布说明明确记载完整逐候选未保存，已核历史范围也没有相应逐用户结果或统一 SID／CCE 十一项，因而这些项目只有本轮的正式数值。复核脚本 `history_compare.py` 以固定历史 CSV、事件、原生结果和本轮原件为输入，检查首次历史测试的 CSV 行，与两个完整验证序列及原生聚合交叉核对，并使用十进制数复算表中的绝对差。不使用其他下游 Beam20 结果填补历史缺项。[历史发布说明](https://github.com/woobowen/recommend/blob/74911e3ce668e265a8c3e7bc9a3929ae9247f11a/LETTER%20SID%20%2B%20TIGER/README.md)
