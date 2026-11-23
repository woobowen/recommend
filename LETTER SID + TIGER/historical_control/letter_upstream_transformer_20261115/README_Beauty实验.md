# LETTER 官方 Beauty SID + 原 TIGER 下游顺序实验

用户授权：直接采用 LETTER 公开 SID 表，仅 Beauty；运行 unshuffle、shuffle_3、shuffle_4。

## 固定输入

官方仓库版本：8d0154e28de37dbb6e24871c508ad8ddb1921cda。

官方文件：third_party/LETTER-8d0154e/data/Beauty/Beauty.index.json。

将 `<a_n> <b_n> <c_n> <d_n>` 按层解析成整数，并仅转为现有下游所需的 `[4,12101]` tensor；不重新学习 SID、不修改四位码、不追加去重位。对应 tensor 在 codex/letter_upstream_transformer_20261115/inputs/beauty/semantic_ids.pt。

逐用户检查现有 training、evaluation、testing：与作者配套交互完全匹配，其中验证/测试沿用原有最多20件历史加1件目标的保存策略。12101件商品的标题一致，只有7件缺标题商品的占位符为原数据 `nan` 对作者 `title_<item_id+1>`；训练、验证、测试交互序列证明其映射仍相同。下游只读取 sequence_data 和 user_id，不输入文本或已有 embedding。

数据：22363用户、12101商品、198502条完整交互。沿用原数据文件，不重建划分。

## 模型与测试

直接复用 codex.plan_v1_1.protocol.ShuffleModel 和原 sid_order_shuffle_model；共享 encoder/decoder embedding，encoder和decoder各4层，d_model128，6 heads，d_ff1024，dropout0.15，内部mlp_layers2。Adam lr0.001、weight_decay0.0001，无scheduler；FP32；源batch16，因果展开上限128，梯度累积16。sequence_length120，验证/测试batch8。最多320000 optimizer steps/10 epochs，每1600训练batch验证，按val/recall@5选最佳checkpoint，early stopping patience10。train seed42，eval seed2026。

与原 Qwen 五seed实验中 Beauty shuffle seed42 的保存配置逐项比对关键参数；只有输入 SID、输出目录改变。原训练从头初始化，不复用旧Qwen下游checkpoint。

只训练一个有序模型，训练和验证不打乱；使用同一个最佳checkpoint依次评估：

- unshuffle：历史SID顺序保持不变。
- shuffle_3：对每个历史商品的前3位embedding做排列，第4位固定。
- shuffle_4：排列每个历史商品的全部4位embedding。

沿用现有可复现shuffle协议，以dataset/user/history-position/eval_seed确定排列；不改变目标SID，不打乱商品之间的时间顺序，不打乱decoder输出位次。全量22363测试用户，beam10，报告Recall@5/10、NDCG@5/10。此请求未要求导出注意力，故关闭导出。

## 官方 SID 冲突的指标含义

官方12101件商品只有12088个不同完整SID；11个冲突组涉及24件商品（9组各2件、2组各3件）。测试集中46个目标属于冲突组。保留原表并沿用现有“完整SID匹配”指标，冲突组内商品不可区分。因此这部分分数不能解释为严格区分商品身份后的指标。不能将结果宣称为LETTER原论文指标复现，也不能把第四位解释为我们旧方案的纯去重位。

## 启动与产物

GPU0独占运行：短流程检查（2个optimizer steps及每种模式2个测试batch）通过后自动正式训练和全量三组测试，GPU1/2/3不占用。短流程与正式结果目录分离。进程脱离SSH运行；失败停止，不隐式恢复旧checkpoint。

latest_beauty_run.json记录当前实验目录。实验目录包含job_status.json、preflight.log、formal.log；formal下保存progress.json、配置、源码快照、映射审计、最佳checkpoint及三组test_<mode>.json。最终stage_result.json和metrics.md自动生成，metrics.md只包含指标。
