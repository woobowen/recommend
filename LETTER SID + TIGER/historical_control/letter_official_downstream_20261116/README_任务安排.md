# LETTER 官方 Instruments / Yelp SID + 自有 TIGER

用户授权：使用 LETTER 论文公开 Instruments 和 Yelp SID，训练自有下游 Transformer，训练后自动执行 unshuffle、shuffle_3、shuffle_4 全量测试，计算 Recall@5/10 和 NDCG@5/10。

官方源码版本 8d0154e28de37dbb6e24871c508ad8ddb1921cda，直接读取对应 data/<Dataset>/<Dataset>.index.json、inter.json、item.json，并与首次下载官方归档的逐文件 SHA256 清单核对。

## 输入与映射

为避免与过去独立筛选后的数据误配，两个数据集均直接使用作者配套 inter.json 的商品整数 ID、用户 ID 和序列顺序。完整保留作者交互，不再去重、不重新5-core、不删冲突商品。仅转换 SID token 为 `[4,N]` 整数tensor；不重训练上游，不改变编码值。

Instruments：24772用户、9922商品、206153交互；9897个不同完整SID，23个冲突组涉及48件商品，额外重复记录25条。测试目标中122个属于冲突组。

Yelp：30431用户、20033商品、316354交互；19201个不同完整SID，415个冲突组涉及1247件商品，额外重复记录832条。作者序列含11830条额外重复用户—商品记录，按原序列保留。测试目标中1306个属于SID冲突组。

完整 SID 匹配沿用原下游评估实现；同SID商品不可区分。这是官方SID在自有下游上的实验，不等价于严格区分商品身份的排名指标，也不是原LETTER论文推荐指标复现。

## 划分与转换核验

- train：每用户 S[:-2]，由原因果展开collator构造训练样本，sequence_length120。
- validation：S[:-1][-21:]，最多20件历史加倒数第二件目标。
- test：S[-21:]，最多20件历史加最后一件目标。
- 数据只保存user_id/sequence_data；不输入商品文本、向量或用户属性。
- 所有生成的TFRecord逐行读回，对照作者序列及拆分策略，用户集合、商品范围、样本数量、顺序全部断言检查；保存源文件与转换文件哈希。

## 训练和评估参数

复用成功的 Beauty LETTER SID 实验 runner（run_beauty.py）的模型、训练、测试代码，生成可审查 diff 仅参数化数据集、输入路径与数量。自有 codex.plan_v1_1.protocol.ShuffleModel 和 sid_order_shuffle_model.py 原文件保持不变。

train seed42，eval seed2026；encoder/decoder共享embedding；各4层，d_model128、6heads、d_ff1024、dropout0.15、内部mlp_layers2；Adam lr0.001、weight_decay0.0001、无调度；FP32；源batch16、因果展开上限128、梯度累积16；验证/测试batch8、beam10。最多320000 optimizer steps/10 epochs，每1600训练batch验证，val/recall@5选择最佳checkpoint，early stopping patience10。实际数据集为无限训练流，常由early stopping停止。

每数据集从头训练一次，ckpt_path=null，训练和验证均不shuffle。训练结束加载同一个最佳checkpoint分别运行unshuffle、shuffle_3、shuffle_4。打乱只作用于encoder历史商品内部的embedding；商品时间顺序、目标、decoder输出顺序保持原协议。shuffle由eval_seed/dataset/user/history-position确定。正式测试对全体用户执行，并独立核对用户不重复及指标计数。

## GPU任务

- GPU0：Instruments短流程检查 → Instruments正式训练 → 三组全量测试。
- GPU1：Yelp短流程检查 → Yelp正式训练 → 三组全量测试。
- GPU2/3：不占用。

进程脱离SSH运行，持有stage GPU文件锁；若GPU被占用则排队，每15秒重新检查。短流程每数据集2个optimizer steps、三模式各2个test batch，成功后自动进入正式任务；失败仅停止本数据集，不自动重启、不影响另一条队列。

latest_queue.json保存当前实验组；组目录的 <dataset>_status.json 为调度状态，<dataset>_<stage>.log 为日志；每数据集正式目录包含源码快照、映射审计、配置、模型核验、progress.json、events.jsonl、最佳checkpoint、test_<mode>.json、stage_result.json、metrics.md。metrics.md只保留指标。

查询：服务器 grid 环境执行 codex/letter_official_downstream_20261116/inspect_queue.py。
