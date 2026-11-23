# LETTER公开SID＋自有Transformer/TIGER

维护：Jingchen Yin（Justin-hub-coder）。原实验2026-11-15/16完成，2026-11-23事后发布；本次提交不是历史运行commit。研究目标是在固定作者SID下，用我们原GRID/TIGER训练一个有序模型，测试历史商品内部SID embedding顺序扰动。不是LETTER官方TIGER后端，也不是Llama推荐。

## 入口与条件

[run_index.json](run_index.json)关联三个模型与[source_snapshots](source_snapshots/)；snapshot逐项对应各run/source_manifest，内容hash是无Git服务器的历史源码身份。实际调用：run_beauty.py或run_dataset.py→Hydra compose→pipeline_launcher→codex.plan_v1_1.protocol.ShuffleModel→sid_order_shuffle_model→T5Stack编码/解码→SIDRetrievalEvaluator。历史入口与准备/launch在historical_control，源src/configs和必要codex都在snapshot，不只两份入口。

官方来源HonghuiBao2000/LETTER@8d0154e28de37dbb6e24871c508ad8ddb1921cda，直接采用index四码不重学上游、不追加去重位。整数tensor为[4,N]，第i列对应作者item_id i，码范围0..255。Beauty12101/22363、Instruments9922/24772、Yelp20033/30431（商品/测试用户）；数字顺序不可与Qwen Yelp轴混用。原index/inter/item随Release提供，正式mapping_audit给全量分片身份、序列对应与SID冲突。

Beauty复用固定GRID分片并按独立作者交互逐用户核序列；Instruments/Yelp由作者S生成training=S[:-2]、validation=S[:-1][-21:]、test=S[-21:]的TFRecord，不进一步去重或core过滤。训练实际采用原causal展开collator与sequence_length120；验证/测试最多20历史+1目标，不把训练保存序列说成只有20历史。

模型encoder/decoder各4层、d_model128、6heads、d_ff1024、dropout.15、mlp_layers2，共享embedding。Adam lr.001/wd.0001，无scheduler，FP32；源batch16、causal展开上限128、累积16；每1600训练batch验证、val/recall@5选最佳、earlystop patience10。seed42，test_seed2026，beam10，eval batch8；最终有效配置以各runs/.hydra/config.yaml为准。三个best分别5200/5400/6500，终步6200/6400/7500。

只测试时gather encoder内每件历史商品的码embedding：S3前三码、第四码固定；S4四码；dataset/user/history-position/seed2026决定全部排列，包含恒等。物品时序、mask、目标及decoder输出码序不变；训练/验证无shuffle。没有本实验注意力导出。完整SID匹配Recall/NDCG，单目标Hit等于Recall；原代码保留生成约束/候选排序，没有本轮更换Trie或去重规则。

官方完整SID并非唯一：Beauty12088/Instruments9897/Yelp19201唯一SID，歧义测试目标46/122/1306；第四码不能称为本项目纯去重位。指标不等同严格item身份命中，也不宣称LETTER原论文精确复现。历史全测试逐候选未保存，当前9份test JSON复核不等同候选级指标重算。

## 结果与取得

[三域九组指标及相对变化](metrics.md)→runs/test_*.json→stage_result、checkpoint hash与run_index。全测试分母22363/24772/30431；同域三组用同一best。预跑成功/Beauty首次失败分别保存在attempts；其结果不纳入正式表。

```bash
python "LETTER SID + TIGER/verify_saved_results.py"
python "LETTER SID + TIGER/fetch_assets.py" --manifest "LETTER SID + TIGER/assets.json" --asset letter-tiger-originals.zip --output ./retrieved
python "LETTER SID + TIGER/assemble_letter_runtime.py" --dataset beauty --assets ./retrieved/letter-tiger-originals --output /your/new-runtime
```

资产包包含3模型、同序SID、原作者JSON及实际TFRecord，整包及成员SHA/CRC验证后解压；约1.80GB压缩，展开所需空间见assets成员bytes。环境见environment/grid_pip_freeze.txt及原requirements，复用既有CUDA框架，不根据名称假定其他环境等价。

## 四种起点

| 起点 | 材料与本轮实做 |
|---|---|
| 保存输出复算 | verify_saved_results核9份JSON及源码，复算相对变化；本轮实际执行 |
| 固定checkpoint重评 | 真实best权重、SID/数据、源码/有效配置均发布；本轮只核hash/配对，未模型重评。用新runtime下的路径适配配置创建独立输出，沿历史trainer.test，不调用fit；新评价需另行授权 |
| 重训 | assemble组装冻结src/configs/codex及资产，在派生副本映射ROOT与mapping审计路径并留diff；按对应run_beauty/run_dataset参数选择新输出。旧baseline只用于配置断言，reference_configs已附；本轮只验证组装不启动训练 |
| 重建上游 | 本实验起点是已公开作者SID，作者RQ-VAE重新学习不属于此次复现；转换原码到tensor及交互→TFRecord脚本原文齐备 |

新runtime示例启动为`python codex/letter_upstream_transformer_20261115/run_beauty.py --output /new/result --gpu 0`；其他域为`python codex/letter_official_downstream_20261116/run_dataset.py --dataset instruments --output /new/result --gpu 0`。这是需要另行授权的计算说明，上传过程不运行。旧source_snapshot/配置保持冻结，本次路径适配与历史版本分开。
