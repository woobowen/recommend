# Instruments GRID：Qwen SID＋自有TIGER基线

维护：Jingchen Yin（Justin-hub-coder）。本目录发布2026年11月25日GPU0完成的 **Qwen2.5-7B-Instruct → RQ-KMeans SID → GRID/TIGER推荐**，仅正常SID顺序、训练seed42。发布获用户明确授权到main/Baseline Results；不是LETTER公开SID或LLaMA推荐实验。运行源码为服务器无Git导出的冻结快照，本次Git提交是结果发布版本。

## 结果与完成状态

| 数据集 | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|
| Instruments | 0.090374 | 0.106793 | 0.076403 | 0.081714 |

原始全精度数据：[run/result.json](run/result.json)、[正式阶段结果](run/formal/stage_result.json)、[训练CSV](run/formal/csv/version_0/metrics.csv)。[metrics.csv](metrics.csv)和[metrics.md](metrics.md)由[verify_saved_results.py](verify_saved_results.py)生成。评价共27,530名用户，每用户一个下一商品目标，完整四位SID匹配，无负采样；完整SID唯一。沿用冻结的beam10生成与SIDRetrievalEvaluator，不额外过滤/补齐候选、不新增约束解码。历史逐用户候选未保存，因此当前复核是聚合结果一致性检查，不能称候选级重算。

| 运行项 | 实际值 |
|---|---|
| run | instruments_grid_qwen_20261125_005542 |
| 启动 | 2026-11-25约00:55，北京时间 |
| 正式完成 | 2026-11-25 10:40:59 +0800，退出0 |
| 最优checkpoint | 6100步，按验证Recall@5选模 |
| 最优验证Recall@5 | 0.13174718618392944 |
| 停止步数 | 7100；连续10次验证未改善早停 |
| GPU | 物理0，一张RTX2080Ti |
| 数据 | 10,611商品、27,530用户、231,312条交互 |

本次没有shuffle_3/4，也没有五种子重复。不能与LETTER的9,922商品/24,772用户直接作同切分对照。单种子结果不支持统计显著性声明。

## 方案、实现与配置

沿用[yjc-grid固定基线](https://github.com/woobowen/recommend/tree/6200d5875e264b5e6ec9841cf2cdb3b4a12d71a4/GRID%20baseline)的Toys最终有效配置；[reference_manifest.json](reference_manifest.json)列直接核对文件。[run/reference/toys_effective.yaml](run/reference/toys_effective.yaml)是原参考，[run/formal/.hydra/config.yaml](run/formal/.hydra/config.yaml)是最终配置，[overrides](run/formal/.hydra/overrides.yaml)为空。适配替换源码根、数据目录、SID路径，正式训练从头初始化，ckpt_path为空；完整生效依据见[launch](run/launch.json)、[source_manifest](run/source_manifest.json)、[effective_changes](run/effective_changes.json)、[argv/cwd/GPU](run/formal/command.json)。

实际链路：

1. [prepare_data.py::main](run/runner/prepare_data.py)：ASIN→item_id核对、按时间和原行号排序、生成TFRecord并逐值回读。
2. [run_experiment.py::worker](run/runner/run_experiment.py)：锁与设备检查→冻结源→2步预跑→独立正式训练/测试。启动脚本的原绝对路径是证据，移植请用[reproduction.md](reproduction.md)，不要直接启动旧runner。
3. [grid_stage.py::main](run/source/_运行/grid_stage.py)→[pipeline_launcher](run/source/src/utils/launcher_utils.py)→[SemanticIDEncoderDecoder](run/source/src/models/modules/semantic_id/tiger_generation_model.py)；causal展开在[collate_functions.py](run/source/src/data/loading/components/collate_functions.py)。
4. 同一入口选择最佳checkpoint并调用trainer.test；完整SID匹配与按样本统计见[SIDRetrievalEvaluator](run/source/src/components/eval_metrics.py)。

encoder/decoder各4层、d_model128、6 heads、d_kv64、d_ff1024、dropout0.15、共享embedding；Adam lr0.001、weight_decay0.0001、无scheduler，FP32。源batch16、causal展开至多128、梯度累积16，训练序列长度120 token；每1600个训练batch验证（约100次更新），patience10，beam10，测试batch8、drop_last=false。训练流无限循环，max_steps320000/max_epochs10是上限，不应把CSV中的epoch0解释成只读了一轮有限样本。seed42，trainer.deterministic=false，保持参考配置，不承诺跨设备逐位复现。

## 输入身份、上游与资产

原Qwen Instruments上游在2026-11-15完成，本轮复用原SID并只训练推荐器。上游：[qwen_config](upstream_metadata/qwen_config.json)、[向量记录](upstream_metadata/upstream/item_embeddings.json)、[量化审计](upstream_metadata/upstream/upstream_audit.json)。Qwen输入文本包含原items_qwen字段；max_length128，WMH MeanAggregation，右padding，输出float32向量[10611,3584]，编码输出未做单位归一化；RQ-KMeans三层、宽256、30步、seed42、normalize_residuals=true，第四位作消歧。SID为[4,10611]，第i列对应item_id i，前三码0–255，第四码0–12；完整SID均唯一。

原ASIN→item_id和交互对应均保持，[data_audit](run/data_audit.json)给内容身份。训练S[:-2]；验证S[:-1][-21:]；测试S[-21:]。不新增k-core或交互去重，保留12,221条重复用户商品交互。只有5名用户的训练序列不足两件商品被排除训练；验证/测试仍各27,530人。TFRecord由准备程序全量逐值回读，序列轴与SID列对应。

原件在[资产manifest](assets.json)对应的单个[Release ZIP](https://github.com/woobowen/recommend/releases/download/instruments-grid-baseline-20261125/instruments-grid-original-assets.zip)：最佳推荐checkpoint、原三份TFRecord、SID、item_mapping、items_qwen、对齐交互、sequences/user_mapping，以及原Qwen向量和量化器checkpoint。约517MB压缩、展开约817MB；完整校验值及每成员大小/哈希在manifest中，获取程序先验hash/CRC再解包。上游Qwen大模型权重不重复打包，其来源与重建边界见reproduction。

## 运行证据、审核与限制

- 实验前：[preflight](run/preflight.json)、[smoke command](run/smoke/command.json)、[smoke exit](run/smoke/exit.json)、[gpu_start](run/gpu_start.json)、冻结源码和输入检查。预跑模型不作为正式初始化，预跑指标不纳入正式表。
- 实验中：[完整训练console](run/formal/console.log)、[CSV](run/formal/csv/version_0/metrics.csv)、启动/结束状态。当前progress.json是最终覆盖值；保留的启动观察在[历史准备说明](historical_preparation/README.md)，其中“未完成”仅描述当时状态。
- 实验后：[status](run/status.json)、[exit](run/formal/exit.json)、[result](run/result.json)、[CPU核验](verification/cpu_verification.json)、[导出清单](verification/export_receipt.json)。CSV的step从0计数：6099对应checkpoint6100，并非选模不一致。
- 首次失败[attempts/failed_005511](attempts/failed_005511/)因TensorFlow分发包名查询失败，保留错误/旧runner与后续正式运行分开；没有用失败结果替代正式记录。
- 历史准备说明记录过非作者静态检查，本轮没有取得其独立审读原文，故只作为历史陈述。本次发布由执行方核对源码/配置/输出和CPU载荷；当前侧对话不可调用子代理，未新增独立非作者审核，不冒称独立通过。网页复核可直接读取本目录及审查包。
- 本轮未重训、未生成新SID、未运行forward或模型重评；[verify_saved_results](verify_saved_results.py)仅复算显示表并核聚合记录。四种复现起点、实际验证范围与缺口见[reproduction.md](reproduction.md)。发布后远端回读记录随交付给出。
