# Qwen Yelp aligned_v1：上游、部分队列及耐心诊断

维护Jingchen Yin。历史实验2026-11-08，2026-11-23补发布。与LETTER官方Yelp不同：本轴为28330用户/18775商品；从LETTER Yelp inter/item去除用户重复商品后重新迭代5-core、编号并转换，源30431/20033，不能据同名当同一切分。处理配置/独立映射/原始指纹在processed_data_audits和historical_control，原文本向量Qwen2.5-7B-Instruct mean pooling、max128，保存FP32[18775,3584]，后3层×256 RQ-KMeans/residual normalization/seed42/30步，再加去重码，SID[4,18775]全唯一。

## 已做及未完成

原计划10次（shuffle/MLP×42–46）只有Y03 shuffle seed43完整三测试，patience10，best200、结束1200；其他Y01/Y02/Y04/Y07用户停止，Y05/Y06/Y08/Y09/Y10用户取消。历史status里failed/尚running字符串由真实user_stop_manifest和过程记录解释，不把五seed计划称完成。保留停止原件及已存模型，不补造缺失结果。

patience40 seed43是独立诊断：best17100、结束21100，三测试已完成，训练从头初始化；不能替代patience10结果或混入原五seed汇总。原运行/source_manifest两份完整snapshot及有效Hydra配置在run_index/source_snapshots/runs；诊断额外源码在runs/patience40_seed43/diagnostic_source_snapshot。核心模型/随机化同原GRID协议，早停等差异以各有效配置为准。

## 数据、结果和原件

源入口historical_control/yelp_aligned_v1/process_dataset.py→Qwen生成→原GRID量化/SID→run_task或run_extended_seed43→ShuffleModel→trainer.fit→按val/recall@5最佳权重→三组test；全部src/configs/codex依赖保留。input_provenance、command/launch、事件、完整配置及原测试是追溯依据，补件提交不是原run版本。

`verify_saved_results.py`读两份stage_result/六份test JSON、冻结源码并汇总。主要指标是完整SID匹配Recall/NDCG@5/10；全测试28330用户，beam10；原始100用户reference或注意力仅是固定样本，不是全测试候选。patience40的300份原注意力、固定样本记录及诊断源码另在assets，原已存诊断JSON保留其中声明的范围。历史CPU replay只说明其8个样本，不能外推全量。

资产：[assets.json](assets.json)，其中upstream包有真实向量、SID、量化器及处理后原TFRecord；downstream包有完成/用户停止尝试的6个checkpoint与原参考；attention包含原始固定样本张量；metadata包含全部此次找到的过程/配置/源码。Qwen基础模型沿yjc-grid已发布固定资产，不重复上传。

| 起点 | 本轮范围 |
|---|---|
| 保存结果复算 | 六份完整测试JSON与stage一致、源码hash验证、表可重算；实际执行 |
| 固定权重重评 | checkpoint、数据/SID/向量、源码和配置可取得；本轮未做模型推理 |
| 重训 | 历史入口和全部生效配置、环境/处理链已保存；绝对路径需在新副本映射并记录，未承诺历史调度器任意机器开箱启动；本轮未训练 |
| 上游重建 | 固定源inter/item指纹、去重/core/编号/转换代码与映射已保存；量化模型与向量是原件；本轮只核内容/轴，没有重新编码量化 |

本轮不恢复已取消队列，不改变数据、SID、patience或评价口径。当前目录保留不利结果，不将诊断改善称显著性证明。
