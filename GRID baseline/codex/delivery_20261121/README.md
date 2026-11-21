# Y1–Y3：基线补全与验收

维护人：Jingchen Yin（Justin-hub-coder）。本次为 2026-11-21 事后补件，未重训、未重新生成推荐。原运行代码、发布与此次补件版本分别保留，不将本提交称为历史运行 commit。

| 条目 | 状态与实际范围 |
|---|---|
| Y1 | 修正导入版本与当前树差别、LFS下载分支、G3发布占位符；G3数据/SID不变。 |
| Y2 | 三域内部商品轴完整、测试用户去重及范围检查通过；Beauty已发布完整ASIN恢复链。Sports/Toys原始独立ASIN映射与最早转换脚本未找到，外部身份链仍受限。 |
| Y3 | 索引关联三个原job、完整配置、选模、checkpoint、历史退出状态和指标；原Git/LFS过程记录按固定提交下载验证。 |

## 逐域三阶段入口

| 数据域 | 历史执行 | 后期停止步 | 测试记录分母 | 追溯 |
|---|---|---:|---:|---|
| beauty | 恢复完整训练状态 | 13500 | 22363 | [机器索引](baseline_index.json)、[原始运行目录](../../experiments/) |
| sports | 从头训练 | 10500 | 35598 | [机器索引](baseline_index.json)、[原始运行目录](../../experiments/) |
| toys | 从头训练 | 7200 | 19412 | [机器索引](baseline_index.json)、[原始运行目录](../../experiments/) |

实验前：有效 Hydra 配置、command/job 与重配请求见 original_records_readback.json 登记的固定原仓库路径和 historical_process；实验中：原日志/CSV/历史进度记录，未保存的独立审核不补造；实验后：stage_result、pipeline_status 与实际最佳checkpoint一致。
测试label函数 padding_token=-1，ID0是有效商品，因此Toys目标ID0那一条不能据此从分母排除。原evaluator按每个batch样本数累加，测试drop_last=false；本轮独立读取原始testing TFRecord分别计数22363/35598/19412，没有重新运行模型。
候选生成使用原 SID 约束与 beam10；分母、完整SID匹配和去重细节由发布的真实模型/evaluator源码及有效配置决定。本轮没有原始全测试候选逐条数据，不能补造历史无效/重复候选率。

## 四种复现起点

| 起点 | 当前能做什么 | 本轮实际验证 |
|---|---|---|
| 保存输出复算 | 原指标JSON/CSV及compare_final_test_metrics.py齐备 | 固定Git/LFS过程记录回读、指标复算；无GPU |
| 固定checkpoint重评 | 既有LFS checkpoint、SID/数据及test_saved_checkpoint.py | 本轮不重复模型推理；必要LFS身份沿用已确认记录 |
| 重跑训练 | 固定SID和完整配置可用；Beauty必须区分恢复与从头运行 | 只核配置/原件，没有新增训练 |
| 重建上游 | 固定TFRecord→Qwen→RQ-KMeans可沿原脚本；Sports/Toys早期Amazon转换受限 | 逐域item ID/向量行/SID列和测试分片边界实查 |

代码导航：`_运行/grid_stage.py`接Hydra配置，经`pipeline_launcher`进入模型训练/生成；`tiger_generation_model.py`的generate输出候选SID与分数，`SIDRetrievalEvaluator`完整码匹配后调用Recall/NDCG累计。输入SID为[4,N]、代码范围0..255，第四位是去重位。
本次未冒充非作者审核；此前清单的只读检查与本次执行方核验是不同证据。需要学长最终独立验收的范围保持待审。

143份基线历史文件已存在于原分支中，本次按固定提交读取真实Git/LFS载荷并核验，不重复提交相同字节；路径、OID和SHA256见original_records_readback.json。
