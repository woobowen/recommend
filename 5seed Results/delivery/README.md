# O1–O5：五种子运行实现与原件

维护人：Jingchen Yin（Justin-hub-coder）。此目录是2026-11-21事后补件，历史运行在2026-11-04/06之后完成；新增交付commit不是历史运行commit。本轮未重训、未重新模型评价。

| 清单项 | 本次补件与核验 |
|---|---|
| O1 | [run_index.json](run_index.json)将20次run映射到5个真实source签名；[source_snapshots](source_snapshots/)逐文件与每次原manifest一致，差异数0。完整src/configs及codex模型、启动/attention/evaluator均可读。 |
| O2 | [assets.json](assets.json)发布20个best checkpoint的真实ZIP及每模型SHA256；普通模型的三次评估共享同一权重，MLP独立。 |
| O3 | 当前README修正已删除mlp/parallel目录，区分早期快照与此次实际实现。 |
| O4 | 原4000份.pt.gz张量已发布，路径/哈希与旧source_manifest对应；保留导出/绘图原程序。 |
| O5 | [control](control/)与[runtime_records](runtime_records/)提供原任务输入、launch/status、全事件流和原日志，包括预跑/失败；已有终态不冒充独立前中审核。 |

## 真实调用与格式

`codex/plan_v1_1/run_task.py::main`解析dataset/variant/seed/gpu/output/sample-dir，组合Hydra有效覆盖→`pipeline_launcher`→`ShuffleModel`或`MLPModel`→Trainer.fit→val/recall@5最佳checkpoint→全测试→固定用户attention导出。run_index给每次使用的snapshot签名，snapshot保留原路径；root `src`/configs是必要依赖，不只提交两种模型文件。

`protocol.py`在测试时按用户ID/历史位置的SHA256、seed2026选择全部6/24种排列（包含恒等排列），只gather历史商品内部码embedding；padding不变。普通模型encoder/decoder共享表。MLP将四码拼接后经可训练MLP成为商品向量，独立从头训练；目标仍为标准四码自回归生成。`attention.py`重算各条件真实Top1前缀，因此cross-attention差异不是控制相同前缀后的因果效应。

输入SID为[4,N]、每码0..255、第四码去重；历史mask采用padding=-1，不将ID0视为padding。`SIDRetrievalEvaluator`对完整四码匹配、Recall/NDCG按batch样本数累加；beam10，测试drop_last=false，全测试与attention100人样本分开。真实loss、早停、优化器、batch、梯度累积和路由以各runtime_records的完整`.hydra/config.yaml`与overrides为准。

## 获取与四种复现起点

```bash
python "5seed Results/delivery/fetch_assets.py" --manifest "5seed Results/delivery/assets.json" --asset order-checkpoints-beauty.zip --output ./retrieved
python "5seed Results/delivery/fetch_assets.py" --manifest "5seed Results/delivery/assets.json" --asset order-checkpoints-toys.zip --output ./retrieved
python "5seed Results/delivery/fetch_assets.py" --manifest "5seed Results/delivery/assets.json" --asset order-attention-originals.zip --output ./retrieved
```

每个ZIP含原始bytes和MANIFEST；下载器核整包SHA256、CRC、路径安全，再解压核每成员SHA256。两个checkpoint包共约2.67GB压缩，展开约3.21GB；原attention约1.31GB。留出压缩包+解包所需空间。metadata.zip含完整现存过程记录和分类/编码原件，只在需要审查过程时取得。

| 起点 | 状态和本轮验证 |
|---|---|
| 保存指标复算 | collect_results.py重生成40行逐seed和8行汇总，与原表数值一致；本轮实际执行 |
| 固定checkpoint重评 | 20份权重+数据/SID身份+源代码齐备；本轮只检验权重bytes，不重新GPU生成 |
| 重跑实验 | 原run_task及完整依赖/配置/计划齐备；assemble_runtime.py组装独立目录，再用原参数启动；本轮只验证组装，不启动训练 |
| 重建上游 | 沿固定GRID基线；从已发布TFRecord开始，早期Amazon转换身份范围沿yjc-grid限制 |

`assemble_runtime.py --run-key beauty/mlp/seed_42 --baseline-root /path/to/GRID_baseline --output /new/workdir`复制对应冻结源码并链接已有资产。原run_task仍含旧quantizer checkpoint路径，组装器只在派生副本做路径映射并留下diff；原snapshot不修改。真实重跑预算和GPU另行授权。新源码不要直接覆盖活动作业目录。

## 验收与限制

[verify_order_delivery.py](verify_order_delivery.py)检查20次run、源码签名、checkpoint SHA配对、4000原张量清单、原40测试JSON与源码入口。服务器无Git导出：snapshot+每文件SHA是历史内容身份。原始full_test_reference只记录固定用户，不可冒称全测试逐用户预测。本轮没有新的训练或独立非作者PASS；本次执行方检查与清单作者的旧只读核查分开。
