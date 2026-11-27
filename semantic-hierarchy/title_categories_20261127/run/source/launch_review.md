# 全量启动实现的独立审核

审核者：`/root/fusion_launch_review`，非实现作者；2026-11-27。范围为本轮四个启动脚本及复用的编码器、统计实现，不代表全量结果审核或已启动证明。

结论：静态实现与已批准实验设计一致，未发现应阻止本轮预跑/全量执行的统计或编码方法问题。实际文本准备、模型加载与最长文本预跑仍须由服务器真实执行证明，失败须保留原记录。

已直接阅读：`prepare_experiment.py`、`encode_fields.py`、`analyze.py`、`launch_fusion.py`，以及旧 `encoder_worker.py`、`semantic_metrics.py`、`runtime_common.py`、`test_metrics.py`，对照本轮实验设计、LETTER 来源摘录、Instruments 对齐审核和 Beauty item mapping 首行结构。

- 文本有效掩码与去重缓存先于读取 SID 构造。两种 SID 共享同一商品轴、字段文本、向量及筛选集合；缓存只复用文本计算，不删除商品或改变商品权重。
- 编码器加载的是 Llama3 8B＋MNTP merge＋supervised adapter，双向、eval、FP16，无向量量化。通过 SSH 直接读取当前服务器 `LLM2Vec.forward`，其返回 `get_pooling(...)` 原始向量，没有在 forward 内部 L2；本轮字段原始向量平均后统一 L2 的调用正确。
- Beauty 类目有序列表保留路径；Instruments 展平类目字符串保持原文本。缺失字段不编码占位符；主分析仅双字段都有效，全可用融合与同集合 title-only 作补充。
- 统计排除自配对与 singleton，proper 边要求 `2<=n_child<n_parent`；路径要求两条 proper 边。路径按第三层不同商品数排序，未按相似度筛选。前 3 层比较没有把 Qwen 去重位与 LETTER 量化位混同。
- 组和公式、父子增益、正增益比例、路径不等式及同分支/跨兄弟分支 Gap 实现与批准口径一致。置换保父节点和子组大小；四个主 Δ12 检验共同 Holm 校正。第 3 层可评估商品覆盖单独输出。
- 已实际远程运行旧 `test_metrics.py`，退出 0，验证精确两两均值、兄弟分支 gap、两种条件置换、singleton/proper 条件、路径 plateau 与 Holm；这不是本轮模型完整评估。
- 监督脚本冻结源码，使用新 run 输出；stdin DEVNULL、日志文件及新 session 脱离 SSH，传递真实子进程失败与超时状态；同时遵守两套 GPU 锁。启动者仍须用新连接检查存活、真实编码进度和 GPU 使用。

工程建议：部署顶层启动脚本时也须放置 `runtime_common.py`，因为在冻结 source 前已有模块导入；Beauty mapping 可再显式断言 `sid_column/grid_item_id` 顺序。模型哈希核验应限定这次用到的权重。旧设计文件中的 Instruments 10611/9922 不同域及 32634 行预算已经过时，应以用户当前同轴 9922 的授权和实际准备审计为准，不据旧 census 报本轮覆盖。

本次所读版本 SHA-256（后续源文件变更需注明并定向复核）：

| 文件 | SHA-256 |
|---|---|
| prepare_experiment.py | 58d6418027d624dbde83d486b6563e825eed9bfefc1d2d0ed1ae07305198464d |
| encode_fields.py | e55819aa1cc97688110c29c8eb993f3ebd5ff2729463c676d83af6bd363c34a5 |
| analyze.py | 11de5e98a60117bfa75e7139a2c76a970ec4bed2e16ec23608a4dee048907900 |
| launch_fusion.py | a5f921eaa5b8c098bde4144df2020ba18d7852670196b4a0b71c345f3effe49d |

未验证：本轮实际准备数量、模型预跑、全量 embedding、最终指标、远端提交/上传。审核通过不代替执行记录，不预判指标必须上升。
