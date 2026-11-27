# Title＋Categories：共同商品轴上的 SID 语义层级实验

维护：Jingchen Yin（Justin-hub-coder）。正式run `20261127_014125`，北京时间2026-11-27 01:41启动、01:54完成，退出0。Beauty原GPU0/1、Instruments原GPU2/3；本次发布只做CPU保存向量复算和归档，不重新编码、训练或改SID。

研究问题：同一个SID共享前缀加深时，商品语义是否更集中，同父节点的不同分支是否能区分。Qwen和LETTER共用同商品、文本、向量及有效掩码，仅SID树不同。Instruments已从Qwen原10611商品筛选/重排至LETTER共同9922商品，不再把两种数据宇宙作纯方法比较。

## 结果导航

- [指标表](metrics.md)、[12组全精度CSV](metrics.csv)、[完整原始汇总](run/all_results.json)、[分析报告](analysis.md)。
- [所有节点、父子边、完整路径、分支Gap、Top20及置换记录](run/analysis)：4个主分析＋8个补充。
- [输入身份与缺失掩码](run/input_audit.json)、[冻结实际配置](run/effective_config.json)、[编码配置](run/encoding_config.json)、[真实命令](run/beauty/command.json)。
- [CPU全量核验](verification/cpu_verification.json)、[资产清单及下载](assets.json)、[复现四种起点](reproduction.md)。

| 范围 | 共同商品 | 双字段主分析 | 单字段回退 |
|---|---:|---:|---:|
| Beauty | 12101 | 12094 | 7 |
| Instruments | 9922 | 9661 | 261 |

## 方法与实际调用

1. [prepare_experiment.py](run/source/prepare_experiment.py)：先构建title/categories、缺失标记、唯一文本缓存，再打开SID；同域两树共用相同文本。Beauty原TFRecord提取类别并通过已核ASIN/商品编号；Instruments使用[作者item2id与ASIN全量映射](alignment/README.md)，Qwen筛选码值逐件不变。
2. [encode_fields.py](run/source/encode_fields.py)调用[original_encoder.py::load_encoder](run/source/original_encoder.py)：Meta-Llama-3-8B-Instruct＋LLM2Vec MNTP adapter合并＋supervised adapter；双向注意力、正文masked mean、4096维、FP16两卡切层，float32保存。未用Llama3.1、未做本数据指令微调、未读SID、未量化向量。
3. title与categories分别编码得到未经字段L2的t、c，融合z=normalize((t+c)/2)。缺字段回退仅用于fusion_all_available；fusion_complete只用两字段有效商品。title_complete使用同一主分析掩码，隔离字段变化；完整序列化文本见[inputs](run/inputs)。相同类别文本只编码一次，不合并商品。类别信息来自平台原件，未人工生成标签。
4. [analyze.py](run/source/analyze.py)→[semantic_metrics.py::TreeMetrics](run/source/semantic_metrics.py)：前3位、组内真实不同商品平均两两余弦；n>=2、排除自比较，float64向量和恒等式精确计算。Qwen第4位去重码与LETTER第4量化位都不用于本次三层统计。
5. [launch_fusion.py](run/source/launch_fusion.py)：冻结源/输入/模型hash、预跑、两GPU worker双锁、服务器独立监督，完成后CPU统计。状态/子进程真实退出码保留，旧绝对路径为历史证据；新机器不得直接运行旧launcher，应按[可移植入口](reproduction.md)。

## 指标与预先固定口径

S(G)=sum(i!=j)z_i·z_j/[n(n−1)]。每层S对可评估节点宏平均；不同层节点集合不同，S2−S1不等于平均父子增益。Δ=S(child)−S(parent)，仅2<=nchild<nparent；正增量Δ>1e−6。完整路径要求n1>n2>n3>=2，S1<S2<=S3（epsilon1e−6）。Gap是父集合内同子分支商品对均值减跨子分支商品对均值，商品对内部加权、最终父节点宏平均。

覆盖率是位于至少两件商品前缀组中的不同商品数/当前分析商品数，不是推荐准确率。第三层主分析覆盖：Beauty Qwen18.98%/LETTER5.57%，Instruments Qwen21.36%/LETTER5.57%；主要证据放1→2，第三层只代表可评估子集。高频路径按第3位商品数预排序，不按余弦好坏筛选，反例保留。

主分析999次父节点内保组大小条件置换，2000次第一层子树簇bootstrap；4个dataset×SID的Δ12主检验统一Holm（均0.004）。这些p检验相对各自随机对照，不是Qwen与LETTER差值检验；相同数据/掩码不等于相同分组大小/可评估范围，不宣称某方法整体更优或跨训练种子显著。

## 研究结论边界

四组均表现出正的平均父子增益、正Gap和多数路径递进，支持编码空间中的统计语义细化，不证明显式的人类类别树，也不是每条路径都单调。Qwen上游用过Categories，故本轮包含共享类目信息的保留效应，不能单独当作完全独立于上游输入的证据；title-only同掩码对照随包保留。加入Categories提高绝对余弦，但Δ/Gap较title-only缩小，不能把高余弦等同于更强层级。

Qwen是在原10611商品宇宙训练后筛到共同9922；LETTER是在作者数据宇宙学习的固定SID。本轮不重训上游，因此只控制当前评估商品集合，不声称上游训练数据也相同。LETTER完整SID碰撞保留，评价单位仍是不同商品。

## 资产、来源、三阶段记录

原run是服务器无Git导出的固定快照；[launch.json](run/launch.json)15文件SHA256是实际执行身份，本次Git提交仅为发布版本。基座/adapter实际核验见[model_verified.json](run/model_verified.json)。基座不重托管；固定HF/ModelScope来源、取得条件和hash见[reproduction.md](reproduction.md)。两个LLM2Vec adapter及实际tokenizer/配置放metadata Release，所有原NPY按域发布并逐成员校验。

前置独立审核[launch_review](history/launch_review.md)，启动实际非作者复核[launch_runtime_review](history/launch_runtime_review.md)，实验中[启动记录](history/启动记录.md)与原console/status，实验后[status](run/status.json)、原全结果和CPU复算。本轮旁支不允许调用子代理，因此没有新发布阶段非作者审核；不将前置独立审核冒称最终发布审核，结果发布由主控自检。该范围限制不会改写任何原始记录。

失败尝试[014041](attempts/failed_014041/status.json)因原Beauty标题换行导致准备正则断言，尚未加载GPU；修正只允许跨换行，新attempt014125成功。原失败源码/日志保留。

归档输入/输出/映射及必要调用均提供；旧Qwen metadata264MB未重复入包，原件仍留服务器，实际新编码文本与完整ASIN链已提供。原模型与Amazon/作者数据各自遵守来源许可和权限；本次不自行给第三方材料声明额外许可证。
