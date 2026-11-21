# 本轮仓库收尾独立非作者复核

审核者为实际独立代理 `/root/review`。我核对本轮任务、现行规则及前轮冻结证据，再直接读取当前候选、原运行材料与执行源码。我没有编写两份学弟文档、仓库说明或包内操作改动。下列独立读回和验证程序不执行实验或修改原件。

## 历史 GRID baseline 原件

- 我直接对照 `new-server:/home/bwwu/recommend/GRID/experiments/` 与已有本地冻结副本的 2026-10-07 两个比较运行。每个作业有四份实际原文件逐字节对应。两份命令均明确 `train=true`、`test=false`、H=5、seed42。原 CSV 分别有 42 和 11 条验证行、无 test 列；现场作业目录含命令、配置、checkpoint、日志和聚合记录。由此限定这些作业的已保存证据为训练／验证。
- 2026-10-10／11 的 RQ-KMeans Qwen 工作 baseline 则有三域正式 `test_completed`。从固定版本 `9767729a5114e21b9c949892af3d759baca6a98e` 的 Git LFS OID 与长度对照取得每域六份原始小文件，共 18 份。原 `stage_result.json` 中五项测试值与发布 JSON 完全一致，CSV 各有一条四指标对应的测试行。命令与 Hydra 原覆盖可直接核对 H=4、单卡、32-true、`train=true`、`test=true` 及验证间隔 1600。Beauty 恢复原 12,500 步 checkpoint，Sports 和 Toys 的 `ckpt_path=null`。最终步分别为 13,500、10,500、7,200；三域各自最佳 checkpoint 文件名和读取入口见机器证据。
- 直接查看上述工作 baseline 的固定 `_运行/grid_stage.py`、`BaseModule.test_step` 与 `SemanticIDEncoderDecoder.eval_step/predict_step`。实际测试路径写 `stage_result.json` 聚合指标，另一个 inference 分支才持久化 `pickle/merged_predictions_tensor.pt`。服务器三域正式目录未发现可绑定该次测试的每用户候选文件。量化阶段保存的 `sid_generation/pickle/merged_predictions*` 由物品编号索引。后续 Shuffle B Normal 属于另一批作业。旧 A800 本地两份目录清单分别核到 12,007 和 5,432 个普通文件，按明确的逐用户预测命名没有命中。这是对可访问目录、实际测试实现和保全目录的定向核查，未取得的历史逐用户原件仍限制从保存输出按用户复算。
- 精确源字节、服务器目录成员及活跃路径分别见[原件目录](evidence/grid-history/review-baseline-originals/)与[查验回执](evidence/grid-history/review-baseline-origins.json)。拟索取材料是原测试逐用户文件或曾保存它的导出、相关 checkpoint、数据 split 与候选／评价配置，及足以绑定原 test 的运行记录。

## CE 权重末位差更正

我从冻结的 `analyze_grouping.py::ce_grouping` 直接确认 `original_grouped_ce_task_weight` 和 `sum_sample_loss_weights` 都使用 Python 内置 `sum`。从原 `grouping.json` 的 per-sample 行按原顺序提取前轮报告的 14 组差异权重。向既有 new-server Python 3.10.22 与本地 Python 3.12.3 送入完全相同的数组，仅执行内置求和，两边均未导入 NumPy。3.10.22 对 14 项逐一得到历史值，亦与普通从左到右折叠相同；3.12.3 对 14 项逐一得到 v1.5 整理值，亦与本轮 `math.fsum` 相同。最大绝对差仍为 2.220446049250313e-16，原值与各复算值保留在[独立跨解释器证据](evidence/review-ce-interpreter-sum.json)。前轮非作者记录中的 NumPy 因果归因经本轮直接证据更正，前轮历史正文维持其当时字节。

## 学弟交付两份全文

我独立通读候选两份全文，对照前轮冻结清单并按编号及状态解析。Y1–Y3、O1–O5、H1–H4 共 12 项完整，三条分支相应为 3、5、4 项。每项含实际位置、证据、影响、补件、旧入口更新方式和验收。已用固定证据支持的结果与仍待学弟或其他负责人补件分别表达，操作顺序具体，保留原范围和核查时点。指南涵盖分工、执行确认、代码关系、有效配置、资产、四类复现、三阶段、发布回读、包与清理，附说明模板和上传前后自查；现行指导以 v1.5 为准。

三个旧的相对规范链接现为固定 GitHub commit `caf3915ccaeef1ed9fee9c5b81e05dd2f6bcdf37` 的 HTTPS 入口。分别把固定版本的两份规则全文与本地现行 v1.5 原文 SHA-256 对比，完全一致。两份独立文件没有相对链接，正文也给出 ZIP 内 `rules/` 路径。本轮复核复用三条学弟分支未变化的固定证据；文档清单不表示对方已经整改。

## 新增历史来源、实际源码与结论范围

- Parallel 两段原 nohup 字节的长度与 SHA-256 均匹配原摘取记录，直接读到明确的 Parallel 初始化。重新读取两套保全路径下六份配置和 hparams，字节与既有审核记录一致，其中两份亦与固定 Git 正文相同。原 2025-12-02 CSV 第 64 行是唯一 test 行，记步 30,625，选用的 best checkpoint 记步 24,375。Recall@5／@10 为 0.010150695219635963，NDCG@5／@10 为 0.00809971522539854。将 2025-12-03 单次原 inference.log 的 19 条按顺序配回原日志摘录，见相同加载路径及 decoder.bos_tokens、SID 表等键不配。此结论只覆盖匹配的失败尝试。直接读取固定 Git 的 Parallel 和 Shuffle 两个 Python 原件，各为相同 61,418 字节与 e03082ad87bfc842c05467580fc479eff4345c6a0d18c5b2f7faf320d4f96c80，早期 e61baf35 的两个 Git blob 也相同。源码现状保留逐码位 decoder 前缀；这些证据无法支持真正非自回归的历史 query、mask、损失和候选协议。
- Toys MLP 从实际原档重算 embedding 及 SID 两次保全身份，分别为 170,943,798 B／db7a21b3aa4389c2bfdcc5a3be81b781cf0c3b90e02e331143e46803666dcc04、382,902 B／26bae0c90859e1ccf7517dbdd077adc75d3550bb9a637df9905c0dc6eca73eb0。包候选中的实际 SID 与保全文件逐字相同。只对 ZIP 容器与序列化元数据做静态解析，实际 storage 类型、字节数和 CRC 支持 float32 [11924,3584] embedding 与 int64 [4,11924] SID。原 config/overrides 绑定 08-44-16 embedding 到 10-05-01 SID，原训练 21-25-31 消费其 SID，21-28-23 则明确读另一份 R-KMeans 文件。保存的同尝试 scheduler 原日志明确发出 T5EncoderModel 从 Qwen2.5-7B-Instruct 初始化且部分权重新初始化的警告。上述证据建立路径、真实载荷和不一致线索，尚缺当时的编码器完整代码、实际权重和 tokenizer revision、可复核的上游有序 item ID。现行正式 Toys 训练沿其已保存 config、request、checkpoint 和 19,412 人结果定位，不因此宣称从原文本重建历史上游。
- LC-Rec 六任务通过既有 Windows SSH 直接从 new-server 读取作者原 Instruments.zip。实际 13,612,638 B、SHA-256 96aad0cf1496a8ffef31bf1ca3a0d1033ca49473dff46b2c2dc27763224f9453。独立解析原 inter、user、映射和 index，验证 24,772 人、9,922 个 item、9,897 个不同 canonical SID。ItemSearch 训练辅助完整配对 12,539 人（12,233 人不配对），测试有 7,473 人不配对。训练辅助目标分别有 3,608／1,623 人等于当前 validation／test 目标，最近 20 个可见辅助物品分别为 1,623／0 人。逐行投影的复算 JSON 与候选回执字节相同。作者辅助生成时点和语义来源仍不能由上述数字推出。按 6,746,034,176 个参数、两份 FP32 moments 与两卡分片复算，每卡 25.1309 GiB 的静态下界确实高于 23.9883 GiB。历史 offload 与 LoRA 有界结果没有扩张为正式六任务训练。
- Shuffle V2 直接核对附带现场 JSON 的原父 job 和源 commit。07:22:39 UTC 的 S2、S3a、S3b 各两条已完成记录都有 exit 0；S4 缺完成退出，父 job 仍 RUNNING。此为原阶段带时间的观测。SID 主题保留历史 21 条和新增 12 条 A 路线之间的证据范围，未把合并入口写成跨路线等条件证明。

## 现行操作及候选正文

逐一读过四条本人分支本轮受影响的入口、当前总报告、历史缺件清单、集中复算说明、GRID／LETTER 紧凑数据说明及 CE 源程序。对两份学弟文件逐字比较待发布分支副本，清单 24,183 B、SHA-256 bdbe0d70ced36d41058758acb306d83bc211441496a3ac9fd54a8ce063162c33，指南 14,898 B、SHA-256 c7f11e780201e4e9ab4a37237ff3fe70f7159b578bea5f0a92dd6fddd96dcec4。现行入口使用一份单包和包外校验，旧卷和旧状态这些引用标明确切历史时间，三类数据保真边界与原件获取路径均保留。

在真实试解包目录独立执行 Beauty MLP 提取，输出 JSONL 的 SHA-256 是 81906804a5ff438e278ad4386d7539e458e08731c282be08b9f482ba2f43ad26，匹配原文件。直接解码 LETTER Instruments A0 Normal 的 row_idx 5928、rank0，并与保留原 gzip 中同一行全部 13 个顶层字段逐项比较，相同。另在独立目录执行实际别名恢复工具，首次还原、再次零改变和目标已有不同字节时拒绝覆盖均符合说明。使用已冻结的同一处理脚本、分组、samples 和差异原件实际重跑新 CE 核验，WSL 3.12.3 的 14 项通过。六任务新配对投影复算回执逐字相同。

主控试解包原始 stdout 确认五个现行命令退出 0。结果报告覆盖 GRID 排名 27 组／706,641 行、注意力 26 组／9,984 行、LETTER 12 个条件和八个配对、六任务 workload 221,225 行。上述新增执行均从已保存输入开始。这里逐项区分我直接执行的测试、直接审读的主控原输出及复用的前轮全量审核。实际最终 ZIP 的 CRC、成员、三处学弟正文身份及容器发布另随包外回读复核。

对集中反馈做了定向复核。新增八份 Parallel 单次原 log、Hydra 配置和 hparams 与独立读取的 A800 原档逐字相同。Toys 调度日志的四条完整记录按原 LF 行号和全文件 SHA-256 对比，逐字相同。V2 入口把底部较早观察标明为 2026-11-20，并明确链接最新阶段，三条已有终态和四组尚未闭合现在可以同时核对。
