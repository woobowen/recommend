# GRID 实验、执行证据与复现入口

`wbw-grid` 是 GRID/TIGER 基线及 SID 顺序、MLP 聚合、注意力和 A 路线恢复的维护分支。2026-11-20 整理以远端起始快照 `026efd1cdfabb9dcf6e23d8003f9e1d72c6554e1` 为准。各 run 的执行版本由下面的原始请求和部署清单定位。本轮从保存的输出核对结果。

## 从输入追到结果

| 实验 | 实际范围与设计 | 配置、执行与结果 |
|---|---|---|
| 工作 baseline | Beauty、Toys、Sports。Qwen2.5-7B-Instruct 文本向量接 RQ-KMeans，使用学长改版 TIGER。各域配置和实际有效参数见历史 Hydra 记录。 | [工作流水线](../../GRID%20baseline/README.md)、[论文比较和本仓测试](../../Baseline%20Results/README.md)、[工作资产清单](../../manifests/sid-extensions/ASSET_MANIFEST.jsonl) |
| Beauty A/B Shuffle | 两条不同 checkpoint／SID 路线，各 22,363 个测试用户、Normal/S3/S4。A 使用固定作者资产与作者结构，B 在普通基线接入 hook。干预发生在 encoder 输入 embedding，按有效物品组重排前三或全部四码位。保持历史物品顺序、分组及 mask，沿用原 beam/evaluator。 | [研究源码说明](experiments/shuffle.md)、[正式回执](../../manifests/grid-formal-20261015/FINAL_RECEIPT.md)、[运行与资产绑定](../../manifests/grid-formal-20261015/FINAL_VERSION_MAP.json)、[逐 job 原件](../../manifests/grid-formal-20261015/runs/) |
| Toys/Sports B Shuffle | 两域使用固定各自 B 路线，测试 19,412／35,598 用户。seed 42、batch 8、32-true、120 SID 输入位、TopK 10；prefix check 保持 false。非法、重复候选按原 evaluator 处理。 | [正式报告](../../manifests/grid-shuffle-20261017/FINAL_REPORT.md)、[执行映射](../../manifests/grid-shuffle-20261017/EXECUTION_SUMMARY.json)、[逐用户配对](../../manifests/grid-shuffle-20261017/results-audit/toys-sports-paired-shuffle-analysis.json) |
| Toys/Sports A 路线 | Toys A-5300/A-5000、Sports A1/A2，各 Normal/S3/S4，共 12 条。前三路线采用原保存结果与对应审读。Sports A2 旧量化因 FD 耗尽受控中断，新任务保持 R-KMeans 算法、数据条件及原 T5 embedding，修改 collate 与 FD 资源绑定后完成量化、训练、正式评测和 attention。 | [最终科研报告](../../manifests/grid-a2-fd-recovery-20261018/final-results/FINAL_RESULTS.md)、[新旧终态](../../manifests/grid-a2-fd-recovery-20261018/final-results/terminal-scene.json)、[生效及运行收据](../../manifests/grid-a2-fd-recovery-20261018/final-results/raw-result-receipts/)、[24 条逐条件绑定](../../manifests/grid-a2-fd-recovery-20261018/final-results/cross-domain/cross-domain-evidence-index-with-a2.json) |
| 三域 MLP | 各域 scratch 全参数训练，使用共享逐物品 MLP，四码位仍逐步自回归生成。由验证 Recall@5 取 best、patience 10，测试使用严格加载的 best。Beauty/Toys/Sports 实际 best step 分别为 2600/4400/4900。具体 raw/accum、早停、输入与批准来源见原配置。Toys 上游 encoder 唯一历史来源未确定。 | [模型及训练说明](experiments/mlp.md)、[结果与资源](../../manifests/grid-formal-20261015/FINAL_RECEIPT.md)、[原运行事实](../../manifests/grid-formal-20261015/final-integration-source-facts-20261017.json) |
| Attention | 分别覆盖 Beauty Shuffle 和 Toys/Sports MLP 核心八组、Toys/Sports B 六组以及后续 A 路线。每条件用预定 fixed32 cohort。最终 Top1 祖先链与每步局部首 beam 分开。统计可见 encoder 列，四个 query 取六对，每用户两条路径。TV/JS 仅在度量运算时归一化。 | [采集与绘图](experiments/attention.md)、[原 CPU 汇总程序](../../experiments/launch_20261017/statistics.py)、[核心八组](../../manifests/grid-formal-20261015/attention-statistics/attention-descriptive-statistics.json)、[六组](../../manifests/grid-shuffle-20261017/attention-statistics/attention-descriptive-statistics.json)、[A 路线汇总](../../manifests/grid-a2-fd-recovery-20261018/final-results/attention-means.csv) |
| Parallel 历史线索 | 保全原件的 parallel 和 shuffle Python 完全同字节，仍由逐码位前缀驱动。真正非自回归实现、当时的 query/mask、分数和候选组合不可据现有记录唯一恢复。 | [原件、配置和缺口](experiments/parallel.md) |

Toys/Sports B 的旧报告将 A 列为当时受阻。后续独立作业已经完成表中的四条 A 路线，原报告保留原时点。10 月 10–11 日 baseline 测试表保留历史日志和结果，现存 Shuffle B Normal 输出只用于相应 Shuffle 对照，尤其 Toys／Sports 的原基线指标与其数值有轻微差异。各资产和模型之间的指标差异只支持各自条件的描述。Attention 行对统计支持固定样本的相似与差异观测。判断置换不变性或非自回归可行性需要独立证据。

## 核心路径

- [工作基线源码](../../GRID%20baseline/src/) 的 `src/train.py`、`src/inference.py` 和 `_运行/run_pipeline.py` 调用数据、编码、量化、训练与推理。数据处理由 `data/loading/components/pre_processing.py` 与 `data/loading/datamodules/sequence_datamodule.py` 组织，文本经过 `_运行/generate_qwen_embeddings.py` 和 [semantic_embedding_inference_module.py](../../GRID%20baseline/src/modules/semantic_embedding_inference_module.py)。实际 Qwen pooling／归一化和 RQ-KMeans 配方按[基线文档](../../GRID%20baseline/README.md)及保存的 Hydra 运行配对。
- [学长原件](../../references/wmh-original/README.md) 保留原模型、配置、日志和脚本身份。跨项目资产按 [P0 来源与映射](../../manifests/sid-extensions/SOURCE_PROVENANCE.md) 定位。用原始 item ID 到 remap、embedding 轴、SID 行、code offset、目标标签和模型词表来验证语义配对。对三域工作资产的既有语义核验见 [A_FINAL_ASSET_REVIEW](review/A_FINAL_ASSET_REVIEW.md) 和 [路径映射](PATH_MAPPING.md)。
- [工程桥接](../../experiments/beauty_extensions/) 的 `common.py::load_config/instantiate` 选择 A/B 对应源码根、落实 Hydra 配置、绑定资产并严格加载权重。输入是实验 spec、数据和 checkpoint，输出是可执行 datamodule、模型及实际配置记录。 `author/models.py::AuthorShuffleModel/AuthorMLPModel` 沿作者原类组织当次输入、推理和评价。 `baseline.py::BaselineShuffleModel` 为 B 路线在基线 encoder embedding 接入扰动。 `results.py::recompute` 将保存预测、目标顺序和原指标对应起来。由 [通用代码总图](CODE_READING_GUIDE.md) 追到候选构造与 [eval_metrics.py](../../GRID%20baseline/src/components/eval_metrics.py)。
- [正式执行](../../experiments/formal_20261015/) 以 `run.py`、`train.py`、按条件 spec 和保存的 request 为入口。训练、评测、attention 分别绑定快照及实际调用。Toys/Sports B 在 [shuffle_20261017](../../experiments/shuffle_20261017/) 复用原 beam 语义并保存各条件结果。A 线队列和有效训练协议见 [launch_20261017](../../experiments/launch_20261017/) 和 [shared GPU](../../manifests/grid-a-launch-20261017/shared-gpu-20261018/)。 `a2_fd_recovery_20261018/adapter.py::collate_items_contiguous` 把对应列表字段合为连续存储，保留值、形状语义和原数据顺序；[等价回执](../../manifests/grid-a2-fd-recovery-20261018/prelaunch/non-author-fd-review/) 记录核查范围。
- [注意力工具](../../experiments/beauty_extensions/attention/) 的 `GenerationAttentionObserver` 保存生成步张量，`records.py::finalize_paths` 重构 beam 祖先、`load_bundle` 读 NPZ 和 JSON，`plot.py::plot_bundle` 对照 mask 与标签绘图。部署、runner、健康和资产辅助工具见 [P0 工具](../../tools/sid-extensions-prep/README.md)。CSV、表格和绘图复核按 [跨域汇总](../../manifests/grid-a2-fd-recovery-20261018/final-results/cross-domain/) 中的原程序与输入。

## 三阶段证据与复现起点

- 实验前从各 `manifests` 的 approvals／asset-audit／prelaunch 读学长或用户批准内容、配置与预跑。原技术准入由对应作业输出证明，说明中的旧日期保留。
- 实验中沿原 job 的状态、工作日志、观察与 attempt 关联追踪。GRID 正式评测和 attention 的 job 保存于 `/home/bwwu/grid-audits/sid-extensions-p0/jobs/<job_id>/`。A2 的实际部署 commit 是 `85b77f5e2a5b497fdc1f620649ba488fc783a245`，快照 SHA-256 是 `22286966cd544773f21b96e4185d56c66afd7183fdc4e8888716d8ea1d77acb2`。
- 实验后比对原 `RESULT.json`、保留的预测与注意力、终态和独立核查。以上报告给出实际运行范围、未提交模板、被替代尝试和结论限制。上一轮及收尾复核的输出见[仓库交付](../delivery/2026-11-20/)。

| 起点 | 可取得材料与本轮验证 |
|---|---|
| 保存输出复算 | 从[单一审核 ZIP 与可运行命令](../delivery/2026-11-20/REPLAY.md)进入。ZIP 的 `compact/grid/` 可逐字节恢复 27 份对应原预测 JSONL，并用同包源码复算排名指标。`grid/attention/` 提供原 record/NPZ 和对应汇总输入，覆盖按 fixed32 指标范围解释。以一次已验收 run 为汇总单位，失败历史单列。 |
| 固定 checkpoint 重评 | 每次运行中的 checkpoint、数据、SID、有效 Hydra 配置和 SHA 见上表。具备项目 SSH 权限的复现者可在作业目录内读取 request 与 status，从其实际 cwd 使用原 `argv` 和解释器重建独立复现环境。外置原件需要授权账号或由保管人交付。此轮未执行固定权重重评。 |
| 重跑约定流程 | 从当次 snapshot 和 request 开始，按原 stage 顺序组织数据、训练、best 选择、评测和采集。共享环境、精度和 GPU 绑定见原记录。Sports A2 用新 11 阶段关系，旧量化失败与未运行模板保留；Toys 上游来源未决和历史环境边界仍有影响。 |
| 重建上游 | 工作基线的 Qwen 权重、上游数据、embedding、SID 及其他大文件按原清单定位。Git LFS 应按清单 OID 从有权限的 endpoint 取得真实载荷，Qwen 分片按 README 校验合并。历史作者资产继续采用已保全原件；从源重建无法替代未确认的历史 producer。此轮实际执行保存输出复算，固定权重重评、训练流程重跑和上游重建均未执行。 |

2026-11-21 收尾定向检查了[历史原件、范围与取得条件](../delivery/2026-11-20/HISTORICAL_SOURCES.md)。最早工作 baseline 的汇总 CSV 可复核，逐用户预测仍待原保管方提供。Toys MLP 的正式下游输入和结果已绑定，唯一历史上游 producer 仍待确认。

SSH 获取前先确认授权和文件身份。仓库只需发布可共享材料。A800 完整原件继续在 `/data/bwwu/archives/wmh-a800-audit-20261010` 保存，工作资产在 `/data/bwwu/sid-extensions-assets/563dc50b1c7e`。在冻结 export 及活动作业之外检出和复查。
