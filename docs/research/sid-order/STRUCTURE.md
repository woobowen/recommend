# SID 顺序研究：结构、来源与维护

[主题总览](README.md) · [GRID 实现](GRID.md) · [LETTER 实现](LETTER.md) · [条件来源](sources.json) · [所有研究主题](../README.md)

## 分支职责与历史身份

| 对象 | 角色 |
|---|---|
| `wbw-grid` | GRID 历史研究来源，起始核对 `9767729a5114e21b9c949892af3d759baca6a98e`；本次不写入。 |
| `wbw-lc-rec` | LETTER 专项及已验收导航来源，起始核对 `2a7d193af47a533a8f0e1ab5691902ae3040970b`；本次不写入。 |
| `wbw-sid-order` | 从上述 LC 提交派生；承接该主题后续开发、联合入口和新的获授权研究，继承其他目录不意味这些内容被重新创建。 |
| `main` 和其他历史分支 | 本次不操作；不因本主题创建或同步其他实验族分支。 |

LC 基点包含 GRID 历史快照的 Git 祖先。共享树对象与祖先关系是不同证据：基点继承历史，不表示所有后续研究来自同一次运行。每次历史训练、生成、评价和补充统计的身份，以 [sources.json](sources.json) 中原始完整 commit、具体 request/state 和结果为准；**主题 HEAD 是当前文档/开发版本，不是历史 RUN**。

## 逻辑维度与物理路径

| 逻辑层级或用途 | 当前物理路径 | 身份及阅读方式 |
|---|---|---|
| 主题入口 | [README.md](README.md)，[GRID.md](GRID.md)，[LETTER.md](LETTER.md) | 统一研究问题、两种干预对象、配置、结果和差异边界。 |
| 条件：GRID A/B × Beauty，GRID B × Toys/Sports，各 Normal/S3/S4 | [sources.json](sources.json) 中 `grid-*` | 12 个条件；各自固定 job、request、effective config、结果和已记录资产；GRID A 的 Toys/Sports 缺口不补造。 |
| 条件：LETTER A0/A1 × Instruments，A0 × Beauty，各 Normal/S3/S4 | [sources.json](sources.json) 中 `letter-*` | 9 个条件；共享 Normal 训练，不代表9次新训练；Beauty 的训练、生成、评价和补算身份分别保存。 |
| GRID 实际桥接与实验 | [experiments/beauty_extensions/](../../../experiments/beauty_extensions/)、[experiments/formal_20261015/](../../../experiments/formal_20261015/)、[experiments/shuffle_20261017/](../../../experiments/shuffle_20261017/) | `common.py`、作者桥接与 baseline hook 是工程/干预入口；具体版本以每个实际 RUN 为准。 |
| GRID 原件及工作 baseline | [references/wmh-original/GRID/](../../../references/wmh-original/GRID/)、[GRID baseline/](../../../GRID%20baseline/) | 原件和不同路线 baseline 保留；不可把不同 SID/checkpoint 混用。 |
| GRID 结果、配置和请求 | [manifests/grid-formal-20261015/](../../../manifests/grid-formal-20261015/)、[manifests/grid-shuffle-20261017/](../../../manifests/grid-shuffle-20261017/) | 实际 request、effective config、结果与状态以固定快照阅读，不从目录名推测。 |
| LETTER Instruments A0 / A1 | [lc_rec/letter/](../../../lc_rec/letter/)、[lc_rec/a1_qwen3_06b/](../../../lc_rec/a1_qwen3_06b/)、[lc_rec/shuffle/](../../../lc_rec/shuffle/) | 原 SeqRec 流程、各自 Normal 和 Shuffle 派生源码/回执；原厂文件与桥接身份分开。 |
| LETTER Beauty A0 | [lc_rec/a0_beauty/](../../../lc_rec/a0_beauty/) | 独立 Beauty `formal_evaluate.py`、`shuffle_entry.py`、`shuffle_runtime.py`；state 记录生成/评价请求摘要，原字节缺口照实保留。 |
| 作者 LC-Rec 与 LETTER vendor | [lc_rec/vendor/original/](../../../lc_rec/vendor/original/)、[lc_rec/letter/vendor/LETTER/](../../../lc_rec/letter/vendor/LETTER/) | 不同作者快照与许可范围，不合并覆盖。 |
| 历史基础结果、公共工具和规范 | [Baseline Results/](../../../Baseline%20Results/)、[tools/](../../../tools/)、[docs/sid-extensions/](../../sid-extensions/) | 保留历史入口；其中 MLP、Attention 和 Parallel 按既有证据分类，不自动成为 SID 顺序正式条件。 |

这种映射将研究主题、路线、模型、数据域、干预条件和原 RUN 分层；它不声称不同数据域的相同名字意味着相同输入或语义。`sources.json` 仍是唯一机器来源清单，没有再复制第二套指标表。

## 后续维护与明确边界

本分支后续可维护主题说明、固定来源和**另行授权**的同主题研究；日常代码演进与正式运行需分别固定源码、配置、数据、权重和输出，不能将最新 HEAD 追认为旧训练版本。历史分支仍按其各自原身份保留；任何反向同步、跨分支合并、其他实验族建分支均需另行指定。

本次没有迁移既有生产路径。`common.py` 的父目录/cwd/`sys.path`、Hydra `_target_`、checkpoint 类路径、冻结入口 SHA、manifest 和服务器 export 都可能依赖原位置；即使文件字节相同，迁移仍可能改变运行。未来若批准迁移，应先逐项映射并单独验证，而不是在导航整理中隐含修改。

Beauty 三组正式生成及评价和后续 S4 补算的来源见 [LETTER.md](LETTER.md)。原 paired attempt 是 `TIMED_OUT`、原 collect 为 `NOT_RUN`；独立补算完成不改变旧控制器历史。生成/评价原始 request 字节未随已有归档提供，state 记录的阶段、路径和摘要不等于本轮重算原始 request 文件。服务器/LFS 大资产本次未重验；这不妨碍按固定记录阅读，也不证明当前实体仍可用。

本分支只组织现有证据，未统一 GRID/LETTER 的 SID、Prompt、位置/mask、Shuffle 分布或评价；严格公平性和机制分析仍待后续独立授权。
