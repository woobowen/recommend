# SID 拓展材料：先读代码，后决定实验

本入口属于 `sid-extensions-p0`。这是材料准备与解释交接阶段；`experiment_execution_authorized=false`，`user_understanding_confirmed=false`。完成本阶段不会授权模型加载、预跑或训练。

受审 CONTENT_COMMIT 为 `84d4533fbdd6b010f54d671b433a9952608911ca`。C 已实际发布、全必需源码/配置从 GitHub 回读并部署核对；最终交付 D 的完整 SHA、最后回读时间及终态见 Codex 最终回复和外部支持目录回执。仓库内报告只记录已经发生的 C 事实，285项受审源码配置在 C/D 保持同字节。

## 三份身份不要混用

|身份|固定来源|本阶段用途|
|---|---|---|
|工作 baseline|`563dc50b1c7e9e2e05cdfddd13e91a3d1888001e` 的 `GRID baseline/`|Qwen → RQ-KMeans → 普通 TIGER；三域必要资产的唯一工作来源|
|学长原生材料|`A800-grid-20261010-v1`，经已验收 v6 映射从 new-server 归档按需恢复|`references/wmh-original/GRID/`；保持原字节，不表示已适配当前 baseline|
|本阶段说明与准备工具|`docs/sid-extensions/`、`manifests/sid-extensions/`、`tools/sid-extensions-prep/`|只做文件准备、静态证据与审核；没有新增研究实现|

`SOURCE_FACT`：基线普通模型与捕获的普通模型正文相同，但项目其他文件存在差异，例如 main 的 `BaseModule.test_step` 已采用普通模型兼容的调用。不能把整项目称为同一实现。

## 建议 GPT 网页版的阅读顺序

1. 读 [REPORT](REPORT.md)、[要求矩阵](REQUIREMENTS_MATRIX.md)、[审核入口](review/README.md)，先确认材料及固定版本身份。
2. 读 [CODE_READING_GUIDE](CODE_READING_GUIDE.md)，从 Beauty 一条交互序列如何变成 SID、embedding、目标标签讲起。
3. 对照 [Shuffle](experiments/shuffle.md) 的三种干预位置，再读 [MLP](experiments/mlp.md) 的聚合与参数变化。
4. 读 [Parallel](experiments/parallel.md)，明确当前文件名与真实算法之间的缺口，保留设计问题。
5. 读 [Attention](experiments/attention.md)，区分四个生成步骤、attention head 与历史图的身份限制。
6. 最后读 [已知问题与待改动](KNOWN_ISSUES_AND_NEXT_CHANGES.md)。让用户逐项决定后续研究与接口方案，另行授权执行。

|实验|实际状态|当前可以讲什么|
|---|---|---|
|Shuffle|`NATIVE_REFERENCE_PRESENT; PROPOSED_NOT_IMPLEMENTED`|查表后重排、双表差异、hook、mask、随机排列协议|
|MLP|`NATIVE_REFERENCE_PRESENT; PROPOSED_NOT_IMPLEMENTED`|有序拼接聚合、SEP 取消、encoder 长度、梯度与参数|
|非自回归 Parallel|`NOT_RECOVERED; RECONSTRUCTION_PENDING_USER_DISCUSSION`|现存文件仍 AR；缺哪些正文、哪些设计问题必须讨论|
|Attention|`NATIVE_REFERENCE_PRESENT; COLLECTION_NOT_RUN`|现存采集/绘图链及 beam、样本、mask 的证据边界|

`HISTORICAL_OBSERVATION`：[Baseline Results](../../Baseline%20Results/README.md) 是固定提交已发布的历史记录，本阶段统一标为 `HISTORICAL_REPORTED_NOT_RERUN`。本次没有重算或挑选指标，也没有把旧 T5 历史结果绑定到 Qwen 工作资产。

## 怎样核查一句话

所有源码引用使用冻结文件的路径、符号和实际行范围；[SOURCE_MANIFEST](../../manifests/sid-extensions/SOURCE_MANIFEST.json) 给出完整 SHA-256、Git blob 或捕获/归档成员身份，[SOURCE_LINES](../../manifests/sid-extensions/SOURCE_LINES.json) 给出静态符号索引。四份说明末尾的源码证据表由同一字节生成。最终交付 SHA 与最后 GitHub 回读收据位于支持目录，避免要求报告包含自身 commit 的 SHA。

陈述性质统一为：`SOURCE_FACT` 源码正文；`CONFIG_DECLARATION` 保存配置声明；`HISTORICAL_OBSERVATION` 既有结果/审核；`STATIC_INFERENCE` 未运行的推导；`TEACHING_EXAMPLE` 人为例子；`PROPOSED_NOT_IMPLEMENTED` 尚未实施建议；`UNKNOWN` 不能确认。配置中声明某 target，不等于已经运行成功。

大型载荷见 [ASSET_MANIFEST](../../manifests/sid-extensions/ASSET_MANIFEST.jsonl)，路径对应见 [DEPLOYMENT_MAP](../../manifests/sid-extensions/DEPLOYMENT_MAP.json) 与 [PATH_MAPPING](PATH_MAPPING.md)。原 baseline `experiments/**` 的小配置也是 LFS 指针，因此必要有效配置另以原字节普通 Git 文本发布在 `manifests/sid-extensions/baseline-run-records/`；这不修改原 baseline 的任何 Git blob。
