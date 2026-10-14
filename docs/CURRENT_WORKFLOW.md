# 当前工作分支与研究流程

用户于 2026-10-14 确认以下规则，适用于 `woobowen/recommend` 后续日常工作。

- `wbw` 是用户唯一日常工作分支，等同 current；代码、配置、实验进度和结果默认在此维护，不为每个小阶段新建分支。
- `main` 仅在用户明确要求时同步；`yinjingchen` 仅在用户指定内容与同步方向后处理。不得据此修改仓库默认分支、可见性或保护规则。
- `sid-extensions`、`wmh` 等分支保留历史，不删除、不重命名。
- 后续流程为：**用户概要确认 → Codex 适配与内部审核 → 必要预跑／正式实验 → 运行中或结果核验后，GPT 根据实际固定版本深入讲解**。新规则替代旧的“必须先完成深入讲解再做实验”要求。代码正确性仍需必要的输入、干预和对照检查，不能只按推荐指标高低判断。
- 每项研究工作仍须取得相应用户授权。本次仅发布现有成果、更新工作规则并清理同步临时件；**未授权代码／接口适配、模型加载、前向／反向、生成、训练、测试、attention 采集或 Parallel 实现**。不把 `user_understanding_confirmed` 改为 true。

## 继承版本与服务器位置

`wbw` 从已审核 P0 交付继承：

- baseline：`563dc50b1c7e9e2e05cdfddd13e91a3d1888001e`
- CONTENT_COMMIT：`84d4533fbdd6b010f54d671b433a9952608911ca`
- DELIVERY_COMMIT：`a2bbdec09eb88f10ea22aed641d389d445039220`
- [固定 P0 阅读入口](https://github.com/woobowen/recommend/blob/a2bbdec09eb88f10ea22aed641d389d445039220/docs/sid-extensions/README.md)

旧 P0 报告、授权记录、部署 marker、审核及回执保持历史原样；其中旧工作分支／先讲解流程的描述不作为后续默认规则。旧 D 部署证据不改写为本次新提交的历史证明。

- 本地 Git 工作副本：`/home/addaswsw/recommend-worktrees/sid-extensions-p0`，当前日常分支为 `wbw`。
- new-server 代码：`/home/bwwu/recommend-worktrees/sid-extensions-p0`。这是原独立源码 export，不假设有 `.git`；本次仅追加本文，研究源码和模型配置保持 D。
- 原件恢复副本：`/home/bwwu/grid-audits/sid-extensions-p0/restored/GRID`。
- 三域正式资产：`/data/bwwu/sid-extensions-assets/563dc50b1c7e`；逐文件身份见原 `manifests/sid-extensions/ASSET_MANIFEST.jsonl`。
- 原件档案：`/data/bwwu/archives/wmh-a800-audit-20261010`；本次不清理。
- 阶段支持目录：WSL `/home/addaswsw/recommend-support/sid-extensions-p0`；服务器 `/home/bwwu/grid-audits/sid-extensions-p0`。本次版本映射另记 `CURRENT_WORK_BRANCH_MAP.md`，不修改冻结 P0 回执。

本次结束后停止。后续适配、预跑和实验等待用户另行确认与授权。
