# 仓库材料整理与交付报告

2026-11-20，America/New_York。通过 WSL 核查 GitHub，并以既有 Windows OpenSSH 对 `new-server` 上属本项目的证据作只读查询。目录扫描核实原 `/home/addaswsw/recommend` 是 v1.4 规范与历史材料入口。四个常用工作副本保留原内容，发布候选各从实时远端 HEAD 在独立工作树准备。

## 范围与上传事实

查询仓库 heads 和 206 条可取得的历史活动，覆盖 2026-09-30 至 2026-11-20T19:38:39Z；可读 PR 列表为空。包内 `evidence/refs-before.tsv`、`github-activity.json` 和 `github-prs.json` 记录查验输入。以工作说明、原源码、实验记录和整理开始前的研究发布核实职责。

| 分支 | 整理前 SHA | 用途与依据 | 本轮范围 |
|---|---|---|---|
| wbw-grid | `026efd1cdfabb9dcf6e23d8003f9e1d72c6554e1` | 由 `woobowen` 2026-11-08 续接原 `wbw` 的 `9767729a5114e21b9c949892af3d759baca6a98e`。GRID Shuffle、MLP、注意力与恢复，用户研究工作分支。 | 本人，读写 |
| wbw-lc-rec | `31983c2ff0df5a0a4d48d9af8f74600114786eb6` | `woobowen` 建立并持续发布；G3 收尾 `7368d301d0d928f77d50551c3b53a5b30c23d63f`、V1 `94b79854a59f52208bc11f14fc6e3069594dab6a` 是整理前的研究记录。 | 本人，读写 |
| wbw-lc-rec-shuffle-v2 | `50507089981cb0b9d36e8977bd1d94cd9085f031` | `woobowen` 2026-11-20 创建并发布 V2 协议及运行，从 `a245824835a3f7457d577bf28320530b6dddf779` 开始。 | 本人，读写 |
| wbw-sid-order | `26fe78593919d4c31a0a8ac3f8642ff5cfd80815` | `woobowen` 2026-11-17 创建于 `7f7697e4baf28fb252b45e3afd356c333c455dbb`，负责跨 GRID 和 LETTER 组织与后续研究。 | 本人，读写 |
| yjc-grid | `abfe7827e9c586a10edcb47c1f4a0c2a6ae62f67` | 旧 `yinjingchen`／`yjc` 的既有工作，`woobowen` 以同 SHA `c92e1576210b3daf2ce6f63f77b65f651f6f7127` 代建现名，随后 `Justin-hub-coder` 发布三个研究提交。 | 学弟，代建事实单列，只读 |
| SID-Order-Invariance | `bc273dbdde56a50392791372172478f3cb8eb459` | `Justin-hub-coder` 创建并发布 SID 扰动和多种子记录。 | 学弟，只读 |
| sid-semantic-hierarchy | `e8f4603ddf5e7783162efc747b2b7cc60d7da2f3` | `Justin-hub-coder` 2026-11-17 创建并发布分层语义分析。 | 学弟，只读 |
| main | `563dc50b1c7e9e2e05cdfddd13e91a3d1888001e` | 公共 baseline，`woobowen` 建立默认主分支，`Justin-hub-coder` 参与发布；共同入口。 | 只读 |
| sid-extensions | `a2bbdec09eb88f10ea22aed641d389d445039220` | `woobowen` 创建和推送的冻结 P0 原交付，现行规范指定保护。 | 历史参考，只读 |
| wmh-grid | `7e943ea9a7d92f16eb10c6c1b27c27761d3f6c20` | `woobowen` 按旧 `wmh` 同一 SHA 代建的王学长 GRID 历史快照。 | 历史参考、代建事实单列，只读 |

当前 10 条 refs 均能分类，仓库活动中的旧已退役分支供溯源。2026-11-20T19:38:39Z 此前四次提交是 v1.3 规范同步，内容只涉及三份规则和 `docs/CURRENT_WORKFLOW.md`。本轮在其提交之上发布 v1.4 正文、入口与研究材料。三份本地规范均符合原 SHA 清单。清单另列的项目设置、任务 Markdown 原包和旧差异报告在规范入口未见，本轮任务从用户消息原文捕获。服务器规范副本在本次查验的项目路径中未见，远端 P0 export 和运行快照继续用各自原记录。网页项目源／设置的状态未核验。

## 本人范围的实际补齐

| 分支 | 补齐与验证 | 复现边界 |
|---|---|---|
| wbw-grid | 根入口加入 GRID 研究、核心源码、请求、资产和四起点说明。将原 baseline、Beauty A/B、Toys/Sports B、Toys A-5300/A-5000、Sports A1/A2、三域 MLP 和对应 attention 连到固定证据。审阅 A2 旧 EMFILE 与新 11 阶段。加入复算脚本及输出。原保存预测复算 27 条、706,641 行、四指标最大差 3.46e-9；26 组 fixed32 原矩阵计算 9,984 行对，核心 14 组逐对值一致，另核 12 组 A 路线两条路径四指标均值。三域历史 baseline 的 5 项测试指标各与原 LFS CSV 完全对应。 | 原始 baseline 逐用户预测未取得。历史真实非自回归 Parallel 源未取得。各域固定 checkpoint 重评需匹配真实资产与环境，未在本轮执行。 |
| wbw-lc-rec | 新增全系列入口，连接原 LC-Rec 六任务技术材料、A0/A1 Instruments、原 LETTER Beauty、G3 Beauty 和 V1。先前未归档的两套 Beauty stage request 原件现场取回，各 9 份；Instruments A0/A1 两个 Normal 逐用户 CSV/NPZ 复核原哈希并补齐。V1/Beauty/G3 共 12 个条件 × 16 项、8 个配对比较由保存矩阵复核，审核包保留 21 份稳定预测分片。 | 原六任务未通过当时的配对／资源门槛，技术预跑按实际范围保留。固定模型重评、训练和上游重建未执行。对应外置 adapter 或环境由项目权限取得。 |
| wbw-lc-rec-shuffle-v2 | 新增分支协议、seed／域与对照语义、生成和评价、历史情形、过程快照及复现入口。实际 RUN、task request、状态、已经写出的部分证据及观察时间随文档保存。 | 作业继续独立执行。正式覆盖与终态以原 controller 和对应子作业为准。 |
| wbw-sid-order | 保留旧 21 条主题语料，补接具有独立执行证据的 GRID A 四路线共 12 条，使 GRID 可回溯至 24 条历史条件。更新 LETTER 原始请求的恢复入口，并链接 V2 独立执行。旧校验复跑覆盖 21 条、137 个固定位置引用及 84 个数字。 | 因果比较、跨路线公平性和上游协议对齐待独立研究。主题文档与历史 RUN 有各自版本。 |

GRID 正式运行绑定示例为 Beauty 评测 `2cf4c793c2e0b2e9dc7e7a9269f5e54d5c667896`、Toys/Sports B `c8acac42d5e13ca80af75faac5d1f7bc33fb589f`、Sports A2 `85b77f5e2a5b497fdc1f620649ba488fc783a245`。完整矩阵中的 SOURCE、request 内容和 cwd 见 GRID 入口。LETTER Beauty 运行 `6e1b211d9c37f358c43b1540b40b17f6703150d2`，G3 `21d495b0be5b606853ec7cec68a3eaa1b58acbd7`，V2 本次新增的运行记录以该研究分支的原 run export 为准。服务器原 job 根位于 `/home/bwwu/grid-audits/sid-extensions-p0/jobs`。LETTER 执行资产在 `/data/bwwu/lc-rec`，GRID 工作资产在 `/data/bwwu/sid-extensions-assets/563dc50b1c7e`。正式结果使用各 job 的真实输入配对，审阅跨域结论时保留这些条件。

## 学弟交付与设备边界

两份文档以固定提交逐分支记述已有证据、整改顺序、所需原件和验收条件。YJC 的 Git LFS 小型有效配置、测试行和 SID 已下载校验。SID-Order-Invariance 现有 40+8 结果行复算一致，仍需据整改清单补程序和固定可取的原 asset。层级实验的已存 6 组验证通过，底层向量与重建权限另行补足。原分支由其维护者继续处理。

本轮无新研究运行、无软件安装或个人配置修改。已确认活动 GPU 由既有作业占用，所有核验为保存输出、文件或 Git 审查。旧失败与进行中数据按原状态解释。原材料不清理。非作者审核保存具体重新检查的源、命令、数据与结果，以及当前待补项。正式发布回读与各卷实际下载 SHA 在发布记录中给出。

## 复现仍依赖的材料

- Parallel 需要学长提供真实非自回归源、成套配置和候选协议，或另行批准从研究定义重新设计。现有同名 Python 可说明历史误配和逐步生成。
- Toys MLP 的早期 encoder 与 SID 的唯一历史 producer 仍需原运行依据。当前条件下可从已存输出复算，固定权重读取仍须按照原资产来源。
- 权重、完整大型输入和原环境仍按各实验记录及访问条件使用。公开审核包提供原预测、配对／统计、关键源码及配置。重评和完整重训另有外部资产依赖。
