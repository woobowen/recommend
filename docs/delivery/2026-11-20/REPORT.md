# 仓库收尾与交付报告 v1.5

2026-11-21，America/New_York。此次交付承接用户确认已通过网页 GPT 二次核验的初次 v1.5 基线。此次修订的复算入口、已找回原件、两份学弟文档和审核证据集中在单一 ZIP。尚未恢复的历史原件继续限制完整复现；原 LC-Rec 六任务的正式完整训练仍未完成，Shuffle V2 由原任务继续执行。本轮没有新训练、固定权重复评、生成或上游重建。新交付的网页二次审核待用户转交后进行。

## 范围与实际发布依据

统一规则入口为 `/home/addaswsw/recommend`，实际 Git 对象库位于 `/home/addaswsw/recommend-worktrees/sid-extensions-p0`。使用显式读取的 v1.5 三份正文和已有受审 Git 快照。起始十条 HEAD 与上一轮回读一致。对原工作区的未提交和暂存状态先行检查，通过独立 Git 索引构建候选。正文规则、旧正式 Release、已有用户控制的 v1.5 单包、原始预测、运行日志和活动目录均保留。

| 分支 | 2026-11-21 收尾起始提交 | 归属与实际范围 |
| --- | --- | --- |
| wbw-grid | `caf3915ccaeef1ed9fee9c5b81e05dd2f6bcdf37` | 本人负责。woobowen 从原 wbw 研究分支续接；更新 GRID 和历史原件入口、总报告、两份学弟文档和审核材料。 |
| wbw-lc-rec | `64239531db7df9114d4e57153a7a00e53317df51` | 本人负责。woobowen 建立并发布 G3、V1 等研究；更新 LC 保存输出复核、六任务资产诊断及归约证据。 |
| wbw-lc-rec-shuffle-v2 | `138017b1d2ac23f363f79f7ae548694464854cdf` | 本人负责。woobowen 创建 V2 原协议并发布；只读观测原运行，更新静态过程入口。 |
| wbw-sid-order | `50411300fce69337b26d19b28e0951d5e87b24b0` | 本人负责。woobowen 创建并维护跨路线主题；接通新复算入口，保留旧 21 条主题证据。 |
| yjc-grid | `abfe7827e9c586a10edcb47c1f4a0c2a6ae62f67` | 学弟维护。woobowen 按旧同 SHA 代建当前分支，Justin-hub-coder 后续发布研究；只读。 |
| SID-Order-Invariance | `bc273dbdde56a50392791372172478f3cb8eb459` | Justin-hub-coder 创建及发布；只读。 |
| sid-semantic-hierarchy | `e8f4603ddf5e7783162efc747b2b7cc60d7da2f3` | Justin-hub-coder 创建及发布；只读。 |
| main | `563dc50b1c7e9e2e05cdfddd13e91a3d1888001e` | 共同工作基线，只读。 |
| sid-extensions | `a2bbdec09eb88f10ea22aed641d389d445039220` | woobowen 发布的冻结 P0，只读。 |
| wmh-grid | `7e943ea9a7d92f16eb10c6c1b27c27761d3f6c20` | woobowen 代建的王学长历史源码，只读。 |

依据为先前 GitHub 活动的研究创建及推送、原方案、分支内容和核验材料。包内 `evidence/github-activity.json`、`evidence/closeout/refs-before.tsv` 保留原始依据。上轮综合报告仍可按[固定版本](https://github.com/woobowen/recommend/blob/caf3915ccaeef1ed9fee9c5b81e05dd2f6bcdf37/docs/delivery/2026-11-20/REPORT.md)审阅，全文同时收入包内 `history/previous-v1.5/REPORT.md`。该报告记录四本人分支原研究补齐、学弟三分支查验、活动记录和三阶段范围。

## 本轮实际结果

| 范围 | 已完成与证据 | 复现限制 |
| --- | --- | --- |
| GRID | 移除原输出卷入口。单包可按原字节还原 27 组、706,641 行 JSONL；实际新复核运行了完整排名、单份恢复及 26 组 attention，另保留三域 baseline CSV 与实际命令和有效条件。恢复 Toys 历史 embedding／SID 路径链、原配置和日志，取回实际 SID 字节并与正式身份匹配。 | Parallel 真正非自回归源码、Toys 唯一编码器 producer 和早期 baseline 逐用户输出仍需原保管人。 |
| LC-Rec | 改用单 ZIP / NPZ 入口，核对 21 个冻结分片、12 条件和八组配对；将原 gzip 的获取方法与字段和值恢复分开。进一步核实六任务 24,772 人配对与原全参资源门槛，加入来源和可复算数值投影。 | 辅助与主数据 split 不配对，原作者生成时点与上下文未取得。原全参训练未完成，后续 LoRA、offload 与 Qwen 仅有证据所限定的技术验证。 |
| Shuffle V2 | 只读跟踪原 controller、进程状态和已保存 CPU 回执。07:22:39 UTC 看到 S2、S3a、S3b 的 GPU 生成及 CPU 评价退出 0；S4 5,725／24,772，仍在生成，保留原状态和观测时间。 | 尚未进行四条件实验后验收。正式运行身份 `e2f54ebd2d7709821de1071693f91644f5ed9a30`，沿原分支记录接续。 |
| SID 顺序主题 | 衔接 GRID 24 个已保存扰动条件、LETTER 原方案及独立 V2 入口。主题较早的 21 条固定来源按原时点保存；补充 12 条 GRID A 结果的收录位置改用当前单包命令。 | 跨模型的因果、数据／SID／模型和评价公平对齐属于另一个研究问题。 |

[历史原件与索取对象](HISTORICAL_SOURCES.md) 给出三项 GRID 缺件的实际查验、找到的原件和四类复现起点的影响。[复算与获取](REPLAY.md) 提供可运行命令、恢复字节和原容器边界。保存输出复算直接核对上述已归档范围；固定 checkpoint 重评从各实验真实权重、映射、数据和生成评价协议开始，本轮未执行；重跑按固定 export、原配置及 attempt 的执行顺序，本轮未执行；重建上游资产依赖真实 producer、原数据和权重，未知项仍未通过。现有权重按原来源和权限获取。应用于旧卷的清理已经由早前回执记载，现行入口不再要求已撤下的审核卷。原预测还在正式项目位置。

## 六任务 CE 权重归约更正

上一轮报告记录旧 NumPy 1.24.4、新 NumPy 1.26.4，并用版本差描述末位变化。该因果措辞在此更正。冻结 `analyze_grouping.py::ce_grouping` 在第 123、128 行调用 Python 内置 `sum`。对冻结的同序权重用逐项相加得到所有 14 个历史 binary64 值，用 `math.fsum` 得到所有 14 个上一轮整理值。原服务器 CPython 3.10.22 和 WSL 3.11.14 的内置 `sum` 命中全部历史值，WSL 3.12.3 命中全部整理值。该因果测试未导入 NumPy，最大绝对差 `2.220446049250313e-16`。结论覆盖这 14 个 mixed global128 汇总字段。实际原始值、原日志、报告和非作者历史意见保持原件；本段的解释是 2026-11-21 新核对所得。 [非作者独立实测](evidence/review-ce-interpreter-sum.json)

## 学弟交接

[学弟分支检查与补全清单](学弟分支检查与补全清单.md) 给出三分支分别 3、5、4 项，以及原件、影响、优先级、建议修改位置、验收与依赖。三条 HEAD 的起始核对与旧查验一致，采用原核查日期和范围。完成顺序是保全和定位原件、补实际代码与资产、补可复算结果和复现入口、自审发布并回读。[GitHub 上传与实验交付说明（学弟适用）](GitHub%20上传与实验交付说明（学弟适用）.md) 含 v1.5 两份详细规则、样例和自查表。三个旧相对规则链接用已验证固定版本，正文也标明 ZIP 的 `rules/` 离线路径。三份交付位置两文正文一致。此处完成只读核查及交接，分支整改归其维护者。

## 审核与取得

[非作者审读](NONAUTHOR_REVIEW_CLOSEOUT.md) 记录关键原件、新差异、命令与复核过程。本次说明和复算检查与先前证据分开，运行 commit、文件内容版本和 ZIP 的 SHA-256 也各自标明。[现行下载、SHA-256 与整理入口](REPLAY.md) 提供同一完整包。单包按既有可复核内容作定向替换和补收，必要的 27 组原 GRID、21 个原 LETTER/G3 分片的全解析字段、原注意力和六任务投影均可直接在包中核查。包不嵌套旧 ZIP，原件相同内容按映射保留。具体文件和包外身份见清单、最终校验和发布回读。
