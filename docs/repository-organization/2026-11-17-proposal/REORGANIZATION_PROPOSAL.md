# GRID / LETTER-LC-Rec 组织方案（盘点草案，PARTIAL）

本轮只调查和提出方案，未实施重组。**建议先建设跨路线的 SID 顺序扰动主题导航，暂不创建新分支、不迁移生产路径**。未来如果确有跨路线持续共同开发，候选 `wbw-sid-order` 可从本轮 LC 起始快照 `12eb8b176b156f3be7d7533ee8f443db02441303` 建立，届时须重新核对是否应以前进后的 LC 交付作为基点并取得批准；不从 orphan 或复制文件建立。GRID 与 LC 原分支继续作为各自权威维护线，历史运行永远引用原 commit；若主题分支启用，明确一个集成维护者和同步方向，不能在三条线重复维护同一文件。优先不建分支的原因是 LC 起点已经包含 GRID 的全部相同对象，现有研究路径与运行身份紧密耦合，导航即可解决发现问题；新分支会增加同步和所有权成本。数据域、模型、seed、Normal/S3/S4 是同一问题的实验维度，不是自动分支边界。

**状态限制：PARTIAL。** 本材料记录已直接读到的代表性源代码、配置和报告，但尚未逐一贯通全部组合的 request→有效配置→代码→资产链；目录枚举不等于逐文件内容审计。非作者代理无文件工具，未完成本方案独立复核；网页审查不能替代缺失的来源核验。具体缺口见第 8 节。不要据此宣称全部通过标准已经满足。

## 1. 快照与覆盖

远端 2026-11-17 09:31:07 UTC：GRID `9767729a5114e21b9c949892af3d759baca6a98e`，LC `12eb8b176b156f3be7d7533ee8f443db02441303`。Git merge-base 等于 GRID，故 GRID 确为 LC 祖先；直接比较 tree，八个共享顶层条目的对象 ID 全等，LC 增加 `lc_rec/`，不是按同名猜测。GRID 4,758 tracked 文件，LC 11,175，差 6,417，见 inventory.json。文件数由本地 Git 完整递归树获得，非截断 API；只枚举元数据，不等于每一文件已读。最终方案提交是新增文档的交付身份，不是任何历史 RUN 身份。

入口 `/home/addaswsw/recommend` 不是 Git 根；三份规则从此处显式读取。工作树 `/home/addaswsw/recommend-worktrees/lc-rec` 与 `/home/addaswsw/recommend-worktrees/sid-extensions-p0` 有先存未跟踪内容，保护不纳入提交。此处无 SSH、模型加载、GPU、LFS 拉取或新实验。仓库内的服务器路径、哈希和以往审核结论只是历史记录，本轮没有现场重验。

## 2. 实验族与状态

| 研究问题 / 实现 | 条件与状态 | 可信入口及限制 |
|---|---|---|
| SID 顺序扰动 / GRID Beauty A 与 B | 两路线 Normal/S3/S4 六组正式全量；各自对各自 Normal | `manifests/grid-formal-20261015/FINAL_RECEIPT.md`、`FINAL_VERSION_MAP.json`、`runs/gf-20261015-{a,b}-{normal,shuffle3,shuffle4}-v1/`。A 与 B 的模型/SID 不同，不是严格同模型对照。 |
| SID 顺序扰动 / GRID Toys、Sports B | 两域 B 各三条件正式；Toys A 来源未决、Sports A 配套缺失 | `manifests/grid-shuffle-20261017/FINAL_REPORT.md`、`EXECUTION_SUMMARY.json`；RUN `c8acac42d5e13ca80af75faac5d1f7bc33fb589f`。缺失 A 未执行。 |
| SID 顺序扰动 / LETTER Instruments A0、A1 | 两模型各自 Normal + S3/S4，四组 Shuffle 正式；报告内部终审 | `lc_rec/shuffle/FINAL_REVIEW_ENTRY.md`；Normal 入口 `lc_rec/letter/A0_NORMAL_FINAL_REVIEW_ENTRY.md`、`lc_rec/a1_qwen3_06b/A1_NORMAL_FINAL_REVIEW_ENTRY.md`。 |
| SID 顺序扰动 / LETTER Beauty A0 | Normal/S3/S4 生成；S4 独立统计补算完成，原 paired 超时、原 collect 未执行 | `lc_rec/a0_beauty/evidence/paired-s4-supplement-20261117/FINAL_AUDIT.md`；真正非作者和网页审核 PENDING。 |
| SID 聚合 / GRID MLP | Beauty、Toys、Sports 正式训练及测试；固定样本注意力另属观察 | `manifests/grid-formal-20261015/FINAL_RECEIPT.md` C、D 节及各 MLP run；不是 Parallel。 |
| 非自回归 Parallel | 真实非 AR 正文未恢复，待讨论，不能视为已执行 | `docs/sid-extensions/experiments/parallel.md`：现存同名代码 generate 仍循环层级；仅历史参考。 |
| 注意力观察 | 与各已完成 GRID 条件关联的固定 32 用户描述统计 | 同一 `FINAL_RECEIPT.md` D 节及 `manifests/grid-shuffle-20261017/FINAL_REPORT.md`；观察维度而非自动新分支。 |
| 原 LC-Rec 六任务、offload、Qwen | 有界技术预跑或受阻，不是 LETTER 单任务正式结果 | `lc_rec/six_task_lora/REVIEW_ENTRY.md`、`lc_rec/offload/REVIEW_ENTRY.md`、`lc_rec/qwen_preflight_20261110/REVIEW_ENTRY.md`、`lc_rec/qwen_six_task_20261111/REVIEW_ENTRY.md`。六任务辅助资产配对与 Qwen mask 问题保留。 |

实验族统一不代表实验条件公平对齐，更不能说明差异只由 LLM 造成。GRID A/B 与 LETTER A0/A1 是不同编号体系；GRID 上游文本编码器 Qwen 与 A1 下游推荐器 Qwen3-0.6B 也不是同一组件。共享 Normal 以各正式原运行记录为唯一来源，主题导航链接，不复制指标或模型。

## 3. 两份实填导航样稿

### GRID Beauty：SID 顺序干预

问题：固定各自的模型和资产，在 test 时比较每件历史物品内的 SID embedding 顺序。`experiments/formal_20261015/run.py` 经 `experiments/beauty_extensions/common.py` 装载配置和数据、实例化模型，`test_lifecycle` 生成及评价，结果再按预测重算。A 使用 `references/wmh-original/GRID` 中作者路线及 `author/models.py::AuthorShuffleModel` 桥接；B 使用 `GRID baseline` 和 `baseline.py::BaselineShuffleModel`，其 encoder lookup hook 取用作者 shuffle 函数。两者是不同资产路线。`common.py` 以自身 `parents[2]` 定位根，插入 `sys.path`、切换 cwd，改写 Hydra `_target_`、SID/分割数据/路径并严格加载 checkpoint。移动它或原件会改变解析，文件字节相同也不能证明行为相同。

A S3 示例：`experiments/formal_20261015/specs/beauty-a-3.json` 固定 seed42、generation_k10、SID SHA `188af0341d8c232de73acd9aa7fc3390db8d430c8c71a94ec9816962d3d94f68`、checkpoint SHA `21b174d051548d28eadc995c365c6c7beaf7410535fe726b8f720e9555a04e00`；实际配置 `manifests/grid-formal-20261015/runs/gf-20261015-a-shuffle3-v1/output/effective_config.yaml`，对应 request/RESULT 和部署、逐 job 版本映射在 `FINAL_VERSION_MAP.json`。B Normal spec 指向不同 SID SHA `d41adcfb22a9a47d5e0424c068f4c74f33cc9fdbaf5db08540fbfa0e3d720e18` 与 checkpoint SHA `3d8f03cf042293f2f7f02132c9d072f573f671d9550d567edd1754f50e53e605`。二者服务器资产是历史引用，未现场检查。完整原精度、全部六组及预测证据见 `FINAL_RECEIPT.md` 与 `results-review/beauty-six-formal-results-independent-summary.json`。例如 A Normal Recall@10 0.066449046，A S3 0.066359609，A S4 0.065733574；不要只凭这一数值推断机制。作者原件、工程桥接、派生实际部署须分开引用。

### LETTER Beauty A0：SID 顺序干预及补算

问题：同一 Beauty A0 adapter 和 Prompt0 下比较 Normal/S3/S4。正式源码 RUN `6e1b211d9c37f358c43b1540b40b17f6703150d2`；`lc_rec/a0_beauty/config/formal_beauty_seed42.json` 记录单 SeqRec、seed42、4 epoch、LoRA、BF16、global64、epoch 验证按最低 eval_loss 选模、beam20。训练 8214 更新，选 checkpoint-6160，不是最后一步。该文件还含存储上限与训练参数两个不同字段，不能未经运行记录核对即合并解释。数据及 SID 哈希、实际 argv、部署与产物引用见最终审计及下属证据；资产未由本轮重取。

`lc_rec/shuffle/entry.py` 对冻结 Normal 入口检查 SHA、生成并保存每 rank 实际派生源码及 diff；`runtime.py::intervene` 在原 tokenizer/Collator 后、generate 前按历史物品分组处理 token IDs；`core.py::transform` 验证组边界与多重集。此处为 Instruments 实现证据；Beauty 具体派生与实际运行绑定仍需逐文件核查，不能只凭同目录推定字节相同。生成、独立评价与补算阶段身份分别保留。统计源码 `1934c01eee767b665ef1043995ffc741958e30e0`，job `letter-a0-beauty-supp-s4-paired-20261117-a0`；原 paired `TIMED_OUT/-15`、原 collect 未运行，独立补算不能写成原控制器全流程成功。最终原精度表及 CCE 定义看 `FINAL_AUDIT.md`；该文标明补算内部审核完成、真正非作者与网页审核待完成。对 S4 与 S3 没有专门配对检验。

Instruments 对应样例：`lc_rec/shuffle/FINAL_REVIEW_ENTRY.md` 固定 A0 GPU Normal `4c6a83d9ba42dc2fe737792759e410ef1874da76`、CPU评价 `3d11258daa6aa4fb2bed0cd013bca50536fe351d`；A1 GPU Normal `0e3e2c6f5c1a76152ea0228390d94ffc1ff75ba5`、CPU评价 `445f1475cd1dd70d19881dbc1639bb311f7cccf7`；Shuffle/评价 RUN `f0242b9f90ae23a4979d688a287734ca9f88cea1`，最终 Shuffle 交付 `3396d45a041424fe4af625bfc0b047467150c5c1`。A0 LLaMA-7B，A1 Qwen3-0.6B；均为该方案各自 Normal 的对照。报告列 24,772 用户、918 SID token、shared seed42 非恒等置换、beam20、各模型所选 step8240；原预测和权重在服务器，Git 有轻量结果和回执。A1 配置中 `load_in_8bit` 的训练配置不能自动当成推理载入方式，正式 Shuffle 报告指出 BF16 Base；需按各实际阶段入口核对。

## 4. 原件、变体和材料层次

`lc_rec/vendor/original/` 是 RUCAIBox/LC-Rec `e6f1b3b3f8b4fee55a49a7b3dfeba29dbfcf0669` 原件；`lc_rec/letter/vendor/LETTER/` 是 HonghuiBao2000/LETTER `8d0154e28de37dbb6e24871c508ad8ddb1921cda` 的独立原件。前者六任务及数据配对问题不能套用后者的单任务正式记录；保留各自来源及许可状态。`references/wmh-original/GRID/` 是作者原件/历史资产，`experiments/beauty_extensions/` 为桥接与研究适配。`GRID baseline/` 中 Qwen→SID 为上游资产生产线，不等于下游 Qwen 推荐模型。

内容可得性分开：源码/部分配置/轻量报告为 Git blob；Git LFS 指针不能证明载荷存在；大模型、数据、checkpoint、完整预测多为服务器路径及哈希引用；本轮不取载荷且不现场核验。个别历史上游生产者无法唯一绑定的应标 UNKNOWN。按每个 RUN 的 request、部署清单和实际 config 判断，模板或最新 DELIVERY 不能代替运行绑定。`inventory.json` 是本次一次性机器目录清单，不是持续维护台账。

## 5. 建议目录（示意，尚未创建）

```text
现有 LC 树（GRID 树为其八项共享子集）
├── GRID baseline/、Baseline Results/    [保留历史及原路径]
├── experiments/、manifests/、references/、tools/ [现存；不移动]
├── docs/sid-extensions/                [现存；历史说明保留]
└── lc_rec/                            [现存；不移动]
    ├── vendor/、letter/vendor/        [原件，保留]
    ├── letter/、a1_qwen3_06b/、a0_beauty/、shuffle/ [现存]
    └── six_task_lora/、offload/、qwen*/ [预跑/参考，保留]
拟新增读者导航：docs/research/sid-order/README.md [未实施]
  ├── GRID / Beauty、Toys、Sports → 固定旧入口链接 [引用]
  └── LETTER / Instruments、Beauty → 固定旧入口链接 [引用]
```

路径动作以 `path-map.csv` 为准：全部原路径保留，新增单一主题入口及必要的两个实现说明；先不迁移。若后续真正迁移，必须事先梳理 Python import、cwd、`parents`、Hydra `_target_`/插值、相对链接、脚本相对路径、checkpoint 的类名模块、原文件 SHA、Plan 与 deployment manifest、源文件派生锚点、LFS 属性/许可及服务器 release/资产引用；保存旧入口与字节身份，轻量接口实测后才可判断行为。研究逻辑修改须单独任务批准。

## 6. README 后续拟议文案，非本轮修改

根 README 可增：『研究主题入口：SID 顺序扰动（GRID 与 LETTER-LC-Rec 的具体实验和固定证据见主题导航）。GRID baseline、历史结果与原资产继续位于原路径；按实际需要获取相应 LFS 载荷。』避免当前“一律 git lfs pull”的广泛下载暗示。LC README 可加阶段说明：『此文是 2026-11-08 原 LC 六任务准备记录，不限制其后经单独授权的 LETTER 正式结果；最新证据入口见主题导航。』保留原阶段限制原文与时间，不能将其改写为所有后续任务均禁止或冒充当前状态。

## 7. 后续实施和验收（需另行授权）

1. 先批准实验族、是否只导航及维护责任；若确定新分支，再确认精确基点与同步规则。
2. 第一独立提交仅新增主题导航和 GRID/LETTER 两份说明，指向固定 commit 的原报告、有效配置、RUN/结果；不移动生产文件。第二独立提交才可修改两个 README 导航；旧阶段文字保留。每份可独立审阅及用 `git revert` 撤回新提交，不改历史。
3. 验证固定链接存在、各数字与原报告一致、状态分层与路径分类正确；用 Git 对比原路径 blob 不变。若将来单独批准迁移，先完成逐路径映射、原件 SHA、静态依赖检查与有界轻量接口；不要求 GPU 实验、完整 Bootstrap 或加载模型。无法验证的行为不得宣称无变化。
4. 范围估计：三个导航文件、两个 README 的小段文字及轻量链接检查；约半天到一天工程量，取决于补全未闭合证据和审核。主要风险是固定链接/版本误绑、旧路径耦合、历史结果状态误述。生产重构不在估计内。

待用户集中确认：①实验族划分及 MLP/attention 的归属；②暂不建分支或未来 `wbw-sid-order` 的名称、数量、精确基点；③维持生产路径还是另行规划迁移；④未来日常维护分支及共享代码的同步责任。以上未获批准，本轮不执行。

## 8. 本轮具体未闭合与审核界限

- 已读 GRID 代表 spec、有效配置、实际执行入口、桥接、六组结果和 Toys/Sports 汇总；没有逐个校验全部 18 个历史 job 的部署字节、所有数据处理及预测。完整同组链以 `FINAL_VERSION_MAP.json` 和对应 request 为后续索引，不把其存在说成本轮全部实证。
- 已读 LETTER Shuffle 核心、派生入口、两域最终审计及配置；Beauty 派生实际运行源码、全部 prompt/data/collator 的对应版本和所有各阶段 request 尚未完整逐项核对。结果和历史内部审核为仓库报告的声明，本轮未独立重算。
- 历史各项外部 ZIP、模型/数据/权重/LFS 实载、服务器 release 本轮未核验；如网页审核必须检查这些外部回执，需要原交付包或指定可核位置。`GRID_LLM_SID_Research_Record_20261114.md` 未取得，未作为证据。
- 三名非作者子代理均报告无文件/命令工具，未真正读原件；本方案非作者审核 NOT_PERFORMED_WITH_REASON。不存在由本轮虚构签名或 PASS；已有历史审核仅引用并保持其原范围。
