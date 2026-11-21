# LC-Rec 分支与实验复现入口

`wbw-lc-rec` 保存 LC-Rec 原六任务准备、技术预跑，以及 LETTER 单任务 Normal 与历史 Shuffle V1 实验。`wbw-grid` 保存独立的 GRID 研究路线，`wbw-sid-order` 汇集跨路线主题，`wbw-lc-rec-shuffle-v2` 从 G3 收尾快照另行研究允许恒等排列的测试期干预。仓库活动记录显示 `woobowen` 上传了本分支的研究提交，例如 G3 收尾 `7368d301d0d928f77d50551c3b53a5b30c23d63f` 和 V1 协议说明 `94b79854a59f52208bc11f14fc6e3069594dab6a`。整理前远端 `31983c2ff0df5a0a4d48d9af8f74600114786eb6` 属于此前规范发布。[仓库活动](https://github.com/woobowen/recommend/activity)

## 原件、适配和运行入口

| 路线与主说明 | 代码与配置 | 实际产物和历史边界 |
|---|---|---|
| [原 LC-Rec 六任务准备](../../lc_rec/REVIEW_ENTRY.md) | `lc_rec/vendor/original/{data,collator,finetune,test_ddp,evaluate}.py`，按顺序构造任务数据、response mask、训练、生成和计算原指标。`lc_rec/scripts/` 数据审核与候选命令，`lc_rec/config/` 约束与候选配置。 | 作者 `e6f1b3b3f8b4fee55a49a7b3dfeba29dbfcf0669`。固定 Instruments 数据 CPU 审计失败两项辅助任务配对断言，原全参无 offload GPU 门槛未通过。六任务五 seed 配方为候选。 |
| [Full-parameter CPU offload](../../lc_rec/offload/RESULTS_20261109.md) | `lc_rec/offload/scripts/` 和 `lc_rec/offload/` 实际 CPU optimizer/master offload、两卡预跑及持久状态。 | 工程可行性及预算证据，保留原六任务资产问题及方法差异。 |
| [六任务 LoRA 技术审核](../../lc_rec/six_task_lora/REVIEW_ENTRY.md)，[Qwen3 技术预检](../../lc_rec/qwen_preflight_20261110/REVIEW_ENTRY.md)，[Qwen3 六任务预跑](../../lc_rec/qwen_six_task_20261111/REVIEW_ENTRY.md) | 各路线 `scripts/` 中的 prepare/train/evaluate 或诊断入口与各自保存的请求，分别读各文档的固定版本和原审查。 | 预跑含监督、更新、显存、保存重载、恢复或生成诊断，实际覆盖因路线而异。监督与配对未决仍制约原六任务的干净复现。 |
| [LETTER 前置审核](../../lc_rec/letter/REVIEW_ENTRY.md)，[A0 Instruments Normal](../../lc_rec/letter/A0_NORMAL_FINAL_REVIEW_ENTRY.md) | 作者原 `lc_rec/letter/vendor/LETTER/LETTER-LC-Rec/{data,collator,prompt,lora_finetune,test_ddp,evaluate}.py`，来源 `8d0154e28de37dbb6e24871c508ad8ddb1921cda`。实际用 `lc_rec/letter/scripts/formal_evaluate.py` 和正式训练调用，运行 config、计划、请求和日志在主说明。 | A0 GPU `4c6a83d9ba42dc2fe737792759e410ef1874da76`，CPU `3d11258daa6aa4fb2bed0cd013bca50536fe351d`。原失败与恢复另见 [Checkpoint 恢复](../../lc_rec/letter/RESUME_DIAGNOSIS_REVIEW_ENTRY.md) 和 [FA2 backward](../../lc_rec/letter/ATTENTION_BACKWARD_DIAGNOSIS_REVIEW_ENTRY.md)。 |
| [A1 Qwen3-0.6B Instruments Normal](../../lc_rec/a1_qwen3_06b/A1_NORMAL_FINAL_REVIEW_ENTRY.md) | `lc_rec/a1_qwen3_06b/scripts/evaluate.py` 与同目录训练和恢复入口，`lc_rec/a1_qwen3_06b/config/` 记录冻结配方。 | GPU `0e3e2c6f5c1a76152ea0228390d94ffc1ff75ba5`，CPU `445f1475cd1dd70d19881dbc1639bb311f7cccf7`，24,772 个测试用户，完整 16 指标与独立审核。 |
| [Instruments A0/A1 S3/S4 V1](../../lc_rec/shuffle/FINAL_REVIEW_ENTRY.md) | `lc_rec/shuffle/entry.py` 固定并派生真实生成源码，`runtime.py` 取得历史组，`core.py::transform` 在 collate 后干预，`evaluate_condition.py` 连续执行原和扩展评价，`paired_and_changes.py` 汇总 paired，`summarize_results.py` 从独立检查的结果生成公开表。 | 研究 RUN `f0242b9f90ae23a4979d688a287734ca9f88cea1`。四组及对应原 Normal 的完整覆盖和结果见 [V1 结果](../../lc_rec/shuffle/results/RESULTS.md)。 |
| [LETTER 原 SID Beauty](../../lc_rec/a0_beauty/evidence/paired-s4-supplement-20261117/FINAL_AUDIT.md) | `lc_rec/a0_beauty/production_pipeline.py` 消费固定 plan 分发训练、生成、评价。`formal_train.py::AuditTrainer` 和 `zero2_partial_state.py` 负责跨 epoch 累积、严格恢复与 checkpoint 语义，`formal_evaluate.py` 配合 `shuffle_runtime.py` 调用作者数据、collator、Prompt0 和生成。`offline_metrics.py` 和 `paired_beauty.py` 保留结果路径。 | RUN `6e1b211d9c37f358c43b1540b40b17f6703150d2`，4 轮 8214 updates，best 6160，22,363 用户。原 paired 超时，S3 原统计保留，S4 由 `1934c01eee767b665ef1043995ffc741958e30e0` 补算。六份阶段 request 原字节见下。 |
| [G3 SID 来源](../../lc_rec/a0_beauty/grid_sid_preparation/identity_audit_abfe/README.md)，[转换与上线准备](../../lc_rec/a0_beauty/grid_sid_preparation/g3_conversion/README.md)，[正式结果](../../lc_rec/a0_beauty/grid_sid_preparation/g3_conversion/production_preparation/cpu_closure/results/README.md) | `grid_sid_preparation/identity_audit_abfe/` 的 bridge 和一致性证据核对原始文本／交互、映射与实体。`g3_conversion/convert.py` 生成四 token G3 index，前三位层级码及第四位去重，既有 LETTER Beauty 入口配合 `config/formal_g3_beauty_seed42.json`。CPU 收尾的 `predictions.py` 独立重算候选，`paired_stats.py` 配对，`review_raw.py` 非作者另行复核。 | 正式 RUN `21d495b0be5b606853ec7cec68a3eaa1b58acbd7`，CPU 补充 `1e29885ae00541aca3574b6b0b75e3ad7820c9fd`。固定单卡训练、best 6160、22,363 个用户，三组生成和评价终态。原 paired `TIMED_OUT/-15` 与未提交的 collect 保留。 |

## 数据、训练及 V1 干预

LETTER 单任务的 `SeqRecDataset` 存储逐用户交互，训练/验证/测试使用各自 split，测试保留最后物品为 target，历史至多最近 20 个。`Collator` 用 response-only CE 监督目标，`TestCollator` 产生输入及 mask。Instruments 原 LETTER index、inter、item 分别为文件 SHA-256 `33c87e28c52ae367985bdf494cb49d676d181e13a39497b4ad035f46111546dd`、`0a11d47791d90483005da93c36d826679e20e4592d1d2b86518d5f56f218e743`、`1d9938e576a8d964e6b977a2b63aa56402840a788eaa6685f26f0d910a7133f2`。三份资产按同一 item ID  join，四 token SID 并非所有 item 一对一。A0/A1 使用同一个已验证用户轴和词表中的 918 个 SID token。原 LC-Rec 六任务的 Instruments 辅助资产及 930 个新增 token 属于另一套输入。Beauty 原 LETTER 使用 965 个注册 token，G3 转换使用自己的字典与对齐，详细配对从上述审计、固定 index 和配置读取。本轮补入 Beauty 三份作者原件，逐文件核对训练计划 SHA-256 和固定 Git blob，核对 22,363 用户的交互涉及物品全部存在于原 LETTER 与 G3 的共同 item 轴，G3 的 12,101 个完整四码 SID 各自唯一。原 LETTER 与 G3 共用同字节 inter/item，采用各自 index。[本轮资产核验](../../lc_rec/evidence/repository-delivery-20261120/retrieved-inputs.json) · [作者固定发布](https://github.com/HonghuiBao2000/LETTER/tree/8d0154e28de37dbb6e24871c508ad8ddb1921cda/data/Beauty)

A0 Instruments 与 Beauty 沿 LoRA 训练，A0/LLaMA base revision `4782ad278652c7c71b72204d462d6d01eaaf7549`，A1/Qwen3-0.6B Base revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`。A0/A1 Instruments 的正式种子均为 42，按最低验证 loss 选 8240。原 LETTER Beauty 和 G3 Beauty 均选 6160，但数据表示和训练拓扑不同。已有 route records 保存解释器、量化或 BF16、ZeRO 阶段、训练 batch、adapter 和真实 argv，入口表之外的数值优先从本行主说明、冻结 config 和 request 读取。

测试期干预只处理 collate 后已识别的每个可见 history item 的一组 4 个完整 token ID。原 V1 S3 固定第四槽，只从 5 种非恒等排列选取，S4 从 23 种中选取，依试验 seed、全局用户行与可见历史出现位置以 SHA-256 拒绝采样，保持 item 时间顺序和真实目标。原生位置与因果上下文因此改变，码位 token 本身保留标识语义。历史正常生成与 Shuffle 同一模型，Prompt0、Beam20/return20、max_new_tokens10，原位置合法 token 回调和 `filter_items=False`。原回调允许位置合法而目录外的组合，统计保留无效候选的排名位置，CCE 显式按 collision group 分配 credit。G3 SID 唯一性和碰撞空集从其专用审计读取，不能推广到其他档案。

## 本轮实际补齐和复核

- 从服务器取回原 LETTER Beauty 和 G3 Beauty 两套各 9 份 stage request 原字节。[清单](../../lc_rec/evidence/repository-delivery-20261120/restored-requests.json)记录原路径、文件 SHA-256、JSON 规范化 SHA-256、当时 stage 状态和版本关系。原 LETTER Beauty 的 `normal`、`normal_eval`、`s3`、`s3_eval`、`s4`、`s4_eval` 六份原件补齐先前仅有账本和 worker argv 的缺口。包括 train、verify、paired 的剩余原件便于和旧记录互校。结果仍采用原先的失败和实际成功尝试。从两个正式 plan 的同一 `runner_sha256` 恢复并保存了 [两套 Beauty 共享的实际 runner 源码](../../lc_rec/evidence/repository-delivery-20261120/execution-source/runner-beauty.py)。
- Instruments A0/A1 的两个 Normal 原逐用户 CSV/NPZ 已按历史 paired 的原始摘要核对并补入 [原件清单](../../lc_rec/evidence/repository-delivery-20261120/normal-output-origin.json)。A0/A1 Shuffle treatment 矩阵、Beauty 三条件合并矩阵、G3 三条件的原 CSV/NPZ 继续各归其原结果入口。四路线的研究数据提供逐用户点估计和配对不确定性复核。
- 用 [复核程序](../../lc_rec/evidence/repository-delivery-20261120/verify_saved_metrics.py) 重算 12 个条件的 16 项点估计、8 个同轴配对的均差，并从已保存的各 5,000×16 抽样均差重新提取线性区间。[实际输出](../../lc_rec/evidence/repository-delivery-20261120/verified-saved-metrics.json)。此过程在 WSL 以 NumPy 1.26.4 读取保存结果，核 row_idx 连续、用户轴摘要、数值和维度，未重新抽样和生成。
- 归档的全候选历史预测用单包的 `compact/letter/` 审查，覆盖 12 个条件、282,810 条预测及 4,524,960 个逐用户指标值。NPZ 可恢复各分片解析后的全部 JSON 结构和值，包括原始解码文字、候选 token、排序分数与输入证据。原 JSONL 文件字节和 gzip 压缩字节的身份见该目录清单。原预测复核的代码与结果继续提供历史核对。[原预测复核程序](../../lc_rec/evidence/repository-delivery-20261120/verify_raw_predictions.py) · [原复核输出](../../lc_rec/evidence/repository-delivery-20261120/verified-raw-predictions.json)

## 从已保存输出复算

从现行 Release 下载单文件审核包 `GRID-Repository-Closeout-v1.5-20261121.zip`，核对同处的包外 SHA-256。旧公开审核卷已撤下，上一轮已清理本地 05/06/07 卷文件。原实验 JSONL 保存在各阶段服务器路径。需要原序列化文件时，从包内 `compact/letter/manifest.json` 查到条件、rank、`server_path` 和原字节 `sha256`，由有权限的账户只读取回并按原摘要核对。包内直接复算使用 NPZ。[现行下载入口](https://github.com/woobowen/recommend/releases/tag/grid-repository-closeout-2026-11-21)

在已下载 ZIP 的目录执行以下命令。使用 Python 3.12.3 与 NumPy 1.26.4。工作目录和结果文件由 `mktemp` 单独创建。

```bash
work=$(mktemp -d)
unzip -q GRID-Repository-Closeout-v1.5-20261121.zip -d "$work/input"
cd "$work/input"
python3 -B tools/restore_aliases.py . repository/wbw-lc-rec/
python3 -B tools/restore_aliases.py . compact/letter/
python3 -B compact/letter/audit_compact_predictions.py \
  --repo repository/wbw-lc-rec --members compact/letter > "$work/letter-audit.json"
```

`letter-audit.json` 的 `status=PASS` 表示 12 条件的所有 16 项逐用户值、候选诊断、固定 V1 置换和 8 组相对 Normal 的配对在该范围内与冻结结果相符。审查某个具体用户的原候选、模型输入和顺序可直接用包内 `compact/letter/README.md` 的逐行示例。该复核从已保存输出开始。

## 原 LC-Rec 六任务定向收尾

2026-11-21 从原服务器目录核对作者 Instruments ZIP 和六个输入文件的 SHA-256，并从原始 interaction 和辅助数据逐用户核对两种历史配对。ItemSearch 直接使用 `user_vague_intention.train` 的 `item`、`inters`、`querys` 和相应 explicit preference。训练辅助 `item/inters` 与 `inter[-3]/inter[:-3]` 不配对 12,233／24,772，测试与 `inter[-1]/inter[:-1]` 不配对 7,473／24,772。训练辅助 target 有 3,608 条等于当前 validation target、1,623 条等于当前 test target，实际最近 20 个可见历史中 1,623 条出现当前 validation 物品。PreferenceObtain 的历史来自当前 `inter[:-3]`，监督文本取作者 `user_explicit_preference`。24,772 人的偏好条目存在，原提示、评论上下文、生成时间与取数分割尚需作者证明。观察只涉及给定数据配对，论文历史采用何种生成上下文仍未知。

`item2id`、metadata、作者 LC-Rec `index` 及 interaction 的物品集合一致，覆盖 9,922 个 item 和 9,897 个 canonical SID，其中 23 个碰撞组。与 LETTER 同域的 `inter`／`item` 原字节相同，SID `index` 不同。原件和来源核对、全 24,772 人的内部数值 ID 投影、可独立运行的配对程序放在本轮证据目录。包内 `compact/workload/` 是 221,225 条声明字段的全行投影。逐样本随机选择和两个文本摘要可查冻结 `records.jsonl.gz`；实际文本及掩码沿保存的诊断样本、作者辅助原件核对，覆盖范围以各原记录为准。 [原资产、配对结果与证据](../../lc_rec/evidence/repository-final-closeout-20261121/SIX_TASK_HISTORY.md)

原 full-parameter ZeRO-3 双卡无 offload 的两组 FP32 Adam moments 下限为每卡 25.1309 GiB，高于实际每卡 23.9883 GiB。该数字是由参数数目与两卡分片推导的静态下限。后续变更执行方式的 offload 和 LoRA 仅完成各自有界工程预跑。LoRA 的双卡 micro2、累积 32 的有界 path 经 GPU 计算与更新验证，micro4/micro8 长尾 OOM 记录完整保留；可信正式六任务训练仍需与当前 split 同源的逐用户辅助原件、辅助生成上下文证明以及经批准并核查的资源方案。 [原六任务技术审核](../../lc_rec/six_task_lora/REVIEW_ENTRY.md)

六任务混合组 CE 权重的 14 个末位差归因于 Python 浮点 `sum` 实现的版本差异。原 Python 3.10.22 与本地 3.11.14 的内置 `sum` 逐字段与历史原值相同，Python 3.12.3 与上一轮整理值相同。两端使用完全相同的已保存浮点权重，未加载 NumPy；旧报告中的 NumPy 版本是环境记录。最大绝对差为 `2.220446049250313e-16`，两个值和原始报告均保留。 [因果核对、原值及命令](../../lc_rec/evidence/repository-final-closeout-20261121/CE_SUM_CORRECTION.md)

## 四种复现起点

| 起点 | 本分支的起始材料及本次验证 |
|---|---|
| 保存输出 | 上表原结果、矩阵、抽样与 CPU 复核入口可核点估计和区间。逐候选检查从单包 `compact/letter/` 以冻结数据和报告复核，G3 的历史主作业从原预测复算，另有独立非作者读取原件复核。 |
| 固定 checkpoint 评价 | 按每个正式 RUN 的 manifest，模型取绑定的 Hugging Face revision 与许可范围，再由获准访问 `new-server` 的账号取评估输入和最佳 adapter。Instruments 冻结数据在 `lc_rec/letter/vendor/LETTER/data/Instruments/`，各阶段 source-root、python、data-root 和实际 `argv` 在该工序 request。Beauty 取 [固定作者资产](../../lc_rec/evidence/repository-delivery-20261120/letter-beauty-author/Beauty/)；G3 用同样的 inter/item 与 [转换 index](../../lc_rec/a0_beauty/grid_sid_preparation/g3_conversion/Beauty.g3.index.json)。没有独立服务器访问的复核者可使用审核包内读取所需的结果材料；重新生成仍需权重与环境。 |
| 重跑约定实验 | 取得固定 source export、工作资产、环境及阶段 plan，按各实际实验从训练、best adapter 复核到生成、评价、配对逐步使用保存脚本。Forensics 复核和预跑的 restore probes 与真实训练是不同任务。当前历史六任务辅助配对失败，原始全参范围未达到准备门槛；不以公开论文候选填实际运行记录。 |
| 重建上游资产 | Instruments 原 LC-Rec 与 LETTER 发布资产来自固定作者版本，缺少可唯一绑定历史 embedding/量化和若干生成链，正式结论限定于给定发布资产。G3 有源上游 Qwen → RQ-KMeans 的身份和转换审计，但本轮未重建 embedding、SID、量化或任一训练。 |

实验前用固定计划、input 配对和当时适用的预审，运行中沿相同 job/attempt 读取观测，实验后沿完整范围、先失败再补算的关系和结果核对。不完整的历史证据沿原审计保留。此次整理的 Git 版本用于定位新增材料，正式 RUN commit 和服务器 export 保持各自历史身份。
