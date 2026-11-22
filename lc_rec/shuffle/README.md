# LETTER 下游推荐的 SID Shuffle

## 本轮实验使用 GRID-LETTER-SHUFFLE-V2

用户在 2026-11-20 批准本轮任务，并将发布分支指定为 `wbw-lc-rec-shuffle-v2`，从 `7368d301d0d928f77d50551c3b53a5b30c23d63f` 建立。正式运行范围是 Instruments LETTER SID、既有 A0 LLaMA-7B adapter，依次执行 `shuffle2`、`shuffle3a`、`shuffle3b`、`shuffle4`。Normal 使用 `4c6a83d9ba42dc2fe737792759e410ef1874da76` 的冻结预测。其余三组依次为原 LETTER Beauty、GRID G3 Beauty、LETTER Instruments A1，须等第一组交付与网页二审后另行授权运行。

### 置换与统计口径

| 模式 | 活动位置（0-based） | 固定位置（0-based） | 排列数 | 理论恒等概率 |
|---|---|---|---:|---:|
| `shuffle2` | 2, 3 | 0, 1 | 2 | 1/2 |
| `shuffle3a` | 0, 1, 2 | 3 | 6 | 1/6 |
| `shuffle3b` | 1, 2, 3 | 0 | 6 | 1/6 |
| `shuffle4` | 0, 1, 2, 3 | 无 | 24 | 1/24 |

`identity_policy=allow_identity`。种子 42，紧凑 UTF-8 JSON `[protocol,42,m,row_idx,visible_item_idx,counter]`，`ensure_ascii=True, sort_keys=True, separators=(',', ':'), allow_nan=False`。SHA-256 完整摘要按无符号大端解释，以 `floor(2**256/m!)*m!` 为接收上界，按余数索引字典序全排列。计数器起始为 0。目标槽取值规则为 `new[A[j]] = old[A[perm[j]]]`。S3a 和 S3b 共享同一相对排列。只在数据集、用户轴与历史出现轴对应时，跨模型或上游共享相对排列。不同 `m` 的键不同，四种模式之间没有嵌套约束。

全排列把自然恒等保留在数据中。对每次出现，移动槽数的理论期望为 `m−1`。即使完整 token ID 及对应 embedding 保持身份，所在序列位置、相对顺序、因果可见上下文和位置编码的配对也可能变化。S3a/S3b 的差值描述两种位置干预，无法单独识别一个 SID 层级的因果作用。去重位的解释只适用于 G3 对应方案。指标相对于 Normal 可能上升或下降，S4 和 S3 间没有预设顺序。

冻结的 [评价协议](../letter/EVALUATION_PROTOCOL_V1.md) 保留 SID Hit、NDCG、MRR 和碰撞 CCE 的 ItemHit、ItemNDCG 共 16 项。主比较为四种模式各减 Normal，预声明次要比较为 S3b−S3a。用户有放回配对 bootstrap 5,000 次，PCG64，统计种子 20261113，95% 线性百分位区间。区间逐指标报告，分析条件为冻结模型、用户轴和单个固定 Shuffle 实例。

### 实际接线与边界

| 来源 | 冻结原入口与 V2 接线 | 模型边界与评价 |
|---|---|---|
| Instruments A0 | `letter/scripts/formal_evaluate.py` → `shuffle_v2.entry` 复用字节固定的派生锚点 → `Runtime.intervene` → `shuffle_v2.core.transform` | 原 `TestCollator`，`seq[:-1][-20:]`，原逐位回调，单 rank BF16 Base+LoRA。调用原 `generate`，实际 decoder prefill 对 token、mask、当前 position_ids 与空初始 cache 逐项检查。独立 CPU 评价读取保存分片。 |
| Instruments A1 | `a1_qwen3_06b/scripts/evaluate.py` → 同一派生入口 → `Runtime.intervene` → V2 core | 原 Qwen Base、tied embedding/head 严格加载，并由 `Runtime.allowed` 先验证冻结回调对原 Prompt 的绑定，再绑定实际置换前缀。仅静态接线核验。 |
| 原 LETTER Beauty | `6e1b211d…` 下冻结 `a0_beauty/formal_evaluate.py` → `shuffle_v2.beauty --variant letter` 使用同版本 `shuffle_entry.py` 派生锚点 → `Runtime.intervene` → V2 core | 原独立 sampler 保留 11,182/11,181 两分片及原回调。仅静态接线核验，后续须单独绑定该数据域 tokenizer 与 CPU 审计。 |
| G3 Beauty | 原 G3 formal_evaluate.py 的 generate_one 经 G3 V2 入口绑定本组 CPU oracle，再调用 V2 core | 单卡固定 FORMAL adapter，PREFLIGHT 范围执行四模式并绑定实际 embedding 和 decoder prefill。 [G3 第三组正式前准备](results/v2-a0-beauty-g3/README.md) |

[V2 固定资产和预算](../shuffle_v2/a0_instruments.json) 绑定既有 Base、step8240 adapter、SID、交互、tokenizer 与 Normal 预测。原 A0 生成配置为 Prompt0、每 rank batch=1、beam=return=20、`max_new_tokens=10`、`filter_items=False`。用户随后将第一组资源修订为单张可用 RTX 4090，`WORLD_SIZE=1`、`rank=local_rank=0`、batch=1。单 rank 处理完整冻结 `DistributedSampler` 随机顺序，即历史两 rank 的交织。全局 `row_idx` 和置换保持一致，历史 Normal 仍读取原双 rank 预测，单卡与双卡的生成数值等价性未预设。预跑保留原 16+16 名单轴合并名单，单次正式作业最多 8 小时，正式累计不超过 36 小时墙钟和 36 GPU-device-hours，预跑累计不超过 60 GPU-device-minutes。装载保持 PEFT 0.5.0 原路径并逐值核验全部 450 个 adapter tensor，包括 embedding/lm_head，未新增参数。正式前门槛依次为全量 CPU 重放、32 名固定真实用户的四个 V2 模型边界预跑、独立预审。预跑只运行上述四个模式。每模式独立 job，顺序由 [控制脚本](../shuffle_v2/control.py) 校验，生成、CPU 评价、配对统计分别保留终态。正常监控间隔为 15 分钟。到达第一整组交付后停止。


### 第一组单卡启动与运行观察

[已发布的 30 分钟观察回执](results/v2-a0-instruments/launch-observation-20261120/README.md)记录本轮正式运行身份、单卡配置、服务器独立调度、预审及七次观察。各条件的正式全量结果和整组审核另在计算完成后交付。

### 第一组 CPU 准备证据

[回放、Normal 重评与作业回执](results/v2-a0-instruments/)保存本轮实测，并附[非作者复核](results/v2-a0-instruments/nonauthor-cpu.md)。每个新模式覆盖 24,772 名用户、172,929 个历史组。校验固定位置、组外 token、mask 和长度，违例数均为 0。两份 Normal 原预测以原评价口径只读重算，逐用户矩阵与历史交付字节一致。

| 模式 | 恒等排列数 | 实际未变组 | 整条输入未变的用户 | 四个位置的 token 移动次数 |
|---|---:|---:|---:|---|
| shuffle2 | 86,483 | 86,483 | 714 | 0 / 0 / 86,446 / 86,446 |
| shuffle3a | 28,935 | 28,935 | 6 | 115,138 / 114,894 / 115,397 / 0 |
| shuffle3b | 28,935 | 28,935 | 6 | 0 / 115,138 / 114,894 / 115,397 |
| shuffle4 | 7,147 | 7,147 | 0 | 129,629 / 129,582 / 130,026 / 129,989 |

四种条件的 CPU token 流与非作者独立重建 SHA-256 一致。真实 Instruments 历史中未发现重复出现的同一商品。补充 CPU 夹具使用真实 tokenizer 和 Prompt0 构造同一历史 SID 重复、左 padding 的输入，核验按出现位置独立索引。独立合成小模型仅验证当前依赖下 hook 接线；A0 真实模型预跑是单独的正式前门槛。

### 历史 V1

V1 的排除恒等注记发布在旧历史线文档提交 `94b79854a59f52208bc11f14fc6e3069594dab6a`。Instruments V1 交付 `3396d45a041424fe4af625bfc0b047467150c5c1` 由注释标签 `letter-shuffle-v1-exclude-identity` 归档。原 LETTER Beauty `6e1b211d9c37f358c43b1540b40b17f6703150d2`、G3 Beauty `21d495b0be5b606853ec7cec68a3eaa1b58acbd7` 和 Instruments `f0242b9f90ae23a4979d688a287734ca9f88cea1` 的 V1 core 字节相同。旧 S3 与新 S3a 位置相同，旧 S3/S4 分别从 5/23 个非恒等排列抽样。旧审核按原时间语义阅读。

## 2026-11-14 历史 Instruments V1 实验

四组正式全量测试及真实非作者方法/统计终审均已通过。请从[最终审核入口](FINAL_REVIEW_ENTRY.md)开始；[完整结果表](results/RESULTS.md)含全部16项、差值/相对差/95%区间、SID诊断、first-hit、分组、推荐变化及成本。GPU/CPU实际RUN均固定为`f0242b9f90ae23a4979d688a287734ca9f88cea1`，后续交付文件没有改变服务器冻结源码。

用户于2026-11-14明确授权的一次性任务。仅在冻结 Normal 模型测试输入中，完整 Prompt 经原 tokenizer 后，对每个可见历史物品的四个 SID Token ID 做组内置换。Token 身份不变；不改训练、目标生成顺序、Beam20、SID、数据或评价协议。

正式顺序：A0-S3 → A0-S4 → A1-S3 → A1-S4，双 rank 严格串行。各条件24,772用户、20候选。A1须先完成原授权最终非作者审核。正式 GPU 门槛：全用户 CPU 输入校验、32真实用户 OFF/ON GPU 预验收、非作者预审均通过。

随机协议 `GRID-LETTER-SHUFFLE-V1`，seed42；compact JSON UTF-8 `[protocol,42,n,row_idx,visible_item_idx,counter]` 的 SHA256 按 unsigned big-endian 解释，拒绝采样避免模偏，索引字典序排除恒等的5/23种排列。方向 `output[j]=original[perm[j]]`。同用户/历史出现位置两模型共用排列。每条件仅一个固定随机实现。

预验收最多60 GPU-device-minutes，含加载、失败及清收。正式各A0最多4h、各A1最多3h，正式累计启动/运行/清收最多18h。任务 round0；随后最多三轮集中修正。原始失败保留。模型加载和生成只用服务器既有环境，无重新训练授权。

代码复用固定A0/A1真实原推理入口。`entry.py`先验证原字节SHA，做唯一锚点最小派生补丁，执行前保存实际派生源码和diff。`runtime.py`仅在传入原生generate之前处理当前历史Token IDs；A1保留原Allowed对原始Prompt及Response边界的验证，再验证置换并绑定当前前缀。正式输入不显式传入旧坐标或旧cache。原Token可能隐式暴露SID层级，不声称层级匿名化或严格置换不变性。

`cpu_inputs.py`全轴校验、排列manifest和固定分层样本；`test_core.py`是作者定向测试，不能替代非作者审核。统计复用冻结16项/CCE/分组/5000次PCG64用户Bootstrap定义。Bootstrap只刻画固定模型、固定置换实现下用户不确定性。

全用户CPU校验、每模型32真实用户GPU OFF/ON验收和非作者技术/统计预审通过。预跑含round0失败及runner清收累计7.8545052 GPU-device-minutes。三轮任务修正全部使用，四组正式GPU均首试成功。A1 Normal最终科研非作者审核已于正式启动前通过并发布至RUN固定提交，见[A1入口](../a1_qwen3_06b/A1_NORMAL_FINAL_REVIEW_ENTRY.md)。四组GPU阶段合计25222.072秒，四runner实际墙钟合计25638.797秒；所有终态/清收及96个export文件最终字节复核通过。最终发布回读和ZIP身份属于本提交冻结后形成的外部交付回执，随实际交付提供。

[验收回执](evidence/preflight_final_receipt.json)、[统计预审](review/STATISTICS_PRELAUNCH.md)、[修正记录](review/CORRECTIONS.md)。`launch.py prepare`只构造固定request，`launch.py submit`提交服务器自主管理的串行作业；已有job只查原身份，失败门槛停止后续依赖链。每条件先生成、清收，再进行独立CPU输入回放/评价，不重新生成已成功预测来修复CPU问题。
