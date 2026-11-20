# Instruments A0 V2 单卡启动与 30 分钟观察

**RUNNING_AFTER_30_MIN_OBSERVATION**。截至 2026-11-20T12:38:40.835667-05:00，第一个正式条件 S2 达到 4,525/24,772 名用户，父作业与原 GPU 子作业仍在运行。其余三个条件已在服务器串行计划内。整组结果与终审等待正式作业完成。

## 七次现场观察

以下时间为 2026-11-20 美东标准时间。七次记录均为原 S2 job、独立 runner 证实的同一 GPU worker PID 3367671、物理卡 1、逻辑卡 0、`WORLD_SIZE=1`。GPU 0 上另有外部进程。预测文件和日志每个区间均继续写入，采样日志未见错误、OOM、异常退出、重试或重复作业。

| 时点 | 时间 EST | 已处理用户 | 区间增量 | GPU worker 显存 MiB | GPU 利用率 | 预测文件 MiB | 日志 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| [T00](T00.json) | 12:08:40 | 1,675 | — | 19,876 | 41% | 25.9 | 进度刷新，未见异常 |
| [T05](T05.json) | 12:13:41 | 2,150 | 475 | 19,876 | 100% | 33.3 | 进度刷新，未见异常 |
| [T10](T10.json) | 12:18:40 | 2,625 | 475 | 19,876 | 41% | 40.8 | 进度刷新，未见异常 |
| [T15](T15.json) | 12:23:41 | 3,100 | 475 | 19,876 | 52% | 48.2 | 进度刷新，未见异常 |
| [T20](T20.json) | 12:28:40 | 3,575 | 475 | 19,876 | 74% | 55.6 | 进度刷新，未见异常 |
| [T25](T25.json) | 12:33:41 | 4,050 | 475 | 19,876 | 100% | 62.9 | 进度刷新，未见异常 |
| [T30](T30.json) | 12:38:40 | 4,525 | 475 | 19,876 | 41% | 70.4 | 进度刷新，未见异常 |

观察期存储最低余量为 2.46 TiB。用户计数来自每 25 人写出的 rank 状态，表中利用率是瞬时采样。

## 进度与预算

当前 S2 已使用 49.9 分钟墙钟，模型装载及严格权重核验用时 117.3 秒，最后一个进度回执中的生成用时 47.7 分钟。还有 20,247 人。最近 10 分钟的有效吞吐为 1.58 人/秒。S2 预计剩余 3.55 小时，按最近四个 5 分钟观测窗口估计约 3.55–3.56 小时。

| 待执行条件 | 估计耗时（含装载） |
| --- | ---: |
| shuffle3a | 4.4–5.4 小时 |
| shuffle3b | 3.8–4.7 小时 |
| shuffle4 | 4.3–5.3 小时 |

GPU 阶段预计剩余 16.1–18.9 小时。两端来自两种外推口径。其一用 S2 最近 10 分钟的实测吞吐按各模式预跑相对速度调整，其二直接用各模式 32 人预跑吞吐外推。历史长度、条件差异和共享服务器资源波动会影响实际时间。

Normal 只读重评用时 33.58 秒，单分片评价回归用时 45.58 秒，旧 A0 的 CPU 评价曾约 102–106 秒。以此作为分钟级参考，每个新条件的 CPU 全量输入检查和指标估计用时约 1–3 分钟。五项新配对的实际总时长待执行观测，当前请求限制为 900 秒。收集、终审和交付留到作业终态，时长待实测。

预跑用去 10.56/60 GPU-device-minutes，剩余 49.44。正式用去 0.832/36 小时，累计剩余 35.168 小时，单次 8 小时上限剩余 7.168 小时。累计 CPU 科研作业用去 397.3/14,400 秒，剩余 14002.7 秒。以上是预算账目。[机器可读汇总](observation-summary.json)。

## 实际研究实现

本轮使用 Instruments 的原 LETTER SID（上游 `HonghuiBao2000/LETTER@8d0154e28de37dbb6e24871c508ad8ddb1921cda`）和 A0 的 `huggyllama/llama-7b@4782ad278652c7c71b72204d462d6d01eaaf7549`。未量化 BF16 Base 配合既有 step8240 LoRA，装载时核对 450 个 adapter tensor 的键集合、形状、有限值和显式类型转换后的精确值，包括完整 embedding 和 lm_head。服务器使用 Python 3.10.22、Torch 2.6.0、Transformers 4.31.0、PEFT 0.5.0、Accelerate 0.21.0。实际装载配置与生成配置保存在[真实预跑 rank0 报告](gpu-preflight-r0/shuffle2/rank0.json)和[正式运行首行](formal-shuffle2-first-row.json)。

| 对象 | 固定身份 |
| --- | --- |
| A0 adapter SHA-256 | `f4651c5c6e6ed841cd3384c0c8ebfab30880e94597e515ffa21916b876001ed5` |
| SID index 文件 SHA-256 | `33c87e28c52ae367985bdf494cb49d676d181e13a39497b4ad035f46111546dd` |
| 交互文件 SHA-256 | `0a11d47791d90483005da93c36d826679e20e4592d1d2b86518d5f56f218e743` |
| 物品文件 SHA-256 | `1d9938e576a8d964e6b977a2b63aa56402840a788eaa6685f26f0d910a7133f2` |
| 用户轴 semantic hash | `1daf63ce4c77b0265777a46b61939006aafab0276b9106c742de602c8c7e867a` |
| tokenizer vocab semantic hash | `7a2f1620a30b53a21addbcf9a0d8827c66fc91e4820cd9679ccc387ea106f22b` |
| 冻结 A0 推理入口 SHA-256 | `47aa87e2d4b1073e076a3d76ef8af07682cb45fc9dd8c3fb09113e1bb4d000d7` |
| Normal GPU / CPU / 交付提交 | `4c6a83d9ba42dc2fe737792759e410ef1874da76` / `3d11258daa6aa4fb2bed0cd013bca50536fe351d` / `55a1371da28a45a2ec81ceee7bd82fd42b7192d5` |
| Normal 两原预测分片 SHA-256 | rank0 `bdcbdde15d3ac6850708888e055aa67ae61212dbb675d9f915b9947b2a208bde`，rank1 `f5d2ce538bfe86b024c38070f74cce3411a457554ae08ec610ec26987570c629` |

词表和用户轴采用冻结口径的有序 JSON、UTF-8 和紧凑分隔语法复算。两份 Normal 原预测各含 12,386 用户，按全局 `row_idx` 匹配用户、目标和原始输入。只读 CPU 重评的 24,772×16 NPZ 与冻结结果逐字节相同。[资产字节核验](../cpu-replay-r1/assets.json)、[Normal 原预测 manifest](../cpu-eval-normal-r0/prediction_manifest.json)、[Normal 重评](../cpu-eval-normal-r0/summary.json)。

| 模式 | 固定位置（0-based） | 活动位置（0-based） | 排列数 | 理论恒等概率 |
| --- | --- | --- | ---: | ---: |
| shuffle2 | `0,1` | `2,3` | 2 | 1/2 |
| shuffle3a | `3` | `0,1,2` | 6 | 1/6 |
| shuffle3b | `0` | `1,2,3` | 6 | 1/6 |
| shuffle4 | 无 | `0,1,2,3` | 24 | 1/24 |

V2 标记为 `GRID-LETTER-SHUFFLE-V2` 和 `identity_policy=allow_identity`。核心使用 `itertools.permutations(range(m))` 的字典序完整目录。每个历史商品出现的抽样键为 UTF-8 紧凑 JSON `[protocol,42,m,row_idx,visible_item_idx,counter]`，`ensure_ascii=True`、`sort_keys=True`、`allow_nan=False`、分隔符 `(',', ':')`。计数器从 0 开始，完整 SHA-256 以无符号大端解释，仅当数值达到 `floor(2^256/m!)*m!` 时重算，接受后取模。模式和活动位置不加入键，两个三位模式用同一个相对排列。目标槽映射为 `new[A[j]] = old[A[perm[j]]]`。所有自然恒等组和用户继续走原模型生成路径。V1 固定历史的 5/23 个排列排除恒等，相关代码与回执保留在 `letter-shuffle-v1-exclude-identity` 及历史说明提交 `94b79854a59f52208bc11f14fc6e3069594dab6a`。

`shuffle_v2.entry` 验证冻结 `letter/scripts/formal_evaluate.py` 的字节哈希，并保存实际派生入口及补丁。原 Prompt0 与 `TestCollator` 完成编码后，`Runtime.intervene` 从实际可见历史 `seq[:-1][-20:]` 逐商品核对四个连续 token 的身份、边界和顺序。它在 `model.generate` 前改变组内完整 token ID 的槽位。目标为 `seq[-1]`，完全排除在历史之外。固定槽、组外 Prompt、长度和 mask 得到保留。真实预跑观测 embedding 输出及首个 decoder prefill，确认所有 beam 收到置换后的 ID、相同 mask、原生位置编排和空初始 KV cache。正式路径为每行检查 CPU manifest 与第一次 prefill 的哈希，证据随原始预测保留。更换 token 的序列位置会改变原生位置编码配对和因果上下文，这是干预的含义。[真实模型预跑与非作者检查](preflight-review.json)、[S2 首评分行及 CPU 对照](formal-shuffle2-first-row.json)。

用户将本轮新实验修订为单 GPU，`WORLD_SIZE=1`、`rank=local_rank=0`、batch 1。单 rank 的完整 `DistributedSampler` 使用原 `shuffle=True, seed=0, epoch=0, drop_last=True`，输出恰好交织原两分片的顺序，保留全局行号，不重定义抽样键。CPU 已对 24,772 人、172,929 组逐位置独立复算并证明一致。预跑沿用原 16+16 的 32 人名单；所有模式分别运行，未做新 Normal、OFF 或 original prefill/generate。自然恒等的 10 个模式–用户对与保存的双 rank Normal 在样本内候选、排序及原分数相同。本轮未据此推定单／双 rank 全量数值等价。[单分片用户轴审计](cpu-single-shard-r1/audit.json)、[只读 Normal 的单分片评价回归](cpu-evaluation-fixture-r0/summary.json)。

实际 `generate` 使用 Prompt0、beam=return=20、max_new_tokens=10、`do_sample=False`、`early_stopping=True`、`use_cache=True`、EOS=2、pad=0、length_penalty=1.0，沿用原逐码位合法 token 回调及 sequence score 降序稳定排序。`filter_items=False`，候选重复和目录外完整 SID 保留，合法码位与目录成员分别核验。对新单 rank 输出的三个冻结评价器仅派生了分片读取和计数门槛，数值公式与 CCE 未变。Normal 仍由原两分片读入。指标为 SID Hit@1/5/10/20、SID NDCG@5/10/20、SID MRR@5/10/20、ItemHit@5/10/20、ItemNDCG@5/10/20。碰撞展开保留冻结 CCE 的均匀 credit 和原无效及重复候选规则。统计比较四模式各减 Normal 及 S3b−S3a。五比较共享冻结用户轴上的 PCG64 配对有放回抽样，统计 seed=20261113、5,000 次、95% 线性百分位区间，逐指标报告。

## 已通过的前置检查

- V1/V2 定向单元测试、四条派生入口编译和原生成调用 AST 比对。原 LETTER Beauty、G3 Beauty、A1 仅做来源核验和静态接线，未运行其 GPU 条件。
- CPU 采用真实 tokenizer/collator 对四模式全量重放，24,772 用户、9,922 物品、9,897 个 SID、918 个新增 token、每模式 172,929 个历史组。固定槽、组外变化及 mask/长度违例均为 0。四个完整输入流 SHA-256 与非作者独立重建相同。S2/S3a/S3b/S4 的恒等组 86,483/28,935/28,935/7,147，整条输入自然不变的用户 714/6/6/0。实测严格检查见[全量 CPU 摘要](../cpu-replay-r1/audit.json)。
- `/root/history_review` 直接复核四个模式真实 A0 预跑的 128 行、972 组、2,560 个候选，各行 prefill IDs、mask、原生位置、起始 cache 长度及真实 embedding。`/root/runner_review` 独立复核单分片评价路线、固定 Normal、样本统计约定及串行运行控制。预跑原 job 退出 0，服务器回执确认全部清收。真实预跑单设备累计 10.5626 分钟。[独立预审](preflight-review.json)、[CPU 非作者复核](../nonauthor-cpu.md)。
- 早期 CPU 重放 r0 在主程序退出后由 runner 回收残留子进程，保留该终态。补齐冻结环境中的 `TORCHINDUCTOR_COMPILE_THREADS=1` 后，r1 退出 0 且 clean。单分片 CPU 审计以已核验的 r1 为依据。未更换资产、降低精度、改 beam、缩短历史或重新选权重。未安装依赖。

## 关闭 Codex 后的执行

实际服务器是 `bwwu@zwlab-01`，runner 为 `/home/bwwu/grid-audits/sid-extensions-p0/scripts/runner-a1-96h-1fefc09a15c1cb73.py`。从新的 SSH 连接读取同一原 ID，series 及当前 GPU 子作业的 supervisor 由 PID 1 承接，进程在自己的 session，没有 TTY，stdin 为 `/dev/null`。服务器持有独立源码 export、解释器、输入、权重、预跑回执、正式计划和全部后续配置。[脱离终端的实际进程回执](autonomy.json)、[服务器串行计划](series-plan.json)。

父作业 `letter-a0-20261112-v2-20261120-group-series-r0` 管理 S2 → S3a → S3b → S4。生成及对应的 CPU 全量验收和评价均成功后才推进，最终提交 CPU 配对 bootstrap。子作业使用固定 ID、请求指纹、独立输出及清收门槛。每个 GPU 请求 28,770 秒，加上运行和清收预留后受 8 小时单次上限约束；运行时累计核算至 36 小时墙钟和单设备 36 小时。依赖失败会停止后续提交，保留原作业和日志。可关闭本地 Codex、SSH、WSL 和 VPN。故障分析或修复、最终非作者终审、证据收集、GitHub 固定交付和最终审核 ZIP 需要重新打开 Codex 接续。返回后不继续发起主动轮询或通知。

- 研究 RUN `e2f54ebd2d7709821de1071693f91644f5ed9a30`，[服务器部署清单](e2f54ebd2d7709821de1071693f91644f5ed9a30.deployment.json)。
- 串行控制快照 `4f5758ffb578f3a8d3d06a54262f59d3f63db370`，[控制器部署清单](4f5758ffb578f3a8d3d06a54262f59d3f63db370.deployment.json)。
- 作业状态与退出回执在 `/home/bwwu/grid-audits/sid-extensions-p0/jobs/<job_id>/`，含 `request.json`、`accepted.json`、`status.json`、`work.log`，终态附 `cleanup.json`。S2 实际生成日志在 `/data/bwwu/lc-rec/letter-shuffle-v2-20261120/gpu-formal-shuffle2-r0/shuffle2.log`，实时 rank 回执和原始生成在该目录的 `test/`。
- 接续时从 `/home/addaswsw/recommend-worktrees/shuffle-v2` 执行 `python3 -B -m lc_rec.shuffle_v2.control query --job group-series` 查询父作业和原任务身份。需要单个子作业时查询原子 ID，例如 `python3 -B -m lc_rec.shuffle_v2.control query --job gpu-formal-shuffle2`。避免重复提交。
