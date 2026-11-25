# 独立非作者审核

审核者为本轮独立子代理 `/root/review`。运行只涉及原件和阶段产物的本地或服务器只读核查。审核者未编写流水线代码，未提交计算作业，未执行 Git 写入。

## 原件与共同轴

- 实际读取规则入口 v1.5 的项目规则第 3、5、8、9 节和远端执行规范的 runner、GPU、记录与证据要求。固定 C 从 `6be303ef7e05968c6a41400702732d954ea1eea1` 的独立工作树读取。B 和 D 从给定的 Git object 读取实际配置、入口和依赖。
- 对固定 A 的本地原件独立计数。用户 24,772，商品 9,922，交互 206,153，用户及商品最小度均为 5，重复 user–item 为 0。按固定切分，训练保留 156,609 条交互，验证可见历史 148,927 项，测试可见历史 172,929 项。每个验证和测试目标均来自原序列的指定位置。
- 在 `new-server` 从既有 LC-Rec 作者 ZIP 直接读 `Instruments.item2id`，其同一目录的交互与商品元数据均字节等于本轮固定 A。由 ASIN 独立重建 `LETTER ID → GRID ID → embedding 行`，核对两端一一对应。逐行将服务器本轮向量与原 B 向量的连续 FP32 字节比较，分别读出 `[9922,3584]` 与 `[10611,3584]`，全部有限，9,922 行全等且各行不重复使用，实际编码文本全部相同。经 ASIN 反查回原编号时 24,772 条完整用户序列逐条精确一致。
- 实际读取 B 的 `assets.json`、`upstream_metadata/qwen_config.json`、`upstream_metadata/source/qwen_worker.py` 和向量元数据。B 向量 SHA-256 为 `65adf94502f35b93fb48964d022f84f2afb7f512282946092ad5202d11baa165`。原文本由 Qwen2.5-7B-Instruct 批处理，批内样本独立，右 padding 128，WMH MeanAggregation 使用每行 attention mask，float16 模型，保存 float32。原权重分片的唯一 HF revision 缺证。图谱按 ASIN 配对，不需要靠标题证明同一商品。

## V2 与位置边界

- 对 `a0_instruments.json`、`replay.py` 及实际 A0 历史 `/data/bwwu/lc-rec/letter-shuffle-v2-20261120/cpu-replay-r1/inputs.jsonl` 独立逐行核查。原 A insertion-order 用户键等于 0 到 24,771 的数字轴，规范用户轴 SHA-256 为 `1daf63ce4c77b0265777a46b61939006aafab0276b9106c742de602c8c7e867a`。原 A0 Replay SHA-256 为 `3acc7910efbbdc86b2676d0c7f383272bd062bbb12a207f09f5b8a73cd00238d`。与 GRID 原有同一 `sequence[:-1][-20:]` 可见商品位置。
- A1 历史 replay SHA-256 为 `93e7a63e53715e639b7d6aea088b4d68d1309658e7a9a8b04cddfbbc0d99d31c`。A1 冻结交互原件与 A 同字节。全量逐用户核对 row_idx、user_id、target_item_id、visible_history_length。直接用本轮 `shuffle.py` SHA-256 `83c2e730af5f176644f082d0f77c0910d94cb3451206859a99a9d421a65a42dc` 的抽样接口比较 A1 参考原始排列编号，345,858 条模式×历史位置全部相同。S3a 6 种组合、28,935 次恒等。S4 24 种组合、7,147 次恒等。强制拒绝采样以及相等裸码、不同层级和不完整历史组边界的 CPU fixture 已执行。
- C 原生 `SemanticIDEncoderDecoder.encoder_forward_pass` 的 sequence_data 经原层级偏移后进入共享 1,024 词嵌入表，随后才插入 SEP。BOS 在原 decoder 路径。真实 mask 将非历史位置置为零，并屏蔽 canonical target。本轮 `EncoderV2` 在共享表的 encoder 调用范围内注册局部 prehook，以原 offset lookup 的完整 token 为单位收集，退出作用域即清理。它保留原位置、SEP、BOS 和相对位置注意力行为。C 的 teacher forcing 与 generate 各一次 encoder，当前实现的候选采集需两次输入计划严格一致。实际 GPU 证据待原件回读。

## 量化、下游及阶段恢复

- B 有效上游为 `rkmeans_train_flat` 与 `rkmeans_inference_flat`。原始训练步上限 30，实际使用三层、每层 256、逐层残差归一化、训练源 batch 2,048、buffer 3,072、KMeans++ 初始化、SGD 0.5。一层初次 buffering 和初始化 raw SGD 均发生在 Lightning 计步之前。对 `deduplicate_rows_in_tensor` 实现正文独立复核发现其说明与实现有差异，代码对多件碰撞组使用 1 到组大小，未碰撞商品使用 0，同组排序取商品轴升序。正式入口需在所有第四码符合 256 槽容量后才启动下游。
- C 实际启动参数和历史校验均固定于 C。读取 `run_normal.py`、`historical/config.yaml`、`historical/overrides.json`、`native/codex/plan_v1_1/protocol.py`、原模型、label/collator、候选指标接口和冻结的 CCE 与 unified SID 计算。初步源码显示复用 C 有效配置检查与原生实例化，保持下游训练 seed 42、普通评测 seed 2026、积累 16，预跑 3 个 optimizer update、48 个 microbatch，于一次 3 批验证选临时 best，三个条件各测事先冻结的 20 个用户。实际配置和运行待原件确认。
- 初检发现阶段复制恢复时，原生量化 `stage_result.json` 中的绝对 checkpoint 路径仍指向旧 attempt，而量化 gate 初版硬要求其等于当前 attempt 路径。已反馈主控与上游作者。最终复查需要以原始回执保持不变、源身份／SHA 和当前相对路径一致检查恢复。
- P0 runner 源码及 C 已完成 job 原始 `status.json` 印证终态 `SUCCEEDED`。受既有 runner 加载检查限制，单次 request 的 timeout 上限为 86,400 秒。这个上限属于执行系统，与正式训练器最大更新数分别记录。

## 真实 r0 回读与修订复核

- 独立从 Git 中重新读取 `b08c63940f5351dce63dc4cb8a0f9c6f1d8ac86e`，逐路径比较原 B native blob 与当前保存版本。81 个原件逐字节相同。另逐项对比服务器 request 哈希和同一 Git 版本的 208 个部署实体，其中 VERSION 为额外部署元数据。不存在未入该快照的执行源码。固定 C 文件与新目录分开。
- 上游每层观测到一批 buffer 只积累不改中心、一批原生初始化改中心、两批 Lightning 计步的中心更新。12 个原生 batch，24,576 个含重复行，覆盖全部 9,922 个商品。实际 checkpoint 三层均初始化。用服务器 CPU 以 float64 独立重算残差归一化和中心最小代价，逐层以真实上一层指派推进，三层的 29,766 个中心索引全部一致。独立复算第三层后的去重位和目录唯一性，9,922 个完整 SID、3,161 个前三码组，其中 1,611 个组发生前三码碰撞，第四码最大值 43。布置出的错商品轴、第四码 256、预跑身份作为正式消费三类直接准入调用均拒绝。
- 从 `bound` 里的三份 TFRecord 实际解码和检查 CRC，用户按原 0 至 24,771 轴逐行一致。训练 156,609 个保存项，验证保留历史加目标 173,699 个项，测试保留历史加目标 197,701 个项。所有行逐序列、目标位置和 NumPy/Python 数值语义比较准确。
- 根据服务器实际 Adam 更新前后参数字节 digest 和梯度范数，48 个 microbatch 中进行了三个真实更新，目标验证批数为三，best step 为 3，采用 `val/recall@5`。实际严格重载记录全部对应同一 SHA-256 `b58e3a8c3daa9da03b276d0764aa18a0c0d74b510da55a4cd43583d5ea1ba13a`，三个模式分别产生 20 用户、200 候选。候选索引、输入 mask、SID 层级身份、canonical target、填充和候选分数原字节均实际读取。Normal 原生测试和新恒等入口候选文件逐字节相同。三个模式覆盖的可见组含两个不同层级裸码相同的真实例子。
- 用原 A0/A1 预先保存的抽样号作为独立排列来源检查预跑每个实际可见组。S3a 与 S4 非恒等和实际 offset token 变化数分别是 147、182，原候选 SID 数组发生变化的用户各一位，score 字节发生变化的用户各 20 位。逐条复算 native top-k、稳定排名的统一 SID 和 CCE 指标含零分母处理，结果对应保存指标。三个模式均无命中，本次预跑的候选均恰好落在目录内。真实模型 `should_check_prefix=False`，原生 beam 的唯一目录过滤分支受此开关控制。预跑观察不作正式效果推断。
- r0 job `grid-letter-rerun-20261125-preview-r0` 的终态为 SUCCEEDED，exit 0，runner cleanup verified。从 17:39:37 至 17:43:20 UTC，未发现本轮前缀下的正式 job 目录。未执行正式量化、正式 SID、正式训练或正式三条件评价。
- 原生绝对路径恢复问题已在 r0 修复。出自原阶段的固定绝对路径与 stage_result 对比，当前目录检查同一相对路径后缀及 checkpoint SHA。审核读取了两个受检路径及原始回执。未在服务器重新提交作业来验证一次跨 attempt 重放。若训练阶段失败且缺少已完成 best 回执，程序阻止自动重复训练，保留该 attempt 等待按证据检查。
