# LC-Rec 六任务 LoRA 非作者审核

审核者：真实子代理 `/root/independent_review`。本报告不由实现作者代写。审核直接读取固定官方原件、本轮源码、请求/配置、两轮 GPU 的两 rank 全部原始 JSON、资源样本、CPU 全量 metadata 与固定样本，以及作者 Instruments index。审核者没有运行 GPU、加载模型、改数据、修改实现或操作 Git 发布。

## 结论及边界

**有界 BF16＋ZeRO-2 GPU 计算链通过；正式六任务数据仍阻塞；完整单 seed 的 96 小时判断为 INCONCLUSIVE。**

可行候选是双卡 `micro2 × accumulation32 = global128`。它完成六任务真实 F/B、真实 AdamW 更新、完整 active embedding/head 与 LoRA 的参数/梯度检查、保存和 fresh-base public PEFT 重载、原 17 prompt/beam20 接口。micro8、micro4 的长短混合极长 batch OOM 均保留，不能把典型 micro8 更新成功当作整个数据域的容量证明。

该结论不等于论文指标复现、已可信的正式训练集、完整 Trainer/DeepSpeed checkpoint 恢复或已经批准的后续研究 baseline。没有新增证据证明某篇论文完整运行过本轮完全相同的六任务 LoRA 配方并报告指标。

## 方法与数据身份

固定官方 `RUCAIBox/LC-Rec@e6f1b3b3f8b4fee55a49a7b3dfeba29dbfcf0669` 的 41 个文件，逐 bytes/SHA-256 对照 manifest 全一致。逐读官方 `lora_finetune.py`、`finetune.py`、Dataset、collator、prompt、utils、test/test_ddp、evaluator、run.sh、DS 配置。LoRA 与全参入口确实共用单个六任务 ConcatDataset；`run.sh` 实际运行全参数入口，不能据它编造 LoRA 的论文历史结果。

原 Instruments 四个实际输入资产的新 hash 均与固定准备审核一致：没有 LETTER SID、数据改写、辅助标签重生成或正式过滤。统计直接覆盖 221,225 条：131,837 / 9,922 / 9,922 / 20,000 / 24,772 / 24,772。独立重算全量 metadata 的长度分位数、均值/方差/计数，并直接检查 304 个样本的文本 SHA、BOS/EOS、930 SID 特殊 token 顺序、IDs/labels/mask 和 shifted labels。全部吻合。

单次 seed42 prompt/query/preference realization 内，完整长度均值依次为 92.562 / 160.014 / 149.044 / 186.203 / 142.688 / 161.297 tokens，最长 1,626；没有 2048 截断或零 shifted-label 样本。保留 Index2Item 的 133 条、FusionSeqRec 的 372 条仅监督 EOS 的空回答；另有 28＋23 条 input/full prefix tokenization 边界不完全相同，仍按原 collator 长度 mask，没有修补。

ItemSearch 直接使用辅助包的 target/history，训练 12,233/24,772 错配、测试 7,473/24,772 错配继续成立；训练中有 3,608 个 validation target、1,623 个 test target，已知 1,623 个可见历史含 validation item。PreferenceObtain 的输入来自当前 inter[:-3]，但 explicit preference 标签的实际生成上下文仍未知。12,539 条精确训练期 ID 配对仅支持不改数据的技术选样，不能认证自然语言标签来源干净。正式训练需作者提供同 split/remapping 的辅助资产及生成输入/版本依据。

## GPU 真实证据

| 作业 | RUN_COMMIT | 原始终态及含义 |
|---|---|---|
| gpu-r2 | `c4a8aef801074f1579d113e041ebf3a24380746c` | FAILED；micro8 典型组 2 次更新、保存/fresh 重载通过，但 `[8,1623]` OOM，生成 finally 的未定义 local 导致退出。 |
| gpu-r3 | `fa51f82f1a2f341e7c33e72dc2db0dd5ada5c6c9` | SUCCEEDED exit0；micro4 `[4,1626]` OOM，micro2 通过实际混合及全局长短 stress 门控；2 次 global128 更新、保存/重载、17 prompt 生成接口通过。 |

独立核对两轮请求中固定源码/配置的 Git blob SHA，以及各轮 24/26 个轻证据成员的大小/hash。原始失败没有重标为成功。初审漏检了 inference finally 的未定义 local，本报告明确保留；round3 仅两处 local→local_rank 修复经逐字 diff、全部 scope 的 symbol-table 检查，并在真实生成中复核。

两 rank 均有 450 个 trainable tensor、289,751,040 参数：LoRA 19,988,480，完整 active embedding/head 269,762,560。输入/输出矩阵不共享；original_module 完整 bytes hash 前后相同。AdamW 精确包含全部且仅包含这些参数；ZeRO-2 FP32 master/moment partitions 在 GPU，没有 CPU optimizer offload。初始化后的 dtype 转换单独记录，更新 delta 基线在 DS init 后，未把舍入算作更新。

micro2 的每次更新均有每 rank 32 个真实 micro-step，450 个梯度每个 micro 都有限，SID 梯度代表性非零，Adam step 依次 1、2，两 rank 完整 450 tensor 的最终 bytes hash 相同。LR 从 0 到 `7.142857142857143e-7`；所有 224 个 LoRA_B 以及新增 SID embedding/head 行发生变化。224 个 LoRA_A 在两次真实更新前向中的梯度都为 0、参数未变：B 零初始化且首次 step 的 LR=0，因此第一次非零 LR 只让 B 改变。这不是已证实的 BF16 舍入停滞；本轮没有验证 A 的真实 optimizer 更新，不为此增加运行。

micro2 长尾附加测试在真实 GPU Adam/master states 常驻时完成 `[2,1626]` F/B；先通过已核源码的 public destroy 移除 ZeRO hooks，未调用 optimizer step，计数与全部 450 参数 hash 不变。它是无通信的容量诊断，不是极长样本的完整 ZeRO optimizer update。

真实 micro2 联合吞吐为 8.2422、8.3145 samples/s；有效 tokens/s 为 904.26、1014.76。两组 padding 为 19.00%、21.92%。每任务 4 个 micro1 F/B 诊断均通过；两 rank 重复同一个代表样本，属于两次计算呈现、一个 distinct record，没有 optimizer/梯度通信。不能把它们冒充正式每任务吞吐或与 LETTER 14.66 samples/s 作严格加速比。

## 保存、生成与正式恢复

实际保存 450 keys、289,751,040 参数，包括完整 active embed_tokens/lm_head、tokenizer 与 SID 顺序。真实释放旧模型后，固定完整 base＋`PeftModel.from_pretrained` 新建模型重载；逐 key/shape 精确值及逆转保存 dtype 后 bytes SHA 全部吻合。448 个 LoRA tensor 由保存 BF16 原生重载为 FP32，值完全相等；两 active 矩阵保持 BF16。没有用 strict=False 掩盖缺失。

原 SeqRec 两模板/用户的验证构造及六个固定真实验证样本通过，未选 checkpoint。原 17 模板每模板 3 个固定真实用户，共 51 cases、1,020 beams，全部 beam20、四位 SID position token＋EOS 合法，未降 beam、未改 trie/去重/指标。审核者独立读取 hash 匹配的原作者完整 index：**1,020 个输出全部非 canonical SID，重复 0。** 因而通过的是原生成协议接口，不能声称生成了合法目录推荐或具有推荐效果。

未实测完整 Trainer/DeepSpeed optimizer/scheduler/RNG checkpoint 保存恢复或实际 best-checkpoint 路径；fresh adapter 重载不能替代它。

## 目标权重与 96 小时成本

全量保存的实际长度上，随机混合 padding 诊断 micro8/4/2/1 分别为 50.20% / 37.51% / 20.81% / 0。按原 micro-batch token-mean CE、再平均 micro/rank 的原规则独立重算权重；global128 相同不能证明目标等价。例如同一个 mixed plan 中，SeqRec 的 CE 权重 micro8=0.27376、micro2=0.50663；Preference micro8=0.33911、micro2=0.19040。正式采用 micro2、BF16 初始化轨迹等须明确研究选择，不能包装成与原训练完全等价。

实际 CPU Accelerate sharding、固定 HF4.31 loop/callback 控制得到 6,912 updates、warmup70；对应循环呈现 884,736，区别于 nominal 四轮 884,900。首 epoch 的 state.epoch<1，使 eval_delay=1 下有效验证 3 次，保守成本仍取 4 次。真实完整 DS/Trainer epoch-tail 行为未跑。

成本计算 RUN 为 `05ecb772bbb8c8940a5515fb13a2b4563895f2cd`。审核者不 import 估算器，直接从 32 个非零 LR micro-step 重建两 rank MAX(F+B)、工作量特征与 7 个非负 active-set 拟合，并逐一重算 17 个 prompt 的测试成本与全部分项。数值一致。

| 已量化项（小时） | 中心 | 保守情景 |
|---|---:|---:|
| 训练 F+B | 29.1205 | 38.0546 |
| CPU 准备＋实测同步/传输/优化器等 | 0.7871 | 1.1248 |
| 验证 | 1.7752 | 3.1163 |
| 17 prompt/beam20 生成 | 33.9028 | 47.3281 |
| Adapter 保存＋fresh restore proxy＋初始化 | 0.0653 | 0.0816 |
| **已量化小计** | **65.6508** | **89.7055** |

完整 training-state 保存/flush/恢复与全测试 CPU decode/evaluator aggregation 未量化，保存为 None；完整 total 也是 None。长期波动、随机模板尾部、正式验证 micro-batch 等仍未约束。可行布局仅有一个非零 LR steady update、32 个 micro 观测，拟合长度 92–503；1626 容量通过不等于已测该尾部的持续通信训练速度。RMSE≈0.01077 秒不是置信区间，拟合 linear 项为零也不证明线性计算不存在。

因此 **INCONCLUSIVE** 正确：保守已知小计距 96 小时约 6.2945 小时，不能宣称足以覆盖未知项且留有明显余量，也不能把两个小计写成完整 seed 保证时间。

## 预算、终态与交付身份

11 个作业有真实终态：8 SUCCEEDED、3 FAILED，原失败保留；均 cleanup_verified，worker/group/supervisor 无遗留，最终两卡没有 compute process。全任务 4 次 global optimizer update，其中 2 次非零 LR；两 GPU 作业共 1,371.0311388522387 GPU-device 秒，即 22.85052 device 分钟、双卡墙钟合计 11.42526 分钟，最长单作业 6.09428 分钟。未增加预算或启动正式 epoch/seed、Int8、Qwen、Shuffle 或其他数据域。

含失败尝试的全程 GPU 最大 allocated≈20.4046GiB、reserved≈22.6582GiB；micro2 resident-Adam stress allocated≈19.0617GiB。GPU guard 采样最大进程树 RSS sum≈24.8686GiB、主机最少 MemAvailable≈231.3825GiB，没有任务 swap/新增 swap。最终定向新增空间（含源码传输包）1,276,698,624 bytes，home/data 两文件系统均满足预留。新增独立 venv/.pth 及 venv 自带的 bootstrap 组件，没有 apt、pip install 或额外工具链操作；未改共享 site-packages 或全局配置，未清理资产。

逐项机器审核见本目录 `source_review.json`、`cpu_runtime_review.json`、`protocol_runtime_review.json`、`grouping_runtime_review.json`、`gpu_r2_runtime_review.json`、`gpu_r3_runtime_review.json`、`budget_runtime_review.json`、`resource_terminal_review.json`。原始作业在 `../evidence/jobs/`，服务器为 `/home/bwwu/grid-audits/sid-extensions-p0/jobs/<job_id>/`，源码是 `/home/bwwu/lc-rec/six-task-20261110/releases/<RUN_COMMIT>/` 无 .git export。

本报告不自嵌自身 DELIVERY_COMMIT，也不预写尚未发生的 GitHub/ZIP 检查。最终固定交付、服务器映射和 GitHub 回读由真实发布后回执绑定；ZIP 成员/CRC/完整解压/哈希/Windows 身份由包外实测回执绑定。发布与打包终核通过后才可宣称交付闭环，不能仅凭本技术报告宣布全部完成。
