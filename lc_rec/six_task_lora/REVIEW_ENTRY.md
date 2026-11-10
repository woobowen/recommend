# LC-Rec SIX-TASK LoRA：一次性 GPU 技术审核入口

**LC-Rec SIX-TASK LoRA GPU TECHNICAL PASS（有界 micro2 计算链）；正式可信六任务训练 BLOCKED；96 小时完整 seed 成本 INCONCLUSIVE。**

本轮已完成所有可独立推进的授权检查，未运行完整 epoch、正式 seed、完整验证/测试集、Int8、Qwen 或 Shuffle。PASS 只覆盖这里列明的真实计算、参数和接口证据，不代表论文指标复现，也不代表这套配置已可直接作为可信研究 baseline。

主要原始证据：六任务全量统计 [`workload_summary.json`](evidence/jobs/lc-six-20261110-cpu-r0/output/workload/workload_summary.json)、[完整记录 metadata](evidence/jobs/lc-six-20261110-cpu-r0/output/workload/records.jsonl.gz)、[确定性样本及 mask](evidence/jobs/lc-six-20261110-cpu-r0/output/workload/samples.json)、最终 GPU [rank0](evidence/jobs/lc-six-20261110-gpu-r3/output/probe/probe_rank0.json)/[rank1](evidence/jobs/lc-six-20261110-gpu-r3/output/probe/probe_rank1.json)、[成本原始计算](evidence/jobs/lc-six-20261110-cost-r3/output/cost/budget.json)、[所有作业终态和资源](evidence/final_remote.json)。

真实非作者审核入口为 [`review/INDEPENDENT_REVIEW.md`](review/INDEPENDENT_REVIEW.md)。审核者直接读取官方源码、全量记录 metadata、样本、两 rank 原始证据，并独立重算成本。GitHub 发布和 ZIP 实际核验在发布后生成外部回执，随 ZIP 的 `final_receipts/` 提供；ZIP 的最终哈希和包外独立核验不放回包内，避免自引用。

## 1. 固定身份及作者实现依据

仓库 `woobowen/recommend`，唯一操作分支 `wbw-lc-rec`。实际 Git 根 `/home/addaswsw/recommend-worktrees/lc-rec`；规范从 `/home/addaswsw/recommend/` 显式读取，未声称同级目录自动继承。初始 HEAD 为 `903b2971a78bcf140491669ca59004beaa628815`；两个先存未跟踪 Qwen tokenizer 文件未改、未加入提交。没有改其他分支、强推或创建新分支。

固定官方源码 [`RUCAIBox/LC-Rec@e6f1b3b3f8b4fee55a49a7b3dfeba29dbfcf0669`](https://github.com/RUCAIBox/LC-Rec/tree/e6f1b3b3f8b4fee55a49a7b3dfeba29dbfcf0669)，本仓库 `lc_rec/vendor/original` 的 41 个文件与已审 manifest 相同。`lora_finetune.py` 与 `finetune.py` 确实共用六任务 Dataset、ConcatDataset 和 Collator；`run.sh` 实际使用全参数入口。官方源码支持完整六任务 LoRA，不等于论文已报告相同 LoRA 配方的指标。此次有界原始资料核查没有获得新增的相同完整配方指标证据，见 [查询边界](evidence/literature_identity.json)。

模型固定 `huggyllama/llama-7b@4782ad278652c7c71b72204d462d6d01eaaf7549`，服务器 `/data/bwwu/lc-rec/assets/llama-7b/4782ad278652c7c71b72204d462d6d01eaaf7549`。复用完整原权重与既有全量 hash 审核，本轮定向核实小文件 hash 和大权重身份，未重复下载或反复全量 hash。数据固定 `/data/bwwu/lc-rec/assets/data/Instruments` 的 LC-Rec 作者资产；四个输入 hash 与准备审核相同。没有 LETTER SID/RQ-VAE/CF/Div、上游重训或重生标签。

24,772 用户、9,922 物品、206,153 交互；原 SID 新增 930 token，词表 32,000→32,930，顺序和 ID 已验证。9,922 物品对应 9,897 canonical SID，23 个碰撞组及 25 个额外物品是原资产性质，未去重或重新量化。

## 2. 六任务监督与实际负载

使用原 Dataset 和原 Collator，全量遍历一份 **seed42 固定随机 prompt/query/preference realization**。这不是所有 epoch 所有随机 prompt 的穷举。参数沿作者 Instruments：`train_prompt_sample_num=1,1,1,1,1,1`、`train_data_sample_num=0,0,0,20000,0,0`、max history20、max length2048。

| 任务 | 记录数 | 输入均值 | 完整序列均值 | shifted label 均值 | 完整 p50/p90/p99/max |
|---|---:|---:|---:|---:|---|
| SeqRec | 131,837 | 87.56 | 92.56 | 5.00 | 83 / 136 / 172 / 175 |
| Item2Index | 9,922 | 155.01 | 160.01 | 5.00 | 110 / 335 / 562 / 1623 |
| Index2Item | 9,922 | 49.29 | 149.04 | 99.75 | 97 / 320 / 553.79 / 1626 |
| FusionSeqRec | 20,000 | 127.25 | 186.20 | 58.95 | 137 / 384 / 607 / 1000 |
| ItemSearch | 24,772 | 137.69 | 142.69 | 5.00 | 123 / 236 / 331 / 617 |
| PreferenceObtain | 24,772 | 82.49 | 161.30 | 78.81 | 147 / 229 / 314 / 629 |
| 合计 | **221,225** | | | | 最长1626 |

单位均为 token；原始摘要另含每任务及联合的 p10/p50/p90/p99/max、方差、回答和有效 label 长度、padding 与截断统计。SeqRec 的短答案不能代表 Index2Item、FusionSeqRec、PreferenceObtain 的计算量。

SeqRec 从当前训练历史预测下一 SID；Item2Index 由标题/描述预测 SID，Index2Item 反向生成标题/描述；FusionSeqRec 按原 prompt 混合 SID/标题历史并生成对应目标；ItemSearch 使用辅助 query/intention/preference 及其指定历史生成 SID；PreferenceObtain 从当前训练历史生成辅助 explicit preference。原 prompt 全部来自 `vendor/original/prompt.py`，未替换成 LETTER 模板。

原训练字符串为 SFT instruction＋response；Collator 对完整文本追加 EOS，BOS=1、EOS=2、PAD=0，右 padding；PAD 和输入前缀 mask 为 -100。模型原 shifted CE 在每个 rank 的 micro-batch 内按有效 label token 求均值，没有 LETTER temperature0.8。930 SID token 是原序列，原 token 前缀保留。304 个选定记录有原 Dataset 索引、文本 hash、token IDs、attention mask、labels 和选择原因；GPU 重放原 Dataset/Collator 与其精确相等。

本 realization：2048 截断0，零有效 shifted-label 样本0。原始空回答仍存在：Index2Item133、FusionSeqRec372 条仅监督 EOS；另外28＋23 条 input/full 的 tokenizer 前缀边界不完全一致，保留原 collator 行为，不假装每一条回答边界都完美对齐。

样本在各任务 p10/p50/p90 和合法最大长度上按确定性长度分位位置与原索引规则选取，不按 loss 或速度挑选。两组 global128 联合技术样本分别按真实比例取 `[76,6,6,12,14,14]`；不是正式过滤或新的 sampler。长度分桶仅诊断。全量固定随机混合的 padding 比例 micro8/4/2/1 为 50.20% /37.51% /20.81% /0，详见 [grouping](evidence/jobs/lc-six-20261110-grouping-r2/output/grouping/grouping.json)。

## 3. 数据可靠性与正式训练阻塞

原辅助训练记录 12,233/24,772 与当前 target/history 不配对；测试错配7,473。训练辅助 target 中3,608 对应当前 validation、1,623 对应 test，已知1,623 条辅助可见历史含当前 validation item。这些是固定资产的观测，不构成论文历史实际运行泄漏的因果证明。

ItemSearch 直接消费这些辅助 target/history，正式可信训练受阻。PreferenceObtain 使用当前 `inter[:-3]` 历史，但辅助 explicit preference 的逐记录生成上下文未确认，不能凭 ID 配对认证标签无未来信息。其余任务不直接消费这份 query/preference 对齐字段；本轮未发现同类直接配对问题，不扩张成所有数据来源均已证明干净。

存在12,539 条与当前训练期 target/history 精确配对的未修改辅助记录，可用于本次技术更新的有界选样；它们不构成已获认证的正式辅助训练集。错配或标签来源未证实的样本仅用于长度/计算诊断，不用于推荐质量结论。没有删除用户、改历史、改目标、重生标签或正式过滤数据。

正式六任务复现所需最小条件：作者提供与当前 interaction split、item remapping、SID 完全一致的辅助资产/清单，逐用户生成输入与版本依据，说明 query/intention/preference 所用训练期上下文及原处理方式。解决前不启动正式训练或指标复现。

## 4. 真实 GPU 候选与适配差异

候选保留官方 LoRA r8、alpha32、dropout0.05，七个投影 `q/k/v/o/gate/down/up_proj`，`modules_to_save=embed_tokens,lm_head`。使用完整 BF16 base、双卡 ZeRO-2、Torch AdamW、FP32 GPU master/moment 分片，无 CPU optimizer offload。正式四轮仅用于成本估计；LR5e-5、wd0.01、cosine、warmup0.01 的来源是作者全参 Instruments 脚本，是有来源的技术候选，不声称 LoRA 论文最优值。实际6912 scheduled steps 对应 warmup70。

与原件的重要差异：官方 `from_pretrained` 未显式 dtype，本轮明确 BF16 初始化，数值舍入轨迹与 FP32 加载后 autocast 不同；以有界脚本替换 `Trainer.train`，使用双4090/ZeRO-2（作者全参示例八卡/ZeRO-3）；关闭 compile；采用固定可重放技术样本；保留原 micro-mean CE，但改变 micro 布局会改变目标权重。没有编辑作者原件或共享 site-packages，没有加载 LETTER 模型/损失/runtime patch。差异原始记录在 `probe_rank*.json:engineering_differences`。

实际 DeepSpeed 配置见 [`deepspeed_effective_rank0.json`](evidence/jobs/lc-six-20261110-gpu-r3/output/probe/deepspeed_effective_rank0.json)：stage2、BF16、accum32、micro2、global128，reduce/allgather bucket各5,000,000，overlap_comm=false，gradient clipping1。DDP 单独路径未测试；Int8 备选没有必要，未运行。QLoRA、4-bit、NVMe、全参 offload、减层等均未运行。

| 布局 | 实测结论 |
|---|---|
| 双卡 micro8×accum8 | 典型联合组两次真实更新成功；`[8,1623]` 长短 stress 在原 dense attention FP32 softmax OOM，不能作为全长域布局 |
| 双卡 micro4×accum16 | `[4,1626]` stress OOM，未运行正式 optimizer update |
| **双卡 micro2×accum32** | 真实六任务 global128 两次更新通过；选为有界候选 |
| 双卡 micro1×accum64 | 未执行完整 ZeRO 更新；每任务 micro1 仅作无更新 F/B 诊断 |

micro2 更新后又在 GPU Adam/master states 常驻时对 `[2,1626]` 完成 F/B。先按已核源码的 DeepSpeed public destroy 移除梯度 hooks，保留模型、GPU optimizer states 和 process group；全部450 参数 hash、step计数不变。此项证明有状态常驻时的无通信容量，不是极长样本完整 ZeRO update 或整个 epoch 的验收。

micro-mean CE 使 global128 相同也不等价：同一 mixed plan 中 SeqRec CE 权重 micro8=0.27376、micro2=0.50663，Preference=0.33911→0.19040。没有擅自改成全局 token-mean loss；正式布局/归一化/BF16 初始化仍需明确方法决策。

## 5. 参数、更新与真实效率

两 rank 各450 个 trainable tensor、**289,751,040 参数**：448 个 LoRA tensor 共19,988,480，两个完整 active embedding/head 共269,762,560。两矩阵各32,930×4,096、互不共享；不仅训练930行。必要 SID 行有有限非零梯度并实际更新。`original_module` 副本冻结，完整 hash 不变；AdamW membership 精确等于 trainable 集合。全部450 参数更新后两 rank 的完整 bytes hash 相同。

每个真实 micro-step 的450梯度均有限；224 LoRA_B 与 SID embedding/head 有非零变化。224 LoRA_A 梯度及变化为0：B零初始化，第一次 LR0，第二次前向 B仍0，随后首次非零 LR 才使 B更新；本轮没有验证 A 的真实数值更新，不能归因于已证明的 BF16 舍入停滞。更新比较基线在 DS 初始化/转换之后，不把 dtype 舍入误算为 optimizer 更新。

本轮两次独立初始化 GPU 作业合计4次 global update，其中2次 LR0、2次非零 LR；各第二次 LR=`7.142857142857143e-7`。原总上限6，没有把失败重跑预算重置。

| 最终 micro2 联合更新 | 墙钟秒 | samples/s | 有效 tokens/s | 监督 tokens/s | padding |
|---|---:|---:|---:|---:|---:|
| warmup LR0 | 15.52985 | 8.24219 | 904.26 | 159.95 | 19.00% |
| 首次非零 LR | 15.39478 | 8.31451 | 1014.76 | 196.24 | 21.92% |

计时使用 CUDA synchronize 和两 rank 最大墙钟，分子为两 rank 实际总呈现量，不相加 rank 耗时。包含完整累积、通信和 engine step；CPU collator 单独测量。非零更新 rank0：CPU collator0.23837s、传输0.00381s、forward3.37195s、backward11.84965s、engine step0.11983s（后者含非边界调用）；不能从各 rank 的独立 phase 最大值拼出另一条真实整体墙钟。模型初始化111.0656s，ZeRO 初始化4.6937s。

下表为各任务短/中/长/极长代表的 **micro1 无更新 F/B 计算呈现吞吐范围**，不是包含 optimizer/梯度通信的独立任务训练速度。两 rank 重复同一条原记录，distinct record=1、compute presentations=2；首个 SeqRec 短样本含冷启动影响，没有删除这个慢点。

| 任务 | full 长度（短/中/长/极长） | samples/s 范围 | 有效 tokens/s 范围 |
|---|---|---:|---:|
| SeqRec | 68/83/136/175 | 1.799–4.526 | 122–791 |
| Item2Index | 65/110/335/1623 | 1.738–4.470 | 252–2821 |
| Index2Item | 61/97/320/1626 | 1.750–4.393 | 268–2845 |
| FusionSeqRec | 88/137/384/1000 | 3.265–4.559 | 390–3265 |
| ItemSearch | 83/123/236/617 | 3.698–4.654 | 386–2282 |
| PreferenceObtain | 109/147/229/629 | 4.135–4.591 | 500–2601 |

24 个逐点 F/B、传输、mask、梯度、资源与样本 ID 原始记录均保留。六任务负载、长答案、padding、micro 布局和 full embedding/head 状态均影响成本；本轮没有控制实验单独识别每项因果贡献。LETTER 已审14.66 samples/s 是不同 SID/SeqRec/Int8/alpha16/temperature0.8/有效batch64/实现的历史参考，未用于本次外推或严格加速比。

## 6. 验证、保存与原推荐协议

原 SeqRec 验证每用户抽2模板，构造49,544条/次；不是17个测试模板。固定 Trainer4.31 loop/callback 检查显示 `eval_delay=1` 叠加首轮 state.epoch<1，四轮 schedule 有效验证3次；保守估算另计4次。原选择较小 eval_loss、`load_best_model_at_end=True`。本轮6个真实短/中/长验证记录 forward/有限 loss 通过，约0.0835–0.0906s/batch1；未完整验证、选模或真正走完 best-checkpoint 恢复。

Adapter 保存/释放/从完整固定 base 新建并用 public PEFT 重载通过，逐一核对全部450 keys、shape、值及转回保存 dtype 的 bytes，包含完整 active embedding/head、tokenizer、930 SID顺序和配置。模型文件579,573,370字节留服务器，未进入 Git/ZIP。保存1.0235s、fresh reload110.9067s。原生 PEFT 将448 LoRA tensor 从 BF16 保存值加载到 FP32，显式双向精确比较通过；两 active 矩阵保持 BF16。没有用 strict=False 掩盖缺失。

**完整 Trainer/DeepSpeed optimizer/scheduler/RNG checkpoint 保存和恢复未实测**。未写无必要巨大 checkpoint，adapter reload 不能证明正式训练可恢复，成本中未量化项也未记零。

原 SeqRec **17 prompts×3固定真实用户=51 cases，每项beam20，共1020 beams** 的生成接口通过，max_new_tokens10、逐位置 token callback、原 postprocess/Hit/NDCG 和逐模板汇总口径保留。原 callback 是逐位置允许集合，不是严格 canonical trie；原脚本 OOM 会减 beam，此 harness 禁止静默降 beam。实际没有降 beam。

**1020/1020 输出位置 token 合法，却全部不在作者 canonical SID 目录；重复0。** 这只能证明原接口可执行，不证明推荐物品合法或推荐效果。独立审核用同 hash 的作者 index 再算确认。未改 trie、去重、metric、prompt或beam以求通过；没有报告本短跑推荐指标。

原双卡测试 DistributedSampler(drop_last=True)，24,772 恰好被2整除，无尾丢失/补重复；奇数诊断子集不适用同结论。本轮显式不重叠分配3用户，全部覆盖。完整17模板指标按每模板分别计算再作原均值/最小/最大汇总，不改成单模板或跨模板融合推荐。逐模板实测见 `generation_rank*.json` 与成本 JSON。

## 7. 96 小时：已知成本与未知项

实际 CPU Accelerate sharding、Trainer4.31 控制流核到各布局6912 scheduled optimizer steps、warmup70。micro2 每 rank loader55,307，accum32；原 loop 的跨 epoch 边界/停止规则模拟呈现884,736，与名义4×221,225=884,900不同。没有将 `ceil(N/128)` 直接冒充真实 Trainer steps；完整真实 epoch-tail DS行为尚未运行。

中心估算从最终非零 LR 更新32个 paired-rank micro-step 的 MAX(F+B) 和完整221,225长度/padding负载建立非负常数＋线性＋二次 work 拟合，另加实际非 F/B 和数据准备。拟合覆盖宽度92–503，完整尾部1626属外推，RMSE约0.01077s，不是置信区间。保留静态算量敏感性；保守情景取较慢敏感性及显式25%波动，并单列未知项，不用系数隐去未知成本。审核者未 import 估算器，独立重建拟合及逐项成本吻合。

| 单 seed 分项（小时） | 中心 | 保守情景 |
|---|---:|---:|
| 训练 F+B | 29.12049 | 38.05461 |
| CPU 数据准备 | 0.45767 | 0.57209 |
| 实测传输/Python/optimizer等 | 0.32938 | 0.55275 |
| **训练已知合计** | **29.90754** | **39.17944** |
| 验证（中心3次/保守4次） | 1.77517 | 3.11631 |
| 24,772用户×17模板beam20 GPU生成 | 33.90276 | 47.32810 |
| **评价已知合计** | **35.67794** | **50.44441** |
| adapter保存4次 | 0.00114 | 0.00142 |
| best restore的fresh-adapter代理 | 0.03081 | 0.03851 |
| 初始化 | 0.03335 | 0.04169 |
| **已量化小计** | **65.65077** | **89.70547** |
| 完整training-state写入/flush/恢复 | **未知** | **未知** |
| 全测试CPU decode/evaluator聚合 | **未知** | **未知** |
| **完整总时间** | **未确定** | **未确定** |

生成按每用户每prompt平均耗时×24,772÷2，不能拿3用户不均匀分rank的整段墙钟直接当饱和吞吐。中心/保守分别保留逐prompt测量。验证正式micro吞吐、随机prompt长尾、持续通信/allocator碎片与长期资源波动仍需补证。

结论 **INCONCLUSIVE**。保守已知小计只剩约6.2945小时，未满足“完整成本已覆盖且明显留余量”的LIKELY条件；也未证明必定超过96小时。没有减少epoch、任务、beam、用户或模板来得到预算通过。


各prompt独立成本如下（每项3真实用户、beam20、batch1；GPU生成，不含完整CPU聚合）。保守列为该prompt较慢用户外推后×1.25。

| prompt ID | 实测均值秒/用户 | 中心全用户小时 | 保守全用户小时 |
|---|---:|---:|---:|
| 0 | 0.67775 | 2.33185 | 3.00143 |
| 1 | 0.57854 | 1.99050 | 2.77551 |
| 2 | 0.58663 | 2.01835 | 2.84682 |
| 3 | 0.56825 | 1.95511 | 2.73687 |
| 4 | 0.56032 | 1.92780 | 2.70852 |
| 5 | 0.55907 | 1.92350 | 2.70806 |
| 6 | 0.55121 | 1.89648 | 2.67425 |
| 7 | 0.56371 | 1.93948 | 2.71320 |
| 8 | 0.55871 | 1.92228 | 2.71352 |
| 9 | 0.56716 | 1.95136 | 2.73782 |
| 10 | 0.56061 | 1.92880 | 2.71237 |
| 11 | 0.55895 | 1.92308 | 2.72071 |
| 12 | 0.59646 | 2.05217 | 2.85233 |
| 13 | 0.59646 | 2.05214 | 2.87223 |
| 14 | 0.58742 | 2.02103 | 2.85056 |
| 15 | 0.58151 | 2.00071 | 2.79625 |
| 16 | 0.60110 | 2.06813 | 2.90764 |

## 8. 作业、错误、预算与环境

| job_id（统一前缀lc-six-20261110-） | 终态 | RUN_COMMIT | 含义 |
|---|---|---|---|
| setup-r0 | SUCCEEDED | e5ff6bcdb438e5266f25d9fd6fbe4edf45cc0e67 | 独立环境 |
| cpu-r0 | SUCCEEDED | a8547adac2dcf0eb2b32e7be932a94f868d11fb1 | 全量负载与样本 |
| protocol-r0 | FAILED | a8547adac2dcf0eb2b32e7be932a94f868d11fb1 | AcceleratorState未初始化 |
| protocol-r1 | SUCCEEDED | b0896f644167344d91b83c4180fb05913078f437 | 初始化后实际loader审查 |
| compatibility-r1 | FAILED | 9f1a164a9e470d9a41861fc9238e735222d673a0 | tiny CPU测试漏抄原件freeze逻辑 |
| compatibility-r2 | SUCCEEDED | 05d38188266d9fcec6f304c6fe6f11ee60b2bab8 | tiny合成API/F/B，非7B证据，无step |
| policy-r2 | SUCCEEDED | 05d38188266d9fcec6f304c6fe6f11ee60b2bab8 | 验证callback |
| grouping-r2 | SUCCEEDED | c4a8aef801074f1579d113e041ebf3a24380746c | 全量padding/CE权重 |
| gpu-r2 | FAILED | c4a8aef801074f1579d113e041ebf3a24380746c | micro8长尾OOM；计算/保存通过后generation finally未定义local退出 |
| gpu-r3 | SUCCEEDED | fa51f82f1a2f341e7c33e72dc2db0dd5ada5c6c9 | micro4失败、micro2有界全链通过 |
| cost-r3 | SUCCEEDED | 05ecb772bbb8c8940a5515fb13a2b4563895f2cd | 真实数据成本计算 |

任务级 round0初次执行；round1修正Accelerator初始化；round2修正tiny测试的原件freeze遗漏；round3修正finally变量，并做已授权micro回退/状态常驻stress。修正额度已用尽，未继续运行修复。首次全量metadata SCP60s超时只做同job身份240s有界传输重试，没有重新计算或冒充研究修正。所有失败保留原始日志与终态。

11作业（8成功、3失败）全部真实终态、cleanup_verified，worker/group/supervisor全无残留，最后两卡无compute process。两GPU作业合计 **1,371.03113885 GPU-device秒=22.85051898 device分钟**；双卡墙钟合计11.42526分钟，最长单job6.09428分钟。guard计费含导入/加载/失败/保存/退出，不只计算kernel时间。4global updates低于6上限；没有完整epoch。

含OOM尝试最大GPU allocated20.40458GiB、reserved22.65820GiB；micro2状态常驻stress峰值19.06171/20.27148GiB。guard采样最大进程树RSS求和24.8686GiB（共享页可重复计数），主机最少MemAvailable231.38GiB，任务VmSwap0/未新增swap。不是声称主机总swap为0。最终新增任务目录连同8个源码传输档案1,276,698,624字节，未清理资产；两个文件系统均保留超过30GiB与10%空间。详见资源原始JSON。

新增系统包0、`pip install`等操作0、新工具链0。新建 `/data/bwwu/envs/lc-six-task-20261110`（`venv --system-site-packages`，含bootstrap pip23.0.1/setuptools79.0.1），新增其pyvenv.cfg及只读框架路径`.pth`；没有改全局rc、CUDA、驱动、SSH/VPN、共享site-packages或LETTER环境。实际复用torch2.6.0/CUDA12.4、transformers4.31、DeepSpeed0.16.9、Accelerate0.21、PEFT0.5.0等已审组件，来源路径记录在环境证据。环境和产物保留，未未经授权清理。

## 9. 固定源码、交付与恢复入口

每个job的request含完整argv、cwd、RUN_COMMIT、配置与输入manifest，stdout/stderr、退出码和cleanup在 `evidence/jobs/<job_id>/`。服务器作业根 `/home/bwwu/grid-audits/sid-extensions-p0/jobs/`；runner为既有 `runner-398768a732bbfebd.py`，未替换共享runner。执行时实际transport/gridctl/runner源码及hash见 [`execution_source`](evidence/execution_source/manifest.json)。

服务器源码为 `/home/bwwu/lc-rec/six-task-20261110/releases/<完整RUN_COMMIT>/` 的 **无.git export**。8份RUN_VERSION/deployment清单及421个实际运行源码/配置文件，已从服务器重新读取hash并对照对应Git对象，见 [`server_source_readback.json`](evidence/server_source_readback.json) 与 `evidence/deployments/`；不是声称服务器checkout。GPU运行固定c4a8…和fa51…；最终成本固定05ec…。DELIVERY只新增结果、审核和本入口，不再改变成功GPU运行使用的脚本/有效配置；最终逐文件对应由包内 `final_receipts/run_delivery_mapping.json` 绑定完整DELIVERY SHA。

GitHub只推送 `wbw-lc-rec`；发布后真实回读branch HEAD及固定SHA关键源码、有效配置、入口和非作者结论，回执收入ZIP。最终ZIP由提交选定树和必要外部回执组成，不含模型/optimizer大文件/私钥；含逐文件SHA-256，实际CRC、完整解压、成员大小/hash、危险路径/重复/秘密pattern检查，以及Windows原生Get-Item/Get-FileHash验证。精确路径/bytes/hash/成员数以包外`.receipt.json`及最终回复为准，不用占位值。

恢复只读查询：在实际Git根执行 `python3 lc_rec/six_task_lora/scripts/control.py query --job lc-six-20261110-gpu-r3`，可核原job，无需重提。adapter留对应服务器job的`output/probe/adapter/`；本轮不再提交GPU作业。完整正式复现仍依赖作者数据补证、正式微批/数值方法决策、完整训练恢复及成本未知项。完成本次审核交付后停止，不启动下一实验。
