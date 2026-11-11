# LC-REC × QWEN3 SIX-TASK PREFLIGHT：PARTIAL

本次为用户明确授权的一次性四路线技术预跑，日期2026-11-11。四路线均完成真实双RTX4090计算、非零学习率更新、最长样本完整ZeRO-2更新、严格保存重载与17模板beam20；**四路线最终均为PARTIAL，而不是完整方法PASS**。原因是Qwen BPE下原LC-Rec独立编码prompt长度的mask不能完整保留所有回答文本的监督；已有辅助数据错配也仍阻止可信正式六任务复现。本轮没有修订这两个研究条件，没有启动正式训练或完整推荐指标评价。

## 1. 四路线终态与96小时结论

显存列为两rank中较大的**训练更新阶段CUDA allocated峰值**，不是整个作业所有阶段的精确CUDA峰值。吞吐为驻留Adam状态、非零LR、真实六任务混合global128完整更新的总样本呈现数除以MAX-rank墙钟；不是六个单任务吞吐拼接。时间为一个seed的条件性中心/保守小时数。

| 路线/最终终态 | 加载 | 双卡F/B | 非零更新 | 最长样本完整ZeRO | allocated峰值GiB | 联合samples/s | 中心h | 保守h | 96h计算状态 |
|---|---|---|---|---|---:|---:|---:|---:|---|
| Q06-L / PARTIAL | PASS | PASS | 3/4次 | PASS，1513 token | 8.8403 | 5.7095 | 70.1046 | 112.5678 | UNLIKELY_WITHIN_96H |
| Q06-F / PARTIAL | PASS | PASS | 2/3次 | PASS，1513 token | 11.2350 | 6.0383 | 59.4405 | 85.7465 | INCONCLUSIVE |
| Q17-L / PARTIAL | PASS | PASS | 3/4次 | PASS，1513 token | 12.6046 | 5.4798 | 72.6054 | 114.6917 | UNLIKELY_WITHIN_96H |
| Q17-F / PARTIAL | PASS | PASS | 2/3次 | PASS，1513 token | 20.9281 | 1.6183 | 178.2663 | 256.2582 | UNLIKELY_WITHIN_96H |

**没有候选达到“可可靠承诺96小时/seed”的验收。** Q06-F是本次已测配置中时间预算最有希望者，但保守余量只有10.2535小时，完整epoch尾部/正式Trainer恢复/长时波动尚未实测，不升级为LIKELY。分级器以保守至少24小时规划余量作为“明显余量”的工程门槛，非论文设定。

上述UNLIKELY仅针对**当前已测micro布局、原mask、包含梯度审计hook/同步/重放/日志的实现**。尤其Q17-F正常更新79.0944秒，其中Backward约70.7943秒，包含逐参数梯度检查。未经实测的精简正式Trainer可能不同；本轮没有把诊断开销擅自减掉，也没有证明所有可能的原生实现必然超96小时。未来如批准修正监督边界，其语义、数值轨迹和成本须另验。本次结果不是baseline选择授权，更不是推荐质量排名。

机器结果：`evidence/execution_results.json`、`evidence/budget.json`。GPU原始rank报告中的`status=PASS`只对应该脚本的计算阶段断言；不能覆盖这里的`PARTIAL_BOUNDARY`。

## 2. 原件、分支、固定资产与复用边界

- 实际规则入口：`/home/addaswsw/recommend/`，实际Git根：`/home/addaswsw/recommend-worktrees/lc-rec`。已真实读取同目录三份规范。只使用`woobowen/recommend:wbw-lc-rec`，起点`1cd37ec681bdb3534108aaaee554dbaa9158a12f`；未改main、GRID或其他历史分支，未force push/reset/clean。原有两个未跟踪8B tokenizer文件保留。
- 官方LC-Rec源码：`RUCAIBox/LC-Rec@e6f1b3b3f8b4fee55a49a7b3dfeba29dbfcf0669`，本地`lc_rec/vendor/original`不改。直接核对`lora_finetune.py`、`finetune.py`、`utils.py`、`data.py`、`collator.py`、`prompt.py`、`run.sh`、`test.py`、`test_ddp.py`与`evaluate.py`；继承原件hash回执并针对新适配回归。
- 六任务资产/固定原始记录与prompt realization复用`cceb550a44835b359c21596507f31dc7765da2d4`的已审核工作；没有重新生成SID、重新下载数据或重做整套旧审计。
- Qwen资产/native模型/PEFT tied-weight工程经验复用`1cd37ec681bdb3534108aaaee554dbaa9158a12f`。该旧Qwen预跑实际是LETTER918/单SeqRec/temperature0.8/Int8 LoRA/global64，不作为本轮LC930、六任务、BF16/global128的运行证据。
- 数据目录：`/data/bwwu/lc-rec/assets/data/Instruments`。LC索引SHA-256：`b94d16398a178220a117abda5effd5508069beaccc2b8110eab25ae50f2af600`；辅助用户文件SHA：`8fe65b5de6222e665d12052c8771f8a314813323b8470f54f2ca2920bf7af5b6`。所有实际数据文件hash见CPU/GPU报告。

| 官方Base模型 | 固定revision | 已审核权重SHA-256 |
|---|---|---|
| Qwen/Qwen3-0.6B-Base | da87bfb608c14b7cf20ba1ce41287e8de496c0cd | cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba |
| Qwen/Qwen3-1.7B-Base | ea980cb0a6c2ae4b936e82123acc929f1cec04c1 | 6df85b39330e5a425ee36253d0f894e4387e4f0a15b9c53cb467d668e6b3a841 |

资产正式入口为`/data/bwwu/lc-rec/qwen-preflight-20261110/models/<模型名>/<完整revision>/`。复用原完整哈希回执，加当前size/inode/mtime及小文件hash核查，没有重新下载或反复重哈希未变化的大权重。见`evidence/preflight.json`。使用Qwen3ForCausalLM原生Base、原架构；没有ChatML/system/thinking、Qwen3.5、量化或第三方模型。保存目录附带的官方chat template文件从未调用。

## 3. LC930 tokenizer与六任务真实长度

两款Qwen tokenizer对**完全相同的221225条原始记录及固定prompt realization**产生逐数组一致的结果；两个`workload.npz`字节SHA均为`803bfe8cc13359fdfff9fff1382b3d587206a2ba150148ba8cb820edd9bbcea4`。记录/prompt原文hash逐条对上旧realization，不按token长度换样本。

词表：原tokenizer151669，新增**930**个按原LC-Rec排序的SID字符串，扩容后152599。原物理embedding/head151936行，扩容后152599行，因此267个新增token复用了原padding行，实际增加663物理行；不能把930 token说成930新增物理行。所有9922物品均逐项验证四个SID token及字符串往返，不拆分；新增ID顺序、合法label范围、保存重载全部一致。

原资产本身只有9897种canonical SID，23个碰撞组、25个重复增量。保留原映射与全部碰撞成员，不声称item→SID为单射，不“清洗”资产。输入embedding与lm_head扩容前后、PEFT active module、ZeRO后以及重载后均共享同一个Parameter/storage。

以下为本轮Qwen**完整序列含EOS**的真实统计，不是历史LLaMA长度。每任务prompt sample均1；data sample为`0,0,0,20000,0,0`；history20/max2048。

| 任务 | 记录数 | 均值 | p10 | p50 | p90 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| SeqRec | 131837 | 86.0597 | 58 | 75 | 138 | 180 | 184 |
| Item2Index | 9922 | 132.7287 | 51 | 89 | 283 | 480 | 1511 |
| Index2Item | 9922 | 123.4612 | 48 | 78 | 270 | 473.79 | 1513 |
| FusionSeqRec | 20000 | 160.9412 | 74 | 120 | 328.10 | 516 | 908 |
| ItemSearch | 24772 | 126.0729 | 70 | 108 | 214 | 308 | 550 |
| PreferenceObtain | 24772 | 142.4178 | 90 | 129 | 210 | 293 | 543 |
| 合计 | 221225 | 107.3913 | 58 | 86 | 178 | 383 | 1513 |

输入/独立回答/shifted有效label分别统计，以下每格均为`p10/p50/p90/p99/max`：

| 任务 | 输入tokens | 独立回答tokens，不含EOS | shifted有效label，含真实EOS |
|---|---|---|---|
| SeqRec | 53/70/133/175/179 | 4/4/4/4/4 | 5/5/5/5/5 |
| Item2Index | 46/84/278/475/1506 | 4/4/4/4/4 | 5/5/5/5/5 |
| Index2Item | 36/38/47/47/47 | 10/38/229/433/1466 | 11/38/229.9/434/1466 |
| FusionSeqRec | 56/82/180/478/699 | 4/15/158/391/843 | 5/15/159/392/844 |
| ItemSearch | 65/103/209/303/545 | 4/4/4/4/4 | 5/5/5/5/5 |
| PreferenceObtain | 52/66/107/164/178 | 31/50/125/187/426 | 32/50/125/187/426 |

输入/完整序列2048截断均0，零有效shifted label为0，最长真实历史20。混合global128任务数固定`76,6,6,12,14,14`；LoRA真实postwarm normal update1/2的padding依次为22.8995%/20.4237%（micro2），full只有update1作steady计时；micro1为0。全221225记录四次固定随机布局的成本场景micro2约21.55–21.61% padding；这不是正式Trainer实际sampler顺序重放。更完整的均值/最小值/计数/分位/任务内padding在两CPU audit中。

## 4. 不能宣称通过的监督语义

官方Collator分别tokenize prompt和prompt+answer+EOS，再按prompt token长度mask。本轮保留该算法与原始自然语言字节；仅在Qwen PAD==EOS时改用attention_mask识别padding，避免错误屏蔽真实EOS。没有temperature0.8，实际为temperature1的FP32 causal shifted CE、ignore_index=-100。

但Qwen会跨`Response:`末尾冒号和回答首字符进行BPE merge：**29574/221225条prompt token不是full token的严格前缀**，Index2Item2993、Fusion4537、Preference22044。部分首字/首词被原mask忽略，例如`index2item:9` token74045覆盖最后冒号与整个`The`，不是只有一个字符差别。

原始空回答505（Index2Item133/Fusion372）；EOS-only监督979，其中474原本非空回答被mask成只监督EOS（183/291）。原算法与本轮CE相互一致，不等于完整回答文本监督正确。独立CE参考最大绝对误差：Q06-L/Q06-F各`9.536743e-7`，Q17-L`4.768372e-7`，Q17-F`1.907349e-6`；这只验证现有labels下的CE，不修复labels语义。

本轮没有offset改mask、分段重新tokenize、加分隔符、删记录或改prompt。这些会改变实际监督/输入，属于下一轮需要确认的重要方法变化。原始GPU脚本中“retained supervision”的自描述仅指保留原mask算法，不能解释为每条回答文本监督完整。独立判断及实际offset例证见`review/mask_boundary_review.json`。

辅助数据仍读取原资产：训练12233、测试7473错配仍在；Preference辅助标签生成上下文未完整证明。全量长度/负载分析保留这些数据，技术更新仅用当前训练期配对可核样本。没有删错配样本制造正式训练集、重生成标签或引入测试目标。**有界计算不被阻止，可信正式六任务质量复现仍BLOCKED。**

## 5. 实际训练范围与数值/工程差异

| 路线 | 扩容后base唯一参数 | 实际trainable唯一参数 | trainable范围 |
|---|---:|---:|---|
| Q06-L | 596728832 | 161307648 | 5046272 LoRA +156261376完整active tied矩阵 |
| Q06-F | 596728832 | 596728832 | 完整模型，310唯一参数张量 |
| Q17-L | 1721932800 | 321239040 | 8716288 LoRA +312522752完整active tied矩阵 |
| Q17-F | 1721932800 | 1721932800 | 完整模型，310唯一参数张量 |

LoRA是196个原生q/k/v/o/gate/down/up projection，r8/alpha32/dropout0.05，392个A/B张量+一个完整共享embedding/head=393个trainable张量；**不是只训练930行，也不是仅几个百万LoRA参数**。PEFT保留frozen original wrapper；物理唯一参数总量Q06为758036480、Q17为2043171840。`ensure_weight_tying=True`，optimizer对active共享Parameter只收一次，frozen original不进入optimizer且完整hash保持不变。

实际均BF16基础权重/参数、原生Qwen3、SDPA、non-reentrant gradient checkpointing、GPU AdamW/FP32 master和moments、ZeRO-2，未Int8、4-bit或CPU offload。LR5e-5、weight decay0.01、cosine、warmup0.01、seed42/global128。官方parser默认LR2e-5；本轮5e-5来自作者全参Instruments脚本及用户明确批准的技术候选，不是LoRA论文最优LR。

相对原件改动位于新目录`common.py/cpu_audit.py/probe.py`等，不静默修改vendor：native Qwen/930扩容/tied PEFT、PAD/EOS适配、按实际Qwen token定位生成分隔符、BF16加载、现代DeepSpeed接口、显式有限engine loop和证据采集。原LLaMA两个独立矩阵不机械复制到Qwen。BF16初始cast/新增行初始化、SDPA/重算、模型本身、现代引擎均可能改变数值轨迹；未声称与原LLaMA逐bit等价。

有限loop显式保留原LC-Rec每micro shifted-token CE mean，再按accum和rank平均；不用新Trainer默认accumulated-window token总归一化。**global128相同不意味着不同micro有相同任务/token梯度权重**，Q17-F的micro1与其他micro2因此不是同一数值轨迹；实际各任务权重计算在`budget.json:micro_loss_weighting`。正式epoch短尾的Trainer+DeepSpeed归一化尚未跑完整验证。

原全参数论文方法换Qwen是下游LLM替换版本；Qwen LoRA又改变训练范围。官方六任务LoRA确有源码，但本轮四路线均没有严格绑定某篇论文完整LoRA指标，也未比较推荐精度或训练范围优劣。Q06-F本次比LoRA快，不能解释为全参数总体更好。

## 6. 更新、梯度、显存、长尾与单任务诊断

| 路线 | 实际双卡micro×accum | 更新MAX-rank秒，含LR0 | normal联合input tokens/s | training reserved峰值GiB | guard进程树采样RSS峰值GiB |
|---|---|---|---:|---:|---:|
| Q06-L | 2×32 | 24.2803 /24.2625 /20.5747 /21.6280(tail) | 592.95 | 9.4883 | 8.4013 |
| Q06-F | 2×32 | 21.6634 /21.1979 /21.5362(tail) | 662.33 | 12.7891 | 8.9611 |
| Q17-L | 2×32 | 24.8309 /24.1651 /22.5516 /23.0726(tail) | 569.09 | 13.2207 | 14.6003 |
| Q17-F | 1×64 | 79.7349 /79.0944 /78.8907(tail) | 177.51 | 22.6289 | 12.8928 |

两LoRA各4次更新512次compute presentations，两个全参各3次384次；每路线原始记录union262，不能说512/384独立记录。两rank参数/模型/tokenizer身份相同，最终全部trainable完整tensor hash同步；LoRA393个与全参310个唯一参数均检查optimizer membership及有限梯度，没有NaN/Inf或意外冻结。真实SID输入位置和输出logit反传均非零，grad checkpointing没有切断梯度；共享参数未重复优化。

第1次LR0只用于状态初始化，不计真实变化。后续LR为`5e-5×step/70`，LoRA前三次非零LR为`7.142857e-7/1.428571e-6/2.142857e-6`；全参两次。LoRA_B零初始化使第一次A梯度为0是预期，之后196个A全部有非零梯度和完整矩阵变化，196个B亦变化；full路线代表性Transformer及SID矩阵真实变化。BF16下不要求每个数值元素都能在极小LR更新后跨过ULP。

每路线先做6个实际max样本刚加载no-grad Forward，再进行已初始化Adam状态常驻、正常ZeRO通信/梯度累积的global128 tail完整更新。六任务实际最大长度`184/1511/1513/908/550/543`全部进入该tail更新；不是只做去hook的容量测试。Q17-F allocated仍约20.93GiB、reserved22.63GiB，相对23.52GiB物理CUDA容量余量有限，短预跑通过不保证长期无碎片风险。

micro2在Q06-L/F、Q17-L真实完整路径可用；micro1在Q17-F真实完整路径可用。其他路线micro1只做了下述无通信诊断；micro4/8均NOT TESTED，不声称不能跑或最快布局已找出。静态状态预算+真实1513长尾决定安全起点，没有故意从micro8逐级OOM。原LLaMA或旧LETTER的OOM不冒充本轮测量。本轮GPU没有OOM/失败。

另有每路线六任务×short/p50/p90/max=24次诊断：Adam状态常驻、去除梯度/activation审计及ZeRO hooks、micro1、无optimizer更新，各卡重复同一条记录，**distinct=1、presentations=2**，全部通过且参数不变。这些只能看计算/容量，不是通信或联合吞吐验收，不能拿较快的无hook F/B替代完整更新成本。示例：各任务max样本两rank较慢的Forward+Backward秒：

| 任务/max tokens | Q06-L | Q06-F | Q17-L | Q17-F |
|---|---:|---:|---:|---:|
| SeqRec/184 | 0.4120 | 0.2315 | 0.3482 | 0.2602 |
| Item2Index/1511 | 0.4224 | 0.2230 | 0.4009 | 0.2529 |
| Index2Item/1513 | 0.4203 | 0.2315 | 0.3974 | 0.2527 |
| Fusion/908 | 0.4173 | 0.2251 | 0.3911 | 0.2559 |
| ItemSearch/550 | 0.4236 | 0.2339 | 0.3920 | 0.2592 |
| Preference/543 | 0.4180 | 0.2257 | 0.3912 | 0.2599 |

每个quantile的CPU collator/H2D/F/B/allocated/reserved/RSS/有效监督tokens均有双rank原始记录；`execution_results.json`给出不混淆distinct/presentations的派生tokens/s。完整更新另保存engine.step时间。NCCL成本包括在Backward/step中，但**未独立profile通信kernel时间**；各phase的MAX也不能直接相加当墙钟。guard有全作业NVML采样，但初始化前未连续reset CUDA peak，故不冒称表中峰值为整个生命周期精确极值。

## 7. 保存重载与训练状态恢复

| 路线 | 模型/adapter保存秒 | 模型转CPU秒 | fresh模型/adapter重载秒 | 完整训练状态 |
|---|---:|---:|---:|---|
| Q06-L | 0.4826 | 1.5286 | 16.4902 | NOT TESTED |
| Q06-F | 1.6604 | 1.2009 | 6.5727 | 同一ZeRO engine真实落盘恢复PASS |
| Q17-L | 0.8155 | 4.8876 | 34.0100 | NOT TESTED |
| Q17-F | 6.1994 | 3.9460 | 20.8200 | NOT TESTED，完整状态静态约24.11GB |

每路线均真正保存模型/adapter+tokenizer，重建原生模型/PEFT，完整key/shape/dtype/值字节hash一致、930ID顺序一致、active tied共享恢复。LoRA不是adapter-only tensor子集糊弄，完整active embedding/head也核对。没有用`strict=False`掩盖缺失参数；key alias允许项按Qwen原生共享语义逐项验证。

Q06-F一次完整状态共7文件、8354402359字节；实际save18.4632秒、MAX-rank load19.5882秒。故意扰动Adam moment和RNG后恢复，模型/optimizer/scheduler/global step/RNG/ZeRO状态与保存时精确一致。**是在原engine内load，不是新进程断线恢复；实际Trainer best-model restore也NOT TESTED。** 其他路线仅模型/adapter重载PASS，不冒称训练恢复PASS。

全状态静态payload公式`2×physical_unique +12×trainable`（BF16模型+FP32 master及两Adam moments）：Q06-L3.4518GB、Q06-F8.3542GB、Q17-L7.9412GB、Q17-F24.1071GB，另有序列化/布局/共享/重复key差异。其他三路I/O成本用Q06-F实测按字节线性外推，明确是proxy而非实测。假设保留4epoch状态+模型及最终模型的持久化场景约15.50/39.47/35.06/113.73GB，**本轮没有实际写这些正式checkpoint**，不受本轮20GiB资产预算冒充授权。

## 8. 原17模板beam20接口

四路线各固定3个真实测试用户（短p10/中p50/长边界max历史）×17模板，共51个user-template case、1020返回序列；不是51个独立用户。测试行索引/用户SHA-256/真实prompt编码见samples和generation cases，可复核而不披露原用户ID。使用原SeqRec prompt、beam20/return20、max_new_tokens10、四位置合法token集合后EOS；**非严格trie**。

Qwen适配按实际编码后的Response分隔符定位生成起点；保持原输出`split('Response:')[-1].strip().replace(' ','')`、原score排序和evaluator，`filter_items=False`，不新增去重。所有case回调轨迹、EOS位置、20个token序列/文本/分数、合法性和峰值均保存；全部位置合法/正常EOS，四路线各**canonical0/1020**。未充分训练且逐位置约束不保证合法组合，这不是正式推荐效果判败，更未报告正式Recall/NDCG。

原`test_ddp.py`在模板循环前构建一次缓存SID callback，而GPU诊断逐case重建catalog，导致残余墙钟不是正式每用户CPU成本。round2保留原GPU源码不改，另用CPU-only原Dataset/TestCollator/decode/evaluate对四路线所有51 case做1次warmup+3次重复；原始输入、20返回序列及evaluator结果逐项精确匹配，无模型/CUDA加载。callback每路线约1秒一次，正常每用户CPU约0.002秒；不得把旧诊断约1秒/用户外推成正式66–77小时。修正前估算保留在`evidence/cost_before_round2_diagnostic_harness.json`。

## 9. 一个seed成本拆分与外推方法

实际安装的Trainer/Accelerate/DeepSpeed sampler协议CPU探针：每epoch**1729** scheduled update，共**6916**，warmup70；不是历史约6912。micro2四epoch实际呈现884912，micro1为884904（名义原始884900），最后global分别44/42；micro4/8协议另有884928呈现。完整global128预跑没有证明短尾更新的数值等价。

Validation按原SeqRec每用户2个采样模板，49544记录/次、4次，不是17模板；以真实有界batch1的F/collator/H2D均值/最大值乘`49544×4/2`外推，正式validation batch吞吐及选最优未实际运行。Testing按`24772×17`，逐模板3个真实用户generate均值/最大值乘`24772/2`，双卡分母只除一次，不重复两rank同一份gather记录。

训练仅用驻留optimizer的normal mixed非零更新校准，tail更新只作容量证据；用每micro的常数/线性padded slots/长度平方非负拟合，加全221225真实长度的4次确定性随机混合布局。中心用实测rank-skew比例与平均非F/B成本；保守包络取不归一化/残差/常数/线性/平方敏感性最大项+最坏非F/B，再×1.25。Backward内的ZeRO通信与梯度审计hook已经进入F/B拟合；collator/H2D/step及其余非F/B成本由完整wall残差计入，两部分合计一次，不再另加通信。LoRA仅2个、full仅1个steady global更新，均不是长期统计置信区间。

单位小时，`中心 / 保守`：

| 成本组成 | Q06-L | Q06-F | Q17-L | Q17-F |
|---|---:|---:|---:|---:|
| 训练 | 43.0533 /68.5569 | 40.6856 /56.6733 | 44.8977 /71.4630 | 151.8812 /205.6126 |
| Validation | 1.8538 /2.8292 | 1.0638 /1.6339 | 2.3086 /2.9145 | 1.5488 /2.2113 |
| 17模板beam20纯生成 | 21.4745 /30.0947 | 14.4573 /18.7331 | 21.5314 /28.9988 | 15.8697 /24.1649 |
| 测试CPU处理proxy | 0.1139 /0.1965 | 0.1197 /0.2378 | 0.1146 /0.2006 | 0.1145 /0.2005 |
| Checkpoint | 0.0113 /0.0141 | 0.0245 /0.0306 | 0.0274 /0.0343 | 0.0733 /0.0916 |
| Best restore proxy | 0.0046 /0.0057 | 0.0054 /0.0068 | 0.0094 /0.0118 | 0.0157 /0.0196 |
| 初始化及明确CPU准备预留 | 0.0961 /0.1826 | 0.0950 /0.1812 | 0.1001 /0.1877 | 0.1156 /0.2070 |
| 运行波动/未知项规划余量 | 3.4970 /10.6880 | 2.9892 /8.2497 | 3.6161 /10.8811 | 8.6476 /23.7507 |
| 总小时 | **70.1046 /112.5678** | **59.4405 /85.7465** | **72.6054 /114.6917** | **178.2663 /256.2582** |
| 总天 | 2.9210 /4.6903 | 2.4767 /3.5728 | 3.0252 /4.7788 | 7.4278 /10.6774 |
| 距96h余量，小时 | +25.8954 /-16.5678 | +36.5595 /+10.2535 | +23.3946 /-18.6917 | -82.2663 /-160.2582 |

完整JSON保留未四舍五入值、原始观测、脚本/所有输入SHA、NNLS系数和敏感性，重新运行`estimate_budget.py`可复算。中心为量化subtotal额外5%+10min，保守为已放大的subtotal再10%+30min；初始化另有5/10min CPU准备预留。CPU数据/回调缓存/模型/ZeRO初始化明确区分实测和规划项，没有把所有guard时间误叫模型加载。

**尚未单独量化/只由规划余量覆盖**：正式数据准备/未来epoch随机realization、精确sampler顺序、全量Dataloader/pinned-memory H2D与输出D2H、GPU scalar score排序同步差异、完整结果gather/barrier/聚合/持久化fsync、实际Trainer最优模型恢复/新进程恢复、全epoch短尾归一化、长时热/碎片/CPU/I/O波动与停机、去审计hook后的正式kernel吞吐。其他路线full-state I/O为线性proxy，非完整恢复验证；17模板仅3用户的方差覆盖有限。因此`MODERATE_CONDITIONAL`只指当前有界工程情景，不是96小时保证或统计置信区间。

## 10. 失败、修正与真实非作者审核

round0有两个**CPU**失败：protocol探针给Trainer错误Tensor容器；CPU adapter错误断言原SID无碰撞。round1分别改dict容器和按原资产验证真实碰撞，不改SID/方法，后继CPU作业通过。另有一次本地错误完整SHA在提交前被拒，未创建服务器job；随后显式使用`git rev-parse --verify REV^{commit}`硬化。一次未终态collect被控制器拒绝，未重启GPU或更改原job。

round2仅修正上述callback初始化成本归因并补CPU缓存实测，不修改GPU实现或增加更新。**整个任务已用2轮集中修正，round3未使用**；不是每路线各3轮。全部原始失败/错误估算/后继回执保留：`evidence/corrections.json`、`evidence/local_client_hardening.json`、`evidence/jobs/`。GPU四作业全部第一次执行成功。

真实非作者`/root/source_reviewer`未实现common/CPU适配/GPU probe/成本计算器，直接读取官方原件、固定源码/config、两rank原始JSON、微步/梯度/参数hash/生成序列与manifest，并以独立标准库NPY解析重数221225条mask边界，独立复算时间与分母。其protocol CPU探针由其作者身份单列，不冒称该小探针独立审核自身；新Qwen适配与实际GPU/cost均由非作者审。

审核入口：`review/source_review.json`、`review/prelaunch_review.json`、`review/mask_boundary_review.json`、`review/q06_L_runtime_review.json`、`review/q06_F_runtime_review.json`、`review/q17_L_runtime_review.json`、`review/q17_F_runtime_review.json`、`review/round2_cost_review.json`、`review/final_cost_review.json`。最终整合/发布/ZIP的真实回读结论以`review/final_delivery_review.json`和ZIP包外审核回执为准，不能用静态脚本审查替代实际发布/ZIP校验。

## 11. 固定源码、真实作业与服务器映射

**所有GPU RUN_COMMIT：`1f3a0bafeebb3a2fa8604a052727ddb3d1eadcff`**。
`probe.py` SHA-256：`8e72c75a961d3ef308c3b88359221cda94632777ead8c891caf89b5979463f3a`。
GPU后没有修改该probe、common或resource_guard。round2只新增CPU replay/修正成本与最终汇总/交付脚本、结果和说明，逐文件diff在最终映射中列出。

| job_id | 内容 | 固定源码commit |
|---|---|---|
| lc-qwen6-20261111-q06-l-r0 | Q06-L，4更新 | 1f3a0bafeebb3a2fa8604a052727ddb3d1eadcff |
| lc-qwen6-20261111-q06-f-r0 | Q06-F，3更新/一次状态恢复 | 1f3a0bafeebb3a2fa8604a052727ddb3d1eadcff |
| lc-qwen6-20261111-q17-l-r0 | Q17-L，4更新 | 1f3a0bafeebb3a2fa8604a052727ddb3d1eadcff |
| lc-qwen6-20261111-q17-f-r0 | Q17-F，3更新 | 1f3a0bafeebb3a2fa8604a052727ddb3d1eadcff |
| lc-qwen6-20261111-postprocess-r2 | CPU缓存测试/原始输出精确重放 | 820e2f30a51c507f65d5f3068eab018868410c8d |

成本计算源码也固定`820e2f30a51c507f65d5f3068eab018868410c8d`，实际脚本SHA/argv/解释器/NumPy/输入hash见`budget.json`。CPU全量token审计为`97ae3fad2cda9e9df5576db36ef6dee323500585`；更早失败commit均在各request中。

服务器固定源码export：`/home/bwwu/lc-rec/qwen-six-task-20261111/releases/<RUN_COMMIT>/`，无`.git`，**不是checkout**；`RUN_VERSION.json`、deployment manifest与最终readback按路径逐字节映射Git对象。所有job保存在`/home/bwwu/grid-audits/sid-extensions-p0/jobs/<job_id>/`，request含完整argv/cwd/环境/config身份，output含两rank/资源/产物manifest/退出状态。复用已验收独立runner，脱离SSH/Codex生命周期；无盲目重跑。

最终DELIVERY_COMMIT不能写入自身文件造成自引用：以最终答复及ZIP内`final_receipts/{github_readback,run_delivery_mapping,server_readback}.json`为准。发布只normal push到`wbw-lc-rec`，实际回读分支HEAD与固定关键源码、四配置、成本、审核入口；本地commit或push退出0不单独算发布完成。

## 12. 资源、恢复入口、ZIP与停止边界

四GPU作业合计**39.6626 GPU-device分钟/240上限**：Q06共15.7566分钟、Q17共23.9060分钟，各<120；最长单job7.1214墙钟分钟/45上限。更新共14，LoRA各4/full各3。包括import/初始化/失败保护窗口/保存/推理所有GPU生命周期，未按仅kernel时间缩小分母。guard最小MemAvailable225.1215GiB，超过40GiB门槛；任务进程树采样RSS最大14.6003GiB。已记录的guard检查中各FS均满足30GiB及10%更严格门槛，原始资源采样及最终实测新文件allocated/logical大小见server回执。

全部本任务job终态且cleanup_verified、进程组空，最终还需真实回查，无任务残留进程才交付；恢复入口就是以上固定job目录，只读查询，不自动重启。没有安装系统包/语言包/工具链，没有修改共享环境、驱动、SSH/VPN/代理或全局配置。任务自产模型/adapter/一次状态checkpoint保留服务器，未授权清理，未触碰旧模型/旧数据/同学作业。

单份轻量ZIP由`scripts/build_review_zip.py`从固定DELIVERY树生成：只包括新源码/config/轻量原始证据、必要原件引用与最小diff、非作者审核、外部GitHub/server回执和逐成员SHA清单；不含模型、数据全集、optimizer state或凭据。实际CRC、完整解压、逐成员size/hash、重复/危险路径/symlink/秘密扫描、Windows原生文件hash验证后才交付；ZIP hash在包外，最终真实完整WSL/Windows路径、字节数、SHA-256、成员数见最终答复和`.zip.receipt.json`。

**GPT_UPLOAD_REQUIRED: YES**。将单份ZIP交网页GPT二次审核，再与学长讨论，不据本次技术探针直接采用baseline。当前授权到此停止：未运行正式4epoch、全量测试、五seed、新数据集、LETTER上游、Qwen3.5、Shuffle或新Loss。监督边界修订、正式数据可信性、精简正式吞吐和baseline选择都留待新明确授权。
