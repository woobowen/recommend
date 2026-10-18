# Sports A2 正式评价、注意力与CPU统计真实非作者审核

`A2_EVALUATION_ATTENTION_STATISTICS_NONAUTHOR_REVIEW_PASS`

实际审核代理：`/root/toys_result_audit`。未编写A2量化修复、推荐模型、训练、评价、注意力或统计生产实现。本轮直接审阅实际终态原件，并执行已准备的独立CPU脚本。此PASS覆盖A2正式输出与其身份依赖；完整量化/训练验收、总预算、最终GitHub/ZIP仍由主控和相应真实审核者另行收口，不将本报告冒充全任务FINAL_INTERNAL_REVIEW_PASS。

## 固定身份与实际执行

- 科研/工程部署source commit：`85b77f5e2a5b497fdc1f620649ba488fc783a245`。snapshot：`22286966cd544773f21b96e4185d56c66afd7183fdc4e8888716d8ea1d77acb2`，实际根`/home/bwwu/grid-audits/sid-extensions-p0/grid-a-launch-20261017/snapshots/22286966cd544773`。20份实际冻结源码与manifest逐字节匹配；没有把旧b443科研原件、共享调度6daf或新发布元数据提交混成一个版本。
- 新T5/R-Kmeans SID：`06cfa6ba0f64d6c68e0e94c4111edd51d086796e8cdb474ac3fcd424a4a2adf0`，原路径`/home/bwwu/grid-audits/sid-extensions-p0/jobs/ga-20261018-sportsa2-quantize-fd-recovery-v1/output/semantic_ids.pt`，588597字节，`[4,18357] int64`。本审核用限制型数值反序列化器独立读取小SID载荷，没有torch/GPU或模型调用；完整SID唯一且全量目标均合法。
- 全部评价/注意力唯一best：step6900，SHA`1287c3f60404726dbda165d5242ba17d40ecaf6520ded69d6e3165170586513d`，161660326字节。原路径`/home/bwwu/grid-audits/sid-extensions-p0/jobs/ga-20261018-sportsa2-train-fd-recovery-v1/output/checkpoints/checkpoint_epoch=000_step=006900.ckpt`。每实际job严格加载178 state tensors，missing/unexpected均空、所有keys/shapes/values完全相等、embedded SID exact，完整双表不共享，参数13259136。没有下载或重新加载大checkpoint；独立训练审核负责直接载荷验收。
- 原训练依赖记录显示完整终态、final step7900、best step6900、79次完整验证，终止原因为`patience10_complete_validation_early_stop`。本报告只核准入依赖与一致性，不代签训练细节审核。
- 三组评价与三组注意力、stats的新job均使用`ga-20261018-sportsa2-*-fd-recovery-v1`独立身份。7份实际request与accepted指纹精确一致；226个runner status artifact hash和24份统计input file hash相符；请求CWD、实际stage argv、capture、SID/best/配置来源均一一对应。原失败job未修改。

## 完整正式评价

| 条件 | testing用户 | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 | 非法候选 | 重复候选 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Normal | 35598 | 0.02250126376748085 | 0.014671024866402149 | 0.034215405583381653 | 0.018478265032172203 | 35 | 0 |
| Shuffle3 | 35598 | 0.020282037556171417 | 0.012955863028764725 | 0.030254509299993515 | 0.016185346990823746 | 25 | 0 |
| Shuffle4 | 35598 | 0.019635934382677078 | 0.012565478682518005 | 0.030928703024983406 | 0.016218414530158043 | 39 | 0 |

表内是原evaluator正式原值，不用独立float64值替换。三条件各35598个唯一用户，sample_key/target multiset与新SID条件下的实际testing索引精确相等，共106794行预测，无遗漏或重复用户。逐用户指标独立复算，汇总对原FP32值最大差`1.397e-09`。Normal/Shuffle3分数无并列；Shuffle4有1行非target并列，不影响命中/目标rank。非法SID保留在原十候选，不过滤、补齐、改分或重排后再评价。

每条件4450个真实CUDA test batches，raw batch8、workers8、120 SID输入位、32-true、seed42、K10、prefix check=False。三个有效配置除预定Shuffle flags和输出目录外完全相同。原loss/generate encoder各4450次：Normal的Shuffle计数0，Shuffle3/4各8900次；保存的实际排列为原rand→argsort方法，padding保持原槽位，Shuffle3第四位不变。源码确认先层级offset与embedding lookup，再在每历史occurrence内部重排完整向量，不重排整数SID或物品时间轴，不改target/decoder顺序。

## 逐用户严格配对

| 与本模型Normal比较 | Top1变化 | Top10顺序变化 | Top10集合变化 | Hit@5丢失/获得 | Hit@10丢失/获得 | 平均Top10重叠 |
|---|---:|---:|---:|---:|---:|---:|
| Shuffle3 | 17847 (50.1348%) | 35530 | 34392 | 231/152 | 326/185 | 5.973594022 |
| Shuffle4 | 17012 (47.7892%) | 35562 | 34516 | 236/134 | 314/197 | 6.175684027 |

每组配对35598人，原target完全一致。`a2-normal-shuffle3-paired.jsonl.gz`和`a2-normal-shuffle4-paired.jsonl.gz`保留逐人变化与排名；本文件rank=0表示Top10未命中，完整rank/Hit转移计数保留于JSON。没有按照test结果挑模型、seed或样本。

## fixed32 注意力与描述统计

cohort SHA为`0a2bbfbd62ab5bbd82b3032330e1f0f411af015d674fccc6b5f23c0ddacafe01`。明确来自**evaluation/validation split**，不是testing；按实际文件/行顺序在看模型输出前固定首32 distinct users，三条件同一cohort。相同user key在两个split不代表相同history/target，未把fixed32 Top1强行等同testing Top1。

每条件4批×8人、4份NPZ＋4份record JSON＋32 PNG＋4份plot receipt；三条件合计12 NPZ/96 PNG。raw最后层6heads、4个autoregressive query、150列encoder全部保留。独立从原beam唯一prefix重建所有parent_indices，并从最终child0逆向追踪完整Top1祖先链；local-first每步parent0单独核验，两个路径的head张量、head mean、最终SID和scores逐项精确一致。Normal/Shuffle3/Shuffle4分别26/27/25个用户的Top1祖链使用过非零parent，未把局部首beam误作最终Top1。

同cohort的实际input SID、mask、item轴和120个SID位置、150个encoder位置、lookup offset、可见/隐藏组及实际shuffle source permutation全部核对。每批observer OFF/ON的all-K SID/scores完全一致、CPU/CUDA RNG结束状态相同、参数hash未变、hooks/方法正常及异常恢复均通过。四批参数hash在各条件内恒定。

96张PNG全部解码并核SHA/大小/来源/2880×1280尺寸/原始共同色阶；另外从实际黑色spine定位双panel，对全部**115200个热力格、14400个mask格**的cell-center RGB与raw NPZ/原YlOrRd LUT独立比较，全部通过（量化差不超过1/255）。未重新渲染或编辑原图，也未作逐行伸缩。对三条件固定第0批第0样本额外实际看图，四行、全150列与mask显示一致。

| 条件 | 路径 | mean Cosine | mean Pearson | mean TV | mean JS(nats) |
|---|---|---:|---:|---:|---:|
| Normal | final_top1_cross | 0.774638107 | 0.163903570 | 0.229967977 | 0.083350841 |
| Normal | local_first_cross | 0.757683830 | 0.064020706 | 0.252022767 | 0.088458694 |
| Shuffle3 | final_top1_cross | 0.674247047 | 0.232591290 | 0.289046436 | 0.112023746 |
| Shuffle3 | local_first_cross | 0.671702784 | 0.154002761 | 0.307894037 | 0.111082820 |
| Shuffle4 | final_top1_cross | 0.709893886 | 0.168490958 | 0.273992690 | 0.104047720 |
| Shuffle4 | local_first_cross | 0.711061788 | 0.104808657 | 0.283527853 | 0.102175607 |

每条件32人×6个query pair×2路径=384行，共1152行独立复算；逐条及min/median/mean/max均匹配原统计，最大差`7.772e-16`。全部masked mass=0，raw单head行和最大误差2.384185791e-7。Cosine/Pearson使用可见原值，TV/JS只在概率计算时除以可见行和，JS为自然对数，包含可见SEP，不改原注意力矩阵。每路径least-three按原声明的六对平均Cosine升序/sample_key平局规则独立复算，六个选择均相符；全32样本和全pairs保留，不只保留最漂亮或最有利案例。

## 作业终态、原件与限制

- 三eval、三attention及stats原状态均SUCCEEDED/exit0/cleanup verified、worker_may_be_live=false、最后owned-group观察为空。它们的输出内容另已验收，没有把exit0当科研证明。当前全服务器进程、控制器总终态、预算与期限由主控实际场景核验另行负责。
- 本轮终态定向收集270成员、95430018原始字节，压缩34834571字节；整体SHA`646784982c92ca2c35a0006662dbee5f7c859be34eb216adcf794b93f4e8b844`，全部成员SHA/大小与安全路径通过。不是整job export，没有复制模型checkpoint或TFRecord。先前preparation PENDING/失败0字节体积门回执保持原样。
- 本审核没有训练、推荐generation、原evaluator调用、GPU模型调用、远端写入、Git写入或新依赖；实际CPU统计来自保存原件。正式输出审核首次即通过，无研究修正或重跑。
- 本审核不代替量化和完整训练的真实非作者审核；checkpoint严格加载从实际每job原记录和固定源码核对，未在本子审核重新加载大checkpoint。
- A2属于T5/R-Kmeans SID下的新双表训练方法复现，不是找回学长原始Sports权重。A1与A2有上游SID与独立scratch训练条件差异，不能将差异归于单一因素。Toys producer非唯一披露仍在跨域交付中保留。
- 单seed42、fixed32 evaluation cohort、各条件内部行间描述统计，不构成数学严格置换不变性、跨seed稳定性或decoder忽略目标prefix的证明。Shuffle导致大量逐用户推荐变化和实际指标下降，均按原结果保留。

## 审核入口

- `A2_OUTPUTS_NONAUTHOR_REVIEW.json`：全量指标、逐用户配对、每条attention统计及身份。
- `a2-final-output-binding-checks.json`：request接受指纹、artifact/统计输入hash、least-three与scores检查。
- `a2-png-pixel-consistency.json`：全部PNG实际格点颜色与mask逐图验收。
- `terminal_outputs/raw/REMOTE_MANIFEST.json`与`collection-receipt.json`：原远端路径/逐成员身份与传输验收；原模型留服务器。
- `preparation-verification.json`和`preparation-v2/raw/`：实际新SID、metadata、cohort、20份固定源码及执行配置。
- `audit_terminal_outputs.py`、`numeric_core.py`、`final_binding_checks.py`、`verify_plot_pixels.py`：本次实际执行的可复核CPU验收程序。

GPT_UPLOAD_REQUIRED: YES（由主控并入唯一最终审核ZIP，本子任务不另要求用户上传散件）。
