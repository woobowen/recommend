**QWEN THREE-MODEL PREFLIGHT：PARTIAL。** 三款官方Base权重全部正式校验，五条训练技术链通过，Qwen3.5 LoRA保留方法映射BLOCKED。

本轮是用户明确授权的一次性 Instruments 技术预检。仅三款官方 Base、LoRA ≤4 次／款、BF16 全参 ≤2 次／款、global64、seed42；没有正式 epoch／seed、完整推荐指标、其他数据集、六任务、上游 SID 训练或 Shuffle。技术预跑不能证明哪款模型正式 Recall/NDCG 最好。

六条候选均已真实收口；全部运行、失败、三轮修正和有界结果保留。发布固定SHA、GitHub实际回读及服务器最终外部回执见ZIP的final_receipts与最终回复。

**1. 先读材料及身份边界**

- 本入口 → `evidence/execution_results.json`（六路径终态）→ `evidence/summary.json`（实测、成本分项和未知成本）。
- `metadata/*/manifest.json` 是完整必要文件的官方大小、LFS SHA／Git blob 身份；`evidence/jobs/*/request.json`、`status.json`、`cleanup.json`、两 rank JSON 和资源原件支持实际运行结论。
- `review/` 是真实非作者对源码、原始记录及独立算术的核验，范围和限制逐文件明确；没有把主控摘要转述称作独立运行。
- ZIP 中 `final_receipts/` 记录最终 GitHub ref／内容回读、服务器无 .git 固定 export、全部 job、资源账本和最终 ZIP 外部身份。ZIP 根仅有一个本入口，`SHA256_MANIFEST.json` 给所有成员的大小和 SHA。

| 官方模型 | 不可变 revision | 单一必要权重 bytes | LFS SHA-256 |
|---|---|---:|---|
| Qwen/Qwen3-0.6B-Base | `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` | 1,192,135,096 | `cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba` |
| Qwen/Qwen3-1.7B-Base | `ea980cb0a6c2ae4b936e82123acc929f1cec04c1` | 3,441,185,608 | `6df85b39330e5a425ee36253d0f894e4387e4f0a15b9c53cb467d668e6b3a841` |
| Qwen/Qwen3.5-0.8B-Base | `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` | 1,746,942,600 | `c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c` |

三者均官方 Apache-2.0 Base；总权重 **6,380,263,304 bytes**，小于 24 GiB。服务器直连官方来源实测不可达（Errno101），因此使用本轮已授权的官方 WSL 下载＋Windows SCP 路线；没有改网络、使用镜像、第三方权重或重复 .bin。WSL 与服务器正式发布分别完成完整文件大小和可信 SHA 校验；保留 job、有效断点及传输片段。正式目录为 `/data/bwwu/lc-rec/qwen-preflight-20261110/models/<model-name>/<full-revision>/`。

**2. LETTER 依据、数据、prompt、监督与生成**

采用 [作者 LETTER 固定源码](https://github.com/HonghuiBao2000/LETTER/tree/8d0154e28de37dbb6e24871c508ad8ddb1921cda) 与 [Issue19 后补参数记录](https://docs.google.com/document/d/1rVeu3JqGXZNjapiMi1ioXIlXdXSt8zl94FmaAFYwX4g/edit)，复用 [已审核 LETTER 技术入口](https://github.com/woobowen/recommend/blob/903b2971a78bcf140491669ca59004beaa628815/lc_rec/letter/AUTOGRAD_REVIEW_ENTRY.md)。完整来源及冻结小文件 SHA 在 `metadata/method_basis.json`；ZIP 还包含这些旧版必要源文件的精确 Git 内容，未夹带完整数据。作者补充是有效参数依据，不能证明历史 argv／checkpoint 绑定。

真实复用作者 **A 版 918 SID**，没有误用服务器另一个 930 SID 数据目录。Instruments：24,772 用户、9,922 物品、四位置；131,837 训练行／epoch，49,544 条两模板验证文本，24,772 prompt0 测试用户。原有 23 组 SID 碰撞、48 个物品（9,897 个唯一 SID）保持，未以 trie 或强行过滤消除。

| 输入 | 固定 SHA-256 |
|---|---|
| Instruments.index.json | `33c87e28c52ae367985bdf494cb49d676d181e13a39497b4ad035f46111546dd` |
| Instruments.inter.json | `0a11d47791d90483005da93c36d826679e20e4592d1d2b86518d5f56f218e743` |
| Instruments.item.json | `1d9938e576a8d964e6b977a2b63aa56402840a788eaa6685f26f0d910a7133f2` |

`common.py` 直接调用冻结作者 SeqRec Dataset，保留 17 套原训练模板、最近20条历史、`, ` 分隔、切分及目标四段 SID。验证保持两随机不重复原模板／用户；测试保持 prompt0、包括验证交互的原测试历史。模型间固定相同 raw row／prompt／user，未按 tokenizer 长度挑短样本。

训练只监督四 SID＋真实 EOS，PAD==EOS 时以 attention_mask 屏蔽 padding；没有误屏蔽有效 EOS。FP32 全词表 causal-shift CE 在 logits／0.8 后求 mean，与独立 masked log_softmax 计算核对。学习率2e-4、weight decay.01、AdamW、cosine、warmup.01、BF16、max_length2048、seed42。四 epochs 仅用于预算；原 global64 在双卡用 micro8×acc4 实现。真实当前输入最大184（Qwen3）／187（Qwen3.5）tokens，没有截断历史。

不使用 chat template、系统消息、thinking 或多模态输入。原 tokenizer 文件中存在模板不等于调用模板。Qwen BPE 对独立 `Response:` 与原 prompt 语境有不同编码；适配层从真实 offset／原 prompt 结尾绑定生成起点，未继续套用 LLaMA 的 `[1:]`。位置候选数仍为 **150／256／256／256／EOS1**，没有严格 trie。beam20、return20、max_new10、原未加训练 temperature 的生成评分保持。

每款 CPU 审计实际检查全部9,922物品的四 token 编解码、tokenizer 保存重载、padding／answer mask、8用户 callback 五位置及所有训练／验证／测试文本的长度分母。非作者还用独立 ASCII byte-BPE 重建固定样本和各 microbatch，不依赖主控 tokenizer 适配器。

**3. Tokenizer、参数及 tied embedding/head**

| 模型 | tokenizer 原始→扩容 | 模型物理词表行 原始→扩容 | 原始→扩容 unique 参数 |
|---|---|---|---|
| Qwen3-0.6B | 151,669→152,587 | 151,936→152,587 | 596,049,920→596,716,544 |
| Qwen3-1.7B | 151,669→152,587 | 151,936→152,587 | 1,720,574,976→1,721,908,224 |
| Qwen3.5 官方文本类 | 248,077→248,995 | 248,320→248,995 | 752,393,024→753,084,224 |

918 个 SID 都是新 tokenizer 身份；其中 Qwen3 的267个、Qwen3.5的243个复用原模型 padded 词表行，其余651／675行实际扩容。初始化使用模型官方 normal initializer、`mean_resizing=False`，不是把918行全称新随机行。

官方三者均 tied embedding/head。实际 resize、PEFT modules_to_save、DeepSpeed、保存重载均检查同一 Parameter／storage。PEFT0.21.2 的 `ensure_weight_tying=True` 保持完整 active embedding/head 共享且进入 optimizer 一次；冻结原模块副本单列，不把 clone 数量算作预训练模型规模。没有静默解绑、只训918行、冻结整个head或重复优化。

两款 Qwen3 的 LoRA 命中28层×7投影=196模块（392个A/B张量）；0.6B LoRA5,046,272参数＋完整共享词表156,249,088=161,295,360 trainable；1.7B LoRA8,716,288＋词表312,498,176=321,214,464。可训练占有效扩容模型约27.03%／18.66%，不能将 r8 误解为只训练极少比例。含PEFT冻结副本的实际分母另记在 rank JSON。

**4. Qwen3.5 架构与路线边界**

官方完整权重 **873,438,784 参数／488张量**：语言752,393,024／320张量、视觉100,592,896／153张量、MTP20,452,864／15张量。使用官方 Transformers5.2 `Qwen3_5ForCausalLM`＋`Qwen3_5TextConfig` 的内置 checkpoint 转换支持纯文本；保留完整官方源权重，不手工删键、伪装Qwen3或替换DeltaNet。全参路线仅表示 **753,084,224 个扩容后已实例化文本参数全部训练**，不表示完整多模态873M模型全参训练。

24层中18层为 Gated DeltaNet、6层为普通全注意力。作者七投影名无法覆盖18个 DeltaNet mixers，fused QKV／gate／step 的 LoRA 映射涉及方法选择。故本轮 **LoRA：BLOCKED_MAPPING_REQUIRES_APPROVAL**；没有以少量名字匹配冒充完整 LETTER LoRA，也没有自创 DeltaNet LoRA 来获得PASS。下一阶段需先由学长／用户审核目标投影及 fused 与独立低秩覆盖差异，r8相同也不代表参数化等价。

官方纯 Torch DeltaNet 路径用于有界预检，未安装可选 FLA／causal-conv1d kernel。官方text-class排除视觉／MTP、此kernel选择及全参身份均独立列明，不能直接将本路线当作原论文 LoRA baseline。

**5. 运行、保存、生成、资源及 seed 成本**

三款资产、三个CPU/tokenizer审计和五条技术训练链通过；Qwen3.5 LoRA因未批准的DeltaNet投影映射保留BLOCKED。F不属于论文原始LoRA配方，短跑PASS不等于完整seed验收。

| 模型 | 权重校验 | LoRA技术状态 | 全参技术状态 | LoRA显存GiB | 全参显存GiB | LoRA预计seed 中心／保守 | 全参预计seed 中心／保守 | 未决问题 |
|---|---|---|---|---:|---:|---|---|---|
| Qwen3-0.6B-Base | PASS | PASS | PASS | 6.378 | 8.212 | 18.02／21.76h；.751／.907天 | 12.41／14.47h；.517／.603天 | 正式seed与质量未测；F需另行采用确认 |
| Qwen3-1.7B-Base | PASS | PASS | PASS（round3后） | 10.060 | 21.082 | 20.56／24.40h；.857／1.017天 | 28.83／33.49h；1.201／1.396天 | F需micro4＋allocator，显存余量窄 |
| Qwen3.5-0.8B-Base | PASS | BLOCKED | PASS（官方文本类） | 未执行 | 11.435 | BLOCKED／NOT TESTED | 34.46／39.91h；1.436／1.663天 | LoRA映射待批准；F排除视觉／MTP |

显存是两rank最大**训练 allocated 峰**，不是静态估值或总卡容量。对应最大reserved：Q06 L/F 16.886719/14.617188GiB；Q17 L/F 19.988281/22.894531GiB；Q35 F21.576172GiB。Q06 L原版本未留存训练后guards/validation reset前区间，因此不能称其整job allocator峰精确完整；训练峰真实，其他已记录phase与NVML采样另列。后续源码补齐该观测，未重跑已通过更新。

| 路线 | 训练精度／量化 | trainable unique参数 | micro×acc×GPU | 实测计时更新秒 | samples/s | 有效tokens/s | 计时分母 |
|---|---|---:|---|---|---:|---:|---|
| Q06-L | Int8基础＋BF16 | 161,295,360 | 8×4×2 | 5.477469／5.942040 | 11.208888 | 983.317198 | 128独立混合row |
| Q06-F | BF16全参 | 596,716,544 | 8×4×2 | 3.761833 | 17.012982 | 2930.486133 | 64呈现、1边界row/17模板 |
| Q17-L | Int8基础＋BF16 | 321,214,464 | 8×4×2 | 6.462683／6.930607 | 9.557025 | 838.404930 | 128独立混合row |
| Q17-F | BF16全参 | 1,721,908,224 | 4×8×2 | 10.335780 | 6.192082 | 1066.586155 | 64呈现、1边界row/17模板 |
| Q35-F | BF16官方文本全参 | 753,084,224 | 8×4×2 | 10.960396 | 5.839205 | 1023.320708 | 64呈现、1边界row/17模板 |

全部为GPU FP32 master／Adam moments、原生AdamW、ZeRO2、梯度检查点、无Offload；实际optimizer组和dtype/device逐项保留。LoRA每款4完整更新／3非零LR，F每款2完整更新／1非零LR。所有预期trainable张量均有有限非零梯度；SID输入位置、监督head logits、SID行及代表性线性层有真实变化。BF16小LR下部分监控元素／norm／A_log不变并不等于冻结，未声称每个参数或元素都数值改变。

L计时是两个post-first混合更新；F只有一个非零边界更新，不能称稳定混合吞吐，也不能直接按表跨模型／跨方法归因“模型本身更快”。完整精度、量化、tokens、padding、长度、数据身份、micro、累积、checkpointing和训练范围均可回查。相同大小micro每样本恰5监督token，mean loss的等样本权重保持；累加顺序仍有数值差异。

声明的256个独立原始row只是选择池，不能称全部实际训练。L实际256呈现／193独立row／209个(row,prompt)，中心计时128独立row；F实际128呈现／65独立row／81个(row,prompt)，中心计时只反复一个真实最长历史row及17套作者模板。原始文本、tokens、监督和padding均保留，未靠缩短历史或tokenizer相关选择提高速度。

| 路线 | 原生保存有效权重bytes | 保存秒 | 严格重载秒（MAX rank） | beam20用户／序列 | canonical |
|---|---:|---:|---:|---|---|
| Q06-L | 322,653,178 | .765477 | 2.006834 | 8／160 | 0/160 |
| Q06-F | 1,193,468,344 | 2.391871 | .455156 | 8／160 | 0/160 |
| Q17-L | 642,491,386 | 1.400502 | 3.770106 | 8／160 | 0/160 |
| Q17-F | 3,443,851,904 | 7.003661 | 1.011609 | 8／160 | 0/160 |
| Q35-F | 1,506,210,008 | 2.912607 | .676480 | 8／160 | 0/160 |

原生adapter保留完整active共享词表及LoRA。394 state key对应393个独立可训练Parameter，embedding/head是共享别名，不是两份独立训练副本。F模型321（Q35）／311（Q3）state key也含共享别名。所有模型／adapter均真实重新加载并全key/shape/dtype/值核对；F没有optimizer/scheduler/RNG checkpoint。

每条成功路径在同一8个不同真实测试用户上运行原prompt0、beam20、return20、max_new10，短/中/长/边界各2用户。共800条跨路径序列，全部4 SID＋EOS、callback五位置和finite序列评分通过；**canonical全为0/800**。这是技术生成事实，不能判定充分训练后推荐能力。每用户秒数、users/s、tokens、EOS、callback事件和GPU峰都在两rank JSON与summary中。

**完整seed的七分项**在[summary.json](evidence/summary.json)各路线 `seed_estimate.center/conservative.seconds_by_component` 中：train、validation、checkpoint、best_restore、beam20_test、init、overhead，逐项精确秒数、小时及天数均保留。实际loader：micro8每rank8240 batches，micro4每rank16480；均对应每epoch2060、四epoch **8240 global updates**，527,360个global64等价呈现，原数据四轮527,348条。没有遍历epoch尾，未编造tail/even_batches行为。原验证49,544×4=198,176条；原prompt0/beam20正式测试为24,772用户。

**96h状态**：Q06-L与Q17-L均 `LIKELY_WITHIN_96H`，是有短跑余量的条件判断；三条F均 `INCONCLUSIVE`，只有一个非零边界步，正式checkpoint和长期成本未实测。Q35-L `BLOCKED`、小时数为空，未用0代替。任何完整seed均不是CONFIRMED。

中心按实测更新、全量原始长度分母和8用户分层生成推导，保守采用慢样本、I/O降速及裕量。每次正式checkpoint的显式代理为 `2×unique当前参数bytes + 12×trainable元素 + 1GiB元数据/临时假设`，四次，再加最终native模型／adapter保存；保守带宽取实测1/4且每次加60秒。这不是实测完整checkpoint，也没有把未测写入时间记为0。正式optimizer/scheduler/RNG恢复、best选择恢复、完整验证聚合、CPU过滤/metrics、长期热状态与I/O波动均未测。10%/25%额外开销是假设，非置信区间。

**资源与修正**：全任务使用round1–3，未按模型重置。round1是Windows命令长度修正；原交互异常stderr未单独留存，只有真实前后源码和已验证成功回执，未伪造raw失败。round2补共享归档/request、精确旧DS缓存保守计数、task内cache及validation峰观测。round3合并F17碎片OOM与Q35元数据set；F17仅启用官方native `expandable_segments:True`、仍micro4/AdamW/BF16/原scheduler；Q35仅修证据序列化。两个r0 FAILED终态与traceback保留，r1各仅一次重试成功，未新增修正轮。

F17首次LR0 foreach_sqrt失败保留1个attempt、0完整global updates及104.613823设备秒；Torch可能已修改moments，这些部分状态随进程结束丢弃，重试从原LR0开始。冻结7份raw SHA及真实非作者源码顺序核查支持显式对账；attempt账本没有删除/覆盖，GPU账本没有扣回失败时间。实际 **8次LoRA＋6次F完整更新，其中9次非零**；attempt总15，含1个经证实失败。该特例不适用于未知失败。

GPU累计 **19.095178/360 device-min**；Q06/Q17/Q35分别4.852848／9.549943／4.692387，均<120，每job<45分钟。收口服务器新增allocated **27,003,990,016 B**，含权重、环境、缓存、原分块、共享归档和产物，<60GiB；最终小型发布export由ZIP外部回执再次计数。WSL资产暂存约6.60GB、Windows中转残留0；原片段/缓存保留，磁盘及40GiB host reserve通过。15个server jobs均terminal且cleanup_verified（13成功＋2失败留存），3个WSL relay均成功清理。两GPU各1MiB，无compute进程，无后台训练／下载。

**建议与未决**：优先考虑Qwen3-0.6B Base的L路线进入下一阶段经审核的正式验证：方法映射清楚、已测显存低、96h条件余量足。F06较低的条件成本可另行讨论，但它是不同训练身份，不能直接替代论文LoRA。F17依赖allocator且显存边界窄；Q35缺经批准的LoRA映射，F只覆盖官方文本子模型与原生Torch DeltaNet回退。所有候选均缺正式训练质量及完整训练状态恢复验证，不能直接称正式baseline，更不能据技术短跑预测Recall/NDCG最好。

**6. 必要工程差异及验证限制**

L：官方 Int8 基础权重＋BF16计算＋r8/alpha16/dropout.05、完整共享词表参数。bnb会把部分BF16输入内部转换为FP16进行Int8量化，真实 warning 保留；不是4-bit QLoRA。推理按作者LoRA分支重新加载BF16 base再载入adapter。Qwen原生SDPA替代只适用LLaMA的旧FastChat patch；未声称逐位复现LLaMA数值。

F：用户额外授权的 BF16 全参数技术对照；GPU FP32 master／Adam moments＋ZeRO2，无CPU optimizer Offload、CPUAdam、冻结层、缩小词表或模型。它不属于 LETTER 论文原始 LoRA 配方，未来正式采用仍需另行确认。通信bucket从历史500M缩至50M是显存工程选择，有效配置完整保留。

微批量改变时保留 global64，并单列 padding及token数。当前每样本恰5个监督token，使等大小microbatch的mean loss权重一致；仍不宣称 dropout、BF16累加和分布式数值逐位等价。最大2048是允许上限，资源预检按 Instruments 实际长度分布及其边界，不能外推任意2048长度任务。

真实 loader 构造而不遍历完整epoch；预算按实际长度、accumulation和tail/even_batches配置推导；不以简单527348÷64冒充生效步数。validation是有限原模板forward；训练以固定有限真实row池，不当作完整epoch。完整Trainer/DeepSpeed optimizer／scheduler／RNG checkpoint未写、完整训练状态恢复未测；model-only／adapter严格重载与正式best checkpoint恢复区分。中心／保守成本的I/O和10%／25%裕量为显式条件假设，不是置信区间。

**7. 环境、安装与已有材料保护**

两款Qwen3共用隔离 `/data/bwwu/envs/qwen-preflight-20261110-qwen3`；Qwen3.5使用独立 `...-qwen35`。只读复用 Torch2.6.0、DeepSpeed0.16.9、bnb0.45.5及既有CUDA12.4。Qwen3固定Transformers4.57.6；Qwen3.5固定5.2.0；共同PEFT0.21.2、Accelerate1.12.0、Tokenizers0.22.2、Safetensors0.7.0；Hub分别0.36.2／1.4.0。官方PyPI wheel来源、版本、SHA、离线安装argv及测试见 `config/dependencies*.json`、setup原始job。

本轮新增15个唯一版本wheel、19个环境安装实例；wheel总28,892,242 bytes。没有系统包、驱动、全局CUDA、共享环境升级或shell启动配置改动。两个新venv中的只读 .pth 引用既有DS／bnb环境；共享环境版本和关键文件前后SHA不变。保留新环境／wheelhouse／已验证下载缓存，未授权清理不自动删除。

初始根与规则入口是 `/home/addaswsw/recommend`；实际Git工作树 `/home/addaswsw/recommend-worktrees/lc-rec`、仅 `wbw-lc-rec`。显式从正式规则入口读取三份规范，未声称同级目录自动继承。原先两个未跟踪8B tokenizer文件保持，未stage；其他项目、旧LLaMA、SID、adapter和训练档案没有改动。

**8. 最终发布及审核包**

发布仅限 `woobowen/recommend:wbw-lc-rec`。RUN_COMMIT与交付SHA不同，逐文件hash证明实际GPU源码／配置身份；服务器是无.git固定export，不称checkout。

| 操作 | 实际固定RUN_COMMIT |
|---|---|
| 三款官方WSL下载及SCP | `3421f12d9a9c12b7e4e3a9c0de3e2353fef43541` |
| 两环境离线安装 | `66a2a82febde2a351df7e48b0b236196ddddbb03` |
| 三款CPU／tokenizer审计 | `7c670e20304d27e724a90bfc255561d0618f6775` |
| Q06正式资产发布／LoRA | `f29527de3b3f94ae5640d2e98faa1a7f74d2e0e8` |
| Q35正式资产发布 | `3fb71a0134a1c6f17e37482d5921bf13295b644c` |
| Q17正式资产／LoRA、Q06-F | `98206e447d21acc53aa0366911c804e01a6ca63f` |
| Q17-F／Q35-F原失败 | `e775a226270f9b3b772e8891a805ad62d7bf2fb2` |
| Round3 CPU回归及Q17-F／Q35-F唯一重试 | `a0903189686329b1029e535dbe2b00a1c34ef427` |
| 统一成本分析 | `b84526eadd6b3edc1a40c8110894221c28f96cd6` |

服务器源码根 `/home/bwwu/lc-rec/qwen-preflight-20261110/releases/<RUN_COMMIT>/`；原始job根 `/home/bwwu/grid-audits/sid-extensions-p0/jobs/<job_id>/`。恢复入口是这些现存终态原件与查询，不是重启计算。`evidence/final_resources.json`和ZIP外部回执给实际服务器脚本/config rehash、无.git的RUN_VERSION和交付比较；模型正式目录仍按第1节完整revision固定。

真实非作者 `/root/official_architecture` 未实现模型、探针或下载器；直接核官方身份、三资产链、CPU环境/语义、六路线结构边界、五成功runtime、两个失败及对账、原始双rank和资源。`/root/letter_semantics` 未实现模型/训练/成本主脚本，独立实现ASCII BPE和算术重算，核五条成功路线及统一70组件。后者参与了新环境准备，因此环境验证由前者独立完成。关键入口：`review/three_assets_non_author.json`、各 `*_runtime_non_author.json`、`review/round3_failure_and_zero_update_non_author.json`、`review/final_cost_non_author.json`、`review/final_delivery_non_author.json`。不冒称独立重新训练模型或完整seed PASS。

最终DELIVERY_COMMIT、GitHub branch ref／源文件／配置／结果／本入口真实回读、最终服务器export及无活动进程回执见ZIP `final_receipts/`，并在最终回复给完整SHA；避免在Git文件内自嵌最终SHA。ZIP使用固定Git树和必要外部回执，实际执行CRC、完整解压逐成员大小/SHA、重复和路径安全、秘密扫描，并由Windows PowerShell再次读取大小与SHA。ZIP整体SHA放包外，权重／adapter／wheel／optimizer不进包。

WSL relay独立于终端和Codex对话，但运行仍需WSL操作系统及笔记本供电；没有声称能跨WSL关机继续计算。本轮下载和中转已全部完成。服务器GPU runner及资产原件没有运行时笔记本依赖，所有真实作业现已终止，不存在需要用户继续看护的长任务。

**GPT_UPLOAD_REQUIRED: YES**。最终回复将给单份可上传ZIP的真实WSL／Windows路径、精确字节、SHA-256、成员数。完成本轮后停止，不自动发起正式训练或扩大研究范围。
