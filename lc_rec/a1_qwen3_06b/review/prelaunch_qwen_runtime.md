# A1 Qwen 正式训练前非作者运行审核

**结论：Qwen 适配、真实双卡训练、完整跨进程恢复、hot best恢复、严格Adapter推理及资源预算的必要预验收门槛 PASS。** 本结论允许主控在汇合其他必要审核、冻结正式RUN并核服务器部署字节后，启动已授权的唯一 Instruments seed42。它不是4 Epoch、完整测试、正式指标或最终实验PASS。

审核者：`/root/qwen_prelaunch_review`。未编写训练、checkpoint、比较器、推理或runner实现；只写审核材料。主控执行服务器SSH/GPU/CPU任务并桥接原始证据。审核者实际读取固定源码、request/status/cleanup、两rank完整事件、详细状态比较、真实原始预测和资产；独立在本地复算事件关联、全部16用户预测与累计计费。审核者未独立SSH/GPU重跑，也未在本地解包多GB训练tensor；这一能力边界不作扩大。

历史依据及round0–3定向静态审核详见 `qwen_historical_basis.md`、`training_static_review.md`。本报告以实际新增运行结果作最终必要门槛判断。

## 1. 实际运行与身份

| 链路 | 固定RUN | 实际job及终态 |
|---|---|---|
| 本轮CPU资产/环境/tokenizer审计 | `a9d36910e20b95ab7df6622f29479b5d542f0b68` | `letter-a1-q06-20261113-cpu-r0`；PASS原始CPU产物 |
| 连续12更新、CP2/12、validation/hot恢复 | `d9de77adfc92328f2bfd5e64ba40b652946f096f` | `letter-a1-q06-20261113-continuous-r2`；训练组件PROBE_COMPLETE；整体FAILED保留，因为其随后原推理字符串断言错误 |
| fresh-process从CP2继续至12 | 同上 | `letter-a1-q06-20261113-resume-r2`；SUCCEEDED、exit0、cleanup verified |
| 全量训练状态CPU比较 | 同上 | `letter-a1-q06-20261113-compare-r2`；SUCCEEDED、exit0、cleanup verified |
| 修正后的独立推理接口 | `ea922c12d5872a95180c84d5660ec853138c10a4` | `letter-a1-q06-20261113-inference-r3`；SUCCEEDED、exit0、cleanup verified |

实际服务器资产根：`/data/bwwu/lc-rec/a1-qwen3-06b-instruments-20261113/`。训练参考目录为其中 `letter-a1-q06-20261113-continuous-r2/training/`。推理任务显式记录独立的training RUN与inference source RUN；它复用已经通过的CP12及Adapter，未重复训练参考路径。

模型为官方 `Qwen/Qwen3-0.6B-Base`，revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`，权重1,192,135,096 bytes，SHA-256 `cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba`。实际CPU审计完整哈希、后续相同manifest/path/size、两rank model config身份与已审历史链一致；不是仅靠目录名或报告常数。

作者A版数据三SHA、9922物品/9897完整SID/23碰撞组/24772用户均实际核验。审核者又独立读取并重哈希本地三份作者原始数据，与推理两rank报告的服务器数据SHA一致。没有使用930 SID或六任务配方。

## 2. 模型、tokenizer与训练计算

- 原tokenizer151669，增加918确定性SID token后152587；物理词表151936→152587，651行物理扩容、267个新SID ID复用原padded行。没有把918行都称为新随机行。
- 本轮CPU审计复核全部9922物品SID编码/解码和tokenizer保存重载。实际Base EOS/PAD均151643，PAD由attention_mask识别；输入四SID+真实EOS共5个监督token。
- Qwen3ForCausalLM，28层×7对应投影=196模块、392 LoRA A/B张量；LoRA5,046,272参数，完整active tied矩阵156,249,088参数；393独立trainable Parameters、161,295,360元素。完整共享矩阵参与训练，不仅训练新增918行。
- LoRA r8/alpha16/dropout0.05；原生SDPA；Int8主干/BF16计算；非reentrant gradient checkpointing；原始作者response-only FP32全词表causal CE/logits除0.8；AdamW2e-4/weight decay0.01/cosine/warmup0.01；双卡micro8×acc4=global64；ZeRO2。
- 实际loader每epoch每rank8240个microbatches，2060更新/epoch、四轮8240更新。每epoch131840呈现比原131837行多3个sampler尾部补齐呈现；实际记录了SeedableRandomSampler/BatchSamplerShard，未仅硬编码更新数。
- 两rank连续12更新各393个Parameter均获48次梯度观测，全部有限且至少一次非零；实际初始基线在DS/BF16构建后采集，全部393监测对象有真实变化，全部918 SID行有梯度。fresh恢复路径对应各40次观测。AdamW成员准确且唯一；共享矩阵仅优化一次。
- 独立CE公式在本轮最大绝对误差`1.9073486328125e-06`，低于既定2e-5公式交叉检查门槛。监督token=8×5。新Trainer明确禁用`num_items_in_batch`语义，native Trainer恰除acc4，DS `scale_wrt_gas=False`，未重复缩放。
- 未调用LLaMA FastChat/FA2/A-only旧补丁。没有为了恢复更换Attention kernel、精度、共享结构、训练seed或数值标准。

## 3. 完整checkpoint、fresh继续及hot best接口

连续路径真实保存CP2、CP12；各manifest23个必要文件，包含两rank完整ZeRO optimizer/master/moments、模型、scheduler、RNG、loader、Adapter/tokenizer和Trainer状态；`exclude_frozen_parameters=false`。完整标记在collective写完后封印，随后才轮换best/latest。

fresh任务实际严格载入连续CP2，HF与DS step均2，重放8个loader yields后执行10次非零LR更新至12。32条真实validation接口通过，随后native hot strict恢复CP12。cold/hot bnb SCB兼容仅临时借用已有scale storage；本轮原生CPU序列化契约以及真实GPU恢复均通过，没有关闭strict或重新量化以通过。

完整比较原件：`evidence/jobs/letter-a1-q06-20261113-compare-r2/run/comparison.json`，2,016,473 bytes，SHA-256 `0b3402cded34494fbcf22f372540a0439619975dcab799029d74227c5d7c495e`。

审核者直接重计详细比较行：**每rank1363个tensor全部exact；0差异元素、0越容差元素。** 全393 trainable/161,295,360元素、196 Int8动态状态、80,647,680元素FP32 master partition及等量两份Adam moments、scheduler、全部RNG与loader状态覆盖完整。原固定atol1e-6/rtol1e-3保持，观测未用到容差余量。两rank各连续/恢复acceptance文件的整体SHA也相同。

审核者另从双方原始事件独立复算：每rank完整40个恢复microbatch的input/label SHA、shape和loader进度exact；40个micro-loss exact；step3–12的10次LR/grad norm exact、LR非零且有限。事件原件哈希与比较器输入哈希一致。

这里的best检查是有界接口：人为指定的CP12被严格恢复并导出Adapter。正式4次完整Validation与最低eval_loss选模仍必须由正式运行产生，未把本次32样本接口结果当成正式best证据。

## 4. 原生PEFT共享与严格推理

PEFT0.21.2原生ensure_weight_tying会将lm_head转入`modules_to_tie`，保存配置的`modules_to_save`变为`[embed_tokens,model.embed_tokens]`。实际安装源码和历史Q06-L原Adapter配置已核实此行为。先前要求字符串集合必须含lm_head的错误审计断言已纠正；模型、权重和训练没有因此修改。

inference-r3真实验证：

- `modules_to_tie=[lm_head]`，两端均为ModulesToSaveWrapper，active default权重是同一Parameter/storage、BF16、shape[152587,1024]。
- 392独立LoRA Parameters+1 active共享矩阵，393独立对象；完整394个保存/加载tensor的键、投影shape、BF16 dtype、有限性和值全部严格核对。没有dtype转换后比较或遗漏共享别名。
- CP12 Adapter与导出的preflight Adapter在CPU逐394个tensor验证同key/shape/BF16/值；两文件SHA均为`095d7f3b7ed843c06a8470030d4048d1f14a42929d8a5f179f2ebdbd300a660f`。`run/adapter_binding.json` SHA为`acebff679bdbf80bd6fb50ac53bb419aacec86df5e4a3907e830b866dc87aa11`。
- 重新加载BF16 Base+Adapter，显式`autocast_adapter_dtype=False`；Base、tokenizer、Adapter和callback身份在两rank一致。没有沿用Int8训练主干作BF16推理的替代品。

## 5. 全部真实预验收预测的独立复核

审核者读取两份完整JSONL共16名不同用户、每人20候选，总320条；未只看summary。原件：

| 文件 | bytes | SHA-256 |
|---|---:|---|
| `inference-r3/run/test_preflight/predictions.rank0.jsonl` |131127|`5abcb977705c3da3ff1252b1d65c5e3a7a1cc90fe6b09993d3fda99d8d37d95d`|
| `inference-r3/run/test_preflight/predictions.rank1.jsonl` |131267|`f6fb04ef1dd8791dab6f8846cf2195dbcd815a46a7be33c8cbd17bdb1b931248`|

独立方法复用历史非作者ASCII byte-BPE编码器，未调用主训练/推理tokenizer适配器或HF tokenizer包；其官方tokenizer输入SHA核为`c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`。审核者从作者交互/SID/prompt源码重新构造全部24772个真实prompt0，独立编码得到最大167 tokens，并重建本次固定分层/最长包络采样。

实际逐条结果：

- 16用户row_idx/UID/目标item/目标SID一一吻合；最近20历史包含验证交互；所有输入token ID与独立编码逐个exact。
- 每rank输入长度包络均为`71,71,71,77,95,113,167,167`；共有4个真实最长167-token用户。跨rank无重复用户。
- 全320个候选各为四个SID位置token+真实EOS151643，EOS在位置4；各位置集合从作者SID独立构造，大小150/256/256/256，全部命中相应集合；后缀无超范围token/非padding残留。
- 每个raw_decoded都等于真实prompt+该四token解码SID；原始解析、finite sequences_scores、20原始generation_order、稳定分数降序和保存relevance全部独立核对一致。
- 不过滤非法完整SID、不去重、不以碰撞消歧。**本12步探针320/320完整SID非canonical，重复0；四个原指标均0。** 这是保留的不利接口观测，不是正式训练质量结论，也不是准入指标阈值。

两rank报告均`PREFLIGHT_INTERFACE_PASS`；真实推理source SHA与已审源码一致。有限预验收不替代后续24772名用户的唯一正式Beam20测试及独立全量复算。

## 6. 资源、累计预算与修正

本轮live对账原件：`evidence/prelaunch_budget_resources.json`，SHA-256 `5956d7acc2c9e1b1039d6b4249e0c3acf9024c694ab0e5a09a27acf92a4a4a5f`，观测UTC2026-11-13 11:48:24。

审核者将全部5个GPU尝试分别与原runner started/ended及pipeline时间对账，独立使用二者较大墙钟×2重算：**590.881410 GPU-device-seconds=9.8480235 device-minutes <60**。失败开销、加载/保存、清收没有扣除；5次全部terminal且cleanup verified。formal ledger为空，当前无正式seed，也无GPU compute-apps。

- 连续训练最大allocated6,397,692,928 bytes、reserved14,933,819,392 bytes；fresh恢复同量级，无OOM/非有限训练值。
- 长包络Beam20两rank峰值allocated2,151,127,552 bytes、reserved2,493,513,728 bytes，显著小于单卡容量。原始16用户生成时延P50=0.349735551秒、P95=1.121987423秒，仅为含冷启动的小样本接口成本。
- 推理期间最小MemAvailable约252.4GB，剩余输出盘约2.957TB；收口可用RAM256,522,924,032 bytes、swap已用152,190,976 bytes。两卡清收后各1MiB、0%利用率、27/28℃。
- GPU UUID：`GPU-dfff0e95-1db6-e2f4-3431-55bb852fa0ae`、`GPU-59cac3bd-3645-668b-79c8-839e0d878893`。

修正轮数为**任务级3/3，已用完**，不按job/代理重置。保留的失败包括：r0可选TF/Keras3导入、r1原生DS logging infinity的报告JSON、r2推理错误字符串审计。前两次0更新；r2训练12步已完成并复用，未为修推理重训。修正没有改变研究配方、训练kernel、共享结构或验收容差。出现新的必要阻塞应按授权停止其依赖并报告，不能开启第四轮。

## 7. 必要门槛最终判断与字节身份

| 必要门槛 | 本报告结论 |
|---|---|
| 官方Qwen资产、作者A918数据、tokenizer/label/EOS | PASS |
| 实际393参数/共享矩阵/196投影及优化器唯一性 | PASS |
| 双卡BF16/Int8、response CE/temperature/累积/ZeRO2 | PASS |
| CP2/12完整保存与真实fresh-process非零LR继续 | PASS，全部详细比较exact |
| native hot strict/best接口、32真实validation | PASS，有界接口范围 |
| BF16 Base+Adapter严格394 tensor重载与CP12绑定 | PASS |
| 16真实用户/320Beam20、167-token最长包络 | PASS，完整原始预测已独立核 |
| 60 device-min累计预算与终态清收 | PASS |
| 正式4 Epoch/完整Validation/best/24772用户/最终指标 | NOT_STARTED，须正式任务产出 |

**本报告范围内所有开训前必要Qwen技术与资源门槛：PASS。** 主控仍须在实际开训前汇合已安排的独立CPU评价流水线/协议审核，冻结正式GPU RUN、校验部署与防重/96小时ledger，并保存外部回执；本报告不引用尚不存在的正式RUN提交或声称已经启动。

受审、实际已运行的精确字节：

| 文件/配置 | SHA-256 |
|---|---|
| `scripts/train.py` | `b145d2f5606a3fa0703d39984860ade9401820d9f9cd2a750f8c21e499dcb88a` |
| `scripts/checkpoint.py` | `11e6191c44abd7402c1e0720e1358bfe0edbf86b28b4dc12f1717daacf79a1cc` |
| `scripts/compare.py` | `2a58c611318f96b315fb260d2dae6eaa577616e17bda62e2513aed2617747170` |
| `scripts/evaluate.py` | `c08f6cceb982147009e6d602a6b693065830e33a5b24e7d40b1d79b7156981f1` |
| `config/formal_instruments_seed42.json` | `5fd3f604d0baffa8a7633935b5a75a4ee8afa33106ea8972dded952795540dbd` |
| `config/ds_z2_bf16.json` | `e4db1e7ed995a6733b754cbfed53af9f5191fb37a4be3eba33481d3a13a31fae` |

训练三核心字节固定于`d9de77adfc92328f2bfd5e64ba40b652946f096f`；严格推理字节固定于`ea922c12d5872a95180c84d5660ec853138c10a4`。任何后续正式R/E源码export须以这些真实已验收字节逐项对应；不能仅因新commit名称相似继承PASS。
