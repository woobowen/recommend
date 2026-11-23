# 非作者审核 A · 研究与模型执行

**结论 PASS。** 实际审核对象为本轮运行提交 `c487f7e7176bc223445d3deecfab87e0bcc30d1a`。

## 源码检查

审核身份为独立子代理 `/root/audit_a`。直接阅读了冻结 Normal GPU 提交 `0e3e2c6f5c1a76152ea0228390d94ffc1ff75ba5` 的 `evaluate.py`、`common.py`、LETTER 的 `data.py` 和 `prompt.py`、V1 Shuffle 派生源码与 GPU 回执、V2 冻结核心 `8dddbe9bcf9f38b3d2b7c5076daa212b2b664b2e`、本轮 A1 实现以及 new-server 实际安装的 Transformers、PEFT 相关源码。生产模块由主控及另一位执行代理编写。

- 对独立生成的派生文件做了 AST 和完整 diff 核对。冻结的 checkpoint 严格加载、Prompt0 数据映射、collate、候选解析与原始 `model.generate` 表达式保留。正式的单分片评价从 GPU 阶段移至随后 CPU 阶段。
- 原收据中的训练模板含 8-bit；实际 Normal 推理时 `strict_reload` 使用未量化 BF16 与原生 SDPA。以实际运行方式为准。加载后检查 394 个保存键、392 个 LoRA 键、152,587 × 1,024 的活动 BF16 embedding，以及同一 Parameter 和 storage 的活动 head。PEFT 0.21.2 的 wrapper 读取活动 adapter 的完整保存行。
- V1 Shuffle 已完成的两个 A1 rank 各报告 16 个唯一全局行。CPU 审核的样本轴与这两个 GPU 回执一致。原 A1 Normal 的早期接口运行只有 8＋8 人。本轮沿用 V1 的完整 32 人名单，以先 rank 0 后 rank 1 定义样本顺序。
- V2 共用冻结原始出现轴，按活动槽长度、全局行号及可见出现位置抽取相对排列。S3a 和 S3b 共用相对排列。本轮使用 `seq[:-1][-20:]` 的四个完整 token ID 守恒映射。冻结历史材料、组外字节和目标轴作为输入核验依据。
- 原 `Allowed` 用原文本和 IDs 生成 byte-BPE offset 并验证边界，包装器检查只有授权槽可变化，再把主动前缀更新为 CUDA generate 所收输入。原类源 SHA 为 `6ef011f28e88802b70bd8d8874dac3d1291a296f6e739dbf2ff5443c0bc95814`。
- 实际安装的 `Qwen3ForCausalLM.forward` 将输入 IDs、2D mask、position IDs、cache position 及 past 直接传入 `Qwen3Model`。Transformers 4.57.6 从显式 mask 累加构造位置，以一维序列位置初始化缓存，并在 beam 扩展中排除 `cache_position`。正常的默认动态缓存构造已分配的 layer 对象，进入首次 decoder forward 时各层 KV 长度为零。经审阅的生产 hook 断言 20 个 beam 共用已置换的输入、mask、位置及活动 embedding 行，证据保留所有 beam 检查结果与可重建的公共行。
- 原 V1 A1 vocab 摘要 `4d0e7267626dde1ab31259f328a81a4be2a923a824e2ae62c2bd74d4d01a4ccc` 采用未排序的 JSON 序列化。实测同一已冻结词表在同一进程的两次 Fast Tokenizer `get_vocab()` 得到不同插入顺序。对同一完整映射按键排序后，摘要为 `7c0e678397740771aa98ef5268786f15020e82537a8bffadf47c4829bb8533e9`。原 tokenizer 文件的固定 SHA 与词到 ID 映射作为另两项身份约束。
- 用适配器的 tokenizer 原文件及冻结索引和交互，独立拼接 Prompt0 并复核全部 24,772 人的原 IDs 与显式 mask。复核历史样本 640 个候选的原始 FP32 分数表示、生成次序、SID 码位、EOS/PAD 和稳定排序。两份冻结 Normal 分片和历史材料保持只读。

现场相关依赖源码的字节标识以下列 SHA-256 为准。

| 文件 | SHA-256 |
|---|---|
| transformers/models/qwen3/modeling_qwen3.py | `4b95c371fd26d40c69083dab36ac1eafd8cf82b415a0bb827275097c5ad2305b` |
| transformers/generation/utils.py | `55fd5c50aa785e226afa03e4507b8d27c0328714c0bd048bc60af9e8813a1173` |
| transformers/cache_utils.py | `f4e11caf14c75bdefa5e3003b93ef0e8b97fb6103375705ad944b0b344d7da8a` |
| peft/peft_model.py | `d22a1c50dea7a046ce28e71a79dfc20b1d96a2d238a9b457be6a88b6068e04d5` |
| peft/utils/other.py | `6b87957915ba399bfb6bbd57b9404aa12f0e9f4e27b19ef1b6c68ac052148f9b` |

## 全量 CPU 独立复算

从独立脚本和原始文件复算成功。直接读取两份原 Normal 分片、冻结目录及相互作用、A1 adapter tokenizer 原文件和本轮全量回执。独立构造 24,772 份 Prompt0、显式 mask 与目标轴，按原双 rank 文件中的可验证采样顺序交错恢复单 rank 顺序。独立脚本不导入本轮生产模块。重新写出每次出现位置的 SHA-256 键和字典序目录，核验 99,088 条模式用户记录、691,716 次模式历史商品组。S3a/S3b 全轴共享排列 ID。

| 条件 | 恒等历史组 | 整条输入恒等用户 | 实际变化 token 数 |
|---|---:|---:|---:|
| S2 | 86,483 | 714 | 172,892 |
| S3a | 28,935 | 6 | 345,429 |
| S3b | 28,935 | 6 | 345,429 |
| S4 | 7,147 | 0 | 519,226 |

CPU manifest SHA-256 为 `93e7a63e53715e639b7d6aea088b4d68d1309658e7a9a8b04cddfbbc0d99d31c`。逐模式输入流、频次与各槽移动的独立结果见复算 JSON。直接和 A0 对比，两者用户轴摘要、排列分布、移动次数和恒等计数完全一致。词到数字 ID 映射不同，四种 A1 输入流摘要也分别不同。部署回执的 25 份文件 SHA-256 及长度与 `c487f7e7176bc223445d3deecfab87e0bcc30d1a` 对应 Git blobs 一致。

## 真实 GPU 边界与自然恒等复核

独立运行从四个真实预测分片与四个 rank 回执读取全部 128 条模式用户记录和 2,560 个候选。实际物理 GPU 1，UUID `GPU-59cac3bd-3645-668b-79c8-839e0d878893`，逻辑 CUDA 0，单 rank、batch 1。复算每条输入与对应历史组的相对及绝对排列，重新验证 Qwen 分隔符 offset 与 SID 四槽合法 ID，核实四码位后真实 EOS 151643、其后 padding、候选生成原序与稳定分数顺序，确认原始分数可用 FP32 精确表示。原预跑生成的 rank 0 派生源码和 diff 均逐字核验，四模式共用派生源码 SHA-256 `9fe9dde11d60e706bcec6f32a066c40d693726bdef3220ca6fca1c42fdbba11e`。

四份真实 GPU 回执的模型、adapter、tokenizer、版本、源文件身份及严格 reload 明细与冻结 Normal 一致。服务器 CPU 资产回执记录了实际 Qwen 基座权重 SHA-256 `cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba` 和 best adapter 权重 SHA-256 `fe96ea74fc3c7253b0c3687239266b0f073e1a51ef9158e7d97bf30b4527bd03`。四个模式分别保存了所有活动参数的内容摘要。各阶段生成前后及四模式间的摘要均相同，为 `a215f6c66b3cf8a42bd5f830d75514d41b8c3e8c0dfed3b6ee35dcd1df08f4a4`。核对了生成调用实际记录的 beam/return 20、最多生成 10 token、确定式 beam search、cache、early stopping、真实 EOS/PAD 和原始 FP32 分数。

逐条比对实际 prefill IDs、显式 mask、自然位置、cache position 及活动 embedding。运行 hook 在首个 Qwen3 decoder forward 验证全部 20 beam 与所保存的共同输入、mask 及位置行精确相等。活动 BF16 embedding 读取的实际输出与按原 embedding 行重排得到的独立期望张量精确相等，两者记录的 SHA-256 相同。每次首次 forward 的 DynamicCache 含有 28 个长度为零的层，cache position 为本次完整输入的从零递增索引。审核者直接检查生产 hook 实现及本轮全部证据，从 CPU 文件独立复算可复算的边界输入和候选检查。模型装载、访问 CUDA 张量及前向计算发生在已记录的预跑作业。

S2 全输入自然恒等的两个全局用户行号为 12005 和 17716。两个用户的输入、共 40 个候选、原序、排序、原始解码文本与 FP32 分数均精确一致，最大绝对分数差为 0。S3a、S3b 和 S4 的此次 32 人样本各有零个全输入自然恒等用户。该对照支持这两个用户的单卡与历史双 rank 结果一致。全量单卡数值轨迹需等正式实验收集。

独立复算程序为 `independent_inputs.py`，输出为 `independent_A.json`，实际退出码为 0。原始 CPU 与 GPU 分片内容仍可直接复查。使用的是已存在的 `tokenizers 0.22.2` wheel 解包后的 Python 导入目录，未修改解释器或系统依赖。
