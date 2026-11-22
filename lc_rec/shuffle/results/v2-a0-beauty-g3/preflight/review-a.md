# 非作者审核 A · 研究实现与模型边界

审核者为 `/root/review_a`。独立读取原始 G3 预测、CPU manifest、实际 rank 记录及源码。审核脚本仅校验已保存数据。

## 源码和资产

- 从 Git 对象 `21d495b0be5b606853ec7cec68a3eaa1b58acbd7` 读取原始生成入口。服务器冻结 `formal_evaluate.py` 的 SHA-256 为 `9c7748ce7659a085d15128e060a1b72f6ac92b62a8642056810d0fa6df6a14e7`。配置列出的 16 个源码与数据文件 hash 均从该固定 Git 对象独立复核。
- 复核 `7368d301d0d928f77d50551c3b53a5b30c23d63f` 对原 G3 预测、训练和 CPU 收尾的交叉引用。从服务器直接读取单 rank Normal 的原始 argv 和 rank 报告。冻结 V2 core、独立 reference 与 V1 OFF 输入合法性 helper 均与第二组 `6212bb302ff4f8db87232ca1e7993f411961eac9` 字节相同。
- 服务器原 `checkpoint-6160/complete.json` SHA-256 为 `6b3f2c13607b093d0b2aeefb36e83780f93d454690da5e5a52f72730e0ee074b`，与实际 best adapter 的 `checkpoint_source` 对应。模型身份文件 SHA-256 为 `3315622866cd97833741608ca21d82b72a10184ab8bff4dc688892146d0bf915`，adapter payload SHA-256 为 `d3a179b59ba7a7218d6e1767884fd1c5961fffb9b756ea5f8bc5522775db6a28`。
- G3 唯一 item SID 为 12,101 个，各位置可用符号数为 256 / 256 / 256 / 18。冻结 tokenizer 的 786 个 SID token ID 为 32000 到 32785。原完整 vocab 内容身份为 `35a8f0b6fef3ce82887e6efa6f92a6d4f75eccf21b8f94064514548dade9e585`。
- G3 冻结 32 人清单 SHA-256 为 `8d72a6525b53be97fee4dc3283f75df68b6cba87fc08b1dfe6ce265ebcad77d0`，原顺序为旧 rank0 后接旧 rank1，并绑定 G3 目标。历史清单中的额外 padding 说明用于原 V1 联跑夹具。此次实际逐行要求人工 padding 为 0。

## 生成调用链

1. 历史 Prompt0、`SeqRecDataset` 和 `TestCollator` 输出输入与 mask。
2. `generate_one` 的 V2 变换替换 V1 干预，沿全局行和可见历史出现位置独立取样。V2 核心借 V1 OFF 核对分组边界与 SID 完整性，该调用只处理 token 序列。完整 token ID 移动到新位置后使用该位置的原生 position ID 与 RoPE，并进入对应的因果上下文。
3. 调用真实模型前并行核对独立 reference、CPU manifest、原始输入、mask、逐组原 token 与 permutation 和变换后输入。CPU manifest hash 与逐行证据 hash 进入预测记录。
4. 冻结模型严格装载继续用原正式训练 scope、commit、配置与文件身份。输出 scope 与 V2 执行提交独立记录。Non-gradient 冻结在 strict reload 后进行，模型仍使用原 eval 和 native generate。
5. 从现场安装的 Transformers 4.31 源码确认 `prepare_inputs_for_generation` 由展开 mask 计算位置，`LlamaModel.forward` 调用活跃 `embed_tokens`。独立审核检查实际 20-beam 输入、mask、position、初始 cache、lookup 及其 hash。生成前后 native defaults 对照覆盖 Transformers 的 legacy config reload 路径。
6. 预测沿原 prefix callback 未过滤地保留原始 20 个候选，FP32 scores、稳定排序及 EOS 行为。额外查验候选的各位置 token ID 是否属于原 G3 callback 集合。

## 独立期望锚点

- 完整交互轴 22,363 人，共 159,966 个可见历史出现。S2 恒等组 79,937、整输入恒等 640。S3a 和 S3b 均为 26,744 组、6 条整输入。S4 6,630 组、0 条整输入。来自固定 inter/index 的独立标准库计算。
- 冻结 32 人中仅 S2 含一个自然恒等，原行 `20014`。原 Normal 的源文件物理行 10，原行字节 SHA-256 `5c0f5c0f440b94717fbd838169390462c5bfd5859e354f0e321c11bf793b59da`。原精度的 20 个分数按生成顺序作紧凑 JSON 的 SHA-256 为 `26a0234f3a701d39e816632bafddb4aeb0cb8d52664a6f63af8c87a8bb794cf3`。

## 实际 CPU 独立复核

实际 `cpu-replay-r2` 为 `SUCCEEDED/0`，runner 正常退出并清收。本地独立脚本直接完整读取 400,029,854 字节的冻结 Normal JSONL 和 `cpu-replay-r2` manifest。独立重建的起点是已有 Normal 编码输入，`cpu-replay-r2` 提供实际 tokenizer/collator 的全量重编码。以冻结 inter/index 和逐用户的归档 Prompt0 验证全部 22,363 个输入与 mask、159,966 个可见历史分组位置和 token、89,452 条模式与用户的编码输出哈希、639,864 次模式与历史出现的独立置换、四个整输入流的 SHA-256。全数符合。详见 `cpu.json`。

## 实际预跑源码身份

预跑请求对应 V2 代码提交 `86660be5a62c8325caf74cf2810581850ca12803`。从 `new-server` 实际 release 读取七个关键 V2 文件 SHA-256，对照该 Git 对象和保存的部署清单均精确一致。这七个文件为 `g3_assets.py`、`g3_replay.py`、`g3_generate.py`、`core.py`、`reference.py`、`a0_beauty_g3.json`、`beauty.py`。实际派生入口 SHA-256 为 `f2d4fc05252dc0215553e0a35d0a5fcdecacd0b29323fb0ffb77e5133a23c80b`。

## 实际 GPU 独立复核 · PASS

本轮四模式在同一张 RTX 4090（`GPU-59cac3bd-3645-668b-79c8-839e0d878893`）依次完成，每模式 32 人、640 候选。审核直接读取四份原始预测和 rank/binding 对应文件，逐一核验全部 128 条真实 collator 输入、CPU manifest 行绑定、实际 20-beam prefill 记录、BF16 lookup 记录，以及 2,560 个候选的 4 位顺序合法性、EOS、FP32 分数、生成/排序顺序与目标匹配。检查每次运行的 450 项 strict reload 结果及其总元素数、LoRA FP32 和其余基础权重 BF16，确认参数冻结和原生 generate 配置。四份实际派生入口字节一致。完整复核出自 `audit.py`，结果在 `gpu.json`，本地运行用时 7.58 秒。

S2 的自然恒等行 20014 与原 Normal 的实际输入、mask、原生位置、全部 20 个候选、生成及排序顺序和 FP32 分数精确一致，最大绝对分差为 0。S3a、S3b、S4 冻结样本中的整输入自然恒等数均为 0。

本审核覆盖冻结 32 人的真实模型边界与全量 CPU 输入。更大范围的 GPU 推理由正式阶段生成。独立审核从实际保存的 embedding hook 证据核对形状、输入绑定和等值检查，离线重新计算 embedding 数值需要同一冻结模型权重。
