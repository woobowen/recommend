# 复现指南

## 1. 复核已有结果（无需GPU或原数据）

在仓库根目录运行：

```bash
python semantic-hierarchy/scripts/summarize_results.py
python semantic-hierarchy/scripts/verify_results.py
```

校验器使用Python标准库检查发布文件SHA256、冻结源码、原统计明细聚合、置换p值、README表格与相对链接。它不重编码商品，也不重新评估完整向量。

统计核心的独立公式测试只需NumPy：

```bash
python -m pip install -r semantic-hierarchy/requirements-metrics.txt
python semantic-hierarchy/archive/llm2vec_beauty_toys/test_metrics.py
python semantic-hierarchy/archive/llm2vec_instruments/test_metrics.py
```

原测试将向量和公式、兄弟Gap及两种条件置换与显式枚举全部商品对的结果比较，并覆盖单商品、父子集合相同、路径持平与Holm校正。

## 2. 用已有向量与SID重新评估（CPU）

新发布入口直接复用归档的 `semantic_metrics.py`，不会调用GPU或修改原有结果。输入：

| 参数 | 格式 |
|---|---|
| `--embeddings` | `N×D`的float32/float64 `.npy`，有限、非零、已L2归一化 |
| `--item-ids` | 与向量行顺序一致的整数ID一维 `.npy`，无重复 |
| `--sids` | JSONL，每行 `{"item_id": 0, "sid": [29, 6, 57, 0]}`；按ID匹配，只使用前三位 |
| `--output` | 全新的输出目录；不会覆盖历史结果 |

```bash
python semantic-hierarchy/scripts/evaluate_embeddings.py \
  --embeddings /path/to/embeddings_normalized.npy \
  --item-ids /path/to/item_ids.npy \
  --sids /path/to/item_sids.jsonl \
  --dataset beauty --variant title \
  --output semantic-hierarchy/local-runs/beauty-title
```

默认seed=20261113、n≥2、epsilon=1e-6、每种条件置换999次、bootstrap2000次，与正式统计一致。返回节点/边/路径/Gap、随机明细和summary；不自动作跨数据集Holm校正，也不产生历史编码完成收据。若改置换次数/样本数，需要报告为新运行，不能当作原结果。

初次发布未附完整向量；2026-11-21已在[补件资产清单](../delivery/assets.json)发布六组原始向量、同序ID和norms原件，实际bytes经SHA256验证，取得方法见[主说明](../delivery/README.md)。已有Beauty/Toys SID和商品TFRecord、Qwen上游代码仅引用[共用路径](shared-code.md)，无需重复上传。

## 3. 重新编码商品（Linux，两张2080Ti/一个模型）

历史入口位于 `archive/`，保留原服务器路径，**不是开箱即可在任意路径直接启动的可移植调度器**。重新运行前需准备输入、模型、运行配置和GPU锁目录，并修改本地副本的路径；归档源码用于追溯，不要直接覆盖冻结版本。

安装独立环境：

```bash
python -m venv .venv-llm2vec
source .venv-llm2vec/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r semantic-hierarchy/requirements-llm2vec.txt
```

依赖版本对应历史编码环境；此发布在本机进行了统计与入口测试，未在本次上传中重新安装Linux CUDA环境或重跑GPU编码。RTX2080Ti不按BF16/FlashAttention2配置。

模型目录结构：

```text
<project-root>/models/
├── Meta-Llama-3-8B-Instruct/
├── LLM2Vec-Meta-Llama-3-8B-Instruct-mntp/
└── LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised/
```

模型来源及固定适配器revision见[依赖文档](shared-code.md)；遵守来源模型访问/使用条件。不要将Llama3.1底座放进上述Llama3目录。

文本JSONL每行严格只有 `item_id`、`text`，ID从0连续；标题为 `Title: ...`，标题＋品牌为 `Title: ...\nBrand: ...`，Unknown/nan/none品牌省略。对应SID放在独立文件，编码配置不包含SID路径。`prepare_inputs.py`说明如何从旧Title/Brand/Categories/Price序列化文本提取两个视图；运行前将ROOT/HERE及基线路径映射到本地。

具体需要替换的历史路径：

1. `runtime_common.py`中的ROOT/HERE；GPU锁位于 `<ROOT>/stage_gpu_locks/gpu_<physical-id>.lock`。
2. 运行目录 `encoding_config.json`的文本路径、输入SHA256和模型manifest指纹；可从已发布配置复制模板，替换路径后重新计算文件哈希。
3. `analysis_config.json`的SID路径、SHA256及商品数。不能为了跑新数据沿用旧商品数/旧指纹。
4. 如果用整个历史调度器，替换其中独立Python环境、数据路径及原GRID阶段入口；若只需编码，直接运行worker并使用新的CPU入口评估。

准备完这些文件后，Beauty/Toys worker的调用形式为：

```bash
CUDA_VISIBLE_DEVICES=0,1 python /path/to/worker-copy/encoder_worker.py \
  --run /path/to/new-run --dataset beauty
```

worker会读取编码配置中的两种文本视图、生成归一化向量/ID/编码收据，并对模型使用的两张卡持有非阻塞锁。Toys使用 `--dataset toys`。Instruments需用 `archive/llm2vec_instruments/encoder_worker.py`副本，参数为 `--dataset instrument`。GPU0/1是示例，应选择当时空闲的卡；本次发布不启动任何任务。

## 4. 重做Llama自动分类

历史分类环境与LLM2Vec环境分开。已记录PyTorch2.6.0+cu124、Transformers4.47.0、Accelerate1.8.1、NumPy2.0.2，结构化解码使用lm-format-enforcer0.11.3及interegular0.3.3；原TFRecord读取还需要对应GRID环境中的TensorFlow依赖。

- Beauty使用 `archive/classification_beauty_v1/`；Toys使用 `archive/classification_toys_v7/`。
- `prompts.py`保存实际发现、分类、修订提示词，`schemas.py`保存结构约束，`inference.py`保存FP16两卡加载及约束生成。
- `prepare.py`读取原GRID商品TFRecord与SID，导出分类文本和前缀组；改 `config.json`的root、模型和数据路径。
- 阶段顺序为 discover → draft → revise → pilot → final → evaluate，`run_stage.py --help`列出单阶段参数。Toys候选归一逻辑见 `compact_taxonomy.py`。
- 原完整控制器还依赖服务器preparation/readiness收据及已经完成的Beauty产物，不应把它当作从空目录复现全部阶段的一条命令。重新执行前需要生成对应准备收据，不能复制旧哈希绕过核验。

最终自动标签不是人工真值；重生成可能受硬件/库影响，不能保证逐token相同。初次发布仅有taxonomy与汇总。2026-11-21已补全逐商品final标签及readiness，所有现存prompt/response、失败重试和控制器日志在metadata.zip；[补件说明](../delivery/README.md)给出阶段路径和真实验证范围。未保存的更早记录仍不补造。

## 5. Instruments数据与SID

`archive/llm2vec_instruments/download_prepare.py`包含官方来源URL、读取校验、元数据匹配、固定ASIN→item_id映射及文本视图生成。运行前改ROOT/HERE和数据目录。原文件/选择指纹见 `results/llm2vec_instruments/download_manifest.json`、`input_audit.json`。

Qwen编码worker调用原GRID均值池化，量化仍调用原 `grid_stage.py` 和 `run_pipeline.py`；仓库路径见[共用代码](shared-code.md)。`run_instrument.py`中的 `qwen_config.json`生成和quantize步骤记录实际覆盖参数：3层×256、30步、seed42、residual normalization、32位量化训练。它同时承担服务器调度，重跑时需要完整GRID环境和路径映射，不是独立于GRID的新量化实现。

## 2026-11-21 四种起点验收

详见[H1–H4主说明](../delivery/README.md)：从原向量CPU复算节点余弦、从全量final标签复算纯度与pair-agreement均已执行；固定模型重新编码、重新分类和从头训练均未执行。现存模型权重的公开访问条件与原manifest身份分开记录。
