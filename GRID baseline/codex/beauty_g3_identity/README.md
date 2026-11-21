# Beauty G3 商品身份与核验补件

本次补件绑定 `yjc-grid` 来源提交 `84f200258ab66b33bcf1fe4104fa44ad43737e4a` 的四位 Beauty G3，补齐全部 12,101 列到 Amazon ASIN 的映射。**原 G3 不变；没有重训、重排商品或重算后替换 SID。** 本次使用 CPU 核验，没有占用 GPU。

## 交付入口

| 文件 | 内容 |
|---|---|
| [beauty_g3_item_mapping.csv](beauty_g3_item_mapping.csv) | 全量 SID 列号、GRID 编号、原始 ASIN、P5 编号及四位码值 |
| [beauty_g3_verification.json](beauty_g3_verification.json) | 实际执行的覆盖、输入、预测键及 tensor 重建结果；包含脚本哈希、命令、UTC 时间、环境与退出状态 |
| [beauty_g3_provenance.json](beauty_g3_provenance.json) | 历史与新导出材料的区别、P5 来源、原件/可读副本对应关系、必要大文件身份 |
| [beauty_g3_input_manifest.json](beauty_g3_input_manifest.json) | 549 个 Beauty TFRecord 分片与历史清单逐一一致的实际哈希 |
| [beauty_g3_keyed_predictions.csv](beauty_g3_keyed_predictions.csv) | 从真实原始 PKL 导出的 `item_id,sid_1,sid_2,sid_3`，不是从最终 tensor 反推 |
| [p5_beauty_item_identity.json](p5_beauty_item_identity.json) | P5 原始 datamaps 的商品字段精确投影；不包含不必要的原始用户身份 |
| [p5_grid_sequence_fingerprints.csv](p5_grid_sequence_fingerprints.csv) | 全部 22,363 个用户测试序列的零基数值指纹 |
| [grid_item_text_fingerprints.csv](grid_item_text_fingerprints.csv) | 全部 12,101 条原始商品 UTF-8 文本的 SHA-256 |
| [historical_records/](historical_records/) | 原完整有效配置、overrides、command、阶段记录及 embedding 元数据的逐字节可读副本 |
| [verify_g3.py](verify_g3.py) / [fetch_p5_identity_inputs.py](fetch_p5_identity_inputs.py) | 确定性导出核验及原始 P5 输入获取工具 |
| [SHA256SUMS](SHA256SUMS) | 本目录必要交付文件的包外可独立复核哈希清单 |

## 固定 G3 身份

相对于仓库根目录：

`GRID baseline/experiments/beauty_rkmeans_qwen_20261010_012253/semantic_ids.pt`

- 大小：388,405 bytes。
- SHA-256：`d41adcfb22a9a47d5e0424c068f4c74f33cc9fdbaf5db08540fbfa0e3d720e18`。
- `int64 [4,12101]`；第一维是码位，第二维是商品列。
- 前三位是三层、每层 256 码字的残差 MiniBatchKMeans；第四位是前三位碰撞的去重编号，不是第四层量化码。
- 前三位各取值 0–255；第四位 0–17；完整四位组合 12,101 个且唯一。

## 映射来源与编号规则

历史 GRID 输入 TFRecord 的商品记录字段为 `id,text,embedding`，没有 ASIN 字段。原始 GRID 到 ASIN 映射文件以及 TFRecord 转换脚本未找到，不能宣称它们在历史训练时已经保留。本映射是 **2026-11-18 的恢复导出件**。

恢复输入来自 P5 官方 README 链接的 `P5_data.zip` 中 `data/beauty/datamaps.json` 与 `data/beauty/sequential_data.txt`。原始成员大小、SHA-256、ZIP CRC32、档案大小及取得地址记录在 provenance；精确商品投影同时发布。只下载必要 ZIP 成员，没有下载或上传完整 P5 档案。

本次不是因商品数或数字范围相同而假设编号一致。实际检查包括：

1. 全部 549 个现存 Beauty TFRecord 的大小与 SHA-256，均与固定提交中的历史来源清单相同。
2. 全部三份划分各 22,363 条用户序列，按下述统一规则与 P5 数值序列逐项一致，差异 0。三份划分合计覆盖所有 12,101 个商品。
3. 每次商品在训练、验证和测试序列中出现时，其文本均与 items 表内对应数字 ID 的原始文本逐字节相同。
4. P5 `item2id` 与 `id2item` 全量互逆；ASIN 直接来自 `id2item`，未利用标题、最终 SID 或模糊匹配猜造。

因此导出关系为：

```text
P5 item ID：原始 datamaps 中的字符串，数值从 1 到 12101
GRID item ID：TFRecord int64，数值从 0 到 12100
sid_column = grid_item_id = int(p5_item_id) - 1
original_item_id = P5 id2item[str(grid_item_id + 1)]
original_id_namespace = amazon_asin
```

ASIN 保留原始字符串，不转换数值、不改变大小写、不去掉前导零。无商品删除、合并、拆分或再次编号；空标题和重复文本不会被过滤。每一列四位冗余码与固定 G3 的对应列完全相等。

### 原始划分的实际关系

设 P5 用户完整交互列表为 `S`，每个商品数字减 1，GRID 用户编号为 P5 用户编号减 1：

| GRID 划分 | 完全匹配的规则 | 用户数 | 差异序列 |
|---|---|---:|---:|
| training | `(S - 1)[:-2]`，不截断 | 22,363 | 0 |
| evaluation | `(S - 1)[:-1][-21:]` | 22,363 | 0 |
| testing | `(S - 1)[-21:]` | 22,363 | 0 |

这里 `S - 1` 表示逐元素减 1，不是删除商品。未考虑末 21 件截断时，验证和测试分别有 935 和 1,019 条长序列不同；按实际截断规则核验后均为 0。原输入没有修补。

上述是从当前原件全量核验得到的数据规则。上游 Amazon/P5 的更早过滤、排序及原始 GRID 转换参数没有完整历史记录，保持未知；本次没有重跑预处理。公共 P5 档案本次取得的内容已经通过全部历史分片身份与全量编号交互绑定，不冒充可证明的历史下载版本。

### embedding 到 SID 列轴

固定源码入口：

- [`../../_运行/generate_qwen_embeddings.py`](../../_运行/generate_qwen_embeddings.py)：按 TFRecord 的显式 `id` 建字典，拒绝重复和不连续 ID，按 `0..N-1` 生成向量；embedding 第 i 行对应 GRID ID i。
- [`../../src/modules/clustering/residual_quantization.py`](../../src/modules/clustering/residual_quantization.py)：预测保留 `batch.item_ids`，输出 `item_id` 和三位 `cluster_ids`。
- [`../../src/utils/inference_utils.py`](../../src/utils/inference_utils.py)：合并原始带键 PKL，并调用键放置、去重、转置。
- [`../../src/utils/tensor_utils.py`](../../src/utils/tensor_utils.py)：按 `item_id` 放入 `[N,3]` 行；唯一三位追加 0，碰撞组按行索引递增追加 1 到组大小；最后转置为 `[4,N]`。

所以原 PKL 记录顺序不决定列轴，最终第 i 列仍对应 GRID ID i。核验先明确排除负键、越界、重复和遗漏，再执行原函数，避免原合并函数未完整防御这些异常的风险。

## 实际核验回执

| 验收项 | 实际结果 |
|---|---|
| A1 固定身份 | 原件 SHA-256、大小、dtype、形状匹配 |
| A2 全量身份 | 12,101 行；列号与 GRID ID 完整；ASIN 无空值、无重复 |
| A3 编号规则 | 全量三个划分和原源码链支持上述编号公式 |
| A4 来源绑定 | 549 个历史分片哈希匹配；全部编号用户序列一致；覆盖全部商品 |
| A5 原始预测 | 原 PKL 为 `list[dict]`，12,101 条；键为 Python int；字段 `item_id,cluster_ids`；完整键集合 0–12100；重复、遗漏、负数、越界均 0；三位码均为整数且在 0–255 |
| A6 四位重建 | 执行固定源码三个函数，仅写临时副本；重建对 G3 差异值 0、差异列 0；同次生成 tensor 对 G3 差异值 0 |
| A7 配置 | 同次完整 SID/量化 Hydra 配置、command、overrides、embedding 元数据均已读取，发布原字节可读副本 |
| A8 入口 | 推送后独立检查固定提交及新文件；该联网发布检查不属于 CPU 核验脚本，最终提交 SHA 与远端结果在交付回复给出 |

核验环境为 Python 3.10.20、PyTorch 2.6.0+cu124，全部 tensor 操作在 CPU。实际 UTC 起止时间、脚本 SHA-256、当时临时工作目录和命令在 JSON 中；退出状态为 0。重建文件的序列化字节不要求与原件相同，报告比较的是形状、列序及全部码值。

### 编码器与量化设置

原 `item_embeddings.json` 与文本生成命令说明实际使用本地 Qwen2.5-7B-Instruct，3584 维，WMH MeanAggregation、右侧 padding、最大长度 128，FP16 权重、FP32 保存。输入是 TFRecord `text`，原 TFRecord 的旧 embedding 字段被忽略。文本包含 title、brand、categories、price；原始文本处理的更早转换细节未知。

完整量化配置记录 seed 42、三层 RQ-KMeans、每层 256 码字、`normalize_residuals=true`、初始化缓冲 3072、量化训练最多 30 steps、FP32，checkpoint 路径与 SID 生成命令一致。配置和元数据均为已保存的有效记录，不是补写默认模板。约 173 MB embedding 与约 531 MB 量化 checkpoint 的历史身份引用现有 server manifest，本次无需重传或加载。

## 获取真实载荷与复核

原件仍在来源提交和本次提交的相同路径。新增映射、脚本、JSON、CSV、配置副本均是普通 Git 文件，可以直接读取。原 tensor 和 PKL 继续由 Git LFS 保存，其指针的 `oid sha256` 是真实载荷 SHA-256，不能与 Git blob SHA 混用。本次已通过 Git LFS 成功取得并哈希核对所需原件。

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --branch yjc-grid https://github.com/woobowen/recommend.git
cd recommend
# 固定已发布的 G3 身份补件；后续交付说明不改变该原件。
git checkout abfe7827e9c586a10edcb47c1f4a0c2a6ae62f67
git lfs pull --include="GRID baseline/experiments/beauty_rkmeans_qwen_20261010_012253/semantic_ids.pt,GRID baseline/experiments/beauty_rkmeans_qwen_20261010_012253/sid_generation/pickle/**"
```

原 PKL SHA-256 为 `dcbfb61369cde2627cea50cc42bb80be753e2d9854b85f1825460174486bd590`，253,952 bytes。同次生成 tensor 为 `d5241c43708cb9e5ce0c1e812b965a0b5f92e38d94aead8246bcebeea62e385f`，388,534 bytes。两 tensor 原容器哈希不同但数值相同。

完整复核需要所有 Beauty 原始 TFRecord 和已列出的历史记录。可按 manifest 仅取得所需 Beauty 数据，无需 Qwen 权重或其他数据集：

```bash
git lfs pull --include="GRID baseline/original_assets/data/amazon_data/beauty/**,GRID baseline/experiments/beauty_rkmeans_qwen_20261010_012253/item_embeddings.json,GRID baseline/experiments/beauty_rkmeans_qwen_20261010_012253/sid_generation/**,GRID baseline/experiments/beauty_rkmeans_qwen_20261010_012253/quantizer_train/.hydra/config.yaml"
python "GRID baseline/codex/beauty_g3_identity/fetch_p5_identity_inputs.py" --output /tmp/p5_beauty_identity
CUDA_VISIBLE_DEVICES="" python "GRID baseline/codex/beauty_g3_identity/verify_g3.py"   --repo "$PWD"   --data-root "$PWD/GRID baseline/original_assets/data/amazon_data/beauty"   --p5-dir /tmp/p5_beauty_identity   --output /tmp/beauty_g3_reverification
```

核验依赖 `torch`、`tfrecord` 及其 protobuf 依赖，使用仓库已记录的兼容环境即可。脚本拒绝非空输出目录，原件只读；只对临时文件执行会写回的去重和转置函数。CSV 是 UTF-8、LF、按列号升序、无索引列，结果行及字节确定；核验 JSON 因时间和路径不同，应比较检查计数与数据关系，而不要求重复运行的 JSON 整文件 SHA 相同。

## 未决资料与责任边界

原始 GRID 转换脚本、原始独立 GRID→ASIN 映射以及历史下载的精确 P5 档案版本未找到。已检查本地历史归档、来源 Git 树、ECNU 的项目 original_assets 和旧 Tiger/data 目录，以及旧 Azure 项目位置；没有用标题补造缺失原件。原资料最可能由历史数据导出目录 `/data_a/mhwang/Rec/GRID/amazon_data` 的维护者进一步提供，但本次已通过上述确定性替代链恢复完整商品身份。

本次 A2–A4 使用的是请求文档允许的“共同预处理编号清单＋全量核验关系”替代方式。更早历史未知不影响这条已执行的完整编号关系；如需审计最早 Amazon→P5 的每个处理参数，仍应向原数据提供者补取，不能称本次已恢复全部历史参数。

本补件仅完成 GRID 上游身份资料。没有验证 LETTER 的数字编号，没有将两侧相同数字直接等同，没有运行 LETTER 适配或下游训练。使用方应继续按 ASIN 连接和核对全量商品集合、双射、交互覆盖及 index 转换。GitHub 固定入口已包含必要轻量补件，不另建全仓库 ZIP。
